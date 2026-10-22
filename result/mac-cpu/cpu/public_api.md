# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261022_182159/run.yaml`
- Timestamp: `20261022_182159`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `quick` profile, 7 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261022_182159`.

- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`

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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20261022_182159/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20261022_182159/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/public_api/20261022_182159/cpu_public_api_t1_20261022_182159.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261022_182159/cpu_public_api_t4_20261022_182159.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20261022_182159/cpu_public_api_20261022_182159.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.409 ± 0.480 | - | 4.731 ± 0.745 | 5.071 ± 2.959 | 1.950 ± 2.812 | 15.492 ± 8.030 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 4.214 ± 2.157 | - | 4.386 ± 0.597 | 4.631 ± 1.071 | 1.446 ± 0.070 | 79.803 ± 10.512 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 16.760 ± 4.920 | - | 13.731 ± 3.383 | 14.017 ± 0.921 | 10.665 ± 0.878 | 23.709 ± 7.746 | 15.308 ± 6.356 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 13.255 ± 1.975 | - | 13.383 ± 2.207 | 12.792 ± 1.349 | 14.470 ± 2.087 | 11.162 ± 3.943 | 9.909 ± 0.849 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 63.017 ± 17.581 | - | 56.145 ± 6.158 | 22.426 ± 3.299 | 43.437 ± 8.607 | 47.594 ± 0.623 | 52.341 ± 6.389 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 29.567 ± 6.020 | - | 21.578 ± 2.358 | 11.936 ± 2.462 | 26.932 ± 7.806 | 65.151 ± 5.776 | 30.268 ± 6.107 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 12.279 ± 4.162 | - | 12.899 ± 0.841 | 14.255 ± 2.724 | 55.582 ± 5.366 | 68.329 ± 3.412 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 11.067 ± 0.885 | - | 11.327 ± 0.747 | 11.535 ± 1.692 | 16.385 ± 0.231 | 48.320 ± 10.026 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 13.930 ± 0.847 | - | unsupported | 12.801 ± 2.109 | 48.000 ± 3.099 | 59.925 ± 14.198 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 11.896 ± 1.301 | - | unsupported | 11.371 ± 5.806 | 17.071 ± 0.855 | 51.843 ± 12.762 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 12.459 ± 2.079 | - | 12.213 ± 0.703 | 14.661 ± 3.637 | 31.823 ± 8.633 | 8.785 ± 4.218 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 11.720 ± 1.413 | - | 11.906 ± 0.853 | 11.524 ± 3.724 | 20.289 ± 0.584 | 473.122 ± 23.512 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 43.795 ± 6.300 | - | 46.055 ± 7.530 | 38.410 ± 1.918 | 45.725 ± 3.687 | 50.595 ± 4.106 | 54.486 ± 17.327 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 25.451 ± 9.496 | - | 20.709 ± 5.382 | 13.100 ± 2.354 | 34.367 ± 15.388 | 69.246 ± 9.071 | 27.550 ± 8.629 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 49.529 ± 5.128 | - | 52.846 ± 2.263 | 52.613 ± 2.905 | 101.503 ± 7.879 | 84.670 ± 12.117 | 80.990 ± 7.920 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 23.279 ± 1.707 | - | 22.571 ± 3.162 | 28.202 ± 14.489 | 49.770 ± 6.873 | 108.849 ± 20.577 | 52.939 ± 10.866 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 11.460 ± 2.840 | - | 13.203 ± 3.945 | 10.467 ± 0.780 | 10.399 ± 4.753 | 9.702 ± 1.710 | 12.744 ± 7.991 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 12.274 ± 1.675 | - | 13.097 ± 0.536 | 12.598 ± 3.114 | 9.360 ± 1.179 | 13.987 ± 3.834 | 13.766 ± 2.553 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.555 ± 0.193 | - | 10.116 ± 2.711 | 6.769 ± 0.977 | 5.333 ± 1.413 | 8.332 ± 0.411 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.169 ± 0.069 | - | 3.969 ± 0.934 | 3.044 ± 0.054 | 3.666 ± 0.261 | 10.552 ± 1.157 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 9.122 ± 0.557 | - | 8.620 ± 1.783 | 8.815 ± 1.196 | 6.761 ± 0.142 | 10.527 ± 0.827 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 8.146 ± 0.253 | - | 8.275 ± 0.212 | 8.947 ± 0.563 | 7.149 ± 0.238 | 216.052 ± 37.368 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.576 ± 1.814 | - | 7.262 ± 0.747 | 6.021 ± 3.175 | 4.282 ± 3.814 | 6.670 ± 0.229 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 6.033 ± 0.792 | - | 5.641 ± 0.629 | 5.600 ± 1.352 | 2.339 ± 0.132 | 250.107 ± 184.436 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 8.608 ± 0.616 | - | 8.502 ± 0.771 | 9.642 ± 3.475 | 7.469 ± 0.908 | 7.527 ± 0.754 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 8.078 ± 0.428 | - | 7.975 ± 0.170 | 8.382 ± 1.156 | 7.113 ± 0.109 | 729.791 ± 482.773 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 14.606 ± 0.636 | - | 12.456 ± 1.369 | 18.288 ± 10.054 | 51.842 ± 4.068 | 66.936 ± 23.546 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 10.251 ± 1.306 | - | 10.882 ± 0.992 | 10.711 ± 0.432 | 22.823 ± 5.660 | 44.329 ± 9.759 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 13.828 ± 1.719 | - | unsupported | 11.114 ± 0.903 | 79.344 ± 10.994 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 13.439 ± 1.350 | - | unsupported | 12.639 ± 4.008 | 20.109 ± 8.845 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 15.720 ± 6.757 | - | 15.170 ± 10.457 | 14.422 ± 1.182 | 18.795 ± 2.399 | 14.634 ± 9.222 | 13.984 ± 2.348 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 11.282 ± 0.750 | - | 12.252 ± 1.269 | 15.222 ± 3.021 | 14.557 ± 2.688 | 13.503 ± 1.887 | 13.562 ± 1.074 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 24.133 ± 1.389 | - | 20.861 ± 11.768 | 19.083 ± 3.235 | 31.194 ± 3.009 | 27.378 ± 3.773 | 24.868 ± 3.771 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 20.006 ± 3.453 | - | 19.185 ± 1.413 | 24.602 ± 3.952 | 28.794 ± 0.946 | 19.442 ± 5.975 | 21.300 ± 1.141 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 37.343 ± 3.016 | - | 40.412 ± 4.865 | 41.534 ± 11.643 | 58.434 ± 17.305 | 62.580 ± 16.745 | 68.004 ± 18.771 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 17.384 ± 4.193 | - | 25.414 ± 3.413 | 26.014 ± 2.950 | 14.993 ± 1.575 | 59.880 ± 8.668 | 33.023 ± 10.623 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 7.673 ± 2.127 | - | 8.440 ± 3.858 | 10.117 ± 3.015 | 11.232 ± 3.534 | 9.525 ± 3.075 | 9.902 ± 2.540 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 8.199 ± 1.230 | - | 8.009 ± 4.895 | 7.469 ± 0.832 | 9.169 ± 0.549 | 7.461 ± 2.811 | 8.059 ± 6.760 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 25.432 ± 4.097 | - | 21.477 ± 7.436 | 21.993 ± 7.220 | 19.493 ± 2.577 | 19.566 ± 2.929 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 13.133 ± 1.115 | - | 12.967 ± 1.689 | 12.906 ± 1.222 | 17.001 ± 2.943 | 16.592 ± 4.905 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 38.212 ± 12.245 | - | 35.488 ± 6.670 | 33.735 ± 2.667 | 34.666 ± 0.771 | 40.318 ± 7.198 | 55.212 ± 19.354 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 27.529 ± 8.370 | - | 28.230 ± 7.833 | 18.808 ± 1.846 | 19.055 ± 3.671 | 94.219 ± 32.776 | 45.896 ± 29.802 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 20.996 ± 6.262 | - | 20.379 ± 5.924 | 19.019 ± 1.899 | 36.084 ± 5.089 | 44.930 ± 112.340 | 20.372 ± 6.055 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 19.329 ± 2.164 | - | 19.872 ± 1.945 | 24.332 ± 3.695 | 25.785 ± 4.166 | 20.927 ± 3.674 | 21.629 ± 0.833 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 19.287 ± 1.457 | - | 20.695 ± 7.016 | 32.031 ± 2.480 | 19.019 ± 1.997 | 38.014 ± 1.811 | 53.987 ± 22.464 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 11.893 ± 4.327 | - | 10.914 ± 2.570 | 15.071 ± 4.575 | 15.409 ± 5.720 | 50.776 ± 15.655 | 24.566 ± 7.173 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 31.488 ± 1.436 | - | 35.083 ± 3.399 | 28.210 ± 1.853 | 27.715 ± 10.833 | 27.966 ± 5.020 | 26.616 ± 1.199 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 14.595 ± 3.234 | - | 14.975 ± 1.834 | 16.711 ± 6.658 | 8.587 ± 0.227 | 40.639 ± 9.733 | 18.808 ± 7.042 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 40.186 ± 12.508 | - | 37.238 ± 6.346 | 45.806 ± 9.119 | 38.970 ± 8.228 | 46.986 ± 48.499 | 48.885 ± 25.678 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 25.648 ± 5.787 | - | 23.236 ± 4.478 | 19.517 ± 6.468 | 18.519 ± 2.149 | 61.296 ± 7.197 | 36.326 ± 7.482 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 20.314 ± 7.190 | - | 17.773 ± 6.332 | 21.965 ± 0.191 | 32.167 ± 14.275 | 24.059 ± 1.952 | 23.905 ± 2.367 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 8.217 ± 0.427 | - | 8.416 ± 0.323 | 13.123 ± 0.767 | 9.081 ± 0.217 | 27.641 ± 9.260 | 16.129 ± 2.397 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.385 ± 0.111 | 0.386 ± 0.104 | unsupported | unsupported | 0.437 ± 0.269 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.621 ± 0.191 | 0.787 ± 0.437 | unsupported | unsupported | 0.487 ± 0.109 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 32.792 ± 11.660 | - | 28.454 ± 3.324 | 28.759 ± 12.382 | 38.224 ± 8.396 | 27.703 ± 31.221 | 29.518 ± 7.675 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 19.786 ± 2.015 | - | 18.544 ± 2.120 | 23.537 ± 5.777 | 24.881 ± 1.190 | 26.321 ± 12.060 | 23.155 ± 5.115 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 34.022 ± 12.817 | - | 46.167 ± 7.602 | 26.297 ± 2.530 | 30.465 ± 7.490 | 25.886 ± 7.058 | 20.403 ± 2.770 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 19.303 ± 2.392 | - | 17.901 ± 0.617 | 17.533 ± 2.378 | 26.982 ± 2.483 | 20.091 ± 2.989 | 23.093 ± 3.128 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 24.058 ± 5.081 | - | 22.383 ± 6.749 | 22.407 ± 2.979 | 35.947 ± 7.329 | 26.815 ± 3.663 | 24.431 ± 4.133 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 18.824 ± 3.986 | - | 20.748 ± 2.982 | 25.611 ± 3.626 | 25.473 ± 0.875 | 20.921 ± 0.856 | 20.187 ± 1.862 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 12.508 ± 0.944 | - | 12.771 ± 4.449 | 13.147 ± 1.424 | 19.654 ± 2.545 | 13.161 ± 3.572 | 14.740 ± 3.373 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 13.155 ± 2.304 | - | 14.219 ± 1.778 | 15.780 ± 3.226 | 13.593 ± 0.660 | 19.381 ± 6.922 | 13.870 ± 9.340 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 41.809 ± 3.499 | - | 39.368 ± 2.693 | 68.823 ± 1.182 | 52.577 ± 18.182 | 99.781 ± 11.578 | 81.059 ± 6.728 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 21.939 ± 7.925 | - | 22.010 ± 4.957 | 36.065 ± 7.933 | 19.028 ± 0.479 | 146.307 ± 20.224 | 62.109 ± 28.013 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 7.217 ± 0.882 | - | 8.214 ± 0.626 | 13.731 ± 0.574 | 29.811 ± 4.074 | 7.210 ± 0.912 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.675 ± 0.270 | - | 7.195 ± 0.399 | 7.800 ± 0.709 | 11.880 ± 1.906 | 7.828 ± 3.002 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 0.922 ± 0.094 | - | 1.084 ± 0.193 | 15.419 ± 0.642 | 4.080 ± 1.693 | 0.867 ± 0.012 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.839 ± 0.070 | - | 0.975 ± 0.179 | 12.551 ± 0.441 | 2.810 ± 0.644 | 1.127 ± 0.288 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.978 ± 0.230 | - | 1.161 ± 0.323 | 1.726 ± 0.319 | 1.072 ± 0.092 | 0.857 ± 0.086 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 2.176 ± 0.602 | - | 1.960 ± 1.900 | 1.176 ± 0.054 | 1.049 ± 0.084 | 0.971 ± 0.098 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 7.201 ± 1.729 | - | 10.077 ± 4.285 | 14.261 ± 2.003 | 33.381 ± 10.541 | 7.930 ± 0.509 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.834 ± 0.233 | - | 6.793 ± 2.310 | 6.973 ± 0.617 | 12.771 ± 2.842 | 8.467 ± 1.656 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.142 ± 0.348 | - | 1.194 ± 1.137 | 16.957 ± 2.667 | 2.740 ± 1.641 | 0.703 ± 0.057 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 1.374 ± 0.763 | - | 1.100 ± 0.294 | 12.584 ± 1.024 | 2.807 ± 1.725 | 0.992 ± 0.073 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 4.114 ± 1.443 | - | 4.776 ± 1.904 | 7.599 ± 1.794 | 6.020 ± 5.126 | 4.131 ± 0.588 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 8.070 ± 2.480 | - | 7.744 ± 1.173 | 4.592 ± 1.868 | 5.930 ± 2.401 | 4.093 ± 0.669 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 7.596 ± 2.497 | - | 8.201 ± 1.593 | 6.482 ± 0.377 | 51.906 ± 16.253 | 7.782 ± 0.681 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.735 ± 3.227 | - | 7.511 ± 2.599 | 6.410 ± 0.304 | 26.016 ± 2.247 | 7.550 ± 1.438 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.917 ± 0.099 | - | 0.910 ± 0.108 | 4.242 ± 0.094 | 3.422 ± 1.765 | 0.752 ± 0.076 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 1.116 ± 0.258 | - | 0.860 ± 0.073 | 2.534 ± 0.151 | 3.401 ± 1.431 | 1.173 ± 0.545 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.028 ± 0.458 | - | 1.055 ± 0.773 | 0.769 ± 0.047 | 1.257 ± 2.430 | 0.981 ± 0.105 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 1.967 ± 0.439 | - | 2.374 ± 0.253 | 0.945 ± 0.247 | 0.869 ± 0.307 | 1.161 ± 0.168 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 6.020 ± 2.019 | - | 6.386 ± 1.778 | 6.202 ± 0.180 | 48.662 ± 11.782 | 8.526 ± 1.361 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.576 ± 1.270 | - | 6.669 ± 1.316 | 6.746 ± 0.890 | 22.014 ± 3.532 | 8.964 ± 2.376 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.896 ± 0.626 | - | 0.861 ± 0.057 | 4.637 ± 0.392 | 6.212 ± 2.287 | 0.741 ± 0.108 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.868 ± 0.074 | - | 0.985 ± 0.308 | 2.577 ± 0.352 | 2.585 ± 1.604 | 1.038 ± 0.215 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.940 ± 0.650 | - | 0.971 ± 0.325 | 0.916 ± 0.232 | 1.471 ± 2.048 | 0.808 ± 0.044 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 2.058 ± 0.147 | - | 2.114 ± 0.186 | 0.691 ± 0.051 | 0.959 ± 0.091 | 0.995 ± 0.210 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.897 ± 0.193 | 1.154 ± 0.059 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.970 ± 0.140 | 0.910 ± 0.111 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 30.683 ± 1.754 | - | 33.063 ± 5.721 | 37.732 ± 2.762 | 51.244 ± 6.377 | 38.043 ± 55.215 | 31.668 ± 0.844 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 15.612 ± 1.012 | - | 15.412 ± 8.011 | 25.104 ± 6.215 | 35.813 ± 14.696 | 37.168 ± 59.245 | 18.478 ± 5.131 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 22.848 ± 8.882 | - | 21.009 ± 3.927 | 18.242 ± 0.296 | 46.905 ± 14.748 | 46.230 ± 16.452 | 46.399 ± 5.980 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 12.230 ± 1.118 | - | 13.102 ± 0.890 | 14.726 ± 3.680 | 24.021 ± 3.945 | 47.445 ± 3.316 | 23.770 ± 3.987 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 39.744 ± 6.510 | - | 39.522 ± 4.819 | 24.777 ± 3.507 | 30.537 ± 1.770 | 30.621 ± 17.879 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 26.917 ± 1.287 | - | 25.981 ± 4.376 | 21.373 ± 0.986 | 30.934 ± 0.767 | 30.295 ± 3.605 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 14.455 ± 5.948 | - | 15.635 ± 4.123 | 18.304 ± 6.316 | 24.672 ± 5.770 | 13.058 ± 6.533 | 22.767 ± 6.356 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 11.086 ± 0.882 | - | 11.381 ± 2.192 | 15.650 ± 2.183 | 15.331 ± 2.087 | 17.314 ± 3.645 | 14.874 ± 2.291 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.358 ± 4.597 | - | 26.672 ± 8.052 | 30.257 ± 3.452 | 39.565 ± 5.918 | 34.661 ± 6.813 | 33.242 ± 2.994 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 21.578 ± 5.450 | - | 21.785 ± 5.750 | 19.197 ± 1.196 | 15.813 ± 1.379 | 52.226 ± 18.842 | 27.217 ± 8.142 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 14.900 ± 1.672 | - | 13.101 ± 1.334 | 12.539 ± 0.346 | 27.112 ± 1.563 | 27.283 ± 1.671 | 32.548 ± 59.989 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 11.621 ± 0.328 | - | 11.180 ± 3.603 | 14.843 ± 1.332 | 20.038 ± 2.367 | 30.797 ± 0.919 | 16.363 ± 3.940 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 20.587 ± 6.010 | - | 19.924 ± 2.070 | 20.711 ± 2.432 | 35.537 ± 3.071 | 27.084 ± 6.465 | 26.608 ± 2.612 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 22.034 ± 2.105 | - | 20.878 ± 4.666 | 23.382 ± 2.291 | 25.606 ± 1.274 | 21.342 ± 1.630 | 21.576 ± 2.074 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 72.936 ± 9.050 | - | 74.894 ± 4.060 | 69.103 ± 3.228 | 18.192 ± 3.438 | 60.724 ± 10.338 | 56.303 ± 12.766 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 44.875 ± 10.490 | - | 60.486 ± 30.511 | 36.580 ± 10.553 | 10.041 ± 1.656 | 63.736 ± 12.445 | 33.799 ± 12.772 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 29.780 ± 4.036 | 39.286 ± 14.829 | unsupported | 26.477 ± 9.183 | 35.893 ± 5.285 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 24.435 ± 1.450 | 23.761 ± 2.555 | unsupported | 21.485 ± 2.435 | 36.568 ± 10.242 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.801 ± 0.086 | - | 0.793 ± 0.053 | 0.578 ± 0.038 | 0.860 ± 0.075 | 1.065 ± 0.403 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.878 ± 0.236 | - | 0.882 ± 0.249 | 0.792 ± 0.030 | 1.065 ± 0.180 | 1.385 ± 1.027 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.879 ± 0.260 | - | 1.075 ± 0.414 | 0.705 ± 0.066 | 0.821 ± 0.043 | 0.963 ± 0.201 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.734 ± 0.119 | - | 0.785 ± 0.075 | 0.928 ± 0.173 | 0.947 ± 0.177 | 1.029 ± 0.324 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.060 ± 0.232 | - | unsupported | 1.012 ± 0.085 | 2.462 ± 0.843 | 1.356 ± 0.354 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.408 ± 0.134 | - | unsupported | 1.137 ± 0.068 | 1.256 ± 0.107 | 2.659 ± 12.771 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.424 ± 0.047 | - | 0.449 ± 0.021 | 0.307 ± 0.040 | 0.420 ± 0.056 | 2.880 ± 2.480 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.738 ± 0.303 | - | 0.953 ± 0.212 | 0.298 ± 0.012 | 0.268 ± 0.384 | 1.259 ± 5.370 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.659 ± 0.036 | - | 0.693 ± 0.013 | 1.106 ± 0.224 | 0.982 ± 0.085 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.686 ± 0.201 | - | 0.769 ± 0.064 | 1.253 ± 0.293 | 1.170 ± 0.207 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.715 ± 0.070 | - | 0.701 ± 0.112 | 0.990 ± 0.210 | 0.951 ± 0.209 | 1.768 ± 13.290 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 1.043 ± 0.234 | - | 0.894 ± 0.171 | 0.845 ± 0.184 | 1.008 ± 0.135 | 1.542 ± 7.963 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.660 ± 0.068 | - | 1.126 ± 0.807 | 0.415 ± 0.071 | 0.498 ± 0.121 | 1.372 ± 4.651 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.839 ± 0.365 | - | 0.980 ± 0.310 | 0.410 ± 0.035 | 0.602 ± 0.074 | 0.396 ± 0.024 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.669 ± 0.261 | - | 0.006 ± 0.001 | 1.133 ± 0.051 | 1.435 ± 1.371 | 1.863 ± 10.222 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.595 ± 0.506 | - | 0.003 ± 0.000 | 1.472 ± 0.133 | 1.526 ± 0.271 | 1.729 ± 0.177 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.062 ± 0.004 | - | 0.067 ± 0.009 | 0.120 ± 0.009 | 0.096 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.201 ± 0.102 | - | 0.162 ± 0.352 | 0.184 ± 0.020 | 0.167 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.207 ± 0.009 | - | 0.235 ± 0.074 | 0.336 ± 0.138 | 0.233 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.052 ± 0.066 | - | 0.050 ± 0.009 | 0.053 ± 0.034 | 0.060 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.003 ± 0.022 | - | 0.004 ± 0.002 | 0.004 ± 0.003 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.003 ± 0.003 | - | 0.005 ± 0.000 | 0.014 ± 0.002 | 0.052 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.011 ± 0.013 | 0.012 ± 0.009 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.002 ± 0.002 | - | 0.005 ± 0.010 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.008 ± 0.003 | 0.006 ± 0.005 | 0.009 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.001 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.005 ± 0.000 | - | 0.005 ± 0.007 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.003 | 0.009 ± 0.007 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.060 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.441 ± 0.180 | - | 0.279 ± 0.127 | 0.278 ± 0.065 | 0.259 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.016 ± 0.004 | 0.004 ± 0.000 | 0.009 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.011 ± 0.000 | - | 0.018 ± 0.003 | 0.025 ± 0.009 | 0.133 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.005 | 0.005 ± 0.011 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.049 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.060 ± 0.001 | - | 0.065 ± 0.002 | 0.258 ± 0.067 | 0.120 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.098 ± 0.003 | - | 0.106 ± 0.002 | 0.190 ± 0.016 | 0.195 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.205 ± 0.004 | - | 0.322 ± 0.165 | 0.306 ± 0.108 | 0.259 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.026 ± 0.001 | - | 0.038 ± 0.004 | 0.098 ± 0.047 | 0.066 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.001 | 0.107 ± 0.022 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.061 ± 0.048 | 0.008 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.098 ± 0.034 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.001 ± 0.000 | - | 0.010 ± 0.000 | 0.053 ± 0.007 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.098 ± 0.015 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.001 | 0.087 ± 0.026 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.012 ± 0.001 | 0.060 ± 0.013 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.001 | 0.106 ± 0.031 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.060 ± 0.017 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.059 ± 0.004 | 0.008 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.182 ± 0.016 | - | 0.247 ± 0.027 | 0.241 ± 0.024 | 0.194 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.061 ± 0.058 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.011 ± 0.000 | - | 0.017 ± 0.001 | 0.077 ± 0.016 | 0.070 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.004 ± 0.005 | - | 0.005 ± 0.001 | 0.064 ± 0.009 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.067 ± 0.018 | 0.008 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.264 ± 0.022 | - | 0.317 ± 0.054 | 0.455 ± 0.022 | 0.998 ± 0.251 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.755 ± 0.281 | - | 1.794 ± 0.437 | 1.854 ± 0.144 | 2.192 ± 0.324 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 7.476 ± 1.348 | - | 7.891 ± 0.564 | 6.910 ± 0.067 | 7.247 ± 0.122 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 7.318 ± 0.633 | - | 7.055 ± 2.362 | 3.814 ± 0.227 | 4.040 ± 0.106 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.038 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.107 ± 0.016 | - | 0.096 ± 0.009 | 0.095 ± 0.002 | 0.189 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.001 | 0.006 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.003 | 0.010 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.002 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.001 | - | 0.008 ± 0.000 | 0.010 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.019 ± 0.002 | - | 0.032 ± 0.013 | 0.024 ± 0.005 | 0.071 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.011 ± 0.002 | 0.013 ± 0.002 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.085 ± 0.070 | - | 0.028 ± 0.004 | 0.029 ± 0.005 | 0.079 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 59.795 ± 3.091 | - | 57.032 ± 7.451 | 31.212 ± 2.662 | 32.783 ± 1.460 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.931 ± 0.171 | - | 0.891 ± 0.191 | 0.791 ± 0.084 | 0.918 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.013 ± 0.034 | - | 0.047 ± 0.000 | 0.018 ± 0.002 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.058 ± 0.009 | - | 0.061 ± 0.008 | 0.054 ± 0.001 | 0.126 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.718 ± 0.007 | - | 0.273 ± 0.017 | 0.480 ± 0.058 | 1.016 ± 0.117 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.567 ± 2.552 | - | 1.705 ± 0.287 | 2.723 ± 2.143 | 1.109 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 7.015 ± 3.505 | - | 6.510 ± 0.355 | 9.133 ± 1.417 | 3.739 ± 0.145 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 7.424 ± 2.107 | - | 7.380 ± 2.663 | 3.682 ± 0.218 | 3.889 ± 0.201 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.015 ± 0.001 | 0.035 ± 0.065 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.090 ± 0.011 | - | 0.093 ± 0.001 | 0.118 ± 0.148 | 0.172 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | 0.019 ± 0.001 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.009 ± 0.002 | - | 0.014 ± 0.000 | 0.011 ± 0.002 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.017 ± 0.001 | 0.035 ± 0.007 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.016 ± 0.000 | - | 0.028 ± 0.002 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.021 ± 0.005 | - | 0.026 ± 0.003 | 0.023 ± 0.001 | 0.089 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.016 ± 0.000 | 0.045 ± 0.054 | 0.027 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.020 ± 0.001 | - | 0.032 ± 0.001 | 0.011 ± 0.003 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.028 ± 0.007 | - | 0.079 ± 0.002 | 0.032 ± 0.006 | 0.096 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 53.974 ± 5.490 | - | 55.595 ± 6.082 | 31.641 ± 2.279 | 19.541 ± 3.304 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.009 ± 0.000 | - | 0.019 ± 0.001 | 0.026 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.731 ± 0.071 | - | 0.745 ± 0.038 | 0.852 ± 0.265 | 0.503 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.036 ± 0.001 | - | 0.016 ± 0.001 | 0.020 ± 0.005 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.050 ± 0.001 | - | 0.051 ± 0.001 | 0.058 ± 0.007 | 0.134 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.241 ± 0.008 | - | 0.249 ± 0.019 | 0.258 ± 0.024 | 0.617 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.445 ± 1.997 | - | 1.346 ± 0.770 | 1.373 ± 0.017 | 1.603 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 6.481 ± 3.265 | - | 5.429 ± 2.398 | 5.524 ± 0.352 | 6.965 ± 1.181 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.964 ± 0.041 | - | 1.745 ± 1.423 | 1.070 ± 0.045 | 1.439 ± 0.804 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.068 ± 0.001 | - | 0.073 ± 0.010 | 0.071 ± 0.007 | 0.145 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.002 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.001 | - | 0.010 ± 0.002 | 0.021 ± 0.002 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.002 | - | 0.008 ± 0.000 | 0.009 ± 0.002 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.016 ± 0.003 | - | 0.022 ± 0.002 | 0.026 ± 0.009 | 0.045 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.009 ± 0.041 | - | 0.012 ± 0.000 | 0.011 ± 0.009 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.020 ± 0.003 | - | 0.026 ± 0.005 | 0.024 ± 0.001 | 0.041 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 9.210 ± 4.479 | - | 8.139 ± 0.280 | 8.379 ± 0.157 | 8.821 ± 0.185 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.006 ± 0.004 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.534 ± 0.105 | - | 0.542 ± 0.011 | 0.590 ± 0.059 | 0.870 ± 0.900 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | 0.017 ± 0.000 | 0.024 ± 0.003 | 0.026 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.040 ± 0.006 | - | 0.050 ± 0.007 | 0.047 ± 0.009 | 0.125 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.250 ± 0.085 | - | 0.265 ± 0.156 | 0.247 ± 0.184 | 0.594 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.897 ± 1.921 | - | 1.360 ± 0.240 | 1.289 ± 0.129 | 0.991 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.579 ± 1.623 | - | 7.937 ± 5.822 | 6.290 ± 6.621 | 2.654 ± 1.042 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.020 ± 0.105 | - | 1.161 ± 0.450 | 1.268 ± 0.243 | 1.059 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.013 ± 0.001 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.167 ± 0.014 | - | 0.094 ± 0.108 | 0.133 ± 0.118 | 0.151 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.011 ± 0.008 | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.009 ± 0.002 | 0.009 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.015 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.023 ± 0.010 | - | 0.025 ± 0.001 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.016 ± 0.004 | - | 0.053 ± 0.002 | 0.019 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.016 ± 0.001 | 0.005 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.018 ± 0.053 | - | 0.028 ± 0.001 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.032 ± 0.095 | - | 0.047 ± 0.014 | 0.023 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 9.957 ± 1.303 | - | 10.020 ± 1.143 | 10.425 ± 1.683 | 4.436 ± 1.237 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.009 ± 0.000 | - | 0.018 ± 0.000 | 0.006 ± 0.000 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.534 ± 0.026 | - | 0.565 ± 0.075 | 0.649 ± 0.337 | 0.348 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.030 ± 0.003 | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.038 ± 0.003 | - | 0.116 ± 0.002 | 0.043 ± 0.001 | 0.104 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.161 ± 0.076 | - | 0.113 ± 0.000 | 0.150 ± 0.013 | 0.694 ± 0.329 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.733 ± 0.016 | - | 1.481 ± 0.900 | 0.837 ± 0.103 | 1.633 ± 0.400 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.696 ± 0.512 | - | 2.726 ± 0.298 | 2.789 ± 0.082 | 5.756 ± 0.334 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.617 ± 0.397 | - | 0.533 ± 0.091 | 0.541 ± 0.045 | 1.138 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.011 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.027 ± 0.000 | - | 0.032 ± 0.004 | 0.034 ± 0.007 | 0.132 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.002 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.002 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.002 | - | 0.013 ± 0.000 | 0.012 ± 0.003 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.012 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.005 ± 0.001 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.010 ± 0.000 | - | 0.016 ± 0.003 | 0.015 ± 0.026 | 0.038 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.451 ± 0.528 | - | 4.361 ± 0.752 | 4.506 ± 0.852 | 8.967 ± 0.556 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.013 ± 0.000 | 0.007 ± 0.002 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.202 ± 0.010 | - | 0.231 ± 0.037 | 0.224 ± 0.018 | 0.638 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.008 ± 0.001 | - | 0.009 ± 0.000 | 0.010 ± 0.002 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.019 ± 0.001 | - | 0.025 ± 0.000 | 0.023 ± 0.004 | 0.095 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.160 ± 0.001 | - | 0.171 ± 0.103 | 0.174 ± 0.081 | 0.560 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.953 ± 0.346 | - | 0.800 ± 0.179 | 0.866 ± 0.105 | 1.010 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.541 ± 0.413 | - | 2.643 ± 0.174 | 2.857 ± 0.452 | 3.755 ± 0.940 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.528 ± 0.059 | - | 0.551 ± 0.035 | 0.510 ± 0.240 | 1.087 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.001 ± 0.001 | - | 0.008 ± 0.002 | 0.007 ± 0.001 | 0.008 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.033 ± 0.034 | - | 0.033 ± 0.001 | 0.031 ± 0.003 | 0.134 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.002 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.011 ± 0.025 | 0.023 ± 0.047 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.008 ± 0.002 | 0.007 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.011 ± 0.002 | 0.010 ± 0.002 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.015 ± 0.000 | - | 0.013 ± 0.000 | 0.029 ± 0.005 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.009 ± 0.001 | 0.007 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.012 ± 0.001 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.011 ± 0.000 | - | 0.016 ± 0.004 | 0.085 ± 0.065 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 4.518 ± 0.246 | - | 4.405 ± 0.295 | 4.085 ± 0.310 | 5.370 ± 1.078 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.199 ± 0.070 | - | 0.213 ± 0.027 | 0.308 ± 0.193 | 0.374 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.008 ± 0.002 | - | 0.013 ± 0.003 | 0.014 ± 0.002 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.020 ± 0.003 | - | 0.025 ± 0.004 | 0.063 ± 0.017 | 0.082 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.119 ± 0.050 | - | 0.126 ± 0.018 | 0.149 ± 0.009 | 0.181 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.253 ± 0.049 | - | 0.253 ± 0.058 | 0.234 ± 0.001 | 0.221 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.517 ± 0.138 | - | 0.618 ± 0.093 | 0.479 ± 0.039 | 0.441 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.066 ± 0.038 | - | 0.069 ± 0.011 | 0.063 ± 0.005 | 0.144 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.006 ± 0.000 | - | 0.013 ± 0.000 | 0.008 ± 0.003 | 0.019 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.010 ± 0.002 | - | 0.015 ± 0.003 | 0.010 ± 0.000 | 0.069 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.008 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.063 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.004 ± 0.001 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.001 | - | 0.007 ± 0.001 | 0.008 ± 0.000 | 0.045 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.004 ± 0.001 | - | 0.010 ± 0.001 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.009 ± 0.002 | 0.047 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.647 ± 0.143 | - | 0.616 ± 0.081 | 0.471 ± 0.017 | 0.611 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.050 ± 0.007 | - | 0.064 ± 0.070 | 0.038 ± 0.000 | 0.104 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.069 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.002 | - | 0.009 ± 0.001 | 0.010 ± 0.000 | 0.069 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.109 ± 0.011 | - | 0.116 ± 0.000 | 0.228 ± 0.033 | 0.195 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.222 ± 0.010 | - | 0.243 ± 0.019 | 0.283 ± 0.080 | 0.262 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.539 ± 0.022 | - | 0.636 ± 0.027 | 0.566 ± 0.222 | 0.559 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.061 ± 0.002 | - | 0.067 ± 0.003 | 0.128 ± 0.017 | 0.271 ± 0.131 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.048 ± 0.020 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.008 ± 0.000 | - | 0.016 ± 0.004 | 0.069 ± 0.037 | 0.090 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.008 ± 0.002 | 0.062 ± 0.036 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.010 ± 0.002 | - | 0.012 ± 0.000 | 0.062 ± 0.020 | 0.098 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.004 ± 0.001 | - | 0.008 ± 0.000 | 0.059 ± 0.011 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.002 | 0.062 ± 0.008 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.001 | - | 0.008 ± 0.000 | 0.149 ± 0.057 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.001 | - | 0.009 ± 0.000 | 0.070 ± 0.015 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.001 | - | 0.008 ± 0.000 | 0.082 ± 0.038 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.067 ± 0.015 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.001 | - | 0.010 ± 0.002 | 0.065 ± 0.021 | 0.085 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.470 ± 0.004 | - | 1.438 ± 0.272 | 0.473 ± 0.015 | 2.082 ± 1.753 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.088 ± 0.019 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.049 ± 0.005 | - | 0.060 ± 0.006 | 0.199 ± 0.133 | 0.145 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.105 ± 0.029 | 0.080 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.008 ± 0.001 | - | 0.013 ± 0.003 | 0.089 ± 0.014 | 0.089 ± 0.106 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.308 ± 0.029 | - | 0.313 ± 0.076 | 0.376 ± 0.214 | 0.573 ± 0.274 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.840 ± 0.244 | - | 0.881 ± 0.842 | 0.936 ± 0.236 | 0.772 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.888 ± 0.871 | - | 1.638 ± 0.160 | 2.107 ± 0.122 | 1.457 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.319 ± 0.029 | - | 0.341 ± 0.613 | 0.399 ± 0.030 | 0.562 ± 0.761 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.004 ± 0.001 | 0.010 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.018 ± 0.004 | - | 0.023 ± 0.000 | 0.023 ± 0.004 | 0.064 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.014 ± 0.000 | - | 0.018 ± 0.001 | 0.026 ± 0.006 | 0.073 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.004 ± 0.006 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.004 ± 0.007 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.006 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.001 | - | 0.011 ± 0.000 | 0.009 ± 0.005 | 0.015 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.004 ± 0.000 | 0.037 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.001 | - | 0.011 ± 0.002 | 0.006 ± 0.003 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.009 ± 0.001 | - | 0.026 ± 0.002 | 0.008 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.694 ± 0.761 | - | 2.636 ± 0.945 | 3.109 ± 0.118 | 3.013 ± 0.583 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.051 | - | 0.010 ± 0.002 | 0.006 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.134 ± 0.010 | - | 0.149 ± 0.012 | 0.179 ± 0.012 | 0.229 ± 0.239 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.007 ± 0.000 | - | 0.012 ± 0.003 | 0.012 ± 0.007 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.016 ± 0.006 | - | 0.019 ± 0.004 | 0.016 ± 0.007 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.222 ± 0.337 | - | 0.220 ± 0.012 | 0.463 ± 0.066 | 0.363 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.519 ± 0.032 | - | 0.772 ± 1.258 | 0.906 ± 0.129 | 1.014 ± 0.159 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.562 ± 0.027 | - | 3.616 ± 2.612 | 2.572 ± 1.097 | 0.814 ± 0.166 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.387 ± 0.616 | - | 0.337 ± 0.024 | 0.374 ± 0.008 | 0.384 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | 0.007 ± 0.000 | 0.034 ± 0.010 | 0.011 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.036 ± 0.003 | - | 0.068 ± 0.410 | 0.029 ± 0.005 | 0.089 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.050 ± 0.040 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.016 ± 0.004 | - | 0.018 ± 0.001 | 0.063 ± 0.050 | 0.077 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.041 ± 0.049 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.009 ± 0.002 | 0.034 ± 0.017 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.029 ± 0.040 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.024 ± 0.001 | 0.030 ± 0.003 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.039 ± 0.009 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.031 ± 0.034 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.018 ± 0.000 | - | 0.013 ± 0.022 | 0.030 ± 0.005 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 3.094 ± 4.917 | - | 2.799 ± 0.338 | 6.709 ± 2.477 | 1.532 ± 0.094 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.036 ± 0.014 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.204 ± 0.234 | - | 0.291 ± 0.059 | 0.186 ± 0.161 | 0.160 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.005 ± 0.001 | - | 0.008 ± 0.000 | 0.077 ± 0.074 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.014 ± 0.003 | - | 0.043 ± 0.014 | 0.051 ± 0.028 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.171 ± 0.009 | - | 0.320 ± 0.105 | 0.126 ± 0.030 | 0.195 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.498 ± 0.216 | - | 0.219 ± 0.043 | 0.190 ± 0.013 | 0.270 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.634 ± 0.495 | - | 0.560 ± 0.439 | 0.421 ± 0.038 | 0.428 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.070 ± 0.044 | - | 0.071 ± 0.006 | 0.089 ± 0.020 | 0.127 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.011 | 0.012 ± 0.007 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.005 ± 0.004 | - | 0.043 ± 0.101 | 0.019 ± 0.006 | 0.060 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.016 ± 0.002 | 0.012 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.032 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.012 ± 0.002 | 0.010 ± 0.005 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.017 ± 0.018 | 0.015 ± 0.018 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.004 | 0.010 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.012 ± 0.001 | 0.010 ± 0.004 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.005 ± 0.000 | - | 0.017 ± 0.001 | 0.023 ± 0.004 | 0.043 ± 0.073 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.011 ± 0.003 | 0.011 ± 0.006 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.431 ± 0.553 | - | 0.464 ± 0.023 | 0.514 ± 0.103 | 1.441 ± 0.948 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.013 ± 0.002 | 0.024 ± 0.030 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.089 ± 0.041 | - | 0.069 ± 0.011 | 0.040 ± 0.012 | 0.114 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.007 ± 0.000 | - | 0.009 ± 0.013 | 0.012 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.006 ± 0.001 | - | 0.015 ± 0.003 | 0.013 ± 0.002 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.103 ± 0.001 | - | 0.323 ± 0.083 | 0.178 ± 0.044 | 0.199 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.175 ± 0.053 | - | 0.159 ± 0.014 | 0.356 ± 0.173 | 0.229 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.345 ± 0.011 | - | 0.406 ± 0.022 | 0.698 ± 0.080 | 0.348 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.052 ± 0.002 | - | 0.061 ± 0.002 | 0.069 ± 0.003 | 0.154 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.026 ± 0.040 | 0.011 ± 0.001 | 0.010 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.040 ± 0.001 | 0.062 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.004 ± 0.004 | - | 0.016 ± 0.000 | 0.030 ± 0.038 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.001 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.025 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.003 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.003 | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.014 ± 0.003 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.413 ± 0.017 | - | 0.535 ± 0.172 | 0.650 ± 0.207 | 0.911 ± 0.366 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.055 ± 0.038 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.027 ± 0.000 | - | 0.035 ± 0.009 | 0.138 ± 0.058 | 0.148 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.035 ± 0.005 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.004 ± 0.001 | - | 0.008 ± 0.000 | 0.044 ± 0.041 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.187 ± 0.038 | - | 1.217 ± 0.091 | 1.398 ± 0.307 | 2.998 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.395 ± 0.331 | - | 4.372 ± 0.073 | 4.810 ± 0.336 | 6.143 ± 0.153 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 15.909 ± 1.041 | - | 15.750 ± 1.246 | 18.045 ± 0.823 | 17.017 ± 0.144 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.520 ± 0.111 | - | 6.024 ± 0.049 | 5.496 ± 0.119 | 5.186 ± 0.218 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.181 ± 0.005 | - | 0.190 ± 0.011 | 0.221 ± 0.048 | 0.245 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.002 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.175 ± 0.155 | - | 0.175 ± 0.011 | 0.171 ± 0.003 | 0.280 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.017 ± 0.003 | - | 0.020 ± 0.000 | 0.024 ± 0.006 | 0.032 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.019 ± 0.001 | 0.032 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.045 ± 0.000 | - | 0.048 ± 0.000 | 0.059 ± 0.011 | 0.065 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.035 ± 0.052 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.022 ± 0.000 | 0.039 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.061 ± 0.000 | - | 0.064 ± 0.001 | 0.070 ± 0.015 | 0.083 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 41.058 ± 5.963 | - | 57.071 ± 5.035 | 40.574 ± 1.947 | 41.809 ± 1.480 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.003 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.433 ± 0.044 | - | 1.514 ± 0.729 | 1.661 ± 0.147 | 1.744 ± 0.127 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.035 ± 0.002 | - | 0.038 ± 0.001 | 0.048 ± 0.006 | 0.060 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.117 ± 0.006 | - | 0.120 ± 0.000 | 0.133 ± 0.024 | 0.200 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.385 ± 0.341 | - | 1.225 ± 0.258 | 1.428 ± 0.109 | 3.066 ± 0.177 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.903 ± 0.341 | - | 4.511 ± 0.356 | 4.703 ± 0.287 | 4.761 ± 1.505 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 15.598 ± 1.386 | - | 15.748 ± 2.211 | 16.034 ± 1.528 | 11.387 ± 4.231 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.861 ± 0.605 | - | 6.209 ± 0.399 | 5.153 ± 0.463 | 8.159 ± 2.659 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.001 | - | 0.013 ± 0.015 | 0.009 ± 0.002 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.189 ± 0.213 | - | 0.183 ± 0.013 | 0.207 ± 0.018 | 0.778 ± 0.453 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.001 | 0.013 ± 0.001 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.166 ± 0.007 | - | 0.176 ± 0.013 | 0.318 ± 0.250 | 0.288 ± 0.071 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.019 ± 0.003 | - | 0.023 ± 0.005 | 0.027 ± 0.004 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.007 ± 0.001 | - | 0.010 ± 0.002 | 0.012 ± 0.002 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.016 ± 0.004 | - | 0.017 ± 0.003 | 0.019 ± 0.004 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.045 ± 0.001 | - | 0.049 ± 0.001 | 0.053 ± 0.009 | 0.073 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.007 ± 0.002 | - | 0.010 ± 0.000 | 0.011 ± 0.003 | 0.034 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.020 ± 0.004 | - | 0.023 ± 0.004 | 0.025 ± 0.006 | 0.042 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.062 ± 0.009 | - | 0.064 ± 0.001 | 0.066 ± 0.012 | 0.098 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 36.116 ± 10.424 | - | 48.415 ± 1.433 | 49.294 ± 14.761 | 29.153 ± 8.983 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.011 ± 0.001 | - | 0.018 ± 0.001 | 0.015 ± 0.000 | 0.084 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.445 ± 0.044 | - | 1.564 ± 0.193 | 1.395 ± 0.026 | 0.912 ± 0.235 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.035 ± 0.003 | - | 0.038 ± 0.003 | 0.041 ± 0.007 | 0.067 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.126 ± 0.003 | - | 0.135 ± 0.014 | 0.135 ± 0.007 | 0.240 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.785 ± 0.049 | - | 0.775 ± 0.078 | 0.731 ± 0.427 | 2.532 ± 0.453 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.227 ± 0.134 | - | 3.247 ± 0.195 | 3.025 ± 0.055 | 4.107 ± 0.117 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 11.931 ± 0.859 | - | 11.976 ± 1.262 | 11.639 ± 0.284 | 12.410 ± 0.310 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.526 ± 0.052 | - | 1.570 ± 1.726 | 1.622 ± 0.053 | 1.714 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.001 | - | 0.008 ± 0.003 | 0.006 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.120 ± 0.001 | - | 0.129 ± 0.214 | 0.134 ± 0.020 | 0.201 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.006 ± 0.002 | - | 0.011 ± 0.003 | 0.012 ± 0.010 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.095 ± 0.002 | - | 0.099 ± 0.001 | 0.106 ± 0.006 | 0.166 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.017 ± 0.018 | - | 0.021 ± 0.002 | 0.016 ± 0.000 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.005 ± 0.001 | - | 0.010 ± 0.000 | 0.007 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.011 ± 0.000 | - | 0.018 ± 0.003 | 0.013 ± 0.003 | 0.030 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.033 ± 0.006 | - | 0.042 ± 0.005 | 0.035 ± 0.007 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.005 ± 0.001 | - | 0.010 ± 0.000 | 0.009 ± 0.003 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.018 ± 0.001 | - | 0.019 ± 0.004 | 0.017 ± 0.001 | 0.029 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.044 ± 0.002 | - | 0.049 ± 0.002 | 0.044 ± 0.000 | 0.058 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 12.812 ± 0.287 | - | 13.038 ± 1.078 | 12.891 ± 0.448 | 12.571 ± 0.308 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.008 ± 0.000 | - | 0.013 ± 0.003 | 0.013 ± 0.002 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.955 ± 0.021 | - | 0.961 ± 0.034 | 1.058 ± 0.063 | 1.123 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.030 ± 0.005 | - | 0.031 ± 0.000 | 0.028 ± 0.002 | 0.045 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.085 ± 0.011 | - | 0.092 ± 0.009 | 0.100 ± 0.005 | 0.165 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.711 ± 0.132 | - | 0.662 ± 0.215 | 0.661 ± 0.139 | 1.952 ± 0.210 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 3.244 ± 0.269 | - | 3.164 ± 0.793 | 3.009 ± 0.285 | 1.645 ± 0.105 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 12.155 ± 2.279 | - | 14.210 ± 3.558 | 11.501 ± 1.574 | 6.143 ± 0.281 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.531 ± 0.080 | - | 1.795 ± 0.914 | 1.551 ± 0.066 | 1.712 ± 0.310 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.001 | - | 0.009 ± 0.002 | 0.007 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.126 ± 0.022 | - | 0.127 ± 0.004 | 0.137 ± 0.031 | 0.206 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.007 ± 0.002 | - | 0.011 ± 0.001 | 0.009 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.095 ± 0.001 | - | 0.110 ± 0.070 | 0.109 ± 0.010 | 0.139 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.015 ± 0.001 | - | 0.015 ± 0.000 | 0.022 ± 0.014 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.020 ± 0.001 | 0.008 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.013 ± 0.003 | - | 0.016 ± 0.004 | 0.014 ± 0.004 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.030 ± 0.005 | - | 0.032 ± 0.005 | 0.043 ± 0.005 | 0.051 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.007 ± 0.001 | - | 0.007 ± 0.001 | 0.008 ± 0.001 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.017 ± 0.003 | - | 0.019 ± 0.004 | 0.017 ± 0.031 | 0.034 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.045 ± 0.008 | - | 0.046 ± 0.009 | 0.046 ± 0.007 | 0.057 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 12.865 ± 1.908 | - | 14.059 ± 1.249 | 14.172 ± 2.877 | 6.864 ± 0.797 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.008 ± 0.002 | - | 0.016 ± 0.027 | 0.010 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.959 ± 0.025 | - | 0.988 ± 0.031 | 1.024 ± 0.072 | 0.633 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.029 ± 0.029 | - | 0.031 ± 0.004 | 0.027 ± 0.000 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.082 ± 0.006 | - | 0.091 ± 0.015 | 0.084 ± 0.004 | 0.151 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.217 ± 0.008 | - | unsupported | 0.220 ± 0.002 | 1.391 ± 0.106 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.437 ± 0.079 | - | unsupported | 1.409 ± 0.061 | 2.654 ± 0.132 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.408 ± 1.287 | - | unsupported | 3.910 ± 0.233 | 5.353 ± 0.260 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.125 ± 0.045 | - | unsupported | 0.794 ± 0.033 | 0.832 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.002 | - | unsupported | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.043 ± 0.001 | - | unsupported | 0.041 ± 0.007 | 0.048 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.033 ± 0.005 | - | unsupported | 0.034 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.013 ± 0.003 | - | unsupported | 0.016 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.001 | - | unsupported | 0.010 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.019 ± 0.003 | - | unsupported | 0.020 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 9.008 ± 2.208 | - | unsupported | 6.221 ± 0.176 | 6.773 ± 0.275 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.352 ± 0.033 | - | unsupported | 0.312 ± 0.010 | 0.328 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.037 ± 0.006 | - | unsupported | 0.034 ± 0.000 | 0.059 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.215 ± 0.024 | - | unsupported | 0.253 ± 0.332 | 1.957 ± 0.327 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.379 ± 0.022 | - | unsupported | 1.497 ± 0.058 | 1.089 ± 0.311 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 4.178 ± 0.062 | - | unsupported | 4.122 ± 0.285 | 2.818 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.831 ± 0.321 | - | unsupported | 0.781 ± 0.072 | 0.828 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.006 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.043 ± 0.005 | - | unsupported | 0.111 ± 0.041 | 0.057 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | unsupported | 0.006 ± 0.000 | 0.014 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.030 ± 0.000 | - | unsupported | 0.039 ± 0.007 | 0.051 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.005 ± 0.001 | - | unsupported | 0.011 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.012 ± 0.000 | - | unsupported | 0.019 ± 0.003 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.001 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.016 ± 0.000 | - | unsupported | 0.021 ± 0.003 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 6.319 ± 0.173 | - | unsupported | 6.183 ± 0.060 | 4.262 ± 0.744 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.006 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.346 ± 0.033 | - | unsupported | 0.294 ± 0.012 | 0.326 ± 0.224 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.012 ± 0.003 | - | unsupported | 0.016 ± 0.004 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.030 ± 0.007 | - | unsupported | 0.037 ± 0.006 | 0.048 ± 0.004 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 2.401 ± 2.601 | - | unsupported | 0.973 ± 0.060 | 1.025 ± 0.017 | 3.863 ± 1.455 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.060 ± 0.009 | - | unsupported | 0.076 ± 0.021 | 0.140 ± 0.013 | 0.237 ± 0.012 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.114 ± 0.001 | - | unsupported | 0.129 ± 0.018 | 0.201 ± 0.022 | 0.167 ± 0.018 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.283 ± 0.065 | - | unsupported | 0.292 ± 0.042 | 0.423 ± 0.019 | 0.747 ± 0.089 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.882 ± 0.218 | - | unsupported | 0.958 ± 0.258 | 0.676 ± 0.147 | 3.538 ± 3.461 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.059 ± 0.003 | - | unsupported | 0.147 ± 0.021 | 0.165 ± 0.014 | 0.093 ± 0.182 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.117 ± 0.010 | - | unsupported | 0.201 ± 0.035 | 0.229 ± 0.045 | 0.208 ± 0.020 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.285 ± 0.091 | - | unsupported | 0.375 ± 0.080 | 0.230 ± 0.058 | 0.753 ± 0.114 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.236 ± 0.217 | - | unsupported | 0.211 ± 0.016 | 0.544 ± 0.142 | 0.483 ± 0.333 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.056 ± 0.012 | - | unsupported | 0.048 ± 0.009 | 0.099 ± 0.025 | 0.068 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.078 ± 0.026 | - | unsupported | 0.064 ± 0.007 | 0.126 ± 0.009 | 0.105 ± 0.014 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.084 ± 0.008 | - | unsupported | 0.100 ± 0.044 | 0.234 ± 0.035 | 0.409 ± 0.225 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.289 ± 0.309 | - | unsupported | 0.284 ± 0.296 | 0.517 ± 0.089 | 0.573 ± 0.088 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.091 ± 0.014 | - | unsupported | 0.043 ± 0.001 | 0.107 ± 0.043 | 0.076 ± 0.015 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.100 ± 0.017 | - | unsupported | 0.195 ± 0.017 | 0.123 ± 0.007 | 0.153 ± 0.078 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.110 ± 0.011 | - | unsupported | 0.291 ± 0.185 | 0.212 ± 0.009 | 0.237 ± 0.017 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.142 ± 0.009 | - | 0.457 ± 0.273 | 0.119 ± 0.018 | 0.580 ± 0.116 | 0.296 ± 0.011 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.058 ± 0.005 | - | 0.099 ± 0.010 | 0.047 ± 0.009 | 0.032 ± 0.003 | 0.042 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.134 ± 0.097 | - | 0.065 ± 0.010 | 0.048 ± 0.010 | 0.031 ± 0.002 | 0.145 ± 0.095 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.174 ± 0.049 | - | 0.092 ± 0.026 | 0.056 ± 0.016 | 0.068 ± 0.006 | 0.106 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.125 ± 0.025 | - | 0.320 ± 0.222 | 0.096 ± 0.015 | 0.232 ± 0.009 | 0.266 ± 0.018 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.043 ± 0.009 | - | 0.094 ± 0.107 | 0.032 ± 0.022 | 0.026 ± 0.003 | 0.044 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.070 ± 0.016 | - | 0.088 ± 0.142 | 0.044 ± 0.064 | 0.035 ± 0.007 | 0.052 ± 0.020 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.173 ± 0.018 | - | 0.118 ± 0.034 | 0.052 ± 0.022 | 0.127 ± 0.090 | 0.114 ± 0.015 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 14.233 ± 0.674 | - | 14.808 ± 2.619 | 24.668 ± 1.415 | 17.985 ± 1.614 | 45.177 ± 11.937 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 11.827 ± 1.796 | - | 13.521 ± 1.779 | 19.372 ± 4.319 | 23.133 ± 4.497 | 398.266 ± 395.199 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 10.673 ± 0.704 | - | 8.881 ± 0.536 | 7.914 ± 2.770 | 9.768 ± 3.746 | 24.393 ± 11.645 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 9.693 ± 0.767 | - | 9.371 ± 1.002 | 8.752 ± 0.791 | 9.197 ± 2.592 | 241.752 ± 22.183 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.432 ± 0.614 | - | 8.208 ± 0.332 | 6.995 ± 0.464 | 9.382 ± 2.314 | 7.826 ± 0.416 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 7.890 ± 1.521 | - | 7.307 ± 2.254 | 7.131 ± 1.348 | 9.939 ± 8.082 | 318.372 ± 27.752 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.844 ± 0.310 | - | 8.936 ± 1.333 | 7.877 ± 0.268 | 11.691 ± 5.558 | 8.861 ± 0.907 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 8.368 ± 0.479 | - | 10.147 ± 3.053 | 8.295 ± 0.555 | 7.861 ± 0.270 | 263.705 ± 14.691 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 27.372 ± 5.729 | - | 28.292 ± 5.953 | 21.643 ± 0.220 | 34.567 ± 15.194 | 15.008 ± 0.507 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 25.031 ± 5.270 | - | 25.149 ± 3.020 | 20.242 ± 1.243 | 32.716 ± 3.602 | 635.995 ± 345.742 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.469 ± 0.145 | 0.361 ± 0.038 | unsupported | 0.352 ± 0.035 | 0.444 ± 0.141 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 4.698 ± 0.743 | 4.567 ± 0.500 | unsupported | 4.620 ± 0.966 | 5.551 ± 0.722 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.306 ± 0.020 | 0.315 ± 0.068 | unsupported | 0.314 ± 0.003 | 0.412 ± 0.073 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.708 ± 0.863 | 3.896 ± 0.076 | unsupported | 4.188 ± 0.460 | 4.782 ± 1.148 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 9.024 ± 0.649 | - | 12.280 ± 4.089 | 12.110 ± 1.128 | 10.380 ± 0.332 | 33.370 ± 2.843 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 9.940 ± 0.684 | - | 12.516 ± 4.612 | 9.486 ± 0.334 | 10.052 ± 2.009 | 378.985 ± 309.937 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 9.796 ± 1.866 | 13.549 ± 5.638 | 6.633 ± 0.555 | 30.468 ± 8.569 | 10.419 ± 3.556 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 11.730 ± 2.837 | 11.138 ± 2.587 | 6.208 ± 1.369 | 26.814 ± 1.911 | 158.003 ± 52.931 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 12.791 ± 4.144 | - | 10.451 ± 1.175 | 16.538 ± 2.501 | 15.610 ± 3.959 | 35.464 ± 22.529 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 9.910 ± 0.548 | - | 10.615 ± 4.432 | 10.067 ± 3.613 | 14.625 ± 2.621 | 260.821 ± 28.808 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.929 ± 0.141 | - | 0.982 ± 0.440 | 2.508 ± 0.181 | 5.856 ± 0.157 | 6.579 ± 0.697 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.879 ± 0.332 | - | 0.761 ± 0.076 | 2.655 ± 0.121 | 4.559 ± 0.648 | 6.825 ± 3.268 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 15.309 ± 4.427 | - | 14.514 ± 4.188 | 13.444 ± 1.728 | 13.367 ± 0.576 | 22.976 ± 5.130 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 14.113 ± 3.353 | - | 12.170 ± 0.503 | 12.772 ± 1.929 | 13.844 ± 4.748 | 980.355 ± 497.396 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 13.095 ± 3.097 | - | 19.068 ± 9.120 | 14.722 ± 0.887 | 15.587 ± 5.721 | 27.570 ± 8.513 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 13.245 ± 1.618 | - | 13.363 ± 3.783 | 11.589 ± 0.695 | 15.748 ± 3.706 | 1042.194 ± 256.179 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.934 ± 0.328 | 1.731 ± 0.155 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 14.611 ± 1.093 | 16.242 ± 2.268 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.876 ± 0.266 | 1.674 ± 0.162 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 15.341 ± 1.344 | 14.245 ± 0.619 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.686 ± 0.316 | 1.585 ± 0.183 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 14.502 ± 2.236 | 16.122 ± 3.630 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.654 ± 0.185 | 1.519 ± 0.304 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 13.780 ± 0.585 | 13.248 ± 0.550 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 8.844 ± 0.405 | - | 9.713 ± 4.002 | 8.655 ± 1.934 | 9.273 ± 0.272 | 22.787 ± 1.870 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 13.120 ± 4.127 | - | 9.818 ± 4.144 | 7.293 ± 0.761 | 6.928 ± 1.185 | 231.662 ± 42.118 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 45.673 ± 13.135 | 45.463 ± 5.449 | 44.328 ± 0.698 | 38.010 ± 5.094 | 83.699 ± 6.658 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 42.342 ± 8.236 | 45.144 ± 9.398 | 38.758 ± 8.157 | 37.568 ± 2.919 | 1810.336 ± 104.621 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 18.455 ± 2.113 | - | 27.922 ± 1.878 | 12.978 ± 0.842 | 80.251 ± 15.787 | 35.434 ± 1.912 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 24.032 ± 7.476 | - | 35.458 ± 10.905 | 13.743 ± 1.638 | 53.877 ± 8.648 | 26.584 ± 7.977 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.336 ± 0.128 | 0.223 ± 0.019 | unsupported | 0.078 ± 0.014 | 0.035 ± 0.002 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 3.879 ± 0.708 | 3.406 ± 0.880 | unsupported | 0.877 ± 0.129 | 0.488 ± 0.024 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.237 ± 0.169 | 0.239 ± 0.145 | unsupported | 0.125 ± 0.010 | 0.060 ± 0.029 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 1.515 ± 0.031 | 1.520 ± 0.400 | unsupported | 0.784 ± 0.174 | 0.620 ± 0.331 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.107 ± 0.021 | 0.112 ± 0.018 | unsupported | 0.109 ± 0.007 | 0.232 ± 0.027 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.521 ± 0.174 | 1.683 ± 0.563 | unsupported | 1.434 ± 0.072 | 3.837 ± 0.389 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.113 ± 0.012 | 0.115 ± 0.010 | unsupported | 0.164 ± 0.007 | 0.187 ± 0.040 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.816 ± 0.325 | 0.716 ± 0.111 | unsupported | 0.832 ± 0.068 | 1.821 ± 0.293 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.417 ± 0.034 | 0.322 ± 0.150 | unsupported | 0.389 ± 0.022 | 0.091 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 5.650 ± 0.479 | 5.323 ± 0.413 | unsupported | 5.744 ± 1.091 | 1.920 ± 0.341 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.267 ± 0.054 | 0.282 ± 0.027 | unsupported | 0.354 ± 0.063 | 0.105 ± 0.027 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 2.346 ± 0.303 | 2.355 ± 0.224 | unsupported | 3.288 ± 0.925 | 1.029 ± 0.398 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.064 ± 0.037 | 0.109 ± 0.013 | unsupported | 0.061 ± 0.009 | 0.091 ± 0.014 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 1.205 ± 0.477 | 0.857 ± 0.121 | unsupported | 0.745 ± 0.071 | 1.040 ± 0.191 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.168 ± 0.069 | 0.153 ± 0.023 | unsupported | 0.073 ± 0.018 | 0.098 ± 0.012 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.817 ± 0.110 | 0.892 ± 0.168 | unsupported | 0.641 ± 0.217 | 1.200 ± 0.264 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.141 ± 0.017 | 0.165 ± 0.014 | unsupported | 0.108 ± 0.010 | 0.116 ± 0.014 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.605 ± 0.421 | 1.877 ± 1.065 | unsupported | 1.646 ± 0.241 | 1.907 ± 0.172 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.293 ± 0.159 | 0.187 ± 0.060 | unsupported | 0.103 ± 0.039 | 0.227 ± 0.085 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 1.333 ± 0.392 | 1.229 ± 0.199 | unsupported | 1.114 ± 0.685 | 1.216 ± 0.434 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.196 ± 0.036 | 0.189 ± 0.010 | unsupported | unsupported | 0.127 ± 0.003 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 2.474 ± 1.147 | 2.512 ± 0.969 | unsupported | unsupported | 2.402 ± 0.535 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.217 ± 0.050 | 0.270 ± 0.021 | unsupported | unsupported | 0.130 ± 0.047 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.554 ± 0.214 | 1.542 ± 0.299 | unsupported | unsupported | 1.732 ± 0.342 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.014 ± 0.002 | 0.021 ± 0.004 | unsupported | 0.009 ± 0.004 | 0.054 ± 0.003 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.124 ± 0.017 | 0.134 ± 0.013 | unsupported | 0.075 ± 0.022 | 0.374 ± 0.090 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.038 ± 0.011 | 0.055 ± 0.036 | unsupported | 0.076 ± 0.015 | 0.070 ± 0.017 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.112 ± 0.023 | 0.112 ± 0.025 | unsupported | 0.168 ± 0.068 | 0.215 ± 0.074 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.047 ± 0.022 | 0.054 ± 0.013 | unsupported | 0.035 ± 0.008 | 0.070 ± 0.015 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.608 ± 0.279 | 0.478 ± 0.088 | unsupported | 0.341 ± 0.085 | 0.866 ± 0.104 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.052 ± 0.020 | 0.086 ± 0.102 | unsupported | 0.174 ± 0.040 | 0.078 ± 0.012 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.404 ± 0.013 | 0.497 ± 0.203 | unsupported | 0.698 ± 0.122 | 0.530 ± 0.352 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.056 ± 0.019 | 0.065 ± 0.032 | unsupported | 0.101 ± 0.015 | 0.089 ± 0.006 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.981 ± 0.343 | 0.856 ± 0.060 | unsupported | 1.309 ± 0.716 | 0.818 ± 0.019 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.077 ± 0.005 | 0.075 ± 0.044 | unsupported | 0.168 ± 0.033 | 0.167 ± 0.027 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.447 ± 0.011 | 0.563 ± 0.221 | unsupported | 0.741 ± 0.273 | 0.849 ± 0.566 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.063 ± 0.013 | 0.059 ± 0.002 | unsupported | 0.098 ± 0.013 | 0.091 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.835 ± 0.077 | 0.953 ± 0.175 | unsupported | 1.524 ± 0.390 | 0.916 ± 0.058 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.085 ± 0.004 | 0.082 ± 0.009 | unsupported | 0.176 ± 0.048 | 0.178 ± 0.032 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.517 ± 0.079 | 0.533 ± 0.053 | unsupported | 0.727 ± 0.143 | 0.598 ± 0.442 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.133 ± 0.033 | 0.147 ± 0.039 | unsupported | 0.116 ± 0.023 | 0.117 ± 0.003 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.581 ± 0.087 | 1.703 ± 0.338 | unsupported | 1.792 ± 0.198 | 2.266 ± 1.142 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.281 ± 0.088 | 0.378 ± 0.029 | unsupported | 0.099 ± 0.040 | 0.152 ± 0.023 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.277 ± 0.351 | 1.361 ± 0.143 | unsupported | 1.303 ± 0.121 | 0.948 ± 0.320 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.296 ± 0.053 | 0.247 ± 0.046 | unsupported | 0.370 ± 0.033 | 0.314 ± 0.004 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 4.496 ± 0.408 | 3.770 ± 0.684 | unsupported | 5.655 ± 0.739 | 6.451 ± 0.981 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.249 ± 0.019 | 0.256 ± 0.030 | unsupported | 0.320 ± 0.078 | 0.290 ± 0.071 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.902 ± 0.077 | 1.918 ± 0.165 | unsupported | 3.051 ± 0.561 | 4.178 ± 1.187 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.909 ± 0.149 | 0.866 ± 0.119 | unsupported | 0.175 ± 0.015 | 0.099 ± 0.009 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 15.867 ± 4.253 | 14.932 ± 4.048 | unsupported | 2.824 ± 0.592 | 1.388 ± 0.310 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.586 ± 0.145 | 1.056 ± 0.559 | unsupported | 0.286 ± 0.040 | 0.205 ± 0.030 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 7.277 ± 0.502 | 12.567 ± 2.499 | unsupported | 2.202 ± 0.469 | 1.610 ± 0.306 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 21.758 ± 4.055 | - | unsupported | 18.286 ± 1.725 | - | 28.942 ± 3.259 | 27.287 ± 7.339 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 23.353 ± 7.611 | - | unsupported | 23.020 ± 3.970 | - | 19.561 ± 4.822 | 20.493 ± 3.974 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 14.121 ± 5.235 | - | unsupported | 12.077 ± 0.609 | - | 18.611 ± 4.822 | 22.302 ± 7.413 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 19.278 ± 3.057 | - | unsupported | 13.999 ± 3.661 | - | 23.733 ± 7.575 | 14.944 ± 2.642 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 11.100 ± 1.060 | - | unsupported | 11.651 ± 0.097 | - | 16.272 ± 4.803 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 11.697 ± 0.980 | - | unsupported | 17.245 ± 1.560 | - | 12.959 ± 0.981 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 21.054 ± 3.446 | - | unsupported | 20.891 ± 2.481 | - | 24.888 ± 4.522 | 23.416 ± 5.224 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 26.824 ± 3.319 | - | unsupported | 20.848 ± 0.410 | - | 21.837 ± 6.056 | 19.149 ± 2.502 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 12.715 ± 1.270 | - | unsupported | 10.724 ± 0.403 | - | 84.314 ± 18.473 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 12.144 ± 0.341 | - | unsupported | 13.414 ± 6.709 | - | 26.885 ± 5.651 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 13.814 ± 5.350 | - | unsupported | 11.918 ± 0.836 | - | 61.677 ± 6.428 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 12.743 ± 1.314 | - | unsupported | 14.081 ± 0.710 | - | 24.540 ± 5.141 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 19.367 ± 0.791 | - | unsupported | 16.936 ± 0.783 | - | 26.275 ± 7.998 | 25.175 ± 7.856 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 22.617 ± 3.050 | - | unsupported | 21.345 ± 0.487 | - | 29.579 ± 8.855 | 24.612 ± 11.388 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 11.966 ± 2.137 | - | unsupported | 11.394 ± 1.794 | - | 16.005 ± 6.195 | 22.757 ± 10.048 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 16.134 ± 1.664 | - | unsupported | 13.348 ± 1.867 | - | 13.578 ± 1.812 | 11.305 ± 0.757 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.312 ± 0.282 | unsupported | unsupported | 0.299 ± 0.082 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 3.463 ± 0.183 | unsupported | unsupported | 4.426 ± 1.871 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.244 ± 0.053 | unsupported | unsupported | 0.449 ± 0.185 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 3.592 ± 0.619 | unsupported | unsupported | 4.165 ± 0.510 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 19.270 ± 1.626 | - | unsupported | 18.052 ± 1.441 | - | 22.913 ± 9.960 | 23.810 ± 13.683 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 23.236 ± 2.511 | - | unsupported | 20.805 ± 2.987 | - | 24.809 ± 4.003 | 30.777 ± 15.552 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 6.194 ± 0.412 | - | 0.002 ± 0.000 | 6.220 ± 0.711 | 11.697 ± 10.352 | 13.352 ± 0.163 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 9.024 ± 1.569 | - | 0.003 ± 0.000 | 5.293 ± 0.996 | 6.998 ± 0.798 | 16.779 ± 4.200 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 10.687 ± 3.249 | - | 23.079 ± 2.264 | 10.237 ± 1.466 | 16.758 ± 4.032 | 10.999 ± 1.413 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 10.293 ± 0.832 | - | 21.434 ± 2.139 | 10.851 ± 0.705 | 9.354 ± 0.903 | 8.967 ± 0.318 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 13.919 ± 1.980 | - | 13.647 ± 1.946 | 11.265 ± 1.892 | 27.757 ± 6.588 | 11.265 ± 1.048 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 15.368 ± 5.763 | - | 13.454 ± 2.519 | 13.595 ± 1.188 | 11.913 ± 1.904 | 12.758 ± 2.672 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 7.186 ± 0.736 | - | 8.345 ± 5.315 | 9.918 ± 0.891 | 19.022 ± 10.166 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 5.450 ± 0.394 | - | 5.302 ± 0.680 | 11.307 ± 0.897 | 11.994 ± 1.653 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 15.965 ± 2.895 | - | 0.003 ± 0.000 | 14.308 ± 2.752 | 20.282 ± 12.243 | 10.140 ± 0.806 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 13.094 ± 1.665 | - | 0.003 ± 0.000 | 13.141 ± 0.748 | 11.798 ± 0.441 | 13.063 ± 3.328 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 52.431 ± 8.896 | - | 0.002 ± 0.006 | 49.840 ± 4.904 | 90.742 ± 11.727 | 235.316 ± 36.217 | 29.457 ± 2.474 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 22.309 ± 4.307 | - | 0.003 ± 0.008 | 50.632 ± 4.363 | 36.230 ± 5.693 | 226.647 ± 9.983 | 25.979 ± 9.318 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 9.469 ± 1.792 | - | 10.097 ± 1.294 | 8.342 ± 0.302 | 6.818 ± 1.013 | 9.168 ± 1.594 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 6.788 ± 0.747 | - | 7.757 ± 1.118 | 7.148 ± 1.240 | 6.790 ± 0.475 | 9.117 ± 0.993 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 8.811 ± 6.634 | - | 10.596 ± 1.074 | 7.828 ± 0.537 | 9.061 ± 6.849 | 8.607 ± 0.455 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 7.624 ± 2.212 | - | 7.805 ± 1.868 | 6.950 ± 0.205 | 6.006 ± 2.448 | 9.926 ± 0.523 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.002 ± 0.002 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.002 ± 0.007 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.989292 | 436830.75 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 6.165833 | 385364.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 6.180791 | 386299.44 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 7.792167 | 487010.44 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 9.936042 | 621002.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 12.599833 | 787489.56 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 14.347750 | 896734.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 18.459708 | 1153731.75 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 15.522917 | 970182.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 14.561708 | 910106.75 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 7.097250 | 443578.12 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 5.638250 | 352390.62 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 7.505458 | 469091.12 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 5.768167 | 360510.44 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 88.815458 | 5550966.12 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 73.922291 | 4620143.19 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 75.169291 | 4698080.69 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 73.076250 | 4567265.62 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.596958 | 412309.88 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.030833 | 314427.06 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 4.898208 | 306138.00 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 5.039625 | 314976.56 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 76.517542 | 4782346.38 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 67.003584 | 4187724.00 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 59.322958 | 3707684.88 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 62.333333 | 3895833.31 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 30.942875 | 1933929.69 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 27.691042 | 1730690.12 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 233.778708 | 14611169.25 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 259.877292 | 16242330.75 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 30.017083 | 1876067.69 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 26.789250 | 1674328.12 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 245.455875 | 15340992.19 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 227.912292 | 14244518.25 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 26.972208 | 1685763.00 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 25.367417 | 1585463.56 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 232.025583 | 14501598.94 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 257.950500 | 16121906.25 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 26.458500 | 1653656.25 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 24.299959 | 1518747.44 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 220.474667 | 13779666.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 211.962583 | 13247661.44 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.242166 | 35033.84 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.969583 | 77649.73 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 21.532000 | 336437.50 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 14.258708 | 222792.31 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.903625 | 487953.12 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.018792 | 877349.00 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 31.032125 | 3879015.62 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 27.246167 | 3405770.88 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.822125 | 59720.70 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.019584 | 125306.00 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 15.169209 | 237018.89 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 15.322083 | 239407.55 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.961083 | 620135.38 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.272209 | 784026.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 12.120500 | 1515062.50 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 12.157458 | 1519682.25 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.841459 | 231897.80 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.001667 | 109401.05 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.828875 | 106701.17 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.185125 | 112267.58 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 30.697292 | 3837161.50 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.473500 | 1434187.50 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 12.166167 | 1520770.88 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 13.460625 | 1682578.12 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.963542 | 186930.34 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.465500 | 163523.44 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.241166 | 113143.22 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.332750 | 114574.22 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.566291 | 1820786.38 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.655875 | 831984.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.527666 | 815958.25 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.730625 | 716328.12 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.823333 | 90989.58 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 24.889625 | 388900.39 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 26.708083 | 417313.80 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 20.583083 | 321610.67 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.360375 | 1920046.88 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 45.950917 | 5743864.62 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 45.200667 | 5650083.38 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 42.583542 | 5322942.75 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.699916 | 104686.19 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 22.652500 | 353945.31 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 17.098375 | 267162.11 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 18.079833 | 282497.39 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.233000 | 1029125.00 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 26.306333 | 3288291.62 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 18.770667 | 2346333.38 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 18.839792 | 2354974.00 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.813750 | 90839.84 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.935375 | 61490.23 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.114584 | 64290.38 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.005167 | 109455.73 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.318000 | 1039750.00 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.960875 | 745109.38 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.643000 | 1205375.00 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.859916 | 857489.50 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.302208 | 98472.00 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.698875 | 73419.92 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.759542 | 168117.84 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.819333 | 153427.08 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.598208 | 1199776.00 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.128750 | 641093.75 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.536000 | 817000.00 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.139000 | 892375.00 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.406667 | 115729.17 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.908375 | 107943.36 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 9.001750 | 140652.34 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.574250 | 165222.66 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.258375 | 1907296.88 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.171708 | 1646463.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 12.837792 | 1604724.00 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 15.013167 | 1876645.88 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.537292 | 227145.19 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.596875 | 103076.17 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 18.772750 | 293324.22 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 11.989375 | 187333.98 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.729333 | 1216166.62 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.910459 | 1113807.38 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 10.663875 | 1332984.38 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.832208 | 1229026.00 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.108958 | 126702.47 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.565459 | 196335.30 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.079250 | 188738.28 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 19.219666 | 2402458.25 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 19.794333 | 2474291.62 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 20.097041 | 2512130.12 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.304250 | 129753.91 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 13.911875 | 217373.05 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 17.280791 | 270012.36 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.856709 | 1732088.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 12.434291 | 1554286.38 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 12.337167 | 1542145.88 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.469750 | 54214.84 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.561250 | 8769.53 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.867916 | 13561.19 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.313792 | 20528.00 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.993208 | 374151.00 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.603709 | 75463.62 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.990542 | 123817.75 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.072584 | 134073.00 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.456375 | 69630.86 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.860292 | 75942.06 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.433292 | 38020.19 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.546542 | 55414.72 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.720583 | 215072.88 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.343875 | 167984.38 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.899792 | 112474.00 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.893042 | 111630.25 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.485750 | 70089.84 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.215125 | 34611.33 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.989208 | 46706.38 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.456625 | 54009.77 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.931625 | 866453.12 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 2.729042 | 341130.25 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.863167 | 607895.88 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.825500 | 478187.50 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.990959 | 77983.73 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.104917 | 173514.33 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.356709 | 52448.58 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.494625 | 85853.52 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.242792 | 530349.00 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.584958 | 698119.75 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.229625 | 403703.12 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.974542 | 496817.75 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.721583 | 89399.73 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.468583 | 101071.61 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.610041 | 56406.89 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.132875 | 64576.17 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.542500 | 817812.50 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.475250 | 1309406.25 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.847750 | 980968.75 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.848000 | 856000.00 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.717041 | 167453.77 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.742292 | 167848.31 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.904333 | 76630.20 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.822000 | 75343.75 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.795791 | 849473.88 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.924792 | 740599.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.576209 | 447026.12 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.506792 | 563349.00 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.813250 | 90832.03 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.240666 | 97510.41 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.012084 | 62688.81 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.796708 | 59323.56 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.331708 | 916463.50 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.193625 | 1524203.12 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.682708 | 835338.50 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.620291 | 952536.38 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.388583 | 177946.61 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.246792 | 175731.12 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.413291 | 84582.67 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.265416 | 82272.12 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.783875 | 597984.38 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.816708 | 727088.50 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.134209 | 516776.13 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.266833 | 533354.12 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.497709 | 117151.70 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.446750 | 116355.47 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.497125 | 132767.58 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.379500 | 146554.69 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.125375 | 2265671.88 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.335500 | 1791937.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 12.647584 | 1580948.00 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 13.622625 | 1702828.12 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.740875 | 152201.17 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.366000 | 99468.75 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 17.982291 | 280973.30 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 24.200959 | 378139.98 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.580250 | 947531.25 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 10.421625 | 1302703.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 10.215917 | 1276989.62 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.887083 | 1360885.38 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 20.083667 | 313807.30 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 23.701375 | 370333.98 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.916584 | 295571.62 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 15.780958 | 246577.47 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 51.610083 | 6451260.38 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 45.242583 | 5655322.88 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 35.964958 | 4495619.75 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 30.162959 | 3770369.88 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 18.534750 | 289605.47 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 20.491833 | 320184.89 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 15.946708 | 249167.31 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 16.385584 | 256024.75 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 33.423917 | 4177989.63 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 24.410416 | 3051302.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 15.215916 | 1901989.50 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.344583 | 1918072.88 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.355541 | 99305.33 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 11.208333 | 175130.20 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 58.207875 | 909498.05 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 55.445292 | 866332.69 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.102334 | 1387791.75 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 22.588292 | 2823536.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 126.935458 | 15866932.25 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 119.457750 | 14932218.75 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.110958 | 204858.72 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 18.317458 | 286210.28 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 37.514125 | 586158.20 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 67.585084 | 1056016.94 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 12.880416 | 1610052.00 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 17.618333 | 2202291.62 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 58.217042 | 7277130.25 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 100.533541 | 12566692.62 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 4.785541 | 299096.31 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 4.993375 | 312085.94 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 70.819666 | 4426229.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 55.400208 | 3462513.00 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 7.176750 | 448546.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3.903375 | 243960.94 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 66.639500 | 4164968.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 57.464667 | 3591541.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012458 | 778.62 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001250 | 78.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.016375 | 1023.44 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001250 | 78.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012166 | 760.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.027792 | 1737.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.013500 | 843.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.031458 | 1966.12 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000208 | 13.00 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.009875 | 617.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.003500 | 218.75 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.010667 | 666.69 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.002375 | 148.44 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000209 | 13.06 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.010708 | 669.25 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002667 | 166.69 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.012917 | 807.31 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002917 | 182.31 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 55.2x faster than the slowest successful cell (`julia-base`, 79.803 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 17.2x faster than the slowest successful cell (`julia-base`, 79.803 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 18.9x faster than the slowest successful cell (`julia-base`, 79.803 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 18.2x faster than the slowest successful cell (`julia-base`, 79.803 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 23.3x faster than the slowest successful cell (`julia-base`, 473.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 41.1x faster than the slowest successful cell (`julia-base`, 473.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 40.4x faster than the slowest successful cell (`julia-base`, 473.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 39.7x faster than the slowest successful cell (`julia-base`, 473.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 30.2x faster than the slowest successful cell (`julia-base`, 216.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 24.1x faster than the slowest successful cell (`julia-base`, 216.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 26.5x faster than the slowest successful cell (`julia-base`, 216.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 26.1x faster than the slowest successful cell (`julia-base`, 216.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 106.9x faster than the slowest successful cell (`julia-base`, 250.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 44.7x faster than the slowest successful cell (`julia-base`, 250.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 41.5x faster than the slowest successful cell (`julia-base`, 250.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 44.3x faster than the slowest successful cell (`julia-base`, 250.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 102.6x faster than the slowest successful cell (`julia-base`, 729.791 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 87.1x faster than the slowest successful cell (`julia-base`, 729.791 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 90.3x faster than the slowest successful cell (`julia-base`, 729.791 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 91.5x faster than the slowest successful cell (`julia-base`, 729.791 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 15.419 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 15.419 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 15.419 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 12.551 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 12.551 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 12.551 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 24.1x faster than the slowest successful cell (`pytorch-cpu`, 16.957 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 16.957 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 16.957 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 12.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 12.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 308.4x faster than the slowest successful cell (`julia-base`, 1.863 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 638.6x faster than the slowest successful cell (`julia-base`, 1.729 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 19.3x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 13.0x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 41.0x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x2x2`): `tenferro-direct` is 14.4x faster than the slowest successful cell (`tenferro-trace`, 0.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x32x32`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.133 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 21.9x faster than the slowest successful cell (`jax-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `jax-cpu` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 135.3x faster than the slowest successful cell (`pytorch-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 32.6x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `jax-cpu` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 123.6x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 22.4x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 57.4x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 117.6x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 21.2x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 86.7x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 23.2x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `jax-cpu` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 127.6x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 23.2x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 62.3x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 20.8x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 52.1x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 29.9x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`1x2x2`): `tenferro-direct` is 23.5x faster than the slowest successful cell (`jax-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`1x2x2`): `tenferro-direct` is 20.5x faster than the slowest successful cell (`pytorch-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 16.9x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 25.0x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 21.3x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 21.2x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `jax-cpu` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 46.6x faster than the slowest successful cell (`pytorch-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`pytorch-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 16.6x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 25.8x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 20.2x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 19.2x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 25.5x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 25.7x faster than the slowest successful cell (`pytorch-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 28.8x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x32x32,rhs=1`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x8x8,rhs=1`): `tenferro-direct` is 17.8x faster than the slowest successful cell (`jax-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 28.2x faster than the slowest successful cell (`pytorch-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x8x8,rhs=1`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 20.7x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 17.2x faster than the slowest successful cell (`julia-base`, 398.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 20.6x faster than the slowest successful cell (`julia-base`, 398.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 33.7x faster than the slowest successful cell (`julia-base`, 398.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 29.5x faster than the slowest successful cell (`julia-base`, 398.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 26.3x faster than the slowest successful cell (`julia-base`, 241.752 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 27.6x faster than the slowest successful cell (`julia-base`, 241.752 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 24.9x faster than the slowest successful cell (`julia-base`, 241.752 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 25.8x faster than the slowest successful cell (`julia-base`, 241.752 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 32.0x faster than the slowest successful cell (`julia-base`, 318.372 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 44.6x faster than the slowest successful cell (`julia-base`, 318.372 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 40.4x faster than the slowest successful cell (`julia-base`, 318.372 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 43.6x faster than the slowest successful cell (`julia-base`, 318.372 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 33.5x faster than the slowest successful cell (`julia-base`, 263.705 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 31.8x faster than the slowest successful cell (`julia-base`, 263.705 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 31.5x faster than the slowest successful cell (`julia-base`, 263.705 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 26.0x faster than the slowest successful cell (`julia-base`, 263.705 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 19.4x faster than the slowest successful cell (`julia-base`, 635.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 31.4x faster than the slowest successful cell (`julia-base`, 635.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 25.4x faster than the slowest successful cell (`julia-base`, 635.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 25.3x faster than the slowest successful cell (`julia-base`, 635.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 37.7x faster than the slowest successful cell (`julia-base`, 378.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 40.0x faster than the slowest successful cell (`julia-base`, 378.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 38.1x faster than the slowest successful cell (`julia-base`, 378.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 30.3x faster than the slowest successful cell (`julia-base`, 378.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 25.5x faster than the slowest successful cell (`julia-base`, 158.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 13.5x faster than the slowest successful cell (`julia-base`, 158.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`julia-base`, 158.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 17.8x faster than the slowest successful cell (`julia-base`, 260.821 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 25.9x faster than the slowest successful cell (`julia-base`, 260.821 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 26.3x faster than the slowest successful cell (`julia-base`, 260.821 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 24.6x faster than the slowest successful cell (`julia-base`, 260.821 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 70.8x faster than the slowest successful cell (`julia-base`, 980.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 76.8x faster than the slowest successful cell (`julia-base`, 980.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 69.5x faster than the slowest successful cell (`julia-base`, 980.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 80.6x faster than the slowest successful cell (`julia-base`, 980.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 66.2x faster than the slowest successful cell (`julia-base`, 1042.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 89.9x faster than the slowest successful cell (`julia-base`, 1042.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 78.7x faster than the slowest successful cell (`julia-base`, 1042.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 78.0x faster than the slowest successful cell (`julia-base`, 1042.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 33.4x faster than the slowest successful cell (`julia-base`, 231.662 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 31.8x faster than the slowest successful cell (`julia-base`, 231.662 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 17.7x faster than the slowest successful cell (`julia-base`, 231.662 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 23.6x faster than the slowest successful cell (`julia-base`, 231.662 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 48.2x faster than the slowest successful cell (`julia-base`, 1810.336 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 46.7x faster than the slowest successful cell (`julia-base`, 1810.336 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 42.8x faster than the slowest successful cell (`julia-base`, 1810.336 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 40.1x faster than the slowest successful cell (`julia-base`, 1810.336 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 11.4x faster than the slowest successful cell (`tenferro-direct`, 15.867 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 5934.3x faster than the slowest successful cell (`julia-base`, 13.352 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 6009.8x faster than the slowest successful cell (`julia-base`, 16.779 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 7489.6x faster than the slowest successful cell (`jax-cpu`, 20.282 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 4505.0x faster than the slowest successful cell (`pytorch-cpu`, 13.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 99080.3x faster than the slowest successful cell (`julia-base`, 235.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`julia-base`, 226.647 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 86341.7x faster than the slowest successful cell (`julia-base`, 226.647 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 347.4x faster than the slowest successful cell (`tenferro-direct`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 245.8x faster than the slowest successful cell (`tenferro-direct`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 47.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 83.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 51.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 161.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 2775.9 GB/s at 0.006 ms. The conservative physical lower bound is 0.168 ms, so the cell is 27.8x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 6192.4 GB/s at 0.003 ms. The conservative physical lower bound is 0.042 ms, so the cell is 15.5x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 119304.6 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1193.0x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 96144.5 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 240.4x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 99126.8 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 991.3x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 92024.5 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 230.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 56512.7 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 565.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 51130.6 GB/s at 0.003 ms. The conservative physical lower bound is 0.336 ms, so the cell is 127.8x faster than the bound.

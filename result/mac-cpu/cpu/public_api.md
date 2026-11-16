# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261116_175245/run.yaml`
- Timestamp: `2026-11-16T17:52:45.053792+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261116_175245`.

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

- CSV: `data/results/mac-cpu/cpu/public_api/20261116_175245/cpu_public_api_t1_20261116_175245.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261116_175245/cpu_public_api_t4_20261116_175245.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.633 ± 0.226 | - | 3.771 ± 0.131 | 4.287 ± 0.148 | 1.160 ± 0.164 | 3.826 ± 0.233 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.913 ± 0.065 | - | 3.565 ± 0.157 | 3.512 ± 0.180 | 1.057 ± 0.020 | 2.960 ± 3.602 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 8.771 ± 1.619 | - | 10.673 ± 1.729 | 10.535 ± 1.584 | 7.445 ± 1.529 | 8.289 ± 0.461 | 7.322 ± 0.576 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 5.430 ± 0.633 | - | 5.304 ± 0.940 | 5.266 ± 0.727 | 4.058 ± 0.128 | 7.828 ± 0.240 | 4.915 ± 2.048 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 51.642 ± 5.064 | - | 50.305 ± 0.874 | 21.391 ± 3.499 | 32.061 ± 3.557 | 43.309 ± 2.190 | 46.557 ± 1.111 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 10.312 ± 0.725 | - | 9.896 ± 0.284 | 7.828 ± 0.151 | 11.244 ± 0.150 | 39.740 ± 3.069 | 11.472 ± 0.036 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 9.296 ± 0.467 | - | 9.232 ± 0.379 | 8.946 ± 0.736 | 48.073 ± 2.743 | 50.015 ± 0.846 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 8.723 ± 1.838 | - | 8.251 ± 0.796 | 7.945 ± 0.280 | 11.576 ± 1.686 | 13.979 ± 1.943 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 10.527 ± 0.786 | - | unsupported | 9.863 ± 0.354 | 48.775 ± 3.409 | 50.467 ± 1.082 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 8.374 ± 0.362 | - | unsupported | 8.262 ± 0.315 | 11.609 ± 0.281 | 14.771 ± 1.466 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 9.799 ± 1.928 | - | 10.114 ± 0.645 | 10.420 ± 0.274 | 17.130 ± 1.678 | 8.414 ± 1.061 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 9.907 ± 0.578 | - | 9.688 ± 0.380 | 9.041 ± 0.155 | 16.221 ± 0.394 | 11.681 ± 3.973 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 37.205 ± 5.767 | - | 43.628 ± 9.329 | 37.966 ± 1.713 | 42.098 ± 4.600 | 45.159 ± 2.950 | 43.915 ± 1.224 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 16.117 ± 7.464 | - | 9.999 ± 3.274 | 14.761 ± 0.293 | 12.954 ± 0.067 | 39.368 ± 2.706 | 11.061 ± 0.431 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 51.786 ± 4.353 | - | 50.627 ± 0.909 | 51.017 ± 2.789 | 85.079 ± 9.227 | 69.233 ± 5.875 | 60.978 ± 3.614 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 10.979 ± 1.451 | - | 10.565 ± 0.436 | 18.713 ± 1.620 | 19.915 ± 0.709 | 73.657 ± 2.687 | 19.773 ± 4.992 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 6.831 ± 0.763 | - | 7.230 ± 0.990 | 7.825 ± 0.548 | 6.731 ± 1.592 | 5.568 ± 0.101 | 5.673 ± 0.066 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 3.513 ± 0.486 | - | 3.173 ± 0.892 | 3.882 ± 0.170 | 3.447 ± 0.111 | 5.686 ± 0.092 | 3.335 ± 0.842 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 0.874 ± 0.061 | - | 7.045 ± 0.486 | 6.059 ± 0.182 | 3.773 ± 0.272 | 7.648 ± 0.277 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.653 ± 0.017 | - | 1.913 ± 0.282 | 1.484 ± 0.575 | 1.800 ± 0.030 | 9.013 ± 0.849 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 7.058 ± 0.424 | - | 6.766 ± 0.501 | 7.243 ± 0.383 | 5.432 ± 0.524 | 8.136 ± 0.835 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 7.408 ± 0.108 | - | 7.438 ± 0.205 | 6.078 ± 0.171 | 5.549 ± 0.120 | 4.681 ± 0.721 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.150 ± 0.420 | - | 4.367 ± 0.961 | 5.474 ± 0.345 | 2.049 ± 0.071 | 5.691 ± 0.114 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 5.504 ± 0.249 | - | 6.503 ± 1.157 | 4.886 ± 0.342 | 1.927 ± 0.074 | 2.715 ± 0.347 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 5.876 ± 0.410 | - | 5.890 ± 0.127 | 6.095 ± 0.171 | 3.750 ± 0.412 | 6.627 ± 0.262 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 6.373 ± 0.300 | - | 6.284 ± 0.111 | 5.707 ± 0.254 | 4.328 ± 0.123 | 5.151 ± 0.587 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 8.500 ± 1.131 | - | 8.431 ± 0.839 | 9.455 ± 0.735 | 45.158 ± 2.496 | 49.926 ± 0.392 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 8.169 ± 0.509 | - | 9.031 ± 0.683 | 8.423 ± 0.528 | 11.106 ± 0.376 | 14.771 ± 0.868 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 8.537 ± 0.877 | - | unsupported | 8.204 ± 0.809 | 56.155 ± 6.998 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 6.949 ± 0.104 | - | unsupported | 7.467 ± 0.579 | 11.588 ± 0.115 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 8.926 ± 1.234 | - | 9.166 ± 0.922 | 9.160 ± 1.753 | 16.771 ± 0.803 | 8.335 ± 4.542 | 12.472 ± 2.191 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 4.454 ± 0.763 | - | 4.425 ± 1.060 | 5.532 ± 1.178 | 5.314 ± 0.890 | 8.699 ± 2.205 | 5.095 ± 0.677 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 17.275 ± 2.606 | - | 16.024 ± 2.963 | 15.476 ± 0.903 | 23.615 ± 1.154 | 13.299 ± 6.025 | 12.097 ± 0.401 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 5.333 ± 0.157 | - | 5.636 ± 0.320 | 7.333 ± 0.048 | 13.177 ± 2.673 | 12.643 ± 0.607 | 6.501 ± 0.297 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 31.797 ± 8.469 | - | 47.102 ± 10.393 | 40.699 ± 1.076 | 31.855 ± 1.527 | 51.174 ± 4.992 | 51.752 ± 2.225 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 6.893 ± 0.620 | - | 8.418 ± 0.970 | 13.854 ± 3.410 | 8.687 ± 0.068 | 54.460 ± 3.100 | 19.737 ± 5.138 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 6.441 ± 0.450 | - | 5.906 ± 0.664 | 4.916 ± 0.786 | 6.851 ± 0.879 | 4.245 ± 0.380 | 5.635 ± 0.236 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 2.629 ± 0.476 | - | 3.357 ± 0.070 | 2.690 ± 0.028 | 3.104 ± 0.058 | 4.503 ± 0.219 | 3.657 ± 0.757 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 13.455 ± 1.038 | - | 12.561 ± 1.112 | 19.993 ± 5.250 | 21.475 ± 5.680 | 13.453 ± 0.486 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 6.773 ± 1.089 | - | 6.865 ± 0.888 | 8.617 ± 0.307 | 9.234 ± 0.127 | 11.072 ± 0.390 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 42.988 ± 3.663 | - | 37.687 ± 8.093 | 36.574 ± 1.766 | 30.569 ± 2.292 | 33.502 ± 1.575 | 34.010 ± 2.804 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 7.282 ± 0.136 | - | 7.313 ± 0.187 | 14.279 ± 0.953 | 9.093 ± 0.413 | 38.389 ± 1.412 | 18.119 ± 2.022 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 14.893 ± 2.029 | - | 16.714 ± 1.278 | 15.861 ± 1.399 | 30.623 ± 17.357 | 12.890 ± 4.255 | 13.817 ± 0.697 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 7.188 ± 1.378 | - | 5.644 ± 2.481 | 7.847 ± 0.189 | 13.922 ± 1.191 | 11.035 ± 0.940 | 6.270 ± 0.130 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 19.543 ± 1.594 | - | 20.443 ± 2.110 | 29.172 ± 1.937 | 16.851 ± 4.240 | 29.432 ± 1.479 | 30.846 ± 2.816 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 4.037 ± 0.100 | - | 4.093 ± 0.189 | 11.463 ± 0.544 | 5.026 ± 0.016 | 30.660 ± 0.651 | 15.209 ± 3.374 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 35.813 ± 2.202 | - | 32.162 ± 10.757 | 30.257 ± 1.647 | 18.036 ± 0.421 | 22.851 ± 1.681 | 26.397 ± 4.810 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 6.652 ± 0.082 | - | 6.718 ± 0.152 | 11.265 ± 0.572 | 5.028 ± 0.082 | 25.915 ± 0.922 | 11.053 ± 4.424 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 33.747 ± 4.943 | - | 47.445 ± 15.362 | 44.869 ± 2.159 | 30.034 ± 3.139 | 40.160 ± 1.180 | 38.443 ± 1.912 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 7.030 ± 0.191 | - | 6.996 ± 0.090 | 16.144 ± 3.264 | 8.195 ± 0.166 | 39.659 ± 0.548 | 15.259 ± 4.651 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 18.116 ± 5.158 | - | 19.134 ± 7.116 | 23.196 ± 1.601 | 18.816 ± 1.594 | 23.841 ± 2.741 | 23.648 ± 2.314 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.756 ± 0.232 | - | 3.886 ± 0.075 | 9.305 ± 0.353 | 5.267 ± 0.047 | 24.315 ± 0.958 | 10.352 ± 6.281 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.475 ± 0.391 | 0.353 ± 0.065 | unsupported | unsupported | 0.302 ± 0.104 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.232 ± 0.018 | 0.261 ± 0.024 | unsupported | unsupported | 0.209 ± 0.017 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 13.999 ± 1.582 | - | 14.833 ± 2.923 | 13.231 ± 2.450 | 23.796 ± 4.772 | 12.924 ± 4.394 | 16.918 ± 4.789 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 7.533 ± 1.530 | - | 7.594 ± 0.216 | 7.909 ± 1.885 | 11.827 ± 4.085 | 12.210 ± 1.984 | 8.304 ± 1.897 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 15.716 ± 1.156 | - | 15.800 ± 1.951 | 14.819 ± 1.176 | 22.009 ± 2.165 | 15.595 ± 2.367 | 14.145 ± 0.839 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 7.273 ± 1.758 | - | 8.041 ± 1.271 | 7.558 ± 0.511 | 13.512 ± 4.418 | 12.380 ± 1.914 | 7.944 ± 1.346 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 14.498 ± 0.951 | - | 16.142 ± 1.915 | 14.707 ± 2.087 | 24.377 ± 3.972 | 10.223 ± 0.299 | 13.860 ± 3.465 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 7.083 ± 0.757 | - | 6.763 ± 0.644 | 8.791 ± 1.988 | 9.804 ± 0.680 | 10.510 ± 1.172 | 5.812 ± 0.246 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 10.190 ± 1.178 | - | 8.855 ± 1.355 | 9.230 ± 1.560 | 18.605 ± 0.972 | 8.367 ± 3.108 | 13.518 ± 0.910 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 4.132 ± 0.483 | - | 3.887 ± 0.470 | 5.134 ± 0.735 | 5.145 ± 0.744 | 7.868 ± 5.535 | 5.305 ± 1.013 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 43.954 ± 2.879 | - | 51.066 ± 6.327 | 73.756 ± 5.972 | 41.399 ± 1.331 | 68.988 ± 5.213 | 82.159 ± 4.860 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 9.236 ± 0.228 | - | 9.219 ± 0.340 | 23.066 ± 0.411 | 10.993 ± 0.518 | 79.149 ± 6.385 | 27.967 ± 8.658 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 11.715 ± 3.944 | - | 9.933 ± 2.754 | 11.656 ± 1.200 | 29.136 ± 1.342 | 6.021 ± 0.196 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 3.162 ± 1.081 | - | 2.679 ± 0.892 | 4.281 ± 0.728 | 9.463 ± 0.417 | 5.670 ± 0.136 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.172 ± 0.116 | - | 1.394 ± 0.322 | 18.954 ± 4.423 | 2.297 ± 0.107 | 0.656 ± 0.074 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.323 ± 0.044 | - | 0.324 ± 1.097 | 8.570 ± 0.536 | 0.950 ± 0.069 | 0.543 ± 0.013 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.078 ± 0.117 | - | 1.161 ± 0.533 | 1.781 ± 0.280 | 0.934 ± 0.108 | 0.721 ± 0.015 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.516 ± 0.027 | - | 0.609 ± 0.380 | 0.818 ± 0.082 | 0.550 ± 0.061 | 0.918 ± 0.111 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 9.516 ± 1.038 | - | 8.617 ± 0.327 | 12.337 ± 1.155 | 28.214 ± 0.733 | 5.934 ± 0.261 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 2.430 ± 0.138 | - | 2.914 ± 0.616 | 4.533 ± 0.724 | 10.128 ± 0.530 | 5.323 ± 0.092 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.764 ± 0.715 | - | 2.176 ± 0.579 | 20.411 ± 1.128 | 2.417 ± 0.138 | 0.666 ± 0.041 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.415 ± 0.008 | - | 0.528 ± 0.021 | 8.521 ± 1.379 | 0.812 ± 0.059 | 0.654 ± 0.046 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 4.524 ± 1.045 | - | 5.683 ± 0.497 | 5.988 ± 0.575 | 3.829 ± 0.164 | 3.171 ± 0.132 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.341 ± 1.024 | - | 3.010 ± 0.366 | 2.562 ± 0.079 | 1.817 ± 0.151 | 3.012 ± 0.133 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 9.428 ± 0.712 | - | 10.784 ± 2.767 | 6.504 ± 0.927 | 45.521 ± 3.787 | 6.121 ± 0.146 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 2.862 ± 0.160 | - | 3.535 ± 1.013 | 3.249 ± 0.311 | 13.786 ± 0.247 | 5.860 ± 0.222 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.178 ± 0.412 | - | 1.027 ± 0.590 | 4.555 ± 0.470 | 2.503 ± 0.245 | 0.636 ± 0.051 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.404 ± 0.046 | - | 0.395 ± 0.013 | 1.585 ± 0.091 | 0.889 ± 0.075 | 0.570 ± 0.046 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.003 ± 0.993 | - | 1.028 ± 0.100 | 0.893 ± 0.082 | 1.234 ± 0.063 | 0.686 ± 0.083 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.585 ± 0.080 | - | 0.566 ± 0.096 | 0.383 ± 0.039 | 0.549 ± 0.042 | 0.646 ± 0.039 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 8.027 ± 1.447 | - | 11.282 ± 4.000 | 6.262 ± 0.275 | 37.289 ± 0.936 | 5.613 ± 0.729 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 3.371 ± 0.711 | - | 2.814 ± 0.106 | 3.240 ± 0.278 | 10.372 ± 0.239 | 5.669 ± 0.143 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.779 ± 0.083 | - | 1.738 ± 0.122 | 4.928 ± 0.316 | 2.399 ± 0.100 | 0.615 ± 0.032 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.545 ± 0.019 | - | 0.373 ± 0.012 | 1.630 ± 0.088 | 0.881 ± 0.101 | 0.670 ± 0.210 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.028 ± 0.096 | - | 1.119 ± 0.454 | 0.609 ± 0.168 | 1.074 ± 0.093 | 0.726 ± 0.094 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.488 ± 0.023 | - | 0.574 ± 0.179 | 0.409 ± 0.017 | 0.425 ± 0.076 | 0.631 ± 0.028 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.934 ± 0.283 | 1.169 ± 0.327 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.231 ± 0.226 | 0.207 ± 0.067 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 29.797 ± 1.833 | - | 29.994 ± 2.405 | 37.689 ± 7.721 | 39.965 ± 1.280 | 26.453 ± 6.435 | 26.968 ± 0.591 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 6.392 ± 0.124 | - | 6.366 ± 0.044 | 13.870 ± 0.618 | 11.081 ± 0.402 | 24.750 ± 1.704 | 12.076 ± 1.499 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 20.422 ± 1.393 | - | 21.689 ± 2.227 | 20.474 ± 1.348 | 39.896 ± 4.228 | 40.385 ± 4.599 | 38.315 ± 5.949 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 4.646 ± 0.411 | - | 4.584 ± 0.262 | 6.089 ± 1.091 | 12.019 ± 0.456 | 38.528 ± 0.944 | 12.924 ± 4.849 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 21.183 ± 3.453 | - | 20.360 ± 2.334 | 23.068 ± 1.116 | 27.944 ± 6.694 | 19.565 ± 6.958 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 11.586 ± 0.566 | - | 10.154 ± 1.992 | 9.328 ± 0.119 | 13.605 ± 0.074 | 16.627 ± 2.193 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 10.018 ± 0.980 | - | 10.084 ± 0.868 | 8.054 ± 1.511 | 20.494 ± 3.318 | 7.341 ± 2.795 | 16.970 ± 1.153 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 4.632 ± 0.913 | - | 4.167 ± 0.570 | 4.733 ± 1.454 | 6.792 ± 0.795 | 8.120 ± 1.962 | 6.789 ± 1.176 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 34.131 ± 8.806 | - | 31.022 ± 6.574 | 33.753 ± 2.403 | 28.767 ± 0.662 | 27.395 ± 3.087 | 28.676 ± 1.031 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 5.593 ± 0.196 | - | 5.579 ± 0.114 | 13.121 ± 1.263 | 8.106 ± 0.108 | 28.747 ± 0.401 | 11.583 ± 2.503 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 15.561 ± 1.524 | - | 14.128 ± 1.130 | 13.865 ± 2.373 | 25.471 ± 1.573 | 29.162 ± 2.497 | 29.357 ± 3.473 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 4.372 ± 0.995 | - | 4.258 ± 0.827 | 4.879 ± 0.363 | 8.098 ± 1.003 | 24.806 ± 0.475 | 8.150 ± 3.953 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 15.587 ± 2.809 | - | 15.039 ± 0.979 | 16.720 ± 3.150 | 25.906 ± 6.200 | 11.115 ± 0.786 | 11.862 ± 0.583 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 7.465 ± 1.608 | - | 6.707 ± 1.121 | 7.577 ± 1.735 | 11.303 ± 1.119 | 13.423 ± 1.330 | 6.375 ± 0.782 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 90.653 ± 7.679 | - | 82.022 ± 7.316 | 72.550 ± 1.779 | 15.279 ± 0.656 | 37.812 ± 1.037 | 37.947 ± 0.428 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 15.977 ± 1.347 | - | 15.515 ± 0.094 | 23.259 ± 1.664 | 4.483 ± 0.121 | 53.895 ± 10.457 | 17.590 ± 7.167 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 22.323 ± 3.144 | 21.644 ± 1.925 | unsupported | 25.931 ± 2.980 | 26.560 ± 8.392 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 11.230 ± 1.556 | 11.328 ± 3.033 | unsupported | 7.866 ± 0.452 | 11.914 ± 0.072 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.671 ± 0.089 | - | 0.598 ± 0.099 | 0.327 ± 0.083 | 0.360 ± 0.012 | 1.484 ± 1.628 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.279 ± 0.033 | - | 0.270 ± 0.014 | 0.431 ± 0.052 | 0.331 ± 0.023 | 0.477 ± 0.052 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.503 ± 0.154 | - | 0.468 ± 0.076 | 0.731 ± 0.045 | 0.571 ± 0.072 | 0.405 ± 0.017 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.278 ± 0.010 | - | 0.225 ± 0.014 | 0.355 ± 0.105 | 0.441 ± 0.100 | 0.431 ± 0.004 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.197 ± 0.576 | - | unsupported | 1.103 ± 0.940 | 0.780 ± 0.041 | 0.595 ± 2.479 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.551 ± 0.234 | - | unsupported | 0.576 ± 0.102 | 0.501 ± 0.043 | 0.595 ± 0.036 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.503 ± 0.303 | - | 1.111 ± 1.111 | 0.314 ± 0.020 | 0.387 ± 0.069 | 0.842 ± 0.106 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.120 ± 0.001 | - | 0.138 ± 0.019 | 0.255 ± 0.023 | 0.233 ± 0.045 | 0.829 ± 0.043 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.775 ± 0.247 | - | 0.708 ± 0.572 | 1.041 ± 0.071 | 0.606 ± 0.154 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.209 ± 0.003 | - | 0.255 ± 0.001 | 0.556 ± 0.072 | 0.432 ± 0.034 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.801 ± 0.362 | - | 0.582 ± 0.390 | 0.582 ± 0.055 | 0.535 ± 0.032 | 1.792 ± 2.479 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.262 ± 0.050 | - | 0.236 ± 0.035 | 0.309 ± 0.058 | 0.455 ± 0.168 | 1.240 ± 1.458 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.621 ± 0.034 | - | 0.849 ± 0.237 | 0.533 ± 0.046 | 0.456 ± 0.129 | 0.304 ± 0.006 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.483 ± 0.037 | - | 0.549 ± 0.036 | 0.416 ± 0.060 | 0.401 ± 0.059 | 0.315 ± 0.027 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.104 ± 0.174 | - | 0.003 ± 0.000 | 0.933 ± 0.027 | 0.845 ± 0.048 | 1.245 ± 0.266 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.414 ± 0.007 | - | 0.002 ± 0.000 | 0.448 ± 0.052 | 0.487 ± 0.016 | 1.158 ± 0.032 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.052 ± 0.005 | - | 0.053 ± 0.002 | 0.108 ± 0.008 | 0.109 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.088 ± 0.004 | - | 0.090 ± 0.011 | 0.168 ± 0.016 | 0.144 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.205 ± 0.001 | - | 0.208 ± 0.007 | 0.257 ± 0.023 | 0.256 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.018 ± 0.001 | - | 0.019 ± 0.002 | 0.028 ± 0.002 | 0.079 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.036 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.046 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.139 ± 0.002 | - | 0.135 ± 0.005 | 0.199 ± 0.017 | 0.225 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.001 | 0.050 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.057 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.060 ± 0.003 | - | 0.061 ± 0.002 | 0.167 ± 0.022 | 0.105 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.096 ± 0.001 | - | 0.104 ± 0.002 | 0.197 ± 0.024 | 0.147 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.190 ± 0.006 | - | 0.205 ± 0.000 | 0.278 ± 0.022 | 0.269 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.018 ± 0.001 | - | 0.020 ± 0.001 | 0.106 ± 0.013 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.089 ± 0.020 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.056 ± 0.004 | 0.056 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.074 ± 0.031 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.070 ± 0.017 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.096 ± 0.040 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.084 ± 0.017 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.084 ± 0.022 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.075 ± 0.016 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.001 | - | 0.004 ± 0.001 | 0.065 ± 0.025 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.071 ± 0.021 | 0.062 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.112 ± 0.003 | - | 0.126 ± 0.002 | 0.231 ± 0.026 | 0.269 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.085 ± 0.008 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.069 ± 0.031 | 0.067 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.003 ± 0.001 | - | 0.004 ± 0.001 | 0.080 ± 0.012 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.083 ± 0.013 | 0.046 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.259 ± 0.016 | - | 0.272 ± 0.030 | 0.417 ± 0.072 | 0.876 ± 0.196 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.664 ± 0.182 | - | 1.644 ± 0.145 | 1.939 ± 0.309 | 2.322 ± 0.153 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.169 ± 0.229 | - | 6.107 ± 1.025 | 6.627 ± 0.191 | 6.913 ± 0.319 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 5.200 ± 0.516 | - | 5.038 ± 2.796 | 3.779 ± 0.254 | 3.179 ± 0.085 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.002 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.107 ± 0.020 | - | 0.103 ± 0.012 | 0.088 ± 0.005 | 0.185 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.001 | - | 0.010 ± 0.003 | 0.009 ± 0.000 | 0.016 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.001 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.021 ± 0.001 | - | 0.021 ± 0.003 | 0.019 ± 0.001 | 0.069 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.001 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.004 | 0.021 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.025 ± 0.003 | - | 0.028 ± 0.003 | 0.024 ± 0.000 | 0.086 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 40.739 ± 2.233 | - | 41.682 ± 1.637 | 29.459 ± 1.050 | 24.898 ± 0.463 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.009 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.793 ± 0.054 | - | 0.840 ± 0.092 | 0.685 ± 0.150 | 0.993 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.014 ± 0.001 | - | 0.016 ± 0.002 | 0.016 ± 0.000 | 0.028 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.053 ± 0.011 | - | 0.054 ± 0.001 | 0.054 ± 0.009 | 0.084 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.252 ± 0.016 | - | 0.263 ± 0.026 | 0.443 ± 0.064 | 0.733 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.450 ± 0.073 | - | 1.529 ± 0.033 | 1.869 ± 0.150 | 0.730 ± 0.257 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 5.855 ± 0.317 | - | 5.888 ± 0.213 | 5.976 ± 0.352 | 3.328 ± 1.401 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 4.637 ± 0.151 | - | 4.709 ± 0.204 | 3.656 ± 0.195 | 2.893 ± 0.157 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.088 ± 0.015 | - | 0.108 ± 0.014 | 0.090 ± 0.001 | 0.167 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.007 ± 0.002 | - | 0.008 ± 0.000 | 0.012 ± 0.004 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.005 ± 0.002 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.008 ± 0.001 | - | 0.007 ± 0.002 | 0.009 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.017 ± 0.001 | - | 0.019 ± 0.000 | 0.024 ± 0.006 | 0.069 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.002 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.010 ± 0.003 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.024 ± 0.003 | - | 0.025 ± 0.001 | 0.025 ± 0.001 | 0.068 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 35.732 ± 3.083 | - | 34.259 ± 2.444 | 28.544 ± 1.233 | 11.217 ± 0.416 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.005 ± 0.001 | - | 0.005 ± 0.000 | 0.009 ± 0.002 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.731 ± 0.047 | - | 0.812 ± 0.068 | 0.705 ± 0.030 | 0.362 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.013 ± 0.001 | - | 0.015 ± 0.004 | 0.017 ± 0.000 | 0.023 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.048 ± 0.007 | - | 0.045 ± 0.002 | 0.049 ± 0.001 | 0.125 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.266 ± 0.046 | - | 0.222 ± 0.023 | 0.246 ± 0.040 | 0.541 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.229 ± 0.167 | - | 1.321 ± 0.059 | 1.325 ± 0.141 | 1.478 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.339 ± 1.253 | - | 6.521 ± 1.902 | 4.843 ± 0.245 | 4.770 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.823 ± 1.145 | - | 1.144 ± 0.526 | 1.062 ± 0.099 | 1.090 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.070 ± 0.004 | - | 0.073 ± 0.013 | 0.063 ± 0.003 | 0.124 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.002 | 0.008 ± 0.002 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.014 ± 0.000 | - | 0.020 ± 0.006 | 0.017 ± 0.003 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.018 ± 0.006 | - | 0.019 ± 0.000 | 0.022 ± 0.005 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 10.011 ± 2.267 | - | 8.189 ± 0.431 | 8.280 ± 0.134 | 8.294 ± 0.396 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.551 ± 0.049 | - | 0.581 ± 0.177 | 0.521 ± 0.012 | 0.601 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.035 ± 0.001 | - | 0.041 ± 0.009 | 0.037 ± 0.010 | 0.100 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.245 ± 0.040 | - | 0.232 ± 0.015 | 0.246 ± 0.011 | 0.499 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.305 ± 0.182 | - | 1.317 ± 0.171 | 1.323 ± 0.111 | 0.983 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 4.805 ± 0.150 | - | 4.820 ± 0.228 | 5.060 ± 0.280 | 2.361 ± 0.912 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.010 ± 0.034 | - | 1.000 ± 0.021 | 1.039 ± 0.030 | 1.094 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.069 ± 0.006 | - | 0.074 ± 0.007 | 0.072 ± 0.001 | 0.130 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.003 ± 0.002 | - | 0.004 ± 0.002 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.007 ± 0.002 | - | 0.009 ± 0.002 | 0.009 ± 0.003 | 0.016 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.003 ± 0.001 | - | 0.003 ± 0.000 | 0.005 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | 0.007 ± 0.002 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.019 ± 0.001 | - | 0.016 ± 0.005 | 0.018 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.004 ± 0.001 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.009 ± 0.001 | - | 0.008 ± 0.001 | 0.009 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.021 ± 0.006 | - | 0.026 ± 0.002 | 0.029 ± 0.002 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 8.109 ± 0.170 | - | 8.121 ± 0.413 | 8.349 ± 0.658 | 3.911 ± 1.691 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.002 | 0.006 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.521 ± 0.023 | - | 0.533 ± 0.010 | 0.573 ± 0.047 | 0.313 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.011 ± 0.003 | - | 0.012 ± 0.001 | 0.016 ± 0.000 | 0.020 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.036 ± 0.007 | - | 0.036 ± 0.001 | 0.039 ± 0.002 | 0.100 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.124 ± 0.005 | - | 0.125 ± 0.015 | 0.132 ± 0.004 | 0.505 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.851 ± 0.042 | - | 0.844 ± 0.047 | 0.841 ± 0.121 | 1.619 ± 0.099 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.679 ± 0.806 | - | 2.568 ± 0.250 | 2.819 ± 0.606 | 5.007 ± 0.199 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.824 ± 0.326 | - | 0.837 ± 0.315 | 0.571 ± 0.088 | 1.069 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.024 ± 0.005 | - | 0.025 ± 0.004 | 0.040 ± 0.007 | 0.114 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.001 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.001 | - | 0.005 ± 0.001 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.007 ± 0.003 | - | 0.011 ± 0.004 | 0.016 ± 0.006 | 0.027 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.001 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.009 ± 0.002 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.001 | 0.029 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.477 ± 2.203 | - | 5.373 ± 1.809 | 4.450 ± 0.248 | 7.974 ± 0.584 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.193 ± 0.020 | - | 0.213 ± 0.015 | 0.268 ± 0.068 | 0.641 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.023 ± 0.004 | - | 0.021 ± 0.004 | 0.027 ± 0.007 | 0.092 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.098 ± 0.000 | - | 0.099 ± 0.001 | 0.114 ± 0.020 | 0.504 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.678 ± 0.048 | - | 0.717 ± 0.076 | 0.743 ± 0.024 | 0.961 ± 0.124 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.411 ± 0.295 | - | 2.450 ± 0.131 | 2.441 ± 0.106 | 1.591 ± 1.178 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.501 ± 0.040 | - | 0.507 ± 0.049 | 0.546 ± 0.027 | 0.946 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.025 ± 0.003 | - | 0.025 ± 0.004 | 0.028 ± 0.009 | 0.110 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.015 ± 0.002 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.007 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.003 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.008 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.005 ± 0.001 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.025 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 4.166 ± 0.075 | - | 4.195 ± 0.116 | 4.247 ± 0.377 | 4.001 ± 0.435 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.001 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.190 ± 0.019 | - | 0.225 ± 0.016 | 0.197 ± 0.039 | 0.267 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.008 ± 0.001 | - | 0.008 ± 0.000 | 0.013 ± 0.007 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.024 ± 0.006 | - | 0.019 ± 0.005 | 0.030 ± 0.006 | 0.091 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.108 ± 0.007 | - | 0.106 ± 0.016 | 0.182 ± 0.007 | 0.225 ± 0.102 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.213 ± 0.013 | - | 0.230 ± 0.093 | 0.273 ± 0.026 | 0.231 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.507 ± 0.032 | - | 0.516 ± 0.027 | 0.579 ± 0.087 | 0.524 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.061 ± 0.004 | - | 0.137 ± 0.079 | 0.064 ± 0.021 | 0.115 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.005 ± 0.002 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.043 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.009 ± 0.003 | - | 0.012 ± 0.001 | 0.015 ± 0.003 | 0.045 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.009 ± 0.003 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.010 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.509 ± 0.058 | - | 0.487 ± 0.015 | 0.452 ± 0.042 | 0.583 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.038 ± 0.002 | - | 0.038 ± 0.002 | 0.040 ± 0.008 | 0.091 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.069 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.001 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.089 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.099 ± 0.001 | - | 0.085 ± 0.013 | 0.230 ± 0.035 | 0.163 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.183 ± 0.009 | - | 0.187 ± 0.036 | 0.298 ± 0.037 | 0.236 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.424 ± 0.027 | - | 0.430 ± 0.021 | 0.380 ± 0.016 | 0.365 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.055 ± 0.002 | - | 0.059 ± 0.001 | 0.117 ± 0.017 | 0.165 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.086 ± 0.038 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.007 ± 0.002 | - | 0.007 ± 0.000 | 0.062 ± 0.019 | 0.056 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.055 ± 0.008 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.009 ± 0.000 | - | 0.009 ± 0.000 | 0.072 ± 0.007 | 0.049 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.080 ± 0.040 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.101 ± 0.024 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.090 ± 0.013 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.082 ± 0.025 | 0.050 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.096 ± 0.018 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.077 ± 0.009 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.005 ± 0.000 | - | 0.005 ± 0.000 | 0.083 ± 0.039 | 0.054 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.413 ± 0.018 | - | 0.450 ± 0.019 | 0.359 ± 0.034 | 0.463 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.125 ± 0.022 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.031 ± 0.003 | - | 0.034 ± 0.010 | 0.099 ± 0.015 | 0.125 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.005 ± 0.000 | - | 0.005 ± 0.000 | 0.127 ± 0.052 | 0.009 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.007 ± 0.000 | - | 0.007 ± 0.000 | 0.113 ± 0.040 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.178 ± 0.014 | - | 0.187 ± 0.021 | 0.252 ± 0.018 | 0.303 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.513 ± 0.019 | - | 0.513 ± 0.015 | 0.563 ± 0.053 | 0.662 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.245 ± 0.038 | - | 1.316 ± 0.099 | 1.384 ± 0.071 | 1.355 ± 0.085 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.309 ± 0.036 | - | 0.296 ± 0.055 | 0.318 ± 0.038 | 0.363 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.015 ± 0.001 | - | 0.014 ± 0.004 | 0.014 ± 0.003 | 0.075 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.002 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.015 ± 0.004 | - | 0.014 ± 0.001 | 0.016 ± 0.003 | 0.085 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.001 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.002 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.004 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.001 | - | 0.005 ± 0.001 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.003 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.509 ± 0.131 | - | 2.496 ± 0.083 | 2.641 ± 0.103 | 2.671 ± 0.306 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.094 ± 0.015 | - | 0.089 ± 0.011 | 0.084 ± 0.013 | 0.130 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.002 | - | 0.007 ± 0.002 | 0.006 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.216 ± 0.013 | - | 0.229 ± 0.011 | 0.251 ± 0.035 | 0.314 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.513 ± 0.032 | - | 0.516 ± 0.043 | 0.613 ± 0.055 | 0.600 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.310 ± 0.132 | - | 1.313 ± 0.035 | 1.222 ± 0.072 | 0.706 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.294 ± 0.009 | - | 0.300 ± 0.011 | 0.359 ± 0.022 | 0.358 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.047 ± 0.020 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.015 ± 0.003 | - | 0.013 ± 0.000 | 0.046 ± 0.006 | 0.077 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.003 | - | 0.003 ± 0.000 | 0.050 ± 0.014 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.016 ± 0.003 | - | 0.018 ± 0.004 | 0.072 ± 0.006 | 0.099 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.002 | - | 0.004 ± 0.000 | 0.045 ± 0.011 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.001 | 0.048 ± 0.015 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.001 | - | 0.006 ± 0.001 | 0.049 ± 0.009 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.051 ± 0.018 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.039 ± 0.010 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.001 | - | 0.005 ± 0.002 | 0.052 ± 0.005 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.001 | 0.051 ± 0.013 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.366 ± 0.100 | - | 2.346 ± 0.047 | 2.389 ± 0.211 | 1.392 ± 0.127 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.050 ± 0.013 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.089 ± 0.015 | - | 0.101 ± 0.025 | 0.120 ± 0.014 | 0.194 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.011 ± 0.004 | 0.044 ± 0.011 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.004 | 0.055 ± 0.013 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.132 ± 0.001 | - | 0.120 ± 0.002 | 0.130 ± 0.016 | 0.185 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.192 ± 0.021 | - | 0.192 ± 0.035 | 0.179 ± 0.014 | 0.224 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.360 ± 0.024 | - | 0.385 ± 0.032 | 0.395 ± 0.048 | 0.464 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.051 ± 0.004 | - | 0.060 ± 0.006 | 0.063 ± 0.002 | 0.138 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.015 ± 0.014 | 0.010 ± 0.001 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.057 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.001 ± 0.001 | - | 0.006 ± 0.001 | 0.011 ± 0.003 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.002 | 0.009 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.003 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.003 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.476 ± 0.125 | - | 0.433 ± 0.033 | 0.370 ± 0.019 | 0.546 ± 0.140 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.024 ± 0.000 | - | 0.030 ± 0.000 | 0.034 ± 0.006 | 0.082 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.003 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.113 ± 0.018 | - | 0.108 ± 0.023 | 0.132 ± 0.026 | 0.155 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.177 ± 0.012 | - | 0.186 ± 0.005 | 0.168 ± 0.032 | 0.242 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.329 ± 0.041 | - | 0.326 ± 0.051 | 0.304 ± 0.017 | 0.423 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.053 ± 0.003 | - | 0.063 ± 0.002 | 0.062 ± 0.005 | 0.141 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.012 | 0.009 ± 0.004 | 0.009 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.006 ± 0.001 | - | 0.011 ± 0.001 | 0.012 ± 0.000 | 0.068 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.001 | 0.009 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.009 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.002 | 0.009 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.008 ± 0.001 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.007 ± 0.002 | 0.011 ± 0.003 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.001 | - | 0.009 ± 0.001 | 0.013 ± 0.004 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.418 ± 0.015 | - | 0.842 ± 0.024 | 0.351 ± 0.008 | 0.506 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.042 ± 0.026 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.032 ± 0.006 | - | 0.032 ± 0.002 | 0.063 ± 0.005 | 0.146 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.061 ± 0.003 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.005 ± 0.001 | - | 0.010 ± 0.000 | 0.063 ± 0.008 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.835 ± 0.851 | - | 2.471 ± 1.077 | 1.359 ± 0.321 | 3.295 ± 0.115 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 5.701 ± 1.159 | - | 6.032 ± 1.706 | 5.920 ± 0.377 | 6.627 ± 0.508 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 16.109 ± 1.293 | - | 16.474 ± 2.296 | 16.750 ± 1.383 | 16.198 ± 0.847 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 5.367 ± 1.245 | - | 4.300 ± 0.115 | 4.628 ± 0.205 | 3.671 ± 0.132 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.013 ± 0.005 | - | 0.024 ± 0.021 | 0.012 ± 0.005 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.207 ± 0.008 | - | 0.213 ± 0.022 | 0.240 ± 0.024 | 0.246 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.003 | - | 0.007 ± 0.001 | 0.025 ± 0.013 | 0.022 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.192 ± 0.014 | - | 0.199 ± 0.015 | 0.195 ± 0.003 | 0.324 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.016 ± 0.005 | - | 0.021 ± 0.003 | 0.032 ± 0.014 | 0.061 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.013 ± 0.004 | - | 0.021 ± 0.010 | 0.011 ± 0.002 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.017 ± 0.005 | - | 0.015 ± 0.001 | 0.029 ± 0.009 | 0.027 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.051 ± 0.004 | - | 0.051 ± 0.011 | 0.064 ± 0.013 | 0.059 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.014 ± 0.008 | - | 0.015 ± 0.011 | 0.012 ± 0.007 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.027 ± 0.086 | - | 0.019 ± 0.000 | 0.029 ± 0.005 | 0.035 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.069 ± 0.003 | - | 0.066 ± 0.019 | 0.077 ± 0.017 | 0.085 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 41.011 ± 5.691 | - | 37.822 ± 4.475 | 36.914 ± 3.989 | 28.407 ± 0.485 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.023 ± 0.010 | - | 0.026 ± 0.013 | 0.026 ± 0.007 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.568 ± 0.677 | - | 1.606 ± 0.092 | 1.669 ± 0.058 | 1.801 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.038 ± 0.013 | - | 0.043 ± 0.003 | 0.048 ± 0.009 | 0.065 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.126 ± 0.006 | - | 0.127 ± 0.007 | 0.148 ± 0.039 | 0.182 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.189 ± 0.051 | - | 1.235 ± 0.022 | 1.382 ± 0.108 | 2.954 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.218 ± 0.105 | - | 4.680 ± 0.199 | 4.649 ± 0.115 | 2.186 ± 0.385 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 13.283 ± 0.237 | - | 13.212 ± 0.245 | 14.643 ± 1.563 | 7.181 ± 3.325 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.244 ± 0.108 | - | 4.407 ± 0.174 | 3.803 ± 0.165 | 3.755 ± 0.337 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.197 ± 0.007 | - | 0.208 ± 0.013 | 0.185 ± 0.010 | 0.261 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.007 ± 0.002 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.163 ± 0.007 | - | 0.173 ± 0.006 | 0.180 ± 0.012 | 0.239 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.017 ± 0.003 | - | 0.021 ± 0.005 | 0.018 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.007 ± 0.001 | - | 0.010 ± 0.001 | 0.012 ± 0.002 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.001 | 0.020 ± 0.005 | 0.028 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.044 ± 0.004 | - | 0.045 ± 0.006 | 0.044 ± 0.007 | 0.053 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.009 ± 0.002 | - | 0.008 ± 0.000 | 0.013 ± 0.003 | 0.029 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.021 ± 0.005 | - | 0.026 ± 0.008 | 0.025 ± 0.006 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.071 ± 0.009 | - | 0.059 ± 0.012 | 0.065 ± 0.013 | 0.064 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 36.792 ± 1.678 | - | 33.369 ± 1.502 | 34.186 ± 3.104 | 15.727 ± 2.163 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.011 ± 0.003 | - | 0.012 ± 0.001 | 0.015 ± 0.007 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.455 ± 0.054 | - | 1.608 ± 0.070 | 1.401 ± 0.025 | 0.610 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.040 ± 0.010 | - | 0.037 ± 0.005 | 0.041 ± 0.007 | 0.053 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.119 ± 0.019 | - | 0.120 ± 0.015 | 0.105 ± 0.010 | 0.190 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.547 ± 0.050 | - | 0.647 ± 0.026 | 0.712 ± 0.043 | 1.963 ± 0.103 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.041 ± 0.080 | - | 3.086 ± 0.962 | 3.425 ± 0.379 | 4.389 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 11.991 ± 2.570 | - | 11.674 ± 2.484 | 11.517 ± 0.973 | 11.083 ± 0.544 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.618 ± 0.061 | - | 1.685 ± 0.090 | 1.665 ± 0.047 | 1.648 ± 0.193 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.113 ± 0.015 | - | 0.107 ± 0.005 | 0.143 ± 0.046 | 0.213 ± 0.100 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.029 ± 0.106 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.131 ± 0.015 | - | 0.227 ± 0.121 | 0.115 ± 0.011 | 0.186 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.013 ± 0.003 | - | 0.015 ± 0.002 | 0.015 ± 0.004 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.007 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.013 ± 0.031 | - | 0.013 ± 0.001 | 0.012 ± 0.001 | 0.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.032 ± 0.003 | - | 0.031 ± 0.007 | 0.038 ± 0.005 | 0.061 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.001 | 0.007 ± 0.003 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.016 ± 0.002 | - | 0.015 ± 0.004 | 0.015 ± 0.001 | 0.026 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.058 ± 0.034 | - | 0.058 ± 0.021 | 0.047 ± 0.008 | 0.060 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 13.533 ± 1.294 | - | 22.544 ± 6.251 | 14.148 ± 1.799 | 13.134 ± 0.223 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.008 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.031 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.895 ± 0.054 | - | 0.885 ± 0.081 | 0.890 ± 0.061 | 1.055 ± 0.431 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.027 ± 0.007 | - | 0.023 ± 0.006 | 0.040 ± 0.015 | 0.043 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.073 ± 0.007 | - | 0.079 ± 0.005 | 0.089 ± 0.018 | 0.127 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.564 ± 0.018 | - | 0.555 ± 0.030 | 0.598 ± 0.046 | 1.804 ± 0.183 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.593 ± 0.144 | - | 2.692 ± 0.128 | 2.883 ± 0.144 | 1.386 ± 0.207 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 8.830 ± 0.527 | - | 9.060 ± 0.098 | 9.995 ± 0.730 | 3.962 ± 2.514 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.399 ± 0.081 | - | 1.515 ± 0.069 | 1.680 ± 0.096 | 1.617 ± 0.137 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.004 ± 0.001 | 0.006 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.106 ± 0.003 | - | 0.105 ± 0.001 | 0.118 ± 0.020 | 0.199 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.006 ± 0.002 | - | 0.006 ± 0.000 | 0.007 ± 0.003 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.094 ± 0.005 | - | 0.102 ± 0.004 | 0.111 ± 0.003 | 0.142 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.011 ± 0.002 | - | 0.016 ± 0.001 | 0.014 ± 0.004 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.001 | 0.007 ± 0.000 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.010 ± 0.003 | - | 0.011 ± 0.001 | 0.012 ± 0.006 | 0.022 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.031 ± 0.005 | - | 0.032 ± 0.008 | 0.032 ± 0.008 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.005 ± 0.001 | - | 0.006 ± 0.000 | 0.013 ± 0.007 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.013 ± 0.002 | - | 0.013 ± 0.000 | 0.019 ± 0.007 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.035 ± 0.001 | - | 0.038 ± 0.002 | 0.047 ± 0.008 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 12.174 ± 0.234 | - | 11.385 ± 0.520 | 13.914 ± 0.307 | 5.066 ± 2.453 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.007 ± 0.002 | - | 0.010 ± 0.001 | 0.015 ± 0.005 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.773 ± 0.042 | - | 0.847 ± 0.030 | 0.886 ± 0.054 | 0.367 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.022 ± 0.000 | - | 0.024 ± 0.002 | 0.026 ± 0.003 | 0.038 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.069 ± 0.012 | - | 0.067 ± 0.013 | 0.085 ± 0.019 | 0.149 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.237 ± 0.012 | - | unsupported | 0.260 ± 0.024 | 1.500 ± 0.243 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.575 ± 0.062 | - | unsupported | 1.653 ± 0.105 | 2.516 ± 0.210 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.538 ± 0.323 | - | unsupported | 4.477 ± 0.416 | 4.849 ± 0.335 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.943 ± 0.034 | - | unsupported | 0.993 ± 0.036 | 0.909 ± 0.102 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.008 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.041 ± 0.009 | - | unsupported | 0.047 ± 0.004 | 0.041 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.036 ± 0.004 | - | unsupported | 0.042 ± 0.003 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | unsupported | 0.011 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.006 ± 0.002 | - | unsupported | 0.010 ± 0.002 | 0.018 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.015 ± 0.002 | - | unsupported | 0.015 ± 0.004 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.001 | - | unsupported | 0.005 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.002 | - | unsupported | 0.012 ± 0.000 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.015 ± 0.005 | - | unsupported | 0.018 ± 0.000 | 0.034 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 7.962 ± 0.775 | - | unsupported | 7.963 ± 0.186 | 7.435 ± 0.190 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.002 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.304 ± 0.044 | - | unsupported | 0.344 ± 0.023 | 0.289 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | unsupported | 0.019 ± 0.001 | 0.031 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.037 ± 0.002 | - | unsupported | 0.041 ± 0.009 | 0.043 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.216 ± 0.016 | - | unsupported | 0.248 ± 0.032 | 1.203 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.332 ± 0.046 | - | unsupported | 1.483 ± 0.133 | 0.845 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 4.255 ± 0.344 | - | unsupported | 4.288 ± 0.105 | 2.538 ± 0.973 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.973 ± 0.023 | - | unsupported | 0.909 ± 0.030 | 0.874 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.035 ± 0.002 | - | unsupported | 0.039 ± 0.001 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.032 ± 0.005 | - | unsupported | 0.032 ± 0.001 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.004 ± 0.001 | - | unsupported | 0.010 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.011 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.011 ± 0.003 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.015 ± 0.003 | - | unsupported | 0.019 ± 0.000 | 0.027 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 7.077 ± 0.055 | - | unsupported | 7.081 ± 0.112 | 3.750 ± 1.673 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.018 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.287 ± 0.037 | - | unsupported | 0.281 ± 0.017 | 0.168 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.011 ± 0.003 | - | unsupported | 0.014 ± 0.000 | 0.027 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.033 ± 0.002 | - | unsupported | 0.033 ± 0.001 | 0.068 ± 0.049 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.928 ± 0.094 | - | unsupported | 0.817 ± 0.063 | 1.075 ± 0.308 | 2.769 ± 0.733 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.068 ± 0.001 | - | unsupported | 0.074 ± 0.052 | 0.149 ± 0.018 | 0.087 ± 0.023 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.134 ± 0.021 | - | unsupported | 0.120 ± 0.016 | 0.179 ± 0.008 | 0.167 ± 0.016 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.281 ± 0.014 | - | unsupported | 0.254 ± 0.014 | 0.362 ± 0.071 | 0.638 ± 0.054 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.817 ± 0.061 | - | unsupported | 0.449 ± 0.056 | 0.459 ± 0.206 | 1.976 ± 0.464 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.058 ± 0.003 | - | unsupported | 0.095 ± 0.029 | 0.165 ± 0.031 | 0.066 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.129 ± 0.010 | - | unsupported | 0.118 ± 0.015 | 0.195 ± 0.014 | 0.154 ± 0.012 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.289 ± 0.029 | - | unsupported | 0.237 ± 0.007 | 0.262 ± 0.024 | 0.553 ± 0.158 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.208 ± 0.018 | - | unsupported | 0.203 ± 0.014 | 0.524 ± 0.049 | 0.341 ± 0.022 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.086 ± 0.018 | - | unsupported | 0.064 ± 0.018 | 0.102 ± 0.031 | 0.061 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.072 ± 0.007 | - | unsupported | 0.052 ± 0.012 | 0.124 ± 0.033 | 0.100 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.098 ± 0.007 | - | unsupported | 0.081 ± 0.013 | 0.185 ± 0.039 | 0.214 ± 0.032 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.222 ± 0.015 | - | unsupported | 0.172 ± 0.030 | 0.524 ± 0.020 | 0.321 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.068 ± 0.004 | - | unsupported | 0.048 ± 0.001 | 0.119 ± 0.012 | 0.053 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.089 ± 0.030 | - | unsupported | 0.046 ± 0.002 | 0.123 ± 0.015 | 0.087 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.096 ± 0.003 | - | unsupported | 0.074 ± 0.004 | 0.215 ± 0.042 | 0.149 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.130 ± 0.028 | - | 0.097 ± 0.012 | 0.070 ± 0.012 | 0.177 ± 0.011 | 0.199 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.042 ± 0.010 | - | 0.050 ± 0.025 | 0.027 ± 0.014 | 0.025 ± 0.005 | 0.028 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.051 ± 0.001 | - | 0.053 ± 0.005 | 0.024 ± 0.001 | 0.027 ± 0.002 | 0.055 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.069 ± 0.004 | - | 0.070 ± 0.011 | 0.038 ± 0.008 | 0.050 ± 0.001 | 0.099 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.103 ± 0.027 | - | 0.123 ± 0.012 | 0.073 ± 0.011 | 0.185 ± 0.011 | 0.140 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.076 ± 0.002 | - | 0.051 ± 0.010 | 0.026 ± 0.007 | 0.022 ± 0.001 | 0.024 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.043 ± 0.011 | - | 0.045 ± 0.003 | 0.029 ± 0.006 | 0.027 ± 0.001 | 0.038 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.071 ± 0.001 | - | 0.057 ± 0.002 | 0.036 ± 0.010 | 0.089 ± 0.025 | 0.069 ± 0.000 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 7.718 ± 2.732 | - | 9.089 ± 2.045 | 24.259 ± 0.987 | 14.813 ± 1.302 | 38.250 ± 6.831 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 5.797 ± 0.128 | - | 5.889 ± 0.325 | 11.376 ± 1.940 | 13.205 ± 3.571 | 13.412 ± 6.096 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 8.404 ± 1.650 | - | 7.699 ± 1.130 | 5.829 ± 0.540 | 5.742 ± 0.189 | 21.255 ± 1.927 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 5.468 ± 0.070 | - | 5.449 ± 0.317 | 7.611 ± 2.517 | 5.082 ± 0.669 | 9.836 ± 4.443 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.835 ± 0.193 | - | 7.140 ± 0.226 | 6.724 ± 0.297 | 7.337 ± 0.971 | 6.819 ± 0.340 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 5.515 ± 0.017 | - | 5.718 ± 0.522 | 5.877 ± 0.078 | 7.395 ± 0.149 | 9.918 ± 4.101 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 9.928 ± 3.212 | - | 10.457 ± 6.438 | 7.849 ± 0.197 | 7.372 ± 0.102 | 7.324 ± 0.587 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 6.380 ± 0.069 | - | 6.427 ± 0.116 | 7.108 ± 0.372 | 6.812 ± 0.227 | 12.654 ± 3.086 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 20.271 ± 4.615 | - | 20.737 ± 2.601 | 19.026 ± 0.869 | 23.574 ± 1.769 | 13.122 ± 1.166 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 15.237 ± 0.141 | - | 15.339 ± 0.032 | 18.792 ± 0.321 | 20.937 ± 0.938 | 11.803 ± 3.124 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.262 ± 0.018 | 0.275 ± 0.032 | unsupported | 0.255 ± 0.014 | 0.379 ± 0.057 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 3.507 ± 0.498 | 3.843 ± 0.733 | unsupported | 3.345 ± 0.216 | 4.632 ± 1.985 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.228 ± 0.003 | 0.257 ± 0.023 | unsupported | 0.245 ± 0.002 | 0.304 ± 0.006 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.152 ± 0.302 | 3.266 ± 0.117 | unsupported | 2.913 ± 0.157 | 3.238 ± 0.067 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 8.947 ± 1.351 | - | 12.599 ± 7.955 | 7.885 ± 0.796 | 8.486 ± 1.506 | 29.349 ± 1.202 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 6.412 ± 0.219 | - | 7.019 ± 0.554 | 6.841 ± 0.199 | 7.221 ± 0.303 | 9.070 ± 0.407 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 8.341 ± 0.481 | 8.799 ± 2.263 | 4.466 ± 0.301 | 27.817 ± 1.241 | 7.986 ± 0.579 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 7.451 ± 0.396 | 7.437 ± 1.023 | 4.444 ± 0.207 | 25.536 ± 1.221 | 5.089 ± 1.214 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 10.755 ± 3.367 | - | 7.742 ± 0.771 | 10.714 ± 0.783 | 10.275 ± 1.228 | 25.567 ± 10.286 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 5.423 ± 0.515 | - | 5.585 ± 0.159 | 8.330 ± 0.288 | 6.300 ± 0.196 | 13.506 ± 6.236 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.868 ± 0.189 | - | 1.062 ± 0.222 | 2.390 ± 0.083 | 5.686 ± 1.227 | 5.842 ± 0.801 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.346 ± 0.025 | - | 0.525 ± 0.271 | 2.066 ± 0.040 | 2.108 ± 0.172 | 5.352 ± 0.095 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 10.642 ± 3.385 | - | 10.098 ± 0.747 | 9.681 ± 0.965 | 11.596 ± 0.406 | 21.355 ± 1.749 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 9.378 ± 0.082 | - | 7.847 ± 0.063 | 8.489 ± 0.141 | 10.189 ± 0.805 | 15.025 ± 1.407 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 9.876 ± 0.423 | - | 11.229 ± 2.446 | 9.528 ± 2.717 | 11.824 ± 1.091 | 20.650 ± 3.017 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 8.876 ± 0.466 | - | 7.865 ± 0.297 | 8.422 ± 0.180 | 9.760 ± 0.204 | 14.611 ± 0.394 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.838 ± 0.284 | 1.459 ± 0.242 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 13.022 ± 2.119 | 12.922 ± 2.399 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.237 ± 0.034 | 1.235 ± 0.021 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 11.790 ± 2.108 | 12.163 ± 1.063 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.508 ± 0.062 | 1.604 ± 0.123 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 10.476 ± 0.795 | 11.276 ± 0.609 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.428 ± 0.134 | 1.426 ± 0.029 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 9.617 ± 0.659 | 10.805 ± 1.335 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 10.521 ± 4.195 | - | 8.008 ± 1.311 | 6.545 ± 2.596 | 7.488 ± 0.746 | 19.683 ± 0.943 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 5.447 ± 0.109 | - | 5.455 ± 0.237 | 5.707 ± 0.092 | 5.183 ± 0.656 | 7.902 ± 1.914 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 32.581 ± 2.470 | 29.973 ± 6.517 | 28.940 ± 1.410 | 27.194 ± 1.493 | 63.718 ± 1.796 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 26.599 ± 2.720 | 25.672 ± 0.683 | 30.944 ± 4.552 | 29.406 ± 1.311 | 47.825 ± 4.904 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 16.052 ± 2.822 | - | 16.952 ± 4.091 | 10.156 ± 0.450 | 59.094 ± 7.881 | 29.464 ± 1.490 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 11.388 ± 1.035 | - | 10.949 ± 0.114 | 10.037 ± 0.586 | 25.812 ± 0.802 | 12.943 ± 6.338 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.199 ± 0.103 | 0.195 ± 0.012 | unsupported | 0.057 ± 0.008 | 0.030 ± 0.003 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 3.019 ± 1.812 | 4.184 ± 1.390 | unsupported | 0.938 ± 0.216 | 0.479 ± 0.186 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.171 ± 0.015 | 0.166 ± 0.024 | unsupported | 0.057 ± 0.011 | 0.064 ± 0.001 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 1.192 ± 0.817 | 1.306 ± 0.603 | unsupported | 0.400 ± 0.067 | 0.235 ± 0.152 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.092 ± 0.027 | 0.085 ± 0.002 | unsupported | 0.094 ± 0.018 | 0.193 ± 0.014 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.369 ± 0.131 | 1.342 ± 0.051 | unsupported | 1.584 ± 0.075 | 2.192 ± 0.181 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.024 ± 0.001 | 0.068 ± 0.195 | unsupported | 0.104 ± 0.004 | 0.104 ± 0.007 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.374 ± 0.146 | 0.358 ± 0.023 | unsupported | 0.351 ± 0.004 | 0.727 ± 0.083 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.303 ± 0.103 | 0.295 ± 0.056 | unsupported | 0.327 ± 0.034 | 0.087 ± 0.006 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.790 ± 0.822 | 4.917 ± 0.899 | unsupported | 5.336 ± 0.669 | 0.874 ± 0.076 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.075 ± 0.065 | 0.075 ± 0.002 | unsupported | 0.193 ± 0.012 | 0.061 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.039 ± 0.049 | 1.063 ± 0.046 | unsupported | 1.131 ± 0.040 | 0.362 ± 0.014 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.045 ± 0.002 | 0.060 ± 0.003 | unsupported | 0.054 ± 0.002 | 0.070 ± 0.006 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.606 ± 0.063 | 0.719 ± 0.216 | unsupported | 0.548 ± 0.027 | 0.543 ± 0.049 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.056 ± 0.007 | 0.074 ± 0.007 | unsupported | 0.065 ± 0.014 | 0.077 ± 0.005 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.351 ± 0.094 | 0.570 ± 0.184 | unsupported | 0.306 ± 0.012 | 0.363 ± 0.031 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.093 ± 0.003 | 0.115 ± 0.010 | unsupported | 0.104 ± 0.004 | 0.100 ± 0.004 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.370 ± 0.678 | 1.612 ± 0.533 | unsupported | 1.518 ± 0.114 | 1.169 ± 0.159 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.091 ± 0.004 | 0.118 ± 0.011 | unsupported | 0.106 ± 0.014 | 0.127 ± 0.019 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.556 ± 0.050 | 0.407 ± 0.012 | unsupported | 0.683 ± 0.059 | 0.552 ± 0.025 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.122 ± 0.011 | 0.144 ± 0.007 | unsupported | unsupported | 0.129 ± 0.015 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 2.065 ± 0.288 | 2.210 ± 0.195 | unsupported | unsupported | 1.466 ± 0.120 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.129 ± 0.012 | 0.160 ± 0.006 | unsupported | unsupported | 0.123 ± 0.007 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 0.684 ± 0.034 | 1.117 ± 0.491 | unsupported | unsupported | 1.334 ± 0.403 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.012 ± 0.001 | 0.018 ± 0.002 | unsupported | 0.008 ± 0.001 | 0.049 ± 0.007 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.108 ± 0.008 | 0.127 ± 0.012 | unsupported | 0.071 ± 0.015 | 0.232 ± 0.055 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.013 ± 0.002 | 0.036 ± 0.019 | unsupported | 0.050 ± 0.016 | 0.031 ± 0.002 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.047 ± 0.001 | 0.057 ± 0.339 | unsupported | 0.060 ± 0.005 | 0.148 ± 0.033 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.024 ± 0.002 | 0.035 ± 0.002 | unsupported | 0.029 ± 0.004 | 0.064 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.344 ± 0.016 | 0.366 ± 0.018 | unsupported | 0.344 ± 0.036 | 0.616 ± 0.048 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.028 ± 0.006 | 0.037 ± 0.004 | unsupported | 0.149 ± 0.027 | 0.077 ± 0.005 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.210 ± 0.141 | 0.195 ± 0.023 | unsupported | 0.372 ± 0.030 | 0.344 ± 0.020 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.057 ± 0.001 | 0.059 ± 0.004 | unsupported | 0.082 ± 0.006 | 0.081 ± 0.006 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.968 ± 0.339 | 0.863 ± 0.028 | unsupported | 1.380 ± 0.072 | 0.750 ± 0.018 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.015 ± 0.001 | 0.016 ± 0.001 | unsupported | 0.070 ± 0.007 | 0.061 ± 0.003 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.224 ± 0.040 | 0.215 ± 0.113 | unsupported | 0.435 ± 0.112 | 0.362 ± 0.102 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.059 ± 0.013 | 0.057 ± 0.005 | unsupported | 0.086 ± 0.010 | 0.078 ± 0.004 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.938 ± 0.542 | 0.917 ± 0.382 | unsupported | 1.358 ± 0.153 | 0.819 ± 0.035 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.014 ± 0.001 | 0.018 ± 0.026 | unsupported | 0.075 ± 0.004 | 0.061 ± 0.004 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.216 ± 0.013 | 0.224 ± 0.007 | unsupported | 0.322 ± 0.027 | 0.376 ± 0.010 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.091 ± 0.006 | 0.101 ± 0.004 | unsupported | 0.102 ± 0.008 | 0.109 ± 0.013 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.415 ± 0.073 | 1.399 ± 0.123 | unsupported | 2.194 ± 0.304 | 1.176 ± 0.209 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.093 ± 0.002 | 0.105 ± 0.004 | unsupported | 0.103 ± 0.004 | 0.120 ± 0.024 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.556 ± 0.043 | 0.396 ± 0.187 | unsupported | 0.802 ± 0.052 | 0.588 ± 0.034 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.257 ± 0.038 | 0.252 ± 0.137 | unsupported | 0.273 ± 0.017 | 0.289 ± 0.013 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 4.043 ± 0.592 | 3.990 ± 0.534 | unsupported | 4.847 ± 0.659 | 4.039 ± 0.255 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.057 ± 0.008 | 0.118 ± 0.059 | unsupported | 0.173 ± 0.012 | 0.163 ± 0.007 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.940 ± 0.825 | 2.041 ± 0.875 | unsupported | 1.030 ± 0.008 | 1.487 ± 0.141 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.542 ± 0.048 | 0.776 ± 0.069 | unsupported | 0.125 ± 0.016 | 0.097 ± 0.006 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 10.375 ± 1.060 | 12.885 ± 1.441 | unsupported | 2.044 ± 0.124 | 0.882 ± 0.060 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.522 ± 0.008 | 0.637 ± 0.026 | unsupported | 0.148 ± 0.028 | 0.063 ± 0.003 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 3.089 ± 0.146 | 3.921 ± 0.082 | unsupported | 0.525 ± 0.016 | 0.337 ± 0.031 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 18.226 ± 3.575 | - | unsupported | 15.291 ± 0.775 | - | 10.699 ± 1.005 | 16.092 ± 1.305 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 6.358 ± 0.507 | - | unsupported | 6.333 ± 0.453 | - | 12.146 ± 2.486 | 5.586 ± 0.174 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 13.537 ± 4.008 | - | unsupported | 9.661 ± 1.033 | - | 8.155 ± 0.711 | 8.643 ± 0.701 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 4.218 ± 0.515 | - | unsupported | 6.292 ± 0.487 | - | 8.174 ± 0.901 | 3.785 ± 0.220 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 8.107 ± 1.704 | - | unsupported | 10.384 ± 0.348 | - | 5.501 ± 0.160 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 3.864 ± 0.733 | - | unsupported | 6.228 ± 0.270 | - | 5.229 ± 0.163 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 17.959 ± 1.289 | - | unsupported | 15.353 ± 1.490 | - | 11.963 ± 1.057 | 13.846 ± 2.092 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 7.519 ± 1.869 | - | unsupported | 6.270 ± 0.293 | - | 11.238 ± 1.141 | 6.804 ± 0.584 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 8.089 ± 1.354 | - | unsupported | 7.454 ± 0.621 | - | 47.652 ± 4.937 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 6.981 ± 0.332 | - | unsupported | 7.680 ± 1.317 | - | 11.206 ± 0.068 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 14.836 ± 9.562 | - | unsupported | 8.208 ± 0.687 | - | 46.130 ± 3.486 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 6.763 ± 0.146 | - | unsupported | 8.395 ± 1.457 | - | 11.160 ± 1.559 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 17.610 ± 5.915 | - | unsupported | 15.045 ± 3.522 | - | 13.143 ± 2.253 | 16.347 ± 2.378 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 6.370 ± 0.800 | - | unsupported | 7.716 ± 1.276 | - | 12.511 ± 0.793 | 6.543 ± 0.803 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 10.753 ± 1.751 | - | unsupported | 9.620 ± 1.410 | - | 7.773 ± 0.389 | 12.600 ± 4.145 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 3.853 ± 0.810 | - | unsupported | 4.438 ± 0.223 | - | 7.462 ± 0.483 | 3.868 ± 0.359 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.230 ± 0.089 | unsupported | unsupported | 0.239 ± 0.031 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 2.700 ± 0.598 | unsupported | unsupported | 3.557 ± 1.742 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.171 ± 0.016 | unsupported | unsupported | 0.301 ± 0.020 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 2.156 ± 0.067 | unsupported | unsupported | 2.578 ± 0.327 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 15.197 ± 2.738 | - | unsupported | 14.380 ± 2.039 | - | 11.913 ± 1.071 | 14.587 ± 2.150 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 6.725 ± 0.655 | - | unsupported | 7.644 ± 1.180 | - | 11.747 ± 2.485 | 6.772 ± 0.670 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 6.753 ± 0.921 | - | 0.002 ± 0.000 | 5.749 ± 0.487 | 6.006 ± 1.318 | 14.498 ± 1.547 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 3.600 ± 0.180 | - | 0.001 ± 0.000 | 4.073 ± 0.353 | 4.491 ± 0.158 | 13.387 ± 2.891 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 8.277 ± 0.636 | - | 21.173 ± 6.340 | 8.843 ± 41.046 | 10.563 ± 0.812 | 8.142 ± 0.629 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 3.503 ± 0.724 | - | 8.063 ± 1.052 | 4.591 ± 0.350 | 4.737 ± 0.817 | 7.120 ± 1.899 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 12.490 ± 4.397 | - | 10.909 ± 0.713 | 12.024 ± 1.257 | 19.536 ± 1.199 | 12.550 ± 0.639 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 7.551 ± 3.395 | - | 6.870 ± 1.190 | 7.255 ± 1.577 | 8.986 ± 0.095 | 11.410 ± 2.031 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 4.976 ± 3.508 | - | 3.767 ± 0.130 | 9.970 ± 1.877 | 7.808 ± 0.638 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 2.435 ± 0.536 | - | 1.776 ± 0.268 | 3.885 ± 0.669 | 4.913 ± 0.965 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 10.887 ± 5.453 | - | 0.001 ± 0.001 | 10.072 ± 0.847 | 7.497 ± 1.167 | 7.188 ± 0.281 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 3.378 ± 0.248 | - | 0.001 ± 0.000 | 5.370 ± 0.176 | 3.580 ± 0.300 | 6.296 ± 0.746 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 44.461 ± 7.527 | - | 0.002 ± 0.007 | 51.361 ± 3.766 | 50.682 ± 0.812 | 90.296 ± 3.942 | 36.034 ± 19.576 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 11.798 ± 0.247 | - | 0.001 ± 0.000 | 52.505 ± 1.945 | 15.236 ± 0.324 | 92.853 ± 18.316 | 11.380 ± 3.318 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 7.303 ± 2.108 | - | 6.653 ± 1.944 | 7.161 ± 0.776 | 5.492 ± 0.495 | 7.588 ± 1.764 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 1.966 ± 0.124 | - | 2.537 ± 0.221 | 2.650 ± 0.099 | 2.073 ± 0.293 | 5.713 ± 0.604 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 7.984 ± 1.649 | - | 7.294 ± 1.307 | 7.324 ± 0.960 | 5.806 ± 0.638 | 6.901 ± 4.494 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 2.134 ± 0.176 | - | 2.110 ± 0.986 | 2.635 ± 0.322 | 2.063 ± 0.101 | 6.088 ± 2.267 | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.825708 | 301606.75 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 7.603750 | 475234.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 5.650375 | 353148.44 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 3.344292 | 209018.25 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 3.705917 | 231619.81 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.181500 | 261343.75 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 14.941834 | 933864.62 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 18.696500 | 1168531.25 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 3.700958 | 231309.88 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 3.312250 | 207015.62 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.062750 | 378921.88 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 4.083625 | 255226.56 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 4.193042 | 262065.12 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 4.398958 | 274934.88 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 74.115209 | 4632200.56 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 53.527250 | 3345453.12 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 56.118584 | 3507411.50 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 61.480042 | 3842502.62 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 4.868000 | 304250.00 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 3.912750 | 244546.88 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 3.649042 | 228065.12 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 4.110833 | 256927.06 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 51.803209 | 3237700.56 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 46.602709 | 2912669.31 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 50.426792 | 3151674.50 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 52.260416 | 3266276.00 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 29.411375 | 1838210.94 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 23.339458 | 1458716.12 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 208.348792 | 13021799.50 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 206.750083 | 12921880.19 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 19.784667 | 1236541.69 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 19.764000 | 1235250.00 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 188.645666 | 11790354.12 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 194.611958 | 12163247.38 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 24.125417 | 1507838.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 25.658750 | 1603671.88 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 167.618875 | 10476179.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 180.412709 | 11275794.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 22.845167 | 1427822.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 22.817583 | 1426098.94 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 153.879625 | 9617476.56 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 172.887166 | 10805447.88 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 1.945708 | 30401.69 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.652708 | 57073.56 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.709042 | 198578.78 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.480583 | 195009.11 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.829500 | 478687.50 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.506208 | 938276.00 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 24.155792 | 3019474.00 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 33.470625 | 4183828.12 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.098208 | 64034.50 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.672583 | 57384.11 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.973625 | 171462.89 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.652000 | 166437.50 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.881083 | 235135.38 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.202000 | 400250.00 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.532417 | 1191552.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.445417 | 1305677.12 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.363958 | 193186.84 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.993667 | 93651.05 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.892208 | 92065.75 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.459292 | 85301.44 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.537042 | 2192130.25 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.671875 | 1583984.38 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.950041 | 1368755.12 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.736500 | 1342062.50 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.658458 | 104038.41 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.674208 | 104284.50 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.507208 | 23550.12 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.353042 | 68016.28 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.814875 | 726859.38 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.805208 | 350651.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.988459 | 373557.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.861500 | 357687.50 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.556709 | 86823.58 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 20.923959 | 326936.86 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 19.370833 | 302669.27 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 18.860167 | 294690.11 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.989333 | 873666.62 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 42.691042 | 5336380.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 38.320583 | 4790072.88 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 39.332333 | 4916541.62 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.922916 | 61295.56 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.323958 | 192561.84 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.808542 | 75133.47 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.801500 | 75023.44 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.897917 | 362239.62 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.049542 | 1131192.75 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.315458 | 1039432.25 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.503792 | 1062974.00 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.506125 | 70408.20 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.470084 | 54220.06 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.853875 | 44591.80 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.851708 | 60182.94 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.347041 | 543380.12 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.380834 | 547604.25 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.845292 | 605661.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.755417 | 719427.12 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.926917 | 76983.08 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.161000 | 65015.62 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.565667 | 55713.55 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.704333 | 73505.20 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.902875 | 362859.38 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.445708 | 305713.50 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.807541 | 350942.62 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.562375 | 570296.88 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.422875 | 100357.42 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.634917 | 103670.58 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.935625 | 92744.14 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.335292 | 114613.94 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.353416 | 1169177.00 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.143000 | 1517875.00 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.956917 | 1369614.62 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 12.895708 | 1611963.50 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.121375 | 126896.48 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.786792 | 106043.62 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.855125 | 91486.33 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.545208 | 117893.88 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.418291 | 552286.38 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.463375 | 682921.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.449708 | 556213.50 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.257083 | 407135.38 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.254958 | 128983.72 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.797167 | 121830.73 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.221916 | 144092.44 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.731208 | 1466401.00 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 16.521791 | 2065223.88 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.678917 | 2209864.62 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.872000 | 123000.00 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.247541 | 128867.83 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.262875 | 160357.42 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.675500 | 1334437.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.469708 | 683713.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.935167 | 1116895.88 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.160792 | 49387.38 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.538000 | 8406.25 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.766458 | 11975.91 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.139334 | 17802.09 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.857167 | 232145.88 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.570417 | 71302.12 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.862250 | 107781.25 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.013334 | 126666.75 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 1.977042 | 30891.28 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.197375 | 49958.98 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.831500 | 12992.19 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.325833 | 36341.14 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.186292 | 148286.50 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.477375 | 59671.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.378500 | 47312.50 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.455375 | 56921.88 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.087834 | 63872.41 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.833333 | 28645.83 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.539292 | 24051.44 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.214667 | 34604.17 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.931666 | 616458.25 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 2.748375 | 343546.88 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.752917 | 344114.62 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.926166 | 365770.75 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.912667 | 76760.42 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.566875 | 149482.42 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.775166 | 27736.97 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.398584 | 37477.88 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.750542 | 343817.75 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.973333 | 371666.62 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.679916 | 209989.50 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.561459 | 195182.38 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.207458 | 81366.53 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.230375 | 81724.61 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.678291 | 57473.30 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.765875 | 58841.80 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.998125 | 749765.62 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.041333 | 1380166.62 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.744750 | 968093.75 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.902458 | 862807.25 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.915709 | 61182.95 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.495583 | 70243.48 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.940542 | 14695.97 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.048459 | 16382.17 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.893916 | 361739.50 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.477541 | 434692.62 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.788166 | 223520.75 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.718500 | 214812.50 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.001125 | 78142.58 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.495500 | 85867.19 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.807917 | 59498.70 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.666417 | 57287.77 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.552500 | 819062.50 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.865166 | 1358145.75 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.500875 | 937609.38 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.339083 | 917385.38 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.908625 | 61072.27 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.773417 | 74584.64 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.883083 | 13798.17 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.121625 | 17525.39 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.006667 | 375833.38 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.578333 | 322291.62 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.729875 | 216234.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.794834 | 224354.25 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.945584 | 108524.75 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.547000 | 102296.88 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.798459 | 90600.92 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.484458 | 101319.66 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.408166 | 1176020.75 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 17.555959 | 2194494.88 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.320833 | 1415104.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.191000 | 1398875.00 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.704750 | 120386.72 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.570542 | 102664.72 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.947167 | 92924.48 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.714334 | 104911.47 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.705542 | 588192.75 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.415125 | 801890.62 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.446250 | 555781.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.169833 | 396229.12 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 18.512333 | 289255.20 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.490834 | 273294.28 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 16.447209 | 256987.64 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 16.124750 | 251949.22 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 32.314041 | 4039255.13 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 38.773792 | 4846724.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 32.347625 | 4043453.12 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 31.916458 | 3989557.25 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.448250 | 163253.91 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.058625 | 172791.02 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.670375 | 57349.61 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.544834 | 117888.03 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.897334 | 1487166.75 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.243459 | 1030432.38 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.520042 | 940005.25 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.327625 | 2040953.13 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.207375 | 96990.23 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.997958 | 124968.09 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 34.691875 | 542060.55 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 49.683458 | 776304.03 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.055791 | 881973.88 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.353042 | 2044130.25 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 82.998833 | 10374854.12 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 103.077791 | 12884723.88 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.062791 | 63481.11 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.482959 | 148171.23 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 33.414500 | 522101.56 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 40.770250 | 637035.16 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.695542 | 336942.75 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.203667 | 525458.38 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 24.714375 | 3089296.88 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 31.366375 | 3920796.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 3.831458 | 239466.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 3.675208 | 229700.50 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 56.910667 | 3556916.69 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 43.194042 | 2699627.62 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 4.820541 | 301283.81 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 2.734209 | 170888.06 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 41.251375 | 2578210.94 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 34.499958 | 2156247.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.015666 | 979.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001708 | 106.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.015792 | 987.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001167 | 72.94 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000166 | 10.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.013417 | 838.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.004417 | 276.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.014583 | 911.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003334 | 208.38 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.012625 | 789.06 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.003584 | 224.00 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.012625 | 789.06 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.001708 | 106.75 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.013584 | 849.00 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003083 | 192.69 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.012875 | 804.69 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002625 | 164.06 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`julia-base`, 9.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 28.9x faster than the slowest successful cell (`pytorch-cpu`, 18.954 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 18.954 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 18.954 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 8.570 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 26.5x faster than the slowest successful cell (`pytorch-cpu`, 8.570 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 26.4x faster than the slowest successful cell (`pytorch-cpu`, 8.570 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 30.6x faster than the slowest successful cell (`pytorch-cpu`, 20.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 20.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 8.521 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 8.521 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 20.5x faster than the slowest successful cell (`pytorch-cpu`, 8.521 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 8.521 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=4, shape=`8388608`): `jax-cpu` is 12.0x faster than the slowest successful cell (`julia-base`, 53.895 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 403.8x faster than the slowest successful cell (`julia-base`, 1.245 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 678.1x faster than the slowest successful cell (`julia-base`, 1.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 18.6x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 30.9x faster than the slowest successful cell (`jax-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 25.6x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `jax-cpu` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 53.7x faster than the slowest successful cell (`pytorch-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 25.9x faster than the slowest successful cell (`pytorch-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 23.4x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 49.5x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 22.8x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 37.3x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 20.7x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 45.9x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 29.1x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 49.0x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 24.8x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `jax-cpu` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 37.2x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 22.3x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 31.8x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 22.4x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 31.2x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 28.9x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 39.1x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 23.1x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 29.7x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 22.6x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 17.2x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 19.1x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 29.5x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 23.8x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 22.6x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 27.8x faster than the slowest successful cell (`pytorch-cpu`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 21.8x faster than the slowest successful cell (`pytorch-cpu`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 24.2x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 24.2x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 26.9x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 20.6x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 20.1x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 19.7x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 32.3x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 24.4x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `jax-cpu` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 26.4x faster than the slowest successful cell (`pytorch-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 27.1x faster than the slowest successful cell (`pytorch-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 17.2x faster than the slowest successful cell (`pytorch-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 25.2x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 20.8x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x2x2,rhs=1`): `tenferro-direct` is 12.1x faster than the slowest successful cell (`tenferro-trace`, 0.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x32x32,rhs=1`): `tenferro-direct` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`4x2x2,rhs=1`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.017 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 21.4x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x8x8,rhs=1`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 15.5x faster than the slowest successful cell (`julia-base`, 5.352 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`julia-base`, 5.352 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 14.6x faster than the slowest successful cell (`tenferro-eager`, 12.885 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 10.0x faster than the slowest successful cell (`tenferro-eager`, 0.637 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 11.6x faster than the slowest successful cell (`tenferro-eager`, 3.921 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 9665.6x faster than the slowest successful cell (`julia-base`, 14.498 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 10709.6x faster than the slowest successful cell (`julia-base`, 13.387 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 7683.3x faster than the slowest successful cell (`tenferro-direct`, 10.887 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 4876.8x faster than the slowest successful cell (`julia-base`, 6.296 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 50388.3x faster than the slowest successful cell (`julia-base`, 90.296 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 76865.3x faster than the slowest successful cell (`julia-base`, 92.853 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 83.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 182.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 98.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 157.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 169.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 161.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 5439.2 GB/s at 0.003 ms. The conservative physical lower bound is 0.168 ms, so the cell is 54.4x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9817.9 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 24.5x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 178957.0 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1789.6x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 214748.4 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 536.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 189439.3 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1894.4x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 207928.3 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 519.8x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 74898.3 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 749.0x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 111107.4 GB/s at 0.001 ms. The conservative physical lower bound is 0.336 ms, so the cell is 277.8x faster than the bound.

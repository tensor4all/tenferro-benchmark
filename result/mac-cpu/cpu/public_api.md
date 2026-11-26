# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261126_182428/run.yaml`
- Timestamp: `2026-11-26T18:24:28.187247+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261126_182428`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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

- CSV: `data/results/mac-cpu/cpu/public_api/20261126_182428/cpu_public_api_t1_20261126_182428.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261126_182428/cpu_public_api_t4_20261126_182428.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 5.724 ± 0.928 | - | 5.691 ± 0.720 | 5.316 ± 2.786 | 2.043 ± 0.125 | 4.772 ± 2.617 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 5.549 ± 0.669 | - | 4.576 ± 0.898 | 3.799 ± 0.161 | 2.008 ± 1.093 | 72.519 ± 15.793 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 11.575 ± 0.342 | - | 11.865 ± 0.498 | 12.783 ± 2.159 | 15.140 ± 4.277 | 20.299 ± 8.004 | 17.816 ± 17.405 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 12.271 ± 1.625 | - | 13.594 ± 0.842 | 14.328 ± 2.252 | 10.730 ± 0.318 | 15.989 ± 4.504 | 18.546 ± 3.918 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 52.221 ± 7.396 | - | 64.865 ± 9.546 | 22.169 ± 2.493 | 41.002 ± 1.935 | 56.412 ± 13.404 | 58.083 ± 44.652 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 22.888 ± 2.818 | - | 24.712 ± 2.605 | 11.057 ± 1.874 | 21.741 ± 15.433 | 56.558 ± 4.430 | 30.172 ± 9.511 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 79.854 ± 6.626 | - | 73.198 ± 8.326 | 13.714 ± 4.199 | 56.808 ± 4.955 | 63.742 ± 3.978 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 33.591 ± 6.703 | - | 31.962 ± 3.523 | 10.323 ± 1.062 | 17.182 ± 1.143 | 44.009 ± 18.895 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 82.243 ± 13.384 | - | unsupported | 13.637 ± 7.139 | 63.403 ± 12.742 | 68.046 ± 12.296 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 35.407 ± 7.143 | - | unsupported | 10.870 ± 1.524 | 18.191 ± 2.925 | 56.731 ± 12.157 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 8.896 ± 0.959 | - | 8.713 ± 1.104 | 13.866 ± 3.056 | 28.097 ± 6.259 | 9.362 ± 1.348 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 11.966 ± 2.432 | - | 12.133 ± 3.972 | 10.993 ± 0.245 | 20.839 ± 0.924 | 472.604 ± 33.549 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 56.600 ± 8.104 | - | 49.188 ± 7.707 | 45.601 ± 3.596 | 52.933 ± 5.102 | 65.490 ± 44.981 | 56.389 ± 9.869 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 18.943 ± 2.534 | - | 21.650 ± 4.340 | 15.161 ± 2.803 | 21.164 ± 0.216 | 55.199 ± 4.803 | 27.447 ± 6.786 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 51.286 ± 3.158 | - | 52.293 ± 5.671 | 70.573 ± 13.758 | 106.583 ± 9.720 | 81.850 ± 4.728 | 79.469 ± 11.095 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 24.945 ± 4.757 | - | 24.118 ± 4.609 | 23.328 ± 4.283 | 31.343 ± 2.687 | 76.750 ± 8.901 | 47.784 ± 16.819 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 9.237 ± 0.080 | - | 9.280 ± 0.373 | 10.929 ± 3.252 | 11.558 ± 1.093 | 10.645 ± 1.339 | 10.700 ± 1.426 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 10.985 ± 1.017 | - | 10.859 ± 1.304 | 7.733 ± 0.348 | 8.982 ± 0.799 | 10.091 ± 1.931 | 11.370 ± 3.554 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.566 ± 0.236 | - | 8.792 ± 2.071 | 7.103 ± 1.013 | 8.336 ± 2.355 | 8.485 ± 0.351 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.617 ± 0.598 | - | 5.228 ± 1.245 | 3.130 ± 0.081 | 4.571 ± 0.462 | 9.015 ± 6.454 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 14.543 ± 4.499 | - | 13.070 ± 5.524 | 9.599 ± 2.640 | 8.512 ± 1.106 | 10.332 ± 1.115 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 7.766 ± 4.007 | - | 8.038 ± 3.484 | 6.948 ± 0.246 | 7.072 ± 0.551 | 177.226 ± 28.007 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 8.252 ± 2.622 | - | 7.408 ± 0.708 | 5.319 ± 3.122 | 3.315 ± 0.698 | 6.962 ± 0.288 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 5.525 ± 0.267 | - | 7.054 ± 0.818 | 5.537 ± 1.428 | 2.336 ± 0.200 | 134.637 ± 28.509 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 13.893 ± 8.949 | - | 11.093 ± 1.864 | 12.750 ± 5.524 | 9.762 ± 2.754 | 7.690 ± 1.260 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 8.100 ± 0.435 | - | 9.601 ± 4.570 | 6.625 ± 2.219 | 5.862 ± 0.359 | 608.337 ± 23.117 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 103.178 ± 9.102 | - | 80.997 ± 7.304 | 18.512 ± 4.549 | 72.754 ± 4.675 | 59.155 ± 5.828 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 33.849 ± 6.195 | - | 42.774 ± 11.417 | 10.192 ± 1.541 | 17.908 ± 3.526 | 39.985 ± 6.912 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 72.546 ± 15.837 | - | unsupported | 13.525 ± 2.704 | 64.230 ± 8.873 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 25.737 ± 4.107 | - | unsupported | 10.799 ± 2.933 | 19.541 ± 1.197 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 19.211 ± 7.588 | - | 12.944 ± 1.288 | 11.632 ± 0.552 | 17.333 ± 2.083 | 12.717 ± 0.747 | 15.807 ± 8.351 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 12.980 ± 1.329 | - | 12.000 ± 2.459 | 13.019 ± 0.964 | 10.507 ± 0.333 | 13.248 ± 3.913 | 14.436 ± 1.644 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 19.095 ± 1.557 | - | 20.713 ± 9.287 | 18.865 ± 6.107 | 27.073 ± 2.382 | 24.312 ± 17.479 | 22.123 ± 4.641 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 16.822 ± 1.063 | - | 25.158 ± 3.851 | 18.363 ± 3.568 | 23.761 ± 1.745 | 24.858 ± 5.452 | 20.915 ± 3.730 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 33.341 ± 10.930 | - | 58.291 ± 6.440 | 69.694 ± 21.907 | 44.353 ± 6.678 | 56.333 ± 6.257 | 57.779 ± 3.352 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 14.213 ± 3.661 | - | 31.140 ± 12.403 | 25.303 ± 5.176 | 16.180 ± 0.347 | 45.953 ± 3.052 | 24.539 ± 3.135 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 6.023 ± 0.191 | - | 6.499 ± 0.353 | 6.882 ± 1.049 | 8.420 ± 0.633 | 7.847 ± 3.071 | 7.364 ± 1.430 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.978 ± 2.141 | - | 10.073 ± 3.076 | 7.195 ± 1.060 | 7.177 ± 0.334 | 7.472 ± 2.713 | 11.384 ± 40.365 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 14.402 ± 1.413 | - | 11.800 ± 1.681 | 19.377 ± 5.933 | 21.933 ± 5.965 | 19.949 ± 11.393 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 14.335 ± 0.580 | - | 13.912 ± 2.426 | 13.327 ± 1.472 | 12.818 ± 2.429 | 14.731 ± 3.868 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 34.578 ± 0.804 | - | 31.537 ± 2.428 | 44.560 ± 6.770 | 30.863 ± 5.240 | 48.423 ± 8.386 | 47.779 ± 19.721 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 13.433 ± 2.442 | - | 15.768 ± 2.328 | 19.208 ± 3.651 | 15.564 ± 0.387 | 38.608 ± 4.258 | 18.450 ± 5.220 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 24.980 ± 3.757 | - | 24.203 ± 1.171 | 20.306 ± 3.515 | 31.974 ± 1.205 | 21.677 ± 8.293 | 32.035 ± 72.452 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 19.090 ± 2.199 | - | 19.305 ± 0.972 | 21.587 ± 2.539 | 22.024 ± 2.372 | 25.226 ± 43.270 | 20.606 ± 4.266 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 18.514 ± 0.723 | - | 18.108 ± 1.898 | 32.426 ± 7.386 | 24.818 ± 5.607 | 56.804 ± 63.005 | 48.106 ± 8.968 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 13.630 ± 3.730 | - | 14.693 ± 8.673 | 12.399 ± 4.057 | 9.274 ± 0.157 | 46.155 ± 11.279 | 26.532 ± 23.595 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 37.095 ± 7.185 | - | 31.640 ± 1.360 | 32.706 ± 4.676 | 21.524 ± 0.746 | 30.362 ± 3.626 | 30.484 ± 4.909 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 14.484 ± 2.499 | - | 12.246 ± 2.026 | 14.427 ± 0.809 | 9.163 ± 0.116 | 24.938 ± 2.583 | 9.833 ± 1.293 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 33.709 ± 2.522 | - | 33.502 ± 1.740 | 58.336 ± 4.018 | 28.774 ± 0.965 | 51.063 ± 14.286 | 73.863 ± 18.431 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 18.787 ± 3.966 | - | 20.880 ± 6.089 | 24.571 ± 15.480 | 15.416 ± 1.304 | 43.467 ± 3.697 | 26.915 ± 14.642 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 17.546 ± 2.294 | - | 17.091 ± 0.921 | 24.339 ± 3.968 | 27.876 ± 8.456 | 27.964 ± 4.260 | 26.539 ± 3.367 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 7.774 ± 0.844 | - | 7.962 ± 3.007 | 10.988 ± 4.014 | 9.849 ± 1.638 | 21.933 ± 2.751 | 8.175 ± 1.803 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.589 ± 0.365 | 0.536 ± 0.211 | unsupported | unsupported | 0.250 ± 0.031 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.267 ± 0.036 | 0.278 ± 0.019 | unsupported | unsupported | 0.191 ± 0.046 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 20.953 ± 3.026 | - | 21.478 ± 5.858 | 22.534 ± 10.343 | 30.326 ± 3.283 | 19.681 ± 0.998 | 153.413 ± 243.145 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 19.335 ± 2.127 | - | 19.198 ± 2.327 | 18.520 ± 1.795 | 22.055 ± 1.243 | 19.815 ± 3.024 | 20.236 ± 2.661 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 19.991 ± 2.957 | - | 20.831 ± 3.303 | 23.202 ± 5.461 | 32.863 ± 6.482 | 25.622 ± 16.784 | 38.052 ± 19.972 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 19.442 ± 0.926 | - | 18.312 ± 4.574 | 20.321 ± 2.248 | 27.289 ± 2.622 | 17.303 ± 4.720 | 19.800 ± 1.976 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 24.126 ± 2.227 | - | 25.184 ± 2.900 | 18.791 ± 1.735 | 33.099 ± 5.858 | 20.608 ± 4.072 | 23.141 ± 17.356 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 18.385 ± 0.733 | - | 21.858 ± 0.835 | 18.758 ± 2.643 | 23.655 ± 4.923 | 20.798 ± 4.243 | 21.559 ± 2.836 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 18.296 ± 4.329 | - | 17.447 ± 6.658 | 13.330 ± 2.471 | 19.488 ± 9.441 | 12.494 ± 1.276 | 21.745 ± 22.070 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 12.051 ± 1.141 | - | 13.024 ± 1.849 | 10.772 ± 2.177 | 11.024 ± 1.189 | 14.559 ± 4.741 | 14.904 ± 8.571 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 44.869 ± 5.246 | - | 38.718 ± 3.017 | 97.879 ± 33.667 | 62.111 ± 18.953 | 110.436 ± 37.021 | 118.632 ± 41.915 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 17.171 ± 1.962 | - | 18.163 ± 3.950 | 30.820 ± 10.238 | 20.592 ± 0.440 | 76.667 ± 12.587 | 32.406 ± 8.284 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 12.817 ± 5.464 | - | 10.558 ± 3.256 | 18.852 ± 15.378 | 33.720 ± 5.263 | 7.556 ± 1.543 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.975 ± 3.529 | - | 10.498 ± 5.049 | 7.000 ± 0.901 | 15.292 ± 0.683 | 6.809 ± 1.607 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.960 ± 0.744 | - | 1.274 ± 0.226 | 19.837 ± 3.974 | 2.775 ± 0.122 | 0.914 ± 0.115 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.843 ± 0.530 | - | 0.765 ± 0.159 | 12.248 ± 1.306 | 1.481 ± 0.234 | 0.858 ± 0.308 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.974 ± 0.185 | - | 1.223 ± 1.042 | 1.793 ± 0.139 | 1.257 ± 0.208 | 1.033 ± 0.077 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 2.853 ± 1.117 | - | 2.614 ± 0.913 | 1.220 ± 0.201 | 0.741 ± 0.086 | 0.937 ± 0.243 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 12.573 ± 4.524 | - | 13.674 ± 9.253 | 14.611 ± 3.596 | 32.234 ± 6.537 | 7.206 ± 1.396 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 8.773 ± 4.016 | - | 6.117 ± 0.672 | 7.380 ± 1.164 | 15.365 ± 2.533 | 5.743 ± 0.580 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.110 ± 0.025 | - | 1.067 ± 0.045 | 20.471 ± 2.656 | 2.586 ± 0.296 | 0.947 ± 0.083 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.966 ± 0.494 | - | 0.875 ± 0.128 | 12.150 ± 1.990 | 1.723 ± 0.231 | 0.828 ± 0.096 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 5.565 ± 3.185 | - | 6.263 ± 4.556 | 6.753 ± 3.467 | 4.544 ± 0.750 | 2.989 ± 0.194 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 8.129 ± 0.844 | - | 9.018 ± 1.611 | 4.840 ± 1.920 | 3.846 ± 0.479 | 3.559 ± 0.344 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 11.300 ± 4.069 | - | 13.185 ± 4.140 | 10.790 ± 5.508 | 40.703 ± 1.800 | 7.071 ± 1.265 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.343 ± 0.566 | - | 6.897 ± 1.170 | 6.601 ± 0.986 | 22.868 ± 1.082 | 6.123 ± 0.416 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.059 ± 0.121 | - | 1.760 ± 1.083 | 4.998 ± 0.371 | 3.223 ± 1.062 | 0.926 ± 0.086 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 1.291 ± 0.250 | - | 0.874 ± 0.063 | 3.006 ± 0.365 | 1.489 ± 0.101 | 1.722 ± 0.117 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 2.552 ± 0.722 | - | 1.101 ± 0.917 | 0.902 ± 0.045 | 1.630 ± 0.807 | 0.927 ± 0.039 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 3.110 ± 1.291 | - | 2.591 ± 0.853 | 1.032 ± 0.082 | 0.879 ± 0.078 | 1.000 ± 1.516 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 11.298 ± 4.001 | - | 9.693 ± 2.042 | 6.730 ± 8.735 | 32.307 ± 4.075 | 6.827 ± 1.860 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.889 ± 0.808 | - | 5.320 ± 0.564 | 6.075 ± 1.060 | 17.384 ± 0.146 | 5.791 ± 0.420 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.295 ± 0.635 | - | 0.893 ± 0.050 | 4.695 ± 0.217 | 2.673 ± 1.389 | 0.907 ± 0.050 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.831 ± 0.144 | - | 0.818 ± 0.138 | 3.004 ± 0.878 | 1.181 ± 0.323 | 1.879 ± 1.109 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.922 ± 0.731 | - | 1.291 ± 1.235 | 0.809 ± 0.089 | 1.334 ± 0.188 | 0.937 ± 0.062 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 1.944 ± 0.998 | - | 2.762 ± 1.121 | 0.871 ± 0.108 | 1.010 ± 0.246 | 0.869 ± 1.158 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.112 ± 0.202 | 1.246 ± 0.350 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 1.111 ± 0.159 | 0.768 ± 0.061 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 29.244 ± 2.097 | - | 30.741 ± 1.052 | 34.737 ± 3.022 | 43.786 ± 6.851 | 83.742 ± 40.512 | 37.172 ± 4.333 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 16.165 ± 7.501 | - | 24.361 ± 2.781 | 15.492 ± 3.580 | 21.632 ± 0.432 | 35.857 ± 11.447 | 18.603 ± 2.926 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 21.445 ± 2.902 | - | 21.598 ± 2.472 | 20.958 ± 1.756 | 38.907 ± 6.684 | 54.586 ± 23.963 | 46.922 ± 4.473 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 20.454 ± 2.853 | - | 16.191 ± 1.935 | 13.700 ± 4.247 | 23.119 ± 2.797 | 45.545 ± 4.850 | 20.012 ± 3.356 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 24.222 ± 2.596 | - | 28.926 ± 7.132 | 27.517 ± 4.082 | 31.010 ± 4.741 | 28.928 ± 1.782 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 27.631 ± 2.669 | - | 25.231 ± 2.764 | 20.944 ± 2.682 | 22.695 ± 1.037 | 29.410 ± 3.967 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 16.984 ± 6.568 | - | 12.262 ± 0.735 | 11.567 ± 0.993 | 21.593 ± 6.161 | 12.133 ± 0.324 | 21.752 ± 4.823 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 12.429 ± 1.390 | - | 11.899 ± 1.317 | 13.176 ± 1.101 | 10.360 ± 0.254 | 13.282 ± 2.952 | 14.514 ± 1.319 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 27.009 ± 0.413 | - | 24.495 ± 3.550 | 39.232 ± 7.654 | 28.034 ± 1.330 | 41.487 ± 14.728 | 49.607 ± 19.030 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 15.320 ± 4.117 | - | 14.005 ± 4.029 | 13.050 ± 2.249 | 14.526 ± 0.708 | 30.385 ± 2.283 | 19.209 ± 6.379 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 12.780 ± 1.504 | - | 11.699 ± 0.766 | 14.874 ± 5.994 | 25.154 ± 5.707 | 60.658 ± 182.351 | 35.543 ± 11.235 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 21.271 ± 4.783 | - | 14.995 ± 2.670 | 12.503 ± 0.468 | 12.681 ± 1.257 | 28.993 ± 1.558 | 15.645 ± 8.017 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 20.773 ± 2.875 | - | 21.692 ± 3.606 | 18.833 ± 0.807 | 34.810 ± 8.147 | 23.243 ± 2.871 | 21.753 ± 2.552 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 23.140 ± 2.267 | - | 23.334 ± 3.907 | 16.704 ± 2.605 | 24.138 ± 1.305 | 25.882 ± 20.871 | 18.800 ± 3.711 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 75.598 ± 4.632 | - | 76.987 ± 11.298 | 75.564 ± 15.473 | 19.274 ± 9.069 | 60.338 ± 17.132 | 63.422 ± 16.634 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 34.631 ± 3.138 | - | 38.916 ± 9.870 | 26.300 ± 7.092 | 7.740 ± 0.639 | 44.622 ± 1.556 | 29.959 ± 8.219 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 23.974 ± 4.056 | 27.669 ± 3.403 | unsupported | 24.312 ± 7.451 | 28.383 ± 4.000 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 27.700 ± 1.038 | 27.023 ± 0.730 | unsupported | 22.576 ± 3.493 | 22.283 ± 1.505 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.848 ± 0.058 | - | 1.237 ± 1.160 | 0.664 ± 0.033 | 0.737 ± 0.063 | 0.827 ± 0.069 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.731 ± 0.073 | - | 0.914 ± 0.104 | 1.009 ± 0.207 | 0.587 ± 0.078 | 0.880 ± 0.068 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.995 ± 0.722 | - | 0.891 ± 0.281 | 0.751 ± 0.104 | 0.753 ± 0.147 | 0.914 ± 0.322 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.729 ± 0.084 | - | 0.744 ± 0.048 | 0.949 ± 0.365 | 1.181 ± 0.078 | 0.835 ± 0.138 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 2.515 ± 1.115 | - | unsupported | 1.184 ± 0.115 | 1.755 ± 0.464 | 1.187 ± 3.748 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.139 ± 0.109 | - | unsupported | 1.429 ± 0.139 | 1.000 ± 0.044 | 1.302 ± 3.153 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.397 ± 0.097 | - | 0.485 ± 0.180 | 0.356 ± 0.809 | 0.401 ± 0.055 | 0.929 ± 0.080 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.275 ± 0.004 | - | 0.349 ± 0.029 | 0.339 ± 0.021 | 0.242 ± 0.043 | 2.568 ± 1.842 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 1.207 ± 0.355 | - | 0.871 ± 0.078 | 1.416 ± 0.544 | 1.018 ± 0.289 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.704 ± 0.065 | - | 0.875 ± 0.078 | 1.454 ± 0.341 | 0.614 ± 0.025 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 1.000 ± 1.083 | - | 1.024 ± 0.677 | 0.812 ± 0.292 | 0.867 ± 0.181 | 1.435 ± 9.234 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.823 ± 0.076 | - | 0.821 ± 0.366 | 0.919 ± 0.127 | 0.796 ± 0.087 | 1.751 ± 5.319 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.684 ± 0.055 | - | 0.785 ± 1.248 | 0.456 ± 0.050 | 0.925 ± 0.282 | 0.376 ± 0.044 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.604 ± 0.106 | - | 0.706 ± 0.044 | 0.908 ± 0.512 | 0.437 ± 0.059 | 0.389 ± 0.196 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.396 ± 0.220 | - | 0.004 ± 0.000 | 1.398 ± 0.177 | 1.412 ± 0.268 | 1.571 ± 0.304 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.110 ± 0.054 | - | 0.001 ± 0.000 | 1.406 ± 0.192 | 1.804 ± 0.407 | 1.826 ± 0.763 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.051 ± 0.009 | - | 0.058 ± 0.005 | 0.120 ± 0.007 | 0.099 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.109 ± 0.009 | - | 0.111 ± 0.031 | 0.171 ± 0.004 | 0.136 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.266 ± 0.070 | - | 0.240 ± 0.053 | 0.284 ± 0.013 | 0.275 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.061 ± 0.004 | - | 0.052 ± 0.004 | 0.039 ± 0.002 | 0.106 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.035 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.001 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.436 ± 0.044 | - | 0.444 ± 0.027 | 0.265 ± 0.012 | 0.375 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.018 ± 0.003 | - | 0.025 ± 0.003 | 0.017 ± 0.002 | 0.051 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.006 ± 0.000 | 0.056 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.048 ± 0.002 | - | 0.049 ± 0.000 | 0.141 ± 0.046 | 0.134 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.100 ± 0.001 | - | 0.101 ± 0.000 | 0.185 ± 0.018 | 0.116 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.219 ± 0.006 | - | 0.221 ± 0.006 | 0.270 ± 0.038 | 0.304 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.050 ± 0.000 | - | 0.051 ± 0.000 | 0.089 ± 0.026 | 0.086 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.043 ± 0.018 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.050 ± 0.006 | 0.063 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.001 | 0.057 ± 0.024 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.001 | 0.055 ± 0.031 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.047 ± 0.012 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.001 | 0.054 ± 0.034 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.041 ± 0.007 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.051 ± 0.011 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.052 ± 0.015 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.045 ± 0.009 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.318 ± 0.040 | - | 0.362 ± 0.143 | 0.246 ± 0.075 | 0.286 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.046 ± 0.025 | 0.027 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.018 ± 0.000 | - | 0.020 ± 0.003 | 0.071 ± 0.026 | 0.083 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.041 ± 0.022 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.045 ± 0.006 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.360 ± 0.026 | - | 0.315 ± 0.217 | 0.490 ± 0.131 | 1.014 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.526 ± 0.127 | - | 1.589 ± 0.286 | 2.015 ± 1.242 | 3.323 ± 0.934 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 5.660 ± 1.165 | - | 5.045 ± 0.385 | 9.283 ± 2.067 | 9.284 ± 2.368 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.413 ± 0.056 | - | 2.521 ± 0.230 | 5.257 ± 0.973 | 4.445 ± 1.851 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.005 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.084 ± 0.012 | - | 0.078 ± 0.000 | 0.101 ± 0.021 | 0.181 ± 0.115 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.008 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.053 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.003 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.015 ± 0.000 | - | 0.020 ± 0.003 | 0.027 ± 0.003 | 0.102 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.012 ± 0.027 | 0.006 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.007 ± 0.002 | - | 0.010 ± 0.002 | 0.014 ± 0.002 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.001 | 0.034 ± 0.002 | 0.073 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 23.068 ± 3.850 | - | 21.501 ± 5.167 | 31.489 ± 5.030 | 43.856 ± 9.941 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.001 | 0.008 ± 0.002 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.617 ± 0.019 | - | 0.628 ± 0.016 | 1.010 ± 0.871 | 1.169 ± 0.112 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.015 ± 0.003 | - | 0.014 ± 0.000 | 0.018 ± 0.004 | 0.052 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.040 ± 0.007 | - | 0.040 ± 0.005 | 0.062 ± 0.013 | 0.121 ± 0.081 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.305 ± 0.014 | - | 0.296 ± 0.017 | 0.493 ± 0.036 | 0.910 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 0.738 ± 0.015 | - | 0.747 ± 0.026 | 1.866 ± 0.197 | 1.520 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 2.614 ± 0.228 | - | 2.679 ± 0.295 | 6.956 ± 0.179 | 4.846 ± 0.845 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 2.739 ± 1.783 | - | 2.385 ± 0.097 | 3.322 ± 0.156 | 4.224 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.013 | 0.014 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.077 ± 0.000 | - | 0.098 ± 0.335 | 0.091 ± 0.002 | 0.158 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.013 ± 0.007 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.007 ± 0.001 | - | 0.008 ± 0.002 | 0.010 ± 0.002 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.015 ± 0.004 | - | 0.016 ± 0.000 | 0.023 ± 0.002 | 0.130 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.015 ± 0.025 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.009 ± 0.001 | - | 0.009 ± 0.002 | 0.011 ± 0.001 | 0.019 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.020 ± 0.003 | - | 0.020 ± 0.004 | 0.034 ± 0.003 | 0.109 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 8.697 ± 2.041 | - | 9.211 ± 2.266 | 26.261 ± 1.101 | 16.831 ± 1.354 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.022 ± 0.008 | 0.016 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.372 ± 0.073 | - | 0.341 ± 0.005 | 0.706 ± 0.023 | 0.696 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.013 ± 0.003 | - | 0.014 ± 0.001 | 0.018 ± 0.003 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.037 ± 0.001 | - | 0.038 ± 0.000 | 0.062 ± 0.006 | 0.141 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.305 ± 0.005 | - | 0.315 ± 0.023 | 0.605 ± 0.421 | 0.633 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.364 ± 0.117 | - | 2.352 ± 1.505 | 1.446 ± 0.380 | 2.106 ± 0.663 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.974 ± 0.698 | - | 4.466 ± 0.344 | 6.586 ± 1.761 | 7.865 ± 1.574 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.505 ± 0.107 | - | 3.213 ± 1.344 | 1.341 ± 0.643 | 1.175 ± 0.240 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.062 ± 0.008 | - | 0.063 ± 0.013 | 0.069 ± 0.002 | 0.113 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.018 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.015 ± 0.005 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.002 | 0.025 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.014 ± 0.003 | - | 0.015 ± 0.002 | 0.019 ± 0.084 | 0.031 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.007 ± 0.001 | - | 0.009 ± 0.002 | 0.029 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.017 ± 0.003 | - | 0.021 ± 0.003 | 0.024 ± 0.006 | 0.035 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 14.111 ± 3.429 | - | 12.296 ± 0.532 | 12.539 ± 4.986 | 11.798 ± 5.883 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.007 ± 0.000 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.542 ± 0.016 | - | 0.547 ± 0.097 | 0.711 ± 0.686 | 0.597 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.013 ± 0.002 | - | 0.012 ± 0.001 | 0.042 ± 0.027 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.035 ± 0.005 | - | 0.036 ± 0.009 | 0.047 ± 0.007 | 0.101 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.358 ± 0.120 | - | 0.305 ± 0.133 | 0.234 ± 0.007 | 0.600 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.336 ± 0.140 | - | 1.230 ± 0.040 | 1.335 ± 0.092 | 1.010 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 2.076 ± 0.022 | - | 2.106 ± 0.044 | 5.290 ± 0.324 | 3.070 ± 0.119 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.599 ± 1.002 | - | 1.380 ± 0.029 | 1.011 ± 0.035 | 1.167 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.024 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.059 ± 0.000 | - | 0.063 ± 0.007 | 0.066 ± 0.005 | 0.142 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.006 ± 0.001 | - | 0.007 ± 0.000 | 0.008 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.005 ± 0.001 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.013 ± 0.004 | - | 0.015 ± 0.000 | 0.021 ± 0.004 | 0.031 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.008 ± 0.001 | - | 0.007 ± 0.002 | 0.009 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.017 ± 0.000 | - | 0.018 ± 0.000 | 0.023 ± 0.000 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 5.437 ± 0.784 | - | 5.756 ± 0.927 | 7.874 ± 0.159 | 5.159 ± 0.360 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.006 ± 0.001 | 0.005 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.283 ± 0.011 | - | 0.285 ± 0.003 | 0.521 ± 0.033 | 0.314 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.012 ± 0.003 | - | 0.013 ± 0.003 | 0.013 ± 0.000 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.034 ± 0.000 | - | 0.034 ± 0.004 | 0.041 ± 0.001 | 0.166 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.555 ± 0.391 | - | 0.236 ± 0.011 | 0.337 ± 0.199 | 0.794 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.316 ± 0.113 | - | 1.234 ± 0.237 | 0.956 ± 0.370 | 2.852 ± 0.917 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 4.628 ± 0.688 | - | 4.667 ± 0.611 | 3.359 ± 0.727 | 6.519 ± 1.936 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.971 ± 0.536 | - | 1.576 ± 0.767 | 0.559 ± 0.388 | 1.288 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.047 ± 0.006 | - | 0.054 ± 0.005 | 0.030 ± 0.005 | 0.135 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.001 | 0.040 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.006 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.014 ± 0.000 | 0.020 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.012 ± 0.000 | - | 0.013 ± 0.003 | 0.014 ± 0.002 | 0.090 ± 0.109 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.011 ± 0.000 | - | 0.007 ± 0.000 | 0.022 ± 0.002 | 0.077 ± 0.191 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.015 ± 0.000 | - | 0.018 ± 0.004 | 0.018 ± 0.002 | 0.033 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 6.453 ± 1.099 | - | 7.219 ± 2.471 | 5.050 ± 1.850 | 8.930 ± 0.469 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.411 ± 0.390 | - | 0.762 ± 0.328 | 0.306 ± 0.312 | 0.637 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.009 ± 0.003 | - | 0.014 ± 0.003 | 0.027 ± 0.000 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.031 ± 0.007 | - | 0.031 ± 0.004 | 0.027 ± 0.005 | 0.206 ± 0.136 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.215 ± 0.002 | - | 0.219 ± 0.018 | 0.137 ± 0.013 | 0.815 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.311 ± 0.311 | - | 1.179 ± 0.045 | 0.799 ± 0.069 | 1.467 ± 0.106 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 1.952 ± 0.039 | - | 1.922 ± 0.289 | 2.542 ± 0.055 | 3.573 ± 0.418 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.796 ± 0.064 | - | 0.778 ± 0.043 | 0.500 ± 0.019 | 1.321 ± 0.488 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.001 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.059 ± 0.028 | - | 0.047 ± 0.000 | 0.030 ± 0.004 | 0.147 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.014 ± 0.003 | - | 0.013 ± 0.003 | 0.011 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.016 ± 0.004 | - | 0.016 ± 0.000 | 0.014 ± 0.000 | 0.029 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 3.039 ± 0.110 | - | 3.101 ± 0.162 | 3.909 ± 0.148 | 5.539 ± 0.565 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.001 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.241 ± 0.028 | - | 0.228 ± 0.016 | 0.205 ± 0.025 | 0.657 ± 0.310 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.010 ± 0.000 | - | 0.013 ± 0.003 | 0.013 ± 0.001 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.030 ± 0.005 | - | 0.031 ± 0.004 | 0.022 ± 0.001 | 0.109 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.072 ± 0.010 | - | 0.122 ± 0.044 | 0.163 ± 0.008 | 0.294 ± 0.220 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.198 ± 0.033 | - | 0.181 ± 0.020 | 0.275 ± 0.026 | 0.274 ± 0.146 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.496 ± 0.071 | - | 0.474 ± 0.034 | 0.770 ± 0.243 | 0.575 ± 0.175 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.166 ± 0.051 | - | 0.152 ± 0.013 | 0.150 ± 0.143 | 0.195 ± 0.108 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.020 ± 0.002 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.010 ± 0.002 | - | 0.010 ± 0.003 | 0.011 ± 0.001 | 0.053 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.002 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.014 ± 0.001 | - | 0.014 ± 0.003 | 0.016 ± 0.035 | 0.077 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.007 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.023 ± 0.000 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.008 ± 0.002 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.002 | 0.058 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.113 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.180 ± 0.043 | - | 1.161 ± 0.008 | 0.583 ± 0.104 | 0.645 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.010 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.093 ± 0.096 | - | 0.075 ± 0.058 | 0.045 ± 0.005 | 0.097 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.051 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.002 | 0.048 ± 0.116 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.071 ± 0.002 | - | 0.075 ± 0.009 | 0.327 ± 0.082 | 0.126 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.163 ± 0.010 | - | 0.173 ± 0.017 | 0.412 ± 0.069 | 0.204 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.434 ± 0.012 | - | 0.476 ± 0.020 | 0.692 ± 0.374 | 0.366 ± 0.118 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.216 ± 0.065 | - | 0.247 ± 0.038 | 0.142 ± 0.081 | 0.128 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.054 ± 0.024 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.011 ± 0.002 | - | 0.012 ± 0.002 | 0.105 ± 0.051 | 0.024 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.072 ± 0.035 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.011 ± 0.003 | - | 0.014 ± 0.003 | 0.130 ± 0.081 | 0.083 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.118 ± 0.063 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.117 ± 0.108 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.110 ± 0.053 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.069 ± 0.011 | 0.060 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.050 ± 0.019 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.094 ± 0.074 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.073 ± 0.032 | 0.012 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.575 ± 0.025 | - | 0.659 ± 0.129 | 0.836 ± 0.153 | 0.502 ± 0.135 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.110 ± 0.089 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.054 ± 0.000 | - | 0.055 ± 0.001 | 0.167 ± 0.029 | 0.123 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.134 ± 0.068 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.005 ± 0.001 | - | 0.006 ± 0.000 | 0.107 ± 0.014 | 0.021 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.176 ± 0.046 | - | 0.150 ± 0.010 | 0.667 ± 0.133 | 0.755 ± 0.357 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.369 ± 0.018 | - | 0.388 ± 0.037 | 0.872 ± 0.500 | 1.065 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.069 ± 0.038 | - | 1.090 ± 0.086 | 2.380 ± 2.614 | 2.334 ± 0.940 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.438 ± 0.051 | - | 0.439 ± 0.100 | 0.362 ± 0.035 | 0.419 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.003 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.022 ± 0.008 | - | 0.021 ± 0.000 | 0.024 ± 0.007 | 0.105 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.016 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.023 ± 0.001 | - | 0.025 ± 0.004 | 0.017 ± 0.004 | 0.075 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.014 ± 0.002 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.021 ± 0.003 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.005 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 4.265 ± 0.744 | - | 3.655 ± 0.685 | 3.712 ± 0.311 | 3.249 ± 0.230 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.152 ± 0.017 | - | 0.163 ± 0.017 | 0.168 ± 0.018 | 0.308 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.005 ± 0.001 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.009 ± 0.000 | - | 0.010 ± 0.001 | 0.135 ± 0.266 | 0.031 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.149 ± 0.010 | - | 0.145 ± 0.004 | 0.282 ± 0.024 | 0.518 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.342 ± 0.001 | - | 0.363 ± 0.029 | 0.629 ± 0.018 | 0.779 ± 0.135 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.532 ± 0.002 | - | 0.536 ± 0.010 | 1.532 ± 0.083 | 1.954 ± 0.832 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.395 ± 0.029 | - | 0.710 ± 0.623 | 0.433 ± 0.045 | 0.402 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.036 ± 0.010 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.020 ± 0.000 | - | 0.021 ± 0.000 | 0.037 ± 0.018 | 0.080 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.022 ± 0.008 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.023 ± 0.000 | - | 0.025 ± 0.001 | 0.042 ± 0.022 | 0.084 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.060 ± 0.037 | 0.049 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.002 ± 0.001 | - | 0.003 ± 0.001 | 0.039 ± 0.017 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.023 ± 0.007 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.042 ± 0.033 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.021 ± 0.009 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.025 ± 0.011 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.005 ± 0.001 | - | 0.008 ± 0.002 | 0.033 ± 0.004 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.546 ± 1.851 | - | 1.699 ± 0.032 | 2.793 ± 0.171 | 2.463 ± 0.204 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.003 ± 0.002 | - | 0.005 ± 0.000 | 0.034 ± 0.019 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.145 ± 0.001 | - | 0.149 ± 0.006 | 0.138 ± 0.020 | 0.403 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.034 ± 0.019 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.011 ± 0.002 | - | 0.010 ± 0.000 | 0.036 ± 0.002 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.115 ± 0.010 | - | 0.092 ± 0.005 | 0.158 ± 0.031 | 0.236 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.159 ± 0.013 | - | 0.145 ± 0.014 | 0.199 ± 0.024 | 0.292 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.399 ± 0.059 | - | 0.441 ± 0.458 | 0.423 ± 0.028 | 0.498 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.091 ± 0.014 | - | 0.104 ± 0.020 | 0.065 ± 0.001 | 0.119 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.014 ± 0.020 | 0.011 ± 0.003 | 0.010 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.005 ± 0.000 | - | 0.010 ± 0.001 | 0.014 ± 0.003 | 0.072 ± 0.094 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.001 | 0.012 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.003 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.010 ± 0.000 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.002 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.788 ± 0.081 | - | 0.814 ± 0.567 | 0.420 ± 0.010 | 0.564 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.012 ± 0.002 | 0.014 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.095 ± 0.001 | - | 0.108 ± 0.004 | 0.046 ± 0.005 | 0.093 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.003 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.096 ± 0.001 | - | 0.077 ± 0.004 | 0.122 ± 0.022 | 0.182 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.152 ± 0.020 | - | 0.130 ± 0.002 | 0.254 ± 0.125 | 0.295 ± 0.108 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.210 ± 0.006 | - | 0.235 ± 0.000 | 0.312 ± 0.024 | 0.630 ± 0.602 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.118 ± 0.004 | - | 0.118 ± 0.010 | 0.070 ± 0.026 | 0.091 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.009 | 0.012 ± 0.002 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.004 | 0.061 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.006 ± 0.002 | 0.010 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.002 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.006 ± 0.002 | 0.013 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.367 ± 0.007 | - | 0.435 ± 0.258 | 0.316 ± 0.036 | 1.354 ± 0.197 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.046 ± 0.010 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.034 ± 0.001 | - | 0.040 ± 0.000 | 0.074 ± 0.041 | 0.124 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.002 | 0.031 ± 0.019 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.005 ± 0.002 | - | 0.007 ± 0.001 | 0.038 ± 0.004 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 0.825 ± 0.111 | - | 0.699 ± 0.259 | 1.381 ± 0.524 | 6.082 ± 1.926 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 3.852 ± 0.864 | - | 3.457 ± 1.682 | 5.098 ± 1.052 | 8.059 ± 2.268 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 12.403 ± 5.701 | - | 10.286 ± 3.446 | 20.125 ± 4.116 | 20.778 ± 8.571 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.990 ± 1.574 | - | 9.217 ± 4.169 | 5.074 ± 0.087 | 5.932 ± 0.500 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.002 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.155 ± 0.006 | - | 0.149 ± 0.009 | 0.183 ± 0.002 | 0.610 ± 0.194 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.011 ± 0.003 | - | 0.016 ± 0.000 | 0.013 ± 0.003 | 0.027 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.132 ± 0.026 | - | 0.131 ± 0.015 | 0.190 ± 0.018 | 0.298 ± 0.409 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.012 ± 0.002 | - | 0.014 ± 0.000 | 0.022 ± 0.003 | 0.032 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.018 | 0.027 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.021 ± 0.001 | - | 0.026 ± 0.000 | 0.019 ± 0.001 | 0.036 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.032 ± 0.006 | - | 0.027 ± 0.000 | 0.059 ± 0.011 | 0.070 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.004 ± 0.001 | - | 0.007 ± 0.000 | 0.011 ± 0.003 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.025 ± 0.015 | - | 0.033 ± 0.001 | 0.024 ± 0.005 | 0.038 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.035 ± 0.005 | - | 0.035 ± 0.000 | 0.071 ± 0.009 | 0.088 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 63.193 ± 6.862 | - | 59.257 ± 10.829 | 42.781 ± 3.546 | 52.702 ± 6.267 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.019 ± 0.001 | 0.047 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.207 ± 0.186 | - | 1.529 ± 1.846 | 1.607 ± 0.061 | 2.004 ± 0.519 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.023 ± 0.017 | - | 0.024 ± 0.004 | 0.043 ± 0.004 | 0.062 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.069 ± 0.008 | - | 0.078 ± 0.006 | 0.150 ± 0.045 | 0.248 ± 0.338 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 0.725 ± 0.029 | - | 0.679 ± 0.054 | 1.248 ± 0.114 | 2.984 ± 0.144 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 1.360 ± 0.067 | - | 2.122 ± 0.274 | 4.508 ± 0.190 | 3.829 ± 0.927 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 5.625 ± 1.008 | - | 4.872 ± 0.606 | 15.281 ± 0.936 | 8.470 ± 0.173 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.728 ± 0.220 | - | 4.656 ± 0.467 | 4.239 ± 0.146 | 4.743 ± 0.295 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.143 ± 0.008 | - | 0.139 ± 0.003 | 0.193 ± 0.014 | 0.243 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.005 ± 0.001 | - | 0.006 ± 0.001 | 0.010 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.124 ± 0.002 | - | 0.127 ± 0.006 | 0.172 ± 0.008 | 0.206 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.013 ± 0.001 | - | 0.013 ± 0.003 | 0.021 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.003 | 0.033 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.030 ± 0.006 | - | 0.032 ± 0.006 | 0.049 ± 0.000 | 0.061 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.005 ± 0.001 | - | 0.007 ± 0.000 | 0.013 ± 0.003 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.012 ± 0.002 | - | 0.014 ± 0.003 | 0.022 ± 0.001 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.037 ± 0.006 | - | 0.042 ± 0.005 | 0.076 ± 0.005 | 0.081 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 17.348 ± 2.742 | - | 19.303 ± 2.927 | 38.430 ± 2.317 | 21.442 ± 1.577 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.008 ± 0.002 | 0.016 ± 0.001 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 0.579 ± 0.007 | - | 0.612 ± 0.048 | 1.448 ± 0.097 | 0.862 ± 0.269 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.022 ± 0.004 | - | 0.025 ± 0.005 | 0.036 ± 0.000 | 0.060 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.068 ± 0.003 | - | 0.089 ± 0.020 | 0.143 ± 0.006 | 0.171 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.612 ± 0.214 | - | 0.505 ± 0.020 | 0.682 ± 0.332 | 2.590 ± 0.226 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.408 ± 0.152 | - | 2.330 ± 0.185 | 3.144 ± 0.284 | 4.741 ± 0.294 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 7.845 ± 0.619 | - | 7.668 ± 0.218 | 14.100 ± 3.609 | 15.751 ± 1.955 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.101 ± 0.276 | - | 2.527 ± 3.156 | 1.608 ± 0.136 | 1.880 ± 0.434 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.112 ± 0.012 | - | 0.104 ± 0.011 | 0.133 ± 0.013 | 0.236 ± 0.115 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.002 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.078 ± 0.007 | - | 0.078 ± 0.009 | 0.111 ± 0.005 | 0.365 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.010 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.001 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.006 | 0.071 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.026 ± 0.005 | - | 0.022 ± 0.000 | 0.120 ± 0.098 | 0.047 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.021 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.021 ± 0.001 | 0.034 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.028 ± 0.000 | - | 0.029 ± 0.007 | 0.123 ± 0.002 | 0.064 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 21.106 ± 4.809 | - | 27.237 ± 8.922 | 14.477 ± 0.766 | 14.326 ± 1.298 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.002 | 0.028 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.888 ± 0.074 | - | 0.902 ± 0.049 | 1.000 ± 0.046 | 2.454 ± 0.824 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.019 ± 0.000 | 0.030 ± 0.005 | 0.051 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.056 ± 0.000 | - | 0.067 ± 0.013 | 0.157 ± 0.077 | 0.139 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.524 ± 0.028 | - | 0.530 ± 0.044 | 0.603 ± 0.035 | 3.210 ± 2.060 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.209 ± 0.107 | - | 2.145 ± 0.068 | 2.550 ± 0.035 | 3.436 ± 0.757 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 3.771 ± 0.097 | - | 4.029 ± 0.721 | 11.118 ± 1.429 | 6.801 ± 0.568 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.981 ± 0.070 | - | 2.063 ± 0.157 | 1.548 ± 0.084 | 1.716 ± 0.252 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.189 ± 0.097 | - | 0.171 ± 0.058 | 0.135 ± 0.013 | 0.174 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.002 | 0.010 ± 0.002 | 0.029 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.077 ± 0.002 | - | 0.078 ± 0.013 | 0.102 ± 0.009 | 0.204 ± 0.304 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.037 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.052 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.008 ± 0.001 | - | 0.010 ± 0.002 | 0.013 ± 0.004 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.021 ± 0.000 | - | 0.026 ± 0.006 | 0.033 ± 0.001 | 0.046 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.012 ± 0.003 | - | 0.014 ± 0.003 | 0.016 ± 0.002 | 0.035 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.028 ± 0.000 | - | 0.029 ± 0.000 | 0.042 ± 0.000 | 0.061 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 9.381 ± 1.491 | - | 8.057 ± 1.569 | 12.651 ± 2.424 | 7.347 ± 2.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.006 ± 0.001 | - | 0.007 ± 0.000 | 0.012 ± 0.003 | 0.078 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.630 ± 0.131 | - | 0.475 ± 0.070 | 1.015 ± 0.057 | 0.672 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.021 ± 0.004 | - | 0.019 ± 0.004 | 0.027 ± 0.003 | 0.055 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.056 ± 0.000 | - | 0.057 ± 0.004 | 0.078 ± 0.002 | 0.401 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.339 ± 0.036 | - | unsupported | 0.245 ± 0.137 | 1.884 ± 0.283 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.517 ± 0.057 | - | unsupported | 1.626 ± 0.881 | 3.993 ± 3.112 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.820 ± 0.295 | - | unsupported | 5.004 ± 0.565 | 6.802 ± 1.413 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.322 ± 0.101 | - | unsupported | 2.218 ± 1.722 | 0.933 ± 0.398 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.047 ± 0.006 | - | unsupported | 0.050 ± 0.003 | 0.057 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.008 ± 0.002 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.052 ± 0.017 | - | unsupported | 0.035 ± 0.006 | 0.109 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.006 ± 0.001 | - | unsupported | 0.011 ± 0.003 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.013 ± 0.000 | - | unsupported | 0.016 ± 0.004 | 0.029 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.013 ± 0.000 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.017 ± 0.000 | - | unsupported | 0.023 ± 0.003 | 0.036 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 11.037 ± 3.036 | - | unsupported | 6.718 ± 0.581 | 9.855 ± 2.586 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.414 ± 0.012 | - | unsupported | 0.353 ± 0.016 | 0.394 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.034 ± 0.002 | - | unsupported | 0.035 ± 0.000 | 0.080 ± 0.081 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.331 ± 0.027 | - | unsupported | 0.238 ± 0.012 | 1.432 ± 0.188 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.579 ± 0.078 | - | unsupported | 1.445 ± 0.145 | 1.108 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 2.358 ± 0.287 | - | unsupported | 4.222 ± 0.533 | 2.585 ± 0.179 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.217 ± 0.038 | - | unsupported | 0.844 ± 0.042 | 0.826 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.006 ± 0.001 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.046 ± 0.002 | - | unsupported | 0.043 ± 0.004 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.037 ± 0.001 | - | unsupported | 0.033 ± 0.001 | 0.038 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.002 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.003 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.013 ± 0.000 | - | unsupported | 0.018 ± 0.004 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.002 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.017 ± 0.000 | - | unsupported | 0.020 ± 0.004 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 4.796 ± 0.500 | - | unsupported | 6.573 ± 0.219 | 3.165 ± 0.189 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | unsupported | 0.006 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.217 ± 0.033 | - | unsupported | 0.311 ± 0.043 | 0.219 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.033 ± 0.000 | - | unsupported | 0.034 ± 0.003 | 0.047 ± 0.001 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.787 ± 0.093 | - | unsupported | 0.998 ± 0.881 | 1.615 ± 0.985 | 3.963 ± 2.320 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.074 ± 0.070 | - | unsupported | 0.094 ± 0.020 | 0.205 ± 0.070 | 0.092 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.252 ± 0.121 | - | unsupported | 0.165 ± 0.021 | 0.314 ± 0.135 | 0.199 ± 0.054 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.230 ± 0.019 | - | unsupported | 0.317 ± 0.030 | 0.397 ± 0.045 | 0.831 ± 0.058 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.422 ± 0.001 | - | unsupported | 0.548 ± 0.062 | 0.667 ± 0.097 | 3.358 ± 1.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.064 ± 0.006 | - | unsupported | 0.108 ± 0.038 | 0.198 ± 0.026 | 0.071 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.128 ± 0.009 | - | unsupported | 0.127 ± 0.020 | 0.228 ± 0.025 | 0.198 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.218 ± 0.023 | - | unsupported | 0.225 ± 0.013 | 0.490 ± 0.185 | 0.732 ± 0.039 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.439 ± 0.184 | - | unsupported | 0.206 ± 0.018 | 0.679 ± 0.094 | 0.521 ± 0.056 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.103 ± 0.029 | - | unsupported | 0.044 ± 0.001 | 0.089 ± 0.011 | 0.072 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.131 ± 0.027 | - | unsupported | 0.051 ± 0.001 | 0.788 ± 0.788 | 0.097 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.150 ± 0.035 | - | unsupported | 0.081 ± 0.010 | 0.195 ± 0.030 | 0.256 ± 0.044 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.283 ± 0.005 | - | unsupported | 0.198 ± 0.021 | 0.642 ± 0.209 | 0.529 ± 0.058 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.032 ± 0.004 | - | unsupported | 0.050 ± 0.006 | 0.230 ± 0.108 | 0.060 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.042 ± 0.000 | - | unsupported | 0.058 ± 0.009 | 0.342 ± 0.090 | 0.087 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.123 ± 0.001 | - | unsupported | 0.095 ± 0.009 | 0.458 ± 0.275 | 0.218 ± 0.017 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.211 ± 0.028 | - | 0.214 ± 0.012 | 0.111 ± 0.029 | 0.185 ± 0.036 | 0.269 ± 0.012 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.021 ± 0.003 | - | 0.031 ± 0.013 | 0.029 ± 0.005 | 0.079 ± 0.003 | 0.042 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.034 ± 0.059 | - | 0.033 ± 0.007 | 0.033 ± 0.019 | 0.035 ± 0.004 | 0.053 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.062 ± 0.001 | - | 0.078 ± 0.007 | 0.047 ± 0.033 | 0.058 ± 0.004 | 0.119 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.187 ± 0.007 | - | 0.196 ± 0.009 | 0.101 ± 0.025 | 0.234 ± 0.021 | 0.248 ± 0.021 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.021 ± 0.000 | - | 0.030 ± 0.009 | 0.025 ± 0.013 | 0.029 ± 0.016 | 0.035 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.026 ± 0.000 | - | 0.028 ± 0.000 | 0.034 ± 0.021 | 0.032 ± 0.002 | 0.045 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.060 ± 0.000 | - | 0.062 ± 0.001 | 0.051 ± 0.018 | 0.101 ± 0.054 | 0.101 ± 0.011 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 50.373 ± 17.682 | - | 61.041 ± 10.642 | 31.186 ± 5.972 | 18.154 ± 8.424 | 42.419 ± 14.357 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 29.618 ± 0.882 | - | 29.597 ± 1.056 | 16.478 ± 1.512 | 15.425 ± 4.407 | 403.918 ± 199.729 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 31.421 ± 6.852 | - | 29.398 ± 7.089 | 8.100 ± 0.775 | 12.101 ± 3.777 | 19.739 ± 1.329 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 26.405 ± 5.249 | - | 24.333 ± 3.757 | 7.060 ± 0.310 | 7.825 ± 2.458 | 237.206 ± 20.469 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 10.053 ± 4.578 | - | 11.988 ± 3.167 | 7.431 ± 1.152 | 8.531 ± 1.137 | 6.728 ± 0.134 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 8.672 ± 0.347 | - | 8.746 ± 0.454 | 6.950 ± 0.351 | 9.323 ± 0.991 | 329.588 ± 33.418 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 14.448 ± 2.600 | - | 14.220 ± 1.196 | 8.104 ± 0.186 | 10.357 ± 6.137 | 6.836 ± 0.089 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 11.495 ± 2.757 | - | 11.803 ± 0.645 | 8.143 ± 0.333 | 7.239 ± 1.001 | 274.712 ± 62.700 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 25.428 ± 12.649 | - | 22.017 ± 8.046 | 24.141 ± 1.898 | 38.996 ± 7.068 | 12.162 ± 0.500 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 19.875 ± 3.236 | - | 19.857 ± 0.657 | 22.086 ± 1.327 | 27.444 ± 3.306 | 645.781 ± 56.543 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.372 ± 0.017 | 0.387 ± 0.141 | unsupported | 0.410 ± 0.103 | 0.397 ± 0.026 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 10.102 ± 2.028 | 8.543 ± 1.854 | unsupported | 4.838 ± 0.664 | 4.865 ± 1.092 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.794 ± 0.089 | 0.737 ± 0.059 | unsupported | 0.314 ± 0.012 | 0.388 ± 0.034 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 8.949 ± 1.482 | 8.314 ± 0.973 | unsupported | 3.820 ± 0.188 | 4.038 ± 0.222 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 42.850 ± 8.906 | - | 37.396 ± 6.028 | 11.099 ± 1.825 | 13.084 ± 1.069 | 28.642 ± 0.573 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 47.474 ± 1.149 | - | 33.500 ± 5.419 | 8.683 ± 0.225 | 9.146 ± 0.317 | 369.340 ± 283.372 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 24.939 ± 8.076 | 21.279 ± 3.995 | 5.479 ± 2.222 | 34.634 ± 3.440 | 7.694 ± 0.041 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 17.565 ± 5.857 | 19.193 ± 14.594 | 4.825 ± 0.433 | 32.321 ± 3.172 | 131.309 ± 18.540 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 32.583 ± 6.126 | - | 34.826 ± 11.637 | 14.197 ± 3.927 | 16.134 ± 3.890 | 21.100 ± 9.670 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 27.213 ± 4.509 | - | 25.428 ± 6.436 | 13.034 ± 5.966 | 11.051 ± 1.183 | 238.244 ± 60.568 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.794 ± 0.217 | - | 0.821 ± 0.035 | 2.579 ± 0.208 | 6.533 ± 0.828 | 5.277 ± 0.052 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.793 ± 0.108 | - | 0.836 ± 0.035 | 2.234 ± 0.047 | 3.471 ± 0.437 | 7.660 ± 1.783 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 24.776 ± 8.987 | - | 21.481 ± 5.806 | 14.211 ± 5.868 | 15.830 ± 1.439 | 17.999 ± 0.552 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 22.926 ± 1.945 | - | 31.523 ± 5.669 | 10.881 ± 0.383 | 12.181 ± 1.834 | 984.193 ± 267.774 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 25.818 ± 9.660 | - | 31.495 ± 11.360 | 11.546 ± 1.554 | 16.198 ± 2.488 | 18.122 ± 0.200 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 17.960 ± 8.821 | - | 21.489 ± 13.837 | 10.356 ± 1.608 | 12.258 ± 2.093 | 999.058 ± 503.523 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.278 ± 0.238 | 1.345 ± 0.263 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 33.533 ± 4.086 | 45.166 ± 9.076 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.110 ± 0.036 | 1.111 ± 0.051 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 22.667 ± 0.667 | 28.263 ± 7.970 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.298 ± 0.091 | 1.439 ± 1.076 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 37.601 ± 3.446 | 32.486 ± 1.787 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.084 ± 0.048 | 1.141 ± 0.054 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 20.897 ± 4.245 | 22.111 ± 5.172 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 30.459 ± 9.715 | - | 25.432 ± 4.441 | 8.865 ± 2.625 | 9.008 ± 0.146 | 19.753 ± 2.117 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 29.660 ± 4.000 | - | 25.338 ± 5.613 | 7.322 ± 1.292 | 8.139 ± 0.569 | 360.970 ± 211.746 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 78.738 ± 6.522 | 75.052 ± 12.588 | 42.517 ± 6.352 | 38.271 ± 1.246 | 64.249 ± 7.559 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 63.938 ± 9.121 | 72.291 ± 8.715 | 36.122 ± 8.394 | 34.496 ± 1.843 | 1676.115 ± 121.714 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 53.793 ± 9.842 | - | 60.072 ± 13.170 | 14.054 ± 1.549 | 80.190 ± 4.183 | 27.383 ± 0.606 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 30.095 ± 1.076 | - | 30.445 ± 2.288 | 12.915 ± 0.626 | 43.789 ± 8.212 | 23.577 ± 4.893 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.194 ± 0.011 | 0.196 ± 0.015 | unsupported | 0.065 ± 0.013 | 0.040 ± 0.018 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 3.007 ± 0.433 | 3.216 ± 1.174 | unsupported | 0.938 ± 0.046 | 0.543 ± 0.050 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.188 ± 0.018 | 0.192 ± 0.007 | unsupported | 0.087 ± 0.011 | 0.045 ± 0.008 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 1.996 ± 0.097 | 1.972 ± 0.486 | unsupported | 0.605 ± 0.196 | 0.243 ± 0.073 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.088 ± 0.003 | 0.088 ± 0.015 | unsupported | 0.102 ± 0.050 | 0.210 ± 0.006 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.372 ± 0.033 | 1.396 ± 0.153 | unsupported | 1.508 ± 0.062 | 3.161 ± 0.367 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.115 ± 0.013 | 0.105 ± 0.011 | unsupported | 0.115 ± 0.018 | 0.132 ± 0.010 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.946 ± 0.059 | 0.792 ± 0.247 | unsupported | 0.834 ± 0.120 | 1.236 ± 0.074 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.298 ± 0.021 | 0.290 ± 0.013 | unsupported | 0.366 ± 0.028 | 0.088 ± 0.006 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.687 ± 0.211 | 4.983 ± 0.611 | unsupported | 5.123 ± 0.476 | 1.343 ± 0.346 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.233 ± 0.029 | 0.229 ± 0.007 | unsupported | 0.234 ± 0.027 | 0.069 ± 0.015 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 2.179 ± 0.068 | 2.151 ± 0.019 | unsupported | 2.539 ± 0.143 | 0.472 ± 0.018 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.052 ± 0.004 | 0.063 ± 0.002 | unsupported | 0.052 ± 0.003 | 0.095 ± 0.022 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.627 ± 0.156 | 0.866 ± 0.232 | unsupported | 0.596 ± 0.033 | 0.933 ± 0.331 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.050 ± 0.003 | 0.070 ± 0.010 | unsupported | 0.088 ± 0.005 | 0.087 ± 0.008 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 1.129 ± 0.108 | 1.255 ± 0.077 | unsupported | 0.327 ± 0.005 | 0.456 ± 0.429 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.097 ± 0.006 | 0.112 ± 0.009 | unsupported | 0.105 ± 0.007 | 0.139 ± 0.070 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.304 ± 0.060 | 1.344 ± 0.176 | unsupported | 1.548 ± 0.066 | 1.470 ± 0.066 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.145 ± 0.093 | 0.133 ± 0.027 | unsupported | 0.113 ± 0.036 | 0.123 ± 0.013 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 1.182 ± 0.066 | 1.328 ± 0.191 | unsupported | 0.751 ± 0.029 | 0.676 ± 0.063 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.130 ± 0.005 | 0.147 ± 0.006 | unsupported | unsupported | 0.141 ± 0.012 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 2.097 ± 0.504 | 1.887 ± 0.156 | unsupported | unsupported | 2.118 ± 0.759 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.164 ± 0.074 | 0.163 ± 0.018 | unsupported | unsupported | 0.129 ± 0.001 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.462 ± 0.358 | 1.896 ± 0.262 | unsupported | unsupported | 0.891 ± 0.007 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.013 ± 0.008 | 0.016 ± 0.010 | unsupported | 0.014 ± 0.016 | 0.059 ± 0.008 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.113 ± 0.010 | 0.115 ± 0.003 | unsupported | 0.065 ± 0.004 | 0.265 ± 0.042 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.013 ± 0.001 | 0.016 ± 0.001 | unsupported | 0.039 ± 0.003 | 0.034 ± 0.002 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.151 ± 0.013 | 0.105 ± 0.080 | unsupported | 0.116 ± 0.022 | 0.242 ± 0.087 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.028 ± 0.001 | 0.036 ± 0.002 | unsupported | 0.031 ± 0.003 | 0.078 ± 0.017 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.360 ± 0.025 | 0.390 ± 0.023 | unsupported | 0.380 ± 0.347 | 0.812 ± 0.171 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.027 ± 0.003 | 0.036 ± 0.003 | unsupported | 0.127 ± 0.017 | 0.073 ± 0.005 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.603 ± 0.028 | 0.526 ± 0.061 | unsupported | 0.384 ± 0.019 | 0.378 ± 0.044 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.054 ± 0.002 | 0.054 ± 0.001 | unsupported | 0.089 ± 0.007 | 0.145 ± 0.043 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.928 ± 0.228 | 0.943 ± 0.122 | unsupported | 1.530 ± 0.382 | 0.940 ± 0.056 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.062 ± 0.052 | 0.081 ± 0.057 | unsupported | 0.081 ± 0.007 | 0.074 ± 0.011 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.557 ± 0.240 | 0.665 ± 0.016 | unsupported | 0.860 ± 0.070 | 0.423 ± 0.034 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.053 ± 0.001 | 0.055 ± 0.002 | unsupported | 0.096 ± 0.019 | 0.096 ± 0.028 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.975 ± 0.107 | 0.891 ± 0.090 | unsupported | 1.435 ± 0.135 | 1.068 ± 0.386 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.066 ± 0.057 | 0.095 ± 0.024 | unsupported | 0.081 ± 0.007 | 0.072 ± 0.024 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.686 ± 0.008 | 0.685 ± 0.004 | unsupported | 1.045 ± 0.386 | 0.488 ± 0.070 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.107 ± 0.016 | 0.107 ± 0.006 | unsupported | 0.115 ± 0.013 | 0.131 ± 0.013 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.382 ± 0.062 | 1.945 ± 1.139 | unsupported | 1.897 ± 0.259 | 1.435 ± 0.458 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.102 ± 0.013 | 0.114 ± 0.034 | unsupported | 0.089 ± 0.007 | 0.122 ± 0.028 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.083 ± 0.036 | 1.205 ± 0.071 | unsupported | 0.887 ± 0.127 | 0.687 ± 0.029 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.244 ± 0.043 | 0.257 ± 0.021 | unsupported | 0.377 ± 0.127 | 0.366 ± 0.039 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.802 ± 0.260 | 3.782 ± 0.290 | unsupported | 5.840 ± 0.716 | 5.503 ± 0.904 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.160 ± 0.026 | 0.207 ± 0.060 | unsupported | 0.206 ± 0.028 | 0.197 ± 0.024 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 2.681 ± 0.273 | 2.671 ± 0.126 | unsupported | 2.340 ± 0.323 | 2.080 ± 0.404 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.599 ± 0.070 | 0.705 ± 0.068 | unsupported | 0.142 ± 0.012 | 0.110 ± 0.011 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 11.643 ± 0.942 | 12.226 ± 1.768 | unsupported | 2.539 ± 0.279 | 1.717 ± 0.427 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.942 ± 0.572 | 1.087 ± 0.131 | unsupported | 0.145 ± 0.014 | 0.082 ± 0.020 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 8.949 ± 1.532 | 10.123 ± 1.013 | unsupported | 1.178 ± 0.054 | 0.540 ± 0.113 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 25.560 ± 8.500 | - | unsupported | 30.384 ± 17.438 | - | 21.368 ± 0.724 | 19.285 ± 8.353 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 20.629 ± 5.296 | - | unsupported | 20.152 ± 6.213 | - | 18.986 ± 1.345 | 17.745 ± 3.341 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 13.098 ± 1.296 | - | unsupported | 20.412 ± 7.472 | - | 16.453 ± 1.277 | 18.747 ± 8.487 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 13.031 ± 0.944 | - | unsupported | 10.302 ± 0.852 | - | 17.177 ± 3.763 | 13.680 ± 3.691 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 14.293 ± 2.695 | - | unsupported | 19.586 ± 2.785 | - | 14.540 ± 1.915 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 13.820 ± 2.494 | - | unsupported | 10.100 ± 1.067 | - | 14.573 ± 0.926 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 24.209 ± 2.218 | - | unsupported | 23.782 ± 4.941 | - | 21.605 ± 2.366 | 24.925 ± 2.782 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 19.593 ± 0.843 | - | unsupported | 16.214 ± 1.700 | - | 19.332 ± 1.637 | 23.496 ± 1.532 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 66.094 ± 10.550 | - | unsupported | 13.708 ± 3.735 | - | 65.447 ± 14.362 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 31.616 ± 7.231 | - | unsupported | 10.686 ± 0.484 | - | 44.324 ± 20.465 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 71.798 ± 17.977 | - | unsupported | 19.663 ± 5.150 | - | 55.652 ± 7.106 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 35.928 ± 2.833 | - | unsupported | 10.587 ± 0.502 | - | 25.179 ± 8.210 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 28.156 ± 10.562 | - | unsupported | 30.373 ± 11.855 | - | 23.659 ± 3.069 | 21.456 ± 3.516 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 22.576 ± 4.348 | - | unsupported | 17.347 ± 2.299 | - | 16.808 ± 0.922 | 16.619 ± 1.508 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 14.686 ± 10.720 | - | unsupported | 18.070 ± 3.888 | - | 13.261 ± 3.458 | 15.036 ± 2.051 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 12.525 ± 0.795 | - | unsupported | 11.033 ± 2.333 | - | 13.608 ± 2.235 | 16.543 ± 4.831 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.619 ± 0.098 | unsupported | unsupported | 0.326 ± 0.139 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 12.494 ± 2.107 | unsupported | unsupported | 4.167 ± 0.504 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 1.122 ± 0.336 | unsupported | unsupported | 0.311 ± 0.036 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 10.435 ± 1.294 | unsupported | unsupported | 3.166 ± 0.137 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 22.936 ± 6.172 | - | unsupported | 25.366 ± 8.397 | - | 19.300 ± 3.178 | 26.366 ± 3.714 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 22.293 ± 4.122 | - | unsupported | 15.959 ± 4.595 | - | 19.404 ± 1.078 | 16.052 ± 0.482 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 11.749 ± 2.297 | - | 0.008 ± 0.005 | 8.257 ± 2.688 | 6.390 ± 0.834 | 15.575 ± 0.690 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 12.551 ± 4.272 | - | 0.002 ± 0.000 | 5.155 ± 0.593 | 5.457 ± 1.665 | 16.895 ± 6.266 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 10.740 ± 0.677 | - | 10.702 ± 2.307 | 11.715 ± 8.786 | 9.798 ± 0.714 | 10.079 ± 0.822 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 11.164 ± 1.803 | - | 12.882 ± 2.330 | 10.587 ± 1.941 | 8.111 ± 0.215 | 14.536 ± 9.145 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 12.831 ± 1.416 | - | 12.682 ± 1.037 | 13.366 ± 0.646 | 18.596 ± 4.751 | 12.289 ± 4.618 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 15.817 ± 5.350 | - | 11.852 ± 0.625 | 9.963 ± 0.556 | 10.181 ± 0.626 | 16.244 ± 1.844 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 8.152 ± 2.898 | - | 8.041 ± 1.628 | 17.292 ± 4.572 | 10.660 ± 1.211 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 7.013 ± 1.288 | - | 5.624 ± 0.971 | 8.082 ± 0.408 | 8.951 ± 0.911 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 13.163 ± 2.244 | - | 0.002 ± 0.000 | 15.849 ± 5.223 | 15.922 ± 5.494 | 14.105 ± 1.966 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 16.314 ± 2.840 | - | 0.002 ± 0.003 | 11.514 ± 1.628 | 10.663 ± 0.748 | 19.789 ± 9.383 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 42.908 ± 8.759 | - | 0.002 ± 0.000 | 60.142 ± 12.405 | 64.697 ± 9.119 | 230.093 ± 25.842 | 39.050 ± 4.719 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 19.828 ± 8.722 | - | 0.002 ± 0.001 | 46.213 ± 3.223 | 28.155 ± 0.918 | 239.477 ± 39.549 | 22.434 ± 3.941 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 9.099 ± 0.842 | - | 8.900 ± 0.401 | 9.392 ± 1.194 | 6.799 ± 1.337 | 8.592 ± 1.868 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 6.675 ± 1.342 | - | 6.279 ± 0.458 | 5.734 ± 0.411 | 6.029 ± 1.246 | 10.108 ± 1.409 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 8.796 ± 1.064 | - | 10.671 ± 2.127 | 10.974 ± 4.039 | 6.515 ± 0.453 | 10.182 ± 1.198 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 5.777 ± 0.356 | - | 6.210 ± 0.985 | 5.972 ± 0.526 | 6.085 ± 0.607 | 10.793 ± 11.317 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.003 ± 0.013 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.003 ± 0.001 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 3.999208 | 249950.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 9.421958 | 588872.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 8.568667 | 535541.69 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 3.059292 | 191205.75 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 4.275625 | 267226.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.449291 | 278080.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 17.790334 | 1111895.88 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 19.941500 | 1246343.75 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 17.782584 | 1111411.50 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 12.285458 | 767841.12 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.352625 | 397039.06 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 6.566708 | 410419.25 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 5.956084 | 372255.25 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 6.198250 | 387390.62 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 77.847209 | 4865450.56 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 77.408792 | 4838049.50 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 161.627750 | 10101734.38 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 136.692416 | 8543276.00 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.214750 | 388421.88 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.031417 | 314463.56 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 12.701292 | 793830.75 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 11.787292 | 736705.75 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 64.615000 | 4038437.50 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 61.120333 | 3820020.81 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 143.179291 | 8948705.69 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 133.019709 | 8313731.81 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 20.443750 | 1277734.38 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 21.514166 | 1344635.38 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 536.524959 | 33532809.94 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 722.650500 | 45165656.25 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 17.752584 | 1109536.50 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 17.768875 | 1110554.69 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 362.665292 | 22666580.75 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 452.206542 | 28262908.88 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 20.766500 | 1297906.25 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 23.017458 | 1438591.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 601.613917 | 37600869.81 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 519.769875 | 32485617.19 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 17.348000 | 1084250.00 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 18.248625 | 1140539.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 334.349042 | 20896815.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 353.771250 | 22110703.12 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.532042 | 39563.16 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.178334 | 65286.47 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.413208 | 193956.38 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.512542 | 195508.47 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.347709 | 543463.62 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.506750 | 938343.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 24.059667 | 3007458.38 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 25.725417 | 3215677.12 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.901250 | 45332.03 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.564166 | 86940.09 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 12.039958 | 188124.34 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 12.283334 | 191927.09 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.944417 | 243052.12 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.838833 | 604854.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 15.965959 | 1995744.88 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.772833 | 1971604.12 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.414167 | 209596.36 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.548375 | 102318.36 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.630167 | 87971.36 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.611708 | 87682.94 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 25.284042 | 3160505.25 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.067083 | 1508385.38 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.974667 | 1371833.38 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.167083 | 1395885.38 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.461125 | 132205.08 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.376833 | 115263.02 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.356542 | 114945.97 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.722917 | 105045.58 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.887958 | 1235994.75 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.668167 | 833520.88 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.565250 | 945656.25 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.337750 | 792218.75 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.632041 | 88000.64 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 23.393250 | 365519.53 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 19.075792 | 298059.25 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 18.542625 | 289728.52 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.745584 | 1343198.00 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 40.986208 | 5123276.00 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 37.495458 | 4686932.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 39.864000 | 4983000.00 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.447708 | 69495.44 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 14.962292 | 233785.81 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 14.903833 | 232872.39 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 14.637917 | 228717.45 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.772250 | 471531.25 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 20.311750 | 2538968.75 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 17.435084 | 2179385.50 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 17.206292 | 2150786.50 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.056834 | 94638.03 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.359250 | 52488.28 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.339583 | 52180.98 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.026625 | 62916.02 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.464958 | 933119.75 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.771542 | 596442.75 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.016875 | 627109.38 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.930417 | 866302.12 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.571750 | 87058.59 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.619292 | 87801.44 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.207709 | 50120.45 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.460250 | 69691.41 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.649291 | 456161.38 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.613125 | 326640.62 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.035625 | 1129453.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.038791 | 1254848.88 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.893959 | 138968.11 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.696292 | 104629.56 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.190833 | 96731.77 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.185750 | 112277.34 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.761458 | 1470182.25 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.387125 | 1548390.62 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.432084 | 1304010.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.750375 | 1343796.88 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.869791 | 122965.48 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.216375 | 112755.86 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.284875 | 145076.17 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.496708 | 132761.06 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.408000 | 676000.00 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.011583 | 751447.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.458125 | 1182265.62 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.625709 | 1328213.62 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.993000 | 140515.62 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.326417 | 130100.27 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.411541 | 147055.33 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 16.947750 | 2118468.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 16.773291 | 2096661.38 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 15.097500 | 1887187.50 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.277625 | 129337.89 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.485458 | 163835.28 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.442583 | 163165.36 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.131209 | 891401.12 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 11.697208 | 1462151.00 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.167875 | 1895984.38 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.761542 | 58774.09 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.912709 | 14261.08 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.825333 | 12895.83 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.029667 | 16088.55 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.122292 | 265286.50 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.518833 | 64854.12 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.901459 | 112682.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.920875 | 115109.38 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.197375 | 34333.98 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.491083 | 38923.17 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.811625 | 12681.64 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.035166 | 16174.47 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.934334 | 241791.75 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.925667 | 115708.38 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.207917 | 150989.62 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.842958 | 105369.75 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.988333 | 77942.70 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.011000 | 31421.88 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.777917 | 27779.95 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.297458 | 35897.78 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.499458 | 812432.25 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.038666 | 379833.25 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.879541 | 359942.62 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.120208 | 390026.00 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.699458 | 73429.03 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.135250 | 127113.28 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.741708 | 27214.19 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.333875 | 36466.80 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.027000 | 378375.00 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.074958 | 384369.75 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.823000 | 602875.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.206541 | 525817.62 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.309042 | 145453.78 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.716500 | 89320.31 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.446875 | 53857.42 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.434208 | 53659.50 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.520291 | 940036.38 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.243500 | 1530437.50 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.423958 | 927994.75 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.540417 | 942552.12 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.716042 | 73688.16 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.196750 | 81199.22 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.941083 | 61579.42 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.200416 | 81256.50 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.385333 | 423166.62 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.877583 | 859697.88 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.454417 | 556802.12 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.317958 | 664744.75 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.137584 | 95899.75 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.116834 | 95575.53 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.394791 | 53043.61 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.526084 | 55095.06 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.546291 | 1068286.38 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.480167 | 1435020.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.799667 | 974958.38 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.124375 | 890546.88 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.610750 | 72042.97 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.178833 | 80919.27 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.218958 | 65921.22 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.055167 | 94611.98 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.906667 | 488333.38 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.360417 | 1045052.12 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.488584 | 686073.00 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.482625 | 685328.12 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.404625 | 131322.27 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.375000 | 115234.38 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.865166 | 107268.22 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.837500 | 106835.94 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.476125 | 1434515.62 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 15.172333 | 1896541.62 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.058000 | 1382250.00 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 15.559584 | 1944948.00 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.777250 | 121519.53 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.691292 | 88926.44 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.517750 | 101839.84 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.280333 | 113755.20 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.494042 | 686755.25 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.093917 | 886739.62 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.663667 | 1082958.38 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.642125 | 1205265.62 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 23.442292 | 366285.81 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 24.138791 | 377168.61 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 15.613500 | 243960.94 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 16.478542 | 257477.22 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 44.026042 | 5503255.25 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 46.719000 | 5839875.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 30.413625 | 3801703.12 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 30.254750 | 3781843.75 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.618625 | 197166.02 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 13.178875 | 205919.92 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.230459 | 159850.92 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 13.219458 | 206554.03 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.642834 | 2080354.25 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 18.722792 | 2340349.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 21.449291 | 2681161.38 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 21.371500 | 2671437.50 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.060083 | 110313.80 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.083500 | 141929.69 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 38.321792 | 598778.00 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 45.095250 | 704613.28 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.733291 | 1716661.38 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 20.315917 | 2539489.62 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 93.147875 | 11643484.38 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 97.804041 | 12225505.12 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.229125 | 81705.08 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.266042 | 144781.91 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 60.272583 | 941759.11 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 69.578500 | 1087164.06 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.316500 | 539562.50 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.421459 | 1177682.38 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 71.592208 | 8949026.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 80.982083 | 10122760.38 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 5.219084 | 326192.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 9.911334 | 619458.38 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 66.672583 | 4167036.44 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 199.899041 | 12493690.06 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 4.976833 | 311052.06 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 17.957625 | 1122351.56 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 50.650625 | 3165664.06 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 166.961750 | 10435109.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.014083 | 880.19 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001375 | 85.94 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.012667 | 791.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001458 | 91.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.011750 | 734.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.047833 | 2989.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.015042 | 940.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.043583 | 2723.94 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000250 | 15.62 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.012458 | 778.62 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.004416 | 276.00 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.011250 | 703.12 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.003000 | 187.50 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.011125 | 695.31 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003209 | 200.56 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.010584 | 661.50 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002792 | 174.50 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 36.1x faster than the slowest successful cell (`julia-base`, 72.519 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 19.1x faster than the slowest successful cell (`julia-base`, 72.519 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 13.1x faster than the slowest successful cell (`julia-base`, 72.519 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`julia-base`, 72.519 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 22.7x faster than the slowest successful cell (`julia-base`, 472.604 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 43.0x faster than the slowest successful cell (`julia-base`, 472.604 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 39.5x faster than the slowest successful cell (`julia-base`, 472.604 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 39.0x faster than the slowest successful cell (`julia-base`, 472.604 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 25.1x faster than the slowest successful cell (`julia-base`, 177.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 25.5x faster than the slowest successful cell (`julia-base`, 177.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 22.8x faster than the slowest successful cell (`julia-base`, 177.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 22.0x faster than the slowest successful cell (`julia-base`, 177.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 57.6x faster than the slowest successful cell (`julia-base`, 134.637 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 24.3x faster than the slowest successful cell (`julia-base`, 134.637 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 24.4x faster than the slowest successful cell (`julia-base`, 134.637 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 19.1x faster than the slowest successful cell (`julia-base`, 134.637 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 103.8x faster than the slowest successful cell (`julia-base`, 608.337 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 91.8x faster than the slowest successful cell (`julia-base`, 608.337 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 75.1x faster than the slowest successful cell (`julia-base`, 608.337 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 63.4x faster than the slowest successful cell (`julia-base`, 608.337 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 21.7x faster than the slowest successful cell (`pytorch-cpu`, 19.837 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 19.837 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 19.837 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 12.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 12.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 12.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 21.6x faster than the slowest successful cell (`pytorch-cpu`, 20.471 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 20.471 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 20.471 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 12.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 12.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 12.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 10.6x faster than the slowest successful cell (`julia-base`, 2.568 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 423.6x faster than the slowest successful cell (`julia-base`, 1.571 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 1327.9x faster than the slowest successful cell (`julia-base`, 1.826 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 17.9x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 44.9x faster than the slowest successful cell (`pytorch-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 21.0x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 43.9x faster than the slowest successful cell (`pytorch-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 23.4x faster than the slowest successful cell (`pytorch-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 42.5x faster than the slowest successful cell (`pytorch-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 21.2x faster than the slowest successful cell (`pytorch-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 46.8x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 20.4x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 36.9x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 20.5x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 21.4x faster than the slowest successful cell (`pytorch-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 37.2x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 33.0x faster than the slowest successful cell (`pytorch-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 20.5x faster than the slowest successful cell (`pytorch-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 20.9x faster than the slowest successful cell (`pytorch-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 20.7x faster than the slowest successful cell (`jax-cpu`, 0.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`3x2x2`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`4x4x4`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 18.7x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 13.0x faster than the slowest successful cell (`jax-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 26.0x faster than the slowest successful cell (`jax-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 24.3x faster than the slowest successful cell (`jax-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 27.1x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 17.3x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 34.6x faster than the slowest successful cell (`pytorch-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 22.5x faster than the slowest successful cell (`pytorch-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x64x24`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `jax-cpu` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 50.5x faster than the slowest successful cell (`pytorch-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 35.4x faster than the slowest successful cell (`pytorch-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 54.2x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 36.6x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 45.3x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 31.7x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 20.8x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 21.7x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 36.5x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 25.7x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 20.0x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 44.0x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 24.0x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `jax-cpu` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 42.9x faster than the slowest successful cell (`pytorch-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 31.5x faster than the slowest successful cell (`pytorch-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x8x8`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 25.2x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.021 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x32x32,rhs=1`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`3x2x2,rhs=1`): `tenferro-direct` is 18.8x faster than the slowest successful cell (`jax-cpu`, 0.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`3x8x8,rhs=1`): `tenferro-direct` is 13.9x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`8x4x4,rhs=1`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 21.0x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 19.6x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_lu_solve` (f64, threads=1, shape=`1024x4x4,rhs=1`): `pytorch-cpu` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.788 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 26.2x faster than the slowest successful cell (`julia-base`, 403.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 24.5x faster than the slowest successful cell (`julia-base`, 403.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 13.6x faster than the slowest successful cell (`julia-base`, 403.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`julia-base`, 403.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 30.3x faster than the slowest successful cell (`julia-base`, 237.206 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 33.6x faster than the slowest successful cell (`julia-base`, 237.206 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 35.4x faster than the slowest successful cell (`julia-base`, 329.588 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 47.4x faster than the slowest successful cell (`julia-base`, 329.588 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 38.0x faster than the slowest successful cell (`julia-base`, 329.588 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 37.7x faster than the slowest successful cell (`julia-base`, 329.588 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 38.0x faster than the slowest successful cell (`julia-base`, 274.712 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 33.7x faster than the slowest successful cell (`julia-base`, 274.712 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 23.9x faster than the slowest successful cell (`julia-base`, 274.712 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 23.3x faster than the slowest successful cell (`julia-base`, 274.712 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 23.5x faster than the slowest successful cell (`julia-base`, 645.781 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 29.2x faster than the slowest successful cell (`julia-base`, 645.781 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 32.5x faster than the slowest successful cell (`julia-base`, 645.781 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 32.5x faster than the slowest successful cell (`julia-base`, 645.781 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 40.4x faster than the slowest successful cell (`julia-base`, 369.340 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 42.5x faster than the slowest successful cell (`julia-base`, 369.340 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`julia-base`, 369.340 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 27.2x faster than the slowest successful cell (`julia-base`, 131.309 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 21.6x faster than the slowest successful cell (`julia-base`, 238.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 18.3x faster than the slowest successful cell (`julia-base`, 238.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 80.8x faster than the slowest successful cell (`julia-base`, 984.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 90.4x faster than the slowest successful cell (`julia-base`, 984.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 42.9x faster than the slowest successful cell (`julia-base`, 984.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 31.2x faster than the slowest successful cell (`julia-base`, 984.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 81.5x faster than the slowest successful cell (`julia-base`, 999.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 96.5x faster than the slowest successful cell (`julia-base`, 999.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 55.6x faster than the slowest successful cell (`julia-base`, 999.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 46.5x faster than the slowest successful cell (`julia-base`, 999.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 44.3x faster than the slowest successful cell (`julia-base`, 360.970 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 49.3x faster than the slowest successful cell (`julia-base`, 360.970 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`julia-base`, 360.970 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`julia-base`, 360.970 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 48.6x faster than the slowest successful cell (`julia-base`, 1676.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 46.4x faster than the slowest successful cell (`julia-base`, 1676.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 26.2x faster than the slowest successful cell (`julia-base`, 1676.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 23.2x faster than the slowest successful cell (`julia-base`, 1676.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 13.3x faster than the slowest successful cell (`tenferro-eager`, 1.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 18.8x faster than the slowest successful cell (`tenferro-eager`, 10.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 2042.6x faster than the slowest successful cell (`julia-base`, 15.575 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 9891.4x faster than the slowest successful cell (`julia-base`, 16.895 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 8686.1x faster than the slowest successful cell (`jax-cpu`, 15.922 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 10554.2x faster than the slowest successful cell (`julia-base`, 19.789 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 128400.2x faster than the slowest successful cell (`julia-base`, 230.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `strided-jl` is 10.7x faster than the slowest successful cell (`julia-base`, 239.477 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 12.1x faster than the slowest successful cell (`julia-base`, 239.477 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 140209.3x faster than the slowest successful cell (`julia-base`, 239.477 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 598.0x faster than the slowest successful cell (`tenferro-direct`, 0.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 340.5x faster than the slowest successful cell (`tenferro-direct`, 0.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 48.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 87.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 139.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 132.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 4521.2 GB/s at 0.004 ms. The conservative physical lower bound is 0.168 ms, so the cell is 45.2x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 12195.7 GB/s at 0.001 ms. The conservative physical lower bound is 0.042 ms, so the cell is 30.5x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 35204.6 GB/s at 0.008 ms. The conservative physical lower bound is 2.684 ms, so the cell is 352.0x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 157163.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 392.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 146446.0 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1464.5x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 143165.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 357.9x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 74898.3 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 749.0x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 78581.8 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 196.5x faster than the bound.

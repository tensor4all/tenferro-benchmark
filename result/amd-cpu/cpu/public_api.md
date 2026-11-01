# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261101_185513/run.yaml`
- Timestamp: `2026-11-01T18:55:13.568870+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261101_185513`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

## CPU Information

- Model: `Intel(R) Xeon(R) 6973P-C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`

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

- CSV: `data/results/amd-cpu/cpu/public_api/20261101_185513/cpu_public_api_t1_20261101_185513.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261101_185513/cpu_public_api_t4_20261101_185513.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 1.808 ± 0.062 | - | 2.115 ± 0.066 | 2.457 ± 0.086 | 3.320 ± 0.044 | 3.633 ± 0.103 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 1.393 ± 0.046 | - | 3.000 ± 1.002 | 1.424 ± 0.015 | 2.984 ± 0.534 | 2.379 ± 2.127 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 95.583 ± 1.021 | - | 94.919 ± 0.264 | 97.291 ± 0.657 | 97.006 ± 6.446 | 103.441 ± 2.333 | 114.037 ± 0.703 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 38.878 ± 0.615 | - | 37.358 ± 2.353 | 37.038 ± 2.817 | 52.337 ± 1.201 | 96.853 ± 3.316 | 43.894 ± 1.251 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 122.226 ± 2.319 | - | 125.432 ± 2.582 | 134.501 ± 0.704 | 139.404 ± 2.746 | 128.436 ± 0.255 | 130.976 ± 0.944 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 31.748 ± 0.438 | - | 32.107 ± 1.529 | 28.518 ± 0.552 | 70.260 ± 0.911 | 128.667 ± 12.811 | 38.513 ± 0.808 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 24.499 ± 6.230 | - | 19.278 ± 0.189 | 20.489 ± 0.068 | 65.475 ± 0.968 | 25.863 ± 0.504 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 12.576 ± 0.975 | - | 11.414 ± 0.244 | 11.639 ± 0.209 | 32.710 ± 0.339 | 12.203 ± 0.558 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 20.101 ± 0.137 | - | unsupported | 20.985 ± 0.803 | 72.224 ± 0.734 | 28.794 ± 15.587 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 11.800 ± 0.404 | - | unsupported | 11.845 ± 0.295 | 33.216 ± 1.257 | 11.990 ± 1.637 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 8.551 ± 0.117 | - | 8.608 ± 0.133 | 8.444 ± 0.405 | 9.125 ± 0.307 | 9.081 ± 0.151 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 10.903 ± 0.153 | - | 17.036 ± 1.655 | 8.619 ± 0.256 | 11.723 ± 0.080 | 11.979 ± 0.245 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 48.842 ± 6.568 | - | 45.084 ± 5.172 | 89.317 ± 0.646 | 80.214 ± 8.295 | 63.064 ± 0.240 | 70.002 ± 9.752 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 22.555 ± 3.503 | - | 22.313 ± 0.350 | 33.908 ± 1.523 | 39.404 ± 1.247 | 54.741 ± 1.760 | 25.182 ± 4.834 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 85.565 ± 13.341 | - | 73.197 ± 2.565 | 110.689 ± 3.349 | 157.954 ± 0.329 | 103.068 ± 1.571 | 104.422 ± 2.134 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 30.667 ± 0.764 | - | 31.179 ± 1.408 | 43.180 ± 0.658 | 58.292 ± 1.397 | 83.882 ± 0.819 | 37.503 ± 0.460 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 58.504 ± 0.757 | - | 58.388 ± 0.380 | 59.395 ± 0.717 | 59.512 ± 0.592 | 61.117 ± 0.380 | 62.591 ± 0.135 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 22.666 ± 0.323 | - | 22.307 ± 0.253 | 22.827 ± 0.172 | 31.455 ± 1.196 | 54.084 ± 1.501 | 24.926 ± 0.653 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.834 ± 0.101 | - | 26.354 ± 2.077 | 25.017 ± 0.968 | 13.025 ± 0.723 | 2.454 ± 1.512 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.894 ± 0.019 | - | 5.719 ± 0.476 | 7.269 ± 0.232 | 3.430 ± 1.106 | 2.364 ± 0.137 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 3.136 ± 0.032 | - | 3.347 ± 0.041 | 3.978 ± 0.107 | 5.145 ± 0.046 | 5.852 ± 0.069 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 3.075 ± 0.073 | - | 8.594 ± 5.501 | 2.898 ± 0.072 | 5.707 ± 0.196 | 4.312 ± 0.642 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 2.225 ± 0.081 | - | 2.465 ± 0.043 | 2.658 ± 0.114 | 3.939 ± 0.042 | 4.082 ± 0.126 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 2.056 ± 0.073 | - | 7.197 ± 0.214 | 1.924 ± 0.105 | 2.814 ± 0.038 | 3.566 ± 0.017 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 2.635 ± 0.047 | - | 2.697 ± 0.090 | 3.184 ± 0.044 | 4.366 ± 0.050 | 4.555 ± 0.503 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.495 ± 0.049 | - | 12.040 ± 0.303 | 3.258 ± 0.056 | 5.436 ± 0.153 | 5.405 ± 0.028 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 19.304 ± 0.748 | - | 19.273 ± 0.485 | 20.116 ± 1.016 | 64.740 ± 2.108 | 25.679 ± 1.233 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 11.326 ± 1.352 | - | 13.614 ± 4.923 | 11.645 ± 0.104 | 32.307 ± 0.321 | 11.209 ± 0.072 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 22.467 ± 0.182 | - | unsupported | 27.092 ± 3.910 | 54.626 ± 1.299 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 11.711 ± 0.089 | - | unsupported | 11.409 ± 1.432 | 27.116 ± 0.499 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 96.307 ± 6.940 | - | 94.911 ± 0.331 | 98.583 ± 1.436 | 103.869 ± 6.835 | 102.671 ± 2.326 | 120.457 ± 0.804 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 36.316 ± 0.255 | - | 38.365 ± 0.911 | 35.956 ± 0.446 | 56.628 ± 1.598 | 97.013 ± 0.517 | 47.815 ± 0.877 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 111.867 ± 0.530 | - | 118.439 ± 1.597 | 112.810 ± 0.525 | 136.874 ± 1.453 | 115.375 ± 4.371 | 131.019 ± 1.657 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 46.650 ± 2.686 | - | 43.764 ± 0.223 | 44.033 ± 6.599 | 70.009 ± 1.340 | 111.930 ± 2.223 | 53.674 ± 3.093 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 39.033 ± 1.212 | - | 50.023 ± 0.968 | 52.881 ± 4.101 | 47.322 ± 0.656 | 48.513 ± 5.523 | 52.480 ± 2.510 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 7.874 ± 0.089 | - | 20.368 ± 0.406 | 21.726 ± 0.790 | 25.775 ± 0.602 | 48.396 ± 5.303 | 20.960 ± 10.557 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 15.686 ± 0.288 | - | 15.796 ± 0.435 | 32.786 ± 0.137 | 36.282 ± 0.482 | 30.884 ± 3.298 | 34.220 ± 5.171 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.766 ± 0.284 | - | 8.266 ± 0.559 | 12.114 ± 0.521 | 24.044 ± 4.583 | 30.537 ± 4.039 | 13.060 ± 6.521 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 33.350 ± 0.212 | - | 33.899 ± 0.661 | 53.495 ± 1.752 | 44.359 ± 0.667 | 32.459 ± 0.169 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 16.883 ± 0.344 | - | 17.217 ± 0.276 | 20.369 ± 0.678 | 23.076 ± 0.311 | 29.494 ± 0.562 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 16.484 ± 0.597 | - | 18.100 ± 0.753 | 30.703 ± 0.396 | 67.406 ± 1.049 | 55.466 ± 6.067 | 55.238 ± 5.193 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 5.983 ± 0.077 | - | 6.113 ± 0.094 | 10.838 ± 0.192 | 34.277 ± 1.113 | 56.186 ± 12.660 | 23.017 ± 7.272 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 126.583 ± 7.869 | - | 112.989 ± 8.154 | 116.631 ± 1.673 | 146.445 ± 0.922 | 115.856 ± 0.437 | 132.592 ± 0.530 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 43.628 ± 2.015 | - | 47.676 ± 19.104 | 44.293 ± 1.266 | 79.071 ± 0.692 | 113.269 ± 2.513 | 51.850 ± 1.026 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 14.139 ± 1.437 | - | 14.013 ± 1.591 | 28.654 ± 0.370 | 50.022 ± 0.788 | 55.604 ± 11.696 | 53.147 ± 5.253 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 4.350 ± 0.099 | - | 4.272 ± 0.122 | 10.149 ± 0.261 | 28.124 ± 1.039 | 49.381 ± 4.965 | 22.158 ± 7.821 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 26.021 ± 3.229 | - | 25.678 ± 0.294 | 26.762 ± 0.426 | 28.580 ± 0.461 | 26.471 ± 5.717 | 28.822 ± 3.260 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 4.461 ± 0.060 | - | 4.509 ± 0.455 | 7.696 ± 0.164 | 12.799 ± 1.002 | 27.408 ± 5.625 | 11.116 ± 3.773 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 43.873 ± 2.180 | - | 44.287 ± 6.527 | 30.623 ± 0.314 | 64.598 ± 0.521 | 61.990 ± 14.318 | 65.545 ± 3.986 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 6.593 ± 0.089 | - | 6.516 ± 0.169 | 10.676 ± 0.230 | 33.025 ± 1.306 | 70.082 ± 7.520 | 32.145 ± 12.393 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 21.581 ± 1.494 | - | 22.545 ± 2.999 | 19.876 ± 0.960 | 39.372 ± 0.549 | 31.931 ± 4.918 | 33.307 ± 3.790 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 2.879 ± 0.135 | - | 3.241 ± 0.021 | 7.348 ± 0.357 | 22.656 ± 1.347 | 29.636 ± 4.784 | 12.766 ± 4.604 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.274 ± 0.035 | 0.300 ± 0.048 | unsupported | unsupported | 0.423 ± 0.007 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.236 ± 0.028 | 0.254 ± 0.056 | unsupported | unsupported | 0.354 ± 0.109 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 106.981 ± 0.398 | - | 107.409 ± 0.507 | 115.024 ± 0.834 | 134.621 ± 0.711 | 122.008 ± 0.891 | 136.110 ± 1.056 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 49.320 ± 2.332 | - | 42.532 ± 0.234 | 42.923 ± 0.258 | 82.570 ± 10.621 | 114.508 ± 0.350 | 51.651 ± 0.364 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 107.306 ± 0.366 | - | 108.603 ± 12.684 | 114.343 ± 1.302 | 134.399 ± 1.744 | 136.249 ± 17.973 | 134.423 ± 2.302 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 43.170 ± 1.924 | - | 42.447 ± 0.361 | 42.896 ± 0.600 | 70.483 ± 0.283 | 115.820 ± 5.142 | 51.701 ± 1.806 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 111.950 ± 0.518 | - | 112.355 ± 2.851 | 114.275 ± 0.819 | 131.016 ± 0.317 | 116.351 ± 0.351 | 129.956 ± 0.948 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 45.784 ± 8.389 | - | 48.079 ± 5.501 | 42.281 ± 0.323 | 69.904 ± 0.435 | 119.061 ± 6.454 | 51.604 ± 0.814 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 95.608 ± 7.197 | - | 95.664 ± 1.535 | 97.991 ± 2.431 | 102.418 ± 0.598 | 104.887 ± 0.266 | 117.228 ± 0.870 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 38.339 ± 2.250 | - | 36.552 ± 0.411 | 37.641 ± 4.003 | 59.029 ± 2.837 | 99.355 ± 0.437 | 44.874 ± 2.019 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 49.473 ± 0.127 | - | 50.045 ± 0.606 | 30.985 ± 0.129 | 59.479 ± 1.048 | 96.367 ± 3.476 | 77.050 ± 3.460 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 23.656 ± 1.421 | - | 22.250 ± 1.128 | 11.497 ± 0.074 | 31.100 ± 0.380 | 104.793 ± 14.393 | 35.405 ± 6.366 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 32.135 ± 2.330 | - | 30.669 ± 2.350 | 17.587 ± 1.470 | 77.285 ± 1.394 | 28.481 ± 0.559 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 8.914 ± 0.200 | - | 7.665 ± 0.663 | 6.250 ± 0.172 | 40.412 ± 0.270 | 27.873 ± 0.752 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.868 ± 0.149 | - | 1.760 ± 0.155 | 27.916 ± 0.662 | 1.550 ± 0.046 | 1.409 ± 0.071 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.737 ± 0.021 | - | 0.775 ± 0.092 | 12.210 ± 0.438 | 0.946 ± 0.218 | 1.484 ± 0.050 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.175 ± 0.032 | - | 1.174 ± 0.024 | 2.035 ± 0.064 | 7.544 ± 4.908 | 1.185 ± 0.053 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.924 ± 0.115 | - | 0.742 ± 0.071 | 0.921 ± 0.026 | 3.536 ± 1.334 | 1.229 ± 0.066 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 27.528 ± 1.795 | - | 27.672 ± 1.758 | 19.569 ± 3.365 | 78.745 ± 6.212 | 24.009 ± 0.274 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 9.085 ± 0.818 | - | 8.418 ± 0.093 | 6.080 ± 0.042 | 40.544 ± 0.626 | 22.575 ± 0.162 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.714 ± 0.104 | - | 1.792 ± 0.101 | 26.741 ± 0.258 | 1.426 ± 0.028 | 1.416 ± 0.056 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.968 ± 0.045 | - | 0.821 ± 0.010 | 12.478 ± 0.828 | 0.927 ± 0.596 | 1.553 ± 0.327 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 10.573 ± 0.556 | - | 11.386 ± 0.452 | 23.470 ± 0.820 | 39.171 ± 0.983 | 9.284 ± 0.864 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 4.141 ± 0.285 | - | 3.899 ± 0.715 | 5.169 ± 0.096 | 20.050 ± 0.214 | 9.221 ± 0.522 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 16.486 ± 0.539 | - | 17.410 ± 0.276 | 18.695 ± 0.543 | 30.850 ± 0.473 | 19.282 ± 1.596 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 8.002 ± 0.299 | - | 8.126 ± 0.410 | 8.320 ± 0.875 | 15.461 ± 0.087 | 16.428 ± 0.795 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.141 ± 0.010 | - | 1.149 ± 0.001 | 1.790 ± 0.015 | 1.277 ± 0.020 | 1.201 ± 0.026 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.593 ± 0.017 | - | 0.602 ± 0.021 | 0.814 ± 0.017 | 1.311 ± 0.301 | 1.144 ± 0.017 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.190 ± 0.029 | - | 1.194 ± 0.007 | 1.169 ± 0.012 | 1.280 ± 0.084 | 1.176 ± 0.065 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.597 ± 0.017 | - | 0.612 ± 0.020 | 0.607 ± 0.020 | 1.306 ± 0.559 | 1.135 ± 0.012 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 17.521 ± 0.828 | - | 17.094 ± 0.665 | 18.094 ± 0.526 | 21.386 ± 0.282 | 19.213 ± 2.479 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 8.113 ± 0.302 | - | 8.260 ± 0.398 | 7.409 ± 0.417 | 11.105 ± 0.343 | 16.583 ± 0.652 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.141 ± 0.016 | - | 1.152 ± 0.032 | 1.791 ± 0.028 | 1.260 ± 0.016 | 1.142 ± 0.012 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.549 ± 0.018 | - | 0.603 ± 0.018 | 0.824 ± 0.008 | 1.346 ± 0.633 | 1.135 ± 0.017 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.152 ± 0.013 | - | 1.159 ± 0.008 | 1.164 ± 0.002 | 1.286 ± 0.034 | 1.145 ± 0.014 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.560 ± 0.019 | - | 0.616 ± 0.011 | 0.605 ± 0.018 | 1.340 ± 0.555 | 1.135 ± 0.009 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.153 ± 0.003 | 1.150 ± 0.010 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.589 ± 0.004 | 0.599 ± 0.004 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 32.038 ± 0.146 | - | 31.634 ± 0.415 | 64.261 ± 2.490 | 58.017 ± 0.968 | 65.610 ± 9.315 | 63.897 ± 3.911 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 10.995 ± 0.104 | - | 10.855 ± 0.338 | 18.540 ± 2.046 | 39.986 ± 4.690 | 61.154 ± 2.919 | 22.043 ± 5.674 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 132.789 ± 1.057 | - | 132.814 ± 0.462 | 132.261 ± 15.233 | 214.078 ± 4.951 | 192.759 ± 0.470 | 192.767 ± 3.087 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 43.373 ± 0.865 | - | 43.680 ± 0.514 | 39.650 ± 0.333 | 144.891 ± 2.837 | 182.933 ± 3.557 | 76.195 ± 4.788 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 120.215 ± 9.373 | - | 110.940 ± 5.438 | 126.050 ± 0.713 | 126.599 ± 4.720 | 122.685 ± 3.058 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 45.597 ± 0.427 | - | 45.191 ± 2.937 | 44.959 ± 0.570 | 65.722 ± 0.449 | 115.354 ± 0.771 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 97.961 ± 0.589 | - | 97.483 ± 0.824 | 104.164 ± 1.224 | 105.121 ± 0.914 | 107.516 ± 0.602 | 122.448 ± 8.670 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 37.416 ± 0.231 | - | 36.983 ± 0.852 | 37.695 ± 1.068 | 60.295 ± 15.531 | 101.963 ± 0.593 | 45.760 ± 0.837 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 20.942 ± 1.225 | - | 21.370 ± 0.483 | 31.656 ± 0.160 | 66.911 ± 0.736 | 55.002 ± 3.838 | 55.491 ± 5.522 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 5.228 ± 0.071 | - | 5.207 ± 0.137 | 10.937 ± 0.232 | 34.601 ± 0.297 | 56.859 ± 6.353 | 20.273 ± 7.369 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 113.574 ± 1.041 | - | 120.710 ± 6.829 | 114.017 ± 0.875 | 113.176 ± 0.824 | 183.971 ± 0.788 | 186.009 ± 8.552 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 40.557 ± 0.407 | - | 40.322 ± 0.320 | 39.374 ± 0.808 | 64.100 ± 0.812 | 175.076 ± 1.935 | 56.835 ± 0.901 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 111.902 ± 1.047 | - | 112.100 ± 1.202 | 116.304 ± 4.003 | 131.332 ± 1.344 | 114.648 ± 1.065 | 132.211 ± 5.773 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 44.398 ± 0.679 | - | 43.775 ± 0.820 | 49.954 ± 9.174 | 70.269 ± 0.710 | 113.962 ± 7.877 | 51.939 ± 1.474 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 48.564 ± 1.481 | - | 50.950 ± 0.616 | 34.346 ± 1.225 | 43.141 ± 1.518 | 57.613 ± 4.588 | 65.660 ± 16.913 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 12.984 ± 0.113 | - | 13.119 ± 0.104 | 12.183 ± 0.189 | 22.741 ± 0.671 | 58.263 ± 6.142 | 22.995 ± 6.455 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 111.145 ± 7.929 | 112.787 ± 1.644 | unsupported | 127.744 ± 12.003 | 120.952 ± 5.785 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 45.266 ± 0.616 | 45.071 ± 0.670 | unsupported | 44.875 ± 0.325 | 65.430 ± 0.699 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 1.145 ± 0.014 | - | 1.157 ± 0.013 | 1.155 ± 0.018 | 1.166 ± 0.007 | 1.741 ± 5.363 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.586 ± 0.014 | - | 0.596 ± 0.021 | 0.617 ± 0.022 | 1.172 ± 0.008 | 4.985 ± 3.641 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 1.136 ± 0.012 | - | 1.147 ± 0.012 | 1.173 ± 0.011 | 1.173 ± 0.050 | 2.038 ± 4.859 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.582 ± 0.017 | - | 0.595 ± 0.011 | 0.603 ± 0.020 | 0.651 ± 0.029 | 5.074 ± 4.159 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.720 ± 0.027 | - | unsupported | 1.782 ± 0.039 | 1.702 ± 0.049 | 7.079 ± 5.213 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.872 ± 0.015 | - | unsupported | 0.916 ± 0.054 | 1.292 ± 0.141 | 3.289 ± 4.772 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.473 ± 0.032 | - | 0.646 ± 0.032 | 0.790 ± 0.009 | 0.335 ± 0.018 | 1.048 ± 0.098 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.312 ± 0.023 | - | 0.431 ± 0.021 | 0.354 ± 0.025 | 0.381 ± 0.108 | 0.936 ± 0.043 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 1.184 ± 0.037 | - | 1.150 ± 0.012 | 1.773 ± 0.017 | 1.167 ± 0.021 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.584 ± 0.008 | - | 0.593 ± 0.016 | 0.968 ± 0.035 | 0.845 ± 0.535 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 1.148 ± 0.024 | - | 1.151 ± 0.026 | 1.172 ± 0.014 | 1.178 ± 0.023 | 2.364 ± 4.259 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.585 ± 0.016 | - | 0.594 ± 0.014 | 0.606 ± 0.014 | 0.654 ± 0.025 | 3.864 ± 4.483 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.700 ± 0.027 | - | 0.972 ± 0.014 | 0.833 ± 0.009 | 0.619 ± 0.030 | 1.595 ± 0.210 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.908 ± 0.016 | - | 1.039 ± 0.019 | 0.691 ± 0.032 | 0.762 ± 0.145 | 1.774 ± 0.155 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.716 ± 0.038 | - | 0.002 ± 0.000 | 1.826 ± 0.079 | 1.775 ± 0.081 | 8.406 ± 1.406 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.876 ± 0.015 | - | 0.002 ± 0.000 | 0.901 ± 0.016 | 0.974 ± 0.836 | 5.462 ± 3.219 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.032 ± 0.000 | - | 0.036 ± 0.001 | 0.058 ± 0.006 | 0.082 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.058 ± 0.004 | - | 0.063 ± 0.001 | 0.074 ± 0.006 | 0.106 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.212 ± 0.009 | - | 0.234 ± 0.016 | 0.288 ± 0.015 | 0.229 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.044 ± 0.002 | - | 0.048 ± 0.000 | 0.129 ± 0.002 | 0.100 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.033 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.002 | 0.006 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.001 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.026 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.358 ± 0.025 | - | 0.444 ± 0.021 | 1.005 ± 0.009 | 0.590 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.015 ± 0.000 | - | 0.019 ± 0.000 | 0.028 ± 0.003 | 0.058 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.002 | 0.059 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.034 ± 0.001 | - | 0.045 ± 0.000 | 0.045 ± 0.002 | 0.064 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.058 ± 0.005 | - | 0.088 ± 0.001 | 0.084 ± 0.004 | 0.094 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.214 ± 0.006 | - | 0.289 ± 0.021 | 0.255 ± 0.003 | 0.264 ± 0.104 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.042 ± 0.000 | - | 0.067 ± 0.001 | 0.089 ± 0.003 | 0.120 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.021 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.001 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.362 ± 0.032 | - | 0.470 ± 0.035 | 0.627 ± 0.017 | 1.094 ± 1.041 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.015 ± 0.000 | - | 0.026 ± 0.000 | 0.037 ± 0.001 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.358 ± 0.016 | - | 0.370 ± 0.012 | 0.510 ± 0.007 | 0.528 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.640 ± 0.035 | - | 1.715 ± 0.004 | 1.698 ± 0.036 | 1.791 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 5.577 ± 0.121 | - | 5.294 ± 0.024 | 5.501 ± 0.048 | 6.240 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 1.732 ± 0.012 | - | 1.798 ± 0.020 | 1.467 ± 0.033 | 2.143 ± 0.098 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.072 ± 0.000 | - | 0.077 ± 0.004 | 0.087 ± 0.008 | 0.120 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.003 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.025 ± 0.003 | 0.049 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | 0.024 ± 0.000 | 0.028 ± 0.007 | 0.062 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 14.132 ± 0.221 | - | 14.393 ± 0.141 | 11.329 ± 0.276 | 17.739 ± 0.168 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.590 ± 0.021 | - | 0.540 ± 0.017 | 0.577 ± 0.011 | 0.697 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.002 | 0.019 ± 0.001 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.045 ± 0.000 | - | 0.047 ± 0.001 | 0.050 ± 0.000 | 0.107 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.623 ± 0.005 | - | 0.622 ± 0.026 | 0.504 ± 0.139 | 0.519 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 2.423 ± 0.025 | - | 2.402 ± 0.015 | 1.706 ± 0.053 | 2.397 ± 0.151 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 7.138 ± 0.118 | - | 7.176 ± 0.107 | 6.948 ± 0.325 | 5.756 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 2.122 ± 0.020 | - | 7.547 ± 0.220 | 1.324 ± 0.027 | 2.764 ± 0.085 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.004 | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.078 ± 0.000 | - | 0.095 ± 0.011 | 0.104 ± 0.020 | 0.128 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.011 ± 0.003 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.010 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.025 ± 0.005 | - | 0.030 ± 0.000 | 0.022 ± 0.000 | 0.041 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.013 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.032 ± 0.001 | - | 0.032 ± 0.000 | 0.029 ± 0.000 | 0.046 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 16.953 ± 1.009 | - | 25.090 ± 0.533 | 10.457 ± 0.096 | 23.686 ± 4.813 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.577 ± 0.005 | - | 0.689 ± 0.021 | 0.782 ± 0.018 | 1.011 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.021 ± 0.000 | - | 0.025 ± 0.000 | 0.020 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.062 ± 0.011 | - | 0.063 ± 0.000 | 0.050 ± 0.000 | 0.076 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.397 ± 0.017 | - | 0.405 ± 0.018 | 0.337 ± 0.012 | 0.288 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.542 ± 0.012 | - | 1.575 ± 0.038 | 1.354 ± 0.025 | 1.447 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.744 ± 0.142 | - | 4.377 ± 0.028 | 4.066 ± 0.016 | 4.311 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.791 ± 0.030 | - | 0.814 ± 0.037 | 0.797 ± 0.015 | 1.073 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.058 ± 0.009 | - | 0.056 ± 0.000 | 0.060 ± 0.009 | 0.099 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.004 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.003 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.015 ± 0.001 | - | 0.018 ± 0.000 | 0.020 ± 0.003 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.020 ± 0.002 | - | 0.023 ± 0.002 | 0.023 ± 0.000 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 6.286 ± 0.118 | - | 6.342 ± 0.182 | 6.149 ± 0.161 | 8.174 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.005 ± 0.001 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.393 ± 0.014 | - | 0.405 ± 0.016 | 0.360 ± 0.017 | 0.533 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.038 ± 0.001 | - | 0.041 ± 0.001 | 0.038 ± 0.000 | 0.086 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.598 ± 0.011 | - | 0.567 ± 0.021 | 0.439 ± 0.082 | 0.256 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 2.364 ± 0.024 | - | 2.397 ± 0.027 | 1.344 ± 0.010 | 1.065 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 6.275 ± 0.185 | - | 6.542 ± 2.813 | 4.969 ± 0.146 | 2.866 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 0.842 ± 0.020 | - | 6.215 ± 0.350 | 0.786 ± 0.295 | 1.996 ± 0.758 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.067 ± 0.004 | - | 0.082 ± 0.002 | 0.094 ± 0.027 | 0.090 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.007 ± 0.002 | - | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.011 ± 0.001 | - | 0.013 ± 0.000 | 0.010 ± 0.002 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.024 ± 0.001 | - | 0.026 ± 0.001 | 0.018 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.012 ± 0.002 | - | 0.017 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.028 ± 0.000 | - | 0.032 ± 0.000 | 0.021 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 6.709 ± 0.090 | - | 12.085 ± 0.191 | 5.905 ± 0.099 | 9.502 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.518 ± 0.025 | - | 0.527 ± 0.028 | 0.527 ± 0.049 | 0.327 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.021 ± 0.000 | - | 0.025 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.053 ± 0.001 | - | 0.052 ± 0.004 | 0.038 ± 0.000 | 0.097 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.181 ± 0.010 | - | 0.183 ± 0.014 | 0.150 ± 0.002 | 0.262 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.830 ± 0.005 | - | 0.840 ± 0.002 | 0.754 ± 0.011 | 1.277 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.433 ± 0.015 | - | 2.451 ± 0.037 | 2.294 ± 0.006 | 4.252 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.336 ± 0.005 | - | 0.374 ± 0.006 | 0.379 ± 0.020 | 1.057 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.003 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.025 ± 0.001 | - | 0.028 ± 0.004 | 0.030 ± 0.002 | 0.110 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.001 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.001 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.010 ± 0.000 | - | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 2.820 ± 0.128 | - | 2.798 ± 0.094 | 3.128 ± 0.200 | 8.109 ± 0.198 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.174 ± 0.005 | - | 0.192 ± 0.017 | 0.197 ± 0.016 | 0.553 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.019 ± 0.000 | - | 0.025 ± 0.000 | 0.025 ± 0.000 | 0.067 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.284 ± 0.011 | - | 0.274 ± 0.010 | 0.238 ± 0.014 | 0.259 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.108 ± 0.022 | - | 1.145 ± 0.039 | 0.846 ± 0.008 | 1.089 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 3.069 ± 0.090 | - | 2.969 ± 0.014 | 2.823 ± 0.014 | 3.113 ± 0.099 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.524 ± 0.024 | - | 2.932 ± 0.980 | 0.550 ± 0.036 | 1.411 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.003 ± 0.003 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.031 ± 0.001 | - | 0.034 ± 0.000 | 0.037 ± 0.003 | 0.106 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.005 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.002 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.014 ± 0.000 | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 4.224 ± 0.197 | - | 9.359 ± 0.388 | 3.852 ± 0.053 | 10.816 ± 4.648 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.232 ± 0.018 | - | 0.218 ± 0.011 | 0.234 ± 0.038 | 0.385 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.026 ± 0.000 | - | 0.028 ± 0.000 | 0.028 ± 0.000 | 0.097 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.098 ± 0.001 | - | 0.102 ± 0.003 | 0.119 ± 0.002 | 0.103 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.127 ± 0.005 | - | 0.131 ± 0.001 | 0.156 ± 0.015 | 0.157 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.495 ± 0.033 | - | 0.523 ± 0.017 | 0.580 ± 0.024 | 0.515 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.076 ± 0.004 | - | 0.080 ± 0.001 | 0.106 ± 0.016 | 0.171 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.012 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.011 ± 0.000 | - | 0.012 ± 0.001 | 0.016 ± 0.001 | 0.041 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.019 ± 0.000 | 0.046 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.002 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.002 | 0.039 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.664 ± 0.025 | - | 0.746 ± 0.022 | 0.930 ± 0.020 | 1.094 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.002 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.052 ± 0.002 | - | 0.058 ± 0.001 | 0.055 ± 0.016 | 0.097 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.042 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.125 ± 0.000 | - | 0.133 ± 0.005 | 0.097 ± 0.002 | 0.094 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.174 ± 0.001 | - | 0.183 ± 0.012 | 0.119 ± 0.002 | 0.170 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.618 ± 0.015 | - | 0.649 ± 0.028 | 0.325 ± 0.031 | 0.588 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.103 ± 0.007 | - | 0.112 ± 0.017 | 0.112 ± 0.001 | 0.146 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.015 ± 0.003 | - | 0.019 ± 0.001 | 0.019 ± 0.000 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.021 ± 0.002 | 0.034 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.005 ± 0.002 | - | 0.008 ± 0.001 | 0.013 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.002 | 0.016 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.001 | 0.038 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.768 ± 0.042 | - | 0.817 ± 0.017 | 0.452 ± 0.041 | 0.756 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.014 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.079 ± 0.002 | - | 0.082 ± 0.000 | 0.049 ± 0.003 | 0.112 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.040 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.008 ± 0.000 | - | 0.012 ± 0.000 | 0.020 ± 0.001 | 0.040 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.151 ± 0.004 | - | 0.154 ± 0.006 | 0.156 ± 0.006 | 0.203 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.365 ± 0.011 | - | 0.369 ± 0.009 | 0.582 ± 0.010 | 0.616 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 0.947 ± 0.008 | - | 0.978 ± 0.012 | 1.235 ± 0.014 | 1.622 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.222 ± 0.013 | - | 0.227 ± 0.014 | 0.309 ± 0.006 | 0.562 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.009 ± 0.000 | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.052 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.003 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.007 | 0.069 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.002 | 0.009 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 1.758 ± 0.057 | - | 2.308 ± 0.021 | 2.462 ± 0.071 | 3.652 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.066 ± 0.003 | - | 0.073 ± 0.006 | 0.078 ± 0.005 | 0.166 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.022 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.155 ± 0.016 | - | 0.242 ± 0.008 | 0.201 ± 0.002 | 0.190 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.369 ± 0.010 | - | 0.570 ± 0.044 | 0.666 ± 0.071 | 0.541 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.987 ± 0.011 | - | 1.342 ± 0.010 | 1.459 ± 0.011 | 0.905 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.314 ± 0.008 | - | 3.005 ± 0.023 | 0.284 ± 0.021 | 1.180 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.013 ± 0.000 | - | 0.020 ± 0.000 | 0.021 ± 0.001 | 0.046 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.021 ± 0.002 | 0.075 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.013 | 0.015 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.006 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.317 ± 0.077 | - | 11.194 ± 2.612 | 2.118 ± 0.120 | 10.139 ± 1.034 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.067 ± 0.000 | - | 0.112 ± 0.005 | 0.106 ± 0.006 | 0.176 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.009 ± 0.000 | - | 0.018 ± 0.002 | 0.019 ± 0.003 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.106 ± 0.003 | - | 0.096 ± 0.002 | 0.065 ± 0.001 | 0.213 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.125 ± 0.004 | - | 0.113 ± 0.017 | 0.085 ± 0.009 | 0.292 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.460 ± 0.014 | - | 0.468 ± 0.028 | 0.634 ± 0.012 | 0.600 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.056 ± 0.005 | - | 0.062 ± 0.007 | 0.097 ± 0.017 | 0.133 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.007 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.001 | 0.041 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.002 | 0.011 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.449 ± 0.015 | - | 0.560 ± 0.022 | 0.735 ± 0.040 | 0.876 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.046 ± 0.001 | - | 0.052 ± 0.000 | 0.056 ± 0.001 | 0.089 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.020 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.107 ± 0.001 | - | 0.120 ± 0.010 | 0.085 ± 0.002 | 0.214 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.125 ± 0.001 | - | 0.152 ± 0.006 | 0.133 ± 0.041 | 0.273 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.468 ± 0.010 | - | 0.598 ± 0.010 | 0.401 ± 0.013 | 0.750 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.065 ± 0.003 | - | 0.083 ± 0.004 | 0.093 ± 0.009 | 0.131 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.005 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.010 ± 0.003 | - | 0.017 ± 0.000 | 0.020 ± 0.000 | 0.041 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.002 | - | 0.010 ± 0.001 | 0.015 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.019 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.020 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.476 ± 0.035 | - | 0.638 ± 0.032 | 0.345 ± 0.018 | 0.682 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.047 ± 0.006 | - | 0.071 ± 0.001 | 0.044 ± 0.001 | 0.115 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.014 ± 0.000 | 0.023 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.175 ± 0.030 | - | 1.209 ± 0.018 | 1.022 ± 0.022 | 1.352 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.424 ± 0.094 | - | 4.532 ± 0.090 | 3.901 ± 0.049 | 3.694 ± 0.203 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 13.119 ± 1.014 | - | 13.194 ± 0.051 | 12.613 ± 0.173 | 10.896 ± 0.169 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 2.319 ± 0.160 | - | 2.393 ± 0.050 | 2.470 ± 0.058 | 3.199 ± 0.129 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.143 ± 0.025 | - | 0.126 ± 0.005 | 0.142 ± 0.018 | 0.172 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.115 ± 0.011 | - | 0.122 ± 0.010 | 0.119 ± 0.010 | 0.160 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.017 ± 0.000 | - | 0.021 ± 0.000 | 0.022 ± 0.003 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.017 ± 0.001 | - | 0.019 ± 0.002 | 0.018 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.038 ± 0.007 | - | 0.043 ± 0.000 | 0.046 ± 0.001 | 0.047 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.020 ± 0.000 | - | 0.025 ± 0.002 | 0.023 ± 0.000 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.051 ± 0.005 | - | 0.054 ± 0.002 | 0.058 ± 0.005 | 0.059 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 19.565 ± 3.274 | - | 19.661 ± 0.506 | 19.297 ± 0.727 | 26.441 ± 0.205 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.011 ± 0.000 | - | 0.015 ± 0.002 | 0.016 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 0.945 ± 0.013 | - | 0.968 ± 0.052 | 1.042 ± 0.029 | 1.115 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.038 ± 0.000 | - | 0.041 ± 0.002 | 0.039 ± 0.001 | 0.046 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.099 ± 0.001 | - | 0.109 ± 0.003 | 0.105 ± 0.010 | 0.140 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.826 ± 0.010 | - | 1.831 ± 0.016 | 1.037 ± 0.060 | 1.283 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 6.364 ± 0.086 | - | 5.178 ± 1.804 | 3.907 ± 0.080 | 1.955 ± 0.209 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 13.455 ± 0.043 | - | 14.715 ± 0.156 | 13.542 ± 0.109 | 6.310 ± 0.273 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 2.773 ± 0.121 | - | 8.015 ± 0.058 | 2.830 ± 0.103 | 4.465 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.126 ± 0.026 | - | 0.143 ± 0.005 | 0.153 ± 0.006 | 0.177 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.013 ± 0.006 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.142 ± 0.014 | - | 0.149 ± 0.011 | 0.134 ± 0.010 | 0.159 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.017 ± 0.000 | - | 0.025 ± 0.000 | 0.021 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.023 ± 0.000 | - | 0.026 ± 0.000 | 0.025 ± 0.010 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.039 ± 0.000 | - | 0.057 ± 0.001 | 0.046 ± 0.028 | 0.054 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.028 ± 0.003 | - | 0.033 ± 0.000 | 0.022 ± 0.000 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.054 ± 0.002 | - | 0.077 ± 0.003 | 0.056 ± 0.005 | 0.063 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 22.447 ± 0.500 | - | 29.267 ± 2.642 | 21.846 ± 0.519 | 58.396 ± 26.958 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.019 ± 0.002 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 0.944 ± 0.013 | - | 0.972 ± 0.053 | 1.168 ± 0.026 | 0.672 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.053 ± 0.001 | - | 0.058 ± 0.000 | 0.037 ± 0.000 | 0.053 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.100 ± 0.001 | - | 0.137 ± 0.001 | 0.104 ± 0.001 | 0.124 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.787 ± 0.005 | - | 0.792 ± 0.008 | 0.774 ± 0.017 | 0.880 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.695 ± 0.016 | - | 3.610 ± 0.021 | 3.164 ± 0.039 | 2.767 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 11.159 ± 0.069 | - | 11.102 ± 0.221 | 9.361 ± 0.011 | 8.988 ± 0.081 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.318 ± 0.020 | - | 1.358 ± 0.029 | 1.490 ± 0.054 | 1.970 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.094 ± 0.007 | - | 0.095 ± 0.007 | 0.097 ± 0.025 | 0.149 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.074 ± 0.003 | - | 0.075 ± 0.006 | 0.073 ± 0.003 | 0.119 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.015 ± 0.000 | - | 0.017 ± 0.002 | 0.018 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.015 ± 0.001 | - | 0.017 ± 0.000 | 0.017 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.034 ± 0.003 | - | 0.034 ± 0.003 | 0.033 ± 0.001 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.002 | 0.010 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.019 ± 0.002 | - | 0.019 ± 0.000 | 0.020 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.042 ± 0.001 | - | 0.044 ± 0.000 | 0.040 ± 0.001 | 0.050 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 10.502 ± 0.063 | - | 10.778 ± 0.363 | 11.717 ± 0.521 | 13.550 ± 0.240 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.699 ± 0.042 | - | 0.717 ± 0.035 | 0.704 ± 0.087 | 0.812 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.031 ± 0.001 | - | 0.034 ± 0.001 | 0.032 ± 0.000 | 0.038 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.084 ± 0.001 | - | 0.086 ± 0.001 | 0.076 ± 0.001 | 0.123 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.296 ± 0.015 | - | 1.281 ± 0.067 | 0.739 ± 0.027 | 0.881 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 5.367 ± 0.016 | - | 5.313 ± 0.144 | 3.148 ± 0.028 | 1.441 ± 0.249 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 10.975 ± 0.116 | - | 12.887 ± 0.055 | 10.368 ± 0.238 | 4.678 ± 0.212 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.645 ± 0.115 | - | 6.994 ± 0.103 | 1.567 ± 0.087 | 2.534 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.099 ± 0.012 | - | 0.115 ± 0.004 | 0.100 ± 0.006 | 0.135 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.010 ± 0.002 | - | 0.013 ± 0.000 | 0.009 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.099 ± 0.006 | - | 0.092 ± 0.006 | 0.090 ± 0.008 | 0.129 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.018 ± 0.003 | - | 0.021 ± 0.000 | 0.016 ± 0.003 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.016 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.046 ± 0.001 | - | 0.047 ± 0.000 | 0.032 ± 0.005 | 0.039 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.024 ± 0.000 | - | 0.025 ± 0.001 | 0.019 ± 0.002 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.061 ± 0.006 | - | 0.061 ± 0.001 | 0.040 ± 0.002 | 0.050 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 12.463 ± 0.183 | - | 18.597 ± 0.689 | 12.206 ± 0.638 | 24.661 ± 5.291 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.015 ± 0.001 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.727 ± 0.017 | - | 0.835 ± 0.104 | 0.737 ± 0.042 | 0.534 ± 0.130 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.045 ± 0.001 | - | 0.046 ± 0.001 | 0.034 ± 0.010 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.122 ± 0.004 | - | 0.115 ± 0.006 | 0.077 ± 0.001 | 0.104 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.294 ± 0.018 | - | unsupported | 0.229 ± 0.010 | 0.463 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.452 ± 0.012 | - | unsupported | 1.286 ± 0.030 | 1.792 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 3.544 ± 0.012 | - | unsupported | 3.454 ± 0.120 | 4.450 ± 0.110 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.669 ± 0.004 | - | unsupported | 0.708 ± 0.011 | 0.851 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.002 | - | unsupported | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.034 ± 0.002 | - | unsupported | 0.038 ± 0.002 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.026 ± 0.001 | - | unsupported | 0.032 ± 0.001 | 0.047 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.011 ± 0.001 | - | unsupported | 0.018 ± 0.000 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.015 ± 0.000 | - | unsupported | 0.020 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 5.303 ± 0.125 | - | unsupported | 5.562 ± 0.250 | 6.954 ± 0.519 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.260 ± 0.009 | - | unsupported | 0.258 ± 0.024 | 0.338 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.029 ± 0.000 | - | unsupported | 0.033 ± 0.000 | 0.048 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.419 ± 0.012 | - | unsupported | 0.332 ± 0.014 | 0.469 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 2.199 ± 0.004 | - | unsupported | 1.300 ± 0.015 | 0.838 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 5.688 ± 0.028 | - | unsupported | 4.433 ± 0.239 | 2.191 ± 0.091 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.785 ± 0.009 | - | unsupported | 0.804 ± 0.005 | 1.472 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.003 ± 0.003 | - | unsupported | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.045 ± 0.001 | - | unsupported | 0.037 ± 0.006 | 0.047 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.008 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.037 ± 0.001 | - | unsupported | 0.035 ± 0.001 | 0.050 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.009 ± 0.001 | - | unsupported | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.010 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.018 ± 0.000 | - | unsupported | 0.017 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.012 ± 0.000 | - | unsupported | 0.013 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.025 ± 0.007 | - | unsupported | 0.021 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 6.240 ± 0.061 | - | unsupported | 6.220 ± 0.027 | 11.771 ± 0.169 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.349 ± 0.008 | - | unsupported | 0.267 ± 0.023 | 0.214 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.020 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.046 ± 0.002 | - | unsupported | 0.034 ± 0.002 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.116 ± 0.013 | - | unsupported | 1.508 ± 0.074 | 2.729 ± 0.080 | 2.104 ± 0.115 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.077 ± 0.006 | - | unsupported | 0.048 ± 0.009 | 0.097 ± 0.006 | 0.113 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.087 ± 0.001 | - | unsupported | 0.059 ± 0.007 | 0.154 ± 0.004 | 0.174 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.334 ± 0.012 | - | unsupported | 0.487 ± 0.130 | 0.346 ± 0.015 | 0.628 ± 0.015 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.147 ± 0.054 | - | unsupported | 0.519 ± 0.029 | 1.153 ± 0.250 | 1.407 ± 0.729 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.080 ± 0.005 | - | unsupported | 0.043 ± 0.009 | 0.073 ± 0.004 | 0.075 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.091 ± 0.001 | - | unsupported | 0.056 ± 0.014 | 0.129 ± 0.003 | 0.123 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.347 ± 0.015 | - | unsupported | 0.153 ± 0.023 | 0.491 ± 0.031 | 0.503 ± 0.021 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.233 ± 0.026 | - | unsupported | 0.516 ± 0.020 | 0.772 ± 0.021 | 0.256 ± 0.018 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.020 ± 0.001 | - | unsupported | 0.029 ± 0.004 | 0.148 ± 0.003 | 0.053 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.037 ± 0.000 | - | unsupported | 0.035 ± 0.004 | 0.161 ± 0.004 | 0.070 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.115 ± 0.005 | - | unsupported | 0.246 ± 0.013 | 0.274 ± 0.020 | 0.123 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.241 ± 0.010 | - | unsupported | 0.567 ± 0.042 | 1.216 ± 0.153 | 0.255 ± 0.021 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.022 ± 0.002 | - | unsupported | 0.040 ± 0.004 | 0.119 ± 0.007 | 0.054 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.039 ± 0.000 | - | unsupported | 0.056 ± 0.009 | 0.148 ± 0.005 | 0.069 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.118 ± 0.001 | - | unsupported | 0.287 ± 0.010 | 0.326 ± 0.037 | 0.123 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.145 ± 0.004 | - | 0.303 ± 0.024 | 0.242 ± 0.014 | 0.373 ± 0.035 | 0.147 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.050 ± 0.003 | - | 0.054 ± 0.003 | 0.052 ± 0.003 | 0.052 ± 0.006 | 0.046 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.051 ± 0.001 | - | 0.058 ± 0.001 | 0.074 ± 0.007 | 0.061 ± 0.009 | 0.053 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.072 ± 0.000 | - | 0.088 ± 0.000 | 0.127 ± 0.003 | 0.112 ± 0.016 | 0.079 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.133 ± 0.006 | - | 0.271 ± 0.031 | 0.258 ± 0.022 | 0.598 ± 0.276 | 0.152 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.051 ± 0.001 | - | 0.078 ± 0.008 | 0.062 ± 0.001 | 0.061 ± 0.001 | 0.044 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.053 ± 0.000 | - | 0.091 ± 0.005 | 0.090 ± 0.005 | 0.061 ± 0.002 | 0.050 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.075 ± 0.001 | - | 0.125 ± 0.010 | 0.140 ± 0.006 | 0.142 ± 0.006 | 0.077 ± 0.011 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 16.576 ± 0.180 | - | 19.253 ± 0.234 | 44.511 ± 2.493 | 69.958 ± 4.777 | 43.713 ± 20.966 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 10.688 ± 0.176 | - | 17.293 ± 1.238 | 17.174 ± 0.376 | 51.979 ± 0.236 | 33.062 ± 7.583 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 13.516 ± 2.041 | - | 12.989 ± 0.677 | 12.317 ± 0.019 | 14.950 ± 1.378 | 18.049 ± 4.486 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 10.287 ± 0.165 | - | 17.878 ± 12.960 | 6.089 ± 0.125 | 11.988 ± 3.343 | 10.107 ± 2.058 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 5.125 ± 0.428 | - | 5.518 ± 0.371 | 5.169 ± 0.181 | 5.880 ± 0.119 | 5.755 ± 0.102 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 7.660 ± 0.160 | - | 14.364 ± 3.540 | 6.727 ± 0.105 | 9.545 ± 1.117 | 7.883 ± 0.158 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.523 ± 0.326 | - | 8.097 ± 0.969 | 6.324 ± 0.049 | 6.580 ± 0.029 | 6.696 ± 0.128 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 9.013 ± 0.083 | - | 14.211 ± 0.366 | 7.436 ± 0.389 | 9.927 ± 0.163 | 10.847 ± 2.554 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 9.713 ± 0.275 | - | 9.575 ± 0.321 | 9.793 ± 0.183 | 23.363 ± 0.872 | 9.637 ± 0.569 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 9.969 ± 0.195 | - | 17.298 ± 2.196 | 9.479 ± 0.080 | 20.980 ± 1.946 | 9.344 ± 1.314 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.234 ± 0.011 | 0.231 ± 0.020 | unsupported | 0.260 ± 0.054 | 0.578 ± 0.023 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 4.694 ± 0.405 | 5.402 ± 0.279 | unsupported | 4.884 ± 0.135 | 7.293 ± 0.865 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.174 ± 0.003 | 3.310 ± 1.125 | unsupported | 0.239 ± 0.034 | 1.587 ± 0.027 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.735 ± 0.087 | 7.169 ± 2.136 | unsupported | 3.253 ± 0.380 | 9.496 ± 0.234 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 17.674 ± 0.279 | - | 18.071 ± 0.467 | 16.337 ± 0.156 | 32.854 ± 0.853 | 21.905 ± 2.685 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 17.904 ± 0.309 | - | 21.592 ± 7.101 | 9.381 ± 0.143 | 18.989 ± 3.455 | 13.691 ± 0.611 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | failed | failed | failed | 29.542 ± 0.497 | 4.922 ± 0.025 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | failed | 32.258 ± 3.013 | failed | 33.020 ± 2.055 | 4.120 ± 0.012 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 13.385 ± 0.280 | - | 14.082 ± 0.115 | 22.786 ± 0.567 | 30.161 ± 1.664 | 25.494 ± 8.445 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 9.783 ± 1.093 | - | 24.122 ± 14.777 | 11.793 ± 0.395 | 26.511 ± 2.988 | 23.708 ± 14.687 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.168 ± 0.034 | - | 1.163 ± 0.021 | 1.174 ± 0.013 | 5.893 ± 0.644 | 1.533 ± 0.113 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 5.999 ± 0.002 | - | 0.607 ± 0.024 | 1.172 ± 0.017 | 10.417 ± 1.807 | 1.612 ± 0.022 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 17.760 ± 1.158 | - | 18.028 ± 0.384 | 16.728 ± 0.158 | 12.895 ± 1.115 | 14.843 ± 0.224 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 56.023 ± 0.086 | - | 81.146 ± 2.992 | 44.135 ± 0.604 | 15.895 ± 0.685 | 16.882 ± 9.987 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 18.131 ± 1.211 | - | 17.302 ± 0.311 | 16.786 ± 0.047 | 14.796 ± 4.720 | 14.601 ± 0.551 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 57.010 ± 7.254 | - | 62.151 ± 12.635 | 43.010 ± 0.360 | 16.176 ± 0.488 | 15.498 ± 1.197 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 0.656 ± 0.012 | 0.671 ± 0.009 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 13.312 ± 0.419 | 14.066 ± 0.714 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 0.964 ± 0.037 | 9.000 ± 2.381 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 11.225 ± 1.239 | 18.029 ± 4.077 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 0.605 ± 0.070 | 0.716 ± 0.043 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 13.016 ± 0.228 | 14.744 ± 1.315 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 0.888 ± 0.016 | 8.302 ± 2.719 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 10.035 ± 0.393 | 15.489 ± 2.165 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 13.096 ± 0.673 | - | 13.053 ± 0.621 | 12.350 ± 0.211 | 16.127 ± 0.429 | 19.562 ± 1.711 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 10.084 ± 0.540 | - | 24.007 ± 10.972 | 6.341 ± 0.184 | 10.901 ± 0.455 | 9.766 ± 0.840 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 81.763 ± 1.936 | 76.872 ± 6.227 | 71.164 ± 2.332 | 40.978 ± 0.743 | 45.640 ± 6.102 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 174.391 ± 21.177 | 177.489 ± 2.720 | 159.820 ± 16.920 | 45.325 ± 0.686 | 46.312 ± 1.336 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 16.373 ± 0.669 | - | 70.666 ± 5.042 | 15.651 ± 0.757 | 157.414 ± 6.672 | 20.916 ± 0.887 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 9.742 ± 0.596 | - | 45.164 ± 14.302 | 8.597 ± 0.162 | 146.307 ± 10.351 | 15.629 ± 3.302 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.068 ± 0.002 | 0.065 ± 0.001 | unsupported | 0.074 ± 0.022 | 0.078 ± 0.022 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 1.097 ± 0.066 | 1.091 ± 0.040 | unsupported | 0.501 ± 0.021 | 0.912 ± 0.192 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.036 ± 0.004 | 0.040 ± 0.002 | unsupported | 0.044 ± 0.087 | 0.101 ± 0.011 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.489 ± 0.013 | 0.505 ± 0.010 | unsupported | 0.564 ± 0.027 | 0.303 ± 0.268 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.032 ± 0.002 | 0.033 ± 0.002 | unsupported | 0.062 ± 0.024 | 0.181 ± 0.014 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 0.611 ± 0.015 | 0.546 ± 0.055 | unsupported | 0.521 ± 0.034 | 2.313 ± 0.465 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.020 ± 0.002 | 0.021 ± 0.001 | unsupported | 0.059 ± 0.003 | 0.136 ± 0.023 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.275 ± 0.010 | 0.268 ± 0.007 | unsupported | 0.510 ± 0.003 | 0.832 ± 0.026 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.107 ± 0.001 | 0.107 ± 0.001 | unsupported | 0.137 ± 0.004 | 0.161 ± 0.009 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 2.151 ± 0.105 | 2.091 ± 0.036 | unsupported | 2.487 ± 0.092 | 1.537 ± 0.080 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.053 ± 0.002 | 0.059 ± 0.013 | unsupported | 0.071 ± 0.001 | 0.100 ± 0.006 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 0.756 ± 0.010 | 0.739 ± 0.005 | unsupported | 1.029 ± 0.004 | 0.881 ± 0.032 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.045 ± 0.021 | 0.070 ± 0.022 | unsupported | 0.101 ± 0.003 | 0.155 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.995 ± 0.221 | 1.043 ± 0.223 | unsupported | 0.663 ± 0.006 | 2.043 ± 0.084 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.096 ± 0.025 | 0.107 ± 0.011 | unsupported | 0.051 ± 0.001 | 0.134 ± 0.006 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.552 ± 0.078 | 0.817 ± 0.212 | unsupported | 0.253 ± 0.004 | 1.292 ± 0.158 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.063 ± 0.015 | 0.071 ± 0.006 | unsupported | 0.068 ± 0.001 | 0.209 ± 0.003 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.335 ± 0.240 | 1.338 ± 0.248 | unsupported | 0.726 ± 0.002 | 2.971 ± 0.059 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.091 ± 0.015 | 0.111 ± 0.007 | unsupported | 0.046 ± 0.002 | 0.199 ± 0.020 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.727 ± 0.049 | 0.813 ± 0.058 | unsupported | 0.327 ± 0.007 | 1.866 ± 0.241 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.211 ± 0.007 | 0.236 ± 0.005 | unsupported | unsupported | 0.229 ± 0.005 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 3.520 ± 0.210 | 3.530 ± 0.476 | unsupported | unsupported | 4.892 ± 0.305 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.197 ± 0.006 | 0.210 ± 0.009 | unsupported | unsupported | 0.231 ± 0.024 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.495 ± 0.081 | 1.622 ± 0.079 | unsupported | unsupported | 2.682 ± 0.442 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.008 ± 0.000 | 0.014 ± 0.001 | unsupported | 0.009 ± 0.001 | 0.040 ± 0.014 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.162 ± 0.007 | 0.174 ± 0.005 | unsupported | 0.155 ± 0.004 | 0.230 ± 0.004 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.011 ± 0.002 | 0.032 ± 0.008 | unsupported | 0.009 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.074 ± 0.006 | 0.086 ± 0.007 | unsupported | 0.054 ± 0.006 | 0.172 ± 0.006 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.028 ± 0.015 | 0.039 ± 0.011 | unsupported | 0.058 ± 0.001 | 0.135 ± 0.011 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.560 ± 0.212 | 0.591 ± 0.227 | unsupported | 0.785 ± 0.003 | 1.764 ± 0.137 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.044 ± 0.008 | 0.059 ± 0.012 | unsupported | 0.070 ± 0.003 | 0.125 ± 0.005 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.314 ± 0.089 | 0.384 ± 0.134 | unsupported | 0.401 ± 0.005 | 1.250 ± 0.096 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.024 ± 0.002 | 0.024 ± 0.001 | unsupported | 0.056 ± 0.002 | 0.136 ± 0.005 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.407 ± 0.110 | 0.432 ± 0.088 | unsupported | 1.109 ± 0.085 | 1.417 ± 0.088 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.016 ± 0.000 | 0.016 ± 0.001 | unsupported | 0.055 ± 0.002 | 0.102 ± 0.004 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.199 ± 0.023 | 0.203 ± 0.014 | unsupported | 0.491 ± 0.043 | 0.946 ± 0.253 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.025 ± 0.001 | 0.025 ± 0.001 | unsupported | 0.057 ± 0.002 | 0.168 ± 0.014 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.426 ± 0.090 | 0.441 ± 0.079 | unsupported | 1.008 ± 0.191 | 1.492 ± 0.045 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.016 ± 0.001 | 0.019 ± 0.011 | unsupported | 0.056 ± 0.002 | 0.112 ± 0.002 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.207 ± 0.035 | 0.206 ± 0.018 | unsupported | 0.499 ± 0.016 | 0.924 ± 0.041 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.063 ± 0.008 | 0.083 ± 0.009 | unsupported | 0.073 ± 0.003 | 0.218 ± 0.012 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.262 ± 0.037 | 1.310 ± 0.072 | unsupported | 0.802 ± 0.006 | 2.883 ± 0.053 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.096 ± 0.020 | 0.113 ± 0.006 | unsupported | 0.044 ± 0.001 | 0.195 ± 0.017 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.718 ± 0.029 | 0.790 ± 0.053 | unsupported | 0.316 ± 0.002 | 1.842 ± 0.121 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.078 ± 0.001 | 0.078 ± 0.000 | unsupported | 0.122 ± 0.001 | 0.234 ± 0.011 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 1.260 ± 0.023 | 1.260 ± 0.004 | unsupported | 2.083 ± 0.133 | 2.795 ± 0.097 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.041 ± 0.001 | 0.043 ± 0.001 | unsupported | 0.096 ± 0.003 | 0.161 ± 0.004 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.562 ± 0.010 | 0.560 ± 0.016 | unsupported | 0.901 ± 0.016 | 1.570 ± 0.043 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.614 ± 0.003 | 1.101 ± 0.094 | unsupported | 0.252 ± 0.009 | 0.156 ± 0.001 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 11.258 ± 0.467 | 20.492 ± 1.768 | unsupported | 5.530 ± 0.234 | 1.483 ± 0.031 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.453 ± 0.004 | 1.005 ± 0.082 | unsupported | 0.157 ± 0.041 | 0.132 ± 0.011 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 5.637 ± 0.128 | 16.663 ± 0.840 | unsupported | 2.381 ± 0.016 | 0.856 ± 0.049 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 51.062 ± 0.277 | - | unsupported | 51.346 ± 0.252 | - | 49.467 ± 0.390 | 48.812 ± 0.322 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 25.345 ± 0.552 | - | unsupported | 25.443 ± 0.618 | - | 48.095 ± 0.475 | 24.650 ± 0.462 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 43.920 ± 0.281 | - | unsupported | 39.497 ± 0.275 | - | 36.749 ± 0.137 | 41.467 ± 0.661 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 18.019 ± 0.369 | - | unsupported | 17.627 ± 0.038 | - | 36.380 ± 0.258 | 15.617 ± 0.273 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 21.589 ± 0.150 | - | unsupported | 38.882 ± 0.396 | - | 21.188 ± 0.358 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 17.930 ± 0.396 | - | unsupported | 17.172 ± 0.295 | - | 20.643 ± 0.156 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 50.524 ± 0.359 | - | unsupported | 50.316 ± 1.144 | - | 49.149 ± 0.308 | 49.760 ± 1.364 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 24.337 ± 0.404 | - | unsupported | 24.829 ± 0.889 | - | 47.024 ± 0.404 | 23.383 ± 0.192 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 21.550 ± 0.921 | - | unsupported | 22.906 ± 0.494 | - | 25.721 ± 0.698 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 11.701 ± 0.191 | - | unsupported | 11.502 ± 0.100 | - | 13.627 ± 0.275 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 22.471 ± 0.169 | - | unsupported | 22.630 ± 0.721 | - | 24.806 ± 0.261 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 13.199 ± 0.104 | - | unsupported | 11.424 ± 0.401 | - | 11.759 ± 0.308 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 51.171 ± 0.385 | - | unsupported | 49.832 ± 0.264 | - | 49.135 ± 0.533 | 50.962 ± 1.843 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 24.685 ± 0.222 | - | unsupported | 24.584 ± 0.281 | - | 46.780 ± 0.371 | 23.674 ± 0.363 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 41.039 ± 0.316 | - | unsupported | 41.441 ± 6.470 | - | 38.742 ± 0.237 | 37.256 ± 0.801 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 17.794 ± 0.431 | - | unsupported | 18.359 ± 2.959 | - | 36.080 ± 1.042 | 15.475 ± 0.399 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.294 ± 0.002 | unsupported | unsupported | 0.394 ± 0.005 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 4.983 ± 0.170 | unsupported | unsupported | 6.357 ± 0.105 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.349 ± 0.015 | unsupported | unsupported | 0.408 ± 0.111 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 3.977 ± 0.222 | unsupported | unsupported | 3.472 ± 0.123 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 51.291 ± 0.291 | - | unsupported | 51.638 ± 0.543 | - | 49.709 ± 0.751 | 49.388 ± 1.113 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 24.643 ± 0.279 | - | unsupported | 25.351 ± 0.232 | - | 47.642 ± 1.722 | 23.291 ± 0.401 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 84.800 ± 0.100 | - | 0.002 ± 0.000 | 85.209 ± 0.539 | 83.943 ± 0.596 | 101.334 ± 3.160 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 40.446 ± 0.406 | - | 0.002 ± 0.000 | 33.148 ± 1.421 | 49.155 ± 0.868 | 92.159 ± 0.381 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 58.517 ± 4.205 | - | 160.821 ± 1.076 | 60.928 ± 0.947 | 63.286 ± 0.527 | 67.841 ± 3.196 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 24.276 ± 0.727 | - | 71.736 ± 1.277 | 22.940 ± 0.229 | 34.306 ± 0.777 | 56.302 ± 2.970 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 170.182 ± 0.669 | - | 182.875 ± 1.229 | 174.339 ± 9.850 | 173.904 ± 2.794 | 188.661 ± 19.860 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 62.633 ± 0.455 | - | 63.154 ± 0.779 | 61.509 ± 3.708 | 96.693 ± 1.295 | 177.952 ± 9.022 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 57.708 ± 0.148 | - | 57.170 ± 0.260 | 61.758 ± 0.364 | 53.282 ± 1.001 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 32.744 ± 0.185 | - | 34.089 ± 1.194 | 24.155 ± 0.461 | 30.498 ± 0.263 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 153.960 ± 0.669 | - | 0.002 ± 0.000 | 97.557 ± 5.334 | 107.141 ± 1.762 | 163.656 ± 2.596 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 39.176 ± 1.256 | - | 0.002 ± 0.000 | 36.546 ± 1.275 | 42.637 ± 1.786 | 158.386 ± 1.844 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 117.608 ± 10.910 | - | 0.002 ± 0.000 | 66.686 ± 0.631 | 81.285 ± 0.624 | 215.526 ± 11.572 | 80.575 ± 5.685 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 33.986 ± 0.302 | - | 0.002 ± 0.000 | 65.814 ± 1.812 | 44.925 ± 0.993 | 217.764 ± 2.600 | 36.663 ± 1.639 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 49.166 ± 0.486 | - | 49.529 ± 0.403 | 54.622 ± 0.465 | 52.111 ± 8.062 | 68.104 ± 1.717 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 19.700 ± 0.353 | - | 19.628 ± 0.157 | 21.113 ± 0.281 | 28.134 ± 0.880 | 61.534 ± 8.261 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 49.680 ± 0.160 | - | 49.684 ± 3.070 | 51.175 ± 0.202 | 49.121 ± 2.237 | 67.678 ± 0.842 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 19.939 ± 0.178 | - | 19.330 ± 0.287 | 18.461 ± 0.108 | 27.463 ± 0.421 | 57.484 ± 1.784 | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.775758 | 423484.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.389285 | 274330.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.805371 | 300335.69 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 5.658227 | 353639.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 3.773295 | 235830.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.062123 | 253882.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 18.449055 | 1153065.94 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 18.397323 | 1149832.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 9.421269 | 588829.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 9.579174 | 598698.38 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 9.240862 | 577553.88 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 4.163545 | 260221.56 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 3.749553 | 234347.06 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 3.690097 | 230631.06 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 116.683071 | 7292691.94 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 78.148639 | 4884289.94 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 75.106085 | 4694130.31 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 86.435497 | 5402218.56 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 25.389333 | 1586833.31 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 3.822618 | 238913.62 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 2.789698 | 174356.12 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 52.957143 | 3309821.44 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 151.936174 | 9496010.88 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 52.052806 | 3253300.38 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 59.759353 | 3734959.56 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 114.704776 | 7169048.50 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 10.488131 | 655508.19 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 10.728545 | 670534.06 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 212.991333 | 13311958.31 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 225.050070 | 14065629.38 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 15.417213 | 963575.81 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 144.005175 | 9000323.44 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 179.592434 | 11224527.12 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 288.466069 | 18029129.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 9.686363 | 605397.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 11.463830 | 716489.38 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 208.249782 | 13015611.38 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 235.904335 | 14744020.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 14.214969 | 888435.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 132.834618 | 8302163.62 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 160.560568 | 10035035.50 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 247.822390 | 15488899.38 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.020723 | 78448.80 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.737222 | 74019.09 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.353145 | 68017.89 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.179424 | 65303.50 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.294736 | 911842.00 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.004612 | 500576.50 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 8.774795 | 1096849.38 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 8.725282 | 1090660.25 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.468139 | 101064.67 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.814743 | 43980.36 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.280368 | 35630.75 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.571857 | 40185.27 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.422044 | 302755.50 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.515694 | 564461.75 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.915642 | 489455.25 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.043348 | 505418.50 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.569986 | 180781.03 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.962284 | 61910.69 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.046562 | 31977.53 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.091770 | 32683.91 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.501388 | 2312673.50 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.168339 | 521042.37 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.885102 | 610637.75 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.371742 | 546467.75 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.672691 | 135510.80 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.785089 | 59142.02 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.291066 | 20172.91 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.358416 | 21225.25 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.658458 | 832307.25 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.077492 | 509686.50 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.201721 | 275215.12 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.143349 | 267918.62 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.321656 | 161275.88 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.757654 | 136838.34 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.835793 | 106809.27 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.879908 | 107498.56 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.297059 | 1537132.38 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 19.899711 | 2487463.88 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.204546 | 2150568.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 16.726051 | 2090756.37 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.417864 | 100279.12 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.524329 | 70692.64 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.363304 | 52551.62 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.765385 | 58834.14 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.048765 | 881095.62 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.230996 | 1028874.50 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.048278 | 756034.75 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.908902 | 738612.75 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.924414 | 155068.97 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.444470 | 100694.84 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.909996 | 45468.69 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.457476 | 69648.06 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 16.347383 | 2043422.88 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.300357 | 662544.62 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.956076 | 994509.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 8.347752 | 1043469.00 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.552188 | 133627.94 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.274305 | 51161.02 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.129024 | 95766.00 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.826038 | 106656.84 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.332490 | 1291561.25 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.025842 | 253230.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.418985 | 552373.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.537057 | 817132.12 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.347294 | 208551.47 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.364492 | 68195.19 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.017313 | 62770.52 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.520586 | 70634.16 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 23.770386 | 2971298.25 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.810737 | 726342.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.679360 | 1334920.00 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.705946 | 1338243.25 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.712557 | 198633.70 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.953830 | 46153.59 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.816658 | 90885.28 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.092413 | 110818.95 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.927448 | 1865931.00 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.614566 | 326820.75 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.813178 | 726647.25 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.506996 | 813374.50 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.661725 | 229089.45 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 13.493249 | 210832.02 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 15.109268 | 236082.31 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 39.133191 | 4891648.88 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 28.160182 | 3520022.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 28.237850 | 3529731.25 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.781759 | 230964.98 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 12.609515 | 197023.67 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 13.464791 | 210387.36 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 21.456157 | 2682019.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 11.959797 | 1494974.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 12.973880 | 1621735.00 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.568168 | 40127.62 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.570209 | 8909.52 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.533139 | 8330.30 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.881944 | 13780.38 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.839903 | 229987.88 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 1.237696 | 154712.00 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 1.292604 | 161575.50 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.391462 | 173932.75 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 1.771767 | 27683.86 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.602731 | 9417.67 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.684869 | 10701.08 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.031226 | 31737.91 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.372266 | 171533.25 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.429547 | 53693.38 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.589498 | 73687.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.685887 | 85735.88 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.643772 | 135058.94 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.723078 | 58173.09 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.817090 | 28392.03 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.482936 | 38795.88 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.114130 | 1764266.25 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.280345 | 785043.12 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.477817 | 559727.12 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.727382 | 590922.75 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.004928 | 125077.00 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.492284 | 70191.94 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.835984 | 44312.25 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.803494 | 59429.59 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.999787 | 1249973.38 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.211807 | 401475.88 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.510271 | 313783.88 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.068738 | 383592.25 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.690318 | 135786.22 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.593518 | 56148.72 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.515501 | 23679.70 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.536159 | 24002.48 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.335320 | 1416915.00 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.868257 | 1108532.12 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.257664 | 407208.00 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.454273 | 431784.12 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.542482 | 102226.28 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.544526 | 55383.22 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.008944 | 15764.75 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.042747 | 16292.92 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.564181 | 945522.62 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.930626 | 491328.25 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.593607 | 199200.88 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.625055 | 203131.88 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.727625 | 167619.14 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.629062 | 56704.09 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.622843 | 25356.92 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.621862 | 25341.59 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.932984 | 1491623.00 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.066603 | 1008325.38 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.404922 | 425615.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.530038 | 441254.75 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.184577 | 112259.02 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.583611 | 55993.92 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.054653 | 16478.95 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.186460 | 18538.44 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.390353 | 923794.12 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.995049 | 499381.12 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.658612 | 207326.50 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.648636 | 206079.50 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.939921 | 217811.27 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.676287 | 73066.98 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.019901 | 62810.95 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.290933 | 82670.83 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 23.062544 | 2882818.00 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.414615 | 801826.88 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.097781 | 1262222.62 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.476185 | 1309523.12 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.488739 | 195136.55 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.825024 | 44141.00 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.159686 | 96245.09 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.226001 | 112906.27 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.737237 | 1842154.62 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.531310 | 316413.75 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.745487 | 718185.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.320667 | 790083.38 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.966850 | 233857.03 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.807821 | 121997.20 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.983193 | 77862.39 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.979071 | 77797.98 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 22.359968 | 2794996.00 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.667840 | 2083480.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.082964 | 1260370.50 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.081347 | 1260168.38 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.294422 | 160850.34 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.163424 | 96303.50 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.633183 | 41143.48 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.720897 | 42514.02 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 12.563034 | 1570379.25 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.207380 | 900922.50 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.494987 | 561873.38 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.478314 | 559789.25 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.984393 | 156006.14 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 16.131730 | 252058.28 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 39.312827 | 614262.92 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 70.484561 | 1101321.27 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.865572 | 1483196.50 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 44.236189 | 5529523.62 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 90.064342 | 11258042.75 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 163.935702 | 20491962.75 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.424718 | 131636.22 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.076416 | 157444.00 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 28.984200 | 452878.12 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 64.339279 | 1005301.23 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.844864 | 855608.00 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 19.044557 | 2380569.62 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 45.093754 | 5636719.25 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 133.305654 | 16663206.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 6.306925 | 394182.81 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 4.696666 | 293541.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 101.718700 | 6357418.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 79.728523 | 4983032.69 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 6.530641 | 408165.06 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 5.578116 | 348632.25 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 55.555900 | 3472243.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 63.624395 | 3976524.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.013297 | 831.06 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001606 | 100.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.013490 | 843.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001552 | 97.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000109 | 6.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012176 | 761.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.006588 | 411.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000105 | 6.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.013151 | 821.94 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.006592 | 412.00 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000227 | 14.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.012698 | 793.62 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.004943 | 308.94 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000637 | 39.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.013128 | 820.50 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.004482 | 280.12 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000309 | 19.31 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.011826 | 739.12 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003622 | 226.38 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000122 | 7.62 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.012550 | 784.38 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003581 | 223.81 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `julia-base` is 10.7x faster than the slowest successful cell (`tenferro-trace`, 26.354 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 14.4x faster than the slowest successful cell (`tenferro-trace`, 26.354 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`julia-base`, 49.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`julia-base`, 49.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`julia-base`, 70.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`julia-base`, 70.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`julia-base`, 29.636 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 18.0x faster than the slowest successful cell (`pytorch-cpu`, 27.916 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 27.916 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 27.916 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 27.916 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 12.210 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 16.6x faster than the slowest successful cell (`pytorch-cpu`, 12.210 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 12.210 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 26.741 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 26.741 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 26.741 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 26.741 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 12.478 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 12.478 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 12.478 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`julia-base`, 56.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`julia-base`, 56.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 4982.6x faster than the slowest successful cell (`julia-base`, 8.406 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 2630.0x faster than the slowest successful cell (`julia-base`, 5.462 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 15.1x faster than the slowest successful cell (`jax-cpu`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 17.4x faster than the slowest successful cell (`jax-cpu`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 24.4x faster than the slowest successful cell (`jax-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x128x128`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`tenferro-trace`, 3.005 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `pytorch-cpu` is 13.9x faster than the slowest successful cell (`tenferro-eager`, 3.310 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 19.0x faster than the slowest successful cell (`tenferro-eager`, 3.310 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 17.2x faster than the slowest successful cell (`jax-cpu`, 10.417 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=1, shape=`4096x4096,rhs=64`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 157.414 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `pytorch-cpu` is 17.0x faster than the slowest successful cell (`jax-cpu`, 146.307 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`jax-cpu`, 146.307 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 13.8x faster than the slowest successful cell (`tenferro-eager`, 20.492 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 19.5x faster than the slowest successful cell (`tenferro-eager`, 16.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 61044.5x faster than the slowest successful cell (`julia-base`, 101.334 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 47406.8x faster than the slowest successful cell (`julia-base`, 92.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 96666.0x faster than the slowest successful cell (`julia-base`, 163.656 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 70299.8x faster than the slowest successful cell (`julia-base`, 158.386 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 134117.0x faster than the slowest successful cell (`julia-base`, 215.526 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 105301.8x faster than the slowest successful cell (`julia-base`, 217.764 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 108.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 117.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 56.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 20.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 38.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 98.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9940.1 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 99.4x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 8073.7 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 20.2x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 161708.1 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1617.1x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 138084.1 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 345.2x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 158556.1 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1585.6x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 119145.8 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 297.9x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 83520.7 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 835.2x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 64902.2 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 162.3x faster than the bound.

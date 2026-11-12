# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/public_api/20261112_201305/run.yaml`
- Timestamp: `2026-11-12T11:13:05.629939+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 4 8`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/zen5-cpu/cpu/public_api/20261112_201305`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `disabled`
- Container: `none`

## Thread Environments

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

### Threads: 8

- OMP_NUM_THREADS: `8`
- OMP_THREAD_LIMIT: `8`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `8`
- OPENBLAS_NUM_THREADS: `8`
- GOTO_NUM_THREADS: `8`
- MKL_NUM_THREADS: `8`
- VECLIB_MAXIMUM_THREADS: `8`
- VECLIB_NUM_THREADS: `8`
- NUMEXPR_NUM_THREADS: `8`
- BLIS_NUM_THREADS: `8`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=8`
- JULIA_NUM_THREADS: `8`

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

## Threads: 4 8

- CSV: `data/results/zen5-cpu/cpu/public_api/20261112_201305/cpu_public_api_t4_20261112_201305.csv`
- CSV: `data/results/zen5-cpu/cpu/public_api/20261112_201305/cpu_public_api_t8_20261112_201305.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 1.953 ± 0.006 | - | 342.984 ± 2.995 | 2.151 ± 0.018 | 3.100 ± 0.099 | 2.704 ± 1.739 | - |
| cpu/complex | `cholesky` | c64 | 8 | `448x448` | 1.638 ± 0.038 | - | 887.000 ± 22.003 | 1.169 ± 0.017 | 2.789 ± 0.101 | 2.003 ± 0.059 | - |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 26.070 ± 0.115 | - | 103.615 ± 2.444 | 25.210 ± 0.136 | 47.678 ± 0.566 | 96.392 ± 1.316 | 43.635 ± 0.484 |
| cpu/complex | `conj` | c64 | 8 | `16777216` | 16.850 ± 0.677 | - | 132.804 ± 4.646 | 17.409 ± 0.587 | 32.149 ± 0.781 | 96.161 ± 1.762 | 35.707 ± 6.270 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 24.477 ± 0.194 | - | 108.014 ± 4.432 | 23.728 ± 0.098 | 44.309 ± 0.750 | 113.668 ± 0.988 | 38.913 ± 2.831 |
| cpu/complex | `div` | c64 | 8 | `8388608` | 13.858 ± 0.313 | - | 137.576 ± 21.236 | 12.942 ± 0.340 | 29.935 ± 0.508 | 115.603 ± 0.851 | 26.937 ± 0.411 |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 17.671 ± 0.026 | - | 80.997 ± 4.416 | 26.675 ± 0.076 | 24.602 ± 0.335 | 21.239 ± 0.684 | - |
| cpu/complex | `dot_general` | c64 | 8 | `640x640` | 9.302 ± 0.039 | - | 98.997 ± 12.194 | 10.631 ± 0.065 | 13.390 ± 1.383 | 11.113 ± 0.230 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 17.688 ± 0.028 | - | unsupported | 27.083 ± 0.072 | 24.980 ± 0.326 | 21.142 ± 2.927 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 8 | `640x640` | 9.294 ± 0.039 | - | unsupported | 11.168 ± 0.086 | 13.438 ± 0.548 | 11.171 ± 0.553 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 9.604 ± 0.027 | - | 5418.995 ± 32.002 | 13.472 ± 0.071 | 16.503 ± 0.027 | 16.580 ± 0.048 | - |
| cpu/complex | `eig` | c64 | 8 | `112x112` | 9.864 ± 0.056 | - | 9879.998 ± 56.991 | 12.951 ± 0.020 | 16.637 ± 0.057 | 16.629 ± 0.051 | - |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 26.314 ± 0.078 | - | 119.999 ± 0.009 | 33.899 ± 0.271 | 53.883 ± 0.164 | 80.564 ± 1.098 | 26.709 ± 2.530 |
| cpu/complex | `exp` | c64 | 8 | `4194304` | 13.234 ± 0.063 | - | 168.000 ± 1.004 | 17.632 ± 0.745 | 36.449 ± 0.868 | 81.693 ± 0.155 | 16.778 ± 4.606 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 33.871 ± 0.139 | - | 155.998 ± 0.016 | 44.476 ± 0.290 | 51.789 ± 3.222 | 153.555 ± 0.854 | 44.962 ± 1.208 |
| cpu/complex | `log` | c64 | 8 | `4194304` | 17.072 ± 1.017 | - | 191.998 ± 0.002 | 23.062 ± 0.410 | 26.297 ± 0.592 | 155.812 ± 0.820 | 26.168 ± 0.796 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 15.115 ± 0.051 | - | 69.022 ± 8.801 | 15.313 ± 0.178 | 28.781 ± 1.599 | 55.097 ± 0.501 | 26.278 ± 2.789 |
| cpu/complex | `mul` | c64 | 8 | `8388608` | 11.133 ± 0.445 | - | 114.006 ± 12.494 | 10.526 ± 0.140 | 19.940 ± 0.201 | 55.648 ± 0.236 | 22.074 ± 2.465 |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 2.228 ± 0.031 | - | 59.997 ± 0.003 | 6.386 ± 0.015 | 3.981 ± 0.095 | 6.360 ± 0.011 | - |
| cpu/complex | `norm_fro` | c64 | 8 | `2048x1536` | 1.211 ± 0.026 | - | 143.999 ± 0.007 | 3.239 ± 0.028 | 3.300 ± 0.062 | 6.368 ± 0.012 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 3.465 ± 0.014 | - | 528.044 ± 1.028 | 5.493 ± 0.028 | 6.995 ± 0.035 | 7.785 ± 0.051 | - |
| cpu/complex | `qr` | c64 | 8 | `256x256` | 2.518 ± 0.028 | - | 1213.997 ± 10.992 | 4.277 ± 0.062 | 6.312 ± 0.037 | 6.402 ± 0.087 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 2.641 ± 0.010 | - | 792.012 ± 0.992 | 2.772 ± 0.018 | 3.608 ± 0.025 | 3.185 ± 0.093 | - |
| cpu/complex | `solve` | c64 | 8 | `384x384,rhs=8` | 2.035 ± 0.024 | - | 1609.001 ± 0.009 | 2.055 ± 0.032 | 2.842 ± 0.103 | 2.074 ± 0.019 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 8.486 ± 0.044 | - | 8.481 ± 0.029 | 4.914 ± 0.030 | 5.377 ± 0.020 | 5.375 ± 0.050 | - |
| cpu/complex | `svd` | c64 | 8 | `160x160` | 8.582 ± 0.067 | - | 8.590 ± 0.047 | 3.906 ± 0.056 | 5.189 ± 0.057 | 4.854 ± 0.073 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 17.676 ± 0.031 | - | 79.409 ± 7.429 | 26.734 ± 0.056 | 24.656 ± 0.259 | 20.566 ± 0.023 | - |
| cpu/complex | `tensordot` | c64 | 8 | `640x640` | 9.298 ± 0.096 | - | 99.998 ± 17.987 | 10.687 ± 0.066 | 13.655 ± 1.731 | 10.641 ± 0.530 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 17.478 ± 0.020 | - | unsupported | 21.874 ± 0.095 | 19.299 ± 0.173 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 8 | `1024x1024` | 8.992 ± 0.032 | - | unsupported | 12.159 ± 0.135 | 11.382 ± 0.857 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 24.932 ± 0.297 | - | 104.526 ± 2.591 | 24.232 ± 0.244 | 49.777 ± 0.165 | 94.541 ± 0.523 | 53.484 ± 1.928 |
| cpu/elementwise_reduction | `abs` | f64 | 8 | `33554432` | 16.235 ± 0.533 | - | 136.099 ± 6.387 | 16.748 ± 0.440 | 33.329 ± 0.750 | 98.895 ± 0.683 | 43.963 ± 2.863 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 28.315 ± 0.413 | - | 104.885 ± 4.473 | 28.755 ± 0.556 | 55.778 ± 0.907 | 100.985 ± 2.126 | 59.462 ± 1.191 |
| cpu/elementwise_reduction | `add` | f64 | 8 | `33554432` | 20.698 ± 0.305 | - | 137.882 ± 7.170 | 21.271 ± 0.526 | 39.640 ± 0.823 | 102.308 ± 0.978 | 50.523 ± 0.586 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 6.943 ± 0.078 | - | 119.999 ± 0.003 | 20.712 ± 0.367 | 29.231 ± 0.110 | 84.702 ± 0.988 | 23.497 ± 7.191 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 8 | `4194304` | 5.973 ± 0.175 | - | 264.000 ± 0.007 | 15.559 ± 0.542 | 19.803 ± 0.192 | 84.968 ± 1.556 | 18.440 ± 12.532 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 5.708 ± 0.022 | - | 23.998 ± 0.010 | 8.591 ± 0.163 | 16.965 ± 0.131 | 29.494 ± 6.133 | 18.959 ± 8.114 |
| cpu/elementwise_reduction | `clamp` | f64 | 8 | `8388608` | 6.108 ± 0.068 | - | 71.999 ± 0.002 | 6.465 ± 0.195 | 12.248 ± 0.230 | 33.495 ± 9.036 | 16.753 ± 12.198 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 11.452 ± 0.031 | - | 35.999 ± 0.003 | 12.620 ± 0.153 | 18.049 ± 0.050 | 16.394 ± 0.133 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 8 | `33554432` | 11.694 ± 0.102 | - | 71.999 ± 0.010 | 12.284 ± 0.105 | 13.977 ± 0.365 | 16.504 ± 0.135 | - |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 6.178 ± 0.032 | - | 48.000 ± 0.004 | 17.493 ± 0.231 | 76.685 ± 0.210 | 73.356 ± 6.653 | 20.463 ± 7.583 |
| cpu/elementwise_reduction | `cos` | f64 | 8 | `8388608` | 3.415 ± 0.163 | - | 95.999 ± 0.002 | 8.662 ± 0.708 | 51.644 ± 0.154 | 74.413 ± 10.194 | 22.060 ± 12.876 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 27.995 ± 0.219 | - | 105.274 ± 1.670 | 27.946 ± 0.439 | 56.662 ± 0.419 | 104.997 ± 0.893 | 59.862 ± 2.090 |
| cpu/elementwise_reduction | `div` | f64 | 8 | `33554432` | 20.482 ± 0.788 | - | 150.589 ± 22.106 | 20.874 ± 0.605 | 38.743 ± 0.405 | 106.599 ± 1.077 | 49.453 ± 1.011 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 3.576 ± 0.021 | - | 35.999 ± 0.004 | 12.312 ± 0.376 | 17.660 ± 0.339 | 65.933 ± 6.071 | 24.699 ± 8.142 |
| cpu/elementwise_reduction | `exp` | f64 | 8 | `8388608` | 3.535 ± 0.275 | - | 71.999 ± 0.004 | 6.507 ± 0.251 | 12.454 ± 0.317 | 71.237 ± 9.443 | 17.482 ± 11.107 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 4.967 ± 0.013 | - | 35.999 ± 0.005 | 7.299 ± 0.269 | 12.784 ± 0.100 | 37.303 ± 1.040 | 10.849 ± 8.974 |
| cpu/elementwise_reduction | `expm1` | f64 | 8 | `4194304` | 2.531 ± 0.034 | - | 71.999 ± 0.005 | 3.957 ± 0.279 | 8.638 ± 0.356 | 38.071 ± 1.517 | 5.935 ± 9.063 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 6.232 ± 0.042 | - | 47.999 ± 0.003 | 14.784 ± 0.132 | 34.328 ± 0.906 | 77.407 ± 9.219 | 21.967 ± 11.278 |
| cpu/elementwise_reduction | `log` | f64 | 8 | `8388608` | 3.477 ± 0.260 | - | 95.999 ± 0.004 | 7.772 ± 0.232 | 23.247 ± 0.348 | 85.876 ± 8.124 | 13.201 ± 12.050 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.684 ± 0.014 | - | 35.997 ± 0.013 | 6.580 ± 0.243 | 18.439 ± 0.487 | 45.528 ± 0.720 | 12.767 ± 7.576 |
| cpu/elementwise_reduction | `log1p` | f64 | 8 | `4194304` | 1.664 ± 0.021 | - | 71.999 ± 0.004 | 3.582 ± 0.105 | 12.843 ± 0.218 | 46.221 ± 0.787 | 6.748 ± 8.189 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.161 ± 0.002 | 119.999 ± 0.688 | unsupported | unsupported | 0.188 ± 0.004 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 8 | `1024x64` | 0.133 ± 0.007 | 252.775 ± 2.375 | unsupported | unsupported | 0.258 ± 0.011 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 29.106 ± 0.290 | - | 116.851 ± 1.231 | 28.610 ± 0.587 | 56.677 ± 0.368 | 105.853 ± 1.228 | 60.684 ± 0.831 |
| cpu/elementwise_reduction | `maximum` | f64 | 8 | `33554432` | 20.641 ± 0.918 | - | 159.925 ± 9.597 | 20.994 ± 0.920 | 38.860 ± 0.423 | 110.986 ± 1.599 | 47.325 ± 4.496 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 29.110 ± 0.330 | - | 117.359 ± 7.440 | 28.630 ± 0.435 | 57.103 ± 0.324 | 105.444 ± 0.442 | 59.634 ± 2.538 |
| cpu/elementwise_reduction | `minimum` | f64 | 8 | `33554432` | 20.756 ± 1.053 | - | 161.106 ± 4.405 | 21.046 ± 0.341 | 38.876 ± 0.677 | 104.509 ± 1.212 | 47.145 ± 6.216 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 28.248 ± 0.253 | - | 105.544 ± 2.587 | 28.294 ± 0.220 | 56.477 ± 0.435 | 100.252 ± 1.833 | 57.553 ± 3.381 |
| cpu/elementwise_reduction | `mul` | f64 | 8 | `33554432` | 20.635 ± 0.596 | - | 135.756 ± 16.368 | 21.464 ± 0.565 | 40.039 ± 0.352 | 109.266 ± 0.995 | 48.777 ± 2.095 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 24.876 ± 0.211 | - | 105.113 ± 3.104 | 24.352 ± 0.118 | 50.328 ± 0.257 | 94.333 ± 2.462 | 53.359 ± 2.335 |
| cpu/elementwise_reduction | `neg` | f64 | 8 | `33554432` | 16.241 ± 0.602 | - | 135.785 ± 6.671 | 16.442 ± 0.329 | 33.206 ± 0.997 | 93.149 ± 3.352 | 44.551 ± 3.878 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 18.637 ± 0.198 | - | 95.999 ± 0.003 | 13.610 ± 0.134 | 43.127 ± 0.028 | 183.748 ± 0.780 | 33.022 ± 6.618 |
| cpu/elementwise_reduction | `pow` | f64 | 8 | `4194304` | 9.403 ± 0.060 | - | 143.999 ± 0.007 | 6.934 ± 0.206 | 28.822 ± 0.371 | 183.330 ± 1.360 | 16.983 ± 11.927 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 5.836 ± 0.035 | - | 35.999 ± 0.003 | 5.345 ± 0.033 | 50.313 ± 0.092 | 15.659 ± 0.182 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 8 | `8192x4096` | 5.431 ± 0.099 | - | 71.999 ± 0.005 | 5.437 ± 0.106 | 33.789 ± 0.059 | 15.071 ± 0.133 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.754 ± 0.015 | - | 24.000 ± 0.004 | 11.571 ± 0.053 | 1.496 ± 0.220 | 1.857 ± 0.275 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 8 | `2048x2048` | 0.654 ± 0.137 | - | 71.999 ± 0.010 | 7.835 ± 0.411 | 1.148 ± 0.378 | 2.050 ± 0.132 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.701 ± 0.017 | - | 23.999 ± 0.003 | 1.107 ± 0.015 | 3.316 ± 0.071 | 0.997 ± 0.027 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 8 | `2048x2048` | 0.705 ± 0.021 | - | 71.999 ± 0.002 | 0.709 ± 0.035 | 2.386 ± 0.049 | 0.980 ± 0.025 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 5.337 ± 0.055 | - | 35.999 ± 0.004 | 5.321 ± 0.013 | 50.284 ± 0.103 | 16.651 ± 0.039 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 8 | `8192x4096` | 5.431 ± 0.071 | - | 71.999 ± 0.004 | 5.402 ± 0.075 | 33.782 ± 0.063 | 16.506 ± 0.075 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.712 ± 0.053 | - | 24.000 ± 0.006 | 11.321 ± 0.044 | 1.472 ± 0.156 | 1.923 ± 0.114 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 8 | `2048x2048` | 0.806 ± 0.390 | - | 71.999 ± 0.008 | 7.730 ± 0.164 | 1.127 ± 0.277 | 1.938 ± 0.125 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.268 ± 0.037 | - | 23.998 ± 0.004 | 4.356 ± 0.010 | 13.180 ± 0.057 | 5.329 ± 0.037 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 8 | `4096x4096` | 2.779 ± 0.011 | - | 71.999 ± 0.008 | 2.723 ± 0.161 | 8.847 ± 0.036 | 5.341 ± 0.045 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 5.313 ± 0.175 | - | 23.999 ± 0.012 | 5.291 ± 0.021 | 24.085 ± 0.083 | 7.336 ± 0.041 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 8 | `8192x4096` | 5.417 ± 0.025 | - | 71.999 ± 0.004 | 5.460 ± 0.036 | 16.260 ± 0.019 | 7.434 ± 0.233 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.409 ± 0.012 | - | 23.999 ± 0.004 | 1.582 ± 0.098 | 1.269 ± 0.138 | 0.807 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 8 | `2048x2048` | 0.526 ± 0.263 | - | 71.999 ± 0.003 | 1.746 ± 0.065 | 0.996 ± 0.100 | 0.917 ± 0.115 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.649 ± 0.137 | - | 23.999 ± 0.003 | 0.424 ± 0.014 | 0.949 ± 0.108 | 0.806 ± 0.108 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 8 | `2048x2048` | 0.737 ± 0.314 | - | 71.999 ± 0.003 | 0.543 ± 0.010 | 0.883 ± 0.459 | 0.924 ± 0.092 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.334 ± 0.032 | - | 23.999 ± 0.003 | 5.329 ± 0.114 | 24.078 ± 0.063 | 7.099 ± 0.036 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 8 | `8192x4096` | 5.421 ± 0.049 | - | 72.000 ± 0.027 | 5.464 ± 0.034 | 16.252 ± 0.033 | 7.319 ± 0.139 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.613 ± 0.144 | - | 23.999 ± 0.002 | 1.584 ± 0.062 | 1.209 ± 0.189 | 0.810 ± 0.046 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 8 | `2048x2048` | 0.566 ± 0.490 | - | 72.000 ± 0.003 | 1.732 ± 0.065 | 0.985 ± 0.082 | 0.883 ± 0.127 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.651 ± 0.129 | - | 24.000 ± 3.972 | 0.420 ± 0.013 | 1.006 ± 0.198 | 0.870 ± 0.027 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 8 | `2048x2048` | 0.717 ± 0.305 | - | 71.999 ± 0.003 | 0.545 ± 0.073 | 0.889 ± 0.464 | 0.913 ± 0.134 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.393 ± 0.004 | 23.999 ± 0.001 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 8 | `2048x2048` | 0.512 ± 0.027 | 72.061 ± 0.313 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 13.881 ± 4.373 | - | 71.999 ± 0.006 | 16.622 ± 0.261 | 37.813 ± 0.686 | 79.234 ± 7.030 | 28.511 ± 6.650 |
| cpu/elementwise_reduction | `rem` | f64 | 8 | `8388608` | 7.210 ± 0.139 | - | 119.997 ± 5.072 | 8.685 ± 0.443 | 26.155 ± 0.538 | 78.367 ± 7.112 | 20.302 ± 11.535 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 33.066 ± 0.358 | - | 140.075 ± 3.327 | 32.209 ± 0.341 | 103.805 ± 0.130 | 278.435 ± 4.238 | 81.070 ± 2.215 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 8 | `33554432` | 18.054 ± 0.381 | - | 184.990 ± 5.283 | 19.127 ± 0.307 | 68.582 ± 0.605 | 279.463 ± 1.718 | 54.592 ± 2.244 |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 29.923 ± 0.280 | - | 108.128 ± 2.122 | 34.114 ± 0.253 | 56.924 ± 1.192 | 104.525 ± 0.783 | - |
| cpu/elementwise_reduction | `select` | f64 | 8 | `33554432` | 22.043 ± 0.553 | - | 159.257 ± 18.100 | 23.054 ± 0.393 | 46.101 ± 0.295 | 104.015 ± 1.559 | - |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 25.561 ± 0.153 | - | 104.651 ± 2.415 | 26.168 ± 0.213 | 50.422 ± 0.588 | 99.070 ± 0.795 | 54.427 ± 2.787 |
| cpu/elementwise_reduction | `sign` | f64 | 8 | `33554432` | 16.490 ± 0.462 | - | 135.236 ± 5.560 | 17.224 ± 0.436 | 35.091 ± 1.103 | 96.289 ± 3.784 | 45.209 ± 2.021 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 4.923 ± 0.048 | - | 35.999 ± 0.002 | 18.079 ± 2.193 | 75.665 ± 0.688 | 69.784 ± 8.831 | 19.319 ± 10.449 |
| cpu/elementwise_reduction | `sin` | f64 | 8 | `8388608` | 3.464 ± 0.055 | - | 71.997 ± 0.011 | 8.956 ± 0.177 | 51.056 ± 0.318 | 70.684 ± 10.985 | 16.445 ± 13.936 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 28.423 ± 0.172 | - | 119.605 ± 2.539 | 47.982 ± 0.199 | 58.999 ± 0.130 | 209.527 ± 2.027 | 65.008 ± 2.539 |
| cpu/elementwise_reduction | `sqrt` | f64 | 8 | `33554432` | 16.437 ± 0.434 | - | 158.586 ± 6.519 | 25.833 ± 0.587 | 39.240 ± 0.133 | 216.681 ± 1.322 | 44.221 ± 1.853 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 28.391 ± 0.316 | - | 106.268 ± 2.525 | 28.691 ± 0.222 | 56.371 ± 0.542 | 101.003 ± 2.854 | 59.021 ± 2.119 |
| cpu/elementwise_reduction | `sub` | f64 | 8 | `33554432` | 20.493 ± 0.177 | - | 135.574 ± 11.549 | 21.603 ± 0.671 | 40.121 ± 0.221 | 104.543 ± 1.323 | 48.535 ± 2.831 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 12.124 ± 0.055 | - | 71.998 ± 0.004 | 24.607 ± 0.261 | 16.771 ± 0.739 | 81.998 ± 6.548 | 23.865 ± 9.079 |
| cpu/elementwise_reduction | `tanh` | f64 | 8 | `8388608` | 6.105 ± 0.016 | - | 119.999 ± 0.003 | 12.114 ± 0.496 | 12.294 ± 0.128 | 90.252 ± 8.400 | 18.968 ± 10.403 |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 28.809 ± 0.120 | 107.012 ± 4.919 | unsupported | 34.385 ± 0.507 | 56.271 ± 0.546 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 8 | `33554432` | 21.217 ± 0.867 | 138.897 ± 7.805 | unsupported | 22.977 ± 0.320 | 47.676 ± 0.238 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.592 ± 0.136 | - | 11.999 ± 0.002 | 0.490 ± 0.012 | 0.784 ± 0.065 | 5.310 ± 2.754 | - |
| cpu/indexing_layout | `concatenate` | f64 | 8 | `1048576+1048576` | 0.667 ± 0.060 | - | 24.000 ± 0.006 | 0.630 ± 0.012 | 0.885 ± 0.037 | 3.358 ± 5.505 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.556 ± 0.135 | - | 11.999 ± 0.003 | 0.486 ± 0.012 | 0.650 ± 0.079 | 1.101 ± 0.084 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 8 | `4194304 -> 2097152` | 0.819 ± 0.307 | - | 23.998 ± 0.009 | 0.677 ± 0.011 | 0.917 ± 0.072 | 6.071 ± 1.971 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.660 ± 0.295 | - | unsupported | 0.704 ± 0.041 | 1.027 ± 0.176 | 6.586 ± 2.837 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 8 | `2097152` | 0.873 ± 0.108 | - | unsupported | 0.916 ± 0.066 | 1.545 ± 0.031 | 1.659 ± 0.045 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.180 ± 0.007 | - | 47.999 ± 0.012 | 0.213 ± 0.007 | 0.214 ± 0.041 | 1.013 ± 0.170 | - |
| cpu/indexing_layout | `gather` | f64 | 8 | `262144` | 0.097 ± 0.002 | - | 143.999 ± 0.993 | 0.114 ± 0.004 | 0.187 ± 0.066 | 1.021 ± 0.044 | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.509 ± 0.021 | - | 12.000 ± 0.005 | 0.616 ± 0.018 | 0.679 ± 0.129 | - | - |
| cpu/indexing_layout | `pad` | f64 | 8 | `2097152` | 0.709 ± 0.023 | - | 23.998 ± 0.030 | 0.958 ± 0.024 | 0.953 ± 0.257 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.526 ± 0.131 | - | 11.999 ± 0.005 | 0.478 ± 0.019 | 0.766 ± 0.194 | 4.239 ± 5.568 | - |
| cpu/indexing_layout | `reverse` | f64 | 8 | `2097152` | 0.694 ± 0.075 | - | 24.000 ± 0.014 | 0.617 ± 0.013 | 0.892 ± 0.203 | 6.813 ± 2.921 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.987 ± 0.007 | - | 35.997 ± 0.014 | 0.383 ± 0.012 | 0.417 ± 0.074 | 1.147 ± 0.696 | - |
| cpu/indexing_layout | `scatter` | f64 | 8 | `262144` | 1.014 ± 0.015 | - | 96.007 ± 0.278 | 0.369 ± 0.009 | 0.470 ± 0.066 | 0.514 ± 0.034 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.821 ± 0.124 | - | 0.002 ± 0.000 | 0.843 ± 0.064 | 1.115 ± 0.079 | 7.316 ± 3.405 | - |
| cpu/indexing_layout | `slice` | f64 | 8 | `4194304 -> 2096128` | 1.081 ± 0.043 | - | 0.003 ± 0.000 | 1.000 ± 0.050 | 1.228 ± 0.061 | 7.587 ± 3.561 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.102 ± 0.000 | - | 0.103 ± 0.001 | 0.054 ± 0.000 | 0.111 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.184 ± 0.000 | - | 0.184 ± 0.001 | 0.083 ± 0.001 | 0.202 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.101 ± 0.000 | - | 12.000 ± 0.002 | 0.264 ± 0.004 | 0.414 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.063 ± 0.002 | - | 24.000 ± 0.001 | 0.162 ± 0.004 | 0.170 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.002 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.129 ± 0.001 | - | 11.999 ± 0.003 | 1.193 ± 0.008 | 1.216 ± 0.855 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.009 ± 0.000 | - | 12.000 ± 0.003 | 0.040 ± 0.000 | 0.040 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1024x2x2` | 0.102 ± 0.000 | - | 0.105 ± 0.001 | 0.049 ± 0.000 | 0.118 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1024x4x4` | 0.184 ± 0.001 | - | 0.189 ± 0.002 | 0.080 ± 0.001 | 0.208 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1024x8x8` | 0.053 ± 0.000 | - | 23.999 ± 0.002 | 0.252 ± 0.004 | 0.401 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1x128x128` | 0.064 ± 0.001 | - | 24.000 ± 0.001 | 0.148 ± 0.003 | 0.148 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1x32x32` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.001 | 0.015 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `3x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `8x128x128` | 0.085 ± 0.018 | - | 24.000 ± 0.002 | 1.095 ± 0.005 | 1.246 ± 0.217 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `8x32x32` | 0.006 ± 0.001 | - | 24.000 ± 0.001 | 0.041 ± 0.001 | 0.041 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `8x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `8x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.158 ± 0.003 | - | 12.000 ± 0.002 | 0.855 ± 0.003 | 1.230 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 0.781 ± 0.008 | - | 12.000 ± 0.002 | 3.028 ± 0.006 | 2.450 ± 0.143 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 2.439 ± 0.004 | - | 12.000 ± 0.003 | 9.173 ± 0.009 | 7.616 ± 0.333 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 3.190 ± 0.013 | - | 3.194 ± 0.006 | 1.905 ± 0.009 | 3.325 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.144 ± 0.000 | - | 0.146 ± 0.001 | 0.155 ± 0.001 | 0.204 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.017 ± 0.001 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.031 ± 0.000 | - | 0.033 ± 0.000 | 0.035 ± 0.000 | 0.051 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.019 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.040 ± 0.000 | - | 0.043 ± 0.000 | 0.043 ± 0.000 | 0.084 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 6.377 ± 0.025 | - | 36.000 ± 0.001 | 14.797 ± 0.036 | 31.967 ± 14.119 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.288 ± 0.005 | - | 12.000 ± 0.002 | 1.191 ± 0.005 | 1.488 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.027 ± 0.000 | - | 0.029 ± 0.001 | 0.031 ± 0.000 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.078 ± 0.000 | - | 0.081 ± 0.000 | 0.079 ± 0.000 | 0.116 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1024x2x2` | 0.082 ± 0.001 | - | 24.000 ± 0.001 | 0.853 ± 0.009 | 1.253 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1024x4x4` | 0.394 ± 0.011 | - | 24.000 ± 0.009 | 3.041 ± 0.003 | 2.489 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1024x8x8` | 1.233 ± 0.018 | - | 24.000 ± 0.004 | 9.169 ± 0.017 | 8.308 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1x128x128` | 3.209 ± 0.012 | - | 3.195 ± 0.008 | 1.587 ± 0.015 | 3.257 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1x32x32` | 0.144 ± 0.002 | - | 0.147 ± 0.001 | 0.155 ± 0.003 | 0.206 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `3x4x4` | 0.012 ± 0.000 | - | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `3x8x8` | 0.031 ± 0.000 | - | 0.033 ± 0.000 | 0.035 ± 0.000 | 0.051 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `4x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `4x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.019 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `4x8x8` | 0.040 ± 0.000 | - | 0.043 ± 0.000 | 0.044 ± 0.000 | 0.074 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `8x128x128` | 3.196 ± 0.016 | - | 48.000 ± 0.009 | 12.222 ± 0.052 | 37.985 ± 28.996 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `8x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `8x32x32` | 0.149 ± 0.002 | - | 24.000 ± 0.003 | 1.200 ± 0.006 | 1.001 ± 3.227 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `8x4x4` | 0.027 ± 0.001 | - | 0.029 ± 0.000 | 0.031 ± 0.000 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `8x8x8` | 0.078 ± 0.000 | - | 0.081 ± 0.000 | 0.079 ± 0.000 | 0.121 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.532 ± 0.007 | - | 0.537 ± 0.007 | 0.501 ± 0.006 | 0.459 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 0.673 ± 0.009 | - | 12.000 ± 0.002 | 2.609 ± 0.010 | 1.697 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 2.175 ± 0.010 | - | 11.999 ± 0.003 | 7.835 ± 0.007 | 4.595 ± 0.093 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 2.263 ± 0.008 | - | 2.259 ± 0.007 | 1.086 ± 0.012 | 1.946 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.115 ± 0.002 | - | 0.118 ± 0.001 | 0.110 ± 0.000 | 0.145 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.028 ± 0.000 | - | 0.030 ± 0.000 | 0.030 ± 0.000 | 0.039 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.018 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.036 ± 0.000 | - | 0.038 ± 0.000 | 0.038 ± 0.000 | 0.048 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 4.558 ± 0.006 | - | 23.999 ± 0.002 | 8.332 ± 0.038 | 20.865 ± 10.762 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.231 ± 0.001 | - | 12.000 ± 0.001 | 0.832 ± 0.012 | 0.400 ± 0.223 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.023 ± 0.000 | - | 0.025 ± 0.000 | 0.027 ± 0.000 | 0.032 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.070 ± 0.000 | - | 0.072 ± 0.000 | 0.068 ± 0.001 | 0.086 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1024x2x2` | 0.540 ± 0.005 | - | 0.537 ± 0.005 | 0.509 ± 0.007 | 0.450 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1024x4x4` | 0.342 ± 0.006 | - | 24.000 ± 0.002 | 2.614 ± 0.010 | 1.753 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1024x8x8` | 1.112 ± 0.015 | - | 23.999 ± 0.001 | 7.870 ± 0.029 | 3.808 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1x128x128` | 2.279 ± 0.009 | - | 2.272 ± 0.015 | 1.025 ± 0.012 | 1.997 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1x32x32` | 0.115 ± 0.002 | - | 0.123 ± 0.001 | 0.111 ± 0.003 | 0.139 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1x8x8` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `3x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `3x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `3x8x8` | 0.028 ± 0.000 | - | 0.030 ± 0.000 | 0.030 ± 0.000 | 0.039 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `4x4x4` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.018 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `4x8x8` | 0.036 ± 0.000 | - | 0.038 ± 0.000 | 0.038 ± 0.000 | 0.048 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `8x128x128` | 2.306 ± 0.017 | - | 23.999 ± 0.009 | 7.869 ± 0.033 | 16.736 ± 16.453 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `8x32x32` | 0.119 ± 0.001 | - | 23.999 ± 0.003 | 0.840 ± 0.005 | 0.363 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `8x4x4` | 0.023 ± 0.000 | - | 0.025 ± 0.000 | 0.027 ± 0.000 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `8x8x8` | 0.070 ± 0.000 | - | 0.074 ± 0.000 | 0.068 ± 0.000 | 0.101 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.362 ± 0.002 | - | 0.367 ± 0.004 | 0.263 ± 0.003 | 0.432 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.545 ± 0.010 | - | 12.000 ± 0.002 | 1.321 ± 0.005 | 1.673 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 1.745 ± 0.009 | - | 12.000 ± 0.002 | 4.120 ± 0.011 | 4.629 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 1.247 ± 0.008 | - | 1.244 ± 0.010 | 0.732 ± 0.010 | 1.896 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.086 ± 0.001 | - | 0.088 ± 0.001 | 0.051 ± 0.001 | 0.157 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.022 ± 0.000 | - | 0.025 ± 0.000 | 0.022 ± 0.001 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.014 ± 0.001 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.029 ± 0.000 | - | 0.031 ± 0.000 | 0.025 ± 0.000 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 2.504 ± 0.009 | - | 12.000 ± 0.003 | 5.599 ± 0.020 | 14.991 ± 3.598 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.174 ± 0.001 | - | 12.000 ± 0.002 | 0.339 ± 0.005 | 0.340 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.019 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.056 ± 0.000 | - | 0.058 ± 0.000 | 0.041 ± 0.000 | 0.083 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1024x2x2` | 0.358 ± 0.003 | - | 0.361 ± 0.003 | 0.265 ± 0.004 | 0.450 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1024x4x4` | 0.280 ± 0.006 | - | 24.000 ± 0.005 | 1.324 ± 0.011 | 1.739 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1024x8x8` | 0.882 ± 0.022 | - | 23.999 ± 0.002 | 4.113 ± 0.011 | 3.716 ± 0.440 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1x128x128` | 1.257 ± 0.015 | - | 1.262 ± 0.015 | 0.745 ± 0.007 | 1.998 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1x32x32` | 0.087 ± 0.001 | - | 0.090 ± 0.001 | 0.051 ± 0.000 | 0.134 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1x8x8` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `3x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `3x8x8` | 0.022 ± 0.000 | - | 0.025 ± 0.000 | 0.022 ± 0.001 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `4x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `4x8x8` | 0.029 ± 0.000 | - | 0.031 ± 0.000 | 0.025 ± 0.000 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `8x128x128` | 1.262 ± 0.010 | - | 24.000 ± 0.002 | 5.732 ± 0.037 | 36.032 ± 24.775 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `8x32x32` | 0.089 ± 0.003 | - | 24.000 ± 0.003 | 0.338 ± 0.003 | 0.381 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `8x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.019 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `8x8x8` | 0.056 ± 0.000 | - | 0.058 ± 0.001 | 0.041 ± 0.000 | 0.084 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.141 ± 0.001 | - | 0.154 ± 0.001 | 0.111 ± 0.001 | 0.144 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.309 ± 0.007 | - | 0.308 ± 0.004 | 0.165 ± 0.001 | 0.280 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.231 ± 0.003 | - | 12.000 ± 0.001 | 0.267 ± 0.004 | 0.602 ± 0.091 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.164 ± 0.007 | - | 72.000 ± 0.023 | 0.207 ± 0.001 | 0.248 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.014 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.012 ± 0.000 | - | 0.015 ± 0.000 | 0.025 ± 0.000 | 0.026 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.027 ± 0.001 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.019 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.345 ± 0.011 | - | 12.000 ± 0.002 | 0.443 ± 0.004 | 1.015 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.023 ± 0.000 | - | 11.999 ± 0.001 | 0.050 ± 0.001 | 0.154 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.020 ± 0.000 | 0.023 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1024x2x2` | 0.142 ± 0.003 | - | 0.145 ± 0.000 | 0.096 ± 0.001 | 0.164 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1024x4x4` | 0.313 ± 0.006 | - | 0.308 ± 0.028 | 0.147 ± 0.002 | 0.272 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1024x8x8` | 0.123 ± 0.001 | - | 23.999 ± 0.004 | 0.221 ± 0.001 | 0.619 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x128x128` | 0.165 ± 0.010 | - | 144.001 ± 0.012 | 0.206 ± 0.003 | 0.268 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x32x32` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.025 ± 0.001 | 0.025 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x64x24` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.027 ± 0.001 | 0.029 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `3x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `3x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `4x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `4x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.019 ± 0.000 | 0.020 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `8x128x128` | 0.182 ± 0.013 | - | 23.999 ± 0.009 | 0.309 ± 0.003 | 0.695 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `8x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `8x32x32` | 0.014 ± 0.000 | - | 23.999 ± 0.003 | 0.042 ± 0.001 | 0.117 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `8x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.020 ± 0.001 | 0.023 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.310 ± 0.003 | - | 0.314 ± 0.003 | 0.206 ± 0.001 | 0.347 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.207 ± 0.005 | - | 12.000 ± 0.002 | 0.627 ± 0.004 | 0.974 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.629 ± 0.006 | - | 12.000 ± 0.004 | 1.629 ± 0.007 | 1.237 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.549 ± 0.008 | - | 0.549 ± 0.005 | 0.491 ± 0.008 | 0.843 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.036 ± 0.000 | - | 0.038 ± 0.000 | 0.031 ± 0.000 | 0.040 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.035 ± 0.000 | - | 0.037 ± 0.000 | 0.039 ± 0.000 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 1.093 ± 0.014 | - | 12.000 ± 0.004 | 3.592 ± 0.011 | 6.925 ± 0.526 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.071 ± 0.000 | - | 12.000 ± 0.001 | 0.202 ± 0.004 | 0.203 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.022 ± 0.000 | - | 0.025 ± 0.000 | 0.020 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1024x2x2` | 0.312 ± 0.005 | - | 0.313 ± 0.004 | 0.201 ± 0.001 | 0.352 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1024x4x4` | 0.107 ± 0.001 | - | 24.000 ± 0.002 | 0.613 ± 0.003 | 0.982 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1024x8x8` | 0.319 ± 0.005 | - | 24.000 ± 0.002 | 1.627 ± 0.012 | 0.939 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x128x128` | 0.554 ± 0.015 | - | 0.553 ± 0.031 | 0.433 ± 0.014 | 0.998 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x32x32` | 0.036 ± 0.000 | - | 0.039 ± 0.000 | 0.035 ± 0.001 | 0.039 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x64x24` | 0.035 ± 0.000 | - | 0.037 ± 0.000 | 0.040 ± 0.000 | 0.037 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `3x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `3x8x8` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `4x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `8x128x128` | 0.593 ± 0.023 | - | 23.998 ± 0.008 | 3.208 ± 0.022 | 8.421 ± 10.929 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `8x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `8x32x32` | 0.039 ± 0.000 | - | 24.000 ± 0.004 | 0.205 ± 0.004 | 0.186 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `8x4x4` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.011 ± 0.001 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `8x8x8` | 0.022 ± 0.000 | - | 0.025 ± 0.000 | 0.020 ± 0.001 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.202 ± 0.001 | - | 0.161 ± 0.003 | 0.083 ± 0.001 | 0.369 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.082 ± 0.000 | - | 23.999 ± 0.004 | 0.116 ± 0.001 | 0.500 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.206 ± 0.001 | - | 35.999 ± 0.002 | 0.332 ± 0.007 | 0.887 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.107 ± 0.004 | - | 84.000 ± 0.005 | 0.177 ± 0.002 | 0.225 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.010 ± 0.000 | - | 0.017 ± 0.000 | 0.028 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.012 ± 0.000 | 0.020 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.013 ± 0.000 | 0.020 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.233 ± 0.007 | - | 36.000 ± 0.003 | 0.398 ± 0.004 | 0.842 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.019 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.020 ± 0.001 | - | 12.000 ± 0.002 | 0.049 ± 0.001 | 0.115 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.005 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.000 | 0.016 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.009 ± 0.000 | - | 0.016 ± 0.000 | 0.022 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1024x2x2,rhs=1` | 0.204 ± 0.002 | - | 0.160 ± 0.004 | 0.079 ± 0.001 | 0.384 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1024x4x4,rhs=1` | 0.044 ± 0.000 | - | 48.000 ± 0.003 | 0.110 ± 0.001 | 0.528 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1024x8x8,rhs=1` | 0.112 ± 0.005 | - | 96.000 ± 0.008 | 0.298 ± 0.004 | 0.957 ± 0.073 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1x128x128,rhs=1` | 0.110 ± 0.005 | - | 168.003 ± 2.998 | 0.179 ± 0.002 | 0.226 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1x32x32,rhs=1` | 0.011 ± 0.000 | - | 0.017 ± 0.000 | 0.028 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `3x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.012 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `3x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `3x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.012 ± 0.000 | 0.021 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `4x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `4x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.019 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `4x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.013 ± 0.000 | 0.021 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `8x128x128,rhs=1` | 0.128 ± 0.002 | - | 96.000 ± 0.007 | 0.281 ± 0.002 | 0.605 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `8x2x2,rhs=1` | 0.005 ± 0.000 | - | 0.011 ± 0.000 | 0.019 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `8x32x32,rhs=1` | 0.012 ± 0.001 | - | 24.000 ± 0.002 | 0.042 ± 0.001 | 0.145 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `8x4x4,rhs=1` | 0.006 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `8x8x8,rhs=1` | 0.009 ± 0.000 | - | 0.016 ± 0.000 | 0.023 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 0.402 ± 0.004 | - | 12.000 ± 0.002 | 1.957 ± 0.005 | 2.567 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 1.378 ± 0.005 | - | 12.000 ± 0.003 | 6.994 ± 0.018 | 2.730 ± 0.541 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 4.233 ± 0.009 | - | 23.999 ± 0.001 | 21.816 ± 0.071 | 10.324 ± 2.579 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 5.537 ± 0.030 | - | 5.541 ± 0.014 | 4.536 ± 0.022 | 5.184 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.249 ± 0.003 | - | 0.253 ± 0.003 | 0.299 ± 0.004 | 0.266 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.001 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.203 ± 0.001 | - | 0.206 ± 0.004 | 0.258 ± 0.003 | 0.286 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.031 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.029 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.051 ± 0.000 | - | 0.053 ± 0.001 | 0.073 ± 0.001 | 0.075 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.016 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.024 ± 0.000 | - | 0.027 ± 0.000 | 0.036 ± 0.000 | 0.042 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.066 ± 0.000 | - | 0.070 ± 0.000 | 0.095 ± 0.000 | 0.097 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 11.067 ± 0.025 | - | 48.000 ± 0.007 | 35.841 ± 0.096 | 100.747 ± 19.302 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.024 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 0.580 ± 0.005 | - | 11.999 ± 0.004 | 2.304 ± 0.006 | 0.667 ± 0.455 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.045 ± 0.000 | - | 0.047 ± 0.000 | 0.063 ± 0.000 | 0.069 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.131 ± 0.000 | - | 0.133 ± 0.001 | 0.180 ± 0.001 | 0.183 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1024x2x2` | 0.204 ± 0.000 | - | 23.999 ± 0.006 | 1.952 ± 0.012 | 2.575 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1024x4x4` | 0.672 ± 0.015 | - | 23.999 ± 0.001 | 7.032 ± 0.023 | 2.278 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1024x8x8` | 2.125 ± 0.010 | - | 23.999 ± 0.007 | 21.962 ± 0.059 | 5.338 ± 0.095 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x128x128` | 5.564 ± 0.019 | - | 5.552 ± 0.022 | 4.000 ± 0.028 | 5.122 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x32x32` | 0.250 ± 0.003 | - | 0.254 ± 0.001 | 0.302 ± 0.007 | 0.289 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x4x4` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.016 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x64x24` | 0.205 ± 0.002 | - | 0.207 ± 0.004 | 0.259 ± 0.004 | 0.283 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.031 ± 0.001 | 0.037 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `3x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.015 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `3x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.029 ± 0.000 | 0.035 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `3x8x8` | 0.051 ± 0.000 | - | 0.054 ± 0.000 | 0.074 ± 0.001 | 0.077 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `4x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.016 ± 0.001 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `4x4x4` | 0.024 ± 0.000 | - | 0.027 ± 0.000 | 0.036 ± 0.000 | 0.042 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `4x8x8` | 0.067 ± 0.000 | - | 0.070 ± 0.001 | 0.096 ± 0.001 | 0.097 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `8x128x128` | 6.256 ± 0.021 | - | 48.000 ± 0.022 | 31.627 ± 0.102 | 139.985 ± 81.326 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `8x2x2` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.024 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `8x32x32` | 0.255 ± 0.004 | - | 24.000 ± 0.003 | 2.318 ± 0.008 | 0.348 ± 0.090 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `8x4x4` | 0.045 ± 0.000 | - | 0.048 ± 0.000 | 0.063 ± 0.001 | 0.069 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `8x8x8` | 0.131 ± 0.001 | - | 0.134 ± 0.001 | 0.180 ± 0.001 | 0.191 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.248 ± 0.008 | - | 1.252 ± 0.014 | 1.365 ± 0.007 | 1.601 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 1.151 ± 0.004 | - | 12.000 ± 0.004 | 5.920 ± 0.022 | 2.159 ± 0.463 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 3.721 ± 0.004 | - | 23.999 ± 0.002 | 19.156 ± 0.049 | 8.383 ± 0.767 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 3.091 ± 0.009 | - | 3.079 ± 0.018 | 2.761 ± 0.039 | 3.149 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.186 ± 0.003 | - | 0.188 ± 0.002 | 0.207 ± 0.003 | 0.204 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.129 ± 0.002 | - | 0.130 ± 0.002 | 0.160 ± 0.004 | 0.211 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.027 ± 0.000 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.026 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.045 ± 0.000 | - | 0.047 ± 0.000 | 0.060 ± 0.000 | 0.062 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.022 ± 0.000 | - | 0.024 ± 0.000 | 0.032 ± 0.000 | 0.037 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.059 ± 0.000 | - | 0.061 ± 0.000 | 0.078 ± 0.000 | 0.078 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 6.199 ± 0.019 | - | 36.000 ± 0.002 | 21.271 ± 0.043 | 33.480 ± 10.296 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.019 ± 0.000 | 0.028 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.367 ± 0.005 | - | 11.999 ± 0.004 | 1.592 ± 0.003 | 0.804 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.039 ± 0.000 | - | 0.041 ± 0.000 | 0.054 ± 0.000 | 0.057 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.117 ± 0.001 | - | 0.119 ± 0.001 | 0.150 ± 0.000 | 0.145 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1024x2x2` | 1.257 ± 0.021 | - | 1.254 ± 0.014 | 1.367 ± 0.011 | 1.592 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1024x4x4` | 0.582 ± 0.010 | - | 24.000 ± 0.002 | 5.948 ± 0.016 | 1.747 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1024x8x8` | 1.881 ± 0.010 | - | 24.000 ± 0.003 | 19.201 ± 0.033 | 4.273 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x128x128` | 3.110 ± 0.014 | - | 3.109 ± 0.029 | 2.630 ± 0.026 | 3.150 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x32x32` | 0.188 ± 0.002 | - | 0.189 ± 0.004 | 0.211 ± 0.004 | 0.213 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.015 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x64x24` | 0.130 ± 0.003 | - | 0.132 ± 0.002 | 0.160 ± 0.002 | 0.189 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x8x8` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.027 ± 0.001 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `3x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `3x4x4` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.026 ± 0.000 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `3x8x8` | 0.045 ± 0.000 | - | 0.047 ± 0.000 | 0.060 ± 0.001 | 0.060 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `4x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `4x4x4` | 0.022 ± 0.000 | - | 0.025 ± 0.000 | 0.032 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `4x8x8` | 0.059 ± 0.000 | - | 0.062 ± 0.000 | 0.078 ± 0.000 | 0.077 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `8x128x128` | 3.228 ± 0.017 | - | 47.999 ± 0.007 | 20.482 ± 0.058 | 30.448 ± 31.705 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `8x2x2` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.020 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `8x32x32` | 0.188 ± 0.003 | - | 23.999 ± 0.002 | 1.618 ± 0.006 | 0.290 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `8x4x4` | 0.040 ± 0.000 | - | 0.042 ± 0.000 | 0.055 ± 0.000 | 0.056 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `8x8x8` | 0.118 ± 0.000 | - | 0.120 ± 0.001 | 0.151 ± 0.001 | 0.141 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.708 ± 0.008 | - | unsupported | 0.393 ± 0.005 | 0.861 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 0.697 ± 0.006 | - | unsupported | 2.596 ± 0.013 | 1.092 ± 0.095 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 1.959 ± 0.007 | - | unsupported | 6.938 ± 0.014 | 4.373 ± 0.127 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.749 ± 0.007 | - | unsupported | 1.497 ± 0.032 | 1.590 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.084 ± 0.002 | - | unsupported | 0.076 ± 0.001 | 0.088 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.064 ± 0.002 | - | unsupported | 0.062 ± 0.001 | 0.075 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.010 ± 0.001 | - | unsupported | 0.014 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.010 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.025 ± 0.000 | - | unsupported | 0.029 ± 0.001 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.013 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.033 ± 0.000 | - | unsupported | 0.035 ± 0.000 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 3.506 ± 0.009 | - | unsupported | 11.703 ± 0.041 | 13.906 ± 1.219 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.008 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.165 ± 0.003 | - | unsupported | 0.558 ± 0.005 | 0.223 ± 0.173 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.023 ± 0.000 | - | unsupported | 0.027 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.062 ± 0.000 | - | unsupported | 0.062 ± 0.000 | 0.076 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1024x2x2` | 0.689 ± 0.013 | - | unsupported | 0.389 ± 0.003 | 0.845 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1024x4x4` | 0.354 ± 0.003 | - | unsupported | 2.598 ± 0.016 | 1.147 ± 0.073 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1024x8x8` | 0.979 ± 0.015 | - | unsupported | 6.945 ± 0.014 | 2.258 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x128x128` | 1.777 ± 0.013 | - | unsupported | 1.500 ± 0.013 | 1.604 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x2x2` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x32x32` | 0.084 ± 0.002 | - | unsupported | 0.076 ± 0.001 | 0.088 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x4x4` | 0.005 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x64x24` | 0.064 ± 0.002 | - | unsupported | 0.063 ± 0.001 | 0.075 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x8x8` | 0.011 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `3x2x2` | 0.005 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `3x4x4` | 0.010 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `3x8x8` | 0.025 ± 0.000 | - | unsupported | 0.028 ± 0.000 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `4x2x2` | 0.005 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `4x4x4` | 0.013 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `4x8x8` | 0.033 ± 0.000 | - | unsupported | 0.035 ± 0.000 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `8x128x128` | 1.772 ± 0.008 | - | unsupported | 11.556 ± 0.037 | 15.941 ± 21.140 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `8x2x2` | 0.008 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `8x32x32` | 0.083 ± 0.001 | - | unsupported | 0.559 ± 0.004 | 0.201 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `8x4x4` | 0.023 ± 0.000 | - | unsupported | 0.028 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `8x8x8` | 0.062 ± 0.000 | - | unsupported | 0.062 ± 0.000 | 0.076 ± 0.001 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.472 ± 0.014 | - | unsupported | 0.373 ± 0.004 | 1.122 ± 0.338 | 2.633 ± 0.611 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.115 ± 0.001 | - | unsupported | 0.042 ± 0.003 | 0.129 ± 0.033 | 0.155 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.208 ± 0.002 | - | unsupported | 0.057 ± 0.001 | 0.229 ± 0.032 | 0.248 ± 0.043 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.188 ± 0.006 | - | unsupported | 0.105 ± 0.001 | 0.492 ± 0.055 | 0.810 ± 0.016 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 8 | `1024x16x16,rhs=1` | 0.273 ± 0.005 | - | unsupported | 0.190 ± 0.001 | 0.884 ± 0.031 | 2.724 ± 0.027 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 8 | `1024x2x2,rhs=1` | 0.138 ± 0.018 | - | unsupported | 0.036 ± 0.001 | 0.140 ± 0.033 | 0.176 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 8 | `1024x4x4,rhs=1` | 0.209 ± 0.001 | - | unsupported | 0.052 ± 0.001 | 0.247 ± 0.030 | 0.278 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 8 | `1024x8x8,rhs=1` | 0.107 ± 0.003 | - | unsupported | 0.078 ± 0.001 | 0.533 ± 0.049 | 0.805 ± 0.193 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.244 ± 0.005 | - | unsupported | 0.502 ± 0.005 | 0.941 ± 0.063 | 0.560 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.063 ± 0.001 | - | unsupported | 0.047 ± 0.001 | 0.244 ± 0.032 | 0.110 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.078 ± 0.000 | - | unsupported | 0.061 ± 0.000 | 0.295 ± 0.033 | 0.169 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.093 ± 0.001 | - | unsupported | 0.225 ± 0.001 | 0.488 ± 0.034 | 0.283 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 8 | `1024x16x16,rhs=1` | 0.165 ± 0.002 | - | unsupported | 0.500 ± 0.003 | 0.956 ± 0.013 | 0.553 ± 0.042 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 8 | `1024x2x2,rhs=1` | 0.063 ± 0.001 | - | unsupported | 0.047 ± 0.000 | 0.272 ± 0.031 | 0.117 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 8 | `1024x4x4,rhs=1` | 0.079 ± 0.000 | - | unsupported | 0.061 ± 0.000 | 0.328 ± 0.028 | 0.169 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 8 | `1024x8x8,rhs=1` | 0.072 ± 0.001 | - | unsupported | 0.227 ± 0.001 | 0.496 ± 0.014 | 0.282 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.100 ± 0.000 | - | 12.000 ± 0.002 | 0.260 ± 0.004 | 0.472 ± 0.042 | 0.358 ± 0.006 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.046 ± 0.001 | - | 0.048 ± 0.000 | 0.083 ± 0.001 | 0.115 ± 0.001 | 0.099 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.053 ± 0.000 | - | 0.056 ± 0.000 | 0.101 ± 0.001 | 0.145 ± 0.002 | 0.120 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.040 ± 0.000 | - | 12.000 ± 0.002 | 0.151 ± 0.001 | 0.243 ± 0.019 | 0.190 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 8 | `1024x16x16,rhs=1` | 0.053 ± 0.000 | - | 23.999 ± 0.006 | 0.263 ± 0.005 | 0.470 ± 0.033 | 0.364 ± 0.031 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 8 | `1024x2x2,rhs=1` | 0.046 ± 0.000 | - | 0.048 ± 0.001 | 0.083 ± 0.002 | 0.118 ± 0.001 | 0.103 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 8 | `1024x4x4,rhs=1` | 0.053 ± 0.000 | - | 0.056 ± 0.001 | 0.102 ± 0.001 | 0.145 ± 0.003 | 0.123 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 8 | `1024x8x8,rhs=1` | 0.022 ± 0.000 | - | 24.000 ± 0.002 | 0.153 ± 0.002 | 0.241 ± 0.040 | 0.194 ± 0.004 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 14.441 ± 0.161 | - | 588.052 ± 1.032 | 25.951 ± 0.044 | 73.212 ± 6.555 | 41.789 ± 12.457 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 8 | `1536x1536` | 9.802 ± 0.064 | - | 1301.999 ± 24.999 | 13.358 ± 0.131 | 55.038 ± 18.545 | 33.066 ± 14.505 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 12.427 ± 1.142 | - | 1848.986 ± 15.014 | 8.939 ± 0.087 | 17.508 ± 0.107 | 12.748 ± 0.075 | - |
| cpu/linalg_uncovered | `det` | f64 | 8 | `1024x1024` | 10.054 ± 0.683 | - | 3763.002 ± 31.998 | 5.680 ± 0.023 | 12.160 ± 1.625 | 7.048 ± 0.682 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 10.364 ± 0.059 | - | 3775.992 ± 13.979 | 9.579 ± 0.043 | 10.467 ± 0.071 | 10.264 ± 0.036 | - |
| cpu/linalg_uncovered | `eig` | f64 | 8 | `160x160` | 10.477 ± 0.086 | - | 7248.998 ± 56.996 | 9.549 ± 0.049 | 10.545 ± 0.043 | 10.296 ± 0.028 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 12.875 ± 0.123 | - | 3151.997 ± 26.011 | 10.837 ± 0.044 | 12.871 ± 0.091 | 12.573 ± 0.031 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 8 | `192x192` | 12.960 ± 0.055 | - | 6184.014 ± 75.999 | 10.643 ± 0.035 | 12.869 ± 0.016 | 12.641 ± 0.027 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 18.708 ± 0.118 | - | 5737.978 ± 6.416 | 13.220 ± 0.024 | 33.379 ± 0.099 | 12.300 ± 0.046 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 8 | `512x512` | 18.228 ± 0.207 | - | 11316.012 ± 52.065 | 11.381 ± 0.092 | 31.055 ± 0.251 | 12.836 ± 0.618 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.252 ± 0.001 | 204.124 ± 0.187 | unsupported | 0.351 ± 0.001 | 1.089 ± 0.021 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 2.991 ± 0.004 | 3093.436 ± 6.938 | unsupported | 4.719 ± 0.013 | 8.038 ± 0.087 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 8 | `256x128` | 0.245 ± 0.001 | 410.561 ± 4.939 | unsupported | 0.349 ± 0.012 | 1.130 ± 0.030 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 8 | `768x384` | 2.414 ± 0.029 | 6264.250 ± 48.627 | unsupported | 2.796 ± 0.007 | 7.793 ± 0.213 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 13.361 ± 0.081 | - | 1584.990 ± 0.015 | 15.899 ± 0.046 | 20.748 ± 0.561 | 15.457 ± 0.070 | - |
| cpu/linalg_uncovered | `inv` | f64 | 8 | `768x768` | 8.796 ± 0.085 | - | 3212.999 ± 90.995 | 9.629 ± 0.132 | 12.680 ± 0.330 | 9.697 ± 0.158 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 7.786 ± 0.076 | 1092.004 ± 12.003 | 5.220 ± 0.019 | 40.786 ± 0.484 | 5.947 ± 0.034 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 8 | `768x384,rhs=16` | - | 6.074 ± 0.062 | 2494.997 ± 39.006 | failed | 37.608 ± 0.170 | 5.921 ± 0.159 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 12.866 ± 2.172 | - | 1777.082 ± 9.599 | 13.531 ± 0.105 | 33.288 ± 0.792 | 22.216 ± 7.683 | - |
| cpu/linalg_uncovered | `lu` | f64 | 8 | `1024x1024` | 10.187 ± 0.414 | - | 3570.013 ± 33.039 | 9.308 ± 0.196 | 26.984 ± 3.916 | 19.221 ± 12.879 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.984 ± 0.012 | - | 24.000 ± 0.003 | 1.650 ± 0.036 | 10.917 ± 0.297 | 4.275 ± 0.049 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 8 | `2048x2048` | 0.666 ± 0.397 | - | 71.999 ± 0.002 | 1.658 ± 0.015 | 9.165 ± 0.553 | 4.238 ± 0.021 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 19.749 ± 0.110 | - | 100.999 ± 0.002 | 16.596 ± 1.941 | 21.172 ± 0.727 | 19.197 ± 0.167 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 8 | `512x256` | 20.059 ± 0.059 | - | 239.001 ± 0.014 | 14.923 ± 0.039 | 21.126 ± 0.929 | 18.890 ± 1.462 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 19.758 ± 0.156 | - | 100.983 ± 3.000 | 16.592 ± 0.033 | 20.882 ± 1.251 | 19.167 ± 0.608 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 8 | `512x256` | 20.037 ± 0.055 | - | 235.995 ± 6.003 | 14.913 ± 0.108 | 21.991 ± 0.739 | 18.573 ± 0.180 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 0.849 ± 0.004 | 1293.372 ± 2.247 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 10.518 ± 0.343 | 4852.559 ± 10.851 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 8 | `256x128` | 0.848 ± 0.043 | 1296.685 ± 6.501 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 8 | `768x384` | 7.547 ± 0.164 | 9672.810 ± 31.187 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 0.844 ± 0.003 | 1302.497 ± 3.745 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 10.297 ± 0.087 | 4863.058 ± 10.399 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 8 | `256x128` | 0.864 ± 0.038 | 1300.872 ± 4.312 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 8 | `768x384` | 7.539 ± 0.021 | 9690.934 ± 22.477 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 12.087 ± 0.280 | - | 1836.991 ± 0.021 | 8.908 ± 0.031 | 17.680 ± 0.241 | 12.733 ± 0.052 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 8 | `1024x1024` | 9.890 ± 0.241 | - | 3772.994 ± 59.003 | 5.554 ± 0.088 | 12.186 ± 1.198 | 6.986 ± 0.088 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 25.762 ± 0.064 | 9487.004 ± 10.993 | 33.608 ± 0.149 | 52.651 ± 0.604 | 52.377 ± 6.184 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 8 | `768x384` | - | 19.945 ± 0.090 | 19318.999 ± 201.962 | 29.371 ± 0.114 | 47.525 ± 0.840 | 48.087 ± 3.445 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 15.163 ± 0.067 | - | 1585.012 ± 25.004 | 14.611 ± 0.038 | 147.898 ± 3.749 | 16.424 ± 0.026 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 8 | `4096x4096,rhs=64` | 11.396 ± 0.245 | - | 3169.999 ± 0.991 | 12.107 ± 0.187 | 113.962 ± 12.251 | 12.327 ± 0.059 | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.037 ± 0.001 | 24.015 ± 0.000 | unsupported | 0.055 ± 0.003 | 0.036 ± 0.001 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.531 ± 0.003 | 23.999 ± 0.001 | unsupported | 0.752 ± 0.014 | 0.875 ± 0.579 | - | - |
| cpu/neural_public_api | `erf` | f32 | 8 | `1024x64` | 0.021 ± 0.001 | 72.000 ± 0.641 | unsupported | 0.036 ± 0.038 | 0.038 ± 0.002 | - | - |
| cpu/neural_public_api | `erf` | f32 | 8 | `4096x256` | 0.274 ± 0.005 | 72.000 ± 0.124 | unsupported | 0.407 ± 0.014 | 0.550 ± 0.194 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.018 ± 0.000 | 12.000 ± 0.062 | unsupported | 0.022 ± 0.000 | 0.094 ± 0.007 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.261 ± 0.007 | 11.999 ± 0.000 | unsupported | 0.413 ± 0.027 | 0.793 ± 0.103 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 8 | `1024x64` | 0.011 ± 0.001 | 24.343 ± 0.750 | unsupported | 0.048 ± 0.013 | 0.062 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 8 | `4096x256` | 0.151 ± 0.028 | 23.999 ± 0.000 | unsupported | 0.268 ± 0.008 | 0.442 ± 0.036 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.050 ± 0.000 | 12.000 ± 0.016 | unsupported | 0.056 ± 0.000 | 0.089 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 0.766 ± 0.004 | 11.999 ± 0.001 | unsupported | 1.045 ± 0.028 | 0.913 ± 0.032 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 8 | `1024x64` | 0.026 ± 0.000 | 24.015 ± 0.016 | unsupported | 0.044 ± 0.001 | 0.089 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 8 | `4096x256` | 0.388 ± 0.002 | 23.999 ± 0.000 | unsupported | 0.552 ± 0.009 | 0.688 ± 0.031 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.058 ± 0.000 | 71.859 ± 0.376 | unsupported | 0.038 ± 0.000 | 0.134 ± 0.004 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.315 ± 0.008 | 96.000 ± 0.125 | unsupported | 0.335 ± 0.004 | 1.056 ± 0.018 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 8 | `1024x64` | 0.062 ± 0.001 | 90.046 ± 0.234 | unsupported | 0.037 ± 0.003 | 0.139 ± 0.004 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 8 | `4096x256` | 0.379 ± 0.016 | 210.196 ± 2.639 | unsupported | 0.174 ± 0.009 | 0.842 ± 0.070 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.070 ± 0.001 | 95.999 ± 0.187 | unsupported | 0.021 ± 0.000 | 0.224 ± 0.010 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.456 ± 0.008 | 120.124 ± 0.126 | unsupported | 0.273 ± 0.005 | 1.872 ± 0.022 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 8 | `1024x64` | 0.071 ± 0.001 | 171.930 ± 1.290 | unsupported | 0.031 ± 0.002 | 0.227 ± 0.003 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 8 | `4096x256` | 0.517 ± 0.026 | 282.375 ± 4.374 | unsupported | 0.157 ± 0.046 | 1.410 ± 0.052 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.125 ± 0.001 | 119.874 ± 0.594 | unsupported | unsupported | 0.251 ± 0.006 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.110 ± 0.006 | 144.124 ± 0.126 | unsupported | unsupported | 2.366 ± 0.029 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 8 | `1024x64` | 0.107 ± 0.004 | 252.197 ± 1.121 | unsupported | unsupported | 0.254 ± 0.007 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 8 | `4096x256` | 0.836 ± 0.056 | 354.498 ± 4.250 | unsupported | unsupported | 1.988 ± 0.093 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.010 ± 0.000 | 24.015 ± 0.016 | unsupported | 0.010 ± 0.000 | 0.037 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.045 ± 0.001 | 24.000 ± 0.001 | unsupported | 0.031 ± 0.001 | 0.224 ± 0.003 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 8 | `1024x64` | 0.010 ± 0.000 | 24.000 ± 0.140 | unsupported | 0.009 ± 0.000 | 0.037 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 8 | `4096x256` | 0.029 ± 0.002 | 72.000 ± 0.126 | unsupported | 0.018 ± 0.001 | 0.164 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.031 ± 0.001 | 35.875 ± 0.235 | unsupported | 0.037 ± 0.000 | 0.113 ± 0.006 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.177 ± 0.005 | 48.000 ± 0.125 | unsupported | 0.226 ± 0.002 | 1.041 ± 0.040 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 8 | `1024x64` | 0.035 ± 0.000 | 44.781 ± 0.453 | unsupported | 0.067 ± 0.003 | 0.118 ± 0.004 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 8 | `4096x256` | 0.189 ± 0.004 | 105.750 ± 4.297 | unsupported | 0.195 ± 0.006 | 0.797 ± 0.047 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.013 ± 0.000 | 12.015 ± 0.016 | unsupported | 0.051 ± 0.002 | 0.079 ± 0.005 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.180 ± 0.003 | 11.999 ± 0.000 | unsupported | 0.395 ± 0.018 | 0.760 ± 0.025 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 8 | `1024x64` | 0.008 ± 0.001 | 24.109 ± 0.750 | unsupported | 0.052 ± 0.002 | 0.088 ± 0.003 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 8 | `4096x256` | 0.130 ± 0.007 | 23.999 ± 0.000 | unsupported | 0.246 ± 0.015 | 0.598 ± 0.022 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.013 ± 0.000 | 12.000 ± 0.016 | unsupported | 0.052 ± 0.002 | 0.082 ± 0.005 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.188 ± 0.001 | 11.999 ± 1.125 | unsupported | 0.406 ± 0.025 | 0.808 ± 0.081 | - | - |
| cpu/neural_public_api | `silu` | f32 | 8 | `1024x64` | 0.008 ± 0.001 | 24.015 ± 0.687 | unsupported | 0.054 ± 0.003 | 0.090 ± 0.002 | - | - |
| cpu/neural_public_api | `silu` | f32 | 8 | `4096x256` | 0.131 ± 0.006 | 23.999 ± 0.000 | unsupported | 0.256 ± 0.011 | 0.611 ± 0.018 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.068 ± 0.001 | 96.046 ± 0.188 | unsupported | 0.023 ± 0.000 | 0.220 ± 0.007 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.458 ± 0.009 | 120.124 ± 0.500 | unsupported | 0.303 ± 0.002 | 1.830 ± 0.027 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 8 | `1024x64` | 0.068 ± 0.001 | 171.609 ± 1.540 | unsupported | 0.029 ± 0.001 | 0.221 ± 0.007 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 8 | `4096x256` | 0.510 ± 0.027 | 280.999 ± 3.125 | unsupported | 0.161 ± 0.007 | 1.368 ± 0.038 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.042 ± 0.000 | 12.140 ± 0.328 | unsupported | 0.115 ± 0.003 | 0.190 ± 0.011 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.642 ± 0.003 | 12.000 ± 0.375 | unsupported | 0.902 ± 0.028 | 2.608 ± 0.018 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 8 | `1024x64` | 0.023 ± 0.001 | 24.015 ± 0.001 | unsupported | 0.117 ± 0.003 | 0.195 ± 0.009 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 8 | `4096x256` | 0.327 ± 0.004 | 23.999 ± 0.000 | unsupported | 0.473 ± 0.011 | 1.828 ± 0.039 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.326 ± 0.002 | 107.671 ± 0.562 | unsupported | 0.184 ± 0.003 | 0.086 ± 0.008 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 5.525 ± 0.072 | 131.999 ± 1.500 | unsupported | 1.854 ± 0.085 | 0.829 ± 0.050 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 8 | `1024x64` | 0.343 ± 0.004 | 272.999 ± 1.234 | unsupported | 0.187 ± 0.004 | 0.074 ± 0.005 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 8 | `4096x256` | 4.707 ± 0.112 | 352.874 ± 3.499 | unsupported | 1.031 ± 0.023 | 0.668 ± 0.033 | - | - |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 17.504 ± 0.107 | - | unsupported | 17.364 ± 0.075 | - | 23.351 ± 0.144 | 26.510 ± 0.068 |
| cpu/output_reuse | `add_into` | f64 | 8 | `33554432` | 18.675 ± 0.190 | - | unsupported | 18.774 ± 0.090 | - | 23.377 ± 0.087 | 19.242 ± 0.467 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 12.435 ± 0.056 | - | unsupported | 12.693 ± 0.167 | - | 16.320 ± 0.075 | 12.500 ± 0.177 |
| cpu/output_reuse | `conj_into` | c64 | 8 | `16777216` | 13.708 ± 0.244 | - | unsupported | 13.665 ± 0.103 | - | 16.722 ± 0.099 | 13.865 ± 0.138 |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 12.475 ± 0.025 | - | unsupported | 12.668 ± 0.087 | - | 13.223 ± 0.054 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 8 | `33554432` | 13.834 ± 0.194 | - | unsupported | 13.813 ± 0.229 | - | 13.198 ± 0.098 | - |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 17.748 ± 0.082 | - | unsupported | 17.348 ± 0.050 | - | 23.517 ± 0.103 | 27.023 ± 0.060 |
| cpu/output_reuse | `div_into` | f64 | 8 | `33554432` | 18.769 ± 0.121 | - | unsupported | 18.779 ± 0.157 | - | 23.779 ± 0.038 | 19.307 ± 0.743 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 17.504 ± 0.042 | - | unsupported | 21.789 ± 0.051 | - | 20.350 ± 0.061 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 8 | `1024x1024` | 8.976 ± 0.052 | - | unsupported | 12.045 ± 0.125 | - | 10.564 ± 0.073 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 17.591 ± 0.015 | - | unsupported | 21.605 ± 0.097 | - | 20.290 ± 0.030 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 8 | `1024x1024` | 9.055 ± 0.023 | - | unsupported | 11.829 ± 0.065 | - | 10.418 ± 0.046 | - |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 17.495 ± 0.068 | - | unsupported | 17.399 ± 0.108 | - | 23.365 ± 0.096 | 26.585 ± 0.143 |
| cpu/output_reuse | `mul_into` | f64 | 8 | `33554432` | 18.520 ± 0.177 | - | unsupported | 18.719 ± 0.137 | - | 23.439 ± 0.067 | 19.257 ± 0.464 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 12.712 ± 0.132 | - | unsupported | 12.662 ± 0.063 | - | 16.443 ± 0.104 | 20.481 ± 0.153 |
| cpu/output_reuse | `neg_into` | f64 | 8 | `33554432` | 13.755 ± 0.158 | - | unsupported | 13.769 ± 0.110 | - | 16.483 ± 0.058 | 14.411 ± 0.158 |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.458 ± 0.002 | unsupported | unsupported | 0.465 ± 0.001 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 4.592 ± 0.005 | unsupported | unsupported | 5.050 ± 0.035 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 8 | `256x256,rhs=16` | 0.424 ± 0.002 | unsupported | unsupported | 0.451 ± 0.003 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 8 | `768x768,rhs=16` | 3.509 ± 0.012 | unsupported | unsupported | 3.057 ± 0.017 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 17.491 ± 0.104 | - | unsupported | 17.369 ± 0.065 | - | 23.716 ± 0.047 | 26.580 ± 0.303 |
| cpu/output_reuse | `sub_into` | f64 | 8 | `33554432` | 18.721 ± 0.160 | - | unsupported | 18.745 ± 0.134 | - | 23.668 ± 0.061 | 19.227 ± 0.741 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 28.139 ± 0.074 | - | 0.002 ± 0.000 | 21.437 ± 0.148 | 42.243 ± 0.359 | 100.046 ± 2.021 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 8 | `8192x1 -> 8192x4096` | 15.787 ± 0.119 | - | 0.002 ± 0.000 | 13.125 ± 0.708 | 30.803 ± 0.346 | 100.461 ± 2.042 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 13.745 ± 0.254 | - | 201.620 ± 1.180 | 13.739 ± 0.206 | 30.466 ± 1.010 | 50.646 ± 0.576 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 8 | `33554432` | 11.077 ± 1.789 | - | 290.378 ± 4.000 | 10.160 ± 0.161 | 21.451 ± 0.387 | 52.857 ± 0.675 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 44.707 ± 0.739 | - | 167.115 ± 1.201 | 43.126 ± 0.279 | 87.109 ± 0.845 | 184.802 ± 1.220 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 8 | `8192 -> 8192x8192` | 25.751 ± 0.431 | - | 167.918 ± 0.446 | 25.585 ± 0.509 | 59.422 ± 0.797 | 191.554 ± 2.607 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 24.446 ± 0.200 | - | 46.317 ± 2.803 | 15.665 ± 0.418 | 25.638 ± 1.358 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 8 | `8388608x2x2 -> 8388608x2` | 24.720 ± 0.141 | - | 48.010 ± 1.001 | 10.447 ± 0.300 | 18.632 ± 0.428 | - | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 25.229 ± 0.293 | - | 0.002 ± 0.000 | 23.888 ± 0.143 | 31.452 ± 4.700 | 105.248 ± 0.725 | - |
| cpu/structural_shape | `reshape` | f64 | 8 | `33554432 -> 8192x4096` | 16.338 ± 0.180 | - | 0.002 ± 0.000 | 16.053 ± 0.545 | 18.647 ± 0.757 | 104.789 ± 0.844 | - |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 22.002 ± 0.273 | - | 0.002 ± 0.000 | 57.010 ± 0.095 | 34.472 ± 0.583 | 224.112 ± 0.427 | 43.890 ± 2.921 |
| cpu/structural_shape | `transpose` | f64 | 8 | `4096x4096` | 15.566 ± 0.200 | - | 0.002 ± 0.000 | 57.888 ± 0.192 | 23.620 ± 1.248 | 223.546 ± 1.045 | 30.700 ± 0.343 |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 13.339 ± 0.191 | - | 54.911 ± 0.202 | 16.098 ± 0.479 | 25.985 ± 0.369 | 54.824 ± 1.361 | - |
| cpu/structural_shape | `tril` | f64 | 8 | `4096x4096` | 8.019 ± 0.210 | - | 70.156 ± 0.708 | 9.153 ± 0.297 | 19.005 ± 0.311 | 54.901 ± 0.773 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 13.427 ± 0.056 | - | 57.654 ± 0.123 | 14.041 ± 0.335 | 25.952 ± 0.181 | 55.952 ± 0.421 | - |
| cpu/structural_shape | `triu` | f64 | 8 | `4096x4096` | 7.889 ± 0.237 | - | 70.400 ± 2.022 | 8.213 ± 0.164 | 18.950 ± 0.298 | 56.290 ± 0.419 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 8 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.001 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 8 | `33554432 -> 8192x4096` | 0.001 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 8 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 8 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 3.005630 | 187851.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 2.579119 | 161194.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 1919.988725 | 119999295.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 16 | 4.131661 | 258228.81 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 8 | tenferro-rs direct API (ms) | 16 | 2.129018 | 133063.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 16 | 4044.396460 | 252774778.75 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 6.295969 | 393498.06 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 383.988603 | 23999287.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 8 | tenferro-rs direct API (ms) | 16 | 8.197737 | 512358.56 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 8 | tenferro-rs eager mode (ms) | 16 | 1152.983612 | 72061475.75 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 17.425678 | 1089104.88 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.610753 | 350672.06 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 4.037945 | 252371.56 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 3265.978777 | 204123673.56 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 128.605971 | 8037873.19 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 75.504359 | 4719022.44 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 47.858673 | 2991167.06 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 49494.975594 | 3093435974.62 |
| householder_qr | f64 | `256x128` | 8 | JAX Python (XLA CPU) (ms) | 16 | 18.073712 | 1129607.00 |
| householder_qr | f64 | `256x128` | 8 | PyTorch Python (ms) | 16 | 5.579365 | 348710.31 |
| householder_qr | f64 | `256x128` | 8 | tenferro-rs direct API (ms) | 16 | 3.915872 | 244742.00 |
| householder_qr | f64 | `256x128` | 8 | tenferro-rs eager mode (ms) | 16 | 6568.976657 | 410561041.06 |
| householder_qr | f64 | `768x384` | 8 | JAX Python (XLA CPU) (ms) | 16 | 124.682189 | 7792636.81 |
| householder_qr | f64 | `768x384` | 8 | PyTorch Python (ms) | 16 | 44.741894 | 2796368.38 |
| householder_qr | f64 | `768x384` | 8 | tenferro-rs direct API (ms) | 16 | 38.621981 | 2413873.81 |
| householder_qr | f64 | `768x384` | 8 | tenferro-rs eager mode (ms) | 16 | 100227.998470 | 6264249904.38 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 13.587372 | 849210.75 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 20693.948455 | 1293371778.44 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 168.287274 | 10517954.62 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 77640.947252 | 4852559203.25 |
| rank_revealing_qr | f64 | `256x128` | 8 | tenferro-rs direct API (ms) | 16 | 13.572984 | 848311.50 |
| rank_revealing_qr | f64 | `256x128` | 8 | tenferro-rs eager mode (ms) | 16 | 20746.955621 | 1296684726.31 |
| rank_revealing_qr | f64 | `768x384` | 8 | tenferro-rs direct API (ms) | 16 | 120.758081 | 7547380.06 |
| rank_revealing_qr | f64 | `768x384` | 8 | tenferro-rs eager mode (ms) | 16 | 154764.953418 | 9672809588.62 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 13.505918 | 844119.88 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 20839.952949 | 1302497059.31 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 164.746958 | 10296684.88 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 77808.935564 | 4863058472.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 8 | tenferro-rs direct API (ms) | 16 | 13.817185 | 863574.06 |
| rank_revealing_qr_deficient | f64 | `256x128` | 8 | tenferro-rs eager mode (ms) | 16 | 20813.959346 | 1300872459.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 8 | tenferro-rs direct API (ms) | 16 | 120.621047 | 7538815.44 |
| rank_revealing_qr_deficient | f64 | `768x384` | 8 | tenferro-rs eager mode (ms) | 16 | 155054.939508 | 9690933719.25 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.330432 | 36413.00 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.505080 | 54766.88 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.357925 | 36842.58 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1536.971485 | 24015179.45 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.001406 | 875175.75 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.016429 | 752053.62 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.246069 | 530758.62 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.994485 | 23999310.62 |
| erf | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 2.425646 | 37900.72 |
| erf | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 2.300334 | 35942.72 |
| erf | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 1.364831 | 21325.48 |
| erf | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 4607.968905 | 71999514.14 |
| erf | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 4.399961 | 549995.12 |
| erf | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 3.259064 | 407383.00 |
| erf | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 2.195683 | 274460.38 |
| erf | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 576.000871 | 72000108.88 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.009127 | 93892.61 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 1.399658 | 21869.66 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.138398 | 17787.47 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.998989 | 11999984.20 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.346214 | 793276.75 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.300218 | 412527.25 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.087698 | 260962.25 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994960 | 11999370.00 |
| gelu | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 3.992291 | 62379.55 |
| gelu | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 3.053509 | 47711.08 |
| gelu | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 0.678343 | 10599.11 |
| gelu | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 1557.969171 | 24343268.30 |
| gelu | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 3.537481 | 442185.12 |
| gelu | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 2.141367 | 267670.88 |
| gelu | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 1.205082 | 150635.25 |
| gelu | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 191.993849 | 23999231.12 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.693007 | 88953.23 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.585309 | 56020.45 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.172913 | 49576.77 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.976413 | 11999631.45 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.303138 | 912892.25 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.356629 | 1044578.62 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.126041 | 765755.12 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994890 | 11999361.25 |
| gelu_tanh | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 5.678567 | 88727.61 |
| gelu_tanh | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 2.815145 | 43986.64 |
| gelu_tanh | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 1.694004 | 26468.81 |
| gelu_tanh | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 1536.969064 | 24015141.62 |
| gelu_tanh | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 5.500285 | 687535.62 |
| gelu_tanh | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 4.416745 | 552093.12 |
| gelu_tanh | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 3.106975 | 388371.88 |
| gelu_tanh | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 191.994036 | 23999254.50 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.577894 | 134029.59 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.461947 | 38467.92 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.701633 | 57838.02 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4598.964270 | 71858816.72 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.448612 | 1056076.50 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.677559 | 334694.88 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.519923 | 314990.38 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 767.996727 | 95999590.88 |
| layer_norm | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 8.876255 | 138691.48 |
| layer_norm | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 2.353904 | 36779.75 |
| layer_norm | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 3.938997 | 61546.83 |
| layer_norm | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 5762.957373 | 90046208.95 |
| layer_norm | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 6.734247 | 841780.88 |
| layer_norm | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 1.389003 | 173625.38 |
| layer_norm | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 3.030530 | 378816.25 |
| layer_norm | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 1681.564498 | 210195562.25 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.309814 | 223590.84 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 1.338444 | 20913.19 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.456995 | 69640.55 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6143.960605 | 95999384.45 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.973230 | 1871653.75 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.187666 | 273458.25 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.647334 | 455916.75 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 960.994089 | 120124261.12 |
| log_softmax | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 14.542879 | 227232.48 |
| log_softmax | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 1.978663 | 30916.61 |
| log_softmax | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 4.535767 | 70871.36 |
| log_softmax | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 11003.519067 | 171929985.42 |
| log_softmax | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 11.279881 | 1409985.12 |
| log_softmax | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 1.259631 | 157453.88 |
| log_softmax | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 4.134755 | 516844.38 |
| log_softmax | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 2258.996384 | 282374548.00 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 16.048383 | 250755.98 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.995644 | 124931.94 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7671.962461 | 119874413.45 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 18.931576 | 2366447.00 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.880441 | 1110055.12 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1152.991823 | 144123977.88 |
| masked_softmax | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 16.256307 | 254004.80 |
| masked_softmax | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 6.836265 | 106816.64 |
| masked_softmax | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 16140.609935 | 252197030.23 |
| masked_softmax | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 15.906905 | 1988363.12 |
| masked_softmax | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 6.690799 | 836349.88 |
| masked_softmax | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 2835.987995 | 354498499.38 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.369856 | 37029.00 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.634141 | 9908.45 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.616096 | 9626.50 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1536.969848 | 24015153.88 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.790727 | 223840.88 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.248603 | 31075.38 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.361282 | 45160.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.997767 | 23999720.88 |
| reduce_mean | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 2.344756 | 36636.81 |
| reduce_mean | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 0.599740 | 9370.94 |
| reduce_mean | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 0.646144 | 10096.00 |
| reduce_mean | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 1535.973822 | 23999590.97 |
| reduce_mean | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 1.315215 | 164401.88 |
| reduce_mean | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 0.141965 | 17745.62 |
| reduce_mean | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 0.235170 | 29396.25 |
| reduce_mean | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 575.998807 | 71999850.88 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.262943 | 113483.48 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.384002 | 37250.03 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.984876 | 31013.69 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2295.975243 | 35874613.17 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.331414 | 1041426.75 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.805394 | 225674.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.413413 | 176676.62 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 383.996612 | 47999576.50 |
| rms_norm | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 7.551061 | 117985.33 |
| rms_norm | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 4.290899 | 67045.30 |
| rms_norm | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 2.219063 | 34672.86 |
| rms_norm | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 2865.963677 | 44780682.45 |
| rms_norm | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 6.372593 | 796574.12 |
| rms_norm | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 1.559360 | 194920.00 |
| rms_norm | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 1.508588 | 188573.50 |
| rms_norm | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 845.997288 | 105749661.00 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.064206 | 79128.22 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.279430 | 51241.09 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.821387 | 12834.17 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 768.970102 | 12015157.84 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.078857 | 759857.12 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.163173 | 395396.62 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.438169 | 179771.12 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.995410 | 11999426.25 |
| sigmoid | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 5.636138 | 88064.66 |
| sigmoid | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 3.355553 | 52430.52 |
| sigmoid | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 0.506002 | 7906.28 |
| sigmoid | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 1542.999929 | 24109373.89 |
| sigmoid | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 4.782986 | 597873.25 |
| sigmoid | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 1.967152 | 245894.00 |
| sigmoid | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 1.042227 | 130278.38 |
| sigmoid | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 191.994572 | 23999321.50 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.241236 | 81894.31 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.343700 | 52245.31 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.855481 | 13366.89 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.975982 | 11999624.72 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.464846 | 808105.75 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.250125 | 406265.62 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.507498 | 188437.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.995771 | 11999471.38 |
| silu | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 5.745723 | 89776.92 |
| silu | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 3.483222 | 54425.34 |
| silu | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 0.535257 | 8363.39 |
| silu | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 1536.972840 | 24015200.62 |
| silu | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 4.888243 | 611030.38 |
| silu | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 2.044617 | 255577.13 |
| silu | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 1.046235 | 130779.38 |
| silu | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 191.994628 | 23999328.50 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.084644 | 220072.56 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 1.502890 | 23482.66 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.378487 | 68413.86 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6146.956913 | 96046201.77 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.643435 | 1830429.38 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.421241 | 302655.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.661790 | 457723.75 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 960.990753 | 120123844.12 |
| softmax | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 14.163391 | 221302.98 |
| softmax | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 1.844292 | 28817.06 |
| softmax | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 4.378362 | 68411.91 |
| softmax | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 10982.955117 | 171608673.70 |
| softmax | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 10.941930 | 1367741.25 |
| softmax | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 1.286011 | 160751.38 |
| softmax | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 4.079312 | 509914.00 |
| softmax | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 2247.992254 | 280999031.75 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.167693 | 190120.20 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.337080 | 114641.88 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.673222 | 41769.09 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 776.957450 | 12139960.16 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 20.862202 | 2607775.25 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.217657 | 902207.12 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.132910 | 641613.75 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.996062 | 11999507.75 |
| softplus | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 12.454453 | 194600.83 |
| softplus | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 7.507915 | 117311.17 |
| softplus | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 1.453887 | 22716.98 |
| softplus | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 1536.971777 | 24015184.02 |
| softplus | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 14.620724 | 1827590.50 |
| softplus | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 3.780587 | 472573.38 |
| softplus | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 2.619264 | 327408.00 |
| softplus | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 191.993662 | 23999207.75 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.473279 | 85519.98 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.800084 | 184376.31 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 20.848656 | 325760.25 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6890.963539 | 107671305.30 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.628881 | 828610.12 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 14.835990 | 1854498.75 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 44.198361 | 5524795.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1055.993219 | 131999152.38 |
| take_along_axis | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 4.755895 | 74310.86 |
| take_along_axis | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 11.950406 | 186725.09 |
| take_along_axis | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 21.956212 | 343065.81 |
| take_along_axis | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 17471.947802 | 272999184.41 |
| take_along_axis | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 5.340316 | 667539.50 |
| take_along_axis | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 8.250670 | 1031333.75 |
| take_along_axis | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 37.653906 | 4706738.25 |
| take_along_axis | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 2822.990228 | 352873778.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 7.442155 | 465134.69 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 7.334525 | 458407.81 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 80.798452 | 5049903.25 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 73.473863 | 4592116.44 |
| solve_read_into | f64 | `256x256,rhs=16` | 8 | PyTorch Python (ms) | 16 | 7.211090 | 450693.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 8 | tenferro-rs direct API (ms) | 16 | 6.786066 | 424129.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 8 | PyTorch Python (ms) | 16 | 48.915595 | 3057224.69 |
| solve_read_into | f64 | `768x768,rhs=16` | 8 | tenferro-rs direct API (ms) | 16 | 56.146416 | 3509151.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.029745 | 1859.06 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002796 | 174.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 8 | PyTorch Python (ms) | 16 | 0.029645 | 1852.81 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 8 | tenferro-rs direct API (ms) | 16 | 0.003226 | 201.62 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000341 | 21.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.027862 | 1741.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.010690 | 668.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 8 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000541 | 33.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 8 | PyTorch Python (ms) | 16 | 0.027361 | 1710.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 8 | tenferro-rs direct API (ms) | 16 | 0.011191 | 699.44 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000270 | 16.88 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.028063 | 1753.94 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.006973 | 435.81 |
| slice_view | f64 | `4194304 -> 2096128` | 8 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000351 | 21.94 |
| slice_view | f64 | `4194304 -> 2096128` | 8 | PyTorch Python (ms) | 16 | 0.027802 | 1737.62 |
| slice_view | f64 | `4194304 -> 2096128` | 8 | tenferro-rs direct API (ms) | 16 | 0.007704 | 481.50 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000130 | 8.12 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.026920 | 1682.50 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.005219 | 326.19 |
| transpose_view | f64 | `4096x4096` | 8 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000140 | 8.75 |
| transpose_view | f64 | `4096x4096` | 8 | PyTorch Python (ms) | 16 | 0.026409 | 1650.56 |
| transpose_view | f64 | `4096x4096` | 8 | tenferro-rs direct API (ms) | 16 | 0.005580 | 348.75 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 110.7x faster than the slowest successful cell (`tenferro-trace`, 342.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `julia-base` is 126.9x faster than the slowest successful cell (`tenferro-trace`, 342.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 159.5x faster than the slowest successful cell (`tenferro-trace`, 342.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 175.6x faster than the slowest successful cell (`tenferro-trace`, 342.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=8, shape=`448x448`): `jax-cpu` is 318.1x faster than the slowest successful cell (`tenferro-trace`, 887.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=8, shape=`448x448`): `julia-base` is 442.9x faster than the slowest successful cell (`tenferro-trace`, 887.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=8, shape=`448x448`): `pytorch-cpu` is 758.7x faster than the slowest successful cell (`tenferro-trace`, 887.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=8, shape=`448x448`): `tenferro-direct` is 541.6x faster than the slowest successful cell (`tenferro-trace`, 887.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/div` (c64, threads=8, shape=`8388608`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`tenferro-trace`, 137.576 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/dot_general` (c64, threads=8, shape=`640x640`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`tenferro-trace`, 98.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 328.4x faster than the slowest successful cell (`tenferro-trace`, 5418.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `julia-base` is 326.8x faster than the slowest successful cell (`tenferro-trace`, 5418.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 402.3x faster than the slowest successful cell (`tenferro-trace`, 5418.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 564.2x faster than the slowest successful cell (`tenferro-trace`, 5418.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=8, shape=`112x112`): `jax-cpu` is 593.9x faster than the slowest successful cell (`tenferro-trace`, 9879.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=8, shape=`112x112`): `julia-base` is 594.1x faster than the slowest successful cell (`tenferro-trace`, 9879.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=8, shape=`112x112`): `pytorch-cpu` is 762.9x faster than the slowest successful cell (`tenferro-trace`, 9879.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=8, shape=`112x112`): `tenferro-direct` is 1001.6x faster than the slowest successful cell (`tenferro-trace`, 9879.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/exp` (c64, threads=8, shape=`4194304`): `strided-jl` is 10.0x faster than the slowest successful cell (`tenferro-trace`, 168.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/exp` (c64, threads=8, shape=`4194304`): `tenferro-direct` is 12.7x faster than the slowest successful cell (`tenferro-trace`, 168.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/log` (c64, threads=8, shape=`4194304`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 191.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/mul` (c64, threads=8, shape=`8388608`): `pytorch-cpu` is 10.8x faster than the slowest successful cell (`tenferro-trace`, 114.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/mul` (c64, threads=8, shape=`8388608`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`tenferro-trace`, 114.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `jax-cpu` is 15.1x faster than the slowest successful cell (`tenferro-trace`, 59.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 26.9x faster than the slowest successful cell (`tenferro-trace`, 59.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=8, shape=`2048x1536`): `jax-cpu` is 43.6x faster than the slowest successful cell (`tenferro-trace`, 143.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=8, shape=`2048x1536`): `julia-base` is 22.6x faster than the slowest successful cell (`tenferro-trace`, 143.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=8, shape=`2048x1536`): `pytorch-cpu` is 44.5x faster than the slowest successful cell (`tenferro-trace`, 143.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=8, shape=`2048x1536`): `tenferro-direct` is 118.9x faster than the slowest successful cell (`tenferro-trace`, 143.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 75.5x faster than the slowest successful cell (`tenferro-trace`, 528.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `julia-base` is 67.8x faster than the slowest successful cell (`tenferro-trace`, 528.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 96.1x faster than the slowest successful cell (`tenferro-trace`, 528.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 152.4x faster than the slowest successful cell (`tenferro-trace`, 528.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=8, shape=`256x256`): `jax-cpu` is 192.3x faster than the slowest successful cell (`tenferro-trace`, 1213.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=8, shape=`256x256`): `julia-base` is 189.6x faster than the slowest successful cell (`tenferro-trace`, 1213.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=8, shape=`256x256`): `pytorch-cpu` is 283.8x faster than the slowest successful cell (`tenferro-trace`, 1213.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=8, shape=`256x256`): `tenferro-direct` is 482.1x faster than the slowest successful cell (`tenferro-trace`, 1213.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 219.5x faster than the slowest successful cell (`tenferro-trace`, 792.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `julia-base` is 248.6x faster than the slowest successful cell (`tenferro-trace`, 792.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 285.7x faster than the slowest successful cell (`tenferro-trace`, 792.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 299.9x faster than the slowest successful cell (`tenferro-trace`, 792.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=8, shape=`384x384,rhs=8`): `jax-cpu` is 566.2x faster than the slowest successful cell (`tenferro-trace`, 1609.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=8, shape=`384x384,rhs=8`): `julia-base` is 775.7x faster than the slowest successful cell (`tenferro-trace`, 1609.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=8, shape=`384x384,rhs=8`): `pytorch-cpu` is 782.8x faster than the slowest successful cell (`tenferro-trace`, 1609.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=8, shape=`384x384,rhs=8`): `tenferro-direct` is 790.5x faster than the slowest successful cell (`tenferro-trace`, 1609.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/tensordot` (c64, threads=8, shape=`640x640`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`tenferro-trace`, 99.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/chain_log1p_exp_mul` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 17.3x faster than the slowest successful cell (`tenferro-trace`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/chain_log1p_exp_mul` (f64, threads=8, shape=`4194304`): `jax-cpu` is 13.3x faster than the slowest successful cell (`tenferro-trace`, 264.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/chain_log1p_exp_mul` (f64, threads=8, shape=`4194304`): `pytorch-cpu` is 17.0x faster than the slowest successful cell (`tenferro-trace`, 264.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/chain_log1p_exp_mul` (f64, threads=8, shape=`4194304`): `strided-jl` is 14.3x faster than the slowest successful cell (`tenferro-trace`, 264.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/chain_log1p_exp_mul` (f64, threads=8, shape=`4194304`): `tenferro-direct` is 44.2x faster than the slowest successful cell (`tenferro-trace`, 264.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/clamp` (f64, threads=8, shape=`8388608`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/clamp` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/cos` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`jax-cpu`, 76.685 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/cos` (f64, threads=8, shape=`8388608`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`tenferro-trace`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/cos` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 28.1x faster than the slowest successful cell (`tenferro-trace`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 18.4x faster than the slowest successful cell (`julia-base`, 65.933 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=8, shape=`8388608`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 20.4x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/expm1` (f64, threads=8, shape=`4194304`): `pytorch-cpu` is 18.2x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/expm1` (f64, threads=8, shape=`4194304`): `strided-jl` is 12.1x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/expm1` (f64, threads=8, shape=`4194304`): `tenferro-direct` is 28.4x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`julia-base`, 77.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log` (f64, threads=8, shape=`8388608`): `pytorch-cpu` is 12.4x faster than the slowest successful cell (`tenferro-trace`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 27.6x faster than the slowest successful cell (`tenferro-trace`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`julia-base`, 45.528 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=8, shape=`4194304`): `pytorch-cpu` is 20.1x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=8, shape=`4194304`): `strided-jl` is 10.7x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=8, shape=`4194304`): `tenferro-direct` is 43.3x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `jax-cpu` is 638.8x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `tenferro-direct` is 744.4x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=8, shape=`1024x64`): `jax-cpu` is 978.9x faster than the slowest successful cell (`tenferro-eager`, 252.775 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=8, shape=`1024x64`): `tenferro-direct` is 1899.6x faster than the slowest successful cell (`tenferro-eager`, 252.775 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/pow` (f64, threads=4, shape=`4194304`): `pytorch-cpu` is 13.5x faster than the slowest successful cell (`julia-base`, 183.748 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/pow` (f64, threads=8, shape=`4194304`): `pytorch-cpu` is 26.4x faster than the slowest successful cell (`julia-base`, 183.330 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/pow` (f64, threads=8, shape=`4194304`): `strided-jl` is 10.8x faster than the slowest successful cell (`julia-base`, 183.330 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/pow` (f64, threads=8, shape=`4194304`): `tenferro-direct` is 19.5x faster than the slowest successful cell (`julia-base`, 183.330 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_all` (f64, threads=8, shape=`8192x4096`): `pytorch-cpu` is 13.2x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_all` (f64, threads=8, shape=`8192x4096`): `tenferro-direct` is 13.3x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 16.0x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 12.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 31.8x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=8, shape=`2048x2048`): `jax-cpu` is 62.7x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=8, shape=`2048x2048`): `julia-base` is 35.1x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=8, shape=`2048x2048`): `tenferro-direct` is 110.1x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 24.1x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 21.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 34.2x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=8, shape=`2048x2048`): `jax-cpu` is 30.2x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=8, shape=`2048x2048`): `julia-base` is 73.4x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=8, shape=`2048x2048`): `pytorch-cpu` is 101.6x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=8, shape=`2048x2048`): `tenferro-direct` is 102.1x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_all` (f64, threads=8, shape=`8192x4096`): `pytorch-cpu` is 13.3x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_all` (f64, threads=8, shape=`8192x4096`): `tenferro-direct` is 13.3x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 16.3x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 12.5x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 33.7x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=8, shape=`2048x2048`): `jax-cpu` is 63.9x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=8, shape=`2048x2048`): `julia-base` is 37.2x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=8, shape=`2048x2048`): `tenferro-direct` is 89.4x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis1` (f64, threads=8, shape=`4096x4096`): `julia-base` is 13.5x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis1` (f64, threads=8, shape=`4096x4096`): `pytorch-cpu` is 26.4x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis1` (f64, threads=8, shape=`4096x4096`): `tenferro-direct` is 25.9x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_all` (f64, threads=8, shape=`8192x4096`): `pytorch-cpu` is 13.2x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_all` (f64, threads=8, shape=`8192x4096`): `tenferro-direct` is 13.3x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 18.9x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 29.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 15.2x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 58.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=8, shape=`2048x2048`): `jax-cpu` is 72.3x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=8, shape=`2048x2048`): `julia-base` is 78.5x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=8, shape=`2048x2048`): `pytorch-cpu` is 41.2x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=8, shape=`2048x2048`): `tenferro-direct` is 137.0x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 25.3x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 29.8x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 56.6x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 37.0x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=8, shape=`2048x2048`): `jax-cpu` is 81.6x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=8, shape=`2048x2048`): `julia-base` is 78.0x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=8, shape=`2048x2048`): `pytorch-cpu` is 132.6x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=8, shape=`2048x2048`): `tenferro-direct` is 97.7x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_all` (f64, threads=8, shape=`8192x4096`): `pytorch-cpu` is 13.2x faster than the slowest successful cell (`tenferro-trace`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_all` (f64, threads=8, shape=`8192x4096`): `tenferro-direct` is 13.3x faster than the slowest successful cell (`tenferro-trace`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 19.8x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 29.6x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 15.2x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 39.1x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=8, shape=`2048x2048`): `jax-cpu` is 73.1x faster than the slowest successful cell (`tenferro-trace`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=8, shape=`2048x2048`): `julia-base` is 81.6x faster than the slowest successful cell (`tenferro-trace`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=8, shape=`2048x2048`): `pytorch-cpu` is 41.6x faster than the slowest successful cell (`tenferro-trace`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=8, shape=`2048x2048`): `tenferro-direct` is 127.3x faster than the slowest successful cell (`tenferro-trace`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 23.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 27.6x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 57.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 36.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=8, shape=`2048x2048`): `jax-cpu` is 81.0x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=8, shape=`2048x2048`): `julia-base` is 78.8x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=8, shape=`2048x2048`): `pytorch-cpu` is 132.0x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=8, shape=`2048x2048`): `tenferro-direct` is 100.4x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_squares_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 61.0x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_squares_axis0` (f64, threads=8, shape=`2048x2048`): `tenferro-direct` is 140.6x faster than the slowest successful cell (`tenferro-eager`, 72.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/rem` (f64, threads=8, shape=`8388608`): `pytorch-cpu` is 13.8x faster than the slowest successful cell (`tenferro-trace`, 119.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/rem` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 16.6x faster than the slowest successful cell (`tenferro-trace`, 119.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/rsqrt` (f64, threads=8, shape=`33554432`): `pytorch-cpu` is 14.6x faster than the slowest successful cell (`julia-base`, 279.463 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/rsqrt` (f64, threads=8, shape=`33554432`): `tenferro-direct` is 15.5x faster than the slowest successful cell (`julia-base`, 279.463 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 15.4x faster than the slowest successful cell (`jax-cpu`, 75.665 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 20.8x faster than the slowest successful cell (`tenferro-trace`, 71.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sqrt` (f64, threads=8, shape=`33554432`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`julia-base`, 216.681 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 19.7x faster than the slowest successful cell (`tenferro-trace`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `jax-cpu` is 15.3x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 24.5x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 20.3x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=8, shape=`1048576+1048576`): `jax-cpu` is 27.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=8, shape=`1048576+1048576`): `pytorch-cpu` is 38.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=8, shape=`1048576+1048576`): `tenferro-direct` is 36.0x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `jax-cpu` is 18.4x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `julia-base` is 10.9x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `pytorch-cpu` is 24.7x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-direct` is 21.6x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=8, shape=`4194304 -> 2097152`): `jax-cpu` is 26.2x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=8, shape=`4194304 -> 2097152`): `pytorch-cpu` is 35.4x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=8, shape=`4194304 -> 2097152`): `tenferro-direct` is 29.3x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 224.6x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `julia-base` is 47.4x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 225.3x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `tenferro-direct` is 266.1x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=8, shape=`262144`): `jax-cpu` is 769.6x faster than the slowest successful cell (`tenferro-trace`, 143.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=8, shape=`262144`): `julia-base` is 141.0x faster than the slowest successful cell (`tenferro-trace`, 143.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=8, shape=`262144`): `pytorch-cpu` is 1264.6x faster than the slowest successful cell (`tenferro-trace`, 143.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=8, shape=`262144`): `tenferro-direct` is 1492.1x faster than the slowest successful cell (`tenferro-trace`, 143.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `jax-cpu` is 17.7x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 19.5x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 23.6x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=8, shape=`2097152`): `jax-cpu` is 25.2x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=8, shape=`2097152`): `pytorch-cpu` is 25.1x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=8, shape=`2097152`): `tenferro-direct` is 33.9x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 15.7x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 25.1x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 22.8x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=8, shape=`2097152`): `jax-cpu` is 26.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=8, shape=`2097152`): `pytorch-cpu` is 38.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=8, shape=`2097152`): `tenferro-direct` is 34.6x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 86.4x faster than the slowest successful cell (`tenferro-trace`, 35.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `julia-base` is 31.4x faster than the slowest successful cell (`tenferro-trace`, 35.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 94.1x faster than the slowest successful cell (`tenferro-trace`, 35.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `tenferro-direct` is 36.5x faster than the slowest successful cell (`tenferro-trace`, 35.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=8, shape=`262144`): `jax-cpu` is 204.2x faster than the slowest successful cell (`tenferro-trace`, 96.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=8, shape=`262144`): `julia-base` is 186.9x faster than the slowest successful cell (`tenferro-trace`, 96.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=8, shape=`262144`): `pytorch-cpu` is 260.3x faster than the slowest successful cell (`tenferro-trace`, 96.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=8, shape=`262144`): `tenferro-direct` is 94.6x faster than the slowest successful cell (`tenferro-trace`, 96.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 2992.1x faster than the slowest successful cell (`julia-base`, 7.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=8, shape=`4194304 -> 2096128`): `tenferro-trace` is 3016.5x faster than the slowest successful cell (`julia-base`, 7.587 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1024x8x8`): `jax-cpu` is 29.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1024x8x8`): `pytorch-cpu` is 45.5x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1024x8x8`): `tenferro-direct` is 118.3x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x128x128`): `jax-cpu` is 141.4x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x128x128`): `pytorch-cpu` is 148.2x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x128x128`): `tenferro-direct` is 381.5x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x128x128`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x128x128`): `tenferro-direct` is 92.8x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 296.3x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x32x32`): `pytorch-cpu` is 296.3x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 1384.7x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=8, shape=`1024x8x8`): `jax-cpu` is 59.8x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=8, shape=`1024x8x8`): `pytorch-cpu` is 95.1x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=8, shape=`1024x8x8`): `tenferro-direct` is 453.9x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=8, shape=`1x128x128`): `jax-cpu` is 161.7x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=8, shape=`1x128x128`): `pytorch-cpu` is 162.7x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=8, shape=`1x128x128`): `tenferro-direct` is 375.2x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=8, shape=`8x128x128`): `jax-cpu` is 19.3x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=8, shape=`8x128x128`): `pytorch-cpu` is 21.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=8, shape=`8x128x128`): `tenferro-direct` is 281.2x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=8, shape=`8x32x32`): `jax-cpu` is 586.4x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=8, shape=`8x32x32`): `pytorch-cpu` is 583.3x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=8, shape=`8x32x32`): `tenferro-direct` is 3882.2x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`1024x2x2`): `pytorch-cpu` is 14.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`1024x2x2`): `tenferro-direct` is 75.9x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`1024x4x4`): `tenferro-direct` is 15.4x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`8x32x32`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`8x32x32`): `tenferro-direct` is 41.7x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=8, shape=`1024x2x2`): `jax-cpu` is 19.2x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=8, shape=`1024x2x2`): `pytorch-cpu` is 28.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=8, shape=`1024x2x2`): `tenferro-direct` is 292.8x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=8, shape=`1024x4x4`): `tenferro-direct` is 60.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=8, shape=`1024x8x8`): `tenferro-direct` is 19.5x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=8, shape=`8x128x128`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`tenferro-trace`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=8, shape=`8x32x32`): `jax-cpu` is 24.0x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=8, shape=`8x32x32`): `pytorch-cpu` is 20.0x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=8, shape=`8x32x32`): `tenferro-direct` is 161.2x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`1024x4x4`): `tenferro-direct` is 17.8x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 30.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`8x32x32`): `pytorch-cpu` is 14.4x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 51.9x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=8, shape=`1024x4x4`): `jax-cpu` is 13.7x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=8, shape=`1024x4x4`): `tenferro-direct` is 70.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=8, shape=`1024x8x8`): `tenferro-direct` is 21.6x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=8, shape=`8x128x128`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=8, shape=`8x32x32`): `jax-cpu` is 66.0x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=8, shape=`8x32x32`): `pytorch-cpu` is 28.6x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=8, shape=`8x32x32`): `tenferro-direct` is 201.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`1024x4x4`): `tenferro-direct` is 22.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 35.3x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`8x32x32`): `pytorch-cpu` is 35.4x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 69.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=8, shape=`1024x4x4`): `jax-cpu` is 13.8x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=8, shape=`1024x4x4`): `pytorch-cpu` is 18.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=8, shape=`1024x4x4`): `tenferro-direct` is 85.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=8, shape=`1024x8x8`): `tenferro-direct` is 27.2x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=8, shape=`8x128x128`): `tenferro-direct` is 28.5x faster than the slowest successful cell (`jax-cpu`, 36.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=8, shape=`8x32x32`): `jax-cpu` is 63.0x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=8, shape=`8x32x32`): `pytorch-cpu` is 71.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=8, shape=`8x32x32`): `tenferro-direct` is 269.0x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1024x8x8`): `jax-cpu` is 19.9x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1024x8x8`): `pytorch-cpu` is 45.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1024x8x8`): `tenferro-direct` is 52.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x128x128`): `jax-cpu` is 289.8x faster than the slowest successful cell (`tenferro-trace`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x128x128`): `pytorch-cpu` is 347.6x faster than the slowest successful cell (`tenferro-trace`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x128x128`): `tenferro-direct` is 439.4x faster than the slowest successful cell (`tenferro-trace`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x128x128`): `jax-cpu` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x128x128`): `pytorch-cpu` is 27.1x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x128x128`): `tenferro-direct` is 34.8x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 78.1x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x32x32`): `pytorch-cpu` is 242.2x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 511.6x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`1024x8x8`): `jax-cpu` is 38.8x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`1024x8x8`): `pytorch-cpu` is 108.6x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`1024x8x8`): `tenferro-direct` is 195.0x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`1x128x128`): `jax-cpu` is 537.7x faster than the slowest successful cell (`tenferro-trace`, 144.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`1x128x128`): `pytorch-cpu` is 699.9x faster than the slowest successful cell (`tenferro-trace`, 144.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`1x128x128`): `tenferro-direct` is 873.6x faster than the slowest successful cell (`tenferro-trace`, 144.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`8x128x128`): `jax-cpu` is 34.5x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`8x128x128`): `pytorch-cpu` is 77.6x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`8x128x128`): `tenferro-direct` is 131.5x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`8x32x32`): `jax-cpu` is 204.8x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`8x32x32`): `pytorch-cpu` is 575.4x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`8x32x32`): `tenferro-direct` is 1660.1x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1024x4x4`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1024x4x4`): `pytorch-cpu` is 19.1x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1024x4x4`): `tenferro-direct` is 58.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1024x8x8`): `tenferro-direct` is 19.1x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x128x128`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 59.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x32x32`): `pytorch-cpu` is 59.3x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 170.1x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=8, shape=`1024x4x4`): `jax-cpu` is 24.4x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=8, shape=`1024x4x4`): `pytorch-cpu` is 39.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=8, shape=`1024x4x4`): `tenferro-direct` is 223.6x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=8, shape=`1024x8x8`): `jax-cpu` is 25.6x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=8, shape=`1024x8x8`): `pytorch-cpu` is 14.8x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=8, shape=`1024x8x8`): `tenferro-direct` is 75.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=8, shape=`8x128x128`): `tenferro-direct` is 40.5x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=8, shape=`8x32x32`): `jax-cpu` is 128.8x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=8, shape=`8x32x32`): `pytorch-cpu` is 117.2x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=8, shape=`8x32x32`): `tenferro-direct` is 616.0x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `jax-cpu` is 48.0x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `pytorch-cpu` is 207.6x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `tenferro-direct` is 293.4x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `jax-cpu` is 40.6x faster than the slowest successful cell (`tenferro-trace`, 35.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 108.5x faster than the slowest successful cell (`tenferro-trace`, 35.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 175.1x faster than the slowest successful cell (`tenferro-trace`, 35.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `jax-cpu` is 373.2x faster than the slowest successful cell (`tenferro-trace`, 84.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `pytorch-cpu` is 475.7x faster than the slowest successful cell (`tenferro-trace`, 84.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `tenferro-direct` is 784.8x faster than the slowest successful cell (`tenferro-trace`, 84.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `jax-cpu` is 42.8x faster than the slowest successful cell (`tenferro-trace`, 36.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 90.5x faster than the slowest successful cell (`tenferro-trace`, 36.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `tenferro-direct` is 154.5x faster than the slowest successful cell (`tenferro-trace`, 36.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x32x32,rhs=1`): `jax-cpu` is 104.7x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x32x32,rhs=1`): `pytorch-cpu` is 243.1x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x32x32,rhs=1`): `tenferro-direct` is 613.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`1024x4x4,rhs=1`): `jax-cpu` is 91.0x faster than the slowest successful cell (`tenferro-trace`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`1024x4x4,rhs=1`): `pytorch-cpu` is 437.9x faster than the slowest successful cell (`tenferro-trace`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`1024x4x4,rhs=1`): `tenferro-direct` is 1094.9x faster than the slowest successful cell (`tenferro-trace`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`1024x8x8,rhs=1`): `jax-cpu` is 100.3x faster than the slowest successful cell (`tenferro-trace`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 322.1x faster than the slowest successful cell (`tenferro-trace`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 859.2x faster than the slowest successful cell (`tenferro-trace`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`1x128x128,rhs=1`): `jax-cpu` is 745.0x faster than the slowest successful cell (`tenferro-trace`, 168.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`1x128x128,rhs=1`): `pytorch-cpu` is 936.4x faster than the slowest successful cell (`tenferro-trace`, 168.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`1x128x128,rhs=1`): `tenferro-direct` is 1531.0x faster than the slowest successful cell (`tenferro-trace`, 168.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`8x128x128,rhs=1`): `jax-cpu` is 158.8x faster than the slowest successful cell (`tenferro-trace`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 341.9x faster than the slowest successful cell (`tenferro-trace`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`8x128x128,rhs=1`): `tenferro-direct` is 752.7x faster than the slowest successful cell (`tenferro-trace`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`8x32x32,rhs=1`): `jax-cpu` is 165.8x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`8x32x32,rhs=1`): `pytorch-cpu` is 570.6x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`8x32x32,rhs=1`): `tenferro-direct` is 1984.8x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=4, shape=`1024x2x2`): `tenferro-direct` is 29.8x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=4, shape=`8x32x32`): `jax-cpu` is 18.0x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=4, shape=`8x32x32`): `tenferro-direct` is 20.7x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=8, shape=`1024x2x2`): `pytorch-cpu` is 12.3x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=8, shape=`1024x2x2`): `tenferro-direct` is 117.5x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=8, shape=`1024x4x4`): `jax-cpu` is 10.5x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=8, shape=`1024x4x4`): `tenferro-direct` is 35.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=8, shape=`1024x8x8`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=8, shape=`8x128x128`): `tenferro-direct` is 22.4x faster than the slowest successful cell (`jax-cpu`, 139.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=8, shape=`8x32x32`): `jax-cpu` is 68.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=8, shape=`8x32x32`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=8, shape=`8x32x32`): `tenferro-direct` is 94.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`1024x4x4`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 14.9x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 32.7x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=8, shape=`1024x4x4`): `jax-cpu` is 13.7x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=8, shape=`1024x4x4`): `tenferro-direct` is 41.2x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=8, shape=`1024x8x8`): `tenferro-direct` is 12.8x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=8, shape=`8x128x128`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=8, shape=`8x32x32`): `jax-cpu` is 82.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=8, shape=`8x32x32`): `pytorch-cpu` is 14.8x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=8, shape=`8x32x32`): `tenferro-direct` is 127.3x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_lu_factor` (f64, threads=8, shape=`1024x16x16,rhs=1`): `pytorch-cpu` is 14.3x faster than the slowest successful cell (`julia-base`, 2.724 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_lu_factor` (f64, threads=8, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`julia-base`, 0.805 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x16x16,rhs=1`): `jax-cpu` is 25.4x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x16x16,rhs=1`): `julia-base` is 33.5x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x16x16,rhs=1`): `pytorch-cpu` is 46.2x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x16x16,rhs=1`): `tenferro-direct` is 119.5x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `jax-cpu` is 49.5x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `julia-base` is 63.2x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 79.5x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 301.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=8, shape=`1024x16x16,rhs=1`): `jax-cpu` is 51.0x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=8, shape=`1024x16x16,rhs=1`): `julia-base` is 65.9x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=8, shape=`1024x16x16,rhs=1`): `pytorch-cpu` is 91.2x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=8, shape=`1024x16x16,rhs=1`): `tenferro-direct` is 454.9x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=8, shape=`1024x8x8,rhs=1`): `jax-cpu` is 99.6x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=8, shape=`1024x8x8,rhs=1`): `julia-base` is 123.7x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=8, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 157.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=8, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 1087.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `julia-base` is 14.1x faster than the slowest successful cell (`tenferro-trace`, 588.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 22.7x faster than the slowest successful cell (`tenferro-trace`, 588.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 40.7x faster than the slowest successful cell (`tenferro-trace`, 588.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=8, shape=`1536x1536`): `jax-cpu` is 23.7x faster than the slowest successful cell (`tenferro-trace`, 1301.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=8, shape=`1536x1536`): `julia-base` is 39.4x faster than the slowest successful cell (`tenferro-trace`, 1301.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=8, shape=`1536x1536`): `pytorch-cpu` is 97.5x faster than the slowest successful cell (`tenferro-trace`, 1301.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=8, shape=`1536x1536`): `tenferro-direct` is 132.8x faster than the slowest successful cell (`tenferro-trace`, 1301.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 105.6x faster than the slowest successful cell (`tenferro-trace`, 1848.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `julia-base` is 145.0x faster than the slowest successful cell (`tenferro-trace`, 1848.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 206.8x faster than the slowest successful cell (`tenferro-trace`, 1848.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 148.8x faster than the slowest successful cell (`tenferro-trace`, 1848.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=8, shape=`1024x1024`): `jax-cpu` is 309.5x faster than the slowest successful cell (`tenferro-trace`, 3763.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=8, shape=`1024x1024`): `julia-base` is 533.9x faster than the slowest successful cell (`tenferro-trace`, 3763.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=8, shape=`1024x1024`): `pytorch-cpu` is 662.4x faster than the slowest successful cell (`tenferro-trace`, 3763.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=8, shape=`1024x1024`): `tenferro-direct` is 374.3x faster than the slowest successful cell (`tenferro-trace`, 3763.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 360.7x faster than the slowest successful cell (`tenferro-trace`, 3775.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `julia-base` is 367.9x faster than the slowest successful cell (`tenferro-trace`, 3775.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 394.2x faster than the slowest successful cell (`tenferro-trace`, 3775.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 364.3x faster than the slowest successful cell (`tenferro-trace`, 3775.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=8, shape=`160x160`): `jax-cpu` is 687.5x faster than the slowest successful cell (`tenferro-trace`, 7248.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=8, shape=`160x160`): `julia-base` is 704.0x faster than the slowest successful cell (`tenferro-trace`, 7248.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=8, shape=`160x160`): `pytorch-cpu` is 759.1x faster than the slowest successful cell (`tenferro-trace`, 7248.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=8, shape=`160x160`): `tenferro-direct` is 691.9x faster than the slowest successful cell (`tenferro-trace`, 7248.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 244.9x faster than the slowest successful cell (`tenferro-trace`, 3151.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `julia-base` is 250.7x faster than the slowest successful cell (`tenferro-trace`, 3151.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 290.8x faster than the slowest successful cell (`tenferro-trace`, 3151.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 244.8x faster than the slowest successful cell (`tenferro-trace`, 3151.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=8, shape=`192x192`): `jax-cpu` is 480.5x faster than the slowest successful cell (`tenferro-trace`, 6184.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=8, shape=`192x192`): `julia-base` is 489.2x faster than the slowest successful cell (`tenferro-trace`, 6184.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=8, shape=`192x192`): `pytorch-cpu` is 581.1x faster than the slowest successful cell (`tenferro-trace`, 6184.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=8, shape=`192x192`): `tenferro-direct` is 477.2x faster than the slowest successful cell (`tenferro-trace`, 6184.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 171.9x faster than the slowest successful cell (`tenferro-trace`, 5737.978 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `julia-base` is 466.5x faster than the slowest successful cell (`tenferro-trace`, 5737.978 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 434.0x faster than the slowest successful cell (`tenferro-trace`, 5737.978 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 306.7x faster than the slowest successful cell (`tenferro-trace`, 5737.978 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=8, shape=`512x512`): `jax-cpu` is 364.4x faster than the slowest successful cell (`tenferro-trace`, 11316.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=8, shape=`512x512`): `julia-base` is 881.6x faster than the slowest successful cell (`tenferro-trace`, 11316.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=8, shape=`512x512`): `pytorch-cpu` is 994.3x faster than the slowest successful cell (`tenferro-trace`, 11316.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=8, shape=`512x512`): `tenferro-direct` is 620.8x faster than the slowest successful cell (`tenferro-trace`, 11316.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `jax-cpu` is 187.4x faster than the slowest successful cell (`tenferro-eager`, 204.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `pytorch-cpu` is 582.1x faster than the slowest successful cell (`tenferro-eager`, 204.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 808.8x faster than the slowest successful cell (`tenferro-eager`, 204.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `jax-cpu` is 384.9x faster than the slowest successful cell (`tenferro-eager`, 3093.436 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 655.5x faster than the slowest successful cell (`tenferro-eager`, 3093.436 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `tenferro-direct` is 1034.2x faster than the slowest successful cell (`tenferro-eager`, 3093.436 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=8, shape=`256x128`): `jax-cpu` is 363.5x faster than the slowest successful cell (`tenferro-eager`, 410.561 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=8, shape=`256x128`): `pytorch-cpu` is 1177.4x faster than the slowest successful cell (`tenferro-eager`, 410.561 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=8, shape=`256x128`): `tenferro-direct` is 1677.5x faster than the slowest successful cell (`tenferro-eager`, 410.561 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=8, shape=`768x384`): `jax-cpu` is 803.9x faster than the slowest successful cell (`tenferro-eager`, 6264.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=8, shape=`768x384`): `pytorch-cpu` is 2240.1x faster than the slowest successful cell (`tenferro-eager`, 6264.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=8, shape=`768x384`): `tenferro-direct` is 2595.1x faster than the slowest successful cell (`tenferro-eager`, 6264.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 76.4x faster than the slowest successful cell (`tenferro-trace`, 1584.990 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `julia-base` is 102.5x faster than the slowest successful cell (`tenferro-trace`, 1584.990 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 99.7x faster than the slowest successful cell (`tenferro-trace`, 1584.990 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 118.6x faster than the slowest successful cell (`tenferro-trace`, 1584.990 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=8, shape=`768x768`): `jax-cpu` is 253.4x faster than the slowest successful cell (`tenferro-trace`, 3212.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=8, shape=`768x768`): `julia-base` is 331.3x faster than the slowest successful cell (`tenferro-trace`, 3212.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=8, shape=`768x768`): `pytorch-cpu` is 333.7x faster than the slowest successful cell (`tenferro-trace`, 3212.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=8, shape=`768x768`): `tenferro-direct` is 365.3x faster than the slowest successful cell (`tenferro-trace`, 3212.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `jax-cpu` is 26.8x faster than the slowest successful cell (`tenferro-trace`, 1092.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `julia-base` is 183.6x faster than the slowest successful cell (`tenferro-trace`, 1092.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 209.2x faster than the slowest successful cell (`tenferro-trace`, 1092.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 140.3x faster than the slowest successful cell (`tenferro-trace`, 1092.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=8, shape=`768x384,rhs=16`): `jax-cpu` is 66.3x faster than the slowest successful cell (`tenferro-trace`, 2494.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=8, shape=`768x384,rhs=16`): `julia-base` is 421.4x faster than the slowest successful cell (`tenferro-trace`, 2494.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=8, shape=`768x384,rhs=16`): `tenferro-eager` is 410.8x faster than the slowest successful cell (`tenferro-trace`, 2494.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 53.4x faster than the slowest successful cell (`tenferro-trace`, 1777.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `julia-base` is 80.0x faster than the slowest successful cell (`tenferro-trace`, 1777.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 131.3x faster than the slowest successful cell (`tenferro-trace`, 1777.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 138.1x faster than the slowest successful cell (`tenferro-trace`, 1777.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=8, shape=`1024x1024`): `jax-cpu` is 132.3x faster than the slowest successful cell (`tenferro-trace`, 3570.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=8, shape=`1024x1024`): `julia-base` is 185.7x faster than the slowest successful cell (`tenferro-trace`, 3570.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=8, shape=`1024x1024`): `pytorch-cpu` is 383.5x faster than the slowest successful cell (`tenferro-trace`, 3570.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=8, shape=`1024x1024`): `tenferro-direct` is 350.4x faster than the slowest successful cell (`tenferro-trace`, 3570.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 14.5x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 24.4x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=8, shape=`2048x2048`): `julia-base` is 17.0x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=8, shape=`2048x2048`): `pytorch-cpu` is 43.4x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=8, shape=`2048x2048`): `tenferro-direct` is 108.1x faster than the slowest successful cell (`tenferro-trace`, 71.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=8, shape=`512x256`): `jax-cpu` is 11.3x faster than the slowest successful cell (`tenferro-trace`, 239.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=8, shape=`512x256`): `julia-base` is 12.7x faster than the slowest successful cell (`tenferro-trace`, 239.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=8, shape=`512x256`): `pytorch-cpu` is 16.0x faster than the slowest successful cell (`tenferro-trace`, 239.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=8, shape=`512x256`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`tenferro-trace`, 239.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=8, shape=`512x256`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-trace`, 235.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=8, shape=`512x256`): `julia-base` is 12.7x faster than the slowest successful cell (`tenferro-trace`, 235.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=8, shape=`512x256`): `pytorch-cpu` is 15.8x faster than the slowest successful cell (`tenferro-trace`, 235.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=8, shape=`512x256`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 235.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 1523.0x faster than the slowest successful cell (`tenferro-eager`, 1293.372 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr` (f64, threads=4, shape=`768x384`): `tenferro-direct` is 461.4x faster than the slowest successful cell (`tenferro-eager`, 4852.559 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr` (f64, threads=8, shape=`256x128`): `tenferro-direct` is 1528.5x faster than the slowest successful cell (`tenferro-eager`, 1296.685 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr` (f64, threads=8, shape=`768x384`): `tenferro-direct` is 1281.6x faster than the slowest successful cell (`tenferro-eager`, 9672.810 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr_deficient` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 1543.0x faster than the slowest successful cell (`tenferro-eager`, 1302.497 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr_deficient` (f64, threads=4, shape=`768x384`): `tenferro-direct` is 472.3x faster than the slowest successful cell (`tenferro-eager`, 4863.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr_deficient` (f64, threads=8, shape=`256x128`): `tenferro-direct` is 1506.4x faster than the slowest successful cell (`tenferro-eager`, 1300.872 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr_deficient` (f64, threads=8, shape=`768x384`): `tenferro-direct` is 1285.5x faster than the slowest successful cell (`tenferro-eager`, 9690.934 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 103.9x faster than the slowest successful cell (`tenferro-trace`, 1836.991 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `julia-base` is 144.3x faster than the slowest successful cell (`tenferro-trace`, 1836.991 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 206.2x faster than the slowest successful cell (`tenferro-trace`, 1836.991 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 152.0x faster than the slowest successful cell (`tenferro-trace`, 1836.991 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=8, shape=`1024x1024`): `jax-cpu` is 309.6x faster than the slowest successful cell (`tenferro-trace`, 3772.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=8, shape=`1024x1024`): `julia-base` is 540.1x faster than the slowest successful cell (`tenferro-trace`, 3772.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=8, shape=`1024x1024`): `pytorch-cpu` is 679.3x faster than the slowest successful cell (`tenferro-trace`, 3772.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=8, shape=`1024x1024`): `tenferro-direct` is 381.5x faster than the slowest successful cell (`tenferro-trace`, 3772.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 180.2x faster than the slowest successful cell (`tenferro-trace`, 9487.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `julia-base` is 181.1x faster than the slowest successful cell (`tenferro-trace`, 9487.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 282.3x faster than the slowest successful cell (`tenferro-trace`, 9487.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 368.3x faster than the slowest successful cell (`tenferro-trace`, 9487.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=8, shape=`768x384`): `jax-cpu` is 406.5x faster than the slowest successful cell (`tenferro-trace`, 19318.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=8, shape=`768x384`): `julia-base` is 401.8x faster than the slowest successful cell (`tenferro-trace`, 19318.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=8, shape=`768x384`): `pytorch-cpu` is 657.8x faster than the slowest successful cell (`tenferro-trace`, 19318.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=8, shape=`768x384`): `tenferro-eager` is 968.6x faster than the slowest successful cell (`tenferro-trace`, 19318.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-trace`, 1585.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `julia-base` is 96.5x faster than the slowest successful cell (`tenferro-trace`, 1585.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `pytorch-cpu` is 108.5x faster than the slowest successful cell (`tenferro-trace`, 1585.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `tenferro-direct` is 104.5x faster than the slowest successful cell (`tenferro-trace`, 1585.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=8, shape=`4096x4096,rhs=64`): `jax-cpu` is 27.8x faster than the slowest successful cell (`tenferro-trace`, 3169.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=8, shape=`4096x4096,rhs=64`): `julia-base` is 257.2x faster than the slowest successful cell (`tenferro-trace`, 3169.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=8, shape=`4096x4096,rhs=64`): `pytorch-cpu` is 261.8x faster than the slowest successful cell (`tenferro-trace`, 3169.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=8, shape=`4096x4096,rhs=64`): `tenferro-direct` is 278.2x faster than the slowest successful cell (`tenferro-trace`, 3169.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 659.5x faster than the slowest successful cell (`tenferro-eager`, 24.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 438.5x faster than the slowest successful cell (`tenferro-eager`, 24.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 651.8x faster than the slowest successful cell (`tenferro-eager`, 24.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 27.4x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 31.9x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 45.2x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=8, shape=`1024x64`): `jax-cpu` is 1899.7x faster than the slowest successful cell (`tenferro-eager`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=8, shape=`1024x64`): `pytorch-cpu` is 2003.2x faster than the slowest successful cell (`tenferro-eager`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 3376.3x faster than the slowest successful cell (`tenferro-eager`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=8, shape=`4096x256`): `jax-cpu` is 130.9x faster than the slowest successful cell (`tenferro-eager`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=8, shape=`4096x256`): `pytorch-cpu` is 176.7x faster than the slowest successful cell (`tenferro-eager`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=8, shape=`4096x256`): `tenferro-direct` is 262.3x faster than the slowest successful cell (`tenferro-eager`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 127.8x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 548.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 674.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 15.1x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 29.1x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 46.0x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=8, shape=`1024x64`): `jax-cpu` is 390.2x faster than the slowest successful cell (`tenferro-eager`, 24.343 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=8, shape=`1024x64`): `pytorch-cpu` is 510.2x faster than the slowest successful cell (`tenferro-eager`, 24.343 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 2296.8x faster than the slowest successful cell (`tenferro-eager`, 24.343 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=8, shape=`4096x256`): `jax-cpu` is 54.3x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=8, shape=`4096x256`): `pytorch-cpu` is 89.7x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=8, shape=`4096x256`): `tenferro-direct` is 159.3x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 134.9x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 214.2x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 242.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 13.1x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 11.5x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 15.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=8, shape=`1024x64`): `jax-cpu` is 270.7x faster than the slowest successful cell (`tenferro-eager`, 24.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=8, shape=`1024x64`): `pytorch-cpu` is 546.0x faster than the slowest successful cell (`tenferro-eager`, 24.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 907.3x faster than the slowest successful cell (`tenferro-eager`, 24.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=8, shape=`4096x256`): `jax-cpu` is 34.9x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=8, shape=`4096x256`): `pytorch-cpu` is 43.5x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=8, shape=`4096x256`): `tenferro-direct` is 61.8x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 536.1x faster than the slowest successful cell (`tenferro-eager`, 71.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1868.0x faster than the slowest successful cell (`tenferro-eager`, 71.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 1242.4x faster than the slowest successful cell (`tenferro-eager`, 71.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 90.9x faster than the slowest successful cell (`tenferro-eager`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 286.8x faster than the slowest successful cell (`tenferro-eager`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 304.8x faster than the slowest successful cell (`tenferro-eager`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=8, shape=`1024x64`): `jax-cpu` is 649.3x faster than the slowest successful cell (`tenferro-eager`, 90.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=8, shape=`1024x64`): `pytorch-cpu` is 2448.2x faster than the slowest successful cell (`tenferro-eager`, 90.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 1463.0x faster than the slowest successful cell (`tenferro-eager`, 90.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=8, shape=`4096x256`): `jax-cpu` is 249.7x faster than the slowest successful cell (`tenferro-eager`, 210.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=8, shape=`4096x256`): `pytorch-cpu` is 1210.6x faster than the slowest successful cell (`tenferro-eager`, 210.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=8, shape=`4096x256`): `tenferro-direct` is 554.9x faster than the slowest successful cell (`tenferro-eager`, 210.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 429.4x faster than the slowest successful cell (`tenferro-eager`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 4590.4x faster than the slowest successful cell (`tenferro-eager`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 1378.5x faster than the slowest successful cell (`tenferro-eager`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 64.2x faster than the slowest successful cell (`tenferro-eager`, 120.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 439.3x faster than the slowest successful cell (`tenferro-eager`, 120.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 263.5x faster than the slowest successful cell (`tenferro-eager`, 120.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=8, shape=`1024x64`): `jax-cpu` is 756.6x faster than the slowest successful cell (`tenferro-eager`, 171.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=8, shape=`1024x64`): `pytorch-cpu` is 5561.0x faster than the slowest successful cell (`tenferro-eager`, 171.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 2426.0x faster than the slowest successful cell (`tenferro-eager`, 171.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=8, shape=`4096x256`): `jax-cpu` is 200.3x faster than the slowest successful cell (`tenferro-eager`, 282.375 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=8, shape=`4096x256`): `pytorch-cpu` is 1793.4x faster than the slowest successful cell (`tenferro-eager`, 282.375 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=8, shape=`4096x256`): `tenferro-direct` is 546.3x faster than the slowest successful cell (`tenferro-eager`, 282.375 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 478.1x faster than the slowest successful cell (`tenferro-eager`, 119.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 959.5x faster than the slowest successful cell (`tenferro-eager`, 119.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 60.9x faster than the slowest successful cell (`tenferro-eager`, 144.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 129.8x faster than the slowest successful cell (`tenferro-eager`, 144.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=8, shape=`1024x64`): `jax-cpu` is 992.9x faster than the slowest successful cell (`tenferro-eager`, 252.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 2361.0x faster than the slowest successful cell (`tenferro-eager`, 252.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=8, shape=`4096x256`): `jax-cpu` is 178.3x faster than the slowest successful cell (`tenferro-eager`, 354.498 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=8, shape=`4096x256`): `tenferro-direct` is 423.9x faster than the slowest successful cell (`tenferro-eager`, 354.498 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 648.5x faster than the slowest successful cell (`tenferro-eager`, 24.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 2423.8x faster than the slowest successful cell (`tenferro-eager`, 24.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 2494.8x faster than the slowest successful cell (`tenferro-eager`, 24.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 107.2x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 772.3x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 531.4x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=8, shape=`1024x64`): `jax-cpu` is 655.1x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=8, shape=`1024x64`): `pytorch-cpu` is 2561.0x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 2377.1x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=8, shape=`4096x256`): `jax-cpu` is 437.9x faster than the slowest successful cell (`tenferro-eager`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=8, shape=`4096x256`): `pytorch-cpu` is 4057.2x faster than the slowest successful cell (`tenferro-eager`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=8, shape=`4096x256`): `tenferro-direct` is 2449.3x faster than the slowest successful cell (`tenferro-eager`, 72.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 316.1x faster than the slowest successful cell (`tenferro-eager`, 35.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 963.1x faster than the slowest successful cell (`tenferro-eager`, 35.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 1156.7x faster than the slowest successful cell (`tenferro-eager`, 35.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 46.1x faster than the slowest successful cell (`tenferro-eager`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 212.7x faster than the slowest successful cell (`tenferro-eager`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 271.7x faster than the slowest successful cell (`tenferro-eager`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=8, shape=`1024x64`): `jax-cpu` is 379.5x faster than the slowest successful cell (`tenferro-eager`, 44.781 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=8, shape=`1024x64`): `pytorch-cpu` is 667.9x faster than the slowest successful cell (`tenferro-eager`, 44.781 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 1291.5x faster than the slowest successful cell (`tenferro-eager`, 44.781 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=8, shape=`4096x256`): `jax-cpu` is 132.8x faster than the slowest successful cell (`tenferro-eager`, 105.750 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=8, shape=`4096x256`): `pytorch-cpu` is 542.5x faster than the slowest successful cell (`tenferro-eager`, 105.750 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=8, shape=`4096x256`): `tenferro-direct` is 560.8x faster than the slowest successful cell (`tenferro-eager`, 105.750 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 151.8x faster than the slowest successful cell (`tenferro-eager`, 12.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 234.5x faster than the slowest successful cell (`tenferro-eager`, 12.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 936.2x faster than the slowest successful cell (`tenferro-eager`, 12.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 15.8x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 30.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 66.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=8, shape=`1024x64`): `jax-cpu` is 273.8x faster than the slowest successful cell (`tenferro-eager`, 24.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=8, shape=`1024x64`): `pytorch-cpu` is 459.8x faster than the slowest successful cell (`tenferro-eager`, 24.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 3049.5x faster than the slowest successful cell (`tenferro-eager`, 24.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=8, shape=`4096x256`): `jax-cpu` is 40.1x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=8, shape=`4096x256`): `pytorch-cpu` is 97.6x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=8, shape=`4096x256`): `tenferro-direct` is 184.2x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 146.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 229.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 897.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 14.8x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 29.5x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 63.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=8, shape=`1024x64`): `jax-cpu` is 267.5x faster than the slowest successful cell (`tenferro-eager`, 24.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=8, shape=`1024x64`): `pytorch-cpu` is 441.3x faster than the slowest successful cell (`tenferro-eager`, 24.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 2871.6x faster than the slowest successful cell (`tenferro-eager`, 24.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=8, shape=`4096x256`): `jax-cpu` is 39.3x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=8, shape=`4096x256`): `pytorch-cpu` is 93.9x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=8, shape=`4096x256`): `tenferro-direct` is 183.5x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 436.4x faster than the slowest successful cell (`tenferro-eager`, 96.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 4090.0x faster than the slowest successful cell (`tenferro-eager`, 96.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 1403.9x faster than the slowest successful cell (`tenferro-eager`, 96.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 65.6x faster than the slowest successful cell (`tenferro-eager`, 120.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 396.9x faster than the slowest successful cell (`tenferro-eager`, 120.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 262.4x faster than the slowest successful cell (`tenferro-eager`, 120.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=8, shape=`1024x64`): `jax-cpu` is 775.4x faster than the slowest successful cell (`tenferro-eager`, 171.609 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=8, shape=`1024x64`): `pytorch-cpu` is 5955.1x faster than the slowest successful cell (`tenferro-eager`, 171.609 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 2508.5x faster than the slowest successful cell (`tenferro-eager`, 171.609 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=8, shape=`4096x256`): `jax-cpu` is 205.4x faster than the slowest successful cell (`tenferro-eager`, 280.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=8, shape=`4096x256`): `pytorch-cpu` is 1748.0x faster than the slowest successful cell (`tenferro-eager`, 280.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=8, shape=`4096x256`): `tenferro-direct` is 551.1x faster than the slowest successful cell (`tenferro-eager`, 280.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 63.9x faster than the slowest successful cell (`tenferro-eager`, 12.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 105.9x faster than the slowest successful cell (`tenferro-eager`, 12.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 290.6x faster than the slowest successful cell (`tenferro-eager`, 12.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 13.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 18.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=8, shape=`1024x64`): `jax-cpu` is 123.4x faster than the slowest successful cell (`tenferro-eager`, 24.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=8, shape=`1024x64`): `pytorch-cpu` is 204.7x faster than the slowest successful cell (`tenferro-eager`, 24.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 1057.1x faster than the slowest successful cell (`tenferro-eager`, 24.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=8, shape=`4096x256`): `jax-cpu` is 13.1x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=8, shape=`4096x256`): `pytorch-cpu` is 50.8x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=8, shape=`4096x256`): `tenferro-direct` is 73.3x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 1259.0x faster than the slowest successful cell (`tenferro-eager`, 107.671 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 584.0x faster than the slowest successful cell (`tenferro-eager`, 107.671 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 330.5x faster than the slowest successful cell (`tenferro-eager`, 107.671 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 159.3x faster than the slowest successful cell (`tenferro-eager`, 131.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 71.2x faster than the slowest successful cell (`tenferro-eager`, 131.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 23.9x faster than the slowest successful cell (`tenferro-eager`, 131.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=8, shape=`1024x64`): `jax-cpu` is 3673.7x faster than the slowest successful cell (`tenferro-eager`, 272.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=8, shape=`1024x64`): `pytorch-cpu` is 1462.0x faster than the slowest successful cell (`tenferro-eager`, 272.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 795.8x faster than the slowest successful cell (`tenferro-eager`, 272.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=8, shape=`4096x256`): `jax-cpu` is 528.6x faster than the slowest successful cell (`tenferro-eager`, 352.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=8, shape=`4096x256`): `pytorch-cpu` is 342.2x faster than the slowest successful cell (`tenferro-eager`, 352.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=8, shape=`4096x256`): `tenferro-direct` is 75.0x faster than the slowest successful cell (`tenferro-eager`, 352.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 44386.1x faster than the slowest successful cell (`julia-base`, 100.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=8, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 43414.4x faster than the slowest successful cell (`julia-base`, 100.461 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `pytorch-cpu` is 14.7x faster than the slowest successful cell (`tenferro-trace`, 201.620 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `tenferro-direct` is 14.7x faster than the slowest successful cell (`tenferro-trace`, 201.620 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=8, shape=`33554432`): `jax-cpu` is 13.5x faster than the slowest successful cell (`tenferro-trace`, 290.378 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=8, shape=`33554432`): `pytorch-cpu` is 28.6x faster than the slowest successful cell (`tenferro-trace`, 290.378 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=8, shape=`33554432`): `tenferro-direct` is 26.2x faster than the slowest successful cell (`tenferro-trace`, 290.378 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 44315.2x faster than the slowest successful cell (`julia-base`, 105.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=8, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 43229.7x faster than the slowest successful cell (`julia-base`, 104.789 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`julia-base`, 224.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 98122.4x faster than the slowest successful cell (`julia-base`, 224.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=8, shape=`4096x4096`): `tenferro-direct` is 14.4x faster than the slowest successful cell (`julia-base`, 223.546 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=8, shape=`4096x4096`): `tenferro-trace` is 98305.1x faster than the slowest successful cell (`julia-base`, 223.546 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 82.9x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=8, shape=`33554432 -> 8192x4096`): `julia-base` is 50.3x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 103.2x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=8, shape=`4194304 -> 2096128`): `julia-base` is 79.0x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 210.4x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=8, shape=`4096x4096`): `julia-base` is 183.4x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 6858.5 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 17.1x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 119092.9 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 297.7x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=8, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 116005.0 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 145.0x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 113025.5 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 282.6x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=8, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 110740.7 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 138.4x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 58764.3 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 146.9x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=8, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 59022.7 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 73.8x faster than the bound.

# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/public_api/20261031_112656/run.yaml`
- Timestamp: `2026-10-31T02:26:56.240637+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4 8`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/zen5-cpu/cpu/public_api/20261031_112656`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

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

## Threads: 1 4 8

- CSV: `data/results/zen5-cpu/cpu/public_api/20261031_112656/cpu_public_api_t1_20261031_112656.csv`
- CSV: `data/results/zen5-cpu/cpu/public_api/20261031_112656/cpu_public_api_t4_20261031_112656.csv`
- CSV: `data/results/zen5-cpu/cpu/public_api/20261031_112656/cpu_public_api_t8_20261031_112656.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 5.824 ± 0.004 | - | 5.966 ± 0.005 | 7.734 ± 0.026 | 6.585 ± 0.017 | 6.476 ± 1.746 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.286 ± 0.008 | - | 12.597 ± 1.131 | 2.121 ± 0.017 | 3.270 ± 0.860 | 2.636 ± 0.039 | - |
| cpu/complex | `cholesky` | c64 | 8 | `448x448` | 1.731 ± 0.009 | - | 12.196 ± 0.066 | 1.174 ± 0.010 | 2.779 ± 0.130 | 1.993 ± 0.031 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 91.509 ± 0.562 | - | 89.539 ± 1.430 | 83.165 ± 0.851 | 101.066 ± 1.232 | 96.673 ± 2.377 | 108.839 ± 2.559 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 27.037 ± 0.174 | - | 26.483 ± 0.188 | 25.962 ± 0.167 | 48.337 ± 0.370 | 97.458 ± 2.359 | 43.504 ± 3.635 |
| cpu/complex | `conj` | c64 | 8 | `16777216` | 17.378 ± 0.429 | - | 16.840 ± 0.158 | 17.595 ± 0.172 | 34.745 ± 0.757 | 97.552 ± 1.380 | 34.429 ± 1.574 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 92.908 ± 0.623 | - | 98.559 ± 0.323 | 89.981 ± 0.352 | 97.222 ± 0.446 | 111.206 ± 0.317 | 122.142 ± 0.648 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 24.842 ± 0.238 | - | 24.644 ± 0.241 | 24.336 ± 0.188 | 44.754 ± 0.750 | 112.499 ± 0.751 | 38.432 ± 1.531 |
| cpu/complex | `div` | c64 | 8 | `8388608` | 13.736 ± 0.279 | - | 13.293 ± 0.170 | 13.748 ± 0.322 | 32.257 ± 0.300 | 115.893 ± 0.650 | 27.004 ± 1.785 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 81.636 ± 0.111 | - | 81.632 ± 0.111 | 78.404 ± 0.106 | 95.472 ± 0.138 | 83.442 ± 0.996 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 20.405 ± 0.027 | - | 20.427 ± 0.031 | 26.633 ± 0.078 | 24.601 ± 0.200 | 21.222 ± 0.683 | - |
| cpu/complex | `dot_general` | c64 | 8 | `640x640` | 10.451 ± 0.048 | - | 10.483 ± 0.017 | 10.645 ± 0.063 | 14.521 ± 1.345 | 11.238 ± 0.517 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 82.802 ± 0.178 | - | unsupported | 79.086 ± 0.241 | 101.949 ± 3.576 | 82.282 ± 0.107 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 21.655 ± 0.042 | - | unsupported | 27.116 ± 0.041 | 24.980 ± 0.255 | 21.126 ± 2.766 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 8 | `640x640` | 11.853 ± 0.029 | - | unsupported | 11.204 ± 0.095 | 13.807 ± 0.599 | 11.303 ± 2.533 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 16.617 ± 0.053 | - | 16.593 ± 0.060 | 15.963 ± 0.026 | 16.445 ± 0.026 | 16.483 ± 0.037 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 16.557 ± 0.030 | - | 27.059 ± 0.021 | 13.452 ± 0.021 | 16.451 ± 0.023 | 16.569 ± 0.037 | - |
| cpu/complex | `eig` | c64 | 8 | `112x112` | 16.777 ± 0.029 | - | 27.306 ± 0.028 | 12.975 ± 0.024 | 16.616 ± 0.113 | 16.765 ± 0.052 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 105.590 ± 0.127 | - | 105.388 ± 0.109 | 128.288 ± 0.379 | 107.116 ± 0.819 | 81.643 ± 0.412 | 84.055 ± 5.706 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 26.288 ± 0.096 | - | 26.279 ± 0.055 | 34.128 ± 0.084 | 54.025 ± 0.178 | 81.367 ± 0.770 | 24.608 ± 1.950 |
| cpu/complex | `exp` | c64 | 8 | `4194304` | 13.259 ± 0.023 | - | 13.254 ± 0.021 | 17.674 ± 0.448 | 38.080 ± 0.323 | 82.694 ± 0.322 | 15.532 ± 3.526 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 137.172 ± 0.203 | - | 136.791 ± 0.365 | 173.250 ± 0.295 | 209.160 ± 0.507 | 154.093 ± 0.472 | 160.885 ± 0.534 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 33.798 ± 0.067 | - | 33.896 ± 0.083 | 44.413 ± 0.588 | 51.591 ± 0.689 | 154.642 ± 0.287 | 44.463 ± 1.100 |
| cpu/complex | `log` | c64 | 8 | `4194304` | 17.134 ± 0.071 | - | 17.111 ± 0.061 | 23.113 ± 0.601 | 26.655 ± 0.555 | 156.633 ± 0.282 | 27.021 ± 1.063 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 49.148 ± 0.127 | - | 51.027 ± 0.646 | 52.766 ± 0.217 | 61.513 ± 0.368 | 53.739 ± 0.322 | 60.809 ± 0.430 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 15.733 ± 0.300 | - | 15.403 ± 0.332 | 15.021 ± 0.100 | 29.477 ± 0.116 | 54.097 ± 0.158 | 24.331 ± 2.151 |
| cpu/complex | `mul` | c64 | 8 | `8388608` | 11.279 ± 0.362 | - | 10.933 ± 0.251 | 11.054 ± 0.169 | 21.292 ± 0.194 | 56.171 ± 0.202 | 22.793 ± 3.448 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.804 ± 0.008 | - | 23.888 ± 0.068 | 25.452 ± 0.053 | 15.465 ± 0.060 | 6.357 ± 0.011 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 2.982 ± 0.144 | - | 6.155 ± 0.043 | 6.379 ± 0.036 | 3.998 ± 0.061 | 6.358 ± 0.015 | - |
| cpu/complex | `norm_fro` | c64 | 8 | `2048x1536` | 3.106 ± 7.549 | - | 3.350 ± 0.073 | 3.216 ± 0.016 | 3.343 ± 0.018 | 6.364 ± 0.042 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 10.368 ± 0.017 | - | 10.391 ± 0.030 | 14.050 ± 0.046 | 11.378 ± 0.044 | 14.374 ± 0.238 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 5.860 ± 0.019 | - | 18.479 ± 3.119 | 5.600 ± 0.020 | 6.960 ± 0.031 | 7.745 ± 0.065 | - |
| cpu/complex | `qr` | c64 | 8 | `256x256` | 5.121 ± 0.010 | - | 21.745 ± 6.915 | 4.426 ± 0.042 | 6.315 ± 0.091 | 6.820 ± 0.087 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 7.632 ± 0.016 | - | 7.738 ± 0.042 | 8.352 ± 0.029 | 8.151 ± 0.022 | 7.842 ± 0.039 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 2.999 ± 0.020 | - | 16.965 ± 1.002 | 2.735 ± 0.017 | 3.642 ± 0.031 | 3.146 ± 0.015 | - |
| cpu/complex | `solve` | c64 | 8 | `384x384,rhs=8` | 2.038 ± 0.013 | - | 16.008 ± 0.036 | 2.059 ± 0.020 | 2.941 ± 0.109 | 2.267 ± 0.123 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 7.367 ± 0.016 | - | 7.413 ± 0.012 | 10.769 ± 0.032 | 7.645 ± 0.046 | 7.541 ± 0.032 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 5.092 ± 0.026 | - | 15.084 ± 0.046 | 4.954 ± 0.024 | 5.363 ± 0.024 | 5.091 ± 0.251 | - |
| cpu/complex | `svd` | c64 | 8 | `160x160` | 4.839 ± 0.024 | - | 18.028 ± 1.003 | 4.062 ± 0.055 | 5.235 ± 0.057 | 4.845 ± 0.029 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 81.652 ± 0.130 | - | 81.717 ± 0.164 | 78.450 ± 0.104 | 100.302 ± 0.144 | 81.662 ± 0.139 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 20.414 ± 0.033 | - | 20.406 ± 0.057 | 26.731 ± 0.035 | 24.552 ± 0.277 | 20.555 ± 0.085 | - |
| cpu/complex | `tensordot` | c64 | 8 | `640x640` | 10.459 ± 0.013 | - | 10.481 ± 0.021 | 10.737 ± 0.024 | 13.336 ± 1.503 | 10.749 ± 0.063 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 80.698 ± 0.106 | - | unsupported | 76.042 ± 0.128 | 74.995 ± 0.082 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 20.505 ± 0.026 | - | unsupported | 21.769 ± 0.086 | 19.284 ± 0.124 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 8 | `1024x1024` | 10.801 ± 0.018 | - | unsupported | 12.101 ± 0.083 | 11.150 ± 0.901 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 85.537 ± 1.677 | - | 84.110 ± 2.209 | 83.620 ± 1.060 | 100.178 ± 2.791 | 92.556 ± 4.369 | 149.179 ± 1.340 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 24.727 ± 0.188 | - | 24.476 ± 0.262 | 24.617 ± 0.119 | 53.652 ± 1.507 | 93.199 ± 2.138 | 53.271 ± 3.951 |
| cpu/elementwise_reduction | `abs` | f64 | 8 | `33554432` | 15.773 ± 0.081 | - | 15.605 ± 0.167 | 17.372 ± 0.318 | 34.854 ± 0.251 | 99.273 ± 1.488 | 45.677 ± 1.869 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 94.792 ± 1.164 | - | 92.003 ± 1.121 | 96.921 ± 0.447 | 103.755 ± 16.783 | 99.823 ± 1.214 | 173.283 ± 0.660 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 27.896 ± 0.226 | - | 26.904 ± 0.134 | 28.772 ± 0.672 | 63.547 ± 2.517 | 99.889 ± 1.582 | 58.033 ± 2.857 |
| cpu/elementwise_reduction | `add` | f64 | 8 | `33554432` | 20.749 ± 0.328 | - | 19.480 ± 0.106 | 21.794 ± 0.230 | 39.053 ± 1.969 | 101.340 ± 4.137 | 47.125 ± 5.676 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 22.039 ± 0.065 | - | 60.801 ± 0.142 | 58.574 ± 0.215 | 56.843 ± 0.137 | 84.272 ± 1.111 | 92.332 ± 1.059 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 6.937 ± 0.034 | - | 16.975 ± 0.164 | 20.485 ± 0.356 | 29.769 ± 0.486 | 84.815 ± 0.587 | 23.529 ± 8.566 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 8 | `4194304` | 5.908 ± 0.055 | - | 10.753 ± 0.048 | 16.758 ± 0.844 | 20.339 ± 0.347 | 85.275 ± 0.849 | 12.476 ± 13.255 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 9.325 ± 0.056 | - | 9.046 ± 0.080 | 28.269 ± 0.200 | 30.194 ± 4.358 | 28.907 ± 5.677 | 46.555 ± 8.523 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 5.662 ± 0.025 | - | 5.702 ± 0.051 | 8.676 ± 0.084 | 16.914 ± 0.384 | 28.909 ± 5.757 | 13.347 ± 6.741 |
| cpu/elementwise_reduction | `clamp` | f64 | 8 | `8388608` | 6.105 ± 0.021 | - | 6.096 ± 0.042 | 6.739 ± 0.214 | 12.255 ± 0.116 | 25.528 ± 8.596 | 12.808 ± 11.426 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 15.899 ± 0.061 | - | 15.654 ± 0.077 | 37.627 ± 0.076 | 32.518 ± 0.092 | 16.470 ± 0.101 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 11.522 ± 0.017 | - | 11.513 ± 0.106 | 12.970 ± 0.147 | 19.722 ± 0.174 | 16.404 ± 0.053 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 8 | `33554432` | 11.654 ± 0.032 | - | 11.676 ± 0.049 | 12.306 ± 0.159 | 14.475 ± 0.061 | 16.896 ± 0.116 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 25.063 ± 0.090 | - | 25.410 ± 0.139 | 60.954 ± 0.540 | 150.123 ± 0.479 | 73.565 ± 8.456 | 79.417 ± 8.497 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 6.162 ± 0.023 | - | 6.197 ± 0.012 | 17.523 ± 0.179 | 75.434 ± 0.273 | 73.883 ± 6.529 | 21.658 ± 8.304 |
| cpu/elementwise_reduction | `cos` | f64 | 8 | `8388608` | 3.502 ± 0.161 | - | 3.495 ± 0.205 | 9.742 ± 0.488 | 52.096 ± 0.123 | 74.454 ± 7.025 | 11.299 ± 11.283 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 96.043 ± 0.915 | - | 94.813 ± 1.250 | 96.667 ± 0.579 | 115.308 ± 2.854 | 103.212 ± 1.275 | 181.326 ± 1.284 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 27.872 ± 0.139 | - | 26.817 ± 0.159 | 28.427 ± 0.171 | 60.354 ± 0.978 | 102.694 ± 1.788 | 57.279 ± 4.139 |
| cpu/elementwise_reduction | `div` | f64 | 8 | `33554432` | 19.803 ± 0.178 | - | 19.511 ± 0.160 | 21.355 ± 0.164 | 39.679 ± 0.460 | 111.467 ± 2.512 | 47.501 ± 6.045 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 14.373 ± 0.052 | - | 14.368 ± 0.040 | 43.739 ± 0.351 | 32.226 ± 0.482 | 60.228 ± 8.256 | 65.994 ± 8.211 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 3.529 ± 0.024 | - | 3.548 ± 0.008 | 12.858 ± 0.176 | 17.669 ± 0.314 | 65.191 ± 5.715 | 22.946 ± 8.113 |
| cpu/elementwise_reduction | `exp` | f64 | 8 | `8388608` | 3.387 ± 0.115 | - | 3.636 ± 2.404 | 7.463 ± 0.309 | 12.602 ± 0.299 | 62.800 ± 8.126 | 15.485 ± 9.890 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 19.807 ± 0.050 | - | 19.826 ± 0.054 | 26.715 ± 0.148 | 23.472 ± 0.057 | 37.198 ± 0.418 | 41.182 ± 5.700 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 4.983 ± 0.025 | - | 5.001 ± 0.016 | 7.464 ± 0.201 | 12.726 ± 0.054 | 37.653 ± 0.463 | 11.321 ± 8.832 |
| cpu/elementwise_reduction | `expm1` | f64 | 8 | `4194304` | 2.540 ± 0.022 | - | 2.533 ± 0.019 | 4.038 ± 0.263 | 8.567 ± 0.245 | 37.987 ± 0.359 | 6.348 ± 7.834 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 25.317 ± 0.084 | - | 25.679 ± 0.049 | 52.095 ± 0.148 | 64.616 ± 0.348 | 77.263 ± 5.554 | 89.250 ± 5.212 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 6.265 ± 0.014 | - | 6.279 ± 0.015 | 14.631 ± 0.352 | 34.408 ± 0.895 | 77.134 ± 9.184 | 24.390 ± 10.916 |
| cpu/elementwise_reduction | `log` | f64 | 8 | `8388608` | 3.351 ± 0.137 | - | 3.554 ± 0.192 | 8.055 ± 0.253 | 22.626 ± 0.205 | 81.770 ± 6.749 | 11.446 ± 14.545 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 12.008 ± 0.027 | - | 12.012 ± 0.033 | 23.421 ± 0.122 | 37.182 ± 0.110 | 45.206 ± 0.591 | 49.289 ± 5.920 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.017 ± 0.008 | - | 3.030 ± 0.009 | 6.675 ± 0.193 | 18.400 ± 0.603 | 45.453 ± 1.100 | 13.602 ± 7.300 |
| cpu/elementwise_reduction | `log1p` | f64 | 8 | `4194304` | 1.668 ± 0.070 | - | 1.649 ± 0.046 | 3.595 ± 0.255 | 12.849 ± 0.194 | 45.476 ± 0.369 | 6.790 ± 8.843 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.362 ± 0.003 | 0.393 ± 0.001 | unsupported | unsupported | 0.452 ± 0.003 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.160 ± 0.002 | 0.195 ± 0.001 | unsupported | unsupported | 0.187 ± 0.010 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 8 | `1024x64` | 0.131 ± 0.001 | 0.168 ± 0.001 | unsupported | unsupported | 0.249 ± 0.006 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 98.321 ± 1.365 | - | 96.612 ± 1.313 | 97.422 ± 0.772 | 114.046 ± 12.857 | 104.127 ± 1.806 | 185.321 ± 1.272 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 28.794 ± 0.259 | - | 27.976 ± 0.345 | 29.026 ± 0.200 | 61.531 ± 1.528 | 106.297 ± 0.745 | 61.586 ± 2.748 |
| cpu/elementwise_reduction | `maximum` | f64 | 8 | `33554432` | 20.025 ± 0.220 | - | 19.450 ± 0.100 | 21.524 ± 0.183 | 40.216 ± 0.263 | 114.787 ± 1.242 | 49.151 ± 3.477 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 98.135 ± 1.466 | - | 96.840 ± 1.939 | 97.620 ± 0.872 | 114.641 ± 1.292 | 105.188 ± 2.322 | 185.588 ± 0.936 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 28.881 ± 0.222 | - | 28.002 ± 0.211 | 29.234 ± 0.133 | 62.600 ± 1.358 | 108.497 ± 0.423 | 60.332 ± 3.238 |
| cpu/elementwise_reduction | `minimum` | f64 | 8 | `33554432` | 20.063 ± 0.174 | - | 19.742 ± 0.157 | 21.575 ± 0.198 | 40.238 ± 0.221 | 112.259 ± 0.997 | 50.253 ± 2.040 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 93.314 ± 1.670 | - | 91.906 ± 1.067 | 94.934 ± 0.681 | 102.667 ± 0.772 | 99.853 ± 1.236 | 173.752 ± 0.960 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 28.199 ± 0.380 | - | 27.015 ± 0.165 | 28.905 ± 0.270 | 62.178 ± 2.225 | 100.425 ± 0.572 | 60.791 ± 1.640 |
| cpu/elementwise_reduction | `mul` | f64 | 8 | `33554432` | 20.800 ± 0.181 | - | 19.493 ± 0.131 | 21.859 ± 0.161 | 38.975 ± 0.189 | 111.136 ± 0.583 | 49.832 ± 0.779 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 85.783 ± 1.769 | - | 84.147 ± 2.369 | 84.651 ± 1.033 | 101.592 ± 2.252 | 93.504 ± 2.468 | 149.140 ± 1.114 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 24.744 ± 0.361 | - | 24.293 ± 0.089 | 24.940 ± 0.203 | 55.367 ± 0.676 | 94.506 ± 3.254 | 53.855 ± 3.014 |
| cpu/elementwise_reduction | `neg` | f64 | 8 | `33554432` | 15.832 ± 0.090 | - | 15.560 ± 0.082 | 17.292 ± 0.115 | 34.897 ± 0.346 | 98.431 ± 1.422 | 44.660 ± 4.266 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 74.513 ± 0.100 | - | 74.499 ± 0.178 | 49.406 ± 0.058 | 85.122 ± 0.173 | 183.234 ± 0.206 | 129.115 ± 5.915 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 18.604 ± 0.038 | - | 18.583 ± 0.056 | 13.298 ± 0.174 | 43.972 ± 0.217 | 183.544 ± 1.100 | 33.330 ± 8.722 |
| cpu/elementwise_reduction | `pow` | f64 | 8 | `4194304` | 9.317 ± 0.074 | - | 9.331 ± 0.023 | 7.049 ± 0.339 | 29.630 ± 0.270 | 182.985 ± 1.179 | 16.777 ± 10.345 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 23.170 ± 0.060 | - | 23.050 ± 0.034 | 19.031 ± 0.239 | 100.216 ± 0.098 | 15.517 ± 0.096 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 5.812 ± 0.053 | - | 6.512 ± 0.056 | 5.370 ± 0.055 | 50.213 ± 0.107 | 16.395 ± 0.053 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 8 | `8192x4096` | 5.389 ± 0.053 | - | 5.395 ± 0.068 | 5.426 ± 0.012 | 33.980 ± 0.077 | 15.424 ± 0.135 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 2.937 ± 0.004 | - | 2.955 ± 0.009 | 45.316 ± 0.127 | 2.566 ± 0.016 | 1.857 ± 0.153 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 2.999 ± 0.021 | - | 0.756 ± 0.021 | 11.534 ± 0.058 | 1.509 ± 0.185 | 2.077 ± 0.268 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 8 | `2048x2048` | 5.997 ± 2.999 | - | 5.999 ± 0.017 | 7.613 ± 0.507 | 1.159 ± 0.240 | 1.920 ± 0.266 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.066 ± 0.036 | - | 1.072 ± 0.074 | 4.368 ± 0.014 | 6.529 ± 0.041 | 0.978 ± 0.034 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.552 ± 0.022 | - | 0.561 ± 0.015 | 1.125 ± 0.023 | 3.318 ± 0.024 | 1.008 ± 0.048 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 8 | `2048x2048` | 0.704 ± 0.010 | - | 0.718 ± 0.011 | 0.705 ± 0.024 | 2.402 ± 0.051 | 0.995 ± 0.032 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 21.084 ± 0.131 | - | 21.069 ± 0.194 | 18.996 ± 0.059 | 100.139 ± 0.053 | 16.383 ± 0.067 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 5.322 ± 0.077 | - | 6.470 ± 0.035 | 5.317 ± 0.019 | 50.232 ± 0.049 | 16.025 ± 0.069 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 8 | `8192x4096` | 5.397 ± 0.055 | - | 5.410 ± 0.038 | 5.432 ± 0.013 | 34.023 ± 0.066 | 16.523 ± 0.045 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 2.700 ± 0.010 | - | 2.711 ± 0.005 | 44.460 ± 0.153 | 2.569 ± 0.022 | 1.769 ± 0.063 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.709 ± 0.024 | - | 0.686 ± 0.006 | 11.314 ± 0.025 | 1.454 ± 0.216 | 1.761 ± 0.084 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 8 | `2048x2048` | 0.671 ± 0.060 | - | 2.047 ± 2.035 | 7.680 ± 0.547 | 1.172 ± 0.250 | 2.086 ± 0.201 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 6.809 ± 0.056 | - | 6.923 ± 0.030 | 17.357 ± 0.150 | 25.890 ± 0.044 | 5.359 ± 0.021 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.118 ± 0.021 | - | 3.143 ± 0.022 | 4.361 ± 0.040 | 13.017 ± 0.059 | 5.343 ± 0.026 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 8 | `4096x4096` | 2.749 ± 0.013 | - | 2.781 ± 1.104 | 2.717 ± 0.073 | 8.900 ± 0.023 | 5.427 ± 0.031 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 6.766 ± 0.034 | - | 6.809 ± 0.147 | 6.592 ± 0.020 | 47.767 ± 0.048 | 7.133 ± 0.032 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 5.293 ± 0.062 | - | 5.317 ± 0.031 | 5.289 ± 0.021 | 24.100 ± 0.051 | 7.112 ± 0.039 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 8 | `8192x4096` | 5.417 ± 0.010 | - | 5.417 ± 0.012 | 5.453 ± 0.026 | 16.382 ± 0.020 | 7.145 ± 0.076 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.772 ± 0.055 | - | 0.790 ± 0.150 | 3.499 ± 0.022 | 2.216 ± 0.021 | 0.824 ± 0.046 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.427 ± 0.028 | - | 0.430 ± 0.011 | 1.746 ± 0.062 | 1.287 ± 0.172 | 0.802 ± 0.069 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 8 | `2048x2048` | 0.530 ± 0.012 | - | 0.533 ± 0.108 | 1.689 ± 0.029 | 1.022 ± 0.303 | 0.898 ± 0.124 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.036 ± 0.143 | - | 0.948 ± 0.010 | 0.732 ± 0.035 | 1.715 ± 0.014 | 0.845 ± 0.074 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.526 ± 0.013 | - | 0.545 ± 0.014 | 0.429 ± 0.014 | 0.974 ± 0.283 | 0.836 ± 0.098 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 8 | `2048x2048` | 0.710 ± 0.012 | - | 0.721 ± 0.007 | 0.507 ± 0.016 | 0.901 ± 0.437 | 0.896 ± 0.087 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 6.802 ± 0.172 | - | 6.832 ± 0.214 | 6.899 ± 0.141 | 47.768 ± 0.034 | 6.910 ± 0.045 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.298 ± 0.021 | - | 5.345 ± 0.048 | 5.309 ± 0.044 | 24.145 ± 0.085 | 6.904 ± 0.025 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 8 | `8192x4096` | 5.415 ± 0.015 | - | 5.431 ± 0.026 | 5.456 ± 0.014 | 16.396 ± 0.034 | 7.256 ± 0.082 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.798 ± 0.189 | - | 0.787 ± 0.037 | 3.291 ± 0.068 | 2.217 ± 0.035 | 0.822 ± 0.108 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.406 ± 0.023 | - | 0.416 ± 0.043 | 1.642 ± 0.049 | 1.269 ± 0.150 | 0.806 ± 0.109 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 8 | `2048x2048` | 0.515 ± 0.074 | - | 0.551 ± 0.066 | 1.708 ± 0.059 | 1.013 ± 0.279 | 0.875 ± 0.107 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.057 ± 0.046 | - | 1.065 ± 0.049 | 0.742 ± 0.027 | 1.723 ± 0.030 | 0.815 ± 0.130 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.523 ± 0.014 | - | 0.528 ± 0.016 | 0.406 ± 0.011 | 0.924 ± 0.130 | 0.843 ± 0.148 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 8 | `2048x2048` | 0.715 ± 0.005 | - | 0.723 ± 0.007 | 0.509 ± 0.088 | 0.928 ± 0.403 | 0.874 ± 0.086 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.737 ± 0.011 | 0.735 ± 0.008 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.391 ± 0.005 | 0.413 ± 0.016 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 8 | `2048x2048` | 0.508 ± 0.017 | 0.519 ± 0.016 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 52.506 ± 0.283 | - | 55.440 ± 0.910 | 62.295 ± 0.167 | 73.158 ± 0.330 | 83.436 ± 6.265 | 91.895 ± 5.917 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 12.582 ± 0.261 | - | 15.565 ± 0.181 | 16.763 ± 0.315 | 37.643 ± 0.776 | 79.958 ± 7.049 | 28.596 ± 8.543 |
| cpu/elementwise_reduction | `rem` | f64 | 8 | `8388608` | 6.502 ± 2.738 | - | 6.522 ± 0.013 | 9.059 ± 0.321 | 24.877 ± 0.273 | 73.408 ± 6.596 | 20.747 ± 10.782 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 122.912 ± 1.758 | - | 123.060 ± 1.375 | 123.458 ± 0.565 | 206.423 ± 0.289 | 275.172 ± 3.313 | 275.982 ± 1.383 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 33.023 ± 0.163 | - | 32.678 ± 0.144 | 32.436 ± 0.091 | 111.452 ± 0.459 | 279.087 ± 1.659 | 84.391 ± 2.196 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 8 | `33554432` | 18.184 ± 0.103 | - | 17.854 ± 0.119 | 19.566 ± 0.211 | 71.360 ± 1.039 | 286.369 ± 6.109 | 53.861 ± 0.748 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 95.081 ± 1.180 | - | 95.433 ± 1.278 | 120.187 ± 1.657 | 116.555 ± 15.523 | 103.656 ± 1.288 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 29.691 ± 0.150 | - | 28.773 ± 0.119 | 34.758 ± 0.260 | 71.590 ± 2.881 | 104.485 ± 2.884 | - |
| cpu/elementwise_reduction | `select` | f64 | 8 | `33554432` | 21.725 ± 0.192 | - | 21.183 ± 0.103 | 23.187 ± 0.261 | 46.805 ± 0.315 | 110.934 ± 0.935 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 88.543 ± 1.913 | - | 88.887 ± 2.989 | 92.248 ± 1.487 | 106.075 ± 13.487 | 97.520 ± 3.459 | 155.533 ± 3.287 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 25.687 ± 0.202 | - | 25.375 ± 0.118 | 26.790 ± 0.208 | 56.982 ± 1.304 | 96.762 ± 0.285 | 55.605 ± 3.189 |
| cpu/elementwise_reduction | `sign` | f64 | 8 | `33554432` | 16.127 ± 0.192 | - | 15.878 ± 0.128 | 18.002 ± 0.119 | 37.046 ± 0.193 | 103.699 ± 1.415 | 46.014 ± 4.013 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 19.751 ± 0.072 | - | 20.115 ± 0.046 | 63.509 ± 0.187 | 148.392 ± 0.076 | 74.295 ± 4.430 | 76.056 ± 8.970 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 4.868 ± 0.012 | - | 4.895 ± 0.009 | 18.176 ± 0.194 | 75.706 ± 0.826 | 69.915 ± 9.422 | 21.179 ± 9.462 |
| cpu/elementwise_reduction | `sin` | f64 | 8 | `8388608` | 3.476 ± 0.052 | - | 3.571 ± 0.576 | 9.851 ± 0.459 | 51.631 ± 0.092 | 70.393 ± 11.431 | 14.800 ± 12.807 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 105.261 ± 1.591 | - | 104.291 ± 1.392 | 186.469 ± 0.905 | 111.442 ± 2.039 | 207.860 ± 4.742 | 209.293 ± 4.485 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 28.294 ± 0.209 | - | 27.789 ± 0.165 | 48.153 ± 0.156 | 66.668 ± 0.354 | 211.225 ± 1.768 | 65.929 ± 3.168 |
| cpu/elementwise_reduction | `sqrt` | f64 | 8 | `33554432` | 16.562 ± 0.200 | - | 16.272 ± 0.158 | 25.875 ± 0.139 | 40.217 ± 1.639 | 221.776 ± 1.421 | 45.032 ± 0.415 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 93.098 ± 1.130 | - | 92.520 ± 0.982 | 97.034 ± 0.420 | 102.994 ± 16.777 | 99.659 ± 0.910 | 173.634 ± 1.129 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 27.918 ± 0.188 | - | 27.044 ± 0.271 | 28.646 ± 0.186 | 63.723 ± 0.387 | 102.343 ± 0.448 | 58.737 ± 3.341 |
| cpu/elementwise_reduction | `sub` | f64 | 8 | `33554432` | 20.710 ± 0.271 | - | 19.740 ± 0.233 | 22.142 ± 0.225 | 38.778 ± 0.286 | 109.259 ± 0.506 | 49.230 ± 2.094 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 50.015 ± 0.102 | - | 49.665 ± 0.085 | 88.091 ± 0.626 | 31.554 ± 0.526 | 83.372 ± 8.692 | 92.560 ± 5.632 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 12.270 ± 0.029 | - | 12.329 ± 0.024 | 24.812 ± 0.133 | 17.504 ± 0.449 | 82.160 ± 6.067 | 24.049 ± 12.720 |
| cpu/elementwise_reduction | `tanh` | f64 | 8 | `8388608` | 6.134 ± 0.026 | - | 6.167 ± 0.024 | 13.097 ± 0.563 | 12.342 ± 0.112 | 85.701 ± 7.791 | 12.718 ± 9.539 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 94.207 ± 1.105 | 93.767 ± 1.561 | unsupported | 120.662 ± 1.066 | 142.383 ± 21.184 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 28.777 ± 0.182 | 27.991 ± 0.125 | unsupported | 34.778 ± 0.286 | 70.426 ± 0.883 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 8 | `33554432` | 21.287 ± 0.116 | 20.638 ± 0.185 | unsupported | 23.028 ± 0.239 | 46.556 ± 0.964 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.797 ± 0.018 | - | 0.780 ± 0.016 | 0.778 ± 0.021 | 0.805 ± 0.216 | 5.303 ± 2.314 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.478 ± 0.017 | - | 0.481 ± 0.010 | 0.492 ± 0.016 | 0.780 ± 0.032 | 1.207 ± 0.241 | - |
| cpu/indexing_layout | `concatenate` | f64 | 8 | `1048576+1048576` | 0.706 ± 0.127 | - | 0.690 ± 0.210 | 0.615 ± 0.009 | 0.888 ± 0.013 | 3.760 ± 5.577 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.716 ± 0.019 | - | 0.721 ± 0.020 | 0.749 ± 0.018 | 0.782 ± 0.079 | 1.083 ± 0.039 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.521 ± 0.017 | - | 0.564 ± 0.044 | 0.483 ± 0.014 | 0.642 ± 0.135 | 1.130 ± 0.067 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 8 | `4194304 -> 2097152` | 0.748 ± 0.057 | - | 0.817 ± 2.201 | 0.678 ± 0.012 | 0.912 ± 0.077 | 6.593 ± 2.821 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.156 ± 0.061 | - | unsupported | 1.145 ± 0.069 | 1.160 ± 0.045 | 4.653 ± 5.332 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.652 ± 0.028 | - | unsupported | 0.705 ± 0.044 | 1.227 ± 0.048 | 6.323 ± 3.142 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 8 | `2097152` | 0.939 ± 0.072 | - | unsupported | 0.918 ± 0.023 | 1.537 ± 0.019 | 1.721 ± 0.158 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.415 ± 0.005 | - | 0.482 ± 0.007 | 0.784 ± 0.005 | 0.400 ± 0.005 | 0.976 ± 0.302 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 2.999 ± 0.007 | - | 0.216 ± 0.011 | 0.206 ± 0.005 | 0.214 ± 0.046 | 0.989 ± 0.326 | - |
| cpu/indexing_layout | `gather` | f64 | 8 | `262144` | 0.102 ± 0.002 | - | 0.134 ± 8.059 | 0.109 ± 0.008 | 0.178 ± 0.079 | 0.983 ± 0.185 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.730 ± 0.029 | - | 0.742 ± 0.024 | 1.132 ± 0.018 | 1.124 ± 0.015 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.512 ± 0.009 | - | 0.516 ± 0.012 | 0.613 ± 0.016 | 0.687 ± 0.209 | - | - |
| cpu/indexing_layout | `pad` | f64 | 8 | `2097152` | 0.709 ± 0.040 | - | 0.730 ± 0.137 | 0.858 ± 0.104 | 0.948 ± 0.245 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.776 ± 0.118 | - | 0.787 ± 0.019 | 0.743 ± 0.007 | 0.779 ± 0.144 | 4.895 ± 5.548 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.490 ± 0.061 | - | 0.622 ± 0.654 | 0.484 ± 0.021 | 0.711 ± 0.152 | 4.803 ± 5.283 | - |
| cpu/indexing_layout | `reverse` | f64 | 8 | `2097152` | 0.657 ± 0.175 | - | 0.748 ± 0.202 | 0.600 ± 0.010 | 0.926 ± 0.306 | 6.875 ± 3.227 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 1.009 ± 0.006 | - | 1.092 ± 0.007 | 0.451 ± 0.025 | 0.400 ± 0.007 | 1.128 ± 0.629 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.988 ± 0.004 | - | 1.014 ± 0.005 | 0.372 ± 0.006 | 0.407 ± 0.087 | 1.150 ± 0.704 | - |
| cpu/indexing_layout | `scatter` | f64 | 8 | `262144` | 0.993 ± 0.007 | - | 1.028 ± 0.004 | 0.391 ± 0.007 | 0.475 ± 0.075 | 0.522 ± 0.018 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.380 ± 0.038 | - | 0.003 ± 0.000 | 1.350 ± 0.033 | 1.218 ± 0.022 | 7.333 ± 1.780 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.803 ± 0.294 | - | 0.003 ± 0.000 | 0.834 ± 0.017 | 1.166 ± 0.099 | 7.297 ± 3.645 | - |
| cpu/indexing_layout | `slice` | f64 | 8 | `4194304 -> 2096128` | 1.090 ± 0.034 | - | 0.003 ± 0.000 | 0.993 ± 0.053 | 1.271 ± 0.071 | 7.436 ± 3.664 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.136 ± 0.000 | - | 0.140 ± 0.000 | 0.094 ± 0.000 | 0.114 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.215 ± 0.004 | - | 0.223 ± 0.003 | 0.134 ± 0.000 | 0.182 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.424 ± 0.005 | - | 0.445 ± 0.005 | 0.349 ± 0.005 | 0.387 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.068 ± 0.001 | - | 0.075 ± 0.001 | 0.226 ± 0.004 | 0.147 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.554 ± 0.006 | - | 0.600 ± 0.006 | 1.808 ± 0.012 | 1.006 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.030 ± 0.001 | - | 0.033 ± 0.000 | 0.044 ± 0.000 | 0.050 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.138 ± 0.000 | - | 0.142 ± 0.000 | 0.054 ± 0.000 | 0.111 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.221 ± 0.004 | - | 0.230 ± 0.005 | 0.083 ± 0.000 | 0.189 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.413 ± 0.005 | - | 0.436 ± 0.012 | 0.265 ± 0.001 | 0.421 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.078 ± 0.001 | - | 10.494 ± 0.015 | 0.163 ± 0.003 | 0.143 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.640 ± 0.007 | - | 13.995 ± 2.696 | 1.194 ± 0.005 | 1.252 ± 0.170 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.028 ± 0.000 | - | 0.034 ± 0.000 | 0.040 ± 0.000 | 0.041 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1024x2x2` | 0.142 ± 0.000 | - | 0.147 ± 0.000 | 0.049 ± 0.000 | 0.117 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1024x4x4` | 0.225 ± 0.002 | - | 0.236 ± 0.003 | 0.075 ± 0.001 | 0.192 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1024x8x8` | 0.420 ± 0.006 | - | 0.450 ± 0.004 | 0.252 ± 0.001 | 0.435 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1x128x128` | 0.122 ± 0.003 | - | 0.141 ± 0.002 | 0.156 ± 0.004 | 0.144 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1x32x32` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `3x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `3x8x8` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `4x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `4x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `8x128x128` | 0.815 ± 0.225 | - | 18.000 ± 0.033 | 1.114 ± 0.009 | 1.479 ± 1.956 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `8x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `8x32x32` | 0.028 ± 0.000 | - | 0.034 ± 0.000 | 0.041 ± 0.000 | 0.041 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `8x4x4` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `8x8x8` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.820 ± 0.006 | - | 0.833 ± 0.005 | 0.899 ± 0.005 | 1.080 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 3.560 ± 0.014 | - | 3.588 ± 0.022 | 3.027 ± 0.006 | 3.736 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 12.234 ± 0.024 | - | 12.290 ± 0.025 | 9.189 ± 0.021 | 12.692 ± 0.212 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 11.455 ± 0.033 | - | 11.471 ± 0.017 | 4.463 ± 0.019 | 3.467 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.271 ± 0.004 | - | 0.275 ± 0.003 | 0.151 ± 0.001 | 0.178 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.015 ± 0.000 | - | 0.018 ± 0.000 | 0.017 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.023 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.039 ± 0.000 | - | 0.042 ± 0.000 | 0.035 ± 0.000 | 0.062 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.016 ± 0.000 | - | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.050 ± 0.000 | - | 0.054 ± 0.000 | 0.043 ± 0.000 | 0.074 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 91.676 ± 0.117 | - | 91.870 ± 0.119 | 35.047 ± 0.175 | 28.625 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 2.134 ± 0.010 | - | 2.137 ± 0.004 | 1.153 ± 0.003 | 1.278 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.030 ± 0.000 | - | 0.034 ± 0.000 | 0.031 ± 0.000 | 0.042 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.098 ± 0.000 | - | 0.102 ± 0.000 | 0.079 ± 0.000 | 0.123 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.857 ± 0.004 | - | 0.862 ± 0.002 | 0.859 ± 0.007 | 1.123 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 3.685 ± 0.008 | - | 3.711 ± 0.011 | 3.003 ± 0.004 | 2.387 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 12.529 ± 0.023 | - | 12.511 ± 0.025 | 9.113 ± 0.015 | 7.532 ± 0.223 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 11.323 ± 0.017 | - | 20.563 ± 6.025 | 1.908 ± 0.012 | 3.317 ± 0.126 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.276 ± 0.003 | - | 0.281 ± 0.002 | 0.153 ± 0.001 | 0.205 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.016 ± 0.000 | - | 0.019 ± 0.000 | 0.016 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.040 ± 0.000 | - | 0.044 ± 0.000 | 0.034 ± 0.000 | 0.051 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.017 ± 0.000 | - | 0.021 ± 0.000 | 0.019 ± 0.000 | 0.027 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.052 ± 0.000 | - | 0.056 ± 0.000 | 0.043 ± 0.000 | 0.064 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 90.537 ± 0.041 | - | 102.014 ± 0.209 | 14.838 ± 0.016 | 32.980 ± 7.649 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.010 ± 0.000 | - | 0.014 ± 0.001 | 0.015 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 2.162 ± 0.004 | - | 2.168 ± 0.003 | 1.184 ± 0.002 | 1.533 ± 2.520 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.032 ± 0.000 | - | 0.036 ± 0.000 | 0.031 ± 0.000 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.101 ± 0.000 | - | 0.105 ± 0.001 | 0.078 ± 0.000 | 0.114 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1024x2x2` | 0.875 ± 0.006 | - | 0.882 ± 0.004 | 0.858 ± 0.009 | 1.177 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1024x4x4` | 3.716 ± 0.021 | - | 3.721 ± 0.016 | 3.035 ± 0.007 | 2.474 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1024x8x8` | 12.578 ± 0.026 | - | 12.678 ± 0.021 | 9.230 ± 0.022 | 8.293 ± 0.110 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1x128x128` | 11.354 ± 0.015 | - | 20.607 ± 0.021 | 1.591 ± 0.015 | 3.210 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1x2x2` | 0.004 ± 0.001 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1x32x32` | 0.277 ± 0.003 | - | 0.282 ± 0.003 | 0.157 ± 0.003 | 0.183 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1x4x4` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1x8x8` | 0.015 ± 0.000 | - | 0.020 ± 0.000 | 0.017 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `3x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `3x4x4` | 0.014 ± 0.000 | - | 0.018 ± 0.000 | 0.016 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `3x8x8` | 0.040 ± 0.000 | - | 0.045 ± 0.000 | 0.035 ± 0.000 | 0.051 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `4x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `4x4x4` | 0.018 ± 0.000 | - | 0.022 ± 0.000 | 0.020 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `4x8x8` | 0.052 ± 0.000 | - | 0.057 ± 0.000 | 0.043 ± 0.000 | 0.064 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `8x128x128` | 90.787 ± 0.076 | - | 101.365 ± 0.069 | 12.292 ± 0.019 | 38.986 ± 24.801 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `8x2x2` | 0.010 ± 0.001 | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `8x32x32` | 2.177 ± 0.006 | - | 2.188 ± 0.016 | 1.211 ± 0.009 | 1.996 ± 0.583 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `8x4x4` | 0.032 ± 0.000 | - | 0.036 ± 0.000 | 0.031 ± 0.000 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `8x8x8` | 0.101 ± 0.000 | - | 0.106 ± 0.001 | 0.079 ± 0.000 | 0.133 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.492 ± 0.004 | - | 0.488 ± 0.004 | 0.501 ± 0.004 | 0.427 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 2.741 ± 0.010 | - | 2.753 ± 0.006 | 2.609 ± 0.005 | 2.546 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 9.070 ± 0.029 | - | 9.067 ± 0.016 | 7.881 ± 0.013 | 8.825 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.816 ± 0.012 | - | 1.821 ± 0.014 | 1.877 ± 0.017 | 1.939 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.121 ± 0.000 | - | 0.125 ± 0.000 | 0.105 ± 0.001 | 0.142 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.015 ± 0.001 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.029 ± 0.000 | - | 0.033 ± 0.000 | 0.030 ± 0.000 | 0.039 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.038 ± 0.000 | - | 0.042 ± 0.000 | 0.038 ± 0.000 | 0.048 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 14.422 ± 0.049 | - | 14.447 ± 0.053 | 14.762 ± 0.033 | 15.157 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.962 ± 0.004 | - | 0.963 ± 0.003 | 0.796 ± 0.003 | 0.976 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.023 ± 0.000 | - | 0.027 ± 0.000 | 0.027 ± 0.000 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.073 ± 0.000 | - | 0.077 ± 0.000 | 0.068 ± 0.000 | 0.095 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.488 ± 0.008 | - | 0.488 ± 0.006 | 0.503 ± 0.004 | 0.427 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 2.727 ± 0.010 | - | 2.736 ± 0.005 | 2.560 ± 0.007 | 1.725 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 8.987 ± 0.012 | - | 9.028 ± 0.021 | 7.827 ± 0.034 | 4.645 ± 0.108 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.798 ± 0.012 | - | 10.884 ± 0.012 | 1.087 ± 0.011 | 1.924 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.122 ± 0.001 | - | 0.126 ± 0.000 | 0.110 ± 0.000 | 0.153 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.029 ± 0.000 | - | 0.033 ± 0.000 | 0.030 ± 0.000 | 0.039 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.014 ± 0.000 | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.038 ± 0.000 | - | 0.042 ± 0.000 | 0.038 ± 0.000 | 0.048 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 14.270 ± 0.031 | - | 23.491 ± 0.353 | 8.315 ± 0.022 | 16.069 ± 10.595 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.959 ± 0.003 | - | 0.960 ± 0.002 | 0.831 ± 0.003 | 0.538 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.024 ± 0.000 | - | 0.028 ± 0.000 | 0.027 ± 0.000 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.073 ± 0.000 | - | 0.077 ± 0.000 | 0.067 ± 0.000 | 0.083 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1024x2x2` | 0.487 ± 0.006 | - | 0.495 ± 0.005 | 0.508 ± 0.007 | 0.457 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1024x4x4` | 2.776 ± 0.005 | - | 2.780 ± 0.007 | 2.589 ± 0.007 | 1.734 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1024x8x8` | 9.167 ± 0.023 | - | 9.128 ± 0.010 | 7.851 ± 0.017 | 3.695 ± 0.464 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1x128x128` | 1.808 ± 0.007 | - | 10.761 ± 0.034 | 1.044 ± 0.008 | 1.997 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1x32x32` | 0.123 ± 0.002 | - | 0.128 ± 0.001 | 0.112 ± 0.002 | 0.146 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1x8x8` | 0.012 ± 0.000 | - | 0.016 ± 0.000 | 0.015 ± 0.001 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `3x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `3x4x4` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `3x8x8` | 0.030 ± 0.000 | - | 0.034 ± 0.000 | 0.030 ± 0.000 | 0.040 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `4x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `4x4x4` | 0.014 ± 0.000 | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `4x8x8` | 0.039 ± 0.000 | - | 0.043 ± 0.001 | 0.038 ± 0.000 | 0.048 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `8x128x128` | 14.379 ± 0.026 | - | 35.277 ± 9.745 | 8.019 ± 0.024 | 27.254 ± 33.210 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `8x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `8x32x32` | 0.970 ± 0.007 | - | 0.983 ± 0.007 | 0.850 ± 0.008 | 0.388 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `8x4x4` | 0.024 ± 0.000 | - | 0.028 ± 0.000 | 0.027 ± 0.000 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `8x8x8` | 0.074 ± 0.001 | - | 0.079 ± 0.000 | 0.068 ± 0.000 | 0.100 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.269 ± 0.007 | - | 0.272 ± 0.004 | 0.265 ± 0.005 | 0.423 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.517 ± 0.005 | - | 1.526 ± 0.006 | 1.315 ± 0.004 | 2.533 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 4.940 ± 0.010 | - | 4.965 ± 0.021 | 4.155 ± 0.017 | 8.825 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.779 ± 0.006 | - | 0.790 ± 0.004 | 0.805 ± 0.005 | 1.898 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.054 ± 0.000 | - | 0.058 ± 0.000 | 0.052 ± 0.001 | 0.136 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.016 ± 0.000 | - | 0.020 ± 0.000 | 0.022 ± 0.001 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.021 ± 0.000 | - | 0.025 ± 0.000 | 0.025 ± 0.000 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 6.219 ± 0.010 | - | 6.265 ± 0.027 | 6.381 ± 0.012 | 14.904 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.410 ± 0.005 | - | 0.416 ± 0.005 | 0.338 ± 0.004 | 0.968 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.029 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.040 ± 0.000 | - | 0.044 ± 0.000 | 0.041 ± 0.000 | 0.090 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.270 ± 0.005 | - | 0.276 ± 0.004 | 0.264 ± 0.005 | 0.436 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.536 ± 0.008 | - | 1.546 ± 0.006 | 1.322 ± 0.014 | 1.683 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 4.980 ± 0.006 | - | 5.008 ± 0.019 | 4.136 ± 0.010 | 4.631 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.768 ± 0.006 | - | 0.782 ± 0.004 | 0.732 ± 0.005 | 1.889 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.054 ± 0.001 | - | 0.059 ± 0.000 | 0.051 ± 0.001 | 0.133 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.017 ± 0.000 | - | 0.021 ± 0.000 | 0.021 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.022 ± 0.000 | - | 0.026 ± 0.000 | 0.025 ± 0.000 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 6.124 ± 0.011 | - | 17.997 ± 0.050 | 5.600 ± 0.015 | 15.533 ± 3.478 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.410 ± 0.005 | - | 0.416 ± 0.005 | 0.338 ± 0.005 | 0.355 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.014 ± 0.000 | - | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.041 ± 0.000 | - | 0.045 ± 0.000 | 0.041 ± 0.000 | 0.082 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1024x2x2` | 0.269 ± 0.003 | - | 0.274 ± 0.005 | 0.284 ± 0.008 | 0.471 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1024x4x4` | 1.548 ± 0.005 | - | 1.551 ± 0.010 | 1.328 ± 0.005 | 1.709 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1024x8x8` | 5.035 ± 0.016 | - | 5.054 ± 0.009 | 4.136 ± 0.005 | 3.722 ± 0.428 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1x128x128` | 0.782 ± 0.002 | - | 0.793 ± 0.005 | 0.776 ± 0.007 | 1.999 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1x32x32` | 0.055 ± 0.000 | - | 0.060 ± 0.001 | 0.051 ± 0.000 | 0.133 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1x8x8` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `3x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `3x4x4` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `3x8x8` | 0.017 ± 0.000 | - | 0.021 ± 0.000 | 0.022 ± 0.001 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `4x4x4` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `4x8x8` | 0.022 ± 0.000 | - | 0.026 ± 0.000 | 0.025 ± 0.000 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `8x128x128` | 6.221 ± 0.010 | - | 8.997 ± 0.025 | 5.944 ± 0.020 | 25.480 ± 27.293 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `8x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `8x32x32` | 0.417 ± 0.004 | - | 0.427 ± 0.005 | 0.340 ± 0.004 | 0.376 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `8x4x4` | 0.014 ± 0.000 | - | 0.018 ± 0.000 | 0.019 ± 0.001 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `8x8x8` | 0.041 ± 0.000 | - | 0.045 ± 0.000 | 0.041 ± 0.000 | 0.082 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.189 ± 0.002 | - | 0.193 ± 0.002 | 0.210 ± 0.001 | 0.148 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.351 ± 0.003 | - | 0.360 ± 0.006 | 0.291 ± 0.007 | 0.259 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.867 ± 0.003 | - | 0.890 ± 0.010 | 0.655 ± 0.004 | 0.751 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.176 ± 0.002 | - | 0.182 ± 0.007 | 0.215 ± 0.008 | 0.257 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.012 ± 0.000 | - | 0.016 ± 0.000 | 0.023 ± 0.000 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.025 ± 0.001 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.382 ± 0.010 | - | 1.416 ± 0.012 | 1.658 ± 0.004 | 1.747 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.073 ± 0.001 | - | 0.078 ± 0.001 | 0.098 ± 0.000 | 0.113 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.019 ± 0.000 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.191 ± 0.002 | - | 0.196 ± 0.002 | 0.111 ± 0.001 | 0.172 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.353 ± 0.006 | - | 0.365 ± 0.005 | 0.166 ± 0.002 | 0.256 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.867 ± 0.008 | - | 0.879 ± 0.005 | 0.268 ± 0.006 | 0.578 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.177 ± 0.003 | - | 10.579 ± 0.015 | 0.206 ± 0.001 | 0.257 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.024 ± 0.000 | 0.025 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.014 ± 0.000 | - | 0.018 ± 0.000 | 0.026 ± 0.000 | 0.029 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.000 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 1.335 ± 0.008 | - | 11.772 ± 0.068 | 0.444 ± 0.005 | 0.994 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.073 ± 0.001 | - | 0.080 ± 0.001 | 0.050 ± 0.001 | 0.163 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.020 ± 0.000 | 0.026 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1024x2x2` | 0.193 ± 0.001 | - | 0.201 ± 0.004 | 0.097 ± 0.002 | 0.175 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1024x4x4` | 0.363 ± 0.004 | - | 0.370 ± 0.005 | 0.145 ± 0.002 | 0.282 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1024x8x8` | 0.875 ± 0.003 | - | 2.997 ± 0.014 | 0.216 ± 0.001 | 0.607 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x128x128` | 0.221 ± 0.005 | - | 10.699 ± 0.043 | 0.206 ± 0.002 | 0.250 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x32x32` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.025 ± 0.000 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x64x24` | 0.014 ± 0.000 | - | 0.018 ± 0.000 | 0.027 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x8x8` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `3x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `3x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `3x8x8` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.001 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `4x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.017 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `4x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `4x8x8` | 0.008 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.000 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `8x128x128` | 1.699 ± 0.005 | - | 15.020 ± 1.440 | 0.311 ± 0.003 | 0.721 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `8x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `8x32x32` | 0.075 ± 0.001 | - | 0.080 ± 0.001 | 0.042 ± 0.001 | 0.120 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `8x4x4` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `8x8x8` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.020 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.362 ± 0.006 | - | 0.362 ± 0.006 | 0.244 ± 0.003 | 0.335 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.984 ± 0.005 | - | 0.988 ± 0.006 | 0.665 ± 0.003 | 0.948 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 2.630 ± 0.007 | - | 2.662 ± 0.024 | 1.731 ± 0.008 | 2.689 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.714 ± 0.006 | - | 0.721 ± 0.006 | 0.822 ± 0.008 | 0.761 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.030 ± 0.000 | - | 0.033 ± 0.000 | 0.031 ± 0.000 | 0.051 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.025 ± 0.000 | - | 0.030 ± 0.001 | 0.035 ± 0.000 | 0.049 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.011 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.012 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 5.681 ± 0.012 | - | 6.186 ± 0.012 | 6.529 ± 0.041 | 5.998 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.219 ± 0.002 | - | 0.222 ± 0.005 | 0.207 ± 0.001 | 0.236 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.014 ± 0.001 | 0.010 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.023 ± 0.000 | - | 0.027 ± 0.000 | 0.019 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.353 ± 0.005 | - | 0.354 ± 0.009 | 0.208 ± 0.000 | 0.358 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.974 ± 0.004 | - | 0.983 ± 0.004 | 0.628 ± 0.005 | 0.946 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 2.624 ± 0.007 | - | 2.640 ± 0.009 | 1.635 ± 0.012 | 1.246 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.561 ± 0.009 | - | 0.856 ± 0.005 | 0.485 ± 0.006 | 0.879 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.030 ± 0.000 | - | 0.034 ± 0.000 | 0.031 ± 0.000 | 0.041 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.026 ± 0.000 | - | 0.031 ± 0.000 | 0.039 ± 0.001 | 0.040 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.011 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.013 ± 0.000 | - | 0.018 ± 0.001 | 0.013 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 4.444 ± 0.014 | - | 17.932 ± 0.025 | 3.581 ± 0.013 | 6.936 ± 0.642 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.213 ± 0.002 | - | 0.219 ± 0.005 | 0.203 ± 0.001 | 0.205 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.024 ± 0.000 | - | 0.027 ± 0.000 | 0.019 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1024x2x2` | 0.367 ± 0.003 | - | 0.369 ± 0.005 | 0.202 ± 0.001 | 0.358 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1024x4x4` | 0.991 ± 0.004 | - | 1.004 ± 0.003 | 0.621 ± 0.006 | 0.970 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1024x8x8` | 2.620 ± 0.010 | - | 2.651 ± 0.352 | 1.646 ± 0.014 | 0.930 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x128x128` | 0.547 ± 0.006 | - | 11.038 ± 0.009 | 0.432 ± 0.010 | 0.887 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x32x32` | 0.030 ± 0.000 | - | 0.034 ± 0.000 | 0.035 ± 0.001 | 0.039 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x64x24` | 0.026 ± 0.000 | - | 0.030 ± 0.000 | 0.039 ± 0.001 | 0.036 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x8x8` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `3x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `3x8x8` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.012 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `4x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `4x4x4` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `4x8x8` | 0.013 ± 0.000 | - | 0.018 ± 0.001 | 0.013 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `8x128x128` | 4.387 ± 0.014 | - | 15.922 ± 0.029 | 3.153 ± 0.021 | 7.284 ± 2.395 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `8x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `8x32x32` | 0.217 ± 0.001 | - | 0.226 ± 0.003 | 0.205 ± 0.001 | 0.197 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `8x4x4` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `8x8x8` | 0.024 ± 0.000 | - | 0.028 ± 0.001 | 0.020 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.298 ± 0.005 | - | 0.272 ± 0.004 | 0.103 ± 0.000 | 0.339 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.469 ± 0.004 | - | 0.442 ± 0.007 | 0.150 ± 0.000 | 0.486 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.967 ± 0.004 | - | 0.959 ± 0.010 | 0.541 ± 0.007 | 1.020 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.147 ± 0.002 | - | 0.158 ± 0.002 | 0.178 ± 0.003 | 0.203 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.011 ± 0.000 | - | 0.017 ± 0.000 | 0.028 ± 0.000 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.012 ± 0.000 | 0.020 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.012 ± 0.000 | 0.020 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 1.171 ± 0.012 | - | 1.237 ± 0.010 | 1.336 ± 0.004 | 1.370 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.071 ± 0.000 | - | 0.078 ± 0.001 | 0.094 ± 0.001 | 0.113 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.006 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.010 ± 0.000 | - | 0.016 ± 0.000 | 0.022 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.304 ± 0.005 | - | 0.286 ± 0.003 | 0.082 ± 0.001 | 0.372 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.476 ± 0.005 | - | 0.453 ± 0.009 | 0.115 ± 0.001 | 0.519 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.962 ± 0.002 | - | 0.965 ± 0.007 | 0.329 ± 0.004 | 0.894 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.140 ± 0.002 | - | 10.557 ± 0.015 | 0.177 ± 0.002 | 0.200 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.011 ± 0.000 | - | 0.019 ± 0.000 | 0.027 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.013 ± 0.000 | 0.020 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.005 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.014 ± 0.000 | 0.020 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 1.112 ± 0.014 | - | 14.505 ± 1.013 | 0.397 ± 0.007 | 0.828 ± 0.228 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.005 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.000 | 0.014 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.071 ± 0.001 | - | 0.079 ± 0.001 | 0.048 ± 0.001 | 0.143 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.006 ± 0.000 | - | 0.014 ± 0.000 | 0.019 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.010 ± 0.000 | - | 0.018 ± 0.000 | 0.022 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1024x2x2,rhs=1` | 0.318 ± 0.004 | - | 0.292 ± 0.005 | 0.079 ± 0.001 | 0.383 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1024x4x4,rhs=1` | 0.480 ± 0.005 | - | 0.464 ± 0.004 | 0.110 ± 0.001 | 0.527 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1024x8x8,rhs=1` | 0.976 ± 0.003 | - | 0.981 ± 0.003 | 0.298 ± 0.006 | 0.944 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1x128x128,rhs=1` | 0.189 ± 0.004 | - | 10.629 ± 0.055 | 0.179 ± 0.002 | 0.224 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1x32x32,rhs=1` | 0.012 ± 0.000 | - | 0.020 ± 0.000 | 0.028 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `3x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `3x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `3x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.014 ± 0.000 | 0.020 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `4x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `4x4x4,rhs=1` | 0.005 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `4x8x8,rhs=1` | 0.007 ± 0.000 | - | 0.015 ± 0.000 | 0.020 ± 0.001 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `8x128x128,rhs=1` | 1.500 ± 0.021 | - | 14.016 ± 0.979 | 0.279 ± 0.004 | 0.597 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `8x2x2,rhs=1` | 0.005 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.000 | 0.015 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `8x32x32,rhs=1` | 0.072 ± 0.001 | - | 0.081 ± 0.001 | 0.042 ± 0.001 | 0.122 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `8x4x4,rhs=1` | 0.006 ± 0.000 | - | 0.014 ± 0.000 | 0.019 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `8x8x8,rhs=1` | 0.010 ± 0.000 | - | 0.018 ± 0.000 | 0.022 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 2.740 ± 0.009 | - | 2.765 ± 0.010 | 2.015 ± 0.004 | 2.595 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 7.099 ± 0.029 | - | 7.139 ± 0.031 | 7.091 ± 0.014 | 6.857 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 20.626 ± 0.072 | - | 20.676 ± 0.027 | 22.119 ± 0.040 | 20.312 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 5.593 ± 0.020 | - | 5.593 ± 0.014 | 7.847 ± 0.050 | 5.883 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.248 ± 0.004 | - | 0.252 ± 0.005 | 0.290 ± 0.005 | 0.269 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.241 ± 0.004 | - | 0.246 ± 0.001 | 0.264 ± 0.004 | 0.269 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.025 ± 0.000 | - | 0.029 ± 0.000 | 0.031 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.012 ± 0.000 | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.024 ± 0.000 | - | 0.029 ± 0.000 | 0.029 ± 0.001 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.066 ± 0.000 | - | 0.070 ± 0.000 | 0.074 ± 0.000 | 0.076 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.015 ± 0.000 | - | 0.019 ± 0.000 | 0.016 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.032 ± 0.000 | - | 0.036 ± 0.000 | 0.036 ± 0.000 | 0.042 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.087 ± 0.000 | - | 0.091 ± 0.000 | 0.097 ± 0.000 | 0.097 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 44.584 ± 0.156 | - | 44.653 ± 0.107 | 62.385 ± 0.026 | 46.370 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.025 ± 0.000 | - | 0.030 ± 0.000 | 0.025 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.956 ± 0.009 | - | 1.980 ± 0.004 | 2.252 ± 0.008 | 2.001 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.059 ± 0.000 | - | 0.064 ± 0.000 | 0.064 ± 0.000 | 0.069 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.167 ± 0.001 | - | 0.172 ± 0.001 | 0.181 ± 0.001 | 0.188 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 2.765 ± 0.008 | - | 2.774 ± 0.012 | 1.924 ± 0.005 | 2.566 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 7.157 ± 0.047 | - | 7.176 ± 0.013 | 6.991 ± 0.018 | 2.725 ± 0.525 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 20.704 ± 0.031 | - | 20.794 ± 0.046 | 22.001 ± 0.061 | 10.224 ± 2.526 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.725 ± 0.034 | - | 14.894 ± 0.024 | 4.543 ± 0.018 | 5.188 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.248 ± 0.004 | - | 0.253 ± 0.004 | 0.298 ± 0.011 | 0.292 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.012 ± 0.000 | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.239 ± 0.005 | - | 0.243 ± 0.004 | 0.260 ± 0.003 | 0.256 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.025 ± 0.000 | - | 0.030 ± 0.000 | 0.031 ± 0.000 | 0.037 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.014 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.025 ± 0.000 | - | 0.029 ± 0.000 | 0.029 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.066 ± 0.000 | - | 0.070 ± 0.000 | 0.073 ± 0.000 | 0.076 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.015 ± 0.000 | - | 0.019 ± 0.000 | 0.016 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.032 ± 0.000 | - | 0.037 ± 0.000 | 0.036 ± 0.000 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.088 ± 0.000 | - | 0.092 ± 0.000 | 0.096 ± 0.000 | 0.098 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 37.522 ± 0.040 | - | 47.683 ± 0.056 | 35.780 ± 0.068 | 79.295 ± 24.311 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.026 ± 0.000 | - | 0.030 ± 0.000 | 0.024 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.947 ± 0.003 | - | 1.974 ± 0.004 | 2.305 ± 0.009 | 1.073 ± 0.121 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.060 ± 0.000 | - | 0.065 ± 0.000 | 0.063 ± 0.000 | 0.069 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.168 ± 0.000 | - | 0.173 ± 0.000 | 0.180 ± 0.001 | 0.179 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1024x2x2` | 2.796 ± 0.008 | - | 2.802 ± 0.008 | 1.938 ± 0.006 | 2.565 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1024x4x4` | 7.193 ± 0.023 | - | 7.157 ± 0.073 | 7.032 ± 0.020 | 2.268 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1024x8x8` | 20.705 ± 0.067 | - | 20.856 ± 0.071 | 22.148 ± 0.050 | 5.289 ± 2.463 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x128x128` | 4.655 ± 0.020 | - | 15.193 ± 0.031 | 4.128 ± 0.017 | 5.166 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x32x32` | 0.249 ± 0.004 | - | 0.254 ± 0.002 | 0.307 ± 0.007 | 0.267 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x4x4` | 0.012 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x64x24` | 0.241 ± 0.004 | - | 0.245 ± 0.003 | 0.260 ± 0.004 | 0.282 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x8x8` | 0.026 ± 0.000 | - | 0.030 ± 0.000 | 0.031 ± 0.001 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `3x2x2` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.015 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `3x4x4` | 0.025 ± 0.000 | - | 0.030 ± 0.000 | 0.029 ± 0.001 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `3x8x8` | 0.066 ± 0.000 | - | 0.071 ± 0.000 | 0.074 ± 0.001 | 0.076 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `4x2x2` | 0.016 ± 0.000 | - | 0.020 ± 0.000 | 0.016 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `4x4x4` | 0.033 ± 0.000 | - | 0.037 ± 0.001 | 0.036 ± 0.000 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `4x8x8` | 0.088 ± 0.000 | - | 0.093 ± 0.001 | 0.096 ± 0.001 | 0.097 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `8x128x128` | 37.117 ± 0.032 | - | 48.876 ± 0.106 | 32.604 ± 0.107 | 160.034 ± 71.401 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `8x2x2` | 0.027 ± 0.000 | - | 0.030 ± 0.000 | 0.025 ± 0.001 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `8x32x32` | 1.954 ± 0.009 | - | 1.973 ± 0.009 | 2.357 ± 0.010 | 0.384 ± 0.183 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `8x4x4` | 0.061 ± 0.000 | - | 0.066 ± 0.002 | 0.063 ± 0.000 | 0.069 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `8x8x8` | 0.169 ± 0.001 | - | 0.174 ± 0.001 | 0.181 ± 0.001 | 0.186 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.725 ± 0.005 | - | 1.727 ± 0.013 | 1.433 ± 0.006 | 1.555 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 5.347 ± 0.016 | - | 5.361 ± 0.013 | 5.929 ± 0.011 | 5.237 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 16.504 ± 0.031 | - | 16.565 ± 0.024 | 19.204 ± 0.033 | 16.563 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.993 ± 0.009 | - | 2.997 ± 0.038 | 3.389 ± 0.015 | 3.113 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.190 ± 0.002 | - | 0.195 ± 0.001 | 0.198 ± 0.004 | 0.214 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.166 ± 0.001 | - | 0.171 ± 0.002 | 0.159 ± 0.003 | 0.187 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.020 ± 0.000 | - | 0.024 ± 0.000 | 0.027 ± 0.000 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.019 ± 0.000 | - | 0.024 ± 0.000 | 0.027 ± 0.000 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.049 ± 0.000 | - | 0.054 ± 0.000 | 0.060 ± 0.000 | 0.062 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.025 ± 0.000 | - | 0.029 ± 0.000 | 0.032 ± 0.000 | 0.037 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.065 ± 0.000 | - | 0.070 ± 0.000 | 0.078 ± 0.000 | 0.078 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 23.762 ± 0.062 | - | 23.668 ± 0.347 | 26.680 ± 0.075 | 24.515 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.017 ± 0.000 | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.491 ± 0.002 | - | 1.501 ± 0.007 | 1.533 ± 0.005 | 1.515 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.046 ± 0.000 | - | 0.050 ± 0.000 | 0.055 ± 0.000 | 0.057 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.129 ± 0.000 | - | 0.133 ± 0.001 | 0.151 ± 0.001 | 0.153 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.720 ± 0.012 | - | 1.724 ± 0.007 | 1.358 ± 0.005 | 1.588 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 5.349 ± 0.005 | - | 5.371 ± 0.008 | 5.876 ± 0.006 | 1.759 ± 0.407 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 16.577 ± 0.012 | - | 16.609 ± 0.031 | 19.202 ± 0.034 | 8.321 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.964 ± 0.015 | - | 13.305 ± 0.026 | 2.720 ± 0.018 | 3.110 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.191 ± 0.002 | - | 0.193 ± 0.001 | 0.206 ± 0.004 | 0.203 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.014 ± 0.001 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.168 ± 0.001 | - | 0.171 ± 0.001 | 0.160 ± 0.001 | 0.211 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.020 ± 0.000 | - | 0.025 ± 0.000 | 0.027 ± 0.000 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.013 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.020 ± 0.000 | - | 0.024 ± 0.000 | 0.026 ± 0.001 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.050 ± 0.000 | - | 0.054 ± 0.000 | 0.060 ± 0.000 | 0.062 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.011 ± 0.000 | - | 0.015 ± 0.001 | 0.014 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.025 ± 0.000 | - | 0.030 ± 0.000 | 0.032 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.066 ± 0.000 | - | 0.071 ± 0.000 | 0.078 ± 0.000 | 0.079 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 23.485 ± 0.053 | - | 33.832 ± 0.175 | 21.261 ± 0.058 | 31.513 ± 9.458 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.018 ± 0.000 | - | 0.022 ± 0.001 | 0.019 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.495 ± 0.005 | - | 1.516 ± 0.009 | 1.598 ± 0.007 | 0.778 ± 0.330 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.046 ± 0.000 | - | 0.051 ± 0.000 | 0.054 ± 0.000 | 0.058 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.129 ± 0.000 | - | 0.134 ± 0.001 | 0.151 ± 0.001 | 0.156 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1024x2x2` | 1.728 ± 0.015 | - | 1.740 ± 0.009 | 1.350 ± 0.012 | 1.585 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1024x4x4` | 5.363 ± 0.018 | - | 5.364 ± 0.016 | 5.911 ± 0.022 | 1.739 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1024x8x8` | 16.617 ± 0.023 | - | 16.709 ± 0.038 | 19.315 ± 0.043 | 4.263 ± 2.050 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x128x128` | 2.968 ± 0.014 | - | 13.391 ± 0.015 | 2.644 ± 0.019 | 3.159 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x32x32` | 0.195 ± 0.003 | - | 0.197 ± 0.001 | 0.212 ± 0.005 | 0.221 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x4x4` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x64x24` | 0.169 ± 0.001 | - | 0.171 ± 0.001 | 0.160 ± 0.002 | 0.216 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x8x8` | 0.020 ± 0.000 | - | 0.025 ± 0.000 | 0.027 ± 0.000 | 0.032 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `3x2x2` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.013 ± 0.001 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `3x4x4` | 0.020 ± 0.000 | - | 0.024 ± 0.000 | 0.026 ± 0.000 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `3x8x8` | 0.050 ± 0.000 | - | 0.054 ± 0.000 | 0.060 ± 0.000 | 0.062 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `4x2x2` | 0.011 ± 0.000 | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `4x4x4` | 0.025 ± 0.000 | - | 0.030 ± 0.000 | 0.032 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `4x8x8` | 0.066 ± 0.000 | - | 0.072 ± 0.000 | 0.079 ± 0.000 | 0.078 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `8x128x128` | 23.655 ± 0.208 | - | 37.935 ± 0.979 | 20.539 ± 0.102 | 58.328 ± 36.640 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `8x2x2` | 0.018 ± 0.000 | - | 0.022 ± 0.000 | 0.020 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `8x32x32` | 1.502 ± 0.010 | - | 1.505 ± 0.018 | 1.637 ± 0.013 | 0.321 ± 0.168 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `8x4x4` | 0.046 ± 0.000 | - | 0.051 ± 0.001 | 0.055 ± 0.000 | 0.057 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `8x8x8` | 0.130 ± 0.000 | - | 0.136 ± 0.001 | 0.152 ± 0.001 | 0.152 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.944 ± 0.005 | - | unsupported | 0.392 ± 0.004 | 0.822 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 3.397 ± 0.013 | - | unsupported | 2.622 ± 0.008 | 3.284 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 8.593 ± 0.020 | - | unsupported | 7.006 ± 0.012 | 8.562 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.458 ± 0.002 | - | unsupported | 1.524 ± 0.017 | 1.532 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.079 ± 0.001 | - | unsupported | 0.076 ± 0.000 | 0.089 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.064 ± 0.000 | - | unsupported | 0.063 ± 0.001 | 0.075 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.011 ± 0.000 | - | unsupported | 0.014 ± 0.001 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.012 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.019 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.027 ± 0.000 | - | unsupported | 0.028 ± 0.001 | 0.035 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.007 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.015 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.036 ± 0.000 | - | unsupported | 0.035 ± 0.000 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 11.647 ± 0.025 | - | unsupported | 12.034 ± 0.036 | 11.901 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.010 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.626 ± 0.003 | - | unsupported | 0.558 ± 0.003 | 0.651 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.028 ± 0.000 | - | unsupported | 0.028 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.069 ± 0.000 | - | unsupported | 0.062 ± 0.000 | 0.076 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.936 ± 0.008 | - | unsupported | 0.394 ± 0.006 | 0.845 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 3.394 ± 0.004 | - | unsupported | 2.597 ± 0.004 | 1.358 ± 0.092 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 8.589 ± 0.015 | - | unsupported | 6.966 ± 0.012 | 4.350 ± 0.104 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.449 ± 0.015 | - | unsupported | 1.500 ± 0.024 | 1.559 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.080 ± 0.001 | - | unsupported | 0.075 ± 0.001 | 0.088 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.065 ± 0.000 | - | unsupported | 0.063 ± 0.001 | 0.074 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.011 ± 0.000 | - | unsupported | 0.014 ± 0.001 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.013 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.028 ± 0.000 | - | unsupported | 0.028 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.016 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.023 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.036 ± 0.000 | - | unsupported | 0.035 ± 0.000 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 11.595 ± 0.090 | - | unsupported | 11.657 ± 0.015 | 13.974 ± 7.194 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.011 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.016 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.625 ± 0.005 | - | unsupported | 0.558 ± 0.006 | 0.354 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.029 ± 0.000 | - | unsupported | 0.028 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.069 ± 0.000 | - | unsupported | 0.062 ± 0.000 | 0.077 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1024x2x2` | 0.944 ± 0.004 | - | unsupported | 0.394 ± 0.004 | 0.843 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1024x4x4` | 3.407 ± 0.021 | - | unsupported | 2.610 ± 0.007 | 1.143 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1024x8x8` | 8.649 ± 0.028 | - | unsupported | 7.008 ± 0.023 | 2.263 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x128x128` | 1.494 ± 0.011 | - | unsupported | 1.506 ± 0.018 | 1.605 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x2x2` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x32x32` | 0.080 ± 0.001 | - | unsupported | 0.076 ± 0.001 | 0.088 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x64x24` | 0.066 ± 0.000 | - | unsupported | 0.063 ± 0.001 | 0.075 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x8x8` | 0.011 ± 0.000 | - | unsupported | 0.014 ± 0.001 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `3x2x2` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.012 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `3x4x4` | 0.013 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `3x8x8` | 0.028 ± 0.000 | - | unsupported | 0.028 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `4x2x2` | 0.007 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `4x4x4` | 0.016 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `4x8x8` | 0.037 ± 0.000 | - | unsupported | 0.035 ± 0.000 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `8x128x128` | 11.942 ± 0.064 | - | unsupported | 11.762 ± 0.029 | 16.811 ± 26.315 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `8x2x2` | 0.011 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `8x32x32` | 0.631 ± 0.005 | - | unsupported | 0.560 ± 0.005 | 0.169 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `8x4x4` | 0.029 ± 0.000 | - | unsupported | 0.028 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `8x8x8` | 0.070 ± 0.000 | - | unsupported | 0.062 ± 0.001 | 0.076 ± 0.001 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 2.066 ± 0.007 | - | unsupported | 1.442 ± 0.003 | 3.146 ± 0.010 | 2.727 ± 0.031 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.141 ± 0.001 | - | unsupported | 0.064 ± 0.002 | 0.130 ± 0.004 | 0.168 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.248 ± 0.008 | - | unsupported | 0.093 ± 0.001 | 0.225 ± 0.005 | 0.269 ± 0.017 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.645 ± 0.008 | - | unsupported | 0.316 ± 0.005 | 0.607 ± 0.006 | 0.844 ± 0.015 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 2.108 ± 0.009 | - | unsupported | 0.380 ± 0.013 | 1.151 ± 0.363 | 2.125 ± 0.703 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.146 ± 0.001 | - | unsupported | 0.040 ± 0.000 | 0.140 ± 0.035 | 0.156 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.254 ± 0.004 | - | unsupported | 0.058 ± 0.001 | 0.232 ± 0.027 | 0.250 ± 0.018 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.650 ± 0.005 | - | unsupported | 0.107 ± 0.001 | 0.466 ± 0.024 | 0.798 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 8 | `1024x16x16,rhs=1` | 2.097 ± 0.007 | - | unsupported | 0.192 ± 0.002 | 0.853 ± 0.051 | 2.760 ± 0.036 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 8 | `1024x2x2,rhs=1` | 0.149 ± 0.000 | - | unsupported | 0.037 ± 0.001 | 0.129 ± 0.032 | 0.177 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 8 | `1024x4x4,rhs=1` | 0.257 ± 0.004 | - | unsupported | 0.052 ± 0.001 | 0.248 ± 0.029 | 0.277 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 8 | `1024x8x8,rhs=1` | 0.659 ± 0.008 | - | unsupported | 0.078 ± 0.001 | 0.483 ± 0.039 | 0.809 ± 0.184 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.588 ± 0.005 | - | unsupported | 0.500 ± 0.004 | 1.004 ± 0.006 | 0.551 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.132 ± 0.000 | - | unsupported | 0.047 ± 0.000 | 0.233 ± 0.003 | 0.111 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.195 ± 0.002 | - | unsupported | 0.062 ± 0.000 | 0.288 ± 0.006 | 0.165 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.300 ± 0.005 | - | unsupported | 0.225 ± 0.001 | 0.464 ± 0.006 | 0.278 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.595 ± 0.004 | - | unsupported | 0.508 ± 0.004 | 0.911 ± 0.037 | 0.562 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.138 ± 0.001 | - | unsupported | 0.046 ± 0.000 | 0.236 ± 0.032 | 0.109 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.196 ± 0.002 | - | unsupported | 0.061 ± 0.000 | 0.321 ± 0.031 | 0.157 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.305 ± 0.005 | - | unsupported | 0.227 ± 0.002 | 0.502 ± 0.009 | 0.259 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 8 | `1024x16x16,rhs=1` | 0.603 ± 0.004 | - | unsupported | 0.507 ± 0.004 | 0.927 ± 0.012 | 0.561 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 8 | `1024x2x2,rhs=1` | 0.145 ± 0.002 | - | unsupported | 0.047 ± 0.000 | 0.269 ± 0.031 | 0.119 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 8 | `1024x4x4,rhs=1` | 0.199 ± 0.002 | - | unsupported | 0.061 ± 0.000 | 0.327 ± 0.030 | 0.171 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 8 | `1024x8x8,rhs=1` | 0.311 ± 0.002 | - | unsupported | 0.228 ± 0.001 | 0.497 ± 0.006 | 0.289 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.339 ± 0.008 | - | 0.443 ± 0.005 | 0.265 ± 0.004 | 0.475 ± 0.008 | 0.305 ± 0.009 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.129 ± 0.000 | - | 0.137 ± 0.000 | 0.083 ± 0.001 | 0.107 ± 0.001 | 0.096 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.141 ± 0.001 | - | 0.151 ± 0.001 | 0.104 ± 0.001 | 0.139 ± 0.001 | 0.116 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.190 ± 0.002 | - | 0.218 ± 0.004 | 0.154 ± 0.001 | 0.219 ± 0.004 | 0.165 ± 0.023 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.345 ± 0.006 | - | 0.389 ± 0.011 | 0.267 ± 0.004 | 0.488 ± 0.020 | 0.364 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.133 ± 0.000 | - | 0.140 ± 0.001 | 0.082 ± 0.001 | 0.113 ± 0.001 | 0.098 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.146 ± 0.002 | - | 0.158 ± 0.001 | 0.102 ± 0.001 | 0.141 ± 0.002 | 0.120 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.195 ± 0.002 | - | 0.224 ± 0.004 | 0.153 ± 0.001 | 0.232 ± 0.027 | 0.191 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 8 | `1024x16x16,rhs=1` | 0.344 ± 0.007 | - | 0.389 ± 0.007 | 0.262 ± 0.004 | 0.458 ± 0.032 | 0.361 ± 0.026 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 8 | `1024x2x2,rhs=1` | 0.138 ± 0.000 | - | 0.146 ± 0.003 | 0.082 ± 0.002 | 0.125 ± 0.002 | 0.103 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 8 | `1024x4x4,rhs=1` | 0.149 ± 0.001 | - | 0.162 ± 0.004 | 0.102 ± 0.001 | 0.154 ± 0.002 | 0.125 ± 0.006 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 8 | `1024x8x8,rhs=1` | 0.202 ± 0.001 | - | 0.233 ± 0.004 | 0.154 ± 0.003 | 0.261 ± 0.016 | 0.200 ± 0.007 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 56.851 ± 0.051 | - | 58.469 ± 0.086 | 101.749 ± 0.224 | 114.117 ± 0.254 | 74.438 ± 7.813 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 20.612 ± 0.026 | - | 32.317 ± 0.147 | 25.993 ± 0.047 | 74.983 ± 7.691 | 39.182 ± 9.257 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 8 | `1536x1536` | 13.226 ± 0.081 | - | 23.452 ± 1.708 | 13.671 ± 0.189 | 51.975 ± 7.047 | 31.881 ± 10.743 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 35.784 ± 0.065 | - | 35.163 ± 0.073 | 40.562 ± 0.149 | 38.323 ± 0.066 | 33.635 ± 0.440 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 14.679 ± 0.031 | - | 23.573 ± 0.130 | 9.004 ± 0.104 | 17.663 ± 0.254 | 12.654 ± 0.540 | - |
| cpu/linalg_uncovered | `det` | f64 | 8 | `1024x1024` | 9.092 ± 0.050 | - | 18.421 ± 15.795 | 5.509 ± 0.028 | 12.119 ± 1.856 | 7.016 ± 0.215 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 10.369 ± 0.022 | - | 10.401 ± 0.150 | 10.615 ± 0.015 | 10.061 ± 0.025 | 10.050 ± 0.109 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 10.224 ± 0.090 | - | 20.696 ± 0.093 | 9.577 ± 0.022 | 10.411 ± 0.080 | 10.181 ± 0.028 | - |
| cpu/linalg_uncovered | `eig` | f64 | 8 | `160x160` | 10.129 ± 0.037 | - | 20.259 ± 0.051 | 9.629 ± 0.034 | 10.641 ± 0.039 | 10.224 ± 0.028 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 12.433 ± 0.034 | - | 12.508 ± 0.025 | 11.773 ± 0.031 | 12.387 ± 0.032 | 12.064 ± 0.030 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 12.508 ± 0.099 | - | 23.105 ± 0.020 | 10.807 ± 0.024 | 12.686 ± 0.029 | 12.494 ± 0.019 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 8 | `192x192` | 12.413 ± 0.026 | - | 22.501 ± 0.475 | 10.748 ± 0.031 | 12.974 ± 0.036 | 12.668 ± 0.037 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 17.709 ± 0.056 | - | 17.681 ± 0.031 | 24.403 ± 0.046 | 52.512 ± 0.063 | 17.285 ± 0.064 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 12.285 ± 0.019 | - | 24.996 ± 0.073 | 13.264 ± 0.018 | 33.344 ± 0.300 | 12.297 ± 0.033 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 8 | `512x512` | 11.662 ± 0.028 | - | 22.998 ± 0.046 | 11.321 ± 0.031 | 30.671 ± 0.412 | 12.779 ± 0.649 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.780 ± 0.001 | 0.801 ± 0.001 | unsupported | 0.764 ± 0.004 | 0.905 ± 0.012 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 12.209 ± 0.010 | 12.346 ± 0.024 | unsupported | 11.366 ± 0.007 | 12.484 ± 0.070 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.455 ± 0.002 | 11.297 ± 5.663 | unsupported | 0.397 ± 0.001 | 1.118 ± 0.010 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 5.915 ± 0.024 | 19.020 ± 0.614 | unsupported | 4.691 ± 0.028 | 8.122 ± 0.106 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 8 | `256x128` | 0.395 ± 0.001 | 3.999 ± 3.000 | unsupported | 0.345 ± 0.004 | 1.131 ± 0.019 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 8 | `768x384` | 4.876 ± 0.032 | 19.180 ± 1.843 | unsupported | 2.791 ± 0.006 | 7.773 ± 0.098 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 62.662 ± 0.157 | - | 62.868 ± 0.131 | 61.017 ± 0.141 | 62.639 ± 0.100 | 44.453 ± 0.432 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 32.984 ± 0.078 | - | 36.498 ± 6.039 | 16.005 ± 0.030 | 20.824 ± 0.061 | 15.477 ± 0.067 | - |
| cpu/linalg_uncovered | `inv` | f64 | 8 | `768x768` | 26.278 ± 7.142 | - | 22.761 ± 5.380 | 9.666 ± 0.055 | 12.429 ± 0.151 | 9.749 ± 0.064 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | failed | failed | 12.551 ± 0.026 | 63.845 ± 0.079 | 10.980 ± 0.026 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 35.007 ± 5.515 | 20.884 ± 0.028 | 5.176 ± 0.009 | 40.974 ± 0.785 | 6.200 ± 0.033 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 8 | `768x384,rhs=16` | - | 33.999 ± 0.004 | 29.820 ± 9.086 | failed | 37.815 ± 0.382 | 5.514 ± 0.408 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 35.834 ± 0.104 | - | 36.351 ± 0.097 | 47.608 ± 0.100 | 54.712 ± 0.119 | 42.869 ± 16.362 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 14.630 ± 0.054 | - | 25.044 ± 0.053 | 13.443 ± 0.050 | 33.299 ± 3.118 | 29.277 ± 13.546 | - |
| cpu/linalg_uncovered | `lu` | f64 | 8 | `1024x1024` | 9.075 ± 0.131 | - | 23.007 ± 0.081 | 9.216 ± 0.049 | 25.459 ± 13.512 | 16.806 ± 19.125 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.092 ± 0.032 | - | 0.737 ± 0.127 | 1.644 ± 0.023 | 18.904 ± 0.130 | 4.238 ± 0.027 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 5.999 ± 0.004 | - | 0.444 ± 0.247 | 1.645 ± 0.014 | 11.117 ± 0.209 | 4.244 ± 0.032 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 8 | `2048x2048` | 0.909 ± 5.117 | - | 0.884 ± 1.283 | 1.668 ± 0.035 | 8.923 ± 0.297 | 4.258 ± 0.007 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 26.637 ± 0.079 | - | 26.440 ± 0.061 | 27.804 ± 0.047 | 27.153 ± 0.065 | 26.754 ± 0.486 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 57.000 ± 0.013 | - | 45.999 ± 13.003 | 17.096 ± 0.087 | 21.536 ± 0.162 | 19.260 ± 1.794 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 8 | `512x256` | 49.000 ± 10.084 | - | 47.998 ± 17.057 | 14.843 ± 0.024 | 21.853 ± 1.506 | 18.624 ± 1.133 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 26.622 ± 0.187 | - | 26.463 ± 0.073 | 27.849 ± 0.053 | 27.134 ± 0.038 | 26.702 ± 0.392 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 57.001 ± 0.997 | - | 32.997 ± 1.004 | 17.108 ± 0.062 | 21.369 ± 0.607 | 19.266 ± 1.041 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 8 | `512x256` | 51.983 ± 25.802 | - | 43.999 ± 2.992 | 14.847 ± 0.071 | 21.211 ± 1.187 | 18.587 ± 0.481 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.696 ± 0.003 | 1.706 ± 0.003 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 28.328 ± 0.054 | 28.555 ± 0.080 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.048 ± 0.002 | 12.498 ± 1.612 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 12.793 ± 0.019 | 23.253 ± 0.699 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 8 | `256x128` | 0.933 ± 0.003 | 12.315 ± 0.208 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 8 | `768x384` | 10.499 ± 0.031 | 31.927 ± 5.604 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.822 ± 0.003 | 1.836 ± 0.003 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 26.837 ± 0.065 | 26.969 ± 0.048 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.339 ± 0.004 | 6.812 ± 9.061 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 12.560 ± 0.014 | 22.813 ± 2.911 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 8 | `256x128` | 1.266 ± 0.012 | 2.999 ± 8.142 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 8 | `768x384` | 10.495 ± 0.014 | 30.782 ± 9.567 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 35.803 ± 0.135 | - | 35.147 ± 0.306 | 40.592 ± 0.065 | 38.360 ± 0.050 | 33.643 ± 0.476 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 14.647 ± 0.065 | - | 23.584 ± 11.460 | 8.990 ± 0.023 | 17.589 ± 0.222 | 12.627 ± 2.314 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 8 | `1024x1024` | 9.123 ± 0.076 | - | 30.994 ± 0.992 | 5.511 ± 0.009 | 12.181 ± 1.280 | 7.046 ± 0.077 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 87.040 ± 0.261 | 86.939 ± 0.331 | 82.807 ± 0.116 | 87.632 ± 0.125 | 86.111 ± 0.770 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 52.812 ± 0.551 | 49.199 ± 0.097 | 33.491 ± 0.038 | 52.606 ± 0.429 | 50.721 ± 1.463 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 8 | `768x384` | - | 44.173 ± 0.556 | 45.047 ± 0.222 | 29.591 ± 0.071 | 47.833 ± 0.238 | 47.814 ± 1.637 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 47.265 ± 0.130 | - | 99.690 ± 0.203 | 43.305 ± 0.173 | 180.005 ± 1.030 | 46.689 ± 0.229 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 16.711 ± 0.030 | - | 63.707 ± 13.106 | 15.282 ± 0.031 | 147.132 ± 27.072 | 16.412 ± 0.017 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 8 | `4096x4096,rhs=64` | 12.602 ± 0.067 | - | 56.091 ± 4.042 | 12.387 ± 0.554 | 113.544 ± 10.458 | 12.369 ± 0.708 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.131 ± 0.000 | 0.132 ± 0.000 | unsupported | 0.173 ± 0.001 | 0.122 ± 0.006 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.096 ± 0.003 | 2.099 ± 0.004 | unsupported | 2.171 ± 0.006 | 1.334 ± 0.189 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.037 ± 0.000 | 0.038 ± 0.000 | unsupported | 0.056 ± 0.033 | 0.037 ± 0.002 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.528 ± 0.004 | 0.646 ± 0.119 | unsupported | 0.750 ± 0.024 | 0.908 ± 0.245 | - | - |
| cpu/neural_public_api | `erf` | f32 | 8 | `1024x64` | 0.022 ± 0.000 | 0.023 ± 0.000 | unsupported | 0.035 ± 0.002 | 0.037 ± 0.001 | - | - |
| cpu/neural_public_api | `erf` | f32 | 8 | `4096x256` | 0.268 ± 0.004 | 0.271 ± 0.004 | unsupported | 0.408 ± 0.026 | 0.355 ± 0.179 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.065 ± 0.000 | 0.065 ± 0.000 | unsupported | 0.055 ± 0.000 | 0.211 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.026 ± 0.002 | 1.045 ± 0.003 | unsupported | 1.377 ± 0.044 | 3.320 ± 0.034 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.018 ± 0.000 | 0.019 ± 0.000 | unsupported | 0.051 ± 0.007 | 0.092 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.260 ± 0.002 | 0.264 ± 0.003 | unsupported | 0.421 ± 0.019 | 0.831 ± 0.147 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 8 | `1024x64` | 0.011 ± 0.000 | 0.011 ± 0.000 | unsupported | 0.044 ± 0.002 | 0.064 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 8 | `4096x256` | 0.143 ± 0.002 | 0.144 ± 0.003 | unsupported | 0.272 ± 0.015 | 0.436 ± 0.028 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.191 ± 0.001 | 0.191 ± 0.000 | unsupported | 0.250 ± 0.001 | 0.151 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.047 ± 0.007 | 3.061 ± 0.004 | unsupported | 3.985 ± 0.016 | 2.370 ± 0.011 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.050 ± 0.000 | 0.050 ± 0.000 | unsupported | 0.072 ± 0.002 | 0.087 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 0.765 ± 0.002 | 0.766 ± 0.004 | unsupported | 0.857 ± 0.005 | 0.917 ± 0.034 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 8 | `1024x64` | 0.026 ± 0.000 | 0.027 ± 0.000 | unsupported | 0.043 ± 0.001 | 0.090 ± 0.007 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 8 | `4096x256` | 0.383 ± 0.002 | 0.387 ± 0.002 | unsupported | 0.547 ± 0.024 | 0.664 ± 0.034 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.064 ± 0.002 | 0.101 ± 0.002 | unsupported | 0.176 ± 0.000 | 0.159 ± 0.001 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.683 ± 0.007 | 0.744 ± 0.006 | unsupported | 1.306 ± 0.005 | 2.587 ± 0.009 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.060 ± 0.000 | 0.093 ± 0.001 | unsupported | 0.057 ± 0.002 | 0.135 ± 0.007 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.317 ± 0.004 | 0.368 ± 0.006 | unsupported | 0.339 ± 0.023 | 1.060 ± 0.018 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 8 | `1024x64` | 0.063 ± 0.000 | 0.094 ± 0.000 | unsupported | 0.037 ± 0.002 | 0.139 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 8 | `4096x256` | 0.360 ± 0.011 | 0.431 ± 0.008 | unsupported | 0.175 ± 0.015 | 0.813 ± 0.036 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.099 ± 0.001 | 0.124 ± 0.002 | unsupported | 0.110 ± 0.001 | 0.263 ± 0.002 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.288 ± 0.008 | 1.365 ± 0.016 | unsupported | 1.042 ± 0.004 | 4.193 ± 0.017 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.070 ± 0.001 | 0.094 ± 0.001 | unsupported | 0.038 ± 0.002 | 0.228 ± 0.007 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.446 ± 0.006 | 0.501 ± 0.007 | unsupported | 0.267 ± 0.003 | 1.872 ± 0.025 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 8 | `1024x64` | 0.073 ± 0.001 | 0.095 ± 0.001 | unsupported | 0.031 ± 0.001 | 0.224 ± 0.003 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 8 | `4096x256` | 0.487 ± 0.007 | 0.590 ± 0.009 | unsupported | 0.174 ± 0.038 | 1.418 ± 0.045 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.247 ± 0.002 | 0.283 ± 0.001 | unsupported | unsupported | 0.285 ± 0.001 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 3.554 ± 0.011 | 3.616 ± 0.020 | unsupported | unsupported | 4.834 ± 0.013 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.125 ± 0.001 | 0.159 ± 0.001 | unsupported | unsupported | 0.248 ± 0.006 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.087 ± 0.008 | 1.155 ± 0.018 | unsupported | unsupported | 2.372 ± 0.037 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 8 | `1024x64` | 0.108 ± 0.000 | 0.140 ± 0.001 | unsupported | unsupported | 0.253 ± 0.001 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 8 | `4096x256` | 0.850 ± 0.017 | 0.894 ± 0.015 | unsupported | unsupported | 1.967 ± 0.064 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.013 ± 0.000 | 0.022 ± 0.000 | unsupported | 0.015 ± 0.000 | 0.041 ± 0.000 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.127 ± 0.001 | 0.138 ± 0.001 | unsupported | 0.079 ± 0.002 | 0.422 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.010 ± 0.000 | 0.019 ± 0.000 | unsupported | 0.010 ± 0.000 | 0.037 ± 0.000 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.046 ± 0.001 | 0.051 ± 0.001 | unsupported | 0.028 ± 0.001 | 0.225 ± 0.008 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 8 | `1024x64` | 0.011 ± 0.000 | 0.020 ± 0.000 | unsupported | 0.009 ± 0.000 | 0.037 ± 0.000 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 8 | `4096x256` | 0.026 ± 0.001 | 0.037 ± 0.001 | unsupported | 0.017 ± 0.001 | 0.164 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.035 ± 0.000 | 0.057 ± 0.001 | unsupported | 0.079 ± 0.002 | 0.163 ± 0.001 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.349 ± 0.004 | 0.386 ± 0.006 | unsupported | 0.510 ± 0.009 | 2.609 ± 0.015 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.032 ± 0.000 | 0.053 ± 0.000 | unsupported | 0.063 ± 0.003 | 0.116 ± 0.004 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.177 ± 0.003 | 0.207 ± 0.001 | unsupported | 0.221 ± 0.003 | 1.082 ± 0.051 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 8 | `1024x64` | 0.036 ± 0.000 | 0.057 ± 0.000 | unsupported | 0.067 ± 0.004 | 0.117 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 8 | `4096x256` | 0.180 ± 0.005 | 0.210 ± 0.002 | unsupported | 0.195 ± 0.003 | 0.789 ± 0.028 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.044 ± 0.000 | 0.044 ± 0.000 | unsupported | 0.083 ± 0.000 | 0.133 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.699 ± 0.001 | 0.719 ± 0.002 | unsupported | 1.311 ± 0.026 | 2.086 ± 0.012 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.013 ± 0.000 | 0.014 ± 0.000 | unsupported | 0.053 ± 0.003 | 0.078 ± 0.006 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.178 ± 0.002 | 0.182 ± 0.003 | unsupported | 0.384 ± 0.025 | 0.765 ± 0.063 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 8 | `1024x64` | 0.008 ± 0.001 | 0.009 ± 0.000 | unsupported | 0.053 ± 0.003 | 0.087 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 8 | `4096x256` | 0.130 ± 0.005 | 0.129 ± 0.004 | unsupported | 0.260 ± 0.025 | 0.584 ± 0.016 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.047 ± 0.000 | 0.047 ± 0.000 | unsupported | 0.085 ± 0.001 | 0.137 ± 0.002 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.734 ± 0.002 | 0.756 ± 0.002 | unsupported | 1.340 ± 0.040 | 2.146 ± 0.014 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.014 ± 0.000 | 0.014 ± 0.000 | unsupported | 0.053 ± 0.003 | 0.080 ± 0.005 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.185 ± 0.002 | 0.191 ± 0.002 | unsupported | 0.416 ± 0.014 | 0.806 ± 0.046 | - | - |
| cpu/neural_public_api | `silu` | f32 | 8 | `1024x64` | 0.008 ± 0.000 | 0.009 ± 0.000 | unsupported | 0.054 ± 0.004 | 0.086 ± 0.004 | - | - |
| cpu/neural_public_api | `silu` | f32 | 8 | `4096x256` | 0.129 ± 0.002 | 0.134 ± 0.010 | unsupported | 0.264 ± 0.044 | 0.609 ± 0.023 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.103 ± 0.000 | 0.133 ± 0.002 | unsupported | 0.122 ± 0.001 | 0.257 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.317 ± 0.006 | 1.373 ± 0.022 | unsupported | 1.160 ± 0.004 | 4.117 ± 0.029 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.069 ± 0.000 | 0.097 ± 0.000 | unsupported | 0.041 ± 0.001 | 0.218 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.446 ± 0.010 | 0.501 ± 0.017 | unsupported | 0.283 ± 0.004 | 1.829 ± 0.025 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 8 | `1024x64` | 0.070 ± 0.000 | 0.095 ± 0.000 | unsupported | 0.028 ± 0.002 | 0.223 ± 0.006 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 8 | `4096x256` | 0.487 ± 0.013 | 0.585 ± 0.009 | unsupported | 0.160 ± 0.010 | 1.351 ± 0.096 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.159 ± 0.000 | 0.160 ± 0.001 | unsupported | 0.210 ± 0.001 | 0.349 ± 0.002 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 2.552 ± 0.006 | 2.565 ± 0.004 | unsupported | 3.347 ± 0.017 | 5.700 ± 0.009 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.042 ± 0.000 | 0.043 ± 0.000 | unsupported | 0.115 ± 0.002 | 0.188 ± 0.006 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.643 ± 0.003 | 0.643 ± 0.003 | unsupported | 0.907 ± 0.030 | 2.583 ± 0.027 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 8 | `1024x64` | 0.022 ± 0.000 | 0.023 ± 0.000 | unsupported | 0.117 ± 0.003 | 0.195 ± 0.004 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 8 | `4096x256` | 0.323 ± 0.003 | 0.326 ± 0.002 | unsupported | 0.476 ± 0.008 | 1.840 ± 0.034 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.818 ± 0.001 | 1.350 ± 0.002 | unsupported | 0.325 ± 0.001 | 0.143 ± 0.001 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 12.455 ± 0.018 | 20.823 ± 0.111 | unsupported | 6.555 ± 0.032 | 2.300 ± 0.054 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.328 ± 0.001 | 0.875 ± 0.004 | unsupported | 0.178 ± 0.005 | 0.085 ± 0.006 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 5.613 ± 0.035 | 13.711 ± 0.174 | unsupported | 1.474 ± 0.025 | 0.836 ± 0.013 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 8 | `1024x64` | 0.345 ± 0.003 | 0.894 ± 0.006 | unsupported | 0.187 ± 0.004 | 0.076 ± 0.005 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 8 | `4096x256` | 4.529 ± 0.061 | 13.446 ± 0.048 | unsupported | 1.060 ± 0.038 | 0.655 ± 0.039 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 23.373 ± 0.067 | - | unsupported | 23.822 ± 0.116 | - | 23.328 ± 0.044 | 90.387 ± 0.298 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 17.529 ± 0.138 | - | unsupported | 17.417 ± 0.097 | - | 23.350 ± 0.057 | 26.611 ± 0.177 |
| cpu/output_reuse | `add_into` | f64 | 8 | `33554432` | 18.563 ± 0.069 | - | unsupported | 18.774 ± 0.117 | - | 23.535 ± 0.063 | 19.087 ± 0.160 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 16.292 ± 0.056 | - | unsupported | 16.392 ± 0.069 | - | 16.288 ± 0.054 | 16.477 ± 0.106 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 12.490 ± 0.127 | - | unsupported | 12.524 ± 0.208 | - | 16.538 ± 0.061 | 12.575 ± 0.134 |
| cpu/output_reuse | `conj_into` | c64 | 8 | `16777216` | 13.742 ± 0.126 | - | unsupported | 13.672 ± 0.119 | - | 16.683 ± 0.048 | 13.857 ± 0.217 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 13.888 ± 0.159 | - | unsupported | 16.476 ± 0.092 | - | 13.230 ± 0.081 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 12.493 ± 0.117 | - | unsupported | 12.468 ± 0.326 | - | 13.456 ± 0.077 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 8 | `33554432` | 13.725 ± 0.109 | - | unsupported | 13.730 ± 0.117 | - | 13.339 ± 0.058 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 23.661 ± 0.186 | - | unsupported | 23.803 ± 0.155 | - | 23.559 ± 0.170 | 91.518 ± 0.251 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 17.735 ± 0.088 | - | unsupported | 17.452 ± 0.071 | - | 23.747 ± 0.170 | 27.060 ± 0.064 |
| cpu/output_reuse | `div_into` | f64 | 8 | `33554432` | 18.763 ± 0.146 | - | unsupported | 18.784 ± 0.110 | - | 24.051 ± 0.057 | 19.262 ± 0.238 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 80.174 ± 0.092 | - | unsupported | 75.905 ± 0.120 | - | 80.042 ± 0.130 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 20.193 ± 0.039 | - | unsupported | 21.703 ± 0.039 | - | 20.313 ± 0.042 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 8 | `1024x1024` | 10.552 ± 0.027 | - | unsupported | 12.066 ± 0.106 | - | 10.658 ± 0.030 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 79.973 ± 0.261 | - | unsupported | 75.539 ± 0.095 | - | 79.861 ± 0.262 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 20.115 ± 0.022 | - | unsupported | 21.447 ± 0.082 | - | 20.243 ± 0.041 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 8 | `1024x1024` | 10.456 ± 0.020 | - | unsupported | 11.768 ± 0.043 | - | 10.502 ± 0.028 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 23.327 ± 0.073 | - | unsupported | 23.763 ± 0.080 | - | 23.335 ± 0.057 | 90.494 ± 0.308 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 17.506 ± 0.075 | - | unsupported | 17.352 ± 0.138 | - | 23.367 ± 0.069 | 26.652 ± 0.091 |
| cpu/output_reuse | `mul_into` | f64 | 8 | `33554432` | 18.617 ± 0.102 | - | unsupported | 18.687 ± 0.192 | - | 23.759 ± 0.082 | 19.165 ± 0.226 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 16.516 ± 0.043 | - | unsupported | 16.607 ± 0.073 | - | 16.485 ± 0.050 | 66.557 ± 0.437 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 12.492 ± 0.063 | - | unsupported | 12.743 ± 0.096 | - | 16.695 ± 0.274 | 20.523 ± 0.180 |
| cpu/output_reuse | `neg_into` | f64 | 8 | `33554432` | 13.709 ± 0.075 | - | unsupported | 13.757 ± 0.094 | - | 16.652 ± 0.076 | 14.419 ± 0.614 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.957 ± 0.003 | unsupported | unsupported | 1.169 ± 0.003 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 16.094 ± 0.013 | unsupported | unsupported | 17.716 ± 0.024 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.544 ± 0.001 | unsupported | unsupported | 0.473 ± 0.002 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 7.529 ± 0.010 | unsupported | unsupported | 5.071 ± 0.018 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 8 | `256x256,rhs=16` | 0.508 ± 0.002 | unsupported | unsupported | 0.458 ± 0.001 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 8 | `768x768,rhs=16` | 4.706 ± 0.019 | unsupported | unsupported | 3.173 ± 0.010 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 23.345 ± 0.095 | - | unsupported | 23.836 ± 0.059 | - | 23.347 ± 0.084 | 90.160 ± 0.221 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 17.493 ± 0.095 | - | unsupported | 17.377 ± 0.092 | - | 23.414 ± 0.300 | 26.757 ± 0.111 |
| cpu/output_reuse | `sub_into` | f64 | 8 | `33554432` | 18.611 ± 0.113 | - | unsupported | 18.702 ± 0.058 | - | 23.386 ± 0.121 | 19.123 ± 0.086 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 80.242 ± 1.173 | - | 0.003 ± 0.000 | 78.740 ± 0.602 | 80.544 ± 1.052 | 98.893 ± 1.731 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 28.309 ± 0.088 | - | 0.003 ± 0.000 | 21.789 ± 0.189 | 42.857 ± 0.806 | 98.235 ± 1.755 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 8 | `8192x1 -> 8192x4096` | 15.999 ± 0.158 | - | 0.003 ± 0.000 | 13.790 ± 0.323 | 31.556 ± 0.247 | 102.445 ± 1.248 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 45.161 ± 1.247 | - | 146.121 ± 1.823 | 45.918 ± 0.228 | 57.921 ± 0.743 | 50.604 ± 0.488 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 14.205 ± 0.177 | - | 56.222 ± 0.763 | 13.708 ± 0.123 | 31.286 ± 1.027 | 50.779 ± 0.693 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 8 | `33554432` | 10.639 ± 0.215 | - | 45.392 ± 0.853 | 10.328 ± 0.573 | 22.644 ± 0.330 | 51.266 ± 0.960 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 165.417 ± 2.299 | - | 164.126 ± 1.993 | 159.861 ± 1.187 | 206.795 ± 0.742 | 188.152 ± 4.098 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 44.112 ± 0.182 | - | 43.982 ± 0.117 | 43.251 ± 0.229 | 86.524 ± 0.473 | 189.592 ± 0.806 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 8 | `8192 -> 8192x8192` | 25.850 ± 0.340 | - | 25.699 ± 0.168 | 26.108 ± 0.210 | 60.395 ± 0.274 | 195.323 ± 6.016 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 43.336 ± 0.323 | - | 42.565 ± 0.347 | 55.137 ± 2.866 | 44.506 ± 1.007 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 24.359 ± 0.282 | - | 23.529 ± 0.242 | 15.756 ± 0.274 | 25.840 ± 0.170 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 8 | `8388608x2x2 -> 8388608x2` | 24.820 ± 0.289 | - | 23.983 ± 0.389 | 10.718 ± 0.254 | 18.953 ± 0.431 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 89.691 ± 0.772 | - | 0.003 ± 0.000 | 82.758 ± 0.293 | 93.102 ± 0.537 | 104.053 ± 0.297 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 25.621 ± 0.151 | - | 0.003 ± 0.000 | 24.245 ± 0.940 | 31.729 ± 4.215 | 105.272 ± 0.853 | - |
| cpu/structural_shape | `reshape` | f64 | 8 | `33554432 -> 8192x4096` | 16.625 ± 0.378 | - | 0.003 ± 0.000 | 16.458 ± 0.807 | 18.788 ± 0.529 | 106.903 ± 0.861 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 71.185 ± 0.100 | - | 0.003 ± 0.000 | 56.919 ± 0.077 | 63.624 ± 0.435 | 222.875 ± 1.038 | 128.427 ± 1.256 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 22.146 ± 0.608 | - | 0.003 ± 0.000 | 56.872 ± 0.054 | 33.894 ± 0.275 | 224.169 ± 1.313 | 42.645 ± 2.944 |
| cpu/structural_shape | `transpose` | f64 | 8 | `4096x4096` | 15.349 ± 0.501 | - | 0.003 ± 0.000 | 57.640 ± 0.184 | 24.302 ± 0.904 | 226.463 ± 0.496 | 33.079 ± 2.029 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 43.065 ± 0.872 | - | 43.583 ± 0.458 | 52.955 ± 0.926 | 54.149 ± 8.758 | 54.746 ± 0.759 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 13.677 ± 0.182 | - | 13.580 ± 0.239 | 15.780 ± 0.151 | 25.901 ± 0.362 | 55.049 ± 0.844 | - |
| cpu/structural_shape | `tril` | f64 | 8 | `4096x4096` | 8.144 ± 0.122 | - | 8.091 ± 0.113 | 9.327 ± 0.143 | 18.623 ± 0.207 | 55.282 ± 0.552 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 42.814 ± 0.343 | - | 42.915 ± 0.525 | 51.547 ± 1.229 | 47.240 ± 1.726 | 52.606 ± 0.284 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 13.389 ± 0.055 | - | 13.419 ± 0.175 | 14.097 ± 0.251 | 25.836 ± 0.731 | 54.275 ± 3.514 | - |
| cpu/structural_shape | `triu` | f64 | 8 | `4096x4096` | 8.007 ± 0.101 | - | 8.030 ± 0.161 | 8.558 ± 0.152 | 18.512 ± 0.618 | 59.265 ± 0.168 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 8 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.001 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.001 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 8 | `33554432 -> 8192x4096` | 0.001 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 8 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 8 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 7.225615 | 451600.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 5.788525 | 361782.81 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 6.287507 | 392969.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 2.989043 | 186815.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 2.565303 | 160331.44 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 3.124607 | 195287.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 16 | 3.985775 | 249110.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 8 | tenferro-rs direct API (ms) | 16 | 2.094815 | 130925.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 16 | 2.683088 | 167693.00 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 11.797430 | 737339.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 11.765240 | 735327.50 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 6.251618 | 390726.12 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 6.602728 | 412670.50 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 8 | tenferro-rs direct API (ms) | 16 | 8.135684 | 508480.25 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 8 | tenferro-rs eager mode (ms) | 16 | 8.302295 | 518893.44 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 14.474161 | 904635.06 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 12.216358 | 763522.38 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 12.476967 | 779810.44 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 12.815589 | 800974.31 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 199.748735 | 12484295.94 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 181.862213 | 11366388.31 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 195.343519 | 12208969.94 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 197.539059 | 12346191.19 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 17.890645 | 1118165.31 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 6.359462 | 397466.38 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 7.277311 | 454831.94 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 180.749822 | 11296863.88 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 129.956284 | 8122267.75 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 75.060017 | 4691251.06 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 94.645635 | 5915352.19 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 304.327515 | 19020469.69 |
| householder_qr | f64 | `256x128` | 8 | JAX Python (XLA CPU) (ms) | 16 | 18.094995 | 1130937.19 |
| householder_qr | f64 | `256x128` | 8 | PyTorch Python (ms) | 16 | 5.527331 | 345458.19 |
| householder_qr | f64 | `256x128` | 8 | tenferro-rs direct API (ms) | 16 | 6.315504 | 394719.00 |
| householder_qr | f64 | `256x128` | 8 | tenferro-rs eager mode (ms) | 16 | 63.991526 | 3999470.38 |
| householder_qr | f64 | `768x384` | 8 | JAX Python (XLA CPU) (ms) | 16 | 124.369026 | 7773064.12 |
| householder_qr | f64 | `768x384` | 8 | PyTorch Python (ms) | 16 | 44.656781 | 2791048.81 |
| householder_qr | f64 | `768x384` | 8 | tenferro-rs direct API (ms) | 16 | 78.009127 | 4875570.44 |
| householder_qr | f64 | `768x384` | 8 | tenferro-rs eager mode (ms) | 16 | 306.874437 | 19179652.31 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 27.129738 | 1695608.62 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 27.288866 | 1705554.12 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 453.242720 | 28327670.00 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 456.877765 | 28554860.31 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 16.771977 | 1048248.56 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 199.963612 | 12497725.75 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 204.680656 | 12792541.00 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 372.054733 | 23253420.81 |
| rank_revealing_qr | f64 | `256x128` | 8 | tenferro-rs direct API (ms) | 16 | 14.926526 | 932907.88 |
| rank_revealing_qr | f64 | `256x128` | 8 | tenferro-rs eager mode (ms) | 16 | 197.042896 | 12315181.00 |
| rank_revealing_qr | f64 | `768x384` | 8 | tenferro-rs direct API (ms) | 16 | 167.981637 | 10498852.31 |
| rank_revealing_qr | f64 | `768x384` | 8 | tenferro-rs eager mode (ms) | 16 | 510.827396 | 31926712.25 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 29.150863 | 1821928.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 29.379180 | 1836198.75 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 429.390564 | 26836910.25 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 431.504980 | 26969061.25 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.417167 | 1338572.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 108.991008 | 6811938.00 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 200.966505 | 12560406.56 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 365.013452 | 22813340.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 8 | tenferro-rs direct API (ms) | 16 | 20.259465 | 1266216.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 8 | tenferro-rs eager mode (ms) | 16 | 47.988309 | 2999269.31 |
| rank_revealing_qr_deficient | f64 | `768x384` | 8 | tenferro-rs direct API (ms) | 16 | 167.912038 | 10494502.38 |
| rank_revealing_qr_deficient | f64 | `768x384` | 8 | tenferro-rs eager mode (ms) | 16 | 492.514074 | 30782129.62 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.831447 | 122366.36 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 11.100193 | 173440.52 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.388028 | 131062.94 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.431650 | 131744.53 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.675232 | 1334404.00 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 17.365315 | 2170664.38 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 16.768622 | 2096077.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 16.789520 | 2098690.00 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.374657 | 37104.02 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.614477 | 56476.20 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.372274 | 37066.78 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.436613 | 38072.08 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.261228 | 907653.50 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.999235 | 749904.38 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.225637 | 528204.62 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.167812 | 645976.50 |
| erf | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 2.370642 | 37041.28 |
| erf | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 2.217144 | 34642.88 |
| erf | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 1.389774 | 21715.22 |
| erf | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 1.449756 | 22652.44 |
| erf | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 2.838703 | 354837.88 |
| erf | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 3.264041 | 408005.12 |
| erf | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 2.147665 | 268458.12 |
| erf | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 2.165127 | 270640.88 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.481905 | 210654.77 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.544791 | 55387.36 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.150343 | 64849.11 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.179427 | 65303.55 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 26.559157 | 3319894.62 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.013210 | 1376651.25 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 8.211969 | 1026496.12 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 8.362550 | 1045318.75 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.864825 | 91637.89 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.285898 | 51342.16 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.157855 | 18091.48 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.195143 | 18674.11 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.649632 | 831204.00 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.367940 | 420992.50 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.076667 | 259583.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.112793 | 264099.12 |
| gelu | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 4.097695 | 64026.48 |
| gelu | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 2.825516 | 44148.69 |
| gelu | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 0.679654 | 10619.59 |
| gelu | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 0.705742 | 11027.22 |
| gelu | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 3.486538 | 435817.25 |
| gelu | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 2.174435 | 271804.38 |
| gelu | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 1.143957 | 142994.62 |
| gelu | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 1.154477 | 144309.62 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.681218 | 151269.03 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 16.011837 | 250184.95 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.194919 | 190545.61 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.234503 | 191164.11 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.961818 | 2370227.25 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 31.876178 | 3984522.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 24.376863 | 3047107.88 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 24.487419 | 3060927.38 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.583063 | 87235.36 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.638564 | 72477.56 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.185230 | 49769.22 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.222078 | 50344.97 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.338060 | 917257.50 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.856384 | 857048.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.118734 | 764841.75 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.127309 | 765913.62 |
| gelu_tanh | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 5.730052 | 89532.06 |
| gelu_tanh | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 2.740678 | 42823.09 |
| gelu_tanh | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 1.689061 | 26391.58 |
| gelu_tanh | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 1.716422 | 26819.09 |
| gelu_tanh | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 5.310191 | 663773.88 |
| gelu_tanh | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 4.378153 | 547269.12 |
| gelu_tanh | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 3.066824 | 383353.00 |
| gelu_tanh | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 3.095858 | 386982.25 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.200901 | 159389.08 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 11.295658 | 176494.66 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.087745 | 63871.02 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.436562 | 100571.28 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.697855 | 2587231.88 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.444268 | 1305533.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.466972 | 683371.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.952479 | 744059.88 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.620701 | 134698.45 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.616771 | 56512.05 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.854520 | 60226.88 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.936927 | 92764.48 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.481953 | 1060244.12 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.709170 | 338646.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.535627 | 316953.38 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.941479 | 367684.88 |
| layer_norm | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 8.891566 | 138930.72 |
| layer_norm | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 2.385156 | 37268.06 |
| layer_norm | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 4.057075 | 63391.80 |
| layer_norm | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 6.006221 | 93847.20 |
| layer_norm | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 6.507510 | 813438.75 |
| layer_norm | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 1.398922 | 174865.25 |
| layer_norm | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 2.876650 | 359581.25 |
| layer_norm | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 3.446461 | 430807.62 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.803815 | 262559.61 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.015383 | 109615.36 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.325525 | 98836.33 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.923009 | 123797.02 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 33.545372 | 4193171.50 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.338756 | 1042344.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.302303 | 1287787.88 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.916030 | 1364503.75 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.578679 | 227791.86 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.462462 | 38475.97 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.453690 | 69588.91 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.989725 | 93589.45 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.976888 | 1872111.00 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.132832 | 266604.00 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.566596 | 445824.50 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.006281 | 500785.13 |
| log_softmax | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 14.348339 | 224192.80 |
| log_softmax | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 2.003536 | 31305.25 |
| log_softmax | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 4.643366 | 72552.59 |
| log_softmax | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 6.085478 | 95085.59 |
| log_softmax | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 11.340112 | 1417514.00 |
| log_softmax | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 1.393201 | 174150.12 |
| log_softmax | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 3.895165 | 486895.62 |
| log_softmax | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 4.722554 | 590319.25 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 18.224461 | 284757.20 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 15.816985 | 247140.39 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 18.143971 | 283499.55 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 38.672334 | 4834041.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 28.435274 | 3554409.25 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 28.924608 | 3615576.00 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.869915 | 247967.42 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.977386 | 124646.66 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.192339 | 159255.30 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 18.974389 | 2371798.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.698452 | 1087306.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.242219 | 1155277.38 |
| masked_softmax | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 16.191669 | 252994.83 |
| masked_softmax | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 6.914971 | 108046.42 |
| masked_softmax | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 8.957580 | 139962.19 |
| masked_softmax | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 15.733798 | 1966724.75 |
| masked_softmax | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 6.802562 | 850320.25 |
| masked_softmax | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 7.150409 | 893801.12 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.632436 | 41131.81 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.961927 | 15030.11 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.851210 | 13300.16 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.421715 | 22214.30 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.372263 | 421532.88 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.631690 | 78961.25 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 1.014275 | 126784.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.105044 | 138130.50 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.370410 | 37037.66 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.631440 | 9866.25 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.621691 | 9713.92 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.202848 | 18794.50 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.797827 | 224728.38 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.226460 | 28307.50 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.365938 | 45742.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.405371 | 50671.38 |
| reduce_mean | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 2.362517 | 36914.33 |
| reduce_mean | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 0.607580 | 9493.44 |
| reduce_mean | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 0.673883 | 10529.42 |
| reduce_mean | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 1.249504 | 19523.50 |
| reduce_mean | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 1.315648 | 164456.00 |
| reduce_mean | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 0.138227 | 17278.38 |
| reduce_mean | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 0.209540 | 26192.50 |
| reduce_mean | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 0.299046 | 37380.75 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.455810 | 163372.03 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.067846 | 79185.09 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.248149 | 35127.33 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.670296 | 57348.38 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.874731 | 2609341.38 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.076384 | 509548.00 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.795812 | 349476.50 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.086004 | 385750.50 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.441542 | 116274.09 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.031680 | 62995.00 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.020672 | 31573.00 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.370373 | 52662.08 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.655125 | 1081890.62 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.767295 | 220911.88 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.419640 | 177455.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.655336 | 206917.00 |
| rms_norm | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 7.508142 | 117314.72 |
| rms_norm | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 4.301331 | 67208.30 |
| rms_norm | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 2.273488 | 35523.25 |
| rms_norm | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 3.638678 | 56854.34 |
| rms_norm | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 6.315822 | 789477.75 |
| rms_norm | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 1.558699 | 194837.38 |
| rms_norm | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 1.438666 | 179833.25 |
| rms_norm | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 1.679744 | 209968.00 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.530748 | 133292.94 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.282216 | 82534.62 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.799029 | 43734.83 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.833973 | 44280.83 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 16.686624 | 2085828.00 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.485675 | 1310709.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.589590 | 698698.75 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.754037 | 719254.62 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.997205 | 78081.33 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.415469 | 53366.70 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.836840 | 13075.62 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 0.867357 | 13552.45 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.118226 | 764778.25 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.075819 | 384477.38 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.420341 | 177542.62 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.454674 | 181834.25 |
| sigmoid | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 5.574823 | 87106.61 |
| sigmoid | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 3.389755 | 52964.92 |
| sigmoid | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 0.525368 | 8208.88 |
| sigmoid | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 0.548591 | 8571.73 |
| sigmoid | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 4.669740 | 583717.50 |
| sigmoid | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 2.080420 | 260052.50 |
| sigmoid | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 1.037089 | 129636.12 |
| sigmoid | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 1.035316 | 129414.50 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.759358 | 136864.97 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.418340 | 84661.56 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.994553 | 46789.89 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.996587 | 46821.67 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.167466 | 2145933.25 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.720965 | 1340120.62 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.870626 | 733828.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.044069 | 755508.62 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.122458 | 80038.41 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.403766 | 53183.84 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.865122 | 13517.53 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 0.901350 | 14083.59 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.451134 | 806391.75 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.328307 | 416038.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.480512 | 185064.00 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.530355 | 191294.38 |
| silu | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 5.514071 | 86157.36 |
| silu | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 3.457080 | 54016.88 |
| silu | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 0.536789 | 8387.33 |
| silu | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 0.567455 | 8866.48 |
| silu | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 4.874150 | 609268.75 |
| silu | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 2.109053 | 263631.62 |
| silu | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 1.034604 | 129325.50 |
| silu | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 1.069469 | 133683.62 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.473650 | 257400.78 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.814356 | 122099.31 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.576925 | 102764.45 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.505668 | 132901.06 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 32.936122 | 4117015.25 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.276206 | 1159525.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.539768 | 1317471.00 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.982745 | 1372843.12 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.961944 | 218155.38 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.615235 | 40863.05 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.430836 | 69231.81 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.214420 | 97100.31 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.634593 | 1829324.12 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.264136 | 283017.00 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.567919 | 445989.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.006191 | 500773.88 |
| softmax | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 14.250989 | 222671.70 |
| softmax | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 1.794818 | 28044.03 |
| softmax | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 4.479963 | 69999.42 |
| softmax | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 6.104894 | 95388.97 |
| softmax | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 10.805638 | 1350704.75 |
| softmax | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 1.282105 | 160263.12 |
| softmax | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 3.896427 | 487053.38 |
| softmax | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 4.676488 | 584561.00 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 22.318034 | 348719.28 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 13.458397 | 210287.45 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 10.202707 | 159417.30 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.226392 | 159787.38 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 45.602271 | 5700283.88 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 26.774810 | 3346851.25 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 20.412448 | 2551556.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 20.519188 | 2564898.50 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.042681 | 188166.89 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.343095 | 114735.86 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.690844 | 42044.44 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.725508 | 42586.06 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 20.667661 | 2583457.62 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.255303 | 906912.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.142235 | 642779.38 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.142145 | 642768.12 |
| softplus | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 12.494900 | 195232.81 |
| softplus | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 7.487296 | 116989.00 |
| softplus | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 1.439857 | 22497.77 |
| softplus | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 1.477398 | 23084.34 |
| softplus | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 14.721384 | 1840173.00 |
| softplus | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 3.810448 | 476306.00 |
| softplus | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 2.583605 | 322950.62 |
| softplus | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 2.607841 | 325980.12 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.126953 | 142608.64 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 20.783268 | 324738.56 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 52.365553 | 818211.77 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 86.426776 | 1350418.38 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.396877 | 2299609.62 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 52.437723 | 6554715.38 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 99.642830 | 12455353.75 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 166.582384 | 20822798.00 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.413177 | 84580.89 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.393129 | 178017.64 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 21.012780 | 328324.69 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 56.031755 | 875496.17 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.688605 | 836075.62 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 11.792880 | 1474110.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 44.902033 | 5612754.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 109.684494 | 13710561.75 |
| take_along_axis | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 4.875753 | 76183.64 |
| take_along_axis | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 11.984822 | 187262.84 |
| take_along_axis | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 22.063114 | 344736.16 |
| take_along_axis | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 57.214402 | 893975.03 |
| take_along_axis | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 5.239701 | 654962.62 |
| take_along_axis | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 8.483570 | 1060446.25 |
| take_along_axis | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 36.229243 | 4528655.38 |
| take_along_axis | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 107.571392 | 13446424.00 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 18.707663 | 1169228.94 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 15.318556 | 957409.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 283.455056 | 17715941.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 257.510510 | 16094406.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 7.561530 | 472595.62 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 8.697546 | 543596.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 81.128853 | 5070553.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 120.460282 | 7528767.62 |
| solve_read_into | f64 | `256x256,rhs=16` | 8 | PyTorch Python (ms) | 16 | 7.333310 | 458331.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 8 | tenferro-rs direct API (ms) | 16 | 8.132602 | 508287.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 8 | PyTorch Python (ms) | 16 | 50.760554 | 3172534.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 8 | tenferro-rs direct API (ms) | 16 | 75.300768 | 4706298.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.029535 | 1845.94 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003005 | 187.81 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.029866 | 1866.62 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002846 | 177.88 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 8 | PyTorch Python (ms) | 16 | 0.029555 | 1847.19 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 8 | tenferro-rs direct API (ms) | 16 | 0.002775 | 173.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000141 | 8.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.027231 | 1701.94 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.009959 | 622.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000311 | 19.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.027320 | 1707.50 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.009527 | 595.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 8 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000270 | 16.88 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 8 | PyTorch Python (ms) | 16 | 0.027221 | 1701.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 8 | tenferro-rs direct API (ms) | 16 | 0.010409 | 650.56 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000210 | 13.12 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.030016 | 1876.00 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.006993 | 437.06 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000280 | 17.50 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.028362 | 1772.62 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.006792 | 424.50 |
| slice_view | f64 | `4194304 -> 2096128` | 8 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000280 | 17.50 |
| slice_view | f64 | `4194304 -> 2096128` | 8 | PyTorch Python (ms) | 16 | 0.028443 | 1777.69 |
| slice_view | f64 | `4194304 -> 2096128` | 8 | tenferro-rs direct API (ms) | 16 | 0.006682 | 417.62 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000140 | 8.75 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.026680 | 1667.50 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.005080 | 317.50 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000140 | 8.75 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.026890 | 1680.62 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.004658 | 291.12 |
| transpose_view | f64 | `4096x4096` | 8 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000140 | 8.75 |
| transpose_view | f64 | `4096x4096` | 8 | PyTorch Python (ms) | 16 | 0.026850 | 1678.12 |
| transpose_view | f64 | `4096x4096` | 8 | tenferro-rs direct API (ms) | 16 | 0.004287 | 267.94 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=8, shape=`448x448`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 12.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 25.452 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/chain_log1p_exp_mul` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`julia-base`, 84.815 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/chain_log1p_exp_mul` (f64, threads=8, shape=`4194304`): `tenferro-direct` is 14.4x faster than the slowest successful cell (`julia-base`, 85.275 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/cos` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`jax-cpu`, 75.434 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/cos` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`jax-cpu`, 75.434 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/cos` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 21.3x faster than the slowest successful cell (`julia-base`, 74.454 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/cos` (f64, threads=8, shape=`8388608`): `tenferro-trace` is 21.3x faster than the slowest successful cell (`julia-base`, 74.454 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 18.5x faster than the slowest successful cell (`julia-base`, 65.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 18.4x faster than the slowest successful cell (`julia-base`, 65.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 18.5x faster than the slowest successful cell (`julia-base`, 62.800 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=8, shape=`8388608`): `tenferro-trace` is 17.3x faster than the slowest successful cell (`julia-base`, 62.800 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/expm1` (f64, threads=8, shape=`4194304`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`julia-base`, 37.987 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/expm1` (f64, threads=8, shape=`4194304`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`julia-base`, 37.987 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`julia-base`, 77.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`julia-base`, 77.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log` (f64, threads=8, shape=`8388608`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`julia-base`, 81.770 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 24.4x faster than the slowest successful cell (`julia-base`, 81.770 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log` (f64, threads=8, shape=`8388608`): `tenferro-trace` is 23.0x faster than the slowest successful cell (`julia-base`, 81.770 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 15.1x faster than the slowest successful cell (`julia-base`, 45.453 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`julia-base`, 45.453 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=8, shape=`4194304`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`julia-base`, 45.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=8, shape=`4194304`): `tenferro-direct` is 27.3x faster than the slowest successful cell (`julia-base`, 45.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=8, shape=`4194304`): `tenferro-trace` is 27.6x faster than the slowest successful cell (`julia-base`, 45.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/pow` (f64, threads=4, shape=`4194304`): `pytorch-cpu` is 13.8x faster than the slowest successful cell (`julia-base`, 183.544 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/pow` (f64, threads=8, shape=`4194304`): `pytorch-cpu` is 26.0x faster than the slowest successful cell (`julia-base`, 182.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/pow` (f64, threads=8, shape=`4194304`): `strided-jl` is 10.9x faster than the slowest successful cell (`julia-base`, 182.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/pow` (f64, threads=8, shape=`4194304`): `tenferro-direct` is 19.6x faster than the slowest successful cell (`julia-base`, 182.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/pow` (f64, threads=8, shape=`4194304`): `tenferro-trace` is 19.6x faster than the slowest successful cell (`julia-base`, 182.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 45.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 24.4x faster than the slowest successful cell (`pytorch-cpu`, 45.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 45.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 45.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 11.534 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 17.3x faster than the slowest successful cell (`pytorch-cpu`, 44.460 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 25.1x faster than the slowest successful cell (`pytorch-cpu`, 44.460 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 44.460 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 44.460 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 11.314 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 11.314 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=8, shape=`2048x2048`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 7.680 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/rem` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`julia-base`, 73.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/rem` (f64, threads=8, shape=`8388608`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`julia-base`, 73.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/rsqrt` (f64, threads=8, shape=`33554432`): `pytorch-cpu` is 14.6x faster than the slowest successful cell (`julia-base`, 286.369 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/rsqrt` (f64, threads=8, shape=`33554432`): `tenferro-direct` is 15.7x faster than the slowest successful cell (`julia-base`, 286.369 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/rsqrt` (f64, threads=8, shape=`33554432`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`julia-base`, 286.369 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 15.6x faster than the slowest successful cell (`jax-cpu`, 75.706 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`jax-cpu`, 75.706 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 20.3x faster than the slowest successful cell (`julia-base`, 70.393 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=8, shape=`8388608`): `tenferro-trace` is 19.7x faster than the slowest successful cell (`julia-base`, 70.393 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sqrt` (f64, threads=8, shape=`33554432`): `tenferro-direct` is 13.4x faster than the slowest successful cell (`julia-base`, 221.776 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sqrt` (f64, threads=8, shape=`33554432`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`julia-base`, 221.776 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`julia-base`, 85.701 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=8, shape=`8388608`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`julia-base`, 85.701 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 14.0x faster than the slowest successful cell (`tenferro-direct`, 2.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 14.6x faster than the slowest successful cell (`tenferro-direct`, 2.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`tenferro-direct`, 2.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=8, shape=`2097152`): `pytorch-cpu` is 11.5x faster than the slowest successful cell (`julia-base`, 6.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=8, shape=`2097152`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`julia-base`, 6.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 2158.7x faster than the slowest successful cell (`julia-base`, 7.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 2206.7x faster than the slowest successful cell (`julia-base`, 7.297 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=8, shape=`4194304 -> 2096128`): `tenferro-trace` is 2256.1x faster than the slowest successful cell (`julia-base`, 7.436 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x128x128`): `jax-cpu` is 73.3x faster than the slowest successful cell (`tenferro-trace`, 10.494 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x128x128`): `pytorch-cpu` is 64.5x faster than the slowest successful cell (`tenferro-trace`, 10.494 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x128x128`): `tenferro-direct` is 134.1x faster than the slowest successful cell (`tenferro-trace`, 10.494 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x128x128`): `jax-cpu` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 13.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x128x128`): `pytorch-cpu` is 11.7x faster than the slowest successful cell (`tenferro-trace`, 13.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x128x128`): `tenferro-direct` is 21.9x faster than the slowest successful cell (`tenferro-trace`, 13.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=8, shape=`8x128x128`): `jax-cpu` is 12.2x faster than the slowest successful cell (`tenferro-trace`, 18.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=8, shape=`8x128x128`): `pytorch-cpu` is 16.2x faster than the slowest successful cell (`tenferro-trace`, 18.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=8, shape=`8x128x128`): `tenferro-direct` is 22.1x faster than the slowest successful cell (`tenferro-trace`, 18.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`1x128x128`): `pytorch-cpu` is 10.8x faster than the slowest successful cell (`tenferro-trace`, 20.563 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=8, shape=`1x128x128`): `pytorch-cpu` is 13.0x faster than the slowest successful cell (`tenferro-trace`, 20.607 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`1x128x128`): `pytorch-cpu` is 10.0x faster than the slowest successful cell (`tenferro-trace`, 10.884 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=8, shape=`1x128x128`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`tenferro-trace`, 10.761 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x128x128`): `jax-cpu` is 41.1x faster than the slowest successful cell (`tenferro-trace`, 10.579 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x128x128`): `pytorch-cpu` is 51.2x faster than the slowest successful cell (`tenferro-trace`, 10.579 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x128x128`): `tenferro-direct` is 59.7x faster than the slowest successful cell (`tenferro-trace`, 10.579 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x128x128`): `jax-cpu` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 11.772 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x128x128`): `pytorch-cpu` is 26.5x faster than the slowest successful cell (`tenferro-trace`, 11.772 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`1024x8x8`): `pytorch-cpu` is 13.9x faster than the slowest successful cell (`tenferro-trace`, 2.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`1x128x128`): `jax-cpu` is 42.8x faster than the slowest successful cell (`tenferro-trace`, 10.699 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`1x128x128`): `pytorch-cpu` is 51.8x faster than the slowest successful cell (`tenferro-trace`, 10.699 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`1x128x128`): `tenferro-direct` is 48.4x faster than the slowest successful cell (`tenferro-trace`, 10.699 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`8x128x128`): `jax-cpu` is 20.8x faster than the slowest successful cell (`tenferro-trace`, 15.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`8x128x128`): `pytorch-cpu` is 48.3x faster than the slowest successful cell (`tenferro-trace`, 15.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=8, shape=`1x128x128`): `jax-cpu` is 12.4x faster than the slowest successful cell (`tenferro-trace`, 11.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=8, shape=`1x128x128`): `pytorch-cpu` is 25.6x faster than the slowest successful cell (`tenferro-trace`, 11.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=8, shape=`1x128x128`): `tenferro-direct` is 20.2x faster than the slowest successful cell (`tenferro-trace`, 11.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `jax-cpu` is 52.7x faster than the slowest successful cell (`tenferro-trace`, 10.557 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `pytorch-cpu` is 59.8x faster than the slowest successful cell (`tenferro-trace`, 10.557 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `tenferro-direct` is 75.2x faster than the slowest successful cell (`tenferro-trace`, 10.557 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `jax-cpu` is 17.5x faster than the slowest successful cell (`tenferro-trace`, 14.505 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 36.5x faster than the slowest successful cell (`tenferro-trace`, 14.505 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `tenferro-direct` is 13.0x faster than the slowest successful cell (`tenferro-trace`, 14.505 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`1x128x128,rhs=1`): `jax-cpu` is 47.4x faster than the slowest successful cell (`tenferro-trace`, 10.629 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`1x128x128,rhs=1`): `pytorch-cpu` is 59.2x faster than the slowest successful cell (`tenferro-trace`, 10.629 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`1x128x128,rhs=1`): `tenferro-direct` is 56.3x faster than the slowest successful cell (`tenferro-trace`, 10.629 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`8x128x128,rhs=1`): `jax-cpu` is 23.5x faster than the slowest successful cell (`tenferro-trace`, 14.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 50.2x faster than the slowest successful cell (`tenferro-trace`, 14.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_lu_factor` (f64, threads=8, shape=`1024x16x16,rhs=1`): `pytorch-cpu` is 14.4x faster than the slowest successful cell (`julia-base`, 2.760 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_lu_factor` (f64, threads=8, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`julia-base`, 0.809 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `jax-cpu` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 11.297 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `pytorch-cpu` is 28.4x faster than the slowest successful cell (`tenferro-eager`, 11.297 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 24.8x faster than the slowest successful cell (`tenferro-eager`, 11.297 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=8, shape=`256x128`): `pytorch-cpu` is 11.6x faster than the slowest successful cell (`tenferro-eager`, 3.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=8, shape=`256x128`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 3.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `pytorch-cpu` is 11.5x faster than the slowest successful cell (`jax-cpu`, 18.904 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 17.3x faster than the slowest successful cell (`jax-cpu`, 18.904 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 25.6x faster than the slowest successful cell (`jax-cpu`, 18.904 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 25.1x faster than the slowest successful cell (`jax-cpu`, 11.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=8, shape=`2048x2048`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 8.923 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`tenferro-eager`, 12.498 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr` (f64, threads=8, shape=`256x128`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`tenferro-eager`, 12.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=8, shape=`1024x64`): `tenferro-eager` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 10.4x faster than the slowest successful cell (`tenferro-eager`, 0.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 16.4x faster than the slowest successful cell (`tenferro-eager`, 13.711 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=8, shape=`1024x64`): `jax-cpu` is 11.7x faster than the slowest successful cell (`tenferro-eager`, 0.894 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=8, shape=`4096x256`): `jax-cpu` is 20.5x faster than the slowest successful cell (`tenferro-eager`, 13.446 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=8, shape=`4096x256`): `pytorch-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 13.446 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 30086.1x faster than the slowest successful cell (`julia-base`, 98.893 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 30843.1x faster than the slowest successful cell (`julia-base`, 98.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=8, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 32667.5x faster than the slowest successful cell (`julia-base`, 102.445 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 32865.9x faster than the slowest successful cell (`julia-base`, 104.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 31556.4x faster than the slowest successful cell (`julia-base`, 105.272 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=8, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 31377.4x faster than the slowest successful cell (`julia-base`, 106.903 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 74415.5x faster than the slowest successful cell (`julia-base`, 222.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`julia-base`, 224.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 76612.6x faster than the slowest successful cell (`julia-base`, 224.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=8, shape=`4096x4096`): `tenferro-direct` is 14.8x faster than the slowest successful cell (`julia-base`, 226.463 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=8, shape=`4096x4096`): `tenferro-trace` is 72445.1x faster than the slowest successful cell (`julia-base`, 226.463 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=8, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 189.1x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 89.8x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=8, shape=`33554432 -> 8192x4096`): `julia-base` is 100.1x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 144.3x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 98.5x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=8, shape=`4194304 -> 2096128`): `julia-base` is 98.8x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 185.3x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 186.8x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=8, shape=`4096x4096`): `julia-base` is 186.4x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 4936.4 GB/s at 0.003 ms. The conservative physical lower bound is 0.168 ms, so the cell is 49.4x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 5070.8 GB/s at 0.003 ms. The conservative physical lower bound is 0.042 ms, so the cell is 12.7x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 81665.8 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 816.7x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 84281.1 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 210.7x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=8, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 85598.0 GB/s at 0.003 ms. The conservative physical lower bound is 0.336 ms, so the cell is 107.0x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 84786.9 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 847.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 80466.3 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 201.2x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=8, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 78789.4 GB/s at 0.003 ms. The conservative physical lower bound is 0.336 ms, so the cell is 98.5x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 44813.9 GB/s at 0.003 ms. The conservative physical lower bound is 1.342 ms, so the cell is 448.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 45870.7 GB/s at 0.003 ms. The conservative physical lower bound is 0.336 ms, so the cell is 114.7x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=8, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 42935.9 GB/s at 0.003 ms. The conservative physical lower bound is 0.168 ms, so the cell is 53.7x faster than the bound.

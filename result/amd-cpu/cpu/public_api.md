# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261124_184947/run.yaml`
- Timestamp: `2026-11-24T18:49:47.772181+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261124_184947`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

## CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

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

- CSV: `data/results/amd-cpu/cpu/public_api/20261124_184947/cpu_public_api_t1_20261124_184947.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261124_184947/cpu_public_api_t4_20261124_184947.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.265 ± 0.034 | - | 3.214 ± 0.062 | 4.250 ± 0.010 | 3.896 ± 0.069 | 4.004 ± 0.041 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.471 ± 0.045 | - | 336.003 ± 0.024 | 2.170 ± 0.024 | 2.819 ± 0.083 | 2.992 ± 2.702 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 116.109 ± 4.278 | - | 117.025 ± 7.020 | 113.837 ± 4.527 | 116.244 ± 5.023 | 127.098 ± 17.847 | 128.864 ± 7.485 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 40.158 ± 0.989 | - | 140.812 ± 9.921 | 40.144 ± 1.247 | 56.851 ± 3.399 | 127.757 ± 6.074 | 63.063 ± 21.526 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 99.264 ± 4.149 | - | 112.784 ± 6.384 | 94.260 ± 1.184 | 84.361 ± 3.656 | 122.816 ± 7.706 | 136.433 ± 9.031 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 40.452 ± 2.087 | - | 117.969 ± 3.398 | 30.645 ± 0.923 | 42.749 ± 2.017 | 116.081 ± 7.320 | 53.115 ± 2.692 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 42.208 ± 0.789 | - | 41.945 ± 0.707 | 44.558 ± 0.630 | 62.191 ± 1.024 | 46.586 ± 1.934 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 22.934 ± 0.856 | - | 51.763 ± 5.922 | 29.153 ± 0.679 | 28.488 ± 0.701 | 24.923 ± 6.889 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 41.569 ± 0.852 | - | unsupported | 44.910 ± 1.286 | 63.863 ± 1.709 | 49.376 ± 6.901 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 22.979 ± 0.419 | - | unsupported | 30.161 ± 0.846 | 29.678 ± 1.161 | 28.053 ± 4.433 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 6.700 ± 0.065 | - | 6.644 ± 0.046 | 10.651 ± 0.088 | 10.801 ± 0.121 | 10.478 ± 0.296 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 9.556 ± 0.131 | - | 5456.989 ± 182.988 | 13.520 ± 0.629 | 16.371 ± 1.732 | 17.113 ± 0.415 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 62.272 ± 1.256 | - | 62.088 ± 0.662 | 104.317 ± 2.418 | 93.319 ± 2.221 | 69.119 ± 10.443 | 67.341 ± 3.223 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 29.762 ± 0.471 | - | 84.000 ± 0.010 | 43.645 ± 0.487 | 45.313 ± 0.603 | 70.385 ± 5.158 | 34.403 ± 6.390 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 99.814 ± 0.347 | - | 99.817 ± 0.888 | 137.823 ± 2.296 | 155.240 ± 6.117 | 121.591 ± 2.648 | 123.546 ± 8.525 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 47.679 ± 0.974 | - | 119.999 ± 12.010 | 61.576 ± 7.768 | 64.590 ± 1.658 | 127.276 ± 2.785 | 58.365 ± 25.603 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 67.168 ± 3.712 | - | 67.022 ± 4.485 | 67.045 ± 0.582 | 69.368 ± 1.585 | 69.356 ± 1.062 | 75.018 ± 3.676 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 23.093 ± 0.693 | - | 82.400 ± 1.278 | 23.626 ± 2.395 | 33.511 ± 1.375 | 71.741 ± 1.213 | 32.949 ± 1.254 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.440 ± 0.029 | - | 17.351 ± 0.205 | 20.299 ± 0.524 | 17.539 ± 0.360 | 3.811 ± 0.086 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.337 ± 0.042 | - | 59.985 ± 0.033 | 10.903 ± 0.443 | 3.432 ± 0.275 | 3.950 ± 0.083 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 6.740 ± 0.037 | - | 6.573 ± 0.049 | 8.860 ± 0.231 | 7.484 ± 0.236 | 8.750 ± 0.227 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.358 ± 0.065 | - | 527.986 ± 0.028 | 5.553 ± 0.086 | 7.378 ± 0.478 | 7.433 ± 0.269 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 4.002 ± 0.041 | - | 4.125 ± 0.031 | 5.504 ± 0.130 | 5.259 ± 0.069 | 4.576 ± 0.084 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.183 ± 2.757 | - | 791.995 ± 0.005 | 3.376 ± 0.127 | 3.994 ± 1.239 | 3.459 ± 0.303 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 5.630 ± 0.022 | - | 5.634 ± 0.063 | 7.148 ± 0.059 | 4.806 ± 0.156 | 4.666 ± 0.039 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 5.950 ± 0.100 | - | 6.040 ± 0.314 | 5.667 ± 0.309 | 5.992 ± 0.350 | 6.348 ± 0.109 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 41.551 ± 0.471 | - | 41.814 ± 0.358 | 44.730 ± 0.905 | 62.396 ± 1.188 | 43.849 ± 0.579 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 23.054 ± 0.235 | - | 53.000 ± 4.724 | 29.365 ± 1.206 | 28.774 ± 0.464 | 23.662 ± 1.440 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 43.300 ± 0.699 | - | unsupported | 50.601 ± 0.584 | 49.691 ± 0.665 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 21.156 ± 0.286 | - | unsupported | 26.436 ± 0.510 | 30.522 ± 0.465 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 115.974 ± 9.074 | - | 114.992 ± 9.550 | 112.839 ± 2.277 | 106.959 ± 7.159 | 124.451 ± 12.958 | 162.615 ± 1.624 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 39.920 ± 2.439 | - | 130.439 ± 4.328 | 41.390 ± 4.101 | 62.453 ± 4.620 | 127.187 ± 12.783 | 73.301 ± 5.468 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 129.140 ± 7.766 | - | 127.980 ± 8.408 | 143.118 ± 3.957 | 130.831 ± 3.820 | 145.392 ± 10.100 | 189.556 ± 6.384 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 43.288 ± 0.305 | - | 142.339 ± 19.082 | 48.314 ± 3.531 | 71.921 ± 2.595 | 143.007 ± 11.353 | 74.231 ± 2.571 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 16.023 ± 0.310 | - | 44.837 ± 1.097 | 61.276 ± 1.205 | 48.074 ± 1.076 | 66.082 ± 8.381 | 72.038 ± 9.867 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 8.589 ± 0.185 | - | 108.007 ± 9.999 | 27.612 ± 0.974 | 27.988 ± 1.311 | 71.966 ± 1.381 | 31.301 ± 14.640 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 8.483 ± 0.301 | - | 9.129 ± 0.309 | 34.222 ± 2.053 | 36.455 ± 0.631 | 34.461 ± 8.014 | 47.411 ± 5.751 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 6.654 ± 0.178 | - | 23.999 ± 0.015 | 12.169 ± 0.434 | 19.749 ± 0.734 | 36.335 ± 8.777 | 18.243 ± 15.418 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 19.638 ± 0.447 | - | 19.858 ± 0.823 | 47.088 ± 0.906 | 43.565 ± 1.583 | 19.499 ± 0.255 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 15.456 ± 0.399 | - | 35.998 ± 0.008 | 18.153 ± 0.475 | 24.774 ± 0.705 | 18.654 ± 0.346 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 14.999 ± 0.072 | - | 15.024 ± 0.299 | 63.478 ± 5.714 | 104.429 ± 1.473 | 52.679 ± 10.165 | 60.316 ± 7.684 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 6.487 ± 0.030 | - | 35.994 ± 0.012 | 22.359 ± 3.927 | 53.463 ± 2.055 | 58.578 ± 13.616 | 25.834 ± 14.794 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 132.501 ± 5.929 | - | 130.422 ± 8.245 | 122.360 ± 4.619 | 136.149 ± 5.612 | 139.424 ± 2.928 | 188.799 ± 10.481 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 43.234 ± 0.876 | - | 141.211 ± 6.286 | 45.147 ± 0.476 | 70.048 ± 2.412 | 143.891 ± 4.878 | 76.990 ± 2.539 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 9.172 ± 2.393 | - | 9.010 ± 0.096 | 42.120 ± 0.668 | 36.017 ± 1.229 | 54.686 ± 9.965 | 58.802 ± 12.735 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 4.366 ± 0.053 | - | 24.003 ± 0.013 | 15.662 ± 2.074 | 19.294 ± 0.651 | 59.659 ± 14.834 | 27.930 ± 17.836 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 12.135 ± 0.188 | - | 12.274 ± 0.126 | 24.142 ± 1.246 | 21.827 ± 0.971 | 31.970 ± 8.617 | 33.550 ± 9.406 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 5.013 ± 0.022 | - | 35.998 ± 0.008 | 9.063 ± 0.519 | 11.633 ± 0.483 | 34.522 ± 5.803 | 15.343 ± 12.251 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 15.433 ± 0.264 | - | 15.339 ± 0.081 | 46.735 ± 0.707 | 57.688 ± 1.183 | 68.836 ± 10.789 | 71.941 ± 12.152 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 7.588 ± 0.103 | - | 35.996 ± 0.005 | 17.655 ± 0.625 | 30.900 ± 0.866 | 73.779 ± 14.098 | 32.218 ± 17.249 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 7.935 ± 0.218 | - | 7.878 ± 0.150 | 21.154 ± 1.327 | 33.199 ± 0.768 | 42.537 ± 9.815 | 39.814 ± 8.900 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.783 ± 0.168 | - | 23.999 ± 0.010 | 7.812 ± 0.973 | 17.684 ± 0.399 | 40.168 ± 2.808 | 18.210 ± 12.360 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.352 ± 0.006 | 0.372 ± 0.017 | unsupported | unsupported | 0.439 ± 0.039 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.245 ± 0.005 | 115.928 ± 28.046 | unsupported | unsupported | 0.350 ± 0.017 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 132.799 ± 8.255 | - | 131.443 ± 10.575 | 127.053 ± 1.937 | 123.471 ± 0.908 | 145.756 ± 26.332 | 192.077 ± 4.861 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 42.306 ± 1.150 | - | 143.552 ± 4.011 | 43.672 ± 1.971 | 68.542 ± 4.739 | 144.165 ± 10.018 | 82.449 ± 2.823 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 131.782 ± 1.783 | - | 129.923 ± 6.687 | 127.110 ± 3.408 | 125.655 ± 2.008 | 140.496 ± 13.677 | 192.244 ± 4.969 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 41.377 ± 0.218 | - | 143.647 ± 0.866 | 43.977 ± 1.093 | 70.879 ± 6.609 | 150.862 ± 6.868 | 82.209 ± 2.147 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 128.021 ± 4.944 | - | 129.832 ± 8.104 | 122.274 ± 3.378 | 130.020 ± 4.437 | 140.979 ± 11.464 | 189.179 ± 4.068 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 43.062 ± 0.990 | - | 142.957 ± 1.493 | 45.356 ± 1.372 | 70.330 ± 1.569 | 141.206 ± 5.663 | 80.338 ± 3.809 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 117.228 ± 14.439 | - | 111.211 ± 7.900 | 110.364 ± 4.023 | 111.553 ± 6.078 | 124.234 ± 2.653 | 165.777 ± 2.052 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 38.216 ± 0.528 | - | 131.005 ± 1.523 | 39.379 ± 1.629 | 63.984 ± 4.789 | 131.182 ± 14.268 | 70.273 ± 2.166 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 49.395 ± 0.508 | - | 51.056 ± 0.480 | 38.071 ± 0.978 | 63.575 ± 1.165 | 112.504 ± 2.394 | 101.974 ± 20.694 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 22.247 ± 0.516 | - | 59.998 ± 0.009 | 13.984 ± 0.465 | 33.581 ± 1.084 | 122.369 ± 1.616 | 41.741 ± 16.884 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 12.887 ± 0.067 | - | 12.876 ± 0.130 | 8.613 ± 0.174 | 43.360 ± 0.298 | 8.232 ± 0.052 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.984 ± 0.071 | - | 35.998 ± 0.008 | 7.545 ± 0.274 | 21.425 ± 0.516 | 8.764 ± 0.187 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.711 ± 0.049 | - | 1.702 ± 0.007 | 24.833 ± 0.450 | 2.246 ± 0.033 | 0.940 ± 0.037 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.758 ± 0.022 | - | 23.996 ± 0.005 | 12.184 ± 1.324 | 1.677 ± 0.107 | 1.030 ± 0.026 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.993 ± 0.023 | - | 1.995 ± 0.015 | 5.247 ± 0.091 | 3.849 ± 0.014 | 1.003 ± 0.096 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.098 ± 0.073 | - | 24.000 ± 0.010 | 2.853 ± 0.197 | 2.896 ± 0.083 | 1.241 ± 0.091 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 12.399 ± 0.077 | - | 12.847 ± 0.208 | 8.569 ± 0.390 | 42.802 ± 0.362 | 8.097 ± 0.061 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.928 ± 0.149 | - | 35.999 ± 0.007 | 7.344 ± 0.093 | 22.104 ± 0.224 | 8.929 ± 0.643 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.650 ± 0.011 | - | 1.621 ± 0.015 | 25.193 ± 0.332 | 2.257 ± 0.017 | 1.030 ± 0.008 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.769 ± 0.063 | - | 24.001 ± 0.010 | 11.902 ± 0.381 | 1.649 ± 0.379 | 1.000 ± 0.025 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 9.756 ± 0.080 | - | 9.668 ± 0.322 | 15.778 ± 0.259 | 15.082 ± 0.165 | 7.835 ± 0.154 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 4.786 ± 0.065 | - | 35.998 ± 0.011 | 8.195 ± 0.391 | 8.530 ± 3.735 | 8.205 ± 0.104 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 9.484 ± 0.228 | - | 9.413 ± 0.433 | 8.531 ± 0.445 | 31.148 ± 0.681 | 7.903 ± 0.194 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.306 ± 0.075 | - | 23.998 ± 0.004 | 7.535 ± 0.125 | 16.218 ± 1.351 | 8.919 ± 0.612 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.788 ± 0.058 | - | 0.753 ± 0.122 | 2.677 ± 0.149 | 1.703 ± 0.033 | 0.637 ± 0.038 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.367 ± 0.046 | - | 23.999 ± 0.009 | 1.078 ± 0.108 | 1.761 ± 0.028 | 0.695 ± 0.081 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.625 ± 0.144 | - | 1.471 ± 0.056 | 0.711 ± 0.013 | 1.145 ± 0.020 | 0.746 ± 0.081 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.886 ± 0.050 | - | 23.998 ± 0.009 | 0.319 ± 0.030 | 1.141 ± 0.040 | 0.898 ± 0.059 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 9.909 ± 0.412 | - | 8.558 ± 0.452 | 8.424 ± 0.294 | 30.579 ± 0.321 | 7.524 ± 0.359 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 7.296 ± 0.242 | - | 23.999 ± 0.017 | 9.966 ± 4.309 | 16.607 ± 1.850 | 9.092 ± 0.283 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.698 ± 0.136 | - | 0.713 ± 0.175 | 2.351 ± 0.073 | 1.751 ± 0.078 | 0.615 ± 0.025 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.436 ± 0.089 | - | 24.000 ± 0.008 | 1.052 ± 0.034 | 1.821 ± 0.038 | 0.915 ± 0.094 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.702 ± 0.092 | - | 1.649 ± 0.164 | 0.708 ± 0.023 | 1.158 ± 0.010 | 0.812 ± 0.121 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.877 ± 0.356 | - | 23.994 ± 0.023 | 0.298 ± 0.029 | 1.062 ± 0.026 | 0.968 ± 0.233 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.744 ± 0.013 | 0.782 ± 0.013 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.301 ± 0.005 | 23.999 ± 0.064 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 34.639 ± 0.603 | - | 37.147 ± 0.053 | 56.861 ± 0.623 | 64.349 ± 1.283 | 61.733 ± 18.351 | 69.197 ± 7.401 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 16.089 ± 0.202 | - | 47.997 ± 0.011 | 20.935 ± 1.500 | 35.919 ± 2.924 | 65.182 ± 13.560 | 39.752 ± 18.757 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 128.352 ± 3.776 | - | 127.896 ± 8.418 | 129.936 ± 7.713 | 217.622 ± 12.304 | 242.641 ± 12.631 | 241.406 ± 10.217 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 38.831 ± 0.312 | - | 141.694 ± 4.376 | 42.954 ± 1.974 | 96.070 ± 2.086 | 249.990 ± 45.009 | 106.605 ± 5.984 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 127.656 ± 5.508 | - | 129.086 ± 7.423 | 143.324 ± 10.114 | 152.270 ± 2.867 | 145.974 ± 3.667 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 44.186 ± 2.161 | - | 143.377 ± 1.396 | 50.165 ± 1.060 | 88.264 ± 6.473 | 145.112 ± 4.105 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 113.848 ± 10.633 | - | 116.633 ± 4.203 | 118.722 ± 3.192 | 104.714 ± 0.878 | 124.372 ± 12.144 | 171.004 ± 3.946 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 38.023 ± 0.501 | - | 131.181 ± 9.402 | 42.127 ± 1.973 | 62.440 ± 3.571 | 127.004 ± 2.185 | 72.212 ± 9.335 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 12.076 ± 0.251 | - | 12.579 ± 0.251 | 57.813 ± 1.262 | 101.912 ± 2.337 | 56.095 ± 17.577 | 59.354 ± 7.978 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 5.396 ± 0.078 | - | 23.997 ± 0.009 | 22.514 ± 0.172 | 52.743 ± 1.162 | 64.236 ± 9.254 | 36.799 ± 12.011 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 115.484 ± 3.346 | - | 115.949 ± 11.027 | 168.829 ± 3.649 | 131.801 ± 5.277 | 195.765 ± 10.375 | 202.355 ± 4.251 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 37.603 ± 0.988 | - | 128.533 ± 3.869 | 63.557 ± 1.638 | 64.932 ± 3.868 | 198.298 ± 6.185 | 88.428 ± 1.890 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 119.743 ± 9.669 | - | 127.849 ± 9.752 | 124.431 ± 5.696 | 129.794 ± 8.683 | 143.923 ± 7.635 | 190.817 ± 3.561 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 43.110 ± 0.327 | - | 143.254 ± 11.794 | 47.529 ± 1.770 | 69.516 ± 3.748 | 141.729 ± 7.154 | 77.202 ± 3.129 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 29.866 ± 0.229 | - | 30.573 ± 0.337 | 78.326 ± 3.543 | 34.322 ± 1.099 | 72.395 ± 14.245 | 73.224 ± 13.996 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 12.764 ± 0.158 | - | 47.997 ± 0.009 | 27.896 ± 1.138 | 17.967 ± 1.162 | 69.456 ± 13.769 | 31.053 ± 13.102 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 121.267 ± 7.553 | 130.021 ± 6.587 | unsupported | 146.657 ± 9.495 | 157.837 ± 5.321 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 43.737 ± 0.319 | 142.450 ± 11.822 | unsupported | 50.578 ± 0.689 | 80.542 ± 2.015 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.794 ± 0.072 | - | 0.826 ± 0.111 | 0.733 ± 0.060 | 0.746 ± 0.039 | 7.781 ± 4.972 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.282 ± 0.042 | - | 11.997 ± 0.021 | 0.331 ± 0.033 | 0.834 ± 0.085 | 7.117 ± 4.021 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.867 ± 0.167 | - | 0.966 ± 0.217 | 0.682 ± 0.040 | 0.709 ± 0.218 | 8.224 ± 5.175 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.413 ± 0.119 | - | 12.000 ± 0.008 | 0.365 ± 0.033 | 0.570 ± 0.343 | 7.645 ± 4.000 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.256 ± 0.062 | - | unsupported | 1.046 ± 0.111 | 1.044 ± 0.135 | 3.795 ± 12.345 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.652 ± 0.171 | - | unsupported | 0.692 ± 0.269 | 0.977 ± 0.210 | 8.016 ± 8.583 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.321 ± 0.073 | - | 0.360 ± 0.016 | 0.722 ± 0.017 | 0.292 ± 0.025 | 1.127 ± 0.110 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.311 ± 0.059 | - | 47.997 ± 0.005 | 0.383 ± 0.015 | 0.175 ± 0.007 | 1.166 ± 0.086 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.869 ± 0.236 | - | 0.754 ± 0.115 | 1.011 ± 0.030 | 0.971 ± 0.031 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.438 ± 0.140 | - | 11.999 ± 0.004 | 0.506 ± 0.026 | 0.985 ± 0.016 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.805 ± 0.068 | - | 0.830 ± 0.092 | 0.948 ± 0.210 | 0.918 ± 0.223 | 7.740 ± 10.673 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.300 ± 0.152 | - | 11.999 ± 0.012 | 0.411 ± 0.062 | 0.637 ± 0.086 | 8.629 ± 8.583 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.679 ± 0.011 | - | 0.750 ± 0.010 | 0.349 ± 0.029 | 0.347 ± 0.011 | 1.232 ± 0.083 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.939 ± 0.008 | - | 36.041 ± 0.244 | 0.470 ± 0.042 | 0.355 ± 0.024 | 1.262 ± 0.079 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.495 ± 0.068 | - | 0.002 ± 0.000 | 1.435 ± 0.083 | 1.269 ± 0.159 | 9.601 ± 2.727 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.824 ± 0.034 | - | 0.002 ± 0.000 | 0.939 ± 0.130 | 1.031 ± 0.308 | 9.457 ± 4.753 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.084 ± 0.024 | - | 0.088 ± 0.023 | 0.066 ± 0.001 | 0.068 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.124 ± 0.001 | - | 0.127 ± 0.006 | 0.092 ± 0.001 | 0.132 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.273 ± 0.017 | - | 0.275 ± 0.059 | 0.265 ± 0.012 | 0.269 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.047 ± 0.000 | - | 0.051 ± 0.006 | 0.129 ± 0.003 | 0.103 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.001 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.001 | 0.004 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.006 ± 0.002 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.381 ± 0.011 | - | 0.335 ± 0.011 | 1.046 ± 0.018 | 0.643 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.016 ± 0.000 | - | 0.021 ± 0.000 | 0.035 ± 0.000 | 0.058 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.070 ± 0.000 | - | 0.071 ± 0.001 | 0.084 ± 0.001 | 0.064 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.121 ± 0.001 | - | 0.125 ± 0.000 | 0.119 ± 0.001 | 0.111 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.089 ± 0.001 | - | 12.000 ± 0.009 | 0.295 ± 0.013 | 0.314 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.069 ± 0.001 | - | 23.998 ± 0.011 | 0.169 ± 0.003 | 0.104 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.172 ± 0.026 | - | 12.001 ± 0.003 | 1.332 ± 0.024 | 0.999 ± 1.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.012 ± 0.001 | - | 11.998 ± 0.007 | 0.054 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.397 ± 0.014 | - | 0.400 ± 0.009 | 0.601 ± 0.015 | 0.738 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.985 ± 0.016 | - | 1.991 ± 0.029 | 1.937 ± 0.021 | 2.220 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.393 ± 0.056 | - | 6.653 ± 0.726 | 5.874 ± 0.086 | 7.824 ± 0.219 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.144 ± 0.039 | - | 2.119 ± 0.012 | 3.019 ± 0.022 | 2.262 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.124 ± 0.016 | - | 0.134 ± 0.028 | 0.105 ± 0.002 | 0.125 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.001 | 0.012 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.025 ± 0.001 | 0.049 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.027 ± 0.004 | - | 0.028 ± 0.000 | 0.029 ± 0.001 | 0.073 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 17.245 ± 0.203 | - | 17.187 ± 0.461 | 24.491 ± 0.460 | 18.724 ± 0.353 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.792 ± 0.168 | - | 0.795 ± 0.017 | 0.790 ± 0.007 | 0.816 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.017 ± 0.000 | - | 0.019 ± 0.000 | 0.021 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.052 ± 0.002 | - | 0.054 ± 0.000 | 0.053 ± 0.001 | 0.102 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.183 ± 0.003 | - | 11.997 ± 0.005 | 0.928 ± 0.318 | 0.746 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 0.754 ± 0.012 | - | 12.000 ± 0.003 | 1.956 ± 0.218 | 2.145 ± 0.172 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 2.260 ± 0.020 | - | 12.001 ± 0.006 | 8.349 ± 0.162 | 5.363 ± 0.505 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 2.145 ± 0.020 | - | 2.184 ± 0.046 | 2.343 ± 0.068 | 3.066 ± 0.093 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.098 ± 0.001 | - | 0.104 ± 0.001 | 0.159 ± 0.007 | 0.121 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.002 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.020 ± 0.000 | - | 0.024 ± 0.000 | 0.025 ± 0.001 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.028 ± 0.000 | - | 0.029 ± 0.000 | 0.031 ± 0.001 | 0.060 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 6.641 ± 0.099 | - | 23.998 ± 0.013 | 17.449 ± 0.547 | 24.403 ± 8.365 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.308 ± 0.010 | - | 12.001 ± 0.013 | 1.240 ± 0.026 | 1.994 ± 3.973 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.021 ± 0.001 | 0.031 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.052 ± 0.002 | - | 0.056 ± 0.001 | 0.055 ± 0.000 | 0.090 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.385 ± 0.023 | - | 0.385 ± 0.012 | 0.372 ± 0.010 | 0.315 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.748 ± 0.021 | - | 1.717 ± 0.011 | 1.668 ± 0.018 | 1.541 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 6.025 ± 0.051 | - | 6.067 ± 0.066 | 4.945 ± 0.115 | 5.684 ± 0.135 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.552 ± 0.022 | - | 1.529 ± 0.021 | 1.229 ± 0.011 | 1.290 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.078 ± 0.001 | - | 0.079 ± 0.001 | 0.074 ± 0.002 | 0.123 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.020 ± 0.000 | - | 0.021 ± 0.000 | 0.022 ± 0.001 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.025 ± 0.000 | - | 0.027 ± 0.000 | 0.026 ± 0.001 | 0.036 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 12.403 ± 1.488 | - | 12.453 ± 0.281 | 9.815 ± 0.204 | 9.548 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.615 ± 0.010 | - | 0.620 ± 0.015 | 0.546 ± 0.009 | 0.684 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.015 ± 0.000 | - | 0.016 ± 0.000 | 0.019 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.049 ± 0.000 | - | 0.051 ± 0.001 | 0.045 ± 0.001 | 0.071 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.382 ± 0.008 | - | 0.396 ± 0.027 | 0.621 ± 0.015 | 0.297 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 0.681 ± 0.012 | - | 12.002 ± 0.044 | 1.698 ± 0.008 | 1.000 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 2.053 ± 0.012 | - | 12.001 ± 0.006 | 7.543 ± 0.185 | 3.153 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.537 ± 0.018 | - | 1.654 ± 0.296 | 1.128 ± 0.007 | 1.992 ± 0.544 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.087 ± 0.003 | - | 0.081 ± 0.002 | 0.112 ± 0.002 | 0.109 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.020 ± 0.000 | - | 0.023 ± 0.000 | 0.021 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.001 | 0.017 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.027 ± 0.000 | - | 0.028 ± 0.000 | 0.026 ± 0.000 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 4.427 ± 0.095 | - | 23.998 ± 0.006 | 9.487 ± 0.625 | 11.529 ± 1.548 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.226 ± 0.002 | - | 11.999 ± 0.007 | 0.819 ± 0.009 | 0.377 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.017 ± 0.000 | - | 0.018 ± 0.000 | 0.027 ± 0.001 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.050 ± 0.000 | - | 0.054 ± 0.000 | 0.046 ± 0.000 | 0.059 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.268 ± 0.015 | - | 0.269 ± 0.009 | 0.190 ± 0.003 | 0.307 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.386 ± 0.008 | - | 1.398 ± 0.009 | 0.809 ± 0.009 | 1.548 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 4.423 ± 0.040 | - | 4.408 ± 0.022 | 2.570 ± 0.028 | 5.566 ± 0.134 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.829 ± 0.014 | - | 0.834 ± 0.014 | 0.504 ± 0.018 | 1.275 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.057 ± 0.000 | - | 0.059 ± 0.000 | 0.036 ± 0.002 | 0.102 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.001 | 0.017 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.018 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 6.434 ± 0.090 | - | 6.550 ± 0.094 | 3.925 ± 0.032 | 9.685 ± 0.214 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.447 ± 0.012 | - | 0.450 ± 0.013 | 0.221 ± 0.002 | 0.639 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.036 ± 0.000 | - | 0.039 ± 0.000 | 0.027 ± 0.001 | 0.066 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.261 ± 0.000 | - | 0.270 ± 0.006 | 0.329 ± 0.008 | 0.321 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.530 ± 0.015 | - | 12.001 ± 0.003 | 0.843 ± 0.018 | 1.014 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 1.526 ± 0.022 | - | 12.001 ± 0.009 | 3.214 ± 0.103 | 3.126 ± 0.126 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.903 ± 0.009 | - | 0.881 ± 5.002 | 0.651 ± 0.010 | 1.417 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.060 ± 0.000 | - | 0.062 ± 0.001 | 0.045 ± 0.002 | 0.089 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.002 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.019 ± 0.001 | - | 0.017 ± 0.000 | 0.016 ± 0.002 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.018 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 2.319 ± 0.101 | - | 11.999 ± 0.003 | 5.426 ± 0.252 | 11.524 ± 1.665 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.152 ± 0.005 | - | 11.998 ± 0.021 | 0.272 ± 0.008 | 0.368 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.019 ± 0.001 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.037 ± 0.000 | - | 0.041 ± 0.000 | 0.028 ± 0.000 | 0.058 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.105 ± 0.001 | - | 0.107 ± 0.002 | 0.162 ± 0.000 | 0.096 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.207 ± 0.002 | - | 0.209 ± 0.005 | 0.224 ± 0.011 | 0.172 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.608 ± 0.007 | - | 0.615 ± 0.011 | 0.467 ± 0.011 | 0.503 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.107 ± 0.002 | - | 0.108 ± 0.005 | 0.149 ± 0.002 | 0.161 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.001 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.010 ± 0.000 | - | 0.012 ± 0.001 | 0.019 ± 0.001 | 0.037 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.800 ± 0.016 | - | 0.801 ± 0.004 | 1.105 ± 0.012 | 1.087 ± 0.147 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.050 ± 0.000 | - | 0.051 ± 0.001 | 0.067 ± 0.003 | 0.085 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.114 ± 0.002 | - | 0.117 ± 0.001 | 0.143 ± 0.003 | 0.085 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.237 ± 0.002 | - | 0.241 ± 0.015 | 0.186 ± 0.007 | 0.171 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.285 ± 0.013 | - | 11.998 ± 0.007 | 0.334 ± 0.013 | 0.548 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.159 ± 0.005 | - | 71.993 ± 0.020 | 0.191 ± 0.015 | 0.191 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.026 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.011 ± 0.000 | - | 0.014 ± 0.001 | 0.029 ± 0.001 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.019 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.019 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.020 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.019 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.019 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.408 ± 0.012 | - | 11.998 ± 0.009 | 0.563 ± 0.021 | 0.765 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.021 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.030 ± 0.001 | - | 11.999 ± 0.004 | 0.060 ± 0.001 | 0.126 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.021 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.023 ± 0.000 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.201 ± 0.008 | - | 0.204 ± 0.001 | 0.191 ± 0.002 | 0.253 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.566 ± 0.012 | - | 0.576 ± 0.009 | 0.465 ± 0.007 | 0.664 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 2.226 ± 0.175 | - | 2.207 ± 0.011 | 1.179 ± 0.037 | 1.861 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.385 ± 0.027 | - | 0.383 ± 0.009 | 0.502 ± 0.009 | 0.540 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.027 ± 0.001 | - | 0.029 ± 0.001 | 0.022 ± 0.000 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.023 ± 0.000 | - | 0.025 ± 0.000 | 0.024 ± 0.000 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.009 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.994 ± 0.010 | - | 2.981 ± 0.047 | 3.906 ± 0.035 | 4.346 ± 0.123 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.199 ± 0.001 | - | 0.205 ± 0.008 | 0.142 ± 0.001 | 0.163 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.013 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.208 ± 0.018 | - | 0.208 ± 0.012 | 0.287 ± 0.010 | 0.264 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.219 ± 0.003 | - | 12.003 ± 0.021 | 1.014 ± 5.608 | 0.655 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.807 ± 0.035 | - | 11.998 ± 0.009 | 1.722 ± 0.053 | 1.110 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.385 ± 0.025 | - | 0.387 ± 0.017 | 0.492 ± 0.014 | 1.090 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.028 ± 0.001 | - | 0.032 ± 0.000 | 0.037 ± 0.000 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.024 ± 0.000 | - | 0.025 ± 0.000 | 0.042 ± 0.001 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.001 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 1.393 ± 0.017 | - | 11.999 ± 0.006 | 3.866 ± 0.075 | 9.990 ± 10.551 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.100 ± 0.002 | - | 11.999 ± 0.004 | 0.246 ± 0.014 | 0.168 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.021 ± 0.000 | 0.027 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.210 ± 0.038 | - | 0.121 ± 0.001 | 0.085 ± 0.001 | 0.203 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.237 ± 0.013 | - | 0.273 ± 0.071 | 0.130 ± 0.004 | 0.289 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.773 ± 0.204 | - | 0.908 ± 0.275 | 0.376 ± 0.013 | 0.634 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.116 ± 0.035 | - | 0.108 ± 0.023 | 0.126 ± 0.015 | 0.169 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.008 ± 0.000 | - | 0.012 ± 0.001 | 0.021 ± 0.001 | 0.048 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.002 | 0.013 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.002 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.004 ± 0.001 | - | 0.009 ± 0.001 | 0.013 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.665 ± 0.109 | - | 0.675 ± 0.089 | 0.910 ± 0.012 | 0.886 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.004 ± 0.002 | - | 0.008 ± 0.005 | 0.013 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.046 ± 0.000 | - | 0.052 ± 0.001 | 0.064 ± 0.001 | 0.098 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.016 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.148 ± 0.001 | - | 0.126 ± 0.002 | 0.113 ± 0.001 | 0.219 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.104 ± 0.001 | - | 24.000 ± 0.006 | 0.159 ± 0.002 | 0.283 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.288 ± 0.020 | - | 35.999 ± 0.005 | 0.367 ± 0.012 | 0.702 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.123 ± 0.003 | - | 84.003 ± 0.034 | 0.177 ± 0.009 | 0.133 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.009 ± 0.001 | - | 0.015 ± 0.000 | 0.030 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.011 ± 0.000 | 0.022 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.019 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.003 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.022 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.301 ± 0.016 | - | 35.997 ± 0.013 | 0.543 ± 0.017 | 0.631 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.022 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.024 ± 0.001 | - | 11.998 ± 0.010 | 0.057 ± 0.003 | 0.106 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.023 ± 0.002 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.008 ± 0.000 | - | 0.015 ± 0.000 | 0.026 ± 0.002 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.056 ± 0.310 | - | 1.596 ± 0.480 | 1.287 ± 0.011 | 1.763 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 3.570 ± 0.626 | - | 3.344 ± 0.039 | 4.573 ± 0.241 | 4.479 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 10.794 ± 0.122 | - | 10.827 ± 0.172 | 14.900 ± 1.438 | 12.870 ± 0.133 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 3.632 ± 0.058 | - | 3.633 ± 0.023 | 5.099 ± 0.125 | 3.830 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.164 ± 0.004 | - | 0.166 ± 0.012 | 0.203 ± 0.019 | 0.218 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.133 ± 0.006 | - | 0.136 ± 0.006 | 0.178 ± 0.013 | 0.183 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.023 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.019 ± 0.001 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.034 ± 0.001 | - | 0.036 ± 0.000 | 0.052 ± 0.001 | 0.052 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.015 ± 0.000 | - | 0.024 ± 0.007 | 0.024 ± 0.001 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.044 ± 0.000 | - | 0.046 ± 0.000 | 0.065 ± 0.001 | 0.067 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 29.590 ± 0.777 | - | 28.840 ± 0.239 | 40.834 ± 0.299 | 29.335 ± 0.397 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.281 ± 0.026 | - | 1.282 ± 0.020 | 1.530 ± 0.008 | 1.305 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.037 ± 0.011 | - | 0.031 ± 0.011 | 0.044 ± 0.001 | 0.048 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.085 ± 0.000 | - | 0.087 ± 0.000 | 0.118 ± 0.001 | 0.147 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 0.455 ± 0.011 | - | 11.997 ± 0.010 | 1.293 ± 0.017 | 1.650 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 1.228 ± 0.007 | - | 11.998 ± 0.006 | 4.493 ± 0.095 | 2.158 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 3.671 ± 0.014 | - | 12.097 ± 11.992 | 16.704 ± 0.482 | 6.514 ± 0.338 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 3.894 ± 0.032 | - | 3.883 ± 0.020 | 4.693 ± 0.032 | 4.876 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.180 ± 0.004 | - | 0.176 ± 0.019 | 0.305 ± 0.019 | 0.180 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.144 ± 0.003 | - | 0.149 ± 0.004 | 0.262 ± 0.016 | 0.165 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.023 ± 0.002 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.001 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | 0.015 ± 0.000 | 0.020 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.035 ± 0.000 | - | 0.038 ± 0.000 | 0.050 ± 0.001 | 0.052 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.025 ± 0.002 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.047 ± 0.001 | - | 0.049 ± 0.001 | 0.065 ± 0.001 | 0.063 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 13.916 ± 0.210 | - | 35.992 ± 0.021 | 36.550 ± 0.918 | 61.969 ± 31.354 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.026 ± 0.002 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 0.533 ± 0.018 | - | 11.997 ± 0.011 | 2.301 ± 0.047 | 0.728 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.032 ± 0.000 | - | 0.033 ± 0.001 | 0.044 ± 0.001 | 0.046 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.089 ± 0.000 | - | 0.091 ± 0.001 | 0.118 ± 0.001 | 0.127 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.796 ± 0.008 | - | 0.797 ± 0.009 | 0.930 ± 0.020 | 1.066 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.793 ± 0.024 | - | 2.797 ± 0.025 | 3.799 ± 0.075 | 3.367 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 9.869 ± 0.240 | - | 10.143 ± 1.146 | 11.672 ± 0.317 | 10.401 ± 0.113 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.135 ± 0.032 | - | 2.136 ± 0.008 | 2.214 ± 0.055 | 2.055 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.127 ± 0.004 | - | 0.127 ± 0.002 | 0.137 ± 0.004 | 0.176 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.017 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.120 ± 0.008 | - | 0.089 ± 0.002 | 0.110 ± 0.004 | 0.149 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.021 ± 0.001 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.020 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.031 ± 0.000 | - | 0.033 ± 0.000 | 0.042 ± 0.002 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.023 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.041 ± 0.000 | - | 0.043 ± 0.000 | 0.052 ± 0.000 | 0.053 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 17.149 ± 0.209 | - | 17.269 ± 0.846 | 17.368 ± 0.128 | 15.643 ± 0.368 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.982 ± 0.012 | - | 1.215 ± 0.022 | 1.019 ± 0.009 | 1.060 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.024 ± 0.000 | - | 0.026 ± 0.001 | 0.038 ± 0.000 | 0.070 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.080 ± 0.000 | - | 0.082 ± 0.000 | 0.098 ± 0.001 | 0.107 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.827 ± 0.013 | - | 0.860 ± 6.018 | 0.938 ± 0.468 | 1.083 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 1.069 ± 0.017 | - | 11.998 ± 0.005 | 3.775 ± 0.080 | 1.621 ± 0.272 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 3.485 ± 0.034 | - | 12.000 ± 0.005 | 14.779 ± 0.578 | 5.275 ± 0.226 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.206 ± 0.022 | - | 2.276 ± 0.016 | 2.776 ± 0.105 | 2.995 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.012 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.130 ± 0.003 | - | 0.132 ± 0.003 | 0.222 ± 0.015 | 0.155 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.093 ± 0.003 | - | 0.092 ± 0.002 | 0.170 ± 0.012 | 0.116 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.032 ± 0.002 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.013 ± 0.000 | - | 0.014 ± 0.001 | 0.019 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.034 ± 0.000 | - | 0.036 ± 0.000 | 0.041 ± 0.001 | 0.042 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.024 ± 0.001 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.044 ± 0.000 | - | 0.045 ± 0.000 | 0.053 ± 0.001 | 0.052 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 7.575 ± 0.442 | - | 23.997 ± 0.014 | 21.318 ± 0.183 | 24.444 ± 6.552 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.001 | 0.022 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.372 ± 0.014 | - | 12.001 ± 0.006 | 1.601 ± 0.021 | 0.575 ± 0.116 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.027 ± 0.000 | - | 0.028 ± 0.001 | 0.039 ± 0.000 | 0.039 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.084 ± 0.000 | - | 0.090 ± 0.001 | 0.098 ± 0.000 | 0.092 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.450 ± 0.010 | - | unsupported | 0.292 ± 0.012 | 0.588 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.666 ± 0.015 | - | unsupported | 1.682 ± 0.110 | 2.178 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.699 ± 0.045 | - | unsupported | 4.366 ± 0.021 | 5.564 ± 0.099 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.075 ± 0.004 | - | unsupported | 1.016 ± 0.006 | 0.990 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.056 ± 0.001 | - | unsupported | 0.054 ± 0.002 | 0.061 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.041 ± 0.001 | - | unsupported | 0.044 ± 0.002 | 0.052 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.015 ± 0.000 | - | unsupported | 0.022 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.020 ± 0.000 | - | unsupported | 0.027 ± 0.001 | 0.034 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 8.639 ± 0.097 | - | unsupported | 7.999 ± 0.158 | 7.796 ± 0.329 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.413 ± 0.012 | - | unsupported | 0.365 ± 0.019 | 0.401 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.038 ± 0.000 | - | unsupported | 0.044 ± 0.001 | 0.052 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.495 ± 0.024 | - | unsupported | 0.450 ± 0.015 | 0.547 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 0.667 ± 0.031 | - | unsupported | 1.597 ± 0.009 | 1.083 ± 0.181 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 1.785 ± 0.049 | - | unsupported | 6.572 ± 0.138 | 2.879 ± 0.143 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.338 ± 0.092 | - | unsupported | 1.384 ± 2.953 | 1.530 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.076 ± 0.004 | - | unsupported | 0.078 ± 0.002 | 0.058 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.065 ± 0.003 | - | unsupported | 0.064 ± 0.016 | 0.068 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.010 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | unsupported | 0.017 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.023 ± 0.000 | - | unsupported | 0.020 ± 0.001 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | unsupported | 0.019 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.030 ± 0.003 | - | unsupported | 0.025 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 3.230 ± 0.115 | - | unsupported | 10.847 ± 0.226 | 12.390 ± 4.384 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | unsupported | 0.013 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.152 ± 0.004 | - | unsupported | 0.388 ± 0.027 | 0.249 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.016 ± 0.000 | - | unsupported | 0.028 ± 0.002 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.056 ± 0.003 | - | unsupported | 0.043 ± 0.000 | 0.050 ± 0.002 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.307 ± 0.015 | - | unsupported | 0.886 ± 0.007 | 2.986 ± 0.019 | 2.172 ± 0.134 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.084 ± 0.001 | - | unsupported | 0.052 ± 0.001 | 0.095 ± 0.002 | 0.114 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.167 ± 0.002 | - | unsupported | 0.078 ± 0.001 | 0.143 ± 0.003 | 0.187 ± 0.136 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.515 ± 0.016 | - | unsupported | 0.223 ± 0.010 | 0.378 ± 0.015 | 0.619 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.554 ± 0.025 | - | unsupported | 0.434 ± 0.015 | 0.932 ± 0.019 | 2.404 ± 0.226 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.159 ± 0.011 | - | unsupported | 0.052 ± 0.002 | 0.089 ± 0.004 | 0.102 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.285 ± 0.008 | - | unsupported | 0.066 ± 0.003 | 0.154 ± 0.007 | 0.159 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.225 ± 0.015 | - | unsupported | 0.131 ± 0.001 | 0.486 ± 0.025 | 0.649 ± 0.035 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.594 ± 0.017 | - | unsupported | 0.340 ± 0.008 | 0.706 ± 0.020 | 0.310 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.046 ± 0.000 | - | unsupported | 0.038 ± 0.001 | 0.154 ± 0.005 | 0.060 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.057 ± 0.001 | - | unsupported | 0.056 ± 0.001 | 0.186 ± 0.006 | 0.092 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.186 ± 0.002 | - | unsupported | 0.158 ± 0.010 | 0.321 ± 0.016 | 0.156 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.304 ± 0.013 | - | unsupported | 0.489 ± 0.012 | 0.786 ± 0.078 | 0.327 ± 0.012 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.082 ± 0.002 | - | unsupported | 0.063 ± 0.002 | 0.138 ± 0.017 | 0.060 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.103 ± 0.002 | - | unsupported | 0.086 ± 0.003 | 0.176 ± 0.004 | 0.091 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.119 ± 0.001 | - | unsupported | 0.201 ± 0.001 | 0.305 ± 0.020 | 0.160 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.315 ± 0.009 | - | 0.328 ± 0.011 | 0.183 ± 0.002 | 0.319 ± 0.011 | 0.171 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.032 ± 0.000 | - | 0.035 ± 0.001 | 0.063 ± 0.002 | 0.060 ± 0.001 | 0.044 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.039 ± 0.000 | - | 0.042 ± 0.000 | 0.077 ± 0.001 | 0.076 ± 0.004 | 0.063 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.111 ± 0.004 | - | 0.116 ± 0.002 | 0.110 ± 0.001 | 0.134 ± 0.010 | 0.099 ± 0.009 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.127 ± 0.008 | - | 12.006 ± 0.018 | 0.241 ± 0.009 | 0.404 ± 0.106 | 0.184 ± 0.009 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.057 ± 0.002 | - | 0.036 ± 0.000 | 0.106 ± 0.001 | 0.065 ± 0.001 | 0.048 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.040 ± 0.000 | - | 0.042 ± 0.000 | 0.121 ± 0.003 | 0.079 ± 0.001 | 0.068 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.049 ± 0.000 | - | 12.001 ± 0.012 | 0.159 ± 0.003 | 0.166 ± 0.004 | 0.108 ± 0.004 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 32.969 ± 1.323 | - | 32.660 ± 0.825 | 61.102 ± 5.403 | 58.583 ± 0.606 | 53.597 ± 6.867 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 17.319 ± 0.423 | - | 587.989 ± 353.743 | 23.796 ± 0.279 | 53.967 ± 2.007 | 36.051 ± 7.479 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 20.489 ± 0.231 | - | 19.014 ± 0.346 | 25.221 ± 0.309 | 23.843 ± 0.222 | 19.582 ± 14.072 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 14.939 ± 0.387 | - | 1817.993 ± 18.056 | 11.232 ± 0.784 | 14.179 ± 0.675 | 12.255 ± 2.416 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.831 ± 0.101 | - | 6.765 ± 0.042 | 7.180 ± 0.054 | 6.868 ± 0.077 | 6.392 ± 0.372 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 9.314 ± 0.101 | - | 3761.991 ± 2.999 | 10.699 ± 0.723 | 10.240 ± 0.440 | 9.667 ± 0.258 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.343 ± 0.109 | - | 8.442 ± 0.288 | 7.946 ± 0.066 | 8.393 ± 1.018 | 7.597 ± 0.083 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 12.398 ± 3.060 | - | 3139.006 ± 237.218 | 11.353 ± 0.200 | 12.951 ± 0.412 | 11.319 ± 0.505 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 18.516 ± 0.170 | - | 18.718 ± 0.099 | 17.176 ± 0.244 | 35.661 ± 0.936 | 11.305 ± 0.397 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 16.043 ± 1.129 | - | 5716.121 ± 406.113 | 15.116 ± 0.207 | 30.812 ± 8.337 | 10.379 ± 0.643 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.273 ± 0.002 | 0.276 ± 0.004 | unsupported | 0.546 ± 0.007 | 0.650 ± 0.024 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 5.093 ± 0.178 | 5.227 ± 0.086 | unsupported | 8.056 ± 0.218 | 8.175 ± 0.192 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.290 ± 0.003 | 207.936 ± 7.936 | unsupported | 0.491 ± 0.013 | 1.669 ± 0.064 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.456 ± 0.319 | 3065.624 ± 69.681 | unsupported | 5.611 ± 0.157 | 10.625 ± 0.347 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 29.077 ± 0.760 | - | 30.663 ± 0.631 | 43.011 ± 0.996 | 39.942 ± 0.348 | 25.983 ± 0.892 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 16.747 ± 0.191 | - | 1654.999 ± 107.999 | 21.991 ± 0.449 | 22.982 ± 1.077 | 15.907 ± 0.616 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 14.631 ± 0.187 | 14.288 ± 0.137 | 8.426 ± 0.167 | 50.651 ± 1.025 | 6.567 ± 0.213 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 8.785 ± 0.160 | 1103.986 ± 21.009 | 5.923 ± 0.213 | 48.596 ± 0.715 | 5.814 ± 0.171 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 21.098 ± 0.225 | - | 20.469 ± 0.223 | 34.345 ± 1.532 | 30.659 ± 1.375 | 24.232 ± 16.707 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 14.603 ± 0.204 | - | 1757.979 ± 4.287 | 16.261 ± 0.421 | 21.033 ± 1.581 | 32.663 ± 7.987 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.828 ± 0.068 | - | 0.765 ± 0.129 | 1.046 ± 0.013 | 20.735 ± 0.521 | 2.421 ± 0.023 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.688 ± 0.025 | - | 23.998 ± 0.007 | 1.073 ± 0.016 | 3.423 ± 0.149 | 2.682 ± 0.046 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 14.884 ± 0.144 | - | 14.849 ± 0.227 | 19.381 ± 0.533 | 17.828 ± 0.234 | 17.868 ± 1.410 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 16.002 ± 0.191 | - | 88.996 ± 3.001 | 16.969 ± 1.252 | 18.745 ± 1.257 | 18.445 ± 0.908 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 15.045 ± 0.206 | - | 14.624 ± 0.133 | 19.551 ± 0.295 | 17.787 ± 0.746 | 16.705 ± 0.453 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 15.860 ± 0.157 | - | 87.998 ± 1.016 | 17.310 ± 0.529 | 19.051 ± 1.384 | 18.117 ± 1.245 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 0.958 ± 0.024 | 0.871 ± 0.014 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 21.020 ± 0.799 | 18.704 ± 0.190 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 0.859 ± 0.021 | 1298.623 ± 128.187 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 12.779 ± 0.862 | 4825.996 ± 137.903 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 0.987 ± 0.013 | 0.944 ± 0.020 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 19.881 ± 0.117 | 19.023 ± 1.056 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 0.834 ± 0.030 | 1295.813 ± 111.438 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 13.285 ± 1.047 | 4814.120 ± 109.498 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 20.745 ± 0.575 | - | 19.202 ± 0.650 | 24.836 ± 0.184 | 23.731 ± 0.893 | 18.982 ± 1.501 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 15.295 ± 0.238 | - | 1838.998 ± 34.995 | 11.533 ± 0.257 | 15.244 ± 1.252 | 12.120 ± 2.173 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 47.147 ± 1.091 | 46.850 ± 0.498 | 56.279 ± 0.209 | 54.486 ± 0.440 | 52.397 ± 9.256 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 30.207 ± 0.794 | 9572.059 ± 1013.630 | 38.191 ± 1.118 | 46.971 ± 0.926 | 53.699 ± 8.685 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 30.244 ± 0.072 | - | 29.654 ± 0.746 | 30.377 ± 2.535 | 141.666 ± 6.220 | 27.318 ± 0.693 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 16.258 ± 0.254 | - | 1584.003 ± 78.003 | 26.493 ± 10.160 | 129.014 ± 15.791 | 18.963 ± 0.918 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.085 ± 0.001 | 0.085 ± 0.001 | unsupported | 0.160 ± 0.002 | 0.032 ± 0.003 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 1.376 ± 0.046 | 1.401 ± 0.050 | unsupported | 2.466 ± 0.052 | 0.361 ± 0.624 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.046 ± 0.017 | 24.000 ± 0.016 | unsupported | 0.076 ± 0.007 | 0.025 ± 0.004 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.665 ± 0.012 | 23.999 ± 0.001 | unsupported | 1.004 ± 0.028 | 0.681 ± 0.093 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.040 ± 0.001 | 0.042 ± 0.000 | unsupported | 0.105 ± 0.014 | 0.186 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 0.640 ± 0.004 | 0.661 ± 0.022 | unsupported | 1.366 ± 0.020 | 2.211 ± 0.076 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.024 ± 0.001 | 12.000 ± 0.000 | unsupported | 0.072 ± 0.047 | 0.127 ± 0.009 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.330 ± 0.012 | 11.999 ± 0.001 | unsupported | 0.596 ± 0.041 | 1.118 ± 0.250 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.114 ± 0.002 | 0.114 ± 0.002 | unsupported | 0.179 ± 0.003 | 0.156 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 1.846 ± 0.026 | 1.803 ± 0.057 | unsupported | 2.804 ± 0.066 | 1.708 ± 0.046 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.059 ± 0.000 | 12.000 ± 0.047 | unsupported | 0.085 ± 0.002 | 0.113 ± 0.008 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 0.868 ± 0.006 | 12.000 ± 0.004 | unsupported | 1.203 ± 0.044 | 0.893 ± 0.050 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.054 ± 0.002 | 0.079 ± 0.005 | unsupported | 0.149 ± 0.001 | 0.179 ± 0.011 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.670 ± 0.011 | 0.697 ± 0.055 | unsupported | 1.581 ± 0.168 | 1.783 ± 0.021 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.088 ± 0.003 | 70.544 ± 10.883 | unsupported | 0.067 ± 0.002 | 0.154 ± 0.011 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.393 ± 0.012 | 82.308 ± 86.505 | unsupported | 0.352 ± 0.002 | 1.077 ± 0.230 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.080 ± 0.007 | 0.091 ± 0.002 | unsupported | 0.109 ± 0.001 | 0.220 ± 0.006 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.040 ± 0.050 | 1.168 ± 0.091 | unsupported | 1.720 ± 0.030 | 2.732 ± 0.057 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.113 ± 0.005 | 95.140 ± 1.000 | unsupported | 0.056 ± 0.002 | 0.205 ± 0.012 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.555 ± 0.016 | 119.999 ± 4.375 | unsupported | 0.427 ± 0.004 | 1.516 ± 0.050 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.251 ± 0.006 | 0.276 ± 0.014 | unsupported | unsupported | 0.235 ± 0.006 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 3.637 ± 0.045 | 3.671 ± 0.127 | unsupported | unsupported | 3.696 ± 0.139 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.216 ± 0.006 | 119.703 ± 1.059 | unsupported | unsupported | 0.245 ± 0.015 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.603 ± 0.036 | 143.999 ± 0.375 | unsupported | unsupported | 1.968 ± 0.078 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.010 ± 0.000 | 0.015 ± 0.000 | unsupported | 0.011 ± 0.000 | 0.040 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.077 ± 0.005 | 0.084 ± 0.001 | unsupported | 0.068 ± 0.003 | 0.298 ± 0.004 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.013 ± 0.000 | 23.938 ± 0.219 | unsupported | 0.012 ± 0.000 | 0.037 ± 0.005 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.056 ± 0.002 | 23.999 ± 0.005 | unsupported | 0.044 ± 0.002 | 0.246 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.027 ± 0.006 | 0.045 ± 0.002 | unsupported | 0.108 ± 0.007 | 0.157 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.361 ± 0.014 | 0.372 ± 0.019 | unsupported | 1.625 ± 0.048 | 1.620 ± 0.044 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.045 ± 0.007 | 35.109 ± 0.323 | unsupported | 0.106 ± 0.003 | 0.129 ± 0.011 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.242 ± 0.006 | 47.999 ± 0.001 | unsupported | 0.280 ± 0.004 | 0.974 ± 0.082 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.028 ± 0.000 | 0.028 ± 0.003 | unsupported | 0.090 ± 0.004 | 0.161 ± 0.003 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.429 ± 0.005 | 0.516 ± 0.003 | unsupported | 1.358 ± 0.026 | 1.496 ± 0.038 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.017 ± 0.000 | 8.112 ± 10.552 | unsupported | 0.073 ± 0.003 | 0.105 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.223 ± 0.002 | 11.999 ± 0.001 | unsupported | 0.528 ± 0.014 | 0.747 ± 0.036 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.029 ± 0.000 | 0.030 ± 0.000 | unsupported | 0.089 ± 0.000 | 0.160 ± 0.002 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.450 ± 0.008 | 0.463 ± 0.005 | unsupported | 1.380 ± 0.020 | 1.514 ± 0.039 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.017 ± 0.000 | 12.000 ± 0.000 | unsupported | 0.075 ± 0.004 | 0.102 ± 0.004 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.231 ± 0.005 | 11.999 ± 0.001 | unsupported | 0.558 ± 0.037 | 0.792 ± 0.057 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.072 ± 0.001 | 0.091 ± 0.001 | unsupported | 0.111 ± 0.003 | 0.219 ± 0.006 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.031 ± 0.027 | 1.001 ± 0.059 | unsupported | 0.962 ± 0.001 | 2.582 ± 0.073 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.105 ± 0.005 | 95.593 ± 15.698 | unsupported | 0.055 ± 0.002 | 0.198 ± 0.013 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.591 ± 0.013 | 119.999 ± 0.002 | unsupported | 0.379 ± 0.003 | 1.460 ± 0.072 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.099 ± 0.000 | 0.099 ± 0.001 | unsupported | 0.166 ± 0.002 | 0.281 ± 0.016 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 1.590 ± 0.060 | 1.616 ± 0.020 | unsupported | 2.613 ± 0.039 | 3.681 ± 0.114 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.046 ± 0.001 | 12.000 ± 1.125 | unsupported | 0.137 ± 0.005 | 0.169 ± 0.008 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.693 ± 0.029 | 11.999 ± 0.002 | unsupported | 1.061 ± 0.074 | 1.841 ± 0.077 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.565 ± 0.008 | 1.194 ± 0.034 | unsupported | 0.261 ± 0.007 | 0.171 ± 0.001 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 12.127 ± 0.083 | 21.182 ± 0.518 | unsupported | 5.809 ± 0.186 | 1.848 ± 0.149 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.447 ± 0.012 | 107.265 ± 0.813 | unsupported | 0.260 ± 0.013 | 0.102 ± 0.012 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 8.198 ± 0.155 | 144.000 ± 1.250 | unsupported | 2.168 ± 0.059 | 0.789 ± 0.134 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 26.493 ± 2.689 | - | unsupported | 23.062 ± 0.952 | - | 23.060 ± 0.141 | 76.813 ± 0.665 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 21.720 ± 0.337 | - | unsupported | 22.049 ± 0.221 | - | 26.706 ± 1.946 | 33.450 ± 1.261 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 18.769 ± 0.978 | - | unsupported | 17.487 ± 0.781 | - | 16.273 ± 0.219 | 16.112 ± 0.147 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 14.902 ± 0.406 | - | unsupported | 15.101 ± 0.347 | - | 20.141 ± 0.688 | 15.241 ± 0.567 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 19.924 ± 0.266 | - | unsupported | 18.688 ± 0.594 | - | 19.608 ± 0.623 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 14.968 ± 0.397 | - | unsupported | 15.058 ± 3.205 | - | 20.164 ± 0.247 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 23.121 ± 0.952 | - | unsupported | 25.294 ± 1.269 | - | 24.889 ± 0.311 | 78.902 ± 1.462 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 21.457 ± 0.711 | - | unsupported | 22.019 ± 6.105 | - | 28.740 ± 1.404 | 32.624 ± 0.988 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 42.158 ± 0.623 | - | unsupported | 53.261 ± 4.131 | - | 40.709 ± 0.817 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 21.337 ± 0.141 | - | unsupported | 26.030 ± 0.852 | - | 22.606 ± 1.059 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 42.805 ± 0.672 | - | unsupported | 50.602 ± 0.756 | - | 42.793 ± 1.258 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 21.504 ± 0.331 | - | unsupported | 25.469 ± 0.763 | - | 22.043 ± 1.045 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 24.799 ± 0.964 | - | unsupported | 24.001 ± 1.936 | - | 23.544 ± 0.515 | 77.634 ± 0.357 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 21.349 ± 0.375 | - | unsupported | 21.956 ± 0.585 | - | 31.085 ± 1.842 | 32.505 ± 4.527 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 19.714 ± 0.399 | - | unsupported | 18.373 ± 0.519 | - | 15.368 ± 0.076 | 61.281 ± 0.566 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 14.865 ± 0.474 | - | unsupported | 15.014 ± 0.111 | - | 23.029 ± 1.992 | 24.769 ± 4.503 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.487 ± 0.013 | unsupported | unsupported | 0.869 ± 0.010 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 8.229 ± 0.124 | unsupported | unsupported | 12.947 ± 0.232 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.540 ± 0.006 | unsupported | unsupported | 0.622 ± 0.030 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 5.711 ± 0.235 | unsupported | unsupported | 6.920 ± 0.191 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 25.475 ± 1.229 | - | unsupported | 22.791 ± 1.119 | - | 23.491 ± 1.232 | 75.849 ± 0.187 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 21.404 ± 0.241 | - | unsupported | 21.730 ± 0.638 | - | 29.734 ± 1.106 | 31.718 ± 1.263 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 100.947 ± 6.085 | - | 0.002 ± 0.000 | 103.328 ± 8.392 | 100.937 ± 9.288 | 121.182 ± 5.180 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 41.259 ± 0.347 | - | 0.002 ± 0.000 | 34.990 ± 3.028 | 51.795 ± 6.169 | 123.682 ± 7.336 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 63.821 ± 3.317 | - | 65.677 ± 2.862 | 67.277 ± 0.592 | 67.990 ± 1.837 | 73.542 ± 2.966 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 22.523 ± 1.156 | - | 82.242 ± 2.457 | 23.564 ± 0.416 | 37.471 ± 2.347 | 77.906 ± 1.204 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 194.507 ± 10.016 | - | 194.007 ± 6.967 | 196.909 ± 7.115 | 192.314 ± 1.554 | 215.498 ± 5.449 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 64.364 ± 1.082 | - | 201.553 ± 12.220 | 68.197 ± 1.552 | 111.802 ± 6.452 | 227.716 ± 8.127 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 65.504 ± 3.633 | - | 66.539 ± 4.240 | 66.228 ± 6.515 | 67.087 ± 3.253 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 40.116 ± 0.184 | - | 69.731 ± 3.962 | 24.136 ± 1.154 | 36.692 ± 0.620 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 127.190 ± 5.852 | - | 0.002 ± 0.000 | 112.375 ± 1.484 | 112.220 ± 6.087 | 140.681 ± 11.382 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 38.515 ± 0.653 | - | 0.002 ± 0.000 | 39.177 ± 0.719 | 47.774 ± 6.478 | 148.856 ± 3.764 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 90.357 ± 1.761 | - | 0.002 ± 0.000 | 79.613 ± 1.638 | 73.961 ± 1.940 | 190.026 ± 13.112 | 121.054 ± 2.718 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 28.509 ± 1.145 | - | 0.002 ± 0.000 | 83.131 ± 3.764 | 38.601 ± 0.870 | 201.158 ± 4.423 | 47.776 ± 0.669 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 53.936 ± 2.538 | - | 54.385 ± 2.810 | 66.246 ± 1.206 | 55.398 ± 4.598 | 72.964 ± 3.408 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 21.575 ± 0.426 | - | 71.814 ± 0.199 | 22.560 ± 1.210 | 33.058 ± 1.495 | 74.891 ± 3.808 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 54.814 ± 2.091 | - | 55.272 ± 4.929 | 60.474 ± 0.725 | 55.583 ± 2.209 | 72.166 ± 1.403 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 20.306 ± 0.327 | - | 70.858 ± 3.563 | 21.192 ± 0.817 | 30.708 ± 1.877 | 76.995 ± 4.941 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.001 ± 0.000 | - | unsupported | 0.001 ± 0.001 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.001 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 7.029273 | 439329.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 5.627043 | 351690.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 5.945795 | 371612.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 5.602515 | 350157.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 3.918300 | 244893.75 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 1854.852128 | 115928258.00 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 11.897359 | 743584.94 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 12.513671 | 782104.44 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 4.809449 | 300590.56 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 383.985980 | 23999123.75 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 10.394213 | 649638.31 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 8.728333 | 545520.81 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 4.364239 | 272764.94 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 4.420973 | 276310.81 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 130.796004 | 8174750.25 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 128.892448 | 8055778.00 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 81.484882 | 5092805.12 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 83.635524 | 5227220.25 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 26.697351 | 1668584.44 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 7.850970 | 490685.62 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 4.639386 | 289961.62 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 3326.978700 | 207936168.75 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 169.997342 | 10624833.88 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 89.770561 | 5610660.06 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 55.300124 | 3456257.75 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 49049.976265 | 3065623516.56 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 15.335886 | 958492.88 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 13.930015 | 870625.94 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 336.317046 | 21019815.38 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 299.258600 | 18703662.50 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 13.750420 | 859401.25 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 20777.961276 | 1298622579.75 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 204.461354 | 12778834.62 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 77215.932318 | 4825995769.88 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 15.790940 | 986933.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 15.099426 | 943714.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 318.093877 | 19880867.31 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 304.363636 | 19022727.25 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 13.336097 | 833506.06 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 20733.004217 | 1295812763.56 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 212.563815 | 13285238.44 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 77025.926740 | 4814120421.25 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.076453 | 32444.58 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.215669 | 159619.83 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.430587 | 84852.92 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.411018 | 84547.16 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.887473 | 360934.12 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 19.730061 | 2466257.62 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.004897 | 1375612.12 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.211353 | 1401419.12 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 1.596883 | 24951.30 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.870949 | 76108.58 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.927323 | 45739.42 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1535.970680 | 23999541.88 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.449133 | 681141.62 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.031287 | 1003910.88 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.316598 | 664574.75 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.989513 | 23998689.12 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.922305 | 186286.02 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.726064 | 105094.75 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.580520 | 40320.62 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.667478 | 41679.34 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.685333 | 2210666.62 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.928554 | 1366069.25 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.116192 | 639524.00 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.291852 | 661481.50 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.133074 | 127079.28 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.606658 | 71979.03 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.505828 | 23528.56 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.977118 | 11999642.47 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.941251 | 1117656.38 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.766114 | 595764.25 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.636222 | 329527.75 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.992407 | 11999050.88 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.964126 | 155689.47 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 11.446852 | 178857.06 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.265854 | 113528.97 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.309619 | 114212.80 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.666465 | 1708308.12 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 22.428945 | 2803618.12 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 14.767456 | 1845932.00 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.420873 | 1802609.12 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.203088 | 112548.25 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.419061 | 84672.83 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.779063 | 59047.86 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.980592 | 11999696.75 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.146155 | 893269.38 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.622152 | 1202769.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.941343 | 867667.88 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.997465 | 11999683.12 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.469484 | 179210.69 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.565848 | 149466.38 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.481172 | 54393.31 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.060920 | 79076.88 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.261916 | 1782739.50 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.647864 | 1580983.00 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.357169 | 669646.12 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.578896 | 697362.00 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.866920 | 154170.62 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.295608 | 67118.88 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.663170 | 88487.03 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4514.795813 | 70543684.58 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.617482 | 1077185.25 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.815807 | 351975.88 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.144014 | 393001.75 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 658.461376 | 82307672.00 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.107368 | 220427.62 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.992148 | 109252.31 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.146558 | 80414.97 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.810219 | 90784.67 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 21.859550 | 2732443.75 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.760808 | 1720101.00 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 8.316689 | 1039586.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.343598 | 1167949.75 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.096839 | 204638.11 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.593359 | 56146.23 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.217236 | 112769.31 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6088.959794 | 95139996.78 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 12.126795 | 1515849.38 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.414384 | 426798.00 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.440472 | 555059.00 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 959.995217 | 119999402.12 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.067739 | 235433.42 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 16.063354 | 250989.91 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 17.675540 | 276180.31 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 29.569571 | 3696196.38 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 29.093478 | 3636684.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 29.365581 | 3670697.62 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.683697 | 245057.77 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 13.838811 | 216231.42 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7660.974935 | 119702733.36 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 15.743787 | 1967973.38 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 12.824210 | 1603026.25 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1151.993413 | 143999176.62 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.547250 | 39800.78 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.685725 | 10714.45 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.668159 | 10439.98 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.970467 | 15163.55 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.386982 | 298372.75 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.544826 | 68103.25 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.615841 | 76980.12 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.673306 | 84163.25 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.344970 | 36640.16 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.789268 | 12332.31 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.849177 | 13268.39 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1532.009339 | 23937645.92 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.965248 | 245656.00 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.355196 | 44399.50 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.447282 | 55910.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.990703 | 23998837.88 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.032768 | 156762.00 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.919711 | 108120.48 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.732845 | 27075.70 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.864900 | 44764.06 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.959489 | 1619936.12 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.003520 | 1625440.00 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.887223 | 360902.88 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.974382 | 371797.75 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.232912 | 128639.25 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.786084 | 106032.56 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.851690 | 44557.66 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2246.987459 | 35109179.05 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.790138 | 973767.25 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.241287 | 280160.88 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.937307 | 242163.38 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 383.993532 | 47999191.50 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.322726 | 161292.59 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.774937 | 90233.39 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.782748 | 27855.44 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.806173 | 28221.45 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.965688 | 1495711.00 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.866132 | 1358266.50 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.432851 | 429106.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.130904 | 516363.00 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.705322 | 104770.66 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.703171 | 73487.05 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.058386 | 16537.28 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 519.142199 | 8111596.86 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.978026 | 747253.25 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.221308 | 527663.50 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.783720 | 222965.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.995602 | 11999450.25 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.219373 | 159677.70 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.715199 | 89299.98 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.828697 | 28573.39 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.898089 | 29657.64 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.114589 | 1514323.62 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.041111 | 1380138.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.597043 | 449630.38 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.704462 | 463057.75 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.542772 | 102230.81 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.827495 | 75429.61 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.076774 | 16824.59 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.978054 | 11999657.09 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.333743 | 791717.88 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.465579 | 558197.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.844700 | 230587.50 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.992628 | 11999078.50 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.023213 | 219112.70 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.133688 | 111463.88 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.592657 | 71760.27 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.851580 | 91430.94 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.654957 | 2581869.62 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.698925 | 962365.62 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 8.248427 | 1031053.37 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 8.011867 | 1001483.38 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.669858 | 197966.53 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.512489 | 54882.64 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.715189 | 104924.83 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6117.969950 | 95593280.47 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.680284 | 1460035.50 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.035052 | 379381.50 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.725033 | 590629.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 959.995223 | 119999402.88 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 17.993950 | 281155.47 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.601681 | 165651.27 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.348406 | 99193.84 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.346633 | 99166.14 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 29.449863 | 3681232.88 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 20.900713 | 2612589.12 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 12.721668 | 1590208.50 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 12.928324 | 1616040.50 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.815674 | 168994.91 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.740607 | 136571.98 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.967371 | 46365.17 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.974586 | 11999602.91 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.731309 | 1841413.62 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.487493 | 1060936.62 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.542233 | 692779.12 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.992057 | 11999007.12 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.936656 | 170885.25 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 16.709951 | 261092.98 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 36.169482 | 565148.16 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 76.431119 | 1194236.23 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.785992 | 1848249.00 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 46.475729 | 5809466.12 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 97.012879 | 12126609.88 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 169.455940 | 21181992.50 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.542741 | 102230.33 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 16.615653 | 259619.58 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 28.636119 | 447439.36 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6864.946428 | 107264787.94 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.309637 | 788704.62 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 17.345756 | 2168219.50 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 65.583584 | 8197948.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1151.997499 | 143999687.38 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 13.902992 | 868937.00 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 7.792971 | 487060.69 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 207.158016 | 12947376.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 131.657471 | 8228591.94 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.957578 | 622348.62 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 8.640527 | 540032.94 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 110.726798 | 6920424.88 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 91.370474 | 5710654.62 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.019739 | 1233.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002343 | 146.44 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.021321 | 1332.56 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002273 | 142.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000130 | 8.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.018407 | 1150.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.008192 | 512.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000621 | 38.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.019089 | 1193.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.008222 | 513.88 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000231 | 14.44 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.019288 | 1205.50 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.006279 | 392.44 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000261 | 16.31 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.021722 | 1357.62 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.006570 | 410.62 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000091 | 5.69 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.017917 | 1119.81 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.004186 | 261.62 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.018287 | 1142.94 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.004046 | 252.88 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 119.2x faster than the slowest successful cell (`tenferro-trace`, 336.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `julia-base` is 112.3x faster than the slowest successful cell (`tenferro-trace`, 336.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 154.8x faster than the slowest successful cell (`tenferro-trace`, 336.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 136.0x faster than the slowest successful cell (`tenferro-trace`, 336.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 333.3x faster than the slowest successful cell (`tenferro-trace`, 5456.989 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `julia-base` is 318.9x faster than the slowest successful cell (`tenferro-trace`, 5456.989 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 403.6x faster than the slowest successful cell (`tenferro-trace`, 5456.989 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 571.1x faster than the slowest successful cell (`tenferro-trace`, 5456.989 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 20.299 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `jax-cpu` is 17.5x faster than the slowest successful cell (`tenferro-trace`, 59.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `julia-base` is 15.2x faster than the slowest successful cell (`tenferro-trace`, 59.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 44.9x faster than the slowest successful cell (`tenferro-trace`, 59.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 71.6x faster than the slowest successful cell (`tenferro-trace`, 527.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `julia-base` is 71.0x faster than the slowest successful cell (`tenferro-trace`, 527.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 95.1x faster than the slowest successful cell (`tenferro-trace`, 527.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 121.2x faster than the slowest successful cell (`tenferro-trace`, 527.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 198.3x faster than the slowest successful cell (`tenferro-trace`, 791.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `julia-base` is 229.0x faster than the slowest successful cell (`tenferro-trace`, 791.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 234.6x faster than the slowest successful cell (`tenferro-trace`, 791.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 248.8x faster than the slowest successful cell (`tenferro-trace`, 791.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/chain_log1p_exp_mul` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 12.6x faster than the slowest successful cell (`tenferro-trace`, 108.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 13.7x faster than the slowest successful cell (`julia-base`, 59.659 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`julia-base`, 40.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `jax-cpu` is 331.1x faster than the slowest successful cell (`tenferro-eager`, 115.928 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `tenferro-direct` is 473.4x faster than the slowest successful cell (`tenferro-eager`, 115.928 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 24.833 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 26.4x faster than the slowest successful cell (`pytorch-cpu`, 24.833 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 24.833 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 24.833 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 14.3x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 23.3x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 31.7x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 19.3x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 21.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 25.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 24.5x faster than the slowest successful cell (`pytorch-cpu`, 25.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 25.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 25.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 14.6x faster than the slowest successful cell (`tenferro-trace`, 24.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 24.0x faster than the slowest successful cell (`tenferro-trace`, 24.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 31.2x faster than the slowest successful cell (`tenferro-trace`, 24.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 13.6x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 34.5x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 22.3x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 65.4x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 21.0x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 26.7x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 75.2x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 27.1x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 13.2x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 26.2x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 22.8x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 55.0x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 22.6x faster than the slowest successful cell (`tenferro-trace`, 23.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 24.8x faster than the slowest successful cell (`tenferro-trace`, 23.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 80.6x faster than the slowest successful cell (`tenferro-trace`, 23.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 27.4x faster than the slowest successful cell (`tenferro-trace`, 23.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_squares_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 79.8x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`julia-base`, 64.236 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=1, shape=`1048576+1048576`): `jax-cpu` is 10.4x faster than the slowest successful cell (`julia-base`, 7.781 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=1, shape=`1048576+1048576`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`julia-base`, 7.781 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `jax-cpu` is 14.4x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 36.2x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 42.5x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=1, shape=`4194304 -> 2097152`): `jax-cpu` is 11.6x faster than the slowest successful cell (`julia-base`, 8.224 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=1, shape=`4194304 -> 2097152`): `pytorch-cpu` is 12.1x faster than the slowest successful cell (`julia-base`, 8.224 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `jax-cpu` is 21.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `pytorch-cpu` is 32.9x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-direct` is 29.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 11.6x faster than the slowest successful cell (`julia-base`, 8.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`julia-base`, 8.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 274.2x faster than the slowest successful cell (`tenferro-trace`, 47.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `julia-base` is 41.2x faster than the slowest successful cell (`tenferro-trace`, 47.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 125.2x faster than the slowest successful cell (`tenferro-trace`, 47.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `tenferro-direct` is 154.3x faster than the slowest successful cell (`tenferro-trace`, 47.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `jax-cpu` is 12.2x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 23.7x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 27.4x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 18.8x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 29.2x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 39.9x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 101.4x faster than the slowest successful cell (`tenferro-trace`, 36.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `julia-base` is 28.6x faster than the slowest successful cell (`tenferro-trace`, 36.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 76.7x faster than the slowest successful cell (`tenferro-trace`, 36.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `tenferro-direct` is 38.4x faster than the slowest successful cell (`tenferro-trace`, 36.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 5240.7x faster than the slowest successful cell (`julia-base`, 9.601 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`julia-base`, 9.457 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`julia-base`, 9.457 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 5219.3x faster than the slowest successful cell (`julia-base`, 9.457 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1024x8x8`): `jax-cpu` is 38.2x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1024x8x8`): `pytorch-cpu` is 40.7x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1024x8x8`): `tenferro-direct` is 134.3x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x128x128`): `jax-cpu` is 230.7x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x128x128`): `pytorch-cpu` is 141.6x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x128x128`): `tenferro-direct` is 347.9x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x128x128`): `jax-cpu` is 12.0x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x128x128`): `tenferro-direct` is 69.7x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 330.7x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x32x32`): `pytorch-cpu` is 220.5x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 963.8x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`1024x2x2`): `jax-cpu` is 16.1x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`1024x2x2`): `pytorch-cpu` is 12.9x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`1024x2x2`): `tenferro-direct` is 65.6x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`1024x4x4`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`8x32x32`): `tenferro-direct` is 39.0x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`1024x4x4`): `jax-cpu` is 12.0x faster than the slowest successful cell (`tenferro-trace`, 12.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`1024x4x4`): `tenferro-direct` is 17.6x faster than the slowest successful cell (`tenferro-trace`, 12.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 31.8x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`8x32x32`): `pytorch-cpu` is 14.7x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 53.0x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`1024x4x4`): `jax-cpu` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`1024x4x4`): `pytorch-cpu` is 14.2x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`1024x4x4`): `tenferro-direct` is 22.6x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 32.6x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`8x32x32`): `pytorch-cpu` is 44.2x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 79.0x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1024x8x8`): `jax-cpu` is 21.9x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1024x8x8`): `pytorch-cpu` is 35.9x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1024x8x8`): `tenferro-direct` is 42.1x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x128x128`): `jax-cpu` is 377.7x faster than the slowest successful cell (`tenferro-trace`, 71.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x128x128`): `pytorch-cpu` is 376.2x faster than the slowest successful cell (`tenferro-trace`, 71.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x128x128`): `tenferro-direct` is 452.6x faster than the slowest successful cell (`tenferro-trace`, 71.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x128x128`): `jax-cpu` is 15.7x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x128x128`): `pytorch-cpu` is 21.3x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x128x128`): `tenferro-direct` is 29.4x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 95.4x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x32x32`): `pytorch-cpu` is 199.1x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 401.5x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1024x4x4`): `jax-cpu` is 18.3x faster than the slowest successful cell (`tenferro-trace`, 12.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1024x4x4`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 12.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1024x4x4`): `tenferro-direct` is 54.8x faster than the slowest successful cell (`tenferro-trace`, 12.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1024x8x8`): `jax-cpu` is 10.8x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1024x8x8`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 71.2x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x32x32`): `pytorch-cpu` is 48.8x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 120.1x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `jax-cpu` is 84.8x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `pytorch-cpu` is 151.0x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `tenferro-direct` is 230.7x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `jax-cpu` is 51.3x faster than the slowest successful cell (`tenferro-trace`, 35.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 98.1x faster than the slowest successful cell (`tenferro-trace`, 35.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 125.0x faster than the slowest successful cell (`tenferro-trace`, 35.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `jax-cpu` is 631.1x faster than the slowest successful cell (`tenferro-trace`, 84.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `pytorch-cpu` is 475.0x faster than the slowest successful cell (`tenferro-trace`, 84.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `tenferro-direct` is 680.3x faster than the slowest successful cell (`tenferro-trace`, 84.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `jax-cpu` is 57.0x faster than the slowest successful cell (`tenferro-trace`, 35.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 66.3x faster than the slowest successful cell (`tenferro-trace`, 35.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `tenferro-direct` is 119.4x faster than the slowest successful cell (`tenferro-trace`, 35.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x32x32,rhs=1`): `jax-cpu` is 112.8x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x32x32,rhs=1`): `pytorch-cpu` is 209.3x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x32x32,rhs=1`): `tenferro-direct` is 496.7x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=4, shape=`1024x2x2`): `tenferro-direct` is 26.4x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=4, shape=`8x32x32`): `jax-cpu` is 16.5x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=4, shape=`8x32x32`): `tenferro-direct` is 22.5x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`1024x4x4`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 20.9x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 32.2x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x16x16,rhs=1`): `jax-cpu` is 29.7x faster than the slowest successful cell (`tenferro-trace`, 12.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x16x16,rhs=1`): `julia-base` is 65.2x faster than the slowest successful cell (`tenferro-trace`, 12.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x16x16,rhs=1`): `pytorch-cpu` is 49.9x faster than the slowest successful cell (`tenferro-trace`, 12.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x16x16,rhs=1`): `tenferro-direct` is 94.3x faster than the slowest successful cell (`tenferro-trace`, 12.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `jax-cpu` is 72.2x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `julia-base` is 111.1x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 75.6x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 246.1x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 10.9x faster than the slowest successful cell (`tenferro-trace`, 587.989 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `julia-base` is 16.3x faster than the slowest successful cell (`tenferro-trace`, 587.989 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 24.7x faster than the slowest successful cell (`tenferro-trace`, 587.989 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 34.0x faster than the slowest successful cell (`tenferro-trace`, 587.989 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 128.2x faster than the slowest successful cell (`tenferro-trace`, 1817.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `julia-base` is 148.3x faster than the slowest successful cell (`tenferro-trace`, 1817.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 161.9x faster than the slowest successful cell (`tenferro-trace`, 1817.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 121.7x faster than the slowest successful cell (`tenferro-trace`, 1817.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 367.4x faster than the slowest successful cell (`tenferro-trace`, 3761.991 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `julia-base` is 389.2x faster than the slowest successful cell (`tenferro-trace`, 3761.991 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 351.6x faster than the slowest successful cell (`tenferro-trace`, 3761.991 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 403.9x faster than the slowest successful cell (`tenferro-trace`, 3761.991 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 242.4x faster than the slowest successful cell (`tenferro-trace`, 3139.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `julia-base` is 277.3x faster than the slowest successful cell (`tenferro-trace`, 3139.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 276.5x faster than the slowest successful cell (`tenferro-trace`, 3139.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 253.2x faster than the slowest successful cell (`tenferro-trace`, 3139.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 185.5x faster than the slowest successful cell (`tenferro-trace`, 5716.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `julia-base` is 550.7x faster than the slowest successful cell (`tenferro-trace`, 5716.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 378.2x faster than the slowest successful cell (`tenferro-trace`, 5716.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 356.3x faster than the slowest successful cell (`tenferro-trace`, 5716.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `jax-cpu` is 124.6x faster than the slowest successful cell (`tenferro-eager`, 207.936 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `pytorch-cpu` is 423.8x faster than the slowest successful cell (`tenferro-eager`, 207.936 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 717.1x faster than the slowest successful cell (`tenferro-eager`, 207.936 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `jax-cpu` is 288.5x faster than the slowest successful cell (`tenferro-eager`, 3065.624 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 546.4x faster than the slowest successful cell (`tenferro-eager`, 3065.624 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `tenferro-direct` is 887.0x faster than the slowest successful cell (`tenferro-eager`, 3065.624 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 72.0x faster than the slowest successful cell (`tenferro-trace`, 1654.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `julia-base` is 104.0x faster than the slowest successful cell (`tenferro-trace`, 1654.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 75.3x faster than the slowest successful cell (`tenferro-trace`, 1654.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 98.8x faster than the slowest successful cell (`tenferro-trace`, 1654.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `jax-cpu` is 22.7x faster than the slowest successful cell (`tenferro-trace`, 1103.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `julia-base` is 189.9x faster than the slowest successful cell (`tenferro-trace`, 1103.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 186.4x faster than the slowest successful cell (`tenferro-trace`, 1103.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 125.7x faster than the slowest successful cell (`tenferro-trace`, 1103.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 83.6x faster than the slowest successful cell (`tenferro-trace`, 1757.979 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `julia-base` is 53.8x faster than the slowest successful cell (`tenferro-trace`, 1757.979 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 108.1x faster than the slowest successful cell (`tenferro-trace`, 1757.979 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 120.4x faster than the slowest successful cell (`tenferro-trace`, 1757.979 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `pytorch-cpu` is 19.8x faster than the slowest successful cell (`jax-cpu`, 20.735 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 25.1x faster than the slowest successful cell (`jax-cpu`, 20.735 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 27.1x faster than the slowest successful cell (`jax-cpu`, 20.735 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 22.4x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 34.9x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 1511.1x faster than the slowest successful cell (`tenferro-eager`, 1298.623 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr` (f64, threads=4, shape=`768x384`): `tenferro-direct` is 377.7x faster than the slowest successful cell (`tenferro-eager`, 4825.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr_deficient` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 1554.7x faster than the slowest successful cell (`tenferro-eager`, 1295.813 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr_deficient` (f64, threads=4, shape=`768x384`): `tenferro-direct` is 362.4x faster than the slowest successful cell (`tenferro-eager`, 4814.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 120.6x faster than the slowest successful cell (`tenferro-trace`, 1838.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `julia-base` is 151.7x faster than the slowest successful cell (`tenferro-trace`, 1838.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 159.5x faster than the slowest successful cell (`tenferro-trace`, 1838.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 120.2x faster than the slowest successful cell (`tenferro-trace`, 1838.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 203.8x faster than the slowest successful cell (`tenferro-trace`, 9572.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `julia-base` is 178.3x faster than the slowest successful cell (`tenferro-trace`, 9572.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 250.6x faster than the slowest successful cell (`tenferro-trace`, 9572.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 316.9x faster than the slowest successful cell (`tenferro-trace`, 9572.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-trace`, 1584.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `julia-base` is 83.5x faster than the slowest successful cell (`tenferro-trace`, 1584.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `pytorch-cpu` is 59.8x faster than the slowest successful cell (`tenferro-trace`, 1584.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `tenferro-direct` is 97.4x faster than the slowest successful cell (`tenferro-trace`, 1584.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 961.9x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 315.3x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 524.7x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 35.2x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 23.9x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 36.1x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 94.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 166.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 510.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 20.1x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 36.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 106.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 141.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 203.2x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 13.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 457.6x faster than the slowest successful cell (`tenferro-eager`, 70.544 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1051.0x faster than the slowest successful cell (`tenferro-eager`, 70.544 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 797.2x faster than the slowest successful cell (`tenferro-eager`, 70.544 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 76.4x faster than the slowest successful cell (`tenferro-eager`, 82.308 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 233.8x faster than the slowest successful cell (`tenferro-eager`, 82.308 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 209.4x faster than the slowest successful cell (`tenferro-eager`, 82.308 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 464.9x faster than the slowest successful cell (`tenferro-eager`, 95.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1694.5x faster than the slowest successful cell (`tenferro-eager`, 95.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 843.7x faster than the slowest successful cell (`tenferro-eager`, 95.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 79.2x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 281.2x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 216.2x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 488.5x faster than the slowest successful cell (`tenferro-eager`, 119.703 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 553.6x faster than the slowest successful cell (`tenferro-eager`, 119.703 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 73.2x faster than the slowest successful cell (`tenferro-eager`, 143.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 89.8x faster than the slowest successful cell (`tenferro-eager`, 143.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 653.3x faster than the slowest successful cell (`tenferro-eager`, 23.938 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1941.1x faster than the slowest successful cell (`tenferro-eager`, 23.938 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 1804.2x faster than the slowest successful cell (`tenferro-eager`, 23.938 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 97.7x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 540.5x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 429.2x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 272.9x faster than the slowest successful cell (`tenferro-eager`, 35.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 331.1x faster than the slowest successful cell (`tenferro-eager`, 35.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 787.9x faster than the slowest successful cell (`tenferro-eager`, 35.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 49.3x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 171.3x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 198.2x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 77.4x faster than the slowest successful cell (`tenferro-eager`, 8.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 110.4x faster than the slowest successful cell (`tenferro-eager`, 8.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 490.5x faster than the slowest successful cell (`tenferro-eager`, 8.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 16.1x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 22.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 53.8x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 117.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 159.1x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 713.2x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 15.2x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 21.5x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 52.0x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 482.9x faster than the slowest successful cell (`tenferro-eager`, 95.593 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1741.8x faster than the slowest successful cell (`tenferro-eager`, 95.593 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 911.1x faster than the slowest successful cell (`tenferro-eager`, 95.593 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 82.2x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 316.3x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 203.2x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 71.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 87.9x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 258.8x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 11.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 17.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 11.5x faster than the slowest successful cell (`tenferro-eager`, 21.182 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 1049.2x faster than the slowest successful cell (`tenferro-eager`, 107.265 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 413.2x faster than the slowest successful cell (`tenferro-eager`, 107.265 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 239.7x faster than the slowest successful cell (`tenferro-eager`, 107.265 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 182.6x faster than the slowest successful cell (`tenferro-eager`, 144.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 66.4x faster than the slowest successful cell (`tenferro-eager`, 144.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 17.6x faster than the slowest successful cell (`tenferro-eager`, 144.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 71199.9x faster than the slowest successful cell (`julia-base`, 121.182 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 71782.9x faster than the slowest successful cell (`julia-base`, 123.682 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 80712.0x faster than the slowest successful cell (`julia-base`, 140.681 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 85944.6x faster than the slowest successful cell (`julia-base`, 148.856 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 115728.3x faster than the slowest successful cell (`julia-base`, 190.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 125566.5x faster than the slowest successful cell (`julia-base`, 201.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 143.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 30.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 86.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 84.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 186.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 163.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9153.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 91.5x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9254.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 23.1x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 157717.7 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1577.2x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 155795.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 389.5x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 154007.7 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1540.1x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 154985.8 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 387.5x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 81740.4 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 817.4x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 83781.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 209.5x faster than the bound.

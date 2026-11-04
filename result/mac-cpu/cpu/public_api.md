# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261104_180459/run.yaml`
- Timestamp: `2026-11-04T18:04:59.407066+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261104_180459`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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

- CSV: `data/results/mac-cpu/cpu/public_api/20261104_180459/cpu_public_api_t1_20261104_180459.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261104_180459/cpu_public_api_t4_20261104_180459.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.220 ± 0.191 | - | 3.557 ± 0.243 | 3.437 ± 0.021 | 1.331 ± 0.035 | 4.687 ± 0.034 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.582 ± 0.045 | - | 3.754 ± 0.058 | 3.805 ± 0.113 | 2.575 ± 0.526 | 64.290 ± 1.302 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 14.034 ± 0.695 | - | 10.732 ± 2.394 | 11.052 ± 2.462 | 11.191 ± 2.583 | 9.354 ± 0.707 | 11.234 ± 0.650 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 12.213 ± 2.605 | - | 10.693 ± 0.318 | 10.293 ± 1.373 | 10.864 ± 0.848 | 11.669 ± 0.707 | 10.891 ± 5.717 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 42.958 ± 2.823 | - | 41.113 ± 0.968 | 17.288 ± 1.094 | 36.559 ± 5.987 | 42.911 ± 2.370 | 47.479 ± 8.096 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 14.625 ± 0.348 | - | 14.611 ± 0.043 | 9.205 ± 0.266 | 16.728 ± 0.661 | 42.967 ± 1.884 | 21.723 ± 5.489 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 8.762 ± 0.482 | - | 10.383 ± 1.990 | 8.837 ± 1.206 | 44.588 ± 2.065 | 57.753 ± 3.198 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 10.077 ± 1.960 | - | 8.897 ± 2.125 | 8.764 ± 0.998 | 16.529 ± 0.515 | 28.923 ± 7.107 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 11.611 ± 1.311 | - | unsupported | 9.082 ± 1.645 | 45.010 ± 2.282 | 55.774 ± 4.145 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 12.299 ± 2.411 | - | unsupported | 9.023 ± 1.705 | 17.641 ± 0.894 | 32.176 ± 7.734 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 11.421 ± 0.066 | - | 11.116 ± 0.486 | 10.809 ± 0.559 | 18.848 ± 1.313 | 7.897 ± 0.079 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 12.086 ± 0.715 | - | 12.236 ± 2.293 | 9.941 ± 0.028 | 19.150 ± 0.926 | 434.973 ± 15.650 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 31.641 ± 0.873 | - | 31.923 ± 0.462 | 32.388 ± 1.368 | 40.784 ± 4.519 | 43.115 ± 2.176 | 42.499 ± 2.559 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 12.500 ± 0.232 | - | 12.379 ± 0.030 | 12.631 ± 0.115 | 22.460 ± 0.287 | 42.522 ± 1.330 | 16.109 ± 0.831 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 41.821 ± 0.688 | - | 41.543 ± 2.185 | 42.111 ± 1.269 | 82.970 ± 9.048 | 65.376 ± 9.076 | 73.359 ± 4.562 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 17.754 ± 3.174 | - | 21.401 ± 0.618 | 15.425 ± 0.146 | 31.689 ± 0.853 | 65.601 ± 2.828 | 26.059 ± 1.049 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 9.824 ± 0.491 | - | 10.395 ± 1.112 | 7.099 ± 0.141 | 9.071 ± 0.697 | 7.835 ± 0.070 | 7.600 ± 1.947 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 7.207 ± 0.533 | - | 7.294 ± 0.254 | 7.161 ± 0.893 | 8.137 ± 1.211 | 8.367 ± 1.656 | 8.833 ± 0.669 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.461 ± 0.125 | - | 6.846 ± 0.347 | 5.821 ± 0.400 | 4.194 ± 0.086 | 8.398 ± 0.147 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.155 ± 0.079 | - | 4.126 ± 0.229 | 2.892 ± 0.005 | 2.820 ± 0.079 | 12.098 ± 3.093 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.271 ± 0.126 | - | 6.936 ± 0.051 | 6.253 ± 0.070 | 5.889 ± 0.628 | 8.438 ± 0.095 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 7.942 ± 0.281 | - | 7.833 ± 0.562 | 7.020 ± 0.155 | 5.856 ± 0.040 | 166.284 ± 26.769 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.102 ± 0.182 | - | 4.372 ± 0.554 | 4.427 ± 0.050 | 2.016 ± 0.013 | 5.312 ± 0.056 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 5.600 ± 1.345 | - | 5.631 ± 0.084 | 4.699 ± 0.095 | 2.650 ± 0.226 | 158.958 ± 5.644 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 7.764 ± 0.454 | - | 8.299 ± 0.432 | 6.345 ± 0.723 | 6.425 ± 0.650 | 6.485 ± 0.123 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 7.810 ± 0.188 | - | 7.218 ± 0.369 | 6.311 ± 0.055 | 6.516 ± 0.362 | 578.255 ± 18.775 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 10.846 ± 1.391 | - | 10.972 ± 1.149 | 9.250 ± 1.742 | 45.296 ± 2.591 | 54.742 ± 2.001 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 11.058 ± 0.993 | - | 10.147 ± 0.730 | 8.719 ± 1.364 | 16.588 ± 0.540 | 27.194 ± 4.229 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 11.809 ± 1.777 | - | unsupported | 7.902 ± 0.842 | 46.413 ± 5.786 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 8.029 ± 0.327 | - | unsupported | 8.192 ± 2.095 | 17.050 ± 0.526 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 10.738 ± 0.799 | - | 12.765 ± 1.755 | 10.914 ± 2.269 | 13.040 ± 1.132 | 12.100 ± 6.009 | 12.474 ± 1.953 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 10.466 ± 0.415 | - | 10.347 ± 1.042 | 10.545 ± 0.313 | 11.547 ± 0.470 | 10.789 ± 0.096 | 9.813 ± 0.234 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 17.701 ± 1.548 | - | 14.341 ± 0.375 | 17.839 ± 3.913 | 24.177 ± 4.927 | 21.006 ± 5.586 | 18.223 ± 3.865 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 16.540 ± 2.540 | - | 16.201 ± 0.625 | 17.980 ± 3.506 | 23.784 ± 0.899 | 18.357 ± 4.176 | 15.712 ± 2.424 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 23.486 ± 1.914 | - | 31.002 ± 1.238 | 34.934 ± 1.037 | 26.527 ± 1.475 | 45.708 ± 1.962 | 50.788 ± 3.234 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 13.195 ± 0.009 | - | 14.809 ± 1.108 | 17.254 ± 0.675 | 13.738 ± 0.036 | 45.125 ± 2.037 | 22.360 ± 4.072 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 6.677 ± 0.564 | - | 4.859 ± 0.133 | 6.707 ± 1.336 | 6.142 ± 0.207 | 6.139 ± 1.080 | 7.529 ± 1.644 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 4.962 ± 0.284 | - | 5.694 ± 0.230 | 5.631 ± 0.814 | 7.022 ± 0.719 | 8.097 ± 3.587 | 9.417 ± 5.835 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 12.777 ± 1.153 | - | 9.978 ± 0.074 | 13.939 ± 1.328 | 13.318 ± 2.182 | 14.533 ± 0.991 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 12.038 ± 1.620 | - | 10.165 ± 0.046 | 10.366 ± 1.440 | 14.403 ± 0.673 | 11.705 ± 1.108 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.422 ± 1.049 | - | 28.338 ± 3.973 | 30.023 ± 1.669 | 28.000 ± 1.003 | 33.552 ± 2.785 | 35.137 ± 1.918 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 15.050 ± 0.390 | - | 14.758 ± 0.051 | 12.165 ± 0.125 | 13.517 ± 0.154 | 35.300 ± 3.428 | 15.216 ± 2.480 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 19.794 ± 3.295 | - | 16.808 ± 0.245 | 14.026 ± 3.553 | 24.069 ± 5.403 | 17.841 ± 1.263 | 14.675 ± 3.925 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 14.237 ± 1.349 | - | 14.493 ± 0.409 | 14.875 ± 2.328 | 25.122 ± 0.808 | 21.173 ± 2.448 | 17.937 ± 0.814 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 15.477 ± 0.840 | - | 16.406 ± 1.335 | 26.794 ± 0.748 | 15.334 ± 0.029 | 33.309 ± 2.284 | 32.040 ± 0.923 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 7.937 ± 1.952 | - | 8.270 ± 0.313 | 10.547 ± 0.224 | 8.640 ± 0.322 | 30.953 ± 1.069 | 13.375 ± 2.066 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 23.990 ± 0.968 | - | 23.961 ± 1.361 | 25.212 ± 0.563 | 15.070 ± 0.072 | 23.517 ± 1.805 | 26.000 ± 0.685 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 11.123 ± 2.134 | - | 10.468 ± 0.194 | 10.769 ± 0.307 | 8.202 ± 0.224 | 23.435 ± 1.131 | 9.188 ± 2.484 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 27.866 ± 0.881 | - | 27.463 ± 0.725 | 38.679 ± 3.845 | 26.239 ± 0.777 | 39.580 ± 1.682 | 40.589 ± 11.581 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 11.270 ± 0.612 | - | 14.660 ± 0.476 | 14.314 ± 3.120 | 12.857 ± 0.327 | 39.650 ± 2.656 | 22.048 ± 4.899 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.981 ± 0.738 | - | 14.302 ± 0.683 | 19.989 ± 0.115 | 16.166 ± 0.202 | 22.633 ± 0.849 | 22.404 ± 2.165 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 7.511 ± 1.555 | - | 7.650 ± 0.113 | 9.157 ± 0.228 | 8.565 ± 0.059 | 21.211 ± 1.290 | 8.758 ± 1.990 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.277 ± 0.017 | 0.316 ± 0.006 | unsupported | unsupported | 0.250 ± 0.019 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.315 ± 0.016 | 0.343 ± 0.019 | unsupported | unsupported | 0.184 ± 0.017 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 16.799 ± 3.428 | - | 14.054 ± 0.342 | 17.972 ± 0.869 | 19.007 ± 5.389 | 14.372 ± 2.774 | 14.396 ± 0.277 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 16.126 ± 1.802 | - | 14.556 ± 0.841 | 16.489 ± 0.744 | 25.145 ± 0.734 | 14.295 ± 2.484 | 16.584 ± 2.635 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 14.219 ± 5.056 | - | 17.806 ± 3.998 | 18.120 ± 4.234 | 21.054 ± 5.764 | 14.466 ± 2.475 | 17.722 ± 4.206 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 16.272 ± 1.942 | - | 16.372 ± 0.487 | 16.640 ± 0.844 | 25.998 ± 0.995 | 18.117 ± 0.492 | 16.810 ± 0.820 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 18.246 ± 2.233 | - | 20.487 ± 1.725 | 15.147 ± 5.903 | 18.918 ± 3.199 | 20.380 ± 2.032 | 18.224 ± 2.765 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 16.052 ± 3.054 | - | 14.274 ± 1.822 | 17.159 ± 2.852 | 24.295 ± 0.840 | 17.802 ± 5.383 | 15.282 ± 0.277 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 12.490 ± 1.301 | - | 12.386 ± 1.586 | 9.248 ± 0.198 | 12.861 ± 0.660 | 10.823 ± 2.695 | 11.749 ± 1.149 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 10.490 ± 0.525 | - | 9.267 ± 1.393 | 9.137 ± 0.067 | 11.238 ± 0.598 | 11.181 ± 1.423 | 10.537 ± 0.684 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 33.635 ± 1.608 | - | 32.712 ± 0.971 | 62.727 ± 2.982 | 41.001 ± 0.993 | 72.101 ± 9.201 | 76.246 ± 10.453 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 12.908 ± 0.075 | - | 16.701 ± 0.819 | 24.350 ± 0.422 | 17.415 ± 0.615 | 75.974 ± 14.138 | 27.988 ± 0.525 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 7.596 ± 0.256 | - | 6.984 ± 0.558 | 11.250 ± 0.748 | 20.340 ± 1.777 | 4.620 ± 0.170 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.554 ± 0.349 | - | 5.253 ± 0.409 | 7.078 ± 0.087 | 11.301 ± 1.021 | 4.544 ± 0.013 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 0.957 ± 0.187 | - | 1.008 ± 0.025 | 15.388 ± 3.632 | 1.801 ± 0.012 | 0.731 ± 0.040 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 1.105 ± 0.608 | - | 1.330 ± 0.139 | 10.947 ± 0.763 | 1.224 ± 0.040 | 0.555 ± 0.025 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.960 ± 0.178 | - | 0.963 ± 0.181 | 1.448 ± 0.079 | 0.736 ± 0.009 | 0.869 ± 0.109 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 2.493 ± 0.525 | - | 2.416 ± 1.115 | 0.970 ± 0.032 | 0.878 ± 0.053 | 0.683 ± 0.019 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 7.263 ± 0.732 | - | 7.711 ± 0.262 | 10.628 ± 0.020 | 20.316 ± 1.854 | 6.004 ± 0.265 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 7.704 ± 1.784 | - | 9.365 ± 2.003 | 7.040 ± 0.104 | 10.471 ± 0.036 | 4.572 ± 1.513 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 0.990 ± 0.082 | - | 0.998 ± 0.101 | 12.177 ± 0.619 | 1.797 ± 0.010 | 0.933 ± 0.200 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 1.093 ± 0.199 | - | 1.322 ± 0.438 | 10.428 ± 1.006 | 1.154 ± 0.015 | 0.633 ± 0.029 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.412 ± 0.191 | - | 3.612 ± 0.142 | 5.544 ± 0.074 | 2.612 ± 0.013 | 2.694 ± 0.082 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 8.672 ± 3.387 | - | 8.964 ± 2.087 | 3.775 ± 0.189 | 3.185 ± 0.263 | 2.648 ± 0.039 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 5.467 ± 0.041 | - | 6.478 ± 1.221 | 5.411 ± 0.041 | 38.353 ± 2.204 | 6.236 ± 0.240 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 5.096 ± 0.446 | - | 6.845 ± 6.009 | 5.949 ± 0.302 | 20.106 ± 0.797 | 5.916 ± 0.778 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.935 ± 0.186 | - | 1.033 ± 0.176 | 4.369 ± 1.122 | 1.828 ± 0.018 | 0.754 ± 0.115 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 1.125 ± 0.333 | - | 1.643 ± 2.357 | 2.705 ± 0.176 | 1.415 ± 0.039 | 0.548 ± 0.037 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.992 ± 0.149 | - | 0.965 ± 0.165 | 0.749 ± 0.010 | 1.014 ± 0.011 | 0.769 ± 0.082 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 1.738 ± 0.487 | - | 2.049 ± 0.160 | 0.763 ± 0.026 | 0.739 ± 0.037 | 0.675 ± 0.078 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 5.368 ± 0.215 | - | 4.727 ± 0.121 | 6.141 ± 0.375 | 28.602 ± 1.988 | 4.588 ± 0.144 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 4.601 ± 0.657 | - | 5.611 ± 0.391 | 4.771 ± 1.157 | 15.629 ± 0.960 | 5.096 ± 0.465 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.841 ± 0.066 | - | 0.895 ± 0.242 | 3.440 ± 0.040 | 2.256 ± 0.207 | 0.708 ± 0.071 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 1.170 ± 0.214 | - | 1.145 ± 0.394 | 2.860 ± 0.786 | 1.202 ± 0.035 | 0.692 ± 0.028 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.896 ± 0.026 | - | 0.907 ± 0.121 | 0.551 ± 0.012 | 0.988 ± 0.009 | 0.773 ± 0.083 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 4.189 ± 0.958 | - | 3.766 ± 1.159 | 0.823 ± 0.127 | 0.720 ± 0.026 | 0.657 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.921 ± 0.051 | 0.760 ± 0.134 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.798 ± 0.150 | 0.829 ± 0.072 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 26.438 ± 1.319 | - | 26.267 ± 0.890 | 29.116 ± 0.967 | 32.636 ± 1.374 | 43.978 ± 20.639 | 31.242 ± 3.965 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 11.018 ± 3.034 | - | 10.804 ± 0.288 | 11.783 ± 0.445 | 20.527 ± 0.981 | 24.417 ± 0.908 | 18.433 ± 7.975 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 16.695 ± 0.051 | - | 17.285 ± 0.590 | 17.026 ± 1.119 | 32.980 ± 2.084 | 37.875 ± 1.408 | 44.422 ± 9.664 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 10.507 ± 0.737 | - | 11.058 ± 0.874 | 10.488 ± 0.460 | 21.523 ± 1.986 | 41.471 ± 2.374 | 19.306 ± 1.986 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 25.232 ± 2.721 | - | 25.371 ± 5.489 | 17.641 ± 2.828 | 27.855 ± 4.341 | 19.806 ± 3.850 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 21.520 ± 4.068 | - | 22.945 ± 3.753 | 16.223 ± 1.131 | 24.060 ± 5.566 | 25.157 ± 9.677 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 12.438 ± 0.837 | - | 10.656 ± 0.658 | 9.415 ± 2.334 | 13.743 ± 0.871 | 12.081 ± 0.837 | 16.804 ± 1.036 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 10.445 ± 0.295 | - | 10.390 ± 0.504 | 9.592 ± 1.344 | 11.434 ± 0.725 | 9.714 ± 1.907 | 11.124 ± 11.129 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 22.081 ± 1.080 | - | 21.912 ± 1.025 | 26.243 ± 0.743 | 26.060 ± 0.919 | 28.875 ± 0.855 | 29.322 ± 3.415 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 9.721 ± 1.790 | - | 10.136 ± 1.702 | 11.204 ± 1.345 | 12.492 ± 0.269 | 31.138 ± 2.607 | 11.688 ± 2.589 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 12.264 ± 0.593 | - | 12.715 ± 1.668 | 12.321 ± 0.483 | 20.846 ± 1.562 | 23.530 ± 2.183 | 24.183 ± 1.339 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 10.445 ± 0.827 | - | 10.827 ± 0.595 | 9.263 ± 0.801 | 13.114 ± 0.340 | 27.369 ± 1.140 | 12.044 ± 1.066 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 14.302 ± 4.344 | - | 16.903 ± 3.993 | 17.792 ± 0.827 | 19.558 ± 5.824 | 23.711 ± 2.390 | 21.605 ± 0.625 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 16.407 ± 0.292 | - | 16.905 ± 0.456 | 16.598 ± 0.824 | 24.377 ± 1.000 | 19.300 ± 14.071 | 16.709 ± 0.592 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 68.178 ± 0.952 | - | 61.733 ± 1.578 | 61.051 ± 2.298 | 12.292 ± 0.157 | 40.936 ± 0.918 | 42.945 ± 0.941 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 31.904 ± 1.166 | - | 31.960 ± 0.842 | 23.624 ± 1.490 | 6.748 ± 0.125 | 40.448 ± 2.033 | 19.086 ± 4.872 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 21.933 ± 6.211 | 24.003 ± 6.035 | unsupported | 21.533 ± 0.523 | 30.464 ± 6.727 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 23.451 ± 3.415 | 23.881 ± 1.791 | unsupported | 17.022 ± 2.243 | 26.311 ± 3.992 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.574 ± 0.166 | - | 0.534 ± 0.013 | 0.561 ± 0.017 | 0.660 ± 0.089 | 0.644 ± 0.014 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.825 ± 0.056 | - | 0.712 ± 0.116 | 0.694 ± 0.037 | 0.670 ± 0.015 | 0.667 ± 0.015 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.818 ± 0.119 | - | 0.690 ± 0.167 | 0.553 ± 0.021 | 0.580 ± 0.009 | 0.707 ± 0.100 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.744 ± 0.036 | - | 0.642 ± 0.034 | 0.698 ± 0.018 | 0.656 ± 0.036 | 0.765 ± 0.173 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.866 ± 0.083 | - | unsupported | 0.880 ± 0.041 | 1.240 ± 0.088 | 0.919 ± 0.079 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.930 ± 0.087 | - | unsupported | 1.080 ± 0.024 | 0.813 ± 0.038 | 0.948 ± 0.039 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.305 ± 0.000 | - | 0.362 ± 0.014 | 0.290 ± 0.020 | 0.298 ± 0.006 | 0.500 ± 0.029 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.269 ± 0.037 | - | 0.351 ± 0.051 | 0.369 ± 0.064 | 0.224 ± 0.035 | 1.269 ± 0.789 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.680 ± 0.101 | - | 0.560 ± 0.046 | 0.837 ± 0.020 | 0.787 ± 0.056 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.690 ± 0.074 | - | 0.679 ± 0.018 | 1.101 ± 0.058 | 0.788 ± 0.033 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.522 ± 0.003 | - | 0.519 ± 0.004 | 0.654 ± 0.141 | 0.564 ± 0.010 | 1.178 ± 0.269 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.631 ± 0.063 | - | 0.680 ± 0.021 | 0.727 ± 0.074 | 0.793 ± 0.123 | 1.341 ± 0.670 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.567 ± 0.021 | - | 0.615 ± 0.009 | 0.383 ± 0.012 | 0.344 ± 0.004 | 0.319 ± 0.017 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.762 ± 0.036 | - | 0.786 ± 0.056 | 0.522 ± 0.082 | 0.385 ± 0.020 | 0.932 ± 1.332 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.315 ± 0.129 | - | 0.002 ± 0.000 | 1.165 ± 0.031 | 1.046 ± 0.017 | 1.409 ± 0.477 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.228 ± 0.089 | - | 0.002 ± 0.000 | 1.109 ± 0.028 | 1.179 ± 0.034 | 1.452 ± 0.425 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.059 ± 0.000 | - | 0.063 ± 0.000 | 0.098 ± 0.001 | 0.087 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.096 ± 0.003 | - | 0.101 ± 0.000 | 0.142 ± 0.000 | 0.114 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.187 ± 0.002 | - | 0.206 ± 0.002 | 0.233 ± 0.000 | 0.203 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.024 ± 0.002 | - | 0.037 ± 0.006 | 0.034 ± 0.000 | 0.091 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.040 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.044 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.206 ± 0.023 | - | 0.238 ± 0.008 | 0.219 ± 0.000 | 0.228 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.010 ± 0.000 | - | 0.016 ± 0.001 | 0.014 ± 0.000 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.056 ± 0.000 | - | 0.060 ± 0.000 | 0.132 ± 0.009 | 0.099 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.089 ± 0.000 | - | 0.095 ± 0.000 | 0.172 ± 0.002 | 0.163 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.189 ± 0.004 | - | 0.205 ± 0.014 | 0.246 ± 0.003 | 0.269 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.026 ± 0.001 | - | 0.039 ± 0.004 | 0.085 ± 0.004 | 0.060 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.058 ± 0.012 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.059 ± 0.013 | 0.070 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.001 | 0.045 ± 0.004 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.048 ± 0.007 | 0.014 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.062 ± 0.008 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.043 ± 0.011 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.035 ± 0.016 | 0.012 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.051 ± 0.013 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.056 ± 0.022 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.045 ± 0.010 | 0.052 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.174 ± 0.001 | - | 0.234 ± 0.010 | 0.207 ± 0.003 | 0.175 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.043 ± 0.028 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.011 ± 0.001 | - | 0.018 ± 0.001 | 0.067 ± 0.007 | 0.063 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.044 ± 0.005 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.046 ± 0.011 | 0.061 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.243 ± 0.003 | - | 0.256 ± 0.002 | 0.393 ± 0.007 | 0.725 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.375 ± 0.080 | - | 1.383 ± 0.014 | 1.549 ± 0.010 | 1.871 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 5.621 ± 0.023 | - | 5.661 ± 0.137 | 6.348 ± 0.180 | 6.402 ± 0.123 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 5.765 ± 0.348 | - | 4.097 ± 0.029 | 2.896 ± 0.037 | 3.451 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.090 ± 0.001 | - | 0.094 ± 0.001 | 0.087 ± 0.002 | 0.170 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.018 ± 0.000 | - | 0.020 ± 0.001 | 0.021 ± 0.000 | 0.059 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | 0.025 ± 0.000 | 0.027 ± 0.000 | 0.065 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 46.036 ± 0.804 | - | 32.775 ± 1.786 | 23.075 ± 1.316 | 28.235 ± 1.960 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.686 ± 0.011 | - | 0.709 ± 0.033 | 0.646 ± 0.061 | 0.757 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.044 ± 0.000 | - | 0.047 ± 0.000 | 0.050 ± 0.001 | 0.091 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.283 ± 0.014 | - | 0.284 ± 0.014 | 0.391 ± 0.002 | 0.747 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.572 ± 0.133 | - | 1.656 ± 0.052 | 1.550 ± 0.006 | 1.057 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 6.187 ± 0.115 | - | 6.741 ± 0.431 | 5.947 ± 0.063 | 3.413 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 6.741 ± 0.447 | - | 4.934 ± 0.049 | 3.050 ± 0.259 | 3.976 ± 0.168 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.001 | 0.004 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.101 ± 0.018 | - | 0.108 ± 0.016 | 0.082 ± 0.001 | 0.152 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.011 ± 0.003 | 0.009 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.007 ± 0.001 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.023 ± 0.005 | - | 0.024 ± 0.005 | 0.021 ± 0.000 | 0.079 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.010 ± 0.002 | - | 0.011 ± 0.001 | 0.010 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.026 ± 0.001 | - | 0.030 ± 0.007 | 0.027 ± 0.000 | 0.089 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 52.698 ± 2.332 | - | 37.928 ± 2.643 | 23.064 ± 1.900 | 15.878 ± 1.203 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.002 | 0.007 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.783 ± 0.052 | - | 0.791 ± 0.029 | 0.630 ± 0.005 | 0.476 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.014 ± 0.003 | - | 0.019 ± 0.004 | 0.016 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.049 ± 0.001 | - | 0.053 ± 0.001 | 0.050 ± 0.001 | 0.109 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.220 ± 0.000 | - | 0.224 ± 0.001 | 0.232 ± 0.000 | 0.522 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.147 ± 0.010 | - | 1.151 ± 0.009 | 1.230 ± 0.062 | 1.392 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.702 ± 0.124 | - | 4.463 ± 0.052 | 4.472 ± 0.024 | 4.812 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.863 ± 0.003 | - | 0.866 ± 0.003 | 0.869 ± 0.003 | 0.936 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.058 ± 0.001 | - | 0.061 ± 0.000 | 0.061 ± 0.001 | 0.096 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.031 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.022 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 6.873 ± 0.012 | - | 7.400 ± 0.760 | 6.882 ± 0.033 | 7.185 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.454 ± 0.000 | - | 0.458 ± 0.001 | 0.457 ± 0.002 | 0.541 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.036 ± 0.003 | - | 0.038 ± 0.001 | 0.040 ± 0.001 | 0.080 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.221 ± 0.001 | - | 0.225 ± 0.001 | 0.222 ± 0.000 | 0.513 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.144 ± 0.018 | - | 1.145 ± 0.015 | 1.144 ± 0.012 | 0.964 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 4.469 ± 0.135 | - | 4.799 ± 0.097 | 4.505 ± 0.017 | 2.800 ± 0.589 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 0.865 ± 0.007 | - | 0.869 ± 0.003 | 0.976 ± 0.042 | 1.116 ± 0.105 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.063 ± 0.000 | - | 0.067 ± 0.002 | 0.061 ± 0.001 | 0.089 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.002 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.007 ± 0.001 | - | 0.009 ± 0.001 | 0.008 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.005 ± 0.001 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.015 ± 0.004 | - | 0.018 ± 0.000 | 0.017 ± 0.001 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.002 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.007 ± 0.001 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.019 ± 0.000 | - | 0.022 ± 0.004 | 0.021 ± 0.000 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 6.881 ± 0.017 | - | 7.872 ± 0.673 | 7.671 ± 0.698 | 3.898 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.455 ± 0.033 | - | 0.459 ± 0.001 | 0.487 ± 0.042 | 0.335 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.012 ± 0.003 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.036 ± 0.001 | - | 0.040 ± 0.000 | 0.038 ± 0.001 | 0.067 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.101 ± 0.000 | - | 0.106 ± 0.003 | 0.118 ± 0.002 | 0.497 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.674 ± 0.010 | - | 0.680 ± 0.011 | 0.684 ± 0.004 | 1.387 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.168 ± 0.016 | - | 2.182 ± 0.022 | 2.194 ± 0.009 | 4.777 ± 0.130 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.445 ± 0.008 | - | 0.436 ± 0.010 | 0.420 ± 0.007 | 1.083 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.001 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.025 ± 0.001 | - | 0.027 ± 0.001 | 0.027 ± 0.000 | 0.095 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.495 ± 0.042 | - | 3.317 ± 0.005 | 3.835 ± 0.164 | 7.372 ± 0.795 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.185 ± 0.002 | - | 0.189 ± 0.002 | 0.181 ± 0.000 | 0.505 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.002 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.023 ± 0.001 | 0.088 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.142 ± 0.026 | - | 0.202 ± 0.034 | 0.126 ± 0.001 | 0.525 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.787 ± 0.174 | - | 0.799 ± 0.056 | 0.712 ± 0.006 | 0.950 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.458 ± 0.063 | - | 2.529 ± 0.089 | 2.217 ± 0.009 | 2.569 ± 0.162 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.455 ± 0.032 | - | 0.474 ± 0.015 | 0.419 ± 0.002 | 0.946 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.026 ± 0.002 | - | 0.032 ± 0.004 | 0.027 ± 0.000 | 0.117 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.002 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.001 | - | 0.008 ± 0.002 | 0.007 ± 0.000 | 0.024 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.010 ± 0.001 | - | 0.012 ± 0.003 | 0.011 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.005 ± 0.001 | - | 0.026 ± 0.009 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.011 ± 0.002 | - | 0.018 ± 0.008 | 0.013 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 3.744 ± 0.217 | - | 3.957 ± 0.075 | 3.354 ± 0.041 | 4.358 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.186 ± 0.005 | - | 0.198 ± 0.016 | 0.181 ± 0.001 | 0.316 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.009 ± 0.001 | - | 0.010 ± 0.003 | 0.010 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.022 ± 0.004 | - | 0.026 ± 0.005 | 0.022 ± 0.000 | 0.101 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.090 ± 0.002 | - | 0.100 ± 0.011 | 0.138 ± 0.001 | 0.131 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.193 ± 0.011 | - | 0.198 ± 0.015 | 0.217 ± 0.002 | 0.198 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.415 ± 0.029 | - | 0.447 ± 0.029 | 0.442 ± 0.001 | 0.431 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.059 ± 0.006 | - | 0.062 ± 0.001 | 0.058 ± 0.001 | 0.112 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.008 ± 0.001 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.046 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.039 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.045 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.443 ± 0.020 | - | 0.482 ± 0.020 | 0.438 ± 0.020 | 0.569 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.032 ± 0.006 | - | 0.039 ± 0.005 | 0.035 ± 0.000 | 0.092 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.001 | 0.010 ± 0.000 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.091 ± 0.018 | - | 0.087 ± 0.004 | 0.146 ± 0.009 | 0.183 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.180 ± 0.001 | - | 0.184 ± 0.001 | 0.226 ± 0.001 | 0.260 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.397 ± 0.002 | - | 0.420 ± 0.008 | 0.419 ± 0.007 | 0.310 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.057 ± 0.001 | - | 0.063 ± 0.001 | 0.111 ± 0.017 | 0.141 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.050 ± 0.012 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.007 ± 0.001 | - | 0.009 ± 0.001 | 0.052 ± 0.002 | 0.062 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.053 ± 0.017 | 0.013 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.009 ± 0.000 | - | 0.012 ± 0.001 | 0.049 ± 0.009 | 0.076 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.048 ± 0.006 | 0.020 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.050 ± 0.008 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.048 ± 0.017 | 0.012 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.002 | 0.036 ± 0.021 | 0.009 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.049 ± 0.007 | 0.018 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.047 ± 0.019 | 0.021 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.002 | 0.051 ± 0.002 | 0.061 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.463 ± 0.053 | - | 0.561 ± 0.043 | 0.367 ± 0.002 | 0.444 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.073 ± 0.024 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.031 ± 0.000 | - | 0.036 ± 0.001 | 0.108 ± 0.011 | 0.122 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.067 ± 0.014 | 0.088 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.075 ± 0.008 | 0.062 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.252 ± 0.002 | - | 0.259 ± 0.002 | 0.277 ± 0.003 | 0.339 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.581 ± 0.066 | - | 0.565 ± 0.003 | 0.587 ± 0.004 | 0.600 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.385 ± 0.009 | - | 1.421 ± 0.126 | 1.418 ± 0.012 | 1.252 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.275 ± 0.002 | - | 0.284 ± 0.001 | 0.305 ± 0.002 | 0.325 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.017 ± 0.000 | - | 0.022 ± 0.000 | 0.018 ± 0.000 | 0.050 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.052 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.004 ± 0.003 | - | 0.008 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.197 ± 0.047 | - | 2.211 ± 0.009 | 2.735 ± 0.199 | 2.649 ± 0.420 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.122 ± 0.000 | - | 0.129 ± 0.000 | 0.127 ± 0.000 | 0.129 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.012 ± 0.000 | - | 0.017 ± 0.000 | 0.014 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.189 ± 0.001 | - | 0.193 ± 0.002 | 0.235 ± 0.002 | 0.334 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.487 ± 0.005 | - | 0.489 ± 0.018 | 0.493 ± 0.003 | 0.589 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.215 ± 0.051 | - | 1.161 ± 0.110 | 1.162 ± 0.006 | 0.779 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.276 ± 0.003 | - | 0.286 ± 0.002 | 0.339 ± 0.004 | 0.353 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.022 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.031 ± 0.003 | 0.070 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.020 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.035 ± 0.003 | 0.075 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.003 | - | 0.006 ± 0.000 | 0.021 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.002 | 0.023 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.001 | - | 0.006 ± 0.002 | 0.020 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.005 ± 0.001 | - | 0.008 ± 0.002 | 0.023 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.022 ± 0.010 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.002 | 0.021 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.007 ± 0.001 | - | 0.009 ± 0.003 | 0.023 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.213 ± 0.092 | - | 2.242 ± 0.023 | 2.350 ± 0.041 | 1.587 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.002 | 0.026 ± 0.021 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.081 ± 0.000 | - | 0.086 ± 0.000 | 0.116 ± 0.001 | 0.175 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.022 ± 0.002 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.012 ± 0.002 | - | 0.014 ± 0.000 | 0.028 ± 0.003 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.134 ± 0.002 | - | 0.120 ± 0.001 | 0.097 ± 0.001 | 0.191 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.210 ± 0.001 | - | 0.178 ± 0.001 | 0.154 ± 0.000 | 0.265 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.377 ± 0.055 | - | 0.363 ± 0.007 | 0.322 ± 0.001 | 0.528 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.051 ± 0.002 | - | 0.056 ± 0.003 | 0.055 ± 0.002 | 0.101 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.016 ± 0.019 | 0.009 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.042 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.361 ± 0.003 | - | 0.398 ± 0.009 | 0.346 ± 0.005 | 0.467 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.027 ± 0.000 | - | 0.034 ± 0.001 | 0.034 ± 0.000 | 0.082 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.006 ± 0.002 | 0.009 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.132 ± 0.003 | - | 0.118 ± 0.003 | 0.117 ± 0.011 | 0.174 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.197 ± 0.001 | - | 0.184 ± 0.004 | 0.150 ± 0.001 | 0.240 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.313 ± 0.040 | - | 0.310 ± 0.032 | 0.287 ± 0.010 | 0.354 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.048 ± 0.000 | - | 0.057 ± 0.000 | 0.060 ± 0.001 | 0.123 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.011 | 0.009 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.005 ± 0.001 | - | 0.010 ± 0.001 | 0.013 ± 0.000 | 0.064 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.001 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.002 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.002 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.002 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.365 ± 0.005 | - | 0.490 ± 0.014 | 0.283 ± 0.006 | 0.399 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.044 ± 0.017 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.026 ± 0.001 | - | 0.034 ± 0.001 | 0.072 ± 0.003 | 0.132 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.001 | - | 0.008 ± 0.001 | 0.030 ± 0.007 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.044 ± 0.013 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.115 ± 0.036 | - | 1.123 ± 0.013 | 1.068 ± 0.007 | 3.125 ± 0.168 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.045 ± 0.265 | - | 3.910 ± 0.018 | 3.889 ± 0.018 | 5.452 ± 0.298 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 14.129 ± 0.774 | - | 13.425 ± 0.061 | 14.235 ± 0.904 | 15.388 ± 0.413 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.598 ± 0.170 | - | 5.083 ± 0.061 | 3.976 ± 0.053 | 4.558 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.163 ± 0.002 | - | 0.165 ± 0.001 | 0.166 ± 0.003 | 0.244 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.021 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.159 ± 0.001 | - | 0.161 ± 0.012 | 0.154 ± 0.001 | 0.236 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.043 ± 0.001 | - | 0.045 ± 0.000 | 0.046 ± 0.000 | 0.058 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.025 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.057 ± 0.000 | - | 0.060 ± 0.000 | 0.060 ± 0.000 | 0.074 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 33.979 ± 3.046 | - | 45.592 ± 1.889 | 32.121 ± 2.580 | 40.253 ± 1.482 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.043 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.260 ± 0.008 | - | 1.358 ± 0.151 | 1.265 ± 0.024 | 1.449 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.035 ± 0.000 | - | 0.037 ± 0.005 | 0.036 ± 0.000 | 0.058 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.108 ± 0.001 | - | 0.112 ± 0.000 | 0.114 ± 0.005 | 0.179 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.063 ± 0.010 | - | 1.069 ± 0.011 | 1.066 ± 0.012 | 2.646 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 3.914 ± 0.257 | - | 4.180 ± 0.084 | 4.188 ± 0.080 | 2.335 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 13.541 ± 0.871 | - | 13.434 ± 0.053 | 13.457 ± 0.072 | 8.007 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.522 ± 0.455 | - | 5.514 ± 0.555 | 4.359 ± 0.159 | 4.555 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.001 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.164 ± 0.003 | - | 0.167 ± 0.000 | 0.168 ± 0.003 | 0.276 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.008 ± 0.001 | - | 0.010 ± 0.000 | 0.010 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.160 ± 0.004 | - | 0.164 ± 0.003 | 0.154 ± 0.002 | 0.230 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.023 ± 0.006 | - | 0.039 ± 0.004 | 0.021 ± 0.001 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.007 ± 0.001 | - | 0.009 ± 0.000 | 0.009 ± 0.003 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.014 ± 0.004 | - | 0.017 ± 0.000 | 0.017 ± 0.001 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.059 ± 0.002 | - | 0.067 ± 0.003 | 0.048 ± 0.001 | 0.060 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.008 ± 0.001 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.018 ± 0.002 | - | 0.021 ± 0.000 | 0.021 ± 0.001 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.068 ± 0.009 | - | 0.085 ± 0.002 | 0.063 ± 0.001 | 0.080 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 34.687 ± 2.756 | - | 41.759 ± 3.397 | 32.888 ± 1.496 | 19.859 ± 1.692 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.012 ± 0.003 | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.261 ± 0.005 | - | 1.273 ± 0.018 | 1.270 ± 0.009 | 0.844 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.033 ± 0.000 | - | 0.036 ± 0.000 | 0.037 ± 0.001 | 0.054 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.144 ± 0.005 | - | 0.147 ± 0.003 | 0.117 ± 0.002 | 0.188 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.521 ± 0.001 | - | 0.525 ± 0.001 | 0.714 ± 0.021 | 1.901 ± 0.183 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.475 ± 0.149 | - | 2.483 ± 0.008 | 3.506 ± 0.614 | 4.157 ± 0.555 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 9.930 ± 0.610 | - | 9.604 ± 0.022 | 9.604 ± 0.064 | 12.956 ± 0.505 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.325 ± 0.010 | - | 1.340 ± 0.012 | 1.325 ± 0.005 | 1.762 ± 0.122 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.016 ± 0.002 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.108 ± 0.002 | - | 0.109 ± 0.001 | 0.115 ± 0.001 | 0.188 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.002 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.090 ± 0.001 | - | 0.093 ± 0.006 | 0.088 ± 0.000 | 0.145 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.022 ± 0.008 | 0.043 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.019 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.025 ± 0.001 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.027 ± 0.000 | - | 0.029 ± 0.000 | 0.042 ± 0.021 | 0.047 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.002 | 0.018 ± 0.005 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.029 ± 0.002 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.037 ± 0.000 | - | 0.039 ± 0.000 | 0.053 ± 0.003 | 0.068 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 11.137 ± 0.776 | - | 10.607 ± 0.124 | 11.389 ± 0.845 | 12.635 ± 1.047 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.005 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.842 ± 0.007 | - | 0.851 ± 0.012 | 0.847 ± 0.013 | 1.087 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.021 ± 0.000 | - | 0.024 ± 0.000 | 0.041 ± 0.002 | 0.040 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.072 ± 0.000 | - | 0.076 ± 0.004 | 0.095 ± 0.034 | 0.161 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.563 ± 0.014 | - | 0.567 ± 0.005 | 0.523 ± 0.002 | 1.796 ± 0.120 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.639 ± 0.069 | - | 2.502 ± 0.049 | 2.471 ± 0.018 | 1.616 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 9.591 ± 0.047 | - | 9.931 ± 0.677 | 10.302 ± 0.245 | 6.480 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.323 ± 0.010 | - | 1.349 ± 0.029 | 1.330 ± 0.006 | 1.707 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.003 | - | 0.007 ± 0.001 | 0.006 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.108 ± 0.001 | - | 0.115 ± 0.003 | 0.112 ± 0.001 | 0.191 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.085 ± 0.001 | - | 0.089 ± 0.002 | 0.088 ± 0.002 | 0.181 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.012 ± 0.001 | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.017 ± 0.003 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.012 ± 0.003 | - | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.028 ± 0.001 | - | 0.031 ± 0.001 | 0.031 ± 0.000 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.001 | 0.007 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.014 ± 0.003 | - | 0.017 ± 0.001 | 0.015 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.037 ± 0.000 | - | 0.041 ± 0.001 | 0.041 ± 0.003 | 0.056 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 11.266 ± 0.786 | - | 10.646 ± 0.081 | 10.642 ± 0.101 | 6.555 ± 0.100 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.009 ± 0.002 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.843 ± 0.011 | - | 0.847 ± 0.016 | 0.846 ± 0.006 | 0.646 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.024 ± 0.000 | - | 0.027 ± 0.000 | 0.025 ± 0.000 | 0.040 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.073 ± 0.000 | - | 0.077 ± 0.001 | 0.076 ± 0.001 | 0.174 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.205 ± 0.018 | - | unsupported | 0.202 ± 0.000 | 1.349 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.375 ± 0.051 | - | unsupported | 1.249 ± 0.020 | 2.320 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 3.903 ± 0.148 | - | unsupported | 3.532 ± 0.005 | 4.689 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.770 ± 0.072 | - | unsupported | 0.707 ± 0.006 | 0.713 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.043 ± 0.006 | - | unsupported | 0.040 ± 0.000 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.031 ± 0.007 | - | unsupported | 0.032 ± 0.001 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.014 ± 0.001 | - | unsupported | 0.014 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.015 ± 0.001 | - | unsupported | 0.018 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 6.284 ± 0.014 | - | unsupported | 5.650 ± 0.159 | 5.676 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.288 ± 0.030 | - | unsupported | 0.272 ± 0.001 | 0.303 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.012 ± 0.003 | - | unsupported | 0.014 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.029 ± 0.000 | - | unsupported | 0.032 ± 0.000 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.202 ± 0.009 | - | unsupported | 0.203 ± 0.001 | 1.228 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.253 ± 0.139 | - | unsupported | 1.260 ± 0.013 | 1.070 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 3.524 ± 0.011 | - | unsupported | 3.850 ± 0.057 | 2.522 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.704 ± 0.010 | - | unsupported | 0.754 ± 0.029 | 0.783 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.035 ± 0.000 | - | unsupported | 0.041 ± 0.002 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.029 ± 0.001 | - | unsupported | 0.032 ± 0.000 | 0.055 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.011 ± 0.000 | - | unsupported | 0.014 ± 0.000 | 0.025 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.015 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 5.612 ± 0.015 | - | unsupported | 5.633 ± 0.024 | 3.088 ± 0.225 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.267 ± 0.000 | - | unsupported | 0.283 ± 0.002 | 0.218 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.011 ± 0.000 | - | unsupported | 0.014 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.029 ± 0.001 | - | unsupported | 0.032 ± 0.016 | 0.047 ± 0.002 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.806 ± 0.034 | - | unsupported | 0.764 ± 0.025 | 0.887 ± 0.012 | 2.341 ± 0.553 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.052 ± 0.000 | - | unsupported | 0.069 ± 0.001 | 0.129 ± 0.009 | 0.063 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.104 ± 0.000 | - | unsupported | 0.119 ± 0.009 | 0.205 ± 0.016 | 0.132 ± 0.014 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.257 ± 0.018 | - | unsupported | 0.258 ± 0.010 | 0.375 ± 0.022 | 0.589 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.732 ± 0.043 | - | unsupported | 0.455 ± 0.015 | 0.707 ± 0.094 | 2.843 ± 0.017 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.052 ± 0.000 | - | unsupported | 0.083 ± 0.012 | 0.142 ± 0.003 | 0.073 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.104 ± 0.000 | - | unsupported | 0.099 ± 0.012 | 0.187 ± 0.003 | 0.141 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.241 ± 0.000 | - | unsupported | 0.218 ± 0.023 | 0.321 ± 0.082 | 0.629 ± 0.025 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.209 ± 0.001 | - | unsupported | 0.173 ± 0.002 | 0.481 ± 0.062 | 0.454 ± 0.023 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.060 ± 0.009 | - | unsupported | 0.040 ± 0.000 | 0.080 ± 0.006 | 0.062 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.062 ± 0.002 | - | unsupported | 0.046 ± 0.000 | 0.114 ± 0.025 | 0.087 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.084 ± 0.009 | - | unsupported | 0.075 ± 0.000 | 0.152 ± 0.004 | 0.224 ± 0.015 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.228 ± 0.015 | - | unsupported | 0.181 ± 0.003 | 0.449 ± 0.008 | 0.447 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.051 ± 0.003 | - | unsupported | 0.041 ± 0.001 | 0.099 ± 0.006 | 0.066 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.061 ± 0.001 | - | unsupported | 0.048 ± 0.001 | 0.110 ± 0.007 | 0.097 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.083 ± 0.002 | - | unsupported | 0.078 ± 0.002 | 0.184 ± 0.012 | 0.227 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.107 ± 0.000 | - | 0.168 ± 0.003 | 0.074 ± 0.001 | 0.193 ± 0.025 | 0.228 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.046 ± 0.000 | - | 0.055 ± 0.025 | 0.030 ± 0.009 | 0.023 ± 0.001 | 0.034 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.047 ± 0.007 | - | 0.052 ± 0.000 | 0.025 ± 0.010 | 0.029 ± 0.001 | 0.048 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.056 ± 0.000 | - | 0.074 ± 0.000 | 0.034 ± 0.000 | 0.053 ± 0.001 | 0.097 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.108 ± 0.001 | - | 0.187 ± 0.008 | 0.081 ± 0.004 | 0.216 ± 0.013 | 0.234 ± 0.126 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.056 ± 0.002 | - | 0.045 ± 0.013 | 0.023 ± 0.013 | 0.025 ± 0.002 | 0.039 ± 0.006 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.043 ± 0.001 | - | 0.053 ± 0.000 | 0.025 ± 0.000 | 0.028 ± 0.001 | 0.048 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.056 ± 0.000 | - | 0.075 ± 0.001 | 0.037 ± 0.001 | 0.070 ± 0.001 | 0.097 ± 0.000 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 10.034 ± 1.892 | - | 11.670 ± 1.197 | 19.634 ± 5.862 | 14.874 ± 3.291 | 39.017 ± 14.413 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 9.138 ± 2.569 | - | 9.531 ± 1.350 | 16.616 ± 4.181 | 11.293 ± 2.811 | 352.729 ± 5.718 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 6.345 ± 0.107 | - | 7.526 ± 0.808 | 6.803 ± 0.558 | 8.356 ± 0.842 | 19.723 ± 2.663 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 6.301 ± 1.263 | - | 6.300 ± 0.228 | 5.499 ± 0.142 | 5.713 ± 0.091 | 231.453 ± 136.583 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.859 ± 0.192 | - | 7.134 ± 0.921 | 7.013 ± 0.327 | 8.403 ± 0.274 | 6.932 ± 0.256 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 6.161 ± 0.286 | - | 6.405 ± 0.550 | 6.137 ± 0.188 | 7.182 ± 0.052 | 307.092 ± 7.068 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.604 ± 0.208 | - | 7.947 ± 0.402 | 7.537 ± 0.505 | 6.330 ± 0.069 | 7.195 ± 0.307 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 6.708 ± 0.193 | - | 6.738 ± 0.453 | 7.894 ± 0.924 | 7.065 ± 0.324 | 247.385 ± 15.233 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 19.692 ± 2.654 | - | 17.791 ± 1.942 | 20.137 ± 0.678 | 23.122 ± 2.146 | 13.157 ± 0.416 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 17.770 ± 0.104 | - | 17.767 ± 0.124 | 18.116 ± 1.333 | 22.565 ± 2.308 | 624.142 ± 35.675 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.245 ± 0.005 | 0.253 ± 0.028 | unsupported | 0.267 ± 0.003 | 0.290 ± 0.003 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 3.222 ± 0.399 | 3.668 ± 0.215 | unsupported | 3.283 ± 0.342 | 3.796 ± 0.564 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.247 ± 0.005 | 0.292 ± 0.014 | unsupported | 0.266 ± 0.001 | 0.360 ± 0.069 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.226 ± 0.324 | 3.296 ± 0.335 | unsupported | 3.189 ± 0.439 | 3.662 ± 0.595 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 7.799 ± 1.952 | - | 7.562 ± 0.291 | 7.253 ± 1.534 | 7.552 ± 0.120 | 27.340 ± 2.784 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 12.327 ± 5.762 | - | 10.671 ± 0.334 | 8.804 ± 0.685 | 7.301 ± 1.458 | 336.695 ± 23.481 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 9.061 ± 1.077 | 8.531 ± 0.588 | 4.862 ± 0.419 | 32.150 ± 7.686 | 7.280 ± 0.518 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 10.688 ± 1.187 | 9.445 ± 1.005 | 4.255 ± 0.018 | 25.349 ± 6.208 | 134.411 ± 20.407 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 8.781 ± 0.640 | - | 8.174 ± 1.051 | 8.556 ± 0.071 | 8.860 ± 0.227 | 28.057 ± 7.320 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 9.770 ± 0.402 | - | 10.233 ± 0.886 | 8.396 ± 0.398 | 10.067 ± 3.068 | 255.526 ± 15.592 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.535 ± 0.014 | - | 0.597 ± 0.002 | 2.118 ± 0.139 | 5.530 ± 0.256 | 5.271 ± 0.017 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.590 ± 0.002 | - | 0.633 ± 0.008 | 2.012 ± 0.026 | 3.595 ± 0.044 | 8.148 ± 3.464 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 11.167 ± 1.275 | - | 9.857 ± 0.644 | 9.481 ± 0.983 | 11.955 ± 1.676 | 17.896 ± 1.647 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 14.741 ± 2.053 | - | 14.825 ± 3.211 | 9.066 ± 0.820 | 10.290 ± 0.138 | 966.334 ± 37.803 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 10.822 ± 0.848 | - | 10.383 ± 1.097 | 9.534 ± 1.404 | 10.732 ± 0.116 | 17.786 ± 1.346 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 12.715 ± 3.792 | - | 12.171 ± 1.221 | 10.374 ± 0.528 | 11.587 ± 0.373 | 941.035 ± 333.853 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.285 ± 0.082 | 1.323 ± 0.102 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 12.054 ± 0.180 | 13.850 ± 1.670 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.290 ± 0.064 | 1.358 ± 0.138 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 12.986 ± 0.081 | 13.092 ± 0.165 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.341 ± 0.077 | 1.319 ± 0.099 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 10.874 ± 0.187 | 12.156 ± 0.351 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.354 ± 0.091 | 1.335 ± 0.103 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 11.175 ± 0.134 | 11.289 ± 0.176 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 6.280 ± 0.108 | - | 6.112 ± 0.179 | 7.422 ± 0.566 | 6.096 ± 2.014 | 18.689 ± 1.695 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 9.393 ± 3.873 | - | 7.745 ± 1.489 | 5.491 ± 0.048 | 6.033 ± 1.725 | 229.141 ± 26.563 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 32.875 ± 3.919 | 31.982 ± 3.416 | 30.066 ± 3.976 | 37.037 ± 6.968 | 64.378 ± 2.053 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 32.014 ± 2.523 | 30.129 ± 3.723 | 29.723 ± 2.845 | 31.212 ± 3.021 | 1636.550 ± 76.179 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 12.164 ± 0.253 | - | 19.912 ± 4.274 | 13.144 ± 0.592 | 60.501 ± 4.110 | 26.333 ± 2.349 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 12.428 ± 2.179 | - | 18.110 ± 2.732 | 12.020 ± 1.802 | 40.614 ± 4.196 | 14.593 ± 1.022 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.162 ± 0.027 | 0.169 ± 0.011 | unsupported | 0.049 ± 0.001 | 0.031 ± 0.000 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.667 ± 0.146 | 2.683 ± 0.197 | unsupported | 0.833 ± 0.037 | 0.421 ± 0.009 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.172 ± 0.013 | 0.163 ± 0.010 | unsupported | 0.048 ± 0.004 | 0.046 ± 0.016 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 1.372 ± 0.063 | 1.357 ± 0.054 | unsupported | 0.443 ± 0.115 | 0.278 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.072 ± 0.000 | 0.073 ± 0.000 | unsupported | 0.080 ± 0.004 | 0.168 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.224 ± 0.068 | 1.157 ± 0.015 | unsupported | 1.343 ± 0.063 | 2.168 ± 0.022 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.047 ± 0.012 | 0.047 ± 0.009 | unsupported | 0.060 ± 0.000 | 0.131 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.621 ± 0.003 | 0.652 ± 0.024 | unsupported | 0.673 ± 0.018 | 1.230 ± 0.087 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.247 ± 0.014 | 0.275 ± 0.043 | unsupported | 0.275 ± 0.015 | 0.078 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.928 ± 0.237 | 3.951 ± 0.212 | unsupported | 4.834 ± 0.210 | 0.836 ± 0.007 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.153 ± 0.013 | 0.164 ± 0.011 | unsupported | 0.176 ± 0.004 | 0.083 ± 0.006 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 2.035 ± 0.071 | 2.097 ± 0.066 | unsupported | 2.290 ± 0.011 | 0.466 ± 0.017 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.041 ± 0.001 | 0.060 ± 0.005 | unsupported | 0.048 ± 0.004 | 0.075 ± 0.007 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.496 ± 0.007 | 0.521 ± 0.001 | unsupported | 0.580 ± 0.023 | 0.585 ± 0.009 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.050 ± 0.001 | 0.068 ± 0.007 | unsupported | 0.051 ± 0.003 | 0.086 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.833 ± 0.129 | 0.686 ± 0.012 | unsupported | 0.332 ± 0.004 | 0.488 ± 0.018 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.082 ± 0.001 | 0.098 ± 0.005 | unsupported | 0.090 ± 0.002 | 0.102 ± 0.011 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.062 ± 0.008 | 1.096 ± 0.048 | unsupported | 1.442 ± 0.087 | 1.232 ± 0.015 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.093 ± 0.001 | 0.104 ± 0.007 | unsupported | 0.061 ± 0.002 | 0.120 ± 0.003 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.928 ± 0.008 | 1.264 ± 0.120 | unsupported | 0.687 ± 0.003 | 0.735 ± 0.080 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.121 ± 0.008 | 0.132 ± 0.002 | unsupported | unsupported | 0.109 ± 0.001 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 1.687 ± 0.154 | 1.874 ± 0.251 | unsupported | unsupported | 1.533 ± 0.272 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.119 ± 0.002 | 0.141 ± 0.009 | unsupported | unsupported | 0.129 ± 0.010 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.240 ± 0.048 | 1.531 ± 0.269 | unsupported | unsupported | 0.895 ± 0.029 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.011 ± 0.000 | 0.017 ± 0.000 | unsupported | 0.007 ± 0.000 | 0.047 ± 0.007 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.104 ± 0.003 | 0.100 ± 0.002 | unsupported | 0.074 ± 0.023 | 0.205 ± 0.004 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.012 ± 0.000 | 0.017 ± 0.000 | unsupported | 0.034 ± 0.001 | 0.059 ± 0.005 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.086 ± 0.004 | 0.107 ± 0.015 | unsupported | 0.065 ± 0.014 | 0.154 ± 0.020 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.022 ± 0.000 | 0.034 ± 0.002 | unsupported | 0.028 ± 0.002 | 0.062 ± 0.001 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.283 ± 0.006 | 0.413 ± 0.125 | unsupported | 0.383 ± 0.022 | 0.602 ± 0.060 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.025 ± 0.001 | 0.038 ± 0.001 | unsupported | 0.108 ± 0.003 | 0.077 ± 0.012 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.374 ± 0.022 | 0.464 ± 0.021 | unsupported | 0.338 ± 0.019 | 0.376 ± 0.017 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.046 ± 0.005 | 0.045 ± 0.000 | unsupported | 0.071 ± 0.004 | 0.076 ± 0.001 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.704 ± 0.002 | 0.753 ± 0.007 | unsupported | 1.125 ± 0.010 | 0.731 ± 0.052 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.029 ± 0.004 | 0.030 ± 0.003 | unsupported | 0.073 ± 0.002 | 0.081 ± 0.018 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.389 ± 0.002 | 0.392 ± 0.018 | unsupported | 0.635 ± 0.011 | 0.456 ± 0.013 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.046 ± 0.000 | 0.047 ± 0.003 | unsupported | 0.077 ± 0.001 | 0.081 ± 0.007 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.735 ± 0.002 | 0.735 ± 0.006 | unsupported | 1.214 ± 0.045 | 0.792 ± 0.004 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.030 ± 0.000 | 0.030 ± 0.000 | unsupported | 0.074 ± 0.003 | 0.066 ± 0.026 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.510 ± 0.113 | 0.493 ± 0.050 | unsupported | 0.625 ± 0.008 | 0.455 ± 0.023 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.084 ± 0.007 | 0.095 ± 0.001 | unsupported | 0.091 ± 0.002 | 0.097 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.053 ± 0.006 | 1.212 ± 0.171 | unsupported | 1.690 ± 0.049 | 1.072 ± 0.012 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.086 ± 0.004 | 0.102 ± 0.001 | unsupported | 0.068 ± 0.002 | 0.130 ± 0.001 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.922 ± 0.005 | 1.267 ± 0.128 | unsupported | 0.822 ± 0.042 | 0.655 ± 0.013 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.203 ± 0.013 | 0.203 ± 0.013 | unsupported | 0.257 ± 0.001 | 0.271 ± 0.003 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.238 ± 0.183 | 3.248 ± 0.121 | unsupported | 4.154 ± 0.089 | 3.818 ± 0.232 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.132 ± 0.005 | 0.136 ± 0.008 | unsupported | 0.167 ± 0.001 | 0.181 ± 0.018 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.701 ± 0.054 | 1.724 ± 0.076 | unsupported | 2.237 ± 0.086 | 2.147 ± 0.110 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.519 ± 0.036 | 0.597 ± 0.073 | unsupported | 0.122 ± 0.004 | 0.085 ± 0.002 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 8.730 ± 0.478 | 10.001 ± 0.691 | unsupported | 2.294 ± 0.138 | 0.868 ± 0.010 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.586 ± 0.053 | 0.729 ± 0.088 | unsupported | 0.134 ± 0.011 | 0.087 ± 0.009 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 6.603 ± 0.365 | 7.720 ± 0.559 | unsupported | 1.085 ± 0.026 | 0.625 ± 0.015 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 14.539 ± 1.617 | - | unsupported | 20.038 ± 3.191 | - | 14.017 ± 0.178 | 18.436 ± 0.995 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 17.061 ± 1.646 | - | unsupported | 16.807 ± 1.232 | - | 14.438 ± 1.661 | 16.335 ± 2.228 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 9.090 ± 0.240 | - | unsupported | 12.697 ± 1.063 | - | 11.101 ± 0.658 | 11.572 ± 0.661 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 10.765 ± 1.111 | - | unsupported | 9.193 ± 1.168 | - | 10.962 ± 1.518 | 9.254 ± 1.085 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 11.206 ± 2.555 | - | unsupported | 10.554 ± 1.625 | - | 8.974 ± 1.727 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 9.156 ± 1.723 | - | unsupported | 10.708 ± 0.612 | - | 9.012 ± 0.299 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 14.349 ± 4.013 | - | unsupported | 20.049 ± 0.582 | - | 14.090 ± 2.886 | 17.667 ± 3.528 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 17.396 ± 1.451 | - | unsupported | 16.974 ± 3.379 | - | 14.011 ± 3.481 | 16.072 ± 2.256 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 7.856 ± 0.164 | - | unsupported | 8.052 ± 0.589 | - | 45.098 ± 2.653 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 7.813 ± 0.839 | - | unsupported | 9.261 ± 2.334 | - | 22.326 ± 1.835 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 11.465 ± 2.189 | - | unsupported | 8.950 ± 2.457 | - | 44.526 ± 1.975 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 8.775 ± 2.045 | - | unsupported | 8.718 ± 0.261 | - | 22.646 ± 3.111 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 17.376 ± 3.454 | - | unsupported | 19.896 ± 0.531 | - | 14.061 ± 0.176 | 14.152 ± 0.424 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 14.931 ± 1.890 | - | unsupported | 16.511 ± 0.576 | - | 17.171 ± 4.054 | 15.900 ± 2.033 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 9.173 ± 1.422 | - | unsupported | 12.396 ± 2.155 | - | 9.126 ± 0.030 | 10.838 ± 1.259 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 9.645 ± 2.066 | - | unsupported | 9.191 ± 1.307 | - | 10.068 ± 0.923 | 9.223 ± 0.939 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.174 ± 0.002 | unsupported | unsupported | 0.255 ± 0.014 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 2.166 ± 0.259 | unsupported | unsupported | 2.588 ± 0.141 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.197 ± 0.008 | unsupported | unsupported | 0.282 ± 0.011 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 2.208 ± 0.458 | unsupported | unsupported | 2.436 ± 0.193 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 16.856 ± 6.329 | - | unsupported | 20.555 ± 0.723 | - | 17.760 ± 2.113 | 14.456 ± 3.970 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 16.600 ± 2.958 | - | unsupported | 18.609 ± 3.977 | - | 17.634 ± 0.690 | 16.488 ± 0.543 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 8.063 ± 0.778 | - | 0.002 ± 0.000 | 5.391 ± 1.314 | 4.238 ± 0.228 | 12.461 ± 0.042 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 8.398 ± 0.741 | - | 0.002 ± 0.001 | 4.434 ± 0.240 | 4.195 ± 0.025 | 12.594 ± 0.803 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 6.992 ± 0.064 | - | 18.875 ± 4.375 | 7.027 ± 0.109 | 9.129 ± 1.063 | 7.405 ± 0.178 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 7.221 ± 0.022 | - | 17.361 ± 2.497 | 9.463 ± 0.801 | 8.525 ± 0.297 | 9.518 ± 0.494 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 11.137 ± 3.106 | - | 8.416 ± 0.076 | 8.425 ± 0.718 | 14.652 ± 1.103 | 12.353 ± 1.486 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 10.204 ± 0.718 | - | 8.457 ± 0.414 | 11.340 ± 0.354 | 8.767 ± 1.665 | 12.005 ± 3.089 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 5.283 ± 0.859 | - | 5.550 ± 1.130 | 8.160 ± 0.047 | 9.039 ± 0.512 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 5.255 ± 0.511 | - | 4.506 ± 0.041 | 9.465 ± 1.232 | 8.550 ± 0.424 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 8.985 ± 0.289 | - | 0.002 ± 0.000 | 11.069 ± 2.867 | 8.995 ± 0.368 | 11.156 ± 0.617 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 9.224 ± 1.115 | - | 0.002 ± 0.001 | 10.921 ± 0.663 | 9.119 ± 0.063 | 9.223 ± 0.774 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 37.354 ± 3.714 | - | 0.002 ± 0.001 | 40.767 ± 3.842 | 49.787 ± 3.883 | 225.484 ± 177.667 | 25.487 ± 4.861 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 17.941 ± 0.822 | - | 0.002 ± 0.007 | 46.162 ± 5.814 | 27.794 ± 1.013 | 207.107 ± 67.182 | 15.555 ± 2.485 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 7.300 ± 0.430 | - | 6.841 ± 0.050 | 8.162 ± 0.886 | 4.685 ± 0.293 | 7.485 ± 0.110 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 5.103 ± 0.664 | - | 5.670 ± 0.106 | 6.552 ± 0.425 | 5.942 ± 0.272 | 7.515 ± 0.159 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 6.878 ± 0.139 | - | 7.752 ± 0.448 | 8.669 ± 0.319 | 5.702 ± 0.551 | 9.034 ± 1.704 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 4.724 ± 0.023 | - | 4.793 ± 0.055 | 5.612 ± 0.546 | 5.376 ± 0.141 | 9.428 ± 1.768 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.001 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.001 ± 0.001 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 3.994875 | 249679.69 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.427792 | 276737.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 5.048542 | 315533.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 2.948292 | 184268.25 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 5.047083 | 315442.69 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 5.489333 | 343083.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 14.728209 | 920513.06 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 12.157541 | 759846.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 12.769750 | 798109.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 13.263000 | 828937.50 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.633791 | 289611.94 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 4.273333 | 267083.31 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 3.927667 | 245479.19 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 4.055917 | 253494.81 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 60.733584 | 3795849.00 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 52.521250 | 3282578.12 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 51.557875 | 3222367.19 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 58.694958 | 3668434.88 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 5.761167 | 360072.94 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 4.260333 | 266270.81 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 3.959917 | 247494.81 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 4.666875 | 291679.69 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 58.588917 | 3661807.31 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 51.024459 | 3189028.69 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 51.623667 | 3226479.19 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 52.734875 | 3295929.69 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 20.559792 | 1284987.00 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 21.175542 | 1323471.38 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 192.864458 | 12054028.62 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 221.597875 | 13849867.19 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 20.640833 | 1290052.06 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 21.723208 | 1357700.50 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 207.780542 | 12986283.88 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 209.470584 | 13091911.50 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 21.457750 | 1341109.38 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 21.105875 | 1319117.19 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 173.979541 | 10873721.31 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 194.497417 | 12156088.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.666750 | 1354171.88 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 21.359208 | 1334950.50 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 178.792542 | 11174533.88 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 180.631000 | 11289437.50 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 1.960125 | 30626.95 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.116000 | 48687.50 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 10.353833 | 161778.64 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.831459 | 169241.55 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.366625 | 420828.12 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.667292 | 833411.50 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 21.334416 | 2666802.00 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 21.464750 | 2683093.75 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.914000 | 45531.25 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.068084 | 47938.81 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.977750 | 171527.34 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.429708 | 162964.19 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.221792 | 277724.00 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.541250 | 442656.25 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 10.977459 | 1372182.38 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.858583 | 1357322.88 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.723834 | 167559.91 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.132208 | 80190.75 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.633417 | 72397.14 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.662708 | 72854.81 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.343000 | 2167875.00 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.742084 | 1342760.50 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.788125 | 1223515.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.257750 | 1157218.75 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.389834 | 131091.16 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.868917 | 60451.83 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.013292 | 47082.69 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.996833 | 46825.52 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.838375 | 1229796.88 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.381083 | 672635.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.969208 | 621151.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.215542 | 651942.75 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.992709 | 78011.08 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.624667 | 275385.42 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 15.808708 | 247011.06 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 17.593125 | 274892.58 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.688625 | 836078.12 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 38.670500 | 4833812.50 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 31.424708 | 3928088.50 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 31.605625 | 3950703.12 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.325333 | 83208.33 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.251709 | 175807.95 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.785500 | 152898.44 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.497459 | 164022.80 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.731500 | 466437.50 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 18.316792 | 2289599.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.276875 | 2034609.38 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.775041 | 2096880.13 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.800667 | 75010.42 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.067542 | 47930.34 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.628084 | 41063.81 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.871375 | 60490.23 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.682167 | 585270.88 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.639708 | 579963.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.965959 | 495744.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.168500 | 521062.50 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.487125 | 85736.33 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.256708 | 50886.06 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.212000 | 50187.50 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.320917 | 67514.33 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.904166 | 488020.75 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.658958 | 332369.75 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.665042 | 833130.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.484917 | 685614.62 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.550167 | 102346.36 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.761584 | 90024.75 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.238708 | 81854.81 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.240375 | 97505.86 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.855000 | 1231875.00 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.539917 | 1442489.62 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 8.492166 | 1061520.75 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 8.769083 | 1096135.38 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.704458 | 120382.16 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.930625 | 61416.02 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.956667 | 93072.92 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.636708 | 103698.56 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.882458 | 735307.25 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.493042 | 686630.25 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.423500 | 927937.50 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.108458 | 1263557.25 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.986292 | 109160.81 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.737833 | 120903.64 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.433167 | 131768.23 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.260625 | 1532578.12 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.499334 | 1687416.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.988333 | 1873541.62 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.287583 | 129493.48 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.598250 | 118722.66 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.039625 | 141244.14 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.158500 | 894812.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.920709 | 1240088.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 12.250500 | 1531312.50 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.994083 | 46782.55 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.439792 | 6871.75 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.711833 | 11122.39 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.077291 | 16832.67 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.639834 | 204979.25 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.593834 | 74229.25 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.831334 | 103916.75 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.802209 | 100276.12 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.773083 | 58954.42 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.182875 | 34107.42 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.742542 | 11602.22 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.095084 | 17110.69 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.231625 | 153953.12 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.520041 | 65005.12 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.687375 | 85921.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.853416 | 106677.00 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.976083 | 62126.30 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.789541 | 27961.58 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.382584 | 21602.88 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.182166 | 34096.34 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.812542 | 601567.75 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.067500 | 383437.50 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.262041 | 282755.12 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.305416 | 413177.00 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.919708 | 76870.44 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.932541 | 108320.95 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.590125 | 24845.70 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.418334 | 37786.47 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.005291 | 375661.38 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.700042 | 337505.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.993750 | 374218.75 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.714167 | 464270.88 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.847917 | 75748.70 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.544958 | 71014.97 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.938959 | 45921.23 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.856250 | 44628.91 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.844209 | 730526.12 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.003750 | 1125468.75 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.632709 | 704088.62 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.024083 | 753010.38 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.209875 | 81404.30 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.675917 | 73061.20 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.854917 | 28983.08 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.912083 | 29876.30 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.649709 | 456213.62 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.080041 | 635005.12 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.110542 | 388817.75 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.139167 | 392395.88 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.215083 | 81485.67 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.941792 | 77215.50 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.956792 | 46199.88 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.986625 | 46666.02 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.334834 | 791854.25 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.711334 | 1213916.75 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.879583 | 734947.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.880917 | 735114.62 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.208083 | 65751.30 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.767458 | 74491.53 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.889500 | 29523.44 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.913375 | 29896.48 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.636333 | 454541.62 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.996458 | 624557.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.083708 | 510463.50 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.947833 | 493479.12 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.185125 | 96642.58 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.810042 | 90781.91 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.350833 | 83606.77 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.087709 | 95120.45 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.572083 | 1071510.38 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.519750 | 1689968.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 8.425042 | 1053130.25 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.692250 | 1211531.25 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.288250 | 129503.91 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.342250 | 67847.66 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.494542 | 85852.22 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.528250 | 102003.91 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.237167 | 654645.88 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.578916 | 822364.50 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.373625 | 921703.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.136875 | 1267109.38 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 17.312125 | 270501.95 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 16.436708 | 256823.56 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.967375 | 202615.23 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.973042 | 202703.78 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 30.545250 | 3818156.25 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 33.229458 | 4153682.25 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.901500 | 3237687.50 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 25.987959 | 3248494.88 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.570583 | 180790.36 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.719583 | 167493.48 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.464500 | 132257.81 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.696250 | 135878.91 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 17.175416 | 2146927.00 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 17.896083 | 2237010.38 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 13.604042 | 1700505.25 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 13.789541 | 1723692.62 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.445583 | 85087.23 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.790791 | 121731.11 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 33.193417 | 518647.14 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 38.177583 | 596524.73 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.947917 | 868489.62 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 18.352459 | 2294057.38 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 69.840292 | 8730036.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 80.008042 | 10001005.25 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.555042 | 86797.53 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.553834 | 133653.66 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 37.531250 | 586425.78 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 46.630791 | 728606.11 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.003125 | 625390.62 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.677416 | 1084677.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 52.827750 | 6603468.75 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 61.759625 | 7719953.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 4.073416 | 254588.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 2.792000 | 174500.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 41.400916 | 2587557.25 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 34.653834 | 2165864.62 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 4.514209 | 282138.06 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3.155458 | 197216.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 38.976834 | 2436052.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 35.330208 | 2208138.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012750 | 796.88 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001333 | 83.31 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.012791 | 799.44 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001292 | 80.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.014959 | 934.94 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.006500 | 406.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.011583 | 723.94 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.019458 | 1216.12 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000250 | 15.62 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.010291 | 643.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.002334 | 145.88 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.010875 | 679.69 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.001541 | 96.31 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.011000 | 687.50 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001958 | 122.38 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.010958 | 684.88 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002084 | 130.25 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 25.0x faster than the slowest successful cell (`julia-base`, 64.290 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 16.9x faster than the slowest successful cell (`julia-base`, 64.290 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 17.9x faster than the slowest successful cell (`julia-base`, 64.290 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`julia-base`, 64.290 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 22.7x faster than the slowest successful cell (`julia-base`, 434.973 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 43.8x faster than the slowest successful cell (`julia-base`, 434.973 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 36.0x faster than the slowest successful cell (`julia-base`, 434.973 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 35.5x faster than the slowest successful cell (`julia-base`, 434.973 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`julia-base`, 12.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 28.4x faster than the slowest successful cell (`julia-base`, 166.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 23.7x faster than the slowest successful cell (`julia-base`, 166.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 20.9x faster than the slowest successful cell (`julia-base`, 166.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 21.2x faster than the slowest successful cell (`julia-base`, 166.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 60.0x faster than the slowest successful cell (`julia-base`, 158.958 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 33.8x faster than the slowest successful cell (`julia-base`, 158.958 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 28.4x faster than the slowest successful cell (`julia-base`, 158.958 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 28.2x faster than the slowest successful cell (`julia-base`, 158.958 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 88.7x faster than the slowest successful cell (`julia-base`, 578.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 91.6x faster than the slowest successful cell (`julia-base`, 578.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 74.0x faster than the slowest successful cell (`julia-base`, 578.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 80.1x faster than the slowest successful cell (`julia-base`, 578.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 21.1x faster than the slowest successful cell (`pytorch-cpu`, 15.388 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 15.388 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 15.388 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 19.7x faster than the slowest successful cell (`pytorch-cpu`, 10.947 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 12.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 12.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 12.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 10.428 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 650.3x faster than the slowest successful cell (`julia-base`, 1.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 670.3x faster than the slowest successful cell (`julia-base`, 1.452 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 23.3x faster than the slowest successful cell (`jax-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 31.2x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 19.9x faster than the slowest successful cell (`jax-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 40.8x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 26.8x faster than the slowest successful cell (`jax-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 33.4x faster than the slowest successful cell (`pytorch-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 31.2x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 41.2x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 25.0x faster than the slowest successful cell (`pytorch-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 17.4x faster than the slowest successful cell (`pytorch-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 33.7x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 32.0x faster than the slowest successful cell (`pytorch-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 23.9x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 20.4x faster than the slowest successful cell (`pytorch-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 20.6x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 21.3x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 20.2x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 17.9x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 16.5x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 23.8x faster than the slowest successful cell (`jax-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x2x2,rhs=1`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`tenferro-trace`, 0.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 31.2x faster than the slowest successful cell (`julia-base`, 352.729 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 21.2x faster than the slowest successful cell (`julia-base`, 352.729 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 38.6x faster than the slowest successful cell (`julia-base`, 352.729 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 37.0x faster than the slowest successful cell (`julia-base`, 352.729 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 40.5x faster than the slowest successful cell (`julia-base`, 231.453 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 42.1x faster than the slowest successful cell (`julia-base`, 231.453 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 36.7x faster than the slowest successful cell (`julia-base`, 231.453 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 36.7x faster than the slowest successful cell (`julia-base`, 231.453 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 42.8x faster than the slowest successful cell (`julia-base`, 307.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 50.0x faster than the slowest successful cell (`julia-base`, 307.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 49.8x faster than the slowest successful cell (`julia-base`, 307.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 47.9x faster than the slowest successful cell (`julia-base`, 307.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 35.0x faster than the slowest successful cell (`julia-base`, 247.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 31.3x faster than the slowest successful cell (`julia-base`, 247.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 36.9x faster than the slowest successful cell (`julia-base`, 247.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 36.7x faster than the slowest successful cell (`julia-base`, 247.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 27.7x faster than the slowest successful cell (`julia-base`, 624.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 34.5x faster than the slowest successful cell (`julia-base`, 624.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 35.1x faster than the slowest successful cell (`julia-base`, 624.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 35.1x faster than the slowest successful cell (`julia-base`, 624.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 46.1x faster than the slowest successful cell (`julia-base`, 336.695 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 38.2x faster than the slowest successful cell (`julia-base`, 336.695 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 27.3x faster than the slowest successful cell (`julia-base`, 336.695 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 31.6x faster than the slowest successful cell (`julia-base`, 336.695 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 31.6x faster than the slowest successful cell (`julia-base`, 134.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 12.6x faster than the slowest successful cell (`julia-base`, 134.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`julia-base`, 134.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 25.4x faster than the slowest successful cell (`julia-base`, 255.526 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 30.4x faster than the slowest successful cell (`julia-base`, 255.526 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 26.2x faster than the slowest successful cell (`julia-base`, 255.526 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 25.0x faster than the slowest successful cell (`julia-base`, 255.526 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 5.530 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`julia-base`, 8.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`julia-base`, 8.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 93.9x faster than the slowest successful cell (`julia-base`, 966.334 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 106.6x faster than the slowest successful cell (`julia-base`, 966.334 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 65.6x faster than the slowest successful cell (`julia-base`, 966.334 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 65.2x faster than the slowest successful cell (`julia-base`, 966.334 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 81.2x faster than the slowest successful cell (`julia-base`, 941.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 90.7x faster than the slowest successful cell (`julia-base`, 941.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 74.0x faster than the slowest successful cell (`julia-base`, 941.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 77.3x faster than the slowest successful cell (`julia-base`, 941.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 38.0x faster than the slowest successful cell (`julia-base`, 229.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 41.7x faster than the slowest successful cell (`julia-base`, 229.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 24.4x faster than the slowest successful cell (`julia-base`, 229.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 29.6x faster than the slowest successful cell (`julia-base`, 229.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 52.4x faster than the slowest successful cell (`julia-base`, 1636.550 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 55.1x faster than the slowest successful cell (`julia-base`, 1636.550 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 51.1x faster than the slowest successful cell (`julia-base`, 1636.550 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 54.3x faster than the slowest successful cell (`julia-base`, 1636.550 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 11.5x faster than the slowest successful cell (`tenferro-eager`, 10.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 7.720 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 6102.1x faster than the slowest successful cell (`julia-base`, 12.461 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 5123.7x faster than the slowest successful cell (`julia-base`, 12.594 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 5466.0x faster than the slowest successful cell (`julia-base`, 11.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 4853.7x faster than the slowest successful cell (`pytorch-cpu`, 10.921 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 100215.3x faster than the slowest successful cell (`julia-base`, 225.484 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `strided-jl` is 13.3x faster than the slowest successful cell (`julia-base`, 207.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`julia-base`, 207.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 85687.5x faster than the slowest successful cell (`julia-base`, 207.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 187.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 243.2x faster than the slowest successful cell (`tenferro-direct`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 40.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 136.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 137.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 137.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 7738.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 77.4x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 7738.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 19.3x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 131457.1 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1314.6x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 109208.9 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 273.0x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 131521.5 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1315.2x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 119304.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 298.3x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 59652.3 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 596.5x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 55530.7 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 138.8x faster than the bound.

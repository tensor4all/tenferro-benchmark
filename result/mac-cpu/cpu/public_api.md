# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260923_181304/run.yaml`
- Timestamp: `20260923_181304`

Latest run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260923_181304`.

- tenferro-rs commit: `412852a8cfb13b14246443fc09b1c0c34f5ce1cd`

## CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260923_181304/run_t1.yaml`
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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`
- JULIA_NUM_THREADS: `1`

### Threads: 4

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260923_181304/run_t4.yaml`
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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4`
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

- CSV: `data/results/mac-cpu/cpu/public_api/20260923_181304/cpu_public_api_t1_20260923_181304.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260923_181304/cpu_public_api_t4_20260923_181304.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260923_181304/cpu_public_api_20260923_181304.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.608 ± 0.155 | - | 2.655 ± 0.112 | 2.756 ± 0.041 | 0.957 ± 0.013 | 2.816 ± 0.328 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.407 ± 0.036 | - | 2.446 ± 0.041 | 2.472 ± 0.036 | 0.763 ± 0.017 | 1.316 ± 0.410 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 4.368 ± 0.227 | - | 4.381 ± 0.122 | 4.515 ± 0.026 | 4.523 ± 0.222 | 4.429 ± 0.033 | 4.431 ± 0.049 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 1.673 ± 0.079 | - | 1.615 ± 0.092 | 1.577 ± 0.030 | 1.867 ± 0.236 | 4.444 ± 0.032 | 7.844 ± 7.061 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 5.660 ± 0.059 | - | 5.748 ± 0.119 | 7.982 ± 0.109 | 23.248 ± 0.587 | 18.639 ± 0.056 | 18.289 ± 0.159 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 1.583 ± 0.045 | - | 1.608 ± 0.059 | 2.789 ± 0.070 | 7.752 ± 0.046 | 19.446 ± 0.761 | 6.817 ± 2.624 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 5.756 ± 0.911 | - | 5.866 ± 0.201 | 5.564 ± 0.235 | 36.543 ± 0.237 | 40.099 ± 0.250 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 5.835 ± 0.847 | - | 5.743 ± 0.108 | 5.478 ± 0.227 | 10.045 ± 0.307 | 11.459 ± 0.631 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 5.833 ± 0.197 | - | unsupported | 5.646 ± 0.198 | 36.784 ± 0.327 | 40.123 ± 0.664 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 5.802 ± 0.194 | - | unsupported | 5.566 ± 0.138 | 10.406 ± 0.812 | 11.305 ± 0.391 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 5.933 ± 0.157 | - | 6.105 ± 0.254 | 5.859 ± 0.046 | 9.952 ± 0.082 | 5.161 ± 0.042 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 5.846 ± 0.057 | - | 5.934 ± 0.031 | 5.862 ± 0.065 | 10.143 ± 0.190 | 5.135 ± 0.331 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 20.711 ± 0.269 | - | 20.404 ± 0.651 | 20.466 ± 0.443 | 23.956 ± 0.053 | 24.819 ± 0.309 | 24.678 ± 1.125 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 5.490 ± 0.705 | - | 5.276 ± 0.047 | 5.950 ± 0.023 | 6.546 ± 0.222 | 26.204 ± 2.894 | 6.866 ± 2.245 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 23.828 ± 0.247 | - | 23.817 ± 0.212 | 25.656 ± 0.340 | 49.574 ± 0.168 | 35.233 ± 0.124 | 34.402 ± 0.101 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 6.503 ± 0.201 | - | 6.477 ± 0.323 | 6.963 ± 0.258 | 13.675 ± 0.441 | 35.395 ± 1.078 | 9.572 ± 1.484 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 4.017 ± 0.250 | - | 3.982 ± 0.125 | 4.305 ± 0.104 | 3.591 ± 0.129 | 3.647 ± 0.047 | 3.452 ± 0.026 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 1.395 ± 0.080 | - | 1.390 ± 0.037 | 1.449 ± 0.026 | 1.435 ± 0.229 | 3.668 ± 0.114 | 4.889 ± 8.265 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 4.873 ± 0.231 | - | 4.035 ± 0.082 | 3.644 ± 0.027 | 1.950 ± 0.049 | 5.570 ± 0.076 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.956 ± 0.066 | - | 1.112 ± 0.015 | 0.941 ± 0.007 | 0.746 ± 0.038 | 5.885 ± 0.012 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.781 ± 0.027 | - | 4.758 ± 0.034 | 5.027 ± 0.043 | 3.650 ± 0.020 | 6.008 ± 0.136 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.695 ± 0.026 | - | 4.801 ± 0.028 | 4.857 ± 0.042 | 3.600 ± 0.114 | 2.794 ± 0.152 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.249 ± 0.314 | - | 3.314 ± 0.181 | 3.143 ± 0.049 | 1.433 ± 0.031 | 3.679 ± 0.037 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.285 ± 0.048 | - | 3.247 ± 0.094 | 3.278 ± 0.093 | 1.310 ± 0.015 | 1.522 ± 0.101 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.842 ± 0.042 | - | 3.985 ± 0.039 | 3.834 ± 0.073 | 2.785 ± 0.035 | 4.437 ± 0.066 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.814 ± 0.065 | - | 3.962 ± 0.036 | 3.836 ± 0.044 | 2.699 ± 0.479 | 2.878 ± 0.100 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 5.498 ± 0.198 | - | 5.798 ± 0.257 | 5.401 ± 0.183 | 36.679 ± 0.282 | 39.640 ± 0.111 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 5.557 ± 0.244 | - | 5.852 ± 0.086 | 5.600 ± 0.203 | 10.924 ± 0.566 | 10.771 ± 0.639 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 4.805 ± 1.006 | - | unsupported | 4.896 ± 0.553 | 33.084 ± 0.912 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 4.977 ± 1.387 | - | unsupported | 4.459 ± 0.031 | 9.173 ± 0.508 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 4.297 ± 0.165 | - | 4.336 ± 0.114 | 4.313 ± 0.080 | 9.312 ± 1.805 | 4.355 ± 0.135 | 7.833 ± 0.131 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 1.583 ± 0.096 | - | 1.636 ± 0.073 | 1.609 ± 0.112 | 2.502 ± 0.050 | 4.455 ± 0.116 | 2.775 ± 3.031 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 6.291 ± 0.149 | - | 6.326 ± 0.117 | 6.530 ± 0.149 | 11.376 ± 0.093 | 6.405 ± 0.209 | 8.277 ± 0.565 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 4.931 ± 1.340 | - | 5.317 ± 1.490 | 2.512 ± 0.153 | 3.524 ± 0.163 | 6.591 ± 0.064 | 5.464 ± 9.201 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 16.603 ± 0.125 | - | 40.923 ± 1.157 | 18.018 ± 0.373 | 17.836 ± 0.048 | 20.982 ± 0.137 | 21.622 ± 0.811 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 4.866 ± 0.238 | - | 15.976 ± 1.091 | 5.200 ± 0.528 | 4.675 ± 0.181 | 21.295 ± 0.270 | 5.876 ± 0.383 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 4.937 ± 1.956 | - | 4.644 ± 0.062 | 2.199 ± 0.095 | 3.801 ± 0.022 | 2.262 ± 0.094 | 3.113 ± 0.158 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 1.611 ± 0.034 | - | 1.625 ± 0.048 | 0.981 ± 0.109 | 1.811 ± 0.023 | 2.282 ± 0.127 | 1.374 ± 0.418 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 4.920 ± 0.029 | - | 4.925 ± 0.025 | 8.605 ± 0.124 | 8.654 ± 0.032 | 5.741 ± 0.065 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 2.081 ± 0.059 | - | 2.116 ± 0.093 | 2.919 ± 0.068 | 2.699 ± 0.048 | 5.774 ± 0.043 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 14.729 ± 2.296 | - | 15.291 ± 1.463 | 17.230 ± 0.466 | 12.724 ± 1.061 | 20.672 ± 0.378 | 20.554 ± 0.224 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 4.215 ± 0.804 | - | 3.959 ± 0.807 | 4.545 ± 0.285 | 3.343 ± 0.035 | 20.442 ± 0.285 | 5.597 ± 0.284 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 6.321 ± 0.128 | - | 6.315 ± 0.132 | 6.843 ± 0.558 | 11.439 ± 0.083 | 6.345 ± 0.144 | 8.595 ± 0.466 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 4.449 ± 0.054 | - | 4.491 ± 0.081 | 2.513 ± 0.121 | 3.549 ± 0.147 | 6.572 ± 0.140 | 3.239 ± 2.655 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 15.763 ± 0.935 | - | 15.995 ± 0.813 | 12.178 ± 0.056 | 10.298 ± 0.056 | 16.547 ± 1.729 | 16.692 ± 2.506 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 4.262 ± 0.049 | - | 4.376 ± 0.146 | 4.415 ± 0.044 | 2.693 ± 0.221 | 17.199 ± 0.532 | 5.942 ± 2.225 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 8.358 ± 0.612 | - | 8.428 ± 0.242 | 14.752 ± 0.218 | 10.892 ± 0.015 | 12.621 ± 0.104 | 12.909 ± 0.335 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.170 ± 0.022 | - | 2.249 ± 0.064 | 3.922 ± 0.202 | 2.900 ± 0.117 | 12.877 ± 0.538 | 3.483 ± 0.708 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 15.120 ± 0.120 | - | 15.230 ± 0.124 | 20.659 ± 0.032 | 14.073 ± 0.045 | 23.001 ± 1.455 | 23.496 ± 1.284 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 4.224 ± 0.288 | - | 4.173 ± 0.143 | 5.391 ± 1.076 | 3.968 ± 0.190 | 22.534 ± 0.309 | 6.256 ± 0.921 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 10.121 ± 0.030 | - | 10.189 ± 0.128 | 11.120 ± 0.137 | 9.892 ± 0.020 | 12.412 ± 0.164 | 12.284 ± 0.342 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 2.755 ± 0.051 | - | 2.791 ± 0.011 | 2.907 ± 0.895 | 2.731 ± 0.109 | 12.188 ± 0.149 | 3.208 ± 0.763 |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 6.477 ± 0.018 | - | 6.503 ± 0.023 | 6.422 ± 0.151 | 11.263 ± 0.127 | 6.363 ± 0.168 | 8.560 ± 0.633 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 2.519 ± 0.124 | - | 2.483 ± 0.111 | 2.534 ± 0.095 | 3.459 ± 0.139 | 6.569 ± 0.127 | 3.238 ± 2.451 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 6.523 ± 0.029 | - | 6.523 ± 0.030 | 6.407 ± 0.088 | 11.252 ± 0.131 | 6.487 ± 0.147 | 8.430 ± 0.491 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 2.519 ± 0.029 | - | 2.579 ± 0.106 | 2.563 ± 0.207 | 3.458 ± 0.101 | 6.570 ± 0.050 | 3.241 ± 2.551 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 6.323 ± 0.133 | - | 6.341 ± 0.110 | 6.466 ± 0.207 | 11.354 ± 0.111 | 6.438 ± 0.161 | 8.349 ± 0.328 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 4.405 ± 0.070 | - | 4.418 ± 0.025 | 2.568 ± 0.572 | 3.434 ± 0.100 | 6.596 ± 0.150 | 3.563 ± 2.641 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 4.257 ± 0.102 | - | 4.273 ± 0.080 | 4.370 ± 0.078 | 9.247 ± 1.366 | 4.344 ± 0.174 | 7.809 ± 0.123 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 2.670 ± 0.020 | - | 1.584 ± 0.051 | 1.550 ± 0.086 | 2.545 ± 0.075 | 4.518 ± 0.180 | 3.121 ± 2.957 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 17.655 ± 0.020 | - | 17.696 ± 0.047 | 38.587 ± 0.955 | 17.538 ± 0.032 | 30.871 ± 1.526 | 31.777 ± 1.265 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 4.879 ± 0.647 | - | 4.918 ± 0.107 | 9.670 ± 0.144 | 5.054 ± 0.503 | 31.483 ± 1.055 | 8.743 ± 0.645 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 4.101 ± 0.396 | - | 4.075 ± 0.018 | 7.399 ± 0.051 | 9.675 ± 0.171 | 2.946 ± 0.054 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 1.349 ± 0.039 | - | 1.568 ± 0.046 | 1.984 ± 0.014 | 2.992 ± 0.020 | 3.075 ± 0.114 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 0.878 ± 0.261 | - | 0.718 ± 0.154 | 10.832 ± 0.578 | 0.924 ± 0.170 | 0.372 ± 0.025 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.294 ± 0.028 | - | 0.281 ± 0.023 | 6.176 ± 0.479 | 0.343 ± 0.075 | 0.384 ± 0.028 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.681 ± 0.043 | - | 0.659 ± 0.027 | 0.978 ± 0.055 | 0.367 ± 0.007 | 0.494 ± 0.007 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.253 ± 0.023 | - | 0.261 ± 0.030 | 0.276 ± 0.004 | 0.260 ± 0.043 | 0.484 ± 0.007 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 4.114 ± 0.257 | - | 4.110 ± 0.044 | 7.455 ± 0.094 | 9.767 ± 0.047 | 2.974 ± 0.075 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 1.354 ± 0.034 | - | 1.531 ± 0.021 | 1.985 ± 0.010 | 3.001 ± 0.024 | 2.997 ± 0.086 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 0.530 ± 0.007 | - | 0.532 ± 0.005 | 12.944 ± 0.834 | 0.896 ± 0.072 | 0.363 ± 0.008 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.194 ± 0.013 | - | 0.166 ± 0.016 | 6.241 ± 0.298 | 0.366 ± 0.096 | 0.361 ± 0.004 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 2.595 ± 0.208 | - | 2.478 ± 0.042 | 3.862 ± 0.083 | 1.584 ± 0.033 | 1.914 ± 0.017 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 1.047 ± 0.060 | - | 0.939 ± 0.060 | 1.069 ± 0.021 | 0.934 ± 0.063 | 1.925 ± 0.029 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 3.180 ± 0.074 | - | 3.191 ± 0.024 | 3.208 ± 0.025 | 17.397 ± 0.512 | 3.220 ± 0.255 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 1.351 ± 0.344 | - | 1.536 ± 0.040 | 1.551 ± 0.023 | 5.390 ± 0.016 | 3.207 ± 0.120 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.362 ± 0.009 | - | 0.379 ± 0.009 | 1.833 ± 0.301 | 1.019 ± 0.085 | 0.359 ± 0.011 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.139 ± 0.010 | - | 0.138 ± 0.007 | 0.764 ± 0.063 | 0.365 ± 0.073 | 0.377 ± 0.006 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.459 ± 0.026 | - | 0.475 ± 0.012 | 0.344 ± 0.007 | 0.378 ± 0.006 | 0.487 ± 0.006 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.181 ± 0.014 | - | 0.198 ± 0.014 | 0.157 ± 0.008 | 0.212 ± 0.024 | 0.486 ± 0.007 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 3.203 ± 0.182 | - | 3.079 ± 0.076 | 2.950 ± 0.046 | 13.633 ± 0.306 | 3.007 ± 0.049 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 1.635 ± 0.167 | - | 1.700 ± 0.101 | 1.559 ± 0.032 | 4.532 ± 0.017 | 3.096 ± 0.045 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.364 ± 0.018 | - | 0.369 ± 0.004 | 1.734 ± 0.210 | 0.829 ± 0.036 | 0.375 ± 0.016 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.170 ± 0.010 | - | 0.136 ± 0.012 | 0.702 ± 0.047 | 0.371 ± 0.061 | 0.371 ± 0.009 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.447 ± 0.009 | - | 0.460 ± 0.006 | 0.320 ± 0.011 | 0.385 ± 0.004 | 0.486 ± 0.020 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.167 ± 0.004 | - | 0.196 ± 0.014 | 0.154 ± 0.005 | 0.259 ± 0.035 | 0.483 ± 0.015 | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 16.092 ± 0.229 | - | 15.958 ± 0.800 | 19.302 ± 0.253 | 14.975 ± 0.334 | 13.997 ± 3.081 | 13.341 ± 0.846 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 4.303 ± 0.514 | - | 4.083 ± 0.078 | 6.375 ± 0.098 | 4.415 ± 0.024 | 13.152 ± 1.830 | 4.299 ± 1.631 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 11.295 ± 0.038 | - | 11.298 ± 0.038 | 12.112 ± 0.075 | 22.669 ± 0.153 | 22.524 ± 0.175 | 22.827 ± 0.236 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 3.080 ± 0.033 | - | 3.088 ± 0.026 | 3.055 ± 0.015 | 6.071 ± 0.042 | 23.365 ± 0.314 | 6.444 ± 2.550 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 22.574 ± 0.208 | - | 22.573 ± 0.319 | 9.138 ± 0.111 | 13.993 ± 0.078 | 7.780 ± 0.091 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 18.891 ± 0.290 | - | 18.913 ± 0.316 | 3.610 ± 0.057 | 4.758 ± 0.151 | 7.834 ± 0.060 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 5.064 ± 0.059 | - | 5.062 ± 0.026 | 5.025 ± 0.049 | 9.840 ± 0.028 | 4.429 ± 0.104 | 11.744 ± 0.160 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 3.223 ± 0.037 | - | 3.132 ± 0.065 | 1.727 ± 0.033 | 3.332 ± 0.045 | 4.453 ± 0.107 | 3.201 ± 2.926 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 13.511 ± 1.222 | - | 13.077 ± 0.101 | 14.300 ± 0.217 | 11.254 ± 0.052 | 17.494 ± 0.318 | 17.419 ± 0.238 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 3.817 ± 0.251 | - | 3.943 ± 0.121 | 3.882 ± 0.055 | 3.160 ± 0.407 | 17.345 ± 0.501 | 4.830 ± 0.652 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 7.499 ± 0.452 | - | 7.356 ± 0.091 | 7.906 ± 0.141 | 8.445 ± 1.218 | 15.293 ± 0.332 | 15.503 ± 0.182 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 2.027 ± 0.130 | - | 2.045 ± 0.047 | 2.697 ± 0.065 | 2.474 ± 0.009 | 15.619 ± 0.079 | 4.168 ± 3.179 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 6.293 ± 0.112 | - | 6.363 ± 0.194 | 6.528 ± 0.135 | 11.350 ± 0.118 | 6.491 ± 0.134 | 8.649 ± 0.427 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 5.188 ± 0.794 | - | 4.628 ± 0.203 | 2.515 ± 0.051 | 3.441 ± 0.066 | 6.573 ± 0.050 | 3.370 ± 2.307 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 21.149 ± 0.032 | - | 21.137 ± 0.035 | 35.896 ± 0.509 | 8.374 ± 0.036 | 23.354 ± 0.169 | 23.631 ± 0.249 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 5.353 ± 0.037 | - | 5.375 ± 0.071 | 9.372 ± 0.229 | 2.309 ± 0.062 | 24.193 ± 0.649 | 6.631 ± 0.399 |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.279 ± 0.022 | - | 0.518 ± 0.006 | 0.215 ± 0.007 | 0.221 ± 0.005 | 0.297 ± 0.014 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.101 ± 0.014 | - | 0.292 ± 0.022 | 0.121 ± 0.008 | 0.223 ± 0.003 | 0.299 ± 0.090 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.224 ± 0.004 | - | 0.780 ± 0.058 | 0.267 ± 0.013 | 0.262 ± 0.005 | 0.288 ± 0.009 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.155 ± 0.016 | - | 0.483 ± 0.023 | 0.101 ± 0.005 | 0.121 ± 0.007 | 0.292 ± 0.005 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.384 ± 0.034 | - | unsupported | 0.415 ± 0.009 | 0.366 ± 0.006 | 0.417 ± 0.086 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.184 ± 0.014 | - | unsupported | 0.201 ± 0.011 | 0.314 ± 0.015 | 0.405 ± 0.006 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.420 ± 0.065 | - | 0.467 ± 0.007 | 0.162 ± 0.006 | 0.176 ± 0.004 | 0.283 ± 0.025 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.276 ± 0.024 | - | 0.325 ± 0.034 | 0.166 ± 0.003 | 0.071 ± 0.013 | 0.271 ± 0.104 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.335 ± 0.006 | - | 0.605 ± 0.068 | 0.400 ± 0.007 | 0.284 ± 0.002 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.174 ± 0.007 | - | 0.361 ± 0.016 | 0.184 ± 0.014 | 0.127 ± 0.011 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.260 ± 0.008 | - | 0.525 ± 0.025 | 0.329 ± 0.061 | 0.268 ± 0.005 | 0.579 ± 0.184 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.100 ± 0.007 | - | 0.277 ± 0.015 | 0.119 ± 0.007 | 0.111 ± 0.010 | 0.641 ± 0.012 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.591 ± 0.202 | - | 0.574 ± 0.042 | 0.235 ± 0.005 | 0.223 ± 0.008 | 0.237 ± 0.009 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.487 ± 0.008 | - | 0.531 ± 0.036 | 0.230 ± 0.003 | 0.216 ± 0.006 | 0.219 ± 0.142 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 0.577 ± 0.025 | - | 1.135 ± 0.096 | 0.422 ± 0.013 | 0.413 ± 0.012 | 0.686 ± 0.056 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.210 ± 0.009 | - | 0.586 ± 0.028 | 0.216 ± 0.007 | 0.297 ± 0.009 | 0.669 ± 0.025 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.757 ± 0.064 | - | unsupported | 0.465 ± 0.018 | 0.558 ± 0.008 | 1.084 ± 0.032 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.069 ± 0.001 | - | unsupported | 0.035 ± 0.002 | 0.065 ± 0.004 | 0.038 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.139 ± 0.003 | - | unsupported | 0.061 ± 0.001 | 0.092 ± 0.002 | 0.088 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.300 ± 0.016 | - | unsupported | 0.156 ± 0.002 | 0.193 ± 0.008 | 0.244 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.705 ± 0.090 | - | unsupported | 0.201 ± 0.006 | 0.302 ± 0.010 | 1.058 ± 0.080 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.067 ± 0.001 | - | unsupported | 0.104 ± 0.021 | 0.073 ± 0.016 | 0.043 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.135 ± 0.000 | - | unsupported | 0.087 ± 0.012 | 0.106 ± 0.007 | 0.072 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.300 ± 0.005 | - | unsupported | 0.115 ± 0.009 | 0.102 ± 0.015 | 0.245 ± 0.014 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.178 ± 0.004 | - | unsupported | 0.124 ± 0.001 | 0.272 ± 0.012 | 0.190 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.044 ± 0.000 | - | unsupported | 0.024 ± 0.001 | 0.042 ± 0.007 | 0.026 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.050 ± 0.000 | - | unsupported | 0.028 ± 0.000 | 0.051 ± 0.007 | 0.045 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.071 ± 0.007 | - | unsupported | 0.051 ± 0.000 | 0.089 ± 0.004 | 0.094 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.177 ± 0.003 | - | unsupported | 0.140 ± 0.006 | 0.237 ± 0.023 | 0.181 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.048 ± 0.000 | - | unsupported | 0.032 ± 0.000 | 0.045 ± 0.008 | 0.026 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.052 ± 0.000 | - | unsupported | 0.039 ± 0.000 | 0.052 ± 0.016 | 0.043 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.070 ± 0.000 | - | unsupported | 0.048 ± 0.001 | 0.085 ± 0.003 | 0.088 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.057 ± 0.000 | - | 0.108 ± 0.003 | 0.046 ± 0.003 | 0.109 ± 0.010 | 0.120 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.026 ± 0.000 | - | 0.029 ± 0.004 | 0.012 ± 0.001 | 0.014 ± 0.001 | 0.021 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.025 ± 0.000 | - | 0.033 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.032 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.030 ± 0.000 | - | 0.049 ± 0.004 | 0.019 ± 0.000 | 0.033 ± 0.001 | 0.059 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.057 ± 0.005 | - | 0.108 ± 0.010 | 0.048 ± 0.001 | 0.086 ± 0.012 | 0.114 ± 0.006 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.022 ± 0.000 | - | 0.058 ± 0.006 | 0.017 ± 0.001 | 0.014 ± 0.001 | 0.022 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.025 ± 0.000 | - | 0.053 ± 0.007 | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.036 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.028 ± 0.000 | - | 0.069 ± 0.008 | 0.019 ± 0.000 | 0.046 ± 0.003 | 0.061 ± 0.002 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 4.288 ± 0.627 | - | 4.440 ± 0.100 | 14.931 ± 0.303 | 9.004 ± 0.804 | 26.620 ± 3.182 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 4.410 ± 0.832 | - | 4.489 ± 0.144 | 8.304 ± 0.557 | 6.348 ± 0.265 | 10.621 ± 12.365 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 4.078 ± 0.043 | - | 4.086 ± 0.022 | 3.965 ± 0.049 | 4.459 ± 0.610 | 13.826 ± 0.529 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 4.149 ± 0.062 | - | 3.949 ± 0.030 | 3.974 ± 0.028 | 4.047 ± 0.390 | 5.061 ± 0.415 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.430 ± 0.052 | - | 4.405 ± 0.050 | 4.462 ± 0.053 | 5.297 ± 0.028 | 4.125 ± 0.114 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.446 ± 0.079 | - | 4.446 ± 0.042 | 4.462 ± 0.064 | 5.551 ± 0.159 | 4.129 ± 0.033 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 4.597 ± 0.082 | - | 4.591 ± 0.064 | 4.604 ± 0.023 | 4.291 ± 0.073 | 4.506 ± 0.074 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 4.613 ± 0.050 | - | 4.620 ± 0.060 | 4.663 ± 0.084 | 4.425 ± 0.082 | 4.509 ± 0.032 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 14.979 ± 0.042 | - | 15.014 ± 0.058 | 14.833 ± 0.140 | 12.563 ± 0.070 | 11.241 ± 0.147 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 14.081 ± 0.774 | - | 14.276 ± 0.542 | 14.811 ± 0.075 | 12.762 ± 0.507 | 6.492 ± 0.058 | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 6.083 ± 0.041 | - | 5.119 ± 0.062 | 5.529 ± 0.091 | 5.494 ± 0.038 | 20.247 ± 1.077 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 6.046 ± 0.028 | - | 5.277 ± 0.260 | 5.274 ± 0.038 | 5.291 ± 0.095 | 6.660 ± 0.251 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 5.622 ± 0.037 | 5.526 ± 0.036 | 3.317 ± 0.061 | 13.326 ± 0.126 | 5.192 ± 0.025 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 5.682 ± 0.098 | 5.593 ± 0.075 | 3.167 ± 0.033 | 13.186 ± 0.141 | 3.069 ± 0.043 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 3.969 ± 0.052 | - | 4.146 ± 0.031 | 6.486 ± 0.345 | 7.080 ± 0.218 | 16.689 ± 2.909 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 3.985 ± 0.019 | - | 4.173 ± 0.048 | 5.507 ± 0.330 | 5.010 ± 0.178 | 7.720 ± 3.100 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.357 ± 0.004 | - | 0.361 ± 0.001 | 1.239 ± 0.037 | 2.855 ± 0.214 | 4.102 ± 0.166 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.118 ± 0.016 | - | 0.151 ± 0.016 | 1.271 ± 0.067 | 0.849 ± 0.128 | 3.928 ± 0.008 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.577 ± 0.023 | - | 6.473 ± 0.053 | 6.410 ± 0.047 | 7.279 ± 0.059 | 13.588 ± 0.355 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.568 ± 0.068 | - | 6.619 ± 0.031 | 6.489 ± 0.066 | 6.728 ± 0.117 | 9.444 ± 0.128 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 6.587 ± 0.064 | - | 6.479 ± 0.064 | 6.423 ± 0.181 | 7.321 ± 0.289 | 12.925 ± 0.427 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 6.496 ± 0.042 | - | 6.624 ± 0.061 | 6.477 ± 0.054 | 6.643 ± 0.161 | 9.396 ± 0.133 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 4.048 ± 0.017 | - | 4.073 ± 0.022 | 3.969 ± 0.048 | 4.298 ± 0.174 | 13.858 ± 0.545 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 4.139 ± 0.037 | - | 3.928 ± 0.063 | 3.976 ± 0.028 | 4.254 ± 0.191 | 5.154 ± 0.176 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 18.372 ± 0.079 | 18.405 ± 0.118 | 18.451 ± 0.110 | 18.328 ± 0.212 | 45.765 ± 1.656 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 18.787 ± 0.113 | 18.809 ± 0.142 | 18.859 ± 0.112 | 19.074 ± 0.748 | 29.327 ± 0.473 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 5.445 ± 0.537 | - | 7.608 ± 0.269 | 3.915 ± 0.034 | 25.456 ± 1.072 | 21.722 ± 0.235 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 5.881 ± 0.288 | - | 7.120 ± 0.397 | 3.753 ± 0.032 | 12.660 ± 1.074 | 7.337 ± 0.205 | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 6.363 ± 0.144 | - | unsupported | 6.719 ± 0.067 | - | 6.444 ± 1.061 | 8.655 ± 0.483 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 2.526 ± 0.255 | - | unsupported | 2.511 ± 0.079 | - | 6.453 ± 0.036 | 3.218 ± 0.124 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 4.342 ± 0.034 | - | unsupported | 4.601 ± 0.119 | - | 4.683 ± 0.293 | 4.446 ± 0.062 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 1.568 ± 0.057 | - | unsupported | 1.592 ± 0.035 | - | 4.335 ± 0.032 | 1.755 ± 0.291 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 3.785 ± 0.050 | - | unsupported | 4.425 ± 0.116 | - | 3.808 ± 0.042 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 1.557 ± 0.044 | - | unsupported | 1.538 ± 0.013 | - | 3.791 ± 0.030 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 6.390 ± 0.145 | - | unsupported | 6.416 ± 0.083 | - | 6.461 ± 0.089 | 8.995 ± 0.615 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 2.517 ± 0.157 | - | unsupported | 2.477 ± 0.019 | - | 6.450 ± 0.039 | 3.243 ± 0.175 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 4.554 ± 0.143 | - | unsupported | 4.629 ± 0.162 | - | 35.967 ± 0.245 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 4.461 ± 0.079 | - | unsupported | 4.443 ± 0.027 | - | 9.597 ± 0.505 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 4.882 ± 0.131 | - | unsupported | 5.030 ± 0.102 | - | 35.838 ± 0.185 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 4.981 ± 0.045 | - | unsupported | 5.000 ± 0.045 | - | 9.759 ± 0.115 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 6.496 ± 0.038 | - | unsupported | 6.443 ± 0.078 | - | 6.588 ± 0.124 | 8.643 ± 0.369 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 2.453 ± 0.117 | - | unsupported | 2.478 ± 0.020 | - | 6.466 ± 0.029 | 3.182 ± 0.286 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 4.344 ± 0.088 | - | unsupported | 4.448 ± 0.153 | - | 4.700 ± 0.591 | 8.119 ± 0.921 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 1.559 ± 0.031 | - | unsupported | 1.531 ± 0.009 | - | 4.323 ± 0.028 | 2.861 ± 0.692 |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 6.381 ± 0.094 | - | unsupported | 6.704 ± 0.165 | - | 6.958 ± 0.435 | 8.611 ± 0.493 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 2.461 ± 0.112 | - | unsupported | 2.486 ± 0.035 | - | 6.448 ± 0.023 | 3.096 ± 0.297 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 2.019 ± 0.013 | - | 2.587 ± 0.584 | 2.042 ± 0.080 | 1.894 ± 0.050 | 8.404 ± 0.130 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 2.157 ± 0.025 | - | 1.130 ± 0.008 | 1.078 ± 0.014 | 1.099 ± 0.038 | 8.477 ± 0.033 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 3.847 ± 0.166 | - | 8.317 ± 0.319 | 3.529 ± 0.071 | 4.883 ± 0.540 | 3.679 ± 0.198 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 1.338 ± 0.117 | - | 2.910 ± 0.087 | 1.321 ± 0.036 | 1.367 ± 0.062 | 3.669 ± 0.037 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 4.614 ± 0.578 | - | 4.196 ± 0.933 | 3.981 ± 0.067 | 6.193 ± 0.205 | 2.005 ± 0.124 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 2.148 ± 0.015 | - | 2.207 ± 0.049 | 2.163 ± 0.050 | 2.361 ± 0.308 | 2.113 ± 0.077 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 2.187 ± 0.071 | - | 6.651 ± 0.047 | 4.401 ± 0.217 | 3.511 ± 0.025 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 2.203 ± 0.090 | - | 3.735 ± 0.129 | 1.750 ± 0.020 | 1.306 ± 0.094 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 4.148 ± 0.344 | - | 4.400 ± 0.058 | 4.282 ± 0.065 | 3.542 ± 0.050 | 3.997 ± 0.216 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 3.795 ± 0.054 | - | 1.607 ± 0.100 | 1.547 ± 0.061 | 1.660 ± 0.082 | 3.938 ± 0.064 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 28.220 ± 0.464 | - | 23.447 ± 1.412 | 19.864 ± 0.911 | 12.566 ± 0.082 | 51.592 ± 0.283 | 31.177 ± 0.308 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 11.738 ± 0.535 | - | 7.310 ± 0.321 | 18.782 ± 0.400 | 5.361 ± 0.387 | 51.030 ± 2.561 | 17.670 ± 7.131 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 3.363 ± 0.292 | - | 5.702 ± 0.231 | 3.418 ± 0.156 | 2.248 ± 0.035 | 4.410 ± 0.245 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 1.414 ± 0.098 | - | 2.288 ± 0.322 | 1.361 ± 0.024 | 0.942 ± 0.062 | 4.297 ± 0.162 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 3.135 ± 0.194 | - | 5.529 ± 0.043 | 3.200 ± 0.109 | 2.209 ± 0.019 | 4.115 ± 0.256 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 1.322 ± 0.083 | - | 2.743 ± 0.445 | 1.220 ± 0.032 | 0.897 ± 0.070 | 3.986 ± 0.159 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---:|---|---:|---:|---:|
| broadcast_in_dim_view | 1 | PyTorch Python (ms) | 16 | 0.005293 | 330.81 |
| broadcast_in_dim_view | 1 | tenferro-rs direct API (ms) | 16 | 0.001417 | 88.56 |
| broadcast_in_dim_view | 4 | PyTorch Python (ms) | 16 | 0.007542 | 471.38 |
| broadcast_in_dim_view | 4 | tenferro-rs direct API (ms) | 16 | 0.001375 | 85.94 |
| reshape_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| reshape_view | 1 | PyTorch Python (ms) | 16 | 0.005166 | 322.88 |
| reshape_view | 1 | tenferro-rs direct API (ms) | 16 | 0.002500 | 156.25 |
| reshape_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| reshape_view | 4 | PyTorch Python (ms) | 16 | 0.007208 | 450.50 |
| reshape_view | 4 | tenferro-rs direct API (ms) | 16 | 0.002500 | 156.25 |
| slice_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| slice_view | 1 | PyTorch Python (ms) | 16 | 0.004750 | 296.88 |
| slice_view | 1 | tenferro-rs direct API (ms) | 16 | 0.002667 | 166.69 |
| slice_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| slice_view | 4 | PyTorch Python (ms) | 16 | 0.006208 | 388.00 |
| slice_view | 4 | tenferro-rs direct API (ms) | 16 | 0.003166 | 197.88 |
| transpose_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| transpose_view | 1 | PyTorch Python (ms) | 16 | 0.004542 | 283.88 |
| transpose_view | 1 | tenferro-rs direct API (ms) | 16 | 0.001917 | 119.81 |
| transpose_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| transpose_view | 4 | PyTorch Python (ms) | 16 | 0.006333 | 395.81 |
| transpose_view | 4 | tenferro-rs direct API (ms) | 16 | 0.002750 | 171.88 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`julia-base`, 19.446 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`julia-base`, 19.446 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 10.832 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 29.1x faster than the slowest successful cell (`pytorch-cpu`, 10.832 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 10.832 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 10.832 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 18.0x faster than the slowest successful cell (`pytorch-cpu`, 6.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 6.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 21.0x faster than the slowest successful cell (`pytorch-cpu`, 6.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 6.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 12.944 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 35.7x faster than the slowest successful cell (`pytorch-cpu`, 12.944 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 24.4x faster than the slowest successful cell (`pytorch-cpu`, 12.944 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 12.944 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 6.241 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 17.3x faster than the slowest successful cell (`pytorch-cpu`, 6.241 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 32.2x faster than the slowest successful cell (`pytorch-cpu`, 6.241 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 37.5x faster than the slowest successful cell (`pytorch-cpu`, 6.241 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=4, shape=`8388608`): `jax-cpu` is 10.5x faster than the slowest successful cell (`julia-base`, 24.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`julia-base`, 4.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`julia-base`, 4.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 33.3x faster than the slowest successful cell (`julia-base`, 3.928 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 26.0x faster than the slowest successful cell (`julia-base`, 3.928 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 107.7x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 150.0x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 99.0x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 129.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 94.7x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 132.0x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

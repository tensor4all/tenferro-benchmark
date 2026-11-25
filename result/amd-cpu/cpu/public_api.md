# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261125_185603/run.yaml`
- Timestamp: `2026-11-25T18:56:03.902364+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261125_185603`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
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

- CSV: `data/results/amd-cpu/cpu/public_api/20261125_185603/cpu_public_api_t1_20261125_185603.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261125_185603/cpu_public_api_t4_20261125_185603.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.705 ± 0.031 | - | 3.726 ± 0.036 | 4.520 ± 0.041 | 3.940 ± 0.153 | 3.725 ± 2.247 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.535 ± 0.231 | - | 2.555 ± 0.035 | 2.233 ± 0.028 | 3.118 ± 0.176 | 3.350 ± 0.372 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 129.277 ± 7.221 | - | 130.604 ± 7.690 | 125.315 ± 5.379 | 126.813 ± 1.129 | 140.702 ± 3.224 | 145.545 ± 2.193 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 40.918 ± 0.185 | - | 40.669 ± 0.422 | 40.181 ± 1.170 | 64.350 ± 2.565 | 147.458 ± 3.916 | 60.681 ± 21.340 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 105.283 ± 3.979 | - | 113.264 ± 4.771 | 95.915 ± 2.222 | 91.502 ± 0.477 | 135.701 ± 4.120 | 139.753 ± 0.831 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 44.335 ± 0.840 | - | 43.568 ± 0.789 | 43.658 ± 19.566 | 46.359 ± 1.214 | 132.710 ± 6.069 | 87.576 ± 27.835 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 47.859 ± 0.386 | - | 47.741 ± 0.810 | 42.671 ± 0.261 | 68.199 ± 0.134 | 46.644 ± 0.231 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 23.476 ± 0.148 | - | 23.474 ± 0.170 | 29.148 ± 0.082 | 29.528 ± 0.328 | 23.453 ± 0.564 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 47.492 ± 0.214 | - | unsupported | 43.147 ± 0.203 | 68.743 ± 0.972 | 47.146 ± 5.968 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 23.562 ± 0.125 | - | unsupported | 29.422 ± 0.313 | 30.114 ± 0.942 | 23.947 ± 4.724 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 7.455 ± 0.051 | - | 7.489 ± 0.057 | 11.539 ± 0.067 | 11.949 ± 0.109 | 11.940 ± 0.109 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 10.070 ± 0.178 | - | 10.069 ± 0.006 | 14.012 ± 0.009 | 17.488 ± 0.078 | 17.651 ± 0.106 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 70.859 ± 0.856 | - | 70.519 ± 0.181 | 107.714 ± 2.149 | 104.682 ± 2.681 | 86.189 ± 12.521 | 83.570 ± 2.385 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 31.022 ± 0.067 | - | 31.249 ± 0.184 | 43.002 ± 0.090 | 52.864 ± 1.599 | 84.381 ± 6.077 | 37.230 ± 0.939 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 117.450 ± 0.285 | - | 116.933 ± 0.335 | 152.981 ± 1.339 | 176.914 ± 1.516 | 150.023 ± 1.886 | 161.470 ± 2.873 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 49.584 ± 0.307 | - | 49.753 ± 1.098 | 63.007 ± 0.983 | 73.635 ± 3.068 | 152.976 ± 2.939 | 64.414 ± 4.882 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 75.203 ± 3.197 | - | 75.320 ± 4.031 | 68.435 ± 1.016 | 72.606 ± 0.631 | 78.629 ± 0.591 | 79.891 ± 3.664 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 24.401 ± 0.875 | - | 24.541 ± 0.650 | 23.108 ± 0.343 | 37.605 ± 0.350 | 77.970 ± 0.881 | 34.938 ± 1.806 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.275 ± 0.081 | - | 20.915 ± 0.084 | 24.249 ± 0.125 | 16.044 ± 0.753 | 4.225 ± 0.055 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.890 ± 0.038 | - | 9.263 ± 0.131 | 11.266 ± 0.014 | 2.495 ± 0.178 | 4.091 ± 0.037 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 7.491 ± 0.031 | - | 7.468 ± 0.097 | 9.604 ± 0.038 | 7.889 ± 0.507 | 9.062 ± 0.256 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.391 ± 0.030 | - | 4.410 ± 0.069 | 5.754 ± 0.014 | 6.962 ± 0.067 | 7.574 ± 1.260 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 4.587 ± 0.009 | - | 4.570 ± 0.021 | 5.418 ± 0.027 | 5.085 ± 0.044 | 4.854 ± 0.116 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.357 ± 0.005 | - | 3.341 ± 0.075 | 3.548 ± 0.036 | 4.616 ± 0.131 | 4.340 ± 0.078 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.514 ± 0.272 | - | 6.550 ± 0.131 | 7.651 ± 0.072 | 5.472 ± 0.043 | 5.393 ± 0.310 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 6.448 ± 0.076 | - | 6.451 ± 0.056 | 5.823 ± 0.029 | 5.951 ± 0.032 | 6.295 ± 0.052 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 47.464 ± 0.239 | - | 47.592 ± 0.284 | 42.755 ± 0.309 | 68.125 ± 0.341 | 44.178 ± 0.230 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 23.449 ± 0.062 | - | 23.548 ± 0.111 | 29.700 ± 0.946 | 29.459 ± 0.539 | 22.245 ± 1.906 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 48.950 ± 0.195 | - | unsupported | 50.948 ± 0.249 | 60.917 ± 1.103 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 23.110 ± 0.076 | - | unsupported | 39.637 ± 5.426 | 34.859 ± 0.248 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 127.565 ± 8.810 | - | 127.222 ± 7.548 | 121.820 ± 0.739 | 126.107 ± 0.844 | 141.624 ± 6.668 | 158.559 ± 8.296 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 40.250 ± 0.200 | - | 40.506 ± 0.391 | 40.170 ± 1.252 | 65.416 ± 0.851 | 142.351 ± 29.326 | 65.714 ± 4.052 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 138.710 ± 6.249 | - | 136.868 ± 9.297 | 131.480 ± 5.097 | 137.391 ± 1.327 | 154.610 ± 14.816 | 173.769 ± 6.163 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 45.062 ± 1.078 | - | 44.495 ± 6.288 | 45.130 ± 0.199 | 71.930 ± 0.446 | 153.514 ± 7.507 | 71.030 ± 0.866 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 27.618 ± 0.454 | - | 56.554 ± 0.252 | 76.704 ± 0.841 | 51.881 ± 0.363 | 81.049 ± 9.844 | 88.267 ± 10.674 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 12.665 ± 0.227 | - | 26.204 ± 0.682 | 33.173 ± 0.908 | 26.295 ± 0.543 | 82.209 ± 10.420 | 32.676 ± 10.766 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 9.904 ± 0.205 | - | 9.587 ± 0.295 | 36.291 ± 2.498 | 38.081 ± 0.465 | 37.909 ± 10.575 | 42.746 ± 6.321 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 6.504 ± 0.133 | - | 7.135 ± 0.433 | 12.523 ± 0.622 | 20.250 ± 0.084 | 37.149 ± 11.173 | 16.838 ± 13.842 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 21.957 ± 0.119 | - | 22.265 ± 0.720 | 42.956 ± 0.635 | 42.622 ± 0.357 | 23.394 ± 0.283 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 13.517 ± 0.231 | - | 14.233 ± 0.261 | 19.388 ± 0.114 | 24.591 ± 0.492 | 22.448 ± 0.292 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.289 ± 0.062 | - | 28.373 ± 0.094 | 55.908 ± 1.756 | 109.037 ± 0.299 | 67.219 ± 10.143 | 71.693 ± 8.459 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 11.895 ± 0.027 | - | 12.074 ± 0.031 | 22.043 ± 0.142 | 54.846 ± 0.197 | 69.251 ± 13.681 | 29.627 ± 14.484 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 136.939 ± 4.560 | - | 137.802 ± 10.293 | 132.196 ± 3.391 | 142.932 ± 1.191 | 152.194 ± 2.941 | 166.403 ± 6.608 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 44.711 ± 1.289 | - | 45.152 ± 0.947 | 43.820 ± 0.266 | 71.746 ± 2.471 | 154.777 ± 6.227 | 69.432 ± 1.588 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 20.708 ± 0.144 | - | 21.074 ± 0.191 | 44.904 ± 0.231 | 42.591 ± 0.428 | 63.768 ± 12.969 | 68.774 ± 12.699 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.462 ± 0.026 | - | 8.481 ± 0.011 | 16.517 ± 0.213 | 21.739 ± 0.130 | 63.426 ± 12.976 | 27.656 ± 15.412 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 20.380 ± 0.029 | - | 20.642 ± 0.090 | 32.638 ± 1.059 | 27.923 ± 0.483 | 38.677 ± 8.471 | 41.838 ± 4.165 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 8.108 ± 0.009 | - | 8.165 ± 0.048 | 11.865 ± 0.160 | 14.342 ± 0.158 | 38.470 ± 10.424 | 18.102 ± 11.228 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 25.932 ± 0.091 | - | 25.569 ± 0.097 | 50.901 ± 0.154 | 66.311 ± 1.545 | 94.611 ± 11.960 | 90.939 ± 11.547 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 10.520 ± 0.017 | - | 10.667 ± 0.028 | 19.182 ± 0.277 | 33.635 ± 0.239 | 83.809 ± 13.575 | 54.251 ± 20.853 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.223 ± 0.034 | - | 13.318 ± 0.032 | 32.611 ± 0.699 | 37.561 ± 0.575 | 46.817 ± 9.133 | 50.273 ± 1.395 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 5.655 ± 0.011 | - | 5.680 ± 0.122 | 12.071 ± 6.000 | 19.113 ± 0.220 | 47.117 ± 10.883 | 20.936 ± 10.412 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.458 ± 0.015 | 0.477 ± 0.008 | unsupported | unsupported | 0.488 ± 0.013 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.463 ± 0.088 | 0.523 ± 0.026 | unsupported | unsupported | 0.411 ± 0.047 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 142.617 ± 8.182 | - | 142.029 ± 8.678 | 131.444 ± 1.188 | 139.791 ± 0.445 | 161.081 ± 34.070 | 168.663 ± 2.275 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 44.619 ± 3.134 | - | 43.801 ± 0.211 | 44.694 ± 0.356 | 72.968 ± 1.909 | 161.030 ± 38.189 | 71.651 ± 1.188 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 137.199 ± 6.496 | - | 138.002 ± 8.788 | 131.295 ± 2.662 | 139.894 ± 0.732 | 156.526 ± 3.151 | 172.116 ± 5.659 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 44.561 ± 0.939 | - | 44.274 ± 0.312 | 44.797 ± 0.929 | 71.426 ± 1.199 | 160.398 ± 6.610 | 71.820 ± 4.510 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 133.728 ± 3.799 | - | 136.197 ± 9.130 | 131.402 ± 5.491 | 137.788 ± 2.114 | 152.248 ± 5.657 | 168.934 ± 27.366 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 44.635 ± 1.052 | - | 44.294 ± 1.011 | 45.189 ± 1.087 | 71.986 ± 1.821 | 153.781 ± 4.775 | 69.710 ± 0.950 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 120.123 ± 8.443 | - | 127.049 ± 7.139 | 125.529 ± 6.304 | 126.430 ± 1.204 | 140.870 ± 4.047 | 158.447 ± 8.664 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 40.832 ± 0.905 | - | 40.260 ± 0.258 | 39.808 ± 0.354 | 65.218 ± 1.347 | 140.908 ± 20.635 | 64.829 ± 2.835 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 58.111 ± 0.104 | - | 58.720 ± 0.056 | 63.137 ± 1.447 | 74.735 ± 0.615 | 125.769 ± 2.352 | 123.914 ± 4.480 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 23.589 ± 0.125 | - | 23.692 ± 0.089 | 22.172 ± 0.259 | 37.320 ± 0.743 | 126.768 ± 1.107 | 50.582 ± 15.623 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 16.276 ± 0.131 | - | 16.094 ± 0.340 | 11.679 ± 0.236 | 26.383 ± 0.373 | 13.694 ± 0.312 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.978 ± 0.342 | - | 6.714 ± 0.111 | 6.304 ± 0.235 | 13.820 ± 0.121 | 13.238 ± 0.311 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.882 ± 0.062 | - | 1.897 ± 0.072 | 24.316 ± 0.123 | 2.443 ± 0.093 | 1.449 ± 0.214 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.789 ± 0.033 | - | 0.779 ± 0.039 | 11.452 ± 0.234 | 1.344 ± 0.063 | 1.194 ± 0.063 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 2.506 ± 0.075 | - | 2.505 ± 0.113 | 4.064 ± 0.039 | 2.455 ± 0.513 | 1.332 ± 0.158 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.351 ± 0.055 | - | 1.364 ± 0.052 | 2.302 ± 0.028 | 1.318 ± 0.131 | 1.640 ± 0.100 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 16.277 ± 0.372 | - | 16.005 ± 0.147 | 11.772 ± 0.596 | 26.350 ± 1.016 | 13.126 ± 0.283 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.937 ± 0.185 | - | 7.175 ± 0.217 | 6.202 ± 0.166 | 13.960 ± 0.137 | 13.124 ± 0.199 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.822 ± 0.087 | - | 1.916 ± 0.085 | 23.202 ± 0.195 | 2.434 ± 0.072 | 1.173 ± 0.122 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.822 ± 0.020 | - | 0.818 ± 0.017 | 10.870 ± 0.259 | 1.335 ± 0.058 | 1.373 ± 0.198 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 10.226 ± 0.165 | - | 10.244 ± 0.084 | 16.401 ± 0.113 | 11.848 ± 0.320 | 6.578 ± 0.433 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.364 ± 0.407 | - | 5.547 ± 0.095 | 8.790 ± 0.076 | 6.404 ± 0.351 | 6.836 ± 0.083 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 10.700 ± 0.155 | - | 11.117 ± 0.209 | 10.533 ± 0.182 | 19.633 ± 0.100 | 11.143 ± 0.210 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.289 ± 0.297 | - | 6.196 ± 0.218 | 6.301 ± 0.041 | 11.008 ± 0.172 | 11.200 ± 0.217 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.047 ± 0.079 | - | 1.149 ± 0.182 | 2.041 ± 0.028 | 1.969 ± 0.032 | 1.047 ± 0.096 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.517 ± 0.032 | - | 0.461 ± 0.228 | 0.973 ± 0.039 | 1.179 ± 0.057 | 0.887 ± 0.051 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.193 ± 0.148 | - | 1.136 ± 0.071 | 0.839 ± 0.028 | 3.151 ± 0.467 | 1.137 ± 0.273 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.614 ± 0.066 | - | 0.678 ± 0.077 | 0.396 ± 0.089 | 1.285 ± 0.105 | 1.006 ± 0.068 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 10.731 ± 0.152 | - | 10.704 ± 0.238 | 10.753 ± 0.211 | 19.965 ± 0.307 | 10.848 ± 0.178 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.472 ± 0.342 | - | 6.271 ± 0.340 | 6.407 ± 0.419 | 10.922 ± 0.182 | 10.915 ± 0.230 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.087 ± 0.065 | - | 1.047 ± 0.048 | 2.030 ± 0.089 | 1.999 ± 0.026 | 1.024 ± 0.080 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.469 ± 0.042 | - | 0.446 ± 0.092 | 0.935 ± 0.027 | 1.112 ± 0.046 | 0.858 ± 0.128 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.131 ± 0.107 | - | 1.084 ± 0.183 | 0.881 ± 0.034 | 3.214 ± 0.265 | 1.065 ± 0.101 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.646 ± 0.023 | - | 0.692 ± 0.030 | 0.364 ± 0.046 | 1.639 ± 0.229 | 1.202 ± 0.078 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.200 ± 0.055 | 1.138 ± 0.052 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.420 ± 0.077 | 0.434 ± 0.055 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 38.897 ± 0.184 | - | 38.079 ± 0.070 | 55.652 ± 1.091 | 71.646 ± 0.493 | 71.847 ± 13.791 | 78.852 ± 7.955 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 18.535 ± 0.111 | - | 19.003 ± 1.146 | 20.870 ± 6.358 | 36.981 ± 0.678 | 72.405 ± 13.126 | 33.771 ± 9.621 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 144.735 ± 2.989 | - | 146.428 ± 7.378 | 146.897 ± 2.006 | 250.679 ± 0.356 | 264.928 ± 3.987 | 265.297 ± 2.815 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 46.621 ± 0.161 | - | 46.948 ± 1.402 | 47.225 ± 0.209 | 127.556 ± 0.525 | 265.432 ± 6.954 | 108.204 ± 22.158 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 139.076 ± 5.866 | - | 139.303 ± 7.297 | 138.542 ± 9.581 | 153.732 ± 3.880 | 158.093 ± 28.543 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 47.295 ± 1.312 | - | 46.606 ± 0.340 | 52.739 ± 0.856 | 83.172 ± 3.256 | 160.274 ± 6.565 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 126.140 ± 9.247 | - | 133.054 ± 7.154 | 124.466 ± 1.120 | 125.818 ± 0.905 | 145.767 ± 32.312 | 151.965 ± 4.755 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 40.270 ± 0.174 | - | 40.225 ± 0.252 | 40.646 ± 0.371 | 69.044 ± 1.388 | 147.736 ± 7.339 | 63.337 ± 1.002 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.470 ± 0.036 | - | 24.501 ± 0.300 | 58.704 ± 2.026 | 108.696 ± 0.493 | 67.282 ± 12.992 | 71.569 ± 9.826 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.261 ± 0.129 | - | 10.400 ± 0.015 | 23.447 ± 0.170 | 54.678 ± 0.157 | 74.981 ± 8.860 | 39.186 ± 10.058 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 136.578 ± 2.770 | - | 137.763 ± 6.789 | 186.321 ± 4.646 | 137.123 ± 0.817 | 215.588 ± 3.302 | 221.979 ± 3.123 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 44.454 ± 0.288 | - | 44.439 ± 0.091 | 68.388 ± 1.783 | 70.576 ± 0.953 | 221.533 ± 5.823 | 94.282 ± 2.667 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 130.919 ± 7.826 | - | 136.272 ± 9.789 | 133.031 ± 4.672 | 138.727 ± 0.800 | 154.890 ± 8.183 | 171.357 ± 7.577 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 44.883 ± 0.266 | - | 44.499 ± 0.608 | 45.061 ± 0.192 | 71.932 ± 2.089 | 153.639 ± 7.541 | 70.055 ± 1.762 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 51.317 ± 0.056 | - | 51.331 ± 0.111 | 75.745 ± 1.785 | 40.329 ± 0.789 | 79.877 ± 12.998 | 88.592 ± 10.550 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 20.417 ± 0.049 | - | 20.593 ± 0.197 | 28.500 ± 12.086 | 20.662 ± 0.231 | 79.780 ± 14.305 | 34.476 ± 15.728 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 140.318 ± 7.546 | 139.278 ± 6.358 | unsupported | 146.879 ± 3.785 | 152.809 ± 2.489 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 46.869 ± 0.187 | 46.837 ± 1.492 | unsupported | 52.650 ± 1.128 | 81.140 ± 2.636 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.836 ± 0.073 | - | 0.800 ± 0.088 | 0.882 ± 0.092 | 0.962 ± 0.030 | 7.834 ± 4.296 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.512 ± 0.174 | - | 0.625 ± 0.227 | 0.368 ± 0.049 | 0.982 ± 0.075 | 7.547 ± 4.038 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 1.001 ± 0.324 | - | 0.907 ± 0.255 | 0.889 ± 0.091 | 0.867 ± 0.295 | 7.946 ± 4.185 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.601 ± 0.227 | - | 0.607 ± 0.246 | 0.333 ± 0.046 | 0.592 ± 0.057 | 7.910 ± 4.328 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.206 ± 0.028 | - | unsupported | 1.425 ± 0.260 | 1.423 ± 0.120 | 8.183 ± 4.083 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.809 ± 0.108 | - | unsupported | 0.630 ± 0.093 | 1.212 ± 0.201 | 6.564 ± 4.250 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.422 ± 0.027 | - | 0.483 ± 0.030 | 0.697 ± 0.035 | 0.436 ± 0.028 | 1.202 ± 0.094 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.357 ± 0.070 | - | 0.417 ± 0.023 | 0.349 ± 0.014 | 0.504 ± 0.066 | 1.188 ± 0.065 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.825 ± 0.222 | - | 0.889 ± 0.075 | 1.337 ± 0.219 | 1.114 ± 0.104 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.520 ± 0.169 | - | 0.614 ± 0.254 | 0.506 ± 0.050 | 1.066 ± 0.014 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.862 ± 0.091 | - | 0.840 ± 0.102 | 0.946 ± 0.095 | 0.938 ± 0.181 | 1.549 ± 11.522 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.504 ± 0.221 | - | 0.467 ± 0.450 | 0.329 ± 0.034 | 0.609 ± 0.162 | 8.517 ± 8.420 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.794 ± 0.020 | - | 0.913 ± 0.008 | 0.476 ± 0.046 | 0.446 ± 0.025 | 1.410 ± 0.257 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 1.208 ± 0.034 | - | 1.327 ± 0.057 | 0.640 ± 0.015 | 0.491 ± 0.032 | 1.392 ± 0.025 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.625 ± 0.102 | - | 0.002 ± 0.000 | 1.827 ± 0.067 | 1.354 ± 0.029 | 9.795 ± 3.037 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.903 ± 0.109 | - | 0.003 ± 0.000 | 0.718 ± 0.068 | 0.881 ± 0.056 | 9.242 ± 5.086 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.076 ± 0.000 | - | 0.077 ± 0.001 | 0.084 ± 0.002 | 0.116 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.133 ± 0.001 | - | 0.136 ± 0.001 | 0.129 ± 0.001 | 0.155 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.260 ± 0.010 | - | 0.259 ± 0.012 | 0.282 ± 0.011 | 0.353 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.050 ± 0.000 | - | 0.053 ± 0.000 | 0.147 ± 0.009 | 0.121 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.056 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.051 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.408 ± 0.013 | - | 0.394 ± 0.009 | 1.155 ± 0.012 | 0.674 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.039 ± 0.002 | 0.086 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.051 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.076 ± 0.000 | - | 0.119 ± 0.001 | 0.087 ± 0.001 | 0.102 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.134 ± 0.001 | - | 0.190 ± 0.012 | 0.125 ± 0.002 | 0.156 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.260 ± 0.014 | - | 0.378 ± 0.012 | 0.340 ± 0.012 | 0.355 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.060 ± 0.000 | - | 0.099 ± 0.007 | 0.177 ± 0.003 | 0.122 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.010 ± 0.002 | 0.012 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.004 ± 0.001 | - | 0.010 ± 0.001 | 0.016 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.011 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.010 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.228 ± 0.055 | - | 0.245 ± 0.007 | 1.388 ± 0.022 | 1.175 ± 0.165 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.018 ± 0.000 | - | 0.038 ± 0.002 | 0.058 ± 0.000 | 0.061 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.003 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.001 | 0.012 ± 0.001 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.529 ± 0.012 | - | 0.534 ± 0.010 | 0.708 ± 0.016 | 0.910 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.128 ± 0.004 | - | 2.135 ± 0.023 | 2.122 ± 0.013 | 2.665 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.683 ± 0.021 | - | 6.666 ± 0.048 | 6.485 ± 0.102 | 8.934 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.435 ± 0.016 | - | 2.446 ± 0.025 | 3.198 ± 0.016 | 2.631 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.103 ± 0.000 | - | 0.106 ± 0.004 | 0.121 ± 0.005 | 0.165 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.028 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.002 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.030 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.021 ± 0.000 | - | 0.023 ± 0.000 | 0.029 ± 0.000 | 0.101 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.026 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.002 | 0.034 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.027 ± 0.000 | - | 0.030 ± 0.002 | 0.035 ± 0.001 | 0.108 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 19.597 ± 0.060 | - | 19.392 ± 0.102 | 25.963 ± 0.427 | 20.595 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.001 | 0.034 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.828 ± 0.019 | - | 0.825 ± 0.003 | 0.874 ± 0.004 | 0.969 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.026 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.054 ± 0.000 | - | 0.056 ± 0.000 | 0.060 ± 0.001 | 0.141 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.533 ± 0.012 | - | 0.869 ± 0.546 | 0.720 ± 0.400 | 0.923 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 0.853 ± 0.067 | - | 0.858 ± 0.023 | 2.110 ± 0.009 | 2.093 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 2.471 ± 0.011 | - | 2.469 ± 0.008 | 9.258 ± 0.015 | 5.874 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 2.411 ± 0.011 | - | 2.413 ± 0.027 | 2.238 ± 0.029 | 3.993 ± 4.126 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.105 ± 0.005 | - | 0.158 ± 0.003 | 0.179 ± 0.005 | 0.203 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.005 ± 0.011 | - | 0.010 ± 0.001 | 0.011 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.009 ± 0.000 | - | 0.016 ± 0.001 | 0.016 ± 0.001 | 0.043 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.001 | 0.016 ± 0.001 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.014 ± 0.008 | - | 0.016 ± 0.001 | 0.015 ± 0.001 | 0.033 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.022 ± 0.000 | - | 0.034 ± 0.002 | 0.029 ± 0.001 | 0.070 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.010 ± 0.001 | 0.017 ± 0.002 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | 0.019 ± 0.001 | 0.018 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.028 ± 0.000 | - | 0.043 ± 0.003 | 0.035 ± 0.001 | 0.072 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 8.116 ± 0.023 | - | 8.164 ± 0.026 | 17.763 ± 0.644 | 28.977 ± 6.596 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.013 ± 0.000 | 0.021 ± 0.004 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.351 ± 0.014 | - | 0.354 ± 0.016 | 1.248 ± 0.010 | 2.473 ± 0.840 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.019 ± 0.006 | - | 0.047 ± 0.005 | 0.025 ± 0.001 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.054 ± 0.001 | - | 0.080 ± 0.001 | 0.061 ± 0.002 | 0.114 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.477 ± 0.012 | - | 0.479 ± 0.007 | 0.512 ± 0.012 | 0.445 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.958 ± 0.019 | - | 1.967 ± 0.005 | 1.942 ± 0.011 | 1.921 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 6.427 ± 0.064 | - | 6.465 ± 0.027 | 5.694 ± 0.038 | 6.480 ± 0.396 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.884 ± 0.009 | - | 1.665 ± 0.011 | 1.332 ± 0.017 | 1.475 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.086 ± 0.016 | - | 0.087 ± 0.007 | 0.089 ± 0.003 | 0.164 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.002 | 0.010 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.002 | 0.034 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.002 | 0.010 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.020 ± 0.001 | - | 0.022 ± 0.000 | 0.025 ± 0.001 | 0.050 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.001 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.026 ± 0.000 | - | 0.028 ± 0.001 | 0.032 ± 0.002 | 0.058 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 14.608 ± 0.086 | - | 14.404 ± 0.036 | 10.485 ± 0.032 | 11.323 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.001 | 0.012 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.694 ± 0.009 | - | 0.698 ± 0.010 | 0.626 ± 0.017 | 0.784 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.017 ± 0.001 | - | 0.019 ± 0.000 | 0.024 ± 0.003 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.051 ± 0.000 | - | 0.054 ± 0.000 | 0.053 ± 0.001 | 0.091 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.482 ± 0.009 | - | 0.779 ± 0.003 | 0.789 ± 0.076 | 0.449 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.918 ± 0.003 | - | 1.925 ± 0.025 | 1.949 ± 0.036 | 1.286 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 2.309 ± 0.012 | - | 2.318 ± 0.031 | 8.432 ± 0.059 | 3.676 ± 0.196 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.875 ± 0.006 | - | 1.696 ± 0.217 | 1.191 ± 0.012 | 1.806 ± 0.194 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.089 ± 0.004 | - | 0.129 ± 0.004 | 0.136 ± 0.004 | 0.151 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.001 | 0.011 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.009 ± 0.000 | - | 0.015 ± 0.003 | 0.014 ± 0.001 | 0.029 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.015 ± 0.000 | 0.015 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.021 ± 0.000 | - | 0.031 ± 0.002 | 0.026 ± 0.001 | 0.043 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.010 ± 0.001 | 0.015 ± 0.001 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.010 ± 0.000 | - | 0.017 ± 0.000 | 0.017 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.027 ± 0.000 | - | 0.040 ± 0.003 | 0.031 ± 0.001 | 0.046 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 5.272 ± 0.024 | - | 5.638 ± 0.078 | 9.296 ± 5.301 | 13.993 ± 0.986 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.012 ± 0.001 | 0.017 ± 0.001 | 0.026 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.272 ± 0.015 | - | 0.282 ± 0.016 | 0.927 ± 0.009 | 0.458 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.017 ± 0.000 | - | 0.028 ± 0.001 | 0.023 ± 0.001 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.051 ± 0.000 | - | 0.075 ± 0.003 | 0.053 ± 0.002 | 0.091 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.338 ± 0.020 | - | 0.339 ± 0.011 | 0.237 ± 0.010 | 0.446 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.560 ± 0.029 | - | 1.560 ± 0.007 | 0.950 ± 0.011 | 1.918 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 4.930 ± 0.082 | - | 4.930 ± 0.016 | 2.923 ± 0.069 | 6.559 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.898 ± 0.003 | - | 0.899 ± 0.003 | 0.552 ± 0.027 | 1.482 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.002 | - | 0.003 ± 0.000 | 0.009 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.062 ± 0.001 | - | 0.064 ± 0.001 | 0.042 ± 0.011 | 0.157 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.020 ± 0.001 | 0.037 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.007 ± 0.009 | - | 0.009 ± 0.000 | 0.013 ± 0.001 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.002 | 0.021 ± 0.000 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 7.193 ± 0.047 | - | 7.226 ± 0.071 | 4.278 ± 0.042 | 11.062 ± 0.108 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.490 ± 0.010 | - | 0.488 ± 0.008 | 0.248 ± 0.021 | 0.775 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.038 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.039 ± 0.000 | - | 0.041 ± 0.003 | 0.032 ± 0.000 | 0.117 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.338 ± 0.008 | - | 0.587 ± 0.009 | 0.400 ± 0.012 | 0.445 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.539 ± 0.007 | - | 1.551 ± 0.348 | 0.945 ± 0.006 | 1.280 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 1.725 ± 0.012 | - | 1.743 ± 0.014 | 3.656 ± 0.018 | 3.668 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.900 ± 0.007 | - | 1.160 ± 0.117 | 0.693 ± 0.015 | 1.752 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.014 ± 0.001 | 0.018 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.063 ± 0.001 | - | 0.084 ± 0.002 | 0.056 ± 0.004 | 0.152 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.015 ± 0.002 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.012 ± 0.001 | 0.013 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.016 ± 0.004 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.003 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.016 ± 0.001 | - | 0.024 ± 0.000 | 0.019 ± 0.001 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.013 ± 0.006 | - | 0.014 ± 0.000 | 0.019 ± 0.002 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.021 ± 0.000 | - | 0.030 ± 0.002 | 0.022 ± 0.001 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 2.468 ± 0.015 | - | 2.482 ± 0.008 | 5.239 ± 0.063 | 14.015 ± 0.690 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.005 ± 0.004 | - | 0.010 ± 0.000 | 0.017 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.177 ± 0.006 | - | 0.182 ± 0.015 | 0.319 ± 0.015 | 0.438 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.014 ± 0.007 | - | 0.022 ± 0.001 | 0.024 ± 0.004 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.040 ± 0.001 | - | 0.057 ± 0.004 | 0.032 ± 0.001 | 0.088 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.126 ± 0.001 | - | 0.129 ± 0.004 | 0.203 ± 0.007 | 0.150 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.266 ± 0.002 | - | 0.269 ± 0.011 | 0.279 ± 0.015 | 0.237 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.704 ± 0.012 | - | 0.709 ± 0.006 | 0.580 ± 0.017 | 0.592 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.131 ± 0.001 | - | 0.133 ± 0.002 | 0.175 ± 0.005 | 0.217 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.025 ± 0.001 | 0.053 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.001 | 0.022 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.027 ± 0.001 | 0.053 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.070 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.015 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.005 ± 0.001 | - | 0.991 ± 0.002 | 1.290 ± 0.021 | 1.325 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.063 ± 0.000 | - | 0.066 ± 0.000 | 0.089 ± 0.009 | 0.163 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.052 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.020 ± 0.001 | 0.045 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.128 ± 0.002 | - | 0.246 ± 0.011 | 0.172 ± 0.004 | 0.124 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.265 ± 0.022 | - | 0.482 ± 0.012 | 0.233 ± 0.001 | 0.212 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.339 ± 0.016 | - | 0.344 ± 0.021 | 0.389 ± 0.018 | 0.633 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.214 ± 0.017 | - | 0.217 ± 0.011 | 0.225 ± 0.018 | 0.234 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.023 ± 0.009 | 0.020 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.011 ± 0.000 | - | 0.021 ± 0.003 | 0.039 ± 0.009 | 0.045 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.023 ± 0.002 | 0.024 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.013 ± 0.000 | - | 0.023 ± 0.001 | 0.042 ± 0.001 | 0.054 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.022 ± 0.004 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.025 ± 0.001 | 0.024 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.028 ± 0.001 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.027 ± 0.006 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.001 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.006 ± 0.000 | - | 0.012 ± 0.000 | 0.028 ± 0.001 | 0.027 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.505 ± 0.014 | - | 0.526 ± 0.020 | 0.702 ± 0.023 | 0.983 ± 0.110 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.027 ± 0.000 | 0.025 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.063 ± 0.000 | - | 0.109 ± 0.002 | 0.076 ± 0.004 | 0.166 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.028 ± 0.001 | 0.022 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.009 ± 0.000 | - | 0.017 ± 0.000 | 0.031 ± 0.000 | 0.022 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.248 ± 0.010 | - | 0.251 ± 0.015 | 0.223 ± 0.013 | 0.321 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.695 ± 0.009 | - | 0.701 ± 0.017 | 0.528 ± 0.013 | 0.862 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 2.135 ± 0.009 | - | 2.178 ± 0.018 | 1.347 ± 0.022 | 2.315 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.475 ± 0.013 | - | 0.469 ± 0.013 | 0.565 ± 0.013 | 0.627 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.030 ± 0.000 | - | 0.032 ± 0.000 | 0.026 ± 0.001 | 0.071 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.027 ± 0.000 | - | 0.029 ± 0.000 | 0.029 ± 0.001 | 0.063 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.003 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.023 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.001 | 0.011 ± 0.000 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.750 ± 0.020 | - | 3.706 ± 0.068 | 4.395 ± 0.034 | 4.723 ± 0.105 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.222 ± 0.011 | - | 0.225 ± 0.001 | 0.175 ± 0.003 | 0.213 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.002 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.019 ± 0.000 | - | 0.020 ± 0.000 | 0.016 ± 0.000 | 0.046 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.244 ± 0.012 | - | 0.382 ± 0.011 | 0.321 ± 0.013 | 0.337 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.691 ± 0.010 | - | 1.028 ± 0.004 | 0.763 ± 0.006 | 0.870 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.867 ± 0.008 | - | 0.876 ± 0.003 | 2.019 ± 0.009 | 1.362 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.464 ± 0.014 | - | 0.753 ± 0.016 | 0.511 ± 0.017 | 0.996 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.030 ± 0.000 | - | 0.051 ± 0.003 | 0.044 ± 0.000 | 0.063 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.011 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.027 ± 0.000 | - | 0.046 ± 0.004 | 0.047 ± 0.001 | 0.061 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.010 ± 0.001 | 0.012 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.002 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.009 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.035 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.001 | 0.011 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.011 ± 0.000 | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 1.794 ± 0.014 | - | 1.813 ± 0.022 | 3.748 ± 0.050 | 7.876 ± 0.981 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.003 | 0.023 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.118 ± 0.007 | - | 0.122 ± 0.003 | 0.290 ± 0.011 | 0.207 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.008 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.030 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.020 ± 0.000 | - | 0.030 ± 0.002 | 0.026 ± 0.000 | 0.047 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.188 ± 0.019 | - | 0.138 ± 0.005 | 0.104 ± 0.005 | 0.257 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.304 ± 0.026 | - | 0.259 ± 0.020 | 0.142 ± 0.001 | 0.362 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.786 ± 0.009 | - | 0.766 ± 0.006 | 0.411 ± 0.014 | 0.834 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.095 ± 0.000 | - | 0.103 ± 0.042 | 0.147 ± 0.003 | 0.229 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.008 ± 0.000 | - | 0.013 ± 0.003 | 0.030 ± 0.001 | 0.077 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.004 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.024 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.019 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.734 ± 0.014 | - | 0.809 ± 0.032 | 1.026 ± 0.006 | 1.025 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.022 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.049 ± 0.000 | - | 0.055 ± 0.004 | 0.080 ± 0.005 | 0.149 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.008 ± 0.000 | - | 0.012 ± 0.000 | 0.021 ± 0.000 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.185 ± 0.021 | - | 0.251 ± 0.015 | 0.130 ± 0.001 | 0.264 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.301 ± 0.010 | - | 0.426 ± 0.014 | 0.172 ± 0.003 | 0.374 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.376 ± 0.013 | - | 0.384 ± 0.014 | 0.396 ± 0.014 | 0.840 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.150 ± 0.009 | - | 0.180 ± 0.076 | 0.203 ± 0.009 | 0.184 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.012 ± 0.000 | 0.026 ± 0.005 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.009 ± 0.000 | - | 0.026 ± 0.005 | 0.043 ± 0.007 | 0.051 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.012 ± 0.000 | 0.026 ± 0.002 | 0.018 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.013 ± 0.000 | 0.026 ± 0.003 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.013 ± 0.004 | 0.025 ± 0.002 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.013 ± 0.001 | 0.026 ± 0.004 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.016 ± 0.001 | 0.028 ± 0.003 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.013 ± 0.000 | 0.026 ± 0.003 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.014 ± 0.000 | 0.026 ± 0.003 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.017 ± 0.002 | 0.029 ± 0.005 | 0.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.367 ± 0.019 | - | 0.443 ± 0.030 | 0.571 ± 0.022 | 0.751 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.014 ± 0.003 | 0.027 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.050 ± 0.002 | - | 0.109 ± 0.036 | 0.072 ± 0.004 | 0.187 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.005 ± 0.000 | - | 0.015 ± 0.001 | 0.028 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.009 ± 0.000 | - | 0.021 ± 0.001 | 0.031 ± 0.001 | 0.040 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.164 ± 0.010 | - | 1.198 ± 0.021 | 1.509 ± 0.017 | 1.973 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 3.558 ± 0.010 | - | 3.566 ± 0.007 | 5.093 ± 0.108 | 5.094 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 11.138 ± 0.112 | - | 11.095 ± 0.079 | 15.909 ± 0.095 | 14.860 ± 0.151 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.207 ± 0.026 | - | 4.190 ± 0.016 | 5.680 ± 0.061 | 4.228 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.015 ± 0.001 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.178 ± 0.002 | - | 0.181 ± 0.005 | 0.231 ± 0.012 | 0.277 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.020 ± 0.001 | 0.034 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.150 ± 0.006 | - | 0.154 ± 0.002 | 0.213 ± 0.020 | 0.259 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.013 ± 0.001 | - | 0.015 ± 0.009 | 0.033 ± 0.003 | 0.054 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.003 | 0.037 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.030 ± 0.001 | 0.042 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.034 ± 0.000 | - | 0.036 ± 0.003 | 0.063 ± 0.001 | 0.072 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.001 | 0.035 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.016 ± 0.000 | - | 0.019 ± 0.000 | 0.035 ± 0.007 | 0.048 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.045 ± 0.000 | - | 0.046 ± 0.004 | 0.079 ± 0.001 | 0.087 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 33.686 ± 0.272 | - | 33.791 ± 0.235 | 44.096 ± 0.344 | 33.378 ± 0.438 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.025 ± 0.002 | 0.051 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.396 ± 0.007 | - | 1.396 ± 0.009 | 1.636 ± 0.009 | 1.597 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.030 ± 0.000 | - | 0.032 ± 0.000 | 0.055 ± 0.001 | 0.068 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.086 ± 0.001 | - | 0.089 ± 0.003 | 0.140 ± 0.003 | 0.195 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.166 ± 0.006 | - | 1.191 ± 0.708 | 1.606 ± 0.027 | 1.982 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 1.407 ± 0.010 | - | 1.418 ± 0.200 | 5.017 ± 0.042 | 2.771 ± 0.106 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 4.200 ± 0.015 | - | 4.220 ± 0.009 | 18.514 ± 0.093 | 7.841 ± 0.732 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.170 ± 0.023 | - | 4.225 ± 0.037 | 4.677 ± 0.026 | 5.086 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.019 ± 0.002 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.180 ± 0.003 | - | 0.272 ± 0.016 | 0.333 ± 0.026 | 0.260 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.014 ± 0.000 | 0.019 ± 0.001 | 0.036 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.153 ± 0.014 | - | 0.229 ± 0.003 | 0.293 ± 0.015 | 0.221 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.014 ± 0.000 | - | 0.023 ± 0.001 | 0.031 ± 0.001 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.013 ± 0.000 | 0.025 ± 0.004 | 0.034 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.014 ± 0.001 | - | 0.023 ± 0.001 | 0.030 ± 0.001 | 0.054 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.034 ± 0.002 | - | 0.054 ± 0.009 | 0.063 ± 0.002 | 0.076 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.015 ± 0.000 | 0.027 ± 0.013 | 0.034 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.017 ± 0.000 | - | 0.028 ± 0.003 | 0.035 ± 0.001 | 0.049 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.045 ± 0.000 | - | 0.071 ± 0.003 | 0.080 ± 0.005 | 0.089 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 15.666 ± 0.173 | - | 15.969 ± 0.076 | 36.751 ± 0.471 | 75.488 ± 47.500 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.012 ± 0.000 | - | 0.022 ± 0.001 | 0.036 ± 0.002 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 0.599 ± 0.010 | - | 0.597 ± 0.015 | 2.340 ± 0.295 | 0.894 ± 0.073 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.031 ± 0.000 | - | 0.050 ± 0.003 | 0.055 ± 0.005 | 0.070 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.087 ± 0.002 | - | 0.132 ± 0.001 | 0.141 ± 0.002 | 0.164 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.947 ± 0.002 | - | 0.948 ± 0.011 | 1.088 ± 0.027 | 1.280 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.193 ± 0.017 | - | 3.203 ± 0.023 | 4.330 ± 0.017 | 4.036 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 10.564 ± 0.049 | - | 10.642 ± 0.017 | 13.833 ± 0.048 | 12.205 ± 0.110 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.288 ± 0.021 | - | 2.232 ± 0.032 | 2.453 ± 0.033 | 2.385 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.014 ± 0.004 | 0.033 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.133 ± 0.002 | - | 0.136 ± 0.007 | 0.163 ± 0.008 | 0.223 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.096 ± 0.006 | - | 0.098 ± 0.008 | 0.136 ± 0.017 | 0.162 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.028 ± 0.001 | 0.040 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.001 | 0.016 ± 0.001 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.012 ± 0.000 | - | 0.014 ± 0.001 | 0.028 ± 0.004 | 0.038 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.032 ± 0.001 | - | 0.034 ± 0.000 | 0.053 ± 0.002 | 0.061 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.031 ± 0.001 | 0.047 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.042 ± 0.002 | - | 0.044 ± 0.000 | 0.065 ± 0.001 | 0.074 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.311 ± 0.088 | - | 17.931 ± 0.097 | 19.620 ± 0.341 | 18.513 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.018 ± 0.006 | - | 0.012 ± 0.003 | 0.022 ± 0.004 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.035 ± 0.001 | - | 1.047 ± 0.004 | 1.163 ± 0.017 | 1.135 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.028 ± 0.001 | - | 0.029 ± 0.000 | 0.048 ± 0.002 | 0.058 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.083 ± 0.000 | - | 0.085 ± 0.000 | 0.118 ± 0.009 | 0.168 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.954 ± 0.002 | - | 1.259 ± 0.559 | 1.017 ± 0.023 | 1.284 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 3.263 ± 1.064 | - | 3.210 ± 0.007 | 4.300 ± 0.045 | 2.083 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 3.904 ± 0.007 | - | 3.894 ± 0.018 | 16.658 ± 0.034 | 6.226 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.254 ± 0.063 | - | 2.270 ± 0.014 | 2.780 ± 0.019 | 3.119 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.005 | 0.032 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.134 ± 0.005 | - | 0.197 ± 0.005 | 0.258 ± 0.015 | 0.208 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.001 | 0.038 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.096 ± 0.004 | - | 0.149 ± 0.004 | 0.202 ± 0.017 | 0.163 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.013 ± 0.000 | - | 0.022 ± 0.001 | 0.027 ± 0.002 | 0.038 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.012 ± 0.002 | 0.022 ± 0.003 | 0.030 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.013 ± 0.006 | - | 0.021 ± 0.001 | 0.027 ± 0.003 | 0.039 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.033 ± 0.000 | - | 0.051 ± 0.003 | 0.052 ± 0.001 | 0.061 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | 0.013 ± 0.000 | 0.024 ± 0.001 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.016 ± 0.010 | - | 0.026 ± 0.000 | 0.031 ± 0.001 | 0.050 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.043 ± 0.000 | - | 0.065 ± 0.002 | 0.066 ± 0.005 | 0.093 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 8.096 ± 0.088 | - | 8.153 ± 0.092 | 21.785 ± 0.441 | 24.992 ± 2.999 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.010 ± 0.000 | - | 0.019 ± 0.001 | 0.030 ± 0.003 | 0.041 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.418 ± 0.011 | - | 0.425 ± 0.026 | 1.750 ± 0.012 | 0.681 ± 0.123 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.028 ± 0.000 | - | 0.044 ± 0.003 | 0.048 ± 0.003 | 0.058 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.084 ± 0.002 | - | 0.122 ± 0.001 | 0.117 ± 0.007 | 0.136 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.583 ± 0.005 | - | unsupported | 0.356 ± 0.014 | 0.787 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.967 ± 0.020 | - | unsupported | 1.907 ± 0.006 | 2.714 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 5.433 ± 0.021 | - | unsupported | 5.292 ± 0.021 | 6.817 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.225 ± 0.008 | - | unsupported | 1.149 ± 0.027 | 1.107 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.059 ± 0.001 | - | unsupported | 0.062 ± 0.002 | 0.075 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.047 ± 0.000 | - | unsupported | 0.055 ± 0.010 | 0.067 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.024 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.007 ± 0.000 | - | unsupported | 0.016 ± 0.004 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.017 ± 0.000 | - | unsupported | 0.025 ± 0.001 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.009 ± 0.000 | - | unsupported | 0.017 ± 0.001 | 0.030 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.031 ± 0.003 | 0.044 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 9.822 ± 0.053 | - | unsupported | 9.035 ± 0.248 | 8.837 ± 0.125 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.459 ± 0.010 | - | unsupported | 0.419 ± 0.013 | 0.471 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.016 ± 0.000 | - | unsupported | 0.024 ± 0.000 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.043 ± 0.000 | - | unsupported | 0.051 ± 0.001 | 0.070 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.585 ± 0.009 | - | unsupported | 0.580 ± 0.013 | 0.744 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.949 ± 0.002 | - | unsupported | 1.899 ± 0.005 | 1.425 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 1.955 ± 0.002 | - | unsupported | 7.617 ± 0.605 | 3.507 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.435 ± 0.256 | - | unsupported | 1.451 ± 0.011 | 1.444 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.005 ± 0.001 | - | unsupported | 0.013 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.079 ± 0.003 | - | unsupported | 0.062 ± 0.005 | 0.074 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.017 ± 0.002 | 0.019 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.069 ± 0.003 | - | unsupported | 0.079 ± 0.003 | 0.069 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | unsupported | 0.015 ± 0.003 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.023 ± 0.005 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.018 ± 0.000 | - | unsupported | 0.027 ± 0.001 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.017 ± 0.006 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | unsupported | 0.025 ± 0.003 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.031 ± 0.001 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 3.242 ± 0.022 | - | unsupported | 11.162 ± 0.092 | 11.854 ± 1.668 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | unsupported | 0.018 ± 0.001 | 0.028 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.169 ± 0.013 | - | unsupported | 0.422 ± 0.020 | 0.316 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.017 ± 0.000 | - | unsupported | 0.025 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.052 ± 0.003 | 0.070 ± 0.002 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.403 ± 0.008 | - | unsupported | 0.959 ± 0.007 | 3.529 ± 0.051 | 2.493 ± 0.868 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.109 ± 0.001 | - | unsupported | 0.064 ± 0.003 | 0.112 ± 0.022 | 0.131 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.186 ± 0.003 | - | unsupported | 0.085 ± 0.008 | 0.181 ± 0.003 | 0.230 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.487 ± 0.026 | - | unsupported | 0.249 ± 0.013 | 0.497 ± 0.016 | 0.682 ± 0.012 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.671 ± 0.050 | - | unsupported | 0.509 ± 1.619 | 1.101 ± 0.016 | 1.764 ± 0.712 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.193 ± 0.002 | - | unsupported | 0.063 ± 0.002 | 0.117 ± 0.021 | 0.100 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.317 ± 0.012 | - | unsupported | 0.082 ± 0.001 | 0.184 ± 0.004 | 0.178 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.227 ± 0.015 | - | unsupported | 0.163 ± 0.007 | 0.538 ± 0.036 | 0.679 ± 0.020 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.564 ± 0.009 | - | unsupported | 0.370 ± 0.009 | 0.849 ± 0.012 | 0.407 ± 0.012 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.055 ± 0.000 | - | unsupported | 0.045 ± 0.006 | 0.189 ± 0.006 | 0.069 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.067 ± 0.000 | - | unsupported | 0.061 ± 0.001 | 0.222 ± 0.007 | 0.110 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.209 ± 0.010 | - | unsupported | 0.165 ± 0.002 | 0.387 ± 0.035 | 0.202 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.324 ± 0.013 | - | unsupported | 0.563 ± 0.012 | 0.806 ± 0.063 | 0.405 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.097 ± 0.002 | - | unsupported | 0.074 ± 0.002 | 0.195 ± 0.004 | 0.066 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.120 ± 0.001 | - | unsupported | 0.099 ± 0.002 | 0.234 ± 0.008 | 0.106 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.357 ± 0.012 | - | unsupported | 0.227 ± 0.002 | 0.386 ± 0.023 | 0.200 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.304 ± 0.010 | - | 0.300 ± 0.008 | 0.193 ± 0.004 | 0.397 ± 0.019 | 0.245 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.039 ± 0.000 | - | 0.041 ± 0.000 | 0.073 ± 0.001 | 0.079 ± 0.011 | 0.052 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.047 ± 0.001 | - | 0.050 ± 0.001 | 0.087 ± 0.001 | 0.103 ± 0.002 | 0.078 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.128 ± 0.001 | - | 0.126 ± 0.001 | 0.121 ± 0.003 | 0.182 ± 0.035 | 0.132 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.142 ± 0.001 | - | 0.144 ± 0.002 | 0.258 ± 0.011 | 0.483 ± 0.058 | 0.252 ± 0.014 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.066 ± 0.002 | - | 0.072 ± 0.002 | 0.126 ± 0.004 | 0.080 ± 0.001 | 0.052 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.047 ± 0.000 | - | 0.085 ± 0.003 | 0.144 ± 0.002 | 0.103 ± 0.003 | 0.077 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.127 ± 0.000 | - | 0.211 ± 0.013 | 0.179 ± 0.012 | 0.222 ± 0.034 | 0.131 ± 0.005 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 40.452 ± 0.199 | - | 45.531 ± 0.756 | 61.571 ± 0.809 | 60.335 ± 0.571 | 52.838 ± 7.368 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 24.067 ± 0.242 | - | 22.170 ± 0.341 | 25.396 ± 0.498 | 64.996 ± 57.745 | 38.043 ± 3.459 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 22.552 ± 0.484 | - | 21.197 ± 0.581 | 26.506 ± 0.208 | 22.430 ± 0.176 | 21.722 ± 3.338 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 17.243 ± 0.241 | - | 12.664 ± 0.266 | 11.343 ± 0.090 | 13.993 ± 0.122 | 12.384 ± 3.511 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.189 ± 0.074 | - | 7.215 ± 0.502 | 7.977 ± 0.129 | 7.943 ± 0.117 | 7.892 ± 0.076 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 9.957 ± 0.035 | - | 9.943 ± 0.031 | 10.484 ± 0.071 | 24.782 ± 6.009 | 11.560 ± 0.050 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.868 ± 0.090 | - | 8.781 ± 0.030 | 8.861 ± 0.044 | 9.550 ± 0.038 | 9.131 ± 0.152 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 12.647 ± 0.024 | - | 12.678 ± 0.085 | 11.840 ± 0.083 | 14.282 ± 0.047 | 13.726 ± 0.046 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 19.728 ± 0.442 | - | 19.615 ± 0.189 | 19.535 ± 0.421 | 40.148 ± 0.628 | 14.281 ± 0.443 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 19.779 ± 0.360 | - | 19.889 ± 0.117 | 15.308 ± 0.256 | 32.670 ± 0.543 | 12.583 ± 0.780 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.336 ± 0.033 | 0.343 ± 0.007 | unsupported | 0.609 ± 0.006 | 0.761 ± 0.041 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 6.281 ± 0.074 | 6.362 ± 0.042 | unsupported | 8.916 ± 0.169 | 9.672 ± 0.119 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.354 ± 0.002 | 0.374 ± 0.022 | unsupported | 0.510 ± 0.003 | 1.263 ± 0.033 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 16.301 ± 4.264 | 4.185 ± 0.165 | unsupported | 5.647 ± 0.106 | 9.969 ± 0.415 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 32.667 ± 0.241 | - | 32.885 ± 0.532 | 44.350 ± 0.234 | 32.405 ± 0.309 | 27.433 ± 0.239 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 19.100 ± 0.221 | - | 18.177 ± 0.263 | 21.771 ± 0.414 | 20.443 ± 0.294 | 16.213 ± 0.243 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 15.743 ± 0.096 | 15.520 ± 0.129 | 9.408 ± 0.033 | 57.366 ± 0.547 | 9.599 ± 0.062 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 10.204 ± 0.080 | 10.116 ± 0.195 | 6.460 ± 0.219 | 54.533 ± 1.880 | 8.779 ± 0.391 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 22.413 ± 0.268 | - | 22.573 ± 0.328 | 34.531 ± 0.983 | 34.615 ± 0.471 | 24.652 ± 10.393 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 17.362 ± 0.330 | - | 17.473 ± 0.432 | 17.416 ± 0.328 | 25.973 ± 1.432 | 20.295 ± 15.559 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.214 ± 0.089 | - | 1.204 ± 0.257 | 1.441 ± 0.041 | 20.242 ± 0.193 | 2.764 ± 0.050 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.448 ± 0.025 | - | 0.484 ± 0.161 | 1.321 ± 0.065 | 3.148 ± 0.644 | 2.710 ± 0.012 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 16.584 ± 0.195 | - | 17.228 ± 0.345 | 20.224 ± 0.139 | 20.398 ± 0.389 | 19.353 ± 1.317 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 19.048 ± 0.338 | - | 18.542 ± 0.059 | 16.406 ± 0.034 | 20.087 ± 0.986 | 18.304 ± 2.973 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 16.426 ± 0.266 | - | 17.024 ± 0.219 | 20.210 ± 0.517 | 20.039 ± 0.332 | 19.620 ± 1.794 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 19.152 ± 0.253 | - | 18.620 ± 0.092 | 16.802 ± 0.266 | 21.826 ± 1.245 | 18.342 ± 1.906 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.093 ± 0.027 | 1.017 ± 0.020 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 23.607 ± 0.836 | 23.177 ± 0.198 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.095 ± 0.024 | 0.999 ± 0.003 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 14.335 ± 0.598 | 14.121 ± 1.161 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.109 ± 0.023 | 1.115 ± 0.020 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 22.822 ± 0.538 | 21.560 ± 0.402 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.082 ± 0.024 | 0.994 ± 0.149 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 13.617 ± 0.504 | 13.170 ± 1.032 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 23.416 ± 1.349 | - | 22.026 ± 0.436 | 26.581 ± 0.923 | 22.478 ± 0.142 | 20.782 ± 0.831 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 17.165 ± 0.087 | - | 12.487 ± 0.214 | 11.407 ± 0.135 | 14.083 ± 0.294 | 12.372 ± 2.397 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 55.428 ± 1.563 | 55.720 ± 0.184 | 62.687 ± 0.311 | 66.145 ± 0.407 | 66.115 ± 2.150 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 54.535 ± 0.431 | 54.798 ± 0.756 | 38.728 ± 0.796 | 54.574 ± 1.154 | 52.890 ± 1.860 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 36.301 ± 0.535 | - | 36.130 ± 0.376 | 30.681 ± 0.329 | 131.158 ± 2.839 | 29.691 ± 0.781 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 18.494 ± 0.180 | - | 18.303 ± 0.064 | 24.174 ± 2.650 | 115.199 ± 19.544 | 19.199 ± 0.657 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.139 ± 0.001 | 0.140 ± 0.002 | unsupported | 0.157 ± 0.029 | 0.149 ± 0.003 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.225 ± 0.008 | 2.217 ± 0.011 | unsupported | 2.479 ± 0.024 | 1.279 ± 0.327 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.081 ± 0.000 | 0.082 ± 0.001 | unsupported | 0.077 ± 0.003 | 0.131 ± 0.010 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.908 ± 0.007 | 0.909 ± 0.004 | unsupported | 1.012 ± 0.006 | 0.349 ± 0.373 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.078 ± 0.001 | 0.079 ± 0.000 | unsupported | 0.137 ± 0.015 | 0.258 ± 0.016 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.264 ± 0.005 | 1.256 ± 0.010 | unsupported | 1.789 ± 0.016 | 2.738 ± 0.089 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.047 ± 0.001 | 0.048 ± 0.007 | unsupported | 0.087 ± 0.002 | 0.161 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.569 ± 0.003 | 0.570 ± 0.463 | unsupported | 0.456 ± 0.153 | 1.664 ± 0.047 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.197 ± 0.002 | 0.197 ± 0.001 | unsupported | 0.285 ± 0.004 | 0.213 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.156 ± 0.008 | 3.123 ± 0.010 | unsupported | 4.431 ± 0.026 | 1.855 ± 0.021 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.085 ± 0.000 | 0.087 ± 0.001 | unsupported | 0.084 ± 0.000 | 0.152 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.187 ± 0.008 | 1.188 ± 0.008 | unsupported | 1.532 ± 0.017 | 1.211 ± 0.060 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.061 ± 0.002 | 0.082 ± 0.007 | unsupported | 0.160 ± 0.002 | 0.186 ± 0.003 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.708 ± 0.023 | 0.730 ± 0.085 | unsupported | 1.774 ± 0.006 | 1.882 ± 0.037 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.140 ± 0.002 | 0.176 ± 0.001 | unsupported | 0.041 ± 0.001 | 0.184 ± 0.004 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.438 ± 0.016 | 0.505 ± 0.036 | unsupported | 0.361 ± 0.002 | 1.383 ± 0.222 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.133 ± 0.008 | 0.146 ± 0.003 | unsupported | 0.136 ± 0.003 | 0.243 ± 0.014 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.708 ± 0.036 | 1.762 ± 0.046 | unsupported | 2.081 ± 0.059 | 2.289 ± 0.040 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.204 ± 0.004 | 0.236 ± 0.002 | unsupported | 0.038 ± 0.000 | 0.213 ± 0.006 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 1.004 ± 0.035 | 1.031 ± 0.028 | unsupported | 0.482 ± 0.005 | 1.632 ± 0.036 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.300 ± 0.003 | 0.321 ± 0.009 | unsupported | unsupported | 0.249 ± 0.008 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 4.329 ± 0.041 | 4.430 ± 0.013 | unsupported | unsupported | 3.255 ± 0.070 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.357 ± 0.008 | 0.399 ± 0.005 | unsupported | unsupported | 0.245 ± 0.008 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 2.017 ± 0.020 | 2.106 ± 0.815 | unsupported | unsupported | 1.958 ± 0.080 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.012 ± 0.001 | 0.015 ± 0.000 | unsupported | 0.014 ± 0.000 | 0.067 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.099 ± 0.003 | 0.111 ± 0.016 | unsupported | 0.090 ± 0.012 | 0.247 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.029 ± 0.001 | 0.038 ± 0.011 | unsupported | 0.015 ± 0.000 | 0.039 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.126 ± 0.005 | 0.104 ± 0.024 | unsupported | 0.103 ± 0.008 | 0.201 ± 0.007 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.031 ± 0.004 | 0.045 ± 0.010 | unsupported | 0.101 ± 0.003 | 0.175 ± 0.004 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.367 ± 0.068 | 0.385 ± 0.021 | unsupported | 1.386 ± 0.013 | 1.683 ± 0.080 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.070 ± 0.002 | 0.095 ± 0.001 | unsupported | 0.056 ± 0.001 | 0.155 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.226 ± 0.014 | 0.269 ± 0.029 | unsupported | 0.267 ± 0.028 | 1.023 ± 0.134 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.054 ± 0.001 | 0.055 ± 0.000 | unsupported | 0.118 ± 0.003 | 0.178 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.866 ± 0.006 | 0.877 ± 0.008 | unsupported | 1.835 ± 0.018 | 1.765 ± 0.021 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.036 ± 0.000 | 0.037 ± 0.016 | unsupported | 0.097 ± 0.003 | 0.146 ± 0.004 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.400 ± 0.017 | 0.399 ± 0.004 | unsupported | 0.704 ± 0.020 | 0.940 ± 0.028 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.056 ± 0.001 | 0.057 ± 0.001 | unsupported | 0.122 ± 0.002 | 0.179 ± 0.004 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.899 ± 0.002 | 0.912 ± 0.004 | unsupported | 1.857 ± 0.023 | 1.809 ± 0.013 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.037 ± 0.001 | 0.038 ± 0.001 | unsupported | 0.099 ± 0.003 | 0.147 ± 0.001 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.415 ± 0.004 | 0.416 ± 0.010 | unsupported | 0.711 ± 0.013 | 0.962 ± 0.007 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.128 ± 0.004 | 0.146 ± 0.004 | unsupported | 0.139 ± 0.001 | 0.220 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.685 ± 0.017 | 1.795 ± 0.060 | unsupported | 1.236 ± 0.011 | 2.223 ± 0.053 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.201 ± 0.002 | 0.238 ± 0.094 | unsupported | 0.036 ± 0.000 | 0.209 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.074 ± 0.184 | 1.040 ± 0.019 | unsupported | 0.472 ± 0.001 | 1.622 ± 0.055 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.182 ± 0.001 | 0.184 ± 0.001 | unsupported | 0.275 ± 0.009 | 0.321 ± 0.004 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 2.933 ± 0.011 | 2.913 ± 0.006 | unsupported | 4.332 ± 0.027 | 4.005 ± 0.012 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.082 ± 0.000 | 0.083 ± 0.002 | unsupported | 0.203 ± 0.001 | 0.218 ± 0.002 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.135 ± 0.005 | 1.138 ± 0.007 | unsupported | 1.563 ± 0.047 | 2.064 ± 0.024 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.773 ± 0.007 | 1.342 ± 0.010 | unsupported | 0.269 ± 0.003 | 0.237 ± 0.024 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 18.972 ± 0.173 | 29.415 ± 0.315 | unsupported | 5.903 ± 0.184 | 2.323 ± 0.023 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 1.011 ± 0.153 | 1.729 ± 0.153 | unsupported | 0.218 ± 0.003 | 0.169 ± 0.002 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 9.286 ± 0.199 | 21.594 ± 1.912 | unsupported | 2.302 ± 0.070 | 1.229 ± 0.024 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 28.216 ± 0.533 | - | unsupported | 26.096 ± 0.663 | - | 27.683 ± 0.543 | 37.961 ± 0.274 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 21.556 ± 0.994 | - | unsupported | 20.526 ± 0.274 | - | 31.354 ± 0.982 | 20.608 ± 0.668 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 21.621 ± 0.489 | - | unsupported | 26.076 ± 0.259 | - | 21.892 ± 0.448 | 22.538 ± 0.489 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 14.675 ± 0.475 | - | unsupported | 21.622 ± 3.959 | - | 22.277 ± 0.593 | 14.906 ± 0.636 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 13.867 ± 0.521 | - | unsupported | 21.400 ± 0.179 | - | 12.574 ± 0.291 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 14.907 ± 0.744 | - | unsupported | 22.518 ± 4.564 | - | 14.170 ± 0.610 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 27.863 ± 0.460 | - | unsupported | 27.058 ± 0.472 | - | 28.385 ± 0.709 | 36.770 ± 0.410 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 20.960 ± 0.682 | - | unsupported | 21.061 ± 0.730 | - | 31.885 ± 0.234 | 22.710 ± 1.560 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 49.096 ± 0.568 | - | unsupported | 50.696 ± 0.253 | - | 48.642 ± 0.257 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 23.073 ± 0.108 | - | unsupported | 39.189 ± 3.216 | - | 23.405 ± 0.101 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 49.365 ± 0.718 | - | unsupported | 50.377 ± 0.567 | - | 48.059 ± 0.163 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 22.989 ± 0.047 | - | unsupported | 36.552 ± 4.432 | - | 23.315 ± 1.487 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 27.990 ± 0.236 | - | unsupported | 27.228 ± 0.633 | - | 28.119 ± 0.554 | 38.156 ± 0.366 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 20.654 ± 0.408 | - | unsupported | 23.168 ± 1.016 | - | 31.233 ± 0.366 | 21.587 ± 2.132 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 21.235 ± 0.532 | - | unsupported | 21.331 ± 0.183 | - | 21.828 ± 0.597 | 26.419 ± 0.271 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 14.509 ± 0.369 | - | unsupported | 14.965 ± 0.304 | - | 25.107 ± 0.421 | 15.024 ± 0.620 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.550 ± 0.003 | unsupported | unsupported | 0.907 ± 0.009 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 9.301 ± 0.097 | unsupported | unsupported | 12.843 ± 0.143 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.622 ± 0.005 | unsupported | unsupported | 0.618 ± 0.004 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 6.525 ± 0.199 | unsupported | unsupported | 6.950 ± 0.106 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 28.146 ± 0.628 | - | unsupported | 26.343 ± 0.811 | - | 27.512 ± 0.328 | 37.813 ± 0.164 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 21.040 ± 0.499 | - | unsupported | 21.218 ± 0.836 | - | 31.165 ± 0.372 | 21.019 ± 0.817 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 109.751 ± 4.162 | - | 0.002 ± 0.000 | 111.701 ± 5.683 | 112.085 ± 5.545 | 136.171 ± 4.607 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 44.203 ± 0.349 | - | 0.003 ± 0.000 | 34.933 ± 11.119 | 56.894 ± 1.048 | 137.481 ± 9.027 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 67.095 ± 3.380 | - | 70.738 ± 3.958 | 69.947 ± 2.660 | 69.491 ± 3.000 | 78.135 ± 4.604 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 23.258 ± 0.753 | - | 22.876 ± 0.188 | 22.820 ± 0.737 | 35.917 ± 0.586 | 78.948 ± 16.340 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 224.914 ± 8.752 | - | 221.661 ± 3.659 | 216.230 ± 8.777 | 223.437 ± 3.710 | 253.054 ± 11.670 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 71.177 ± 1.693 | - | 72.012 ± 2.247 | 70.538 ± 1.890 | 114.027 ± 5.850 | 258.012 ± 11.778 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 69.805 ± 4.469 | - | 68.976 ± 1.832 | 73.171 ± 1.459 | 69.594 ± 3.412 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 42.416 ± 0.656 | - | 42.527 ± 0.294 | 24.234 ± 0.630 | 35.852 ± 0.158 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 140.384 ± 8.021 | - | 0.002 ± 0.000 | 125.148 ± 4.892 | 119.178 ± 3.543 | 157.827 ± 8.487 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 74.315 ± 2.409 | - | 0.003 ± 0.000 | 39.824 ± 1.050 | 46.504 ± 8.583 | 160.263 ± 6.497 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 87.034 ± 1.558 | - | 0.002 ± 0.000 | 79.723 ± 0.559 | 84.969 ± 0.888 | 214.939 ± 9.615 | 129.256 ± 2.275 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 39.052 ± 5.601 | - | 0.002 ± 0.000 | 80.432 ± 0.625 | 43.747 ± 0.880 | 214.541 ± 2.254 | 52.335 ± 1.698 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 59.656 ± 2.612 | - | 60.328 ± 2.514 | 64.833 ± 2.318 | 66.000 ± 2.603 | 81.449 ± 3.185 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 21.424 ± 0.435 | - | 21.188 ± 0.187 | 23.353 ± 0.350 | 33.294 ± 2.430 | 82.905 ± 2.637 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 60.493 ± 1.968 | - | 59.525 ± 2.725 | 65.637 ± 2.358 | 64.166 ± 1.043 | 82.870 ± 1.781 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 20.552 ± 0.386 | - | 20.338 ± 0.165 | 22.593 ± 0.252 | 32.757 ± 0.752 | 83.720 ± 3.923 | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 7.813611 | 488350.69 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 7.334779 | 458423.69 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 7.628707 | 476794.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.572591 | 410786.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 7.406574 | 462910.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 8.367043 | 522940.19 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 19.202209 | 1200138.06 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 18.207297 | 1137956.06 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 6.717590 | 419849.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 6.944263 | 434016.44 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 12.173592 | 760849.50 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 9.736340 | 608521.25 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 5.372354 | 335772.12 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 5.487418 | 342963.62 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 154.754754 | 9672172.12 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 142.648779 | 8915548.69 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 100.488673 | 6280542.06 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 101.784657 | 6361541.06 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 20.204132 | 1262758.25 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 8.167032 | 510439.50 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 5.669497 | 354343.56 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 5.977060 | 373566.25 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 159.496597 | 9968537.31 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 90.348255 | 5646765.94 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 260.821088 | 16301318.00 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 66.966573 | 4185410.81 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.488325 | 1093020.31 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 16.269766 | 1016860.38 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 377.714333 | 23607145.81 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 370.826186 | 23176636.62 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 17.515656 | 1094728.50 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 15.990997 | 999437.31 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 229.367308 | 14335456.75 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 225.940248 | 14121265.50 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.750424 | 1109401.50 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.833438 | 1114589.88 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 365.152460 | 22822028.75 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 344.952411 | 21559525.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 17.307829 | 1081739.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 15.901440 | 993840.00 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 217.867485 | 13616717.81 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 210.726484 | 13170405.25 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.516171 | 148690.17 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.072537 | 157383.39 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.881549 | 138774.20 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.961778 | 140027.78 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.229269 | 1278658.62 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 19.835478 | 2479434.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.803944 | 2225493.00 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.734525 | 2216815.62 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.370387 | 130787.30 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.911567 | 76743.23 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.161118 | 80642.47 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.248621 | 82009.70 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.790566 | 348820.75 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.099345 | 1012418.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.267593 | 908449.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.270688 | 908836.00 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.532898 | 258326.53 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.765894 | 136967.09 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.990613 | 77978.33 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.051737 | 78933.39 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 21.906966 | 2738370.75 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.311270 | 1788908.75 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.110357 | 1263794.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.047660 | 1255957.50 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.278080 | 160595.00 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.577148 | 87142.94 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.000231 | 46878.61 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.055173 | 47737.08 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.312449 | 1664056.12 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.646733 | 455841.62 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.554507 | 569313.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.561580 | 570197.50 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.608894 | 212638.97 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.257048 | 285266.38 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.583803 | 196621.92 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.636522 | 197445.66 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.840557 | 1855069.62 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 35.449583 | 4431197.88 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.247115 | 3155889.38 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 24.985898 | 3123237.25 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.735970 | 152124.53 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.359482 | 83741.91 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.468521 | 85445.64 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.553579 | 86774.67 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.686909 | 1210863.62 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.257721 | 1532215.12 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.493141 | 1186642.62 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.502823 | 1187852.88 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.908657 | 186072.77 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.228337 | 159817.77 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.913778 | 61152.78 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.247271 | 81988.61 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.057982 | 1882247.75 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.191998 | 1773999.75 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.663747 | 707968.38 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.840637 | 730079.62 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.794614 | 184290.84 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.630159 | 41096.23 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.975556 | 140243.06 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 11.289678 | 176401.22 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.061164 | 1382645.50 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.891456 | 361432.00 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.504529 | 438066.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.043132 | 505391.50 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.555429 | 243053.58 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.708877 | 136076.20 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.512101 | 133001.58 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.328250 | 145753.91 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.313074 | 2289134.25 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.649304 | 2081163.00 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.667320 | 1708415.00 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.094286 | 1761785.75 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.626034 | 212906.78 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.403627 | 37556.67 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 13.037786 | 203715.41 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 15.092134 | 235814.59 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.059026 | 1632378.25 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.852055 | 481506.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.032564 | 1004070.50 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.249028 | 1031128.50 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.906002 | 248531.28 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 19.215873 | 300248.02 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 20.523849 | 320685.14 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 26.039622 | 3254952.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 34.632362 | 4329045.25 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 35.441569 | 4430196.12 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.701640 | 245338.12 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 22.846845 | 356981.95 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 25.521088 | 398767.00 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 15.662968 | 1957871.00 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.134306 | 2016788.25 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.849118 | 2106139.75 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.276063 | 66813.48 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.865291 | 13520.17 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.784060 | 12250.94 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.967141 | 15111.58 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.974597 | 246824.62 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.722886 | 90360.75 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.795792 | 99474.00 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.884137 | 110517.12 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.464079 | 38501.23 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.940982 | 14702.84 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.875274 | 29301.16 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.406615 | 37603.36 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.605570 | 200696.25 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.826459 | 103307.38 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.008108 | 126013.50 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.835596 | 104449.50 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.219864 | 175310.38 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.472313 | 101129.89 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.969518 | 30773.72 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.852091 | 44563.92 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.467531 | 1683441.38 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.090763 | 1386345.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.935046 | 366880.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.080116 | 385014.50 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.930802 | 155168.78 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.555162 | 55549.41 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.485589 | 70087.33 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.093455 | 95210.23 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.180544 | 1022568.00 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.137060 | 267132.50 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.806846 | 225855.75 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.149464 | 268683.00 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.382186 | 177846.66 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.548697 | 117948.39 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.443392 | 53803.00 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.512671 | 54885.48 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.123091 | 1765386.38 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.680187 | 1835023.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.930085 | 866260.62 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.018509 | 877313.62 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.359720 | 146245.62 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.190550 | 96727.34 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.302470 | 35976.09 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.375587 | 37118.55 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.516197 | 939524.62 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.632300 | 704037.50 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.202628 | 400328.50 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.188171 | 398521.38 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.449452 | 178897.69 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.788424 | 121694.12 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.614150 | 56471.09 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.659865 | 57185.39 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.468203 | 1808525.38 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.854953 | 1856869.12 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.193395 | 899174.38 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.293853 | 911731.62 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.425352 | 147271.12 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.338886 | 99045.09 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.364076 | 36938.69 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.416072 | 37751.12 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.692666 | 961583.25 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.684799 | 710599.88 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.318785 | 414848.12 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.326329 | 415791.12 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.056337 | 219630.27 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.868725 | 138573.83 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.176376 | 127755.87 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.374307 | 146473.55 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.787515 | 2223439.38 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.890888 | 1236361.00 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.479441 | 1684930.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.357757 | 1794719.62 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.388321 | 209192.52 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.290456 | 35788.38 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 12.842271 | 200660.48 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 15.254497 | 238351.52 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 12.972079 | 1621509.88 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.776374 | 472046.75 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.589040 | 1073630.00 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.321292 | 1040161.50 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 20.542563 | 320977.55 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.580458 | 274694.66 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.677967 | 182468.23 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.771080 | 183923.12 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 32.042221 | 4005277.62 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.657859 | 4332232.38 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.464586 | 2933073.25 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.304207 | 2913025.88 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.925422 | 217584.72 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 13.012155 | 203314.92 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.239794 | 81871.78 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.315846 | 83060.09 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.508302 | 2063537.75 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.504661 | 1563082.62 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.078077 | 1134759.62 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.102963 | 1137870.38 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.177317 | 237145.58 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.207133 | 268861.45 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 49.502603 | 773478.17 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 85.875826 | 1341809.78 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.580826 | 2322603.25 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 47.221065 | 5902633.12 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 151.777228 | 18972153.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 235.323386 | 29415423.25 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.831350 | 169239.84 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 13.945995 | 217906.17 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 64.672556 | 1010508.69 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 110.683637 | 1729431.83 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.834865 | 1229358.12 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 18.413967 | 2301745.88 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 74.285216 | 9285652.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 172.750457 | 21593807.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 14.517003 | 907312.69 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 8.798696 | 549918.50 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 205.493837 | 12843364.81 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 148.816516 | 9301032.25 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.880543 | 617533.94 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 9.955760 | 622235.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 111.196294 | 6949768.38 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 104.397551 | 6524846.94 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.023293 | 1455.81 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002645 | 165.31 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.023324 | 1457.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002354 | 147.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.021801 | 1362.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.007294 | 455.88 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000141 | 8.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.021660 | 1353.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.006773 | 423.31 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000140 | 8.75 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.022492 | 1405.75 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005830 | 364.38 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.022301 | 1393.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005340 | 333.75 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.020138 | 1258.62 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003186 | 199.12 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000131 | 8.19 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.020488 | 1280.50 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003086 | 192.88 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 24.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 11.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 24.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 24.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 24.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 11.452 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 11.452 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 23.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 23.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 23.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 10.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 10.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 20.5x faster than the slowest successful cell (`julia-base`, 7.547 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 14.7x faster than the slowest successful cell (`julia-base`, 7.547 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`julia-base`, 7.547 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `jax-cpu` is 13.4x faster than the slowest successful cell (`julia-base`, 7.910 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `pytorch-cpu` is 23.8x faster than the slowest successful cell (`julia-base`, 7.910 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`julia-base`, 7.910 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`julia-base`, 7.910 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`julia-base`, 6.564 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 14.0x faster than the slowest successful cell (`julia-base`, 8.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 25.9x faster than the slowest successful cell (`julia-base`, 8.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 16.9x faster than the slowest successful cell (`julia-base`, 8.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`julia-base`, 8.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 5286.1x faster than the slowest successful cell (`julia-base`, 9.795 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `jax-cpu` is 10.5x faster than the slowest successful cell (`julia-base`, 9.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 12.9x faster than the slowest successful cell (`julia-base`, 9.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`julia-base`, 9.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 3442.2x faster than the slowest successful cell (`julia-base`, 9.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 23.6x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.023 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `pytorch-cpu` is 14.0x faster than the slowest successful cell (`jax-cpu`, 20.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`jax-cpu`, 20.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 16.8x faster than the slowest successful cell (`jax-cpu`, 20.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 29.415 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 10.2x faster than the slowest successful cell (`tenferro-eager`, 1.729 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 17.6x faster than the slowest successful cell (`tenferro-eager`, 21.594 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 78124.6x faster than the slowest successful cell (`julia-base`, 136.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 52373.6x faster than the slowest successful cell (`julia-base`, 137.481 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 90032.3x faster than the slowest successful cell (`julia-base`, 157.827 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 61997.3x faster than the slowest successful cell (`julia-base`, 160.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 129950.8x faster than the slowest successful cell (`julia-base`, 214.939 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 88470.6x faster than the slowest successful cell (`julia-base`, 214.541 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 194.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 150.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 156.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 174.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 157.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 160.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9049.7 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 90.5x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 6245.4 GB/s at 0.003 ms. The conservative physical lower bound is 0.042 ms, so the cell is 15.6x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 154007.7 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1540.1x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 102261.1 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 255.7x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 153129.2 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1531.3x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 103843.5 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 259.6x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 81147.4 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 811.5x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 55347.5 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 138.4x faster than the bound.

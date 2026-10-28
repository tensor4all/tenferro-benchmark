# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/public_api/20261028_004801/run.yaml`
- Timestamp: `20261028_004801`

Latest run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4 8`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/zen5-cpu/cpu/public_api/20261028_004801`.

- tenferro-rs commit: `8fec1ca70852b930c092e7177c30bb8af7594a57`

## CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `enabled`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/zen5-cpu/cpu/public_api/20261028_004801/run_t1.yaml`
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

- Run metadata: `data/results/zen5-cpu/cpu/public_api/20261028_004801/run_t4.yaml`
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

- Run metadata: `data/results/zen5-cpu/cpu/public_api/20261028_004801/run_t8.yaml`
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

- CSV: `data/results/zen5-cpu/cpu/public_api/20261028_004801/cpu_public_api_t1_20261028_004801.csv`
- CSV: `data/results/zen5-cpu/cpu/public_api/20261028_004801/cpu_public_api_t4_20261028_004801.csv`
- CSV: `data/results/zen5-cpu/cpu/public_api/20261028_004801/cpu_public_api_t8_20261028_004801.csv`
- Source table: `data/results/zen5-cpu/cpu/public_api/20261028_004801/cpu_public_api_20261028_004801.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.439 ± 0.005 | - | 2.546 ± 0.008 | 2.953 ± 0.009 | 2.531 ± 0.008 | 2.550 ± 0.820 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 0.746 ± 0.005 | - | 1.211 ± 0.926 | 0.853 ± 0.006 | 1.215 ± 0.010 | 1.146 ± 0.889 | - |
| cpu/complex | `cholesky` | c64 | 8 | `448x448` | 0.788 ± 0.008 | - | 2.556 ± 0.640 | 0.712 ± 0.005 | 1.923 ± 1.094 | 1.276 ± 0.627 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 46.554 ± 0.736 | - | 46.547 ± 1.529 | 46.428 ± 0.981 | 54.619 ± 8.423 | 51.708 ± 0.844 | 59.576 ± 1.660 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 18.140 ± 0.200 | - | 17.504 ± 0.108 | 18.599 ± 0.116 | 29.292 ± 0.558 | 52.507 ± 0.846 | 27.885 ± 1.721 |
| cpu/complex | `conj` | c64 | 8 | `16777216` | 15.856 ± 0.445 | - | 15.163 ± 0.215 | 16.413 ± 0.210 | 26.903 ± 0.494 | 69.735 ± 1.215 | 28.469 ± 0.679 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 46.621 ± 0.226 | - | 55.341 ± 0.700 | 46.200 ± 0.292 | 52.712 ± 1.086 | 56.704 ± 0.351 | 59.592 ± 0.595 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 14.611 ± 0.285 | - | 14.026 ± 0.166 | 15.108 ± 0.222 | 27.092 ± 0.725 | 54.204 ± 0.390 | 19.536 ± 0.911 |
| cpu/complex | `div` | c64 | 8 | `8388608` | 11.630 ± 0.315 | - | 10.984 ± 0.373 | 11.380 ± 0.134 | 21.510 ± 0.322 | 77.617 ± 0.087 | 20.663 ± 2.256 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 31.076 ± 0.107 | - | 31.139 ± 0.040 | 30.198 ± 0.043 | 36.868 ± 0.059 | 32.319 ± 0.931 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 11.089 ± 10.690 | - | 11.074 ± 0.020 | 10.650 ± 0.025 | 10.405 ± 0.140 | 8.558 ± 0.135 | - |
| cpu/complex | `dot_general` | c64 | 8 | `640x640` | 13.668 ± 5.355 | - | 7.469 ± 0.310 | 6.877 ± 0.050 | 8.807 ± 1.063 | 6.932 ± 0.297 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 31.539 ± 0.089 | - | unsupported | 30.833 ± 0.039 | 37.380 ± 0.055 | 31.964 ± 0.216 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 11.729 ± 0.021 | - | unsupported | 11.139 ± 0.055 | 10.850 ± 0.331 | 8.775 ± 0.986 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 8 | `640x640` | 8.592 ± 0.319 | - | unsupported | 7.286 ± 0.041 | 8.572 ± 0.402 | 7.112 ± 0.457 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 5.623 ± 0.013 | - | 5.627 ± 0.008 | 6.089 ± 0.015 | 6.271 ± 0.026 | 6.967 ± 0.022 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 5.184 ± 0.016 | - | 5.215 ± 0.007 | 5.237 ± 0.022 | 6.999 ± 0.002 | 6.343 ± 0.016 | - |
| cpu/complex | `eig` | c64 | 8 | `112x112` | 7.759 ± 0.020 | - | 8.485 ± 0.030 | 7.821 ± 0.021 | 10.016 ± 0.020 | 10.057 ± 0.029 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 40.295 ± 0.123 | - | 41.080 ± 0.046 | 55.229 ± 0.480 | 44.639 ± 0.835 | 36.978 ± 2.935 | 37.299 ± 2.705 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 10.553 ± 0.015 | - | 14.741 ± 10.797 | 14.342 ± 0.213 | 22.667 ± 0.193 | 36.585 ± 0.210 | 12.147 ± 1.191 |
| cpu/complex | `exp` | c64 | 8 | `4194304` | 8.003 ± 0.028 | - | 16.286 ± 8.623 | 11.185 ± 0.445 | 24.649 ± 0.605 | 55.286 ± 0.790 | 12.477 ± 1.157 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 52.057 ± 0.040 | - | 51.926 ± 0.108 | 70.089 ± 0.336 | 84.762 ± 0.164 | 61.798 ± 0.135 | 64.859 ± 0.234 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 13.608 ± 0.030 | - | 13.710 ± 6.025 | 18.811 ± 0.142 | 21.936 ± 0.528 | 63.365 ± 0.824 | 19.135 ± 0.896 |
| cpu/complex | `log` | c64 | 8 | `4194304` | 10.159 ± 0.040 | - | 10.234 ± 10.505 | 14.622 ± 0.446 | 17.215 ± 0.442 | 96.941 ± 0.249 | 17.942 ± 0.805 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 34.618 ± 0.181 | - | 34.822 ± 0.130 | 32.575 ± 0.198 | 38.019 ± 0.422 | 32.877 ± 0.201 | 39.247 ± 0.285 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 13.288 ± 0.561 | - | 12.702 ± 0.343 | 11.812 ± 0.338 | 19.386 ± 0.522 | 30.778 ± 0.110 | 16.097 ± 1.125 |
| cpu/complex | `mul` | c64 | 8 | `8388608` | 10.624 ± 0.312 | - | 10.412 ± 0.493 | 10.397 ± 0.201 | 16.757 ± 0.224 | 39.671 ± 0.092 | 19.413 ± 2.367 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.041 ± 0.081 | - | 9.534 ± 0.014 | 9.713 ± 0.013 | 8.010 ± 0.149 | 2.430 ± 0.005 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.240 ± 0.031 | - | 9.175 ± 0.054 | 2.562 ± 0.032 | 2.608 ± 0.150 | 2.428 ± 0.008 | - |
| cpu/complex | `norm_fro` | c64 | 8 | `2048x1536` | 1.591 ± 0.085 | - | 7.866 ± 3.281 | 1.956 ± 0.019 | 2.720 ± 0.043 | 3.851 ± 0.018 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.560 ± 0.008 | - | 4.676 ± 0.020 | 5.399 ± 0.009 | 4.320 ± 0.008 | 5.539 ± 0.043 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 1.933 ± 0.008 | - | 2.387 ± 0.450 | 2.145 ± 0.007 | 2.686 ± 0.028 | 2.933 ± 0.158 | - |
| cpu/complex | `qr` | c64 | 8 | `256x256` | 2.399 ± 0.015 | - | 3.570 ± 0.827 | 2.790 ± 0.012 | 3.812 ± 0.021 | 4.185 ± 0.043 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.146 ± 0.007 | - | 3.182 ± 0.036 | 3.190 ± 0.014 | 3.123 ± 0.006 | 3.062 ± 0.014 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 1.116 ± 0.026 | - | 1.567 ± 0.008 | 1.091 ± 0.012 | 1.534 ± 0.102 | 1.284 ± 0.013 | - |
| cpu/complex | `solve` | c64 | 8 | `384x384,rhs=8` | 1.150 ± 0.010 | - | 2.251 ± 0.156 | 1.262 ± 0.009 | 2.013 ± 0.093 | 1.491 ± 0.023 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.401 ± 0.017 | - | 3.405 ± 0.007 | 4.096 ± 0.012 | 2.920 ± 0.004 | 2.933 ± 0.014 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 1.760 ± 0.016 | - | 2.208 ± 0.006 | 1.941 ± 0.006 | 2.092 ± 0.036 | 1.972 ± 0.013 | - |
| cpu/complex | `svd` | c64 | 8 | `160x160` | 2.153 ± 0.041 | - | 2.868 ± 0.012 | 2.456 ± 0.060 | 3.146 ± 0.051 | 3.168 ± 0.037 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 31.142 ± 0.087 | - | 31.024 ± 0.347 | 30.248 ± 0.038 | 36.879 ± 0.025 | 31.477 ± 0.031 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 11.080 ± 0.041 | - | 11.088 ± 0.142 | 10.711 ± 0.036 | 10.391 ± 0.097 | 8.341 ± 0.063 | - |
| cpu/complex | `tensordot` | c64 | 8 | `640x640` | 7.628 ± 0.385 | - | 7.092 ± 0.060 | 6.911 ± 0.077 | 8.317 ± 0.987 | 6.683 ± 0.026 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 29.906 ± 0.025 | - | unsupported | 30.122 ± 0.065 | 29.537 ± 0.065 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 16.356 ± 7.294 | - | unsupported | 9.954 ± 0.057 | 8.219 ± 0.114 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 8 | `1024x1024` | 13.680 ± 1.658 | - | unsupported | 8.649 ± 0.030 | 7.046 ± 0.666 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 46.078 ± 1.267 | - | 47.237 ± 1.374 | 46.327 ± 0.222 | 55.931 ± 2.112 | 46.937 ± 2.161 | 87.355 ± 0.930 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 17.094 ± 0.099 | - | 16.731 ± 0.099 | 17.751 ± 0.260 | 31.457 ± 1.161 | 47.965 ± 2.290 | 32.556 ± 3.229 |
| cpu/elementwise_reduction | `abs` | f64 | 8 | `33554432` | 14.611 ± 0.075 | - | 14.352 ± 0.188 | 15.575 ± 0.220 | 27.117 ± 0.226 | 71.695 ± 0.813 | 37.607 ± 0.684 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 57.781 ± 0.984 | - | 55.794 ± 0.649 | 57.687 ± 0.782 | 70.241 ± 0.458 | 53.893 ± 0.525 | 102.452 ± 0.969 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 22.101 ± 0.176 | - | 21.079 ± 0.069 | 22.911 ± 0.219 | 36.325 ± 0.846 | 53.866 ± 0.367 | 38.333 ± 2.676 |
| cpu/elementwise_reduction | `add` | f64 | 8 | `33554432` | 19.388 ± 0.207 | - | 18.702 ± 0.066 | 20.929 ± 0.214 | 31.735 ± 0.399 | 69.502 ± 0.519 | 41.867 ± 3.882 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 9.909 ± 0.126 | - | 28.479 ± 0.159 | 28.954 ± 0.621 | 24.470 ± 0.204 | 33.428 ± 2.339 | 36.403 ± 1.054 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 5.077 ± 0.076 | - | 16.336 ± 7.387 | 12.675 ± 0.424 | 14.916 ± 0.399 | 33.492 ± 0.717 | 9.810 ± 4.728 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 8 | `4194304` | 5.783 ± 0.149 | - | 14.034 ± 6.246 | 13.536 ± 0.819 | 12.747 ± 0.240 | 52.281 ± 0.425 | 12.742 ± 7.645 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 7.068 ± 0.029 | - | 7.185 ± 0.017 | 18.478 ± 0.140 | 22.304 ± 0.262 | 16.276 ± 2.897 | 26.095 ± 3.277 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 5.691 ± 0.053 | - | 5.627 ± 0.098 | 6.981 ± 0.117 | 11.192 ± 0.151 | 14.113 ± 3.081 | 9.302 ± 5.624 |
| cpu/elementwise_reduction | `clamp` | f64 | 8 | `8388608` | 6.108 ± 0.020 | - | 6.780 ± 0.354 | 6.376 ± 0.193 | 9.723 ± 0.269 | 17.584 ± 4.498 | 10.957 ± 8.267 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 13.279 ± 0.031 | - | 13.155 ± 0.089 | 23.432 ± 0.112 | 22.905 ± 0.114 | 13.390 ± 0.060 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 11.414 ± 0.020 | - | 11.446 ± 0.630 | 12.087 ± 0.166 | 14.316 ± 0.249 | 13.557 ± 0.099 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 8 | `33554432` | 11.645 ± 0.022 | - | 11.677 ± 0.025 | 12.112 ± 0.113 | 13.431 ± 0.082 | 14.577 ± 0.149 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 9.342 ± 0.027 | - | 9.372 ± 0.021 | 31.191 ± 1.153 | 59.968 ± 0.186 | 31.405 ± 3.935 | 33.596 ± 3.360 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 3.088 ± 2.036 | - | 5.316 ± 0.012 | 9.232 ± 0.154 | 31.282 ± 0.285 | 32.882 ± 4.608 | 9.372 ± 4.945 |
| cpu/elementwise_reduction | `cos` | f64 | 8 | `8388608` | 3.350 ± 0.044 | - | 5.098 ± 0.400 | 6.904 ± 0.337 | 31.749 ± 0.654 | 47.484 ± 5.870 | 11.867 ± 9.994 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 56.094 ± 0.600 | - | 53.939 ± 0.857 | 57.005 ± 0.708 | 64.742 ± 1.051 | 53.101 ± 0.757 | 95.174 ± 0.699 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 21.155 ± 0.111 | - | 20.453 ± 0.111 | 22.216 ± 0.258 | 36.445 ± 1.301 | 56.280 ± 1.508 | 41.615 ± 3.671 |
| cpu/elementwise_reduction | `div` | f64 | 8 | `33554432` | 19.044 ± 0.134 | - | 18.588 ± 0.122 | 20.282 ± 0.165 | 31.670 ± 0.591 | 78.448 ± 0.295 | 40.324 ± 4.904 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 5.360 ± 0.012 | - | 5.367 ± 0.015 | 23.602 ± 0.182 | 15.857 ± 0.640 | 26.135 ± 3.930 | 28.317 ± 3.251 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 2.901 ± 0.022 | - | 3.751 ± 0.023 | 7.438 ± 0.267 | 8.969 ± 1.276 | 27.251 ± 4.363 | 7.791 ± 4.504 |
| cpu/elementwise_reduction | `exp` | f64 | 8 | `8388608` | 3.401 ± 0.109 | - | 4.336 ± 1.500 | 5.705 ± 0.615 | 8.891 ± 0.236 | 40.129 ± 5.815 | 9.327 ± 9.031 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 7.519 ± 0.036 | - | 7.527 ± 0.042 | 11.802 ± 0.150 | 10.853 ± 0.042 | 15.669 ± 3.022 | 17.098 ± 2.688 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 1.946 ± 0.008 | - | 4.154 ± 0.013 | 3.651 ± 0.150 | 5.798 ± 0.066 | 15.839 ± 0.787 | 4.640 ± 3.456 |
| cpu/elementwise_reduction | `expm1` | f64 | 8 | `4194304` | 1.606 ± 0.027 | - | 3.687 ± 0.030 | 2.876 ± 0.220 | 6.089 ± 0.244 | 23.819 ± 0.351 | 6.763 ± 6.258 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 9.449 ± 0.013 | - | 9.467 ± 0.043 | 25.692 ± 0.459 | 29.137 ± 0.437 | 32.975 ± 2.395 | 37.381 ± 2.852 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 2.983 ± 0.020 | - | 2.964 ± 0.036 | 7.553 ± 0.252 | 15.139 ± 0.389 | 33.202 ± 3.155 | 9.525 ± 6.105 |
| cpu/elementwise_reduction | `log` | f64 | 8 | `8388608` | 3.324 ± 0.041 | - | 5.207 ± 0.768 | 6.238 ± 0.385 | 14.793 ± 0.460 | 50.140 ± 5.287 | 7.870 ± 8.860 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 4.576 ± 0.015 | - | 4.606 ± 0.016 | 10.825 ± 0.154 | 15.447 ± 0.058 | 18.672 ± 2.308 | 20.122 ± 2.615 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 1.327 ± 0.030 | - | 2.635 ± 0.016 | 3.322 ± 0.125 | 9.277 ± 0.118 | 18.648 ± 0.174 | 5.363 ± 4.019 |
| cpu/elementwise_reduction | `log1p` | f64 | 8 | `4194304` | 1.738 ± 0.031 | - | 2.971 ± 0.039 | 2.836 ± 0.283 | 8.525 ± 0.382 | 29.044 ± 0.302 | 4.480 ± 8.252 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.105 ± 0.002 | 0.118 ± 0.001 | unsupported | unsupported | 0.179 ± 0.000 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.050 ± 0.001 | 0.065 ± 0.002 | unsupported | unsupported | 0.071 ± 0.015 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 8 | `1024x64` | 0.071 ± 0.002 | 0.093 ± 2.757 | unsupported | unsupported | 0.164 ± 0.006 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 60.526 ± 0.552 | - | 59.061 ± 0.457 | 57.734 ± 0.206 | 60.213 ± 1.235 | 58.489 ± 0.821 | 104.418 ± 0.830 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 21.928 ± 0.112 | - | 21.365 ± 0.116 | 22.488 ± 0.219 | 34.074 ± 0.220 | 57.340 ± 0.841 | 39.983 ± 3.413 |
| cpu/elementwise_reduction | `maximum` | f64 | 8 | `33554432` | 19.343 ± 0.124 | - | 18.660 ± 0.101 | 20.002 ± 0.123 | 32.035 ± 0.235 | 78.189 ± 0.816 | 42.240 ± 0.558 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 61.866 ± 0.677 | - | 58.475 ± 0.900 | 57.661 ± 0.256 | 60.260 ± 0.958 | 58.996 ± 1.477 | 104.245 ± 0.458 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 22.043 ± 0.160 | - | 21.424 ± 0.113 | 22.483 ± 0.265 | 35.042 ± 1.147 | 58.840 ± 0.592 | 39.474 ± 2.717 |
| cpu/elementwise_reduction | `minimum` | f64 | 8 | `33554432` | 19.287 ± 0.093 | - | 18.715 ± 0.177 | 20.010 ± 0.180 | 32.008 ± 0.223 | 72.968 ± 1.932 | 38.662 ± 4.216 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 56.975 ± 1.268 | - | 56.043 ± 0.715 | 56.197 ± 0.792 | 70.393 ± 0.478 | 55.384 ± 0.918 | 101.466 ± 0.679 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 22.120 ± 0.118 | - | 21.038 ± 0.049 | 22.874 ± 0.125 | 36.814 ± 0.776 | 59.090 ± 0.267 | 39.770 ± 3.901 |
| cpu/elementwise_reduction | `mul` | f64 | 8 | `33554432` | 19.598 ± 0.191 | - | 18.756 ± 0.079 | 20.881 ± 0.177 | 31.791 ± 0.248 | 79.269 ± 0.369 | 41.527 ± 2.813 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 45.855 ± 1.563 | - | 48.575 ± 2.474 | 47.070 ± 0.669 | 49.140 ± 8.156 | 48.259 ± 1.717 | 86.884 ± 0.963 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 16.944 ± 0.160 | - | 16.825 ± 0.195 | 17.624 ± 0.322 | 30.350 ± 0.203 | 51.064 ± 1.049 | 35.322 ± 3.004 |
| cpu/elementwise_reduction | `neg` | f64 | 8 | `33554432` | 14.595 ± 0.069 | - | 14.286 ± 5.027 | 15.520 ± 0.179 | 27.101 ± 0.182 | 69.935 ± 0.743 | 35.182 ± 3.440 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 28.251 ± 0.057 | - | 28.170 ± 0.084 | 22.420 ± 0.229 | 35.425 ± 0.201 | 72.059 ± 3.289 | 50.442 ± 3.049 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 7.431 ± 0.015 | - | 15.118 ± 7.776 | 6.062 ± 0.266 | 19.822 ± 0.195 | 71.578 ± 0.833 | 14.068 ± 3.825 |
| cpu/elementwise_reduction | `pow` | f64 | 8 | `4194304` | 5.644 ± 0.024 | - | 11.695 ± 5.926 | 4.864 ± 0.202 | 17.818 ± 0.258 | 112.591 ± 0.441 | 10.652 ± 5.786 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 9.122 ± 0.017 | - | 9.048 ± 0.031 | 7.457 ± 0.020 | 38.309 ± 0.109 | 8.862 ± 0.113 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 5.323 ± 0.024 | - | 5.862 ± 0.220 | 5.300 ± 0.063 | 19.533 ± 0.093 | 9.276 ± 0.084 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 8 | `8192x4096` | 5.329 ± 0.013 | - | 5.935 ± 0.253 | 5.414 ± 0.033 | 20.915 ± 0.381 | 10.314 ± 0.363 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.154 ± 0.014 | - | 1.170 ± 0.012 | 17.859 ± 0.733 | 1.254 ± 0.088 | 1.121 ± 0.151 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.400 ± 0.017 | - | 1.126 ± 0.022 | 4.974 ± 0.120 | 0.869 ± 0.051 | 0.879 ± 0.218 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 8 | `2048x2048` | 0.661 ± 0.068 | - | 1.607 ± 0.704 | 5.152 ± 0.883 | 1.011 ± 0.362 | 1.484 ± 0.276 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.001 ± 0.068 | - | 1.002 ± 0.015 | 1.701 ± 0.014 | 2.514 ± 0.017 | 0.771 ± 0.216 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.819 ± 0.030 | - | 1.927 ± 0.059 | 0.465 ± 0.018 | 1.296 ± 0.076 | 0.709 ± 0.249 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 8 | `2048x2048` | 0.594 ± 0.037 | - | 1.711 ± 0.622 | 0.654 ± 0.030 | 1.816 ± 0.401 | 0.810 ± 0.197 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 8.390 ± 0.083 | - | 8.317 ± 0.033 | 7.459 ± 0.021 | 38.345 ± 0.105 | 9.918 ± 0.074 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 5.317 ± 0.031 | - | 6.584 ± 0.286 | 5.274 ± 0.021 | 19.518 ± 0.071 | 10.665 ± 0.025 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 8 | `8192x4096` | 5.327 ± 0.019 | - | 5.808 ± 0.226 | 5.392 ± 0.049 | 20.615 ± 0.219 | 11.889 ± 0.140 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.072 ± 0.008 | - | 1.081 ± 0.015 | 18.235 ± 0.056 | 1.217 ± 0.111 | 1.157 ± 0.313 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.408 ± 0.014 | - | 0.414 ± 0.023 | 5.082 ± 0.091 | 1.002 ± 0.335 | 1.133 ± 0.267 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 8 | `2048x2048` | 1.371 ± 0.071 | - | 1.954 ± 0.875 | 5.271 ± 0.697 | 0.968 ± 0.378 | 1.261 ± 0.068 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 6.080 ± 0.044 | - | 6.093 ± 0.047 | 6.622 ± 0.023 | 9.896 ± 0.019 | 3.614 ± 0.047 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.849 ± 0.045 | - | 7.411 ± 0.118 | 2.524 ± 0.009 | 5.085 ± 0.909 | 3.556 ± 0.050 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 8 | `4096x4096` | 2.740 ± 0.014 | - | 4.937 ± 2.225 | 2.701 ± 0.049 | 5.972 ± 0.107 | 4.052 ± 0.028 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 5.980 ± 0.155 | - | 5.883 ± 0.022 | 5.921 ± 0.038 | 18.801 ± 0.129 | 5.723 ± 0.034 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 5.254 ± 0.015 | - | 5.772 ± 0.028 | 5.282 ± 0.020 | 10.227 ± 0.400 | 6.210 ± 0.401 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 8 | `8192x4096` | 5.409 ± 0.013 | - | 5.373 ± 0.072 | 5.433 ± 0.010 | 10.009 ± 0.042 | 6.298 ± 0.016 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.604 ± 0.011 | - | 0.629 ± 0.115 | 2.546 ± 0.017 | 1.312 ± 0.093 | 0.755 ± 0.177 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.389 ± 0.014 | - | 1.311 ± 0.040 | 1.458 ± 0.110 | 0.850 ± 0.067 | 0.702 ± 0.182 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 8 | `2048x2048` | 0.540 ± 0.013 | - | 1.532 ± 0.959 | 1.806 ± 0.021 | 0.916 ± 0.371 | 0.845 ± 0.198 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.653 ± 0.106 | - | 0.722 ± 0.094 | 0.597 ± 0.176 | 0.907 ± 0.272 | 0.730 ± 0.195 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.479 ± 0.014 | - | 2.070 ± 0.033 | 0.409 ± 0.011 | 0.571 ± 0.056 | 0.678 ± 0.246 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 8 | `2048x2048` | 1.201 ± 0.536 | - | 2.057 ± 0.644 | 0.543 ± 0.016 | 0.883 ± 0.367 | 0.875 ± 0.212 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 5.935 ± 0.029 | - | 6.018 ± 0.074 | 6.338 ± 0.015 | 18.811 ± 0.102 | 5.756 ± 0.044 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.298 ± 0.037 | - | 5.666 ± 0.146 | 5.296 ± 0.068 | 9.597 ± 0.576 | 6.241 ± 0.314 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 8 | `8192x4096` | 5.402 ± 0.011 | - | 5.998 ± 0.224 | 5.451 ± 0.019 | 10.067 ± 0.098 | 6.394 ± 0.020 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.627 ± 0.110 | - | 0.676 ± 0.135 | 2.407 ± 0.013 | 1.305 ± 0.037 | 0.731 ± 0.179 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.393 ± 0.023 | - | 1.227 ± 0.106 | 1.513 ± 0.043 | 0.855 ± 0.087 | 0.716 ± 0.122 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 8 | `2048x2048` | 0.916 ± 0.016 | - | 2.412 ± 0.080 | 1.730 ± 0.041 | 0.912 ± 0.253 | 0.815 ± 0.188 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.698 ± 0.070 | - | 0.692 ± 0.095 | 0.610 ± 0.015 | 0.890 ± 0.293 | 0.707 ± 0.258 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.472 ± 0.019 | - | 1.095 ± 0.820 | 0.401 ± 0.012 | 0.592 ± 0.203 | 0.689 ± 0.177 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 8 | `2048x2048` | 1.538 ± 0.014 | - | 2.035 ± 0.494 | 0.520 ± 0.067 | 0.927 ± 0.393 | 0.859 ± 0.206 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.617 ± 0.006 | 0.617 ± 0.008 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.387 ± 0.006 | 0.926 ± 0.009 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 8 | `2048x2048` | 0.543 ± 0.011 | 0.888 ± 0.345 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 21.463 ± 0.077 | - | 20.877 ± 0.097 | 31.516 ± 0.296 | 36.871 ± 0.489 | 37.371 ± 3.051 | 43.939 ± 3.717 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 5.508 ± 0.307 | - | 10.805 ± 0.210 | 9.234 ± 0.168 | 17.880 ± 0.582 | 34.823 ± 2.379 | 10.992 ± 4.738 |
| cpu/elementwise_reduction | `rem` | f64 | 8 | `8388608` | 5.400 ± 0.037 | - | 8.580 ± 0.155 | 6.793 ± 0.216 | 17.517 ± 0.222 | 45.249 ± 3.698 | 7.761 ± 6.841 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 56.731 ± 1.124 | - | 56.029 ± 1.062 | 58.964 ± 0.496 | 94.856 ± 0.137 | 116.851 ± 2.269 | 116.569 ± 2.118 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 18.476 ± 0.108 | - | 17.978 ± 0.112 | 19.441 ± 0.329 | 49.565 ± 1.677 | 116.882 ± 1.730 | 40.762 ± 0.152 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 8 | `33554432` | 14.792 ± 0.125 | - | 14.543 ± 0.141 | 16.354 ± 0.236 | 45.549 ± 0.353 | 176.818 ± 2.164 | 38.835 ± 2.372 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 56.993 ± 0.436 | - | 55.924 ± 0.824 | 73.811 ± 0.281 | 75.431 ± 0.988 | 57.244 ± 0.682 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 22.672 ± 0.070 | - | 22.026 ± 0.062 | 25.502 ± 0.191 | 38.436 ± 0.966 | 64.819 ± 0.535 | - |
| cpu/elementwise_reduction | `select` | f64 | 8 | `33554432` | 20.744 ± 0.187 | - | 20.141 ± 0.062 | 21.679 ± 0.155 | 36.462 ± 0.144 | 74.272 ± 0.448 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 51.505 ± 2.388 | - | 52.460 ± 2.959 | 51.662 ± 0.178 | 60.087 ± 2.605 | 50.209 ± 1.896 | 89.138 ± 0.754 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 18.098 ± 0.127 | - | 18.149 ± 0.104 | 18.844 ± 0.424 | 29.157 ± 0.400 | 50.033 ± 0.264 | 34.002 ± 2.502 |
| cpu/elementwise_reduction | `sign` | f64 | 8 | `33554432` | 15.102 ± 0.126 | - | 14.726 ± 0.280 | 15.874 ± 0.209 | 29.143 ± 0.183 | 74.535 ± 0.451 | 35.262 ± 4.226 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 7.365 ± 0.026 | - | 7.403 ± 0.039 | 32.342 ± 0.512 | 59.382 ± 0.167 | 32.286 ± 2.168 | 32.429 ± 3.737 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 2.921 ± 0.007 | - | 2.930 ± 0.035 | 8.916 ± 0.079 | 31.171 ± 0.316 | 31.247 ± 4.320 | 8.577 ± 5.698 |
| cpu/elementwise_reduction | `sin` | f64 | 8 | `8388608` | 3.346 ± 0.055 | - | 4.698 ± 0.067 | 6.802 ± 0.401 | 31.912 ± 0.279 | 45.487 ± 7.418 | 10.304 ± 7.718 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 51.015 ± 2.709 | - | 49.028 ± 1.000 | 87.306 ± 0.925 | 57.533 ± 0.090 | 92.422 ± 2.648 | 92.866 ± 3.105 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 17.366 ± 0.132 | - | 16.888 ± 0.053 | 24.722 ± 0.367 | 33.144 ± 1.743 | 104.720 ± 0.725 | 28.626 ± 1.495 |
| cpu/elementwise_reduction | `sqrt` | f64 | 8 | `33554432` | 14.521 ± 0.051 | - | 14.457 ± 0.165 | 19.610 ± 0.239 | 27.071 ± 0.119 | 142.688 ± 0.637 | 31.177 ± 2.040 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 55.707 ± 0.822 | - | 56.047 ± 0.817 | 57.739 ± 0.768 | 70.093 ± 0.543 | 54.096 ± 0.607 | 102.455 ± 0.633 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 22.126 ± 0.133 | - | 21.010 ± 0.094 | 22.912 ± 0.236 | 36.185 ± 0.486 | 58.989 ± 0.457 | 40.762 ± 2.817 |
| cpu/elementwise_reduction | `sub` | f64 | 8 | `33554432` | 19.499 ± 0.136 | - | 18.772 ± 0.141 | 21.036 ± 0.198 | 31.741 ± 0.291 | 81.180 ± 0.110 | 42.046 ± 3.406 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 18.470 ± 0.039 | - | 18.540 ± 0.041 | 43.978 ± 0.666 | 15.389 ± 0.249 | 34.259 ± 3.428 | 38.355 ± 3.653 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 4.766 ± 0.011 | - | 4.762 ± 0.010 | 11.556 ± 0.099 | 9.701 ± 0.407 | 34.848 ± 2.294 | 10.319 ± 5.315 |
| cpu/elementwise_reduction | `tanh` | f64 | 8 | `8388608` | 3.688 ± 0.010 | - | 3.728 ± 0.029 | 8.645 ± 0.373 | 8.396 ± 0.393 | 52.927 ± 5.899 | 10.546 ± 7.970 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 56.250 ± 0.818 | 54.446 ± 1.188 | unsupported | 73.091 ± 0.401 | 89.822 ± 0.398 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 22.147 ± 0.105 | 21.368 ± 0.093 | unsupported | 25.238 ± 0.152 | 39.216 ± 0.463 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 8 | `33554432` | 20.194 ± 0.120 | 19.625 ± 0.062 | unsupported | 21.421 ± 0.170 | 36.268 ± 0.135 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.708 ± 0.024 | - | 0.696 ± 0.027 | 0.676 ± 0.020 | 0.704 ± 0.151 | 2.996 ± 1.312 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.467 ± 0.021 | - | 0.745 ± 0.016 | 0.491 ± 0.012 | 0.675 ± 0.174 | 1.108 ± 2.331 | - |
| cpu/indexing_layout | `concatenate` | f64 | 8 | `1048576+1048576` | 1.008 ± 0.056 | - | 1.378 ± 0.334 | 0.618 ± 0.011 | 0.747 ± 0.028 | 4.387 ± 1.852 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.592 ± 0.019 | - | 0.607 ± 0.015 | 0.659 ± 0.027 | 0.743 ± 0.105 | 0.768 ± 0.066 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.528 ± 0.017 | - | 1.022 ± 0.040 | 0.468 ± 0.014 | 0.591 ± 0.099 | 0.806 ± 0.046 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 8 | `4194304 -> 2097152` | 0.797 ± 0.028 | - | 1.200 ± 1.384 | 0.662 ± 0.007 | 0.874 ± 0.232 | 4.447 ± 1.782 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.980 ± 0.080 | - | unsupported | 0.955 ± 0.087 | 0.976 ± 0.044 | 2.547 ± 2.700 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.659 ± 0.036 | - | unsupported | 0.691 ± 0.053 | 0.951 ± 0.125 | 3.656 ± 1.282 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 8 | `2097152` | 0.859 ± 0.049 | - | unsupported | 0.903 ± 0.033 | 1.499 ± 0.030 | 1.525 ± 0.055 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.155 ± 0.009 | - | 0.186 ± 0.005 | 0.297 ± 0.003 | 0.152 ± 0.002 | 0.470 ± 0.280 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.071 ± 0.004 | - | 1.317 ± 0.559 | 0.078 ± 0.003 | 0.091 ± 0.046 | 0.458 ± 0.012 | - |
| cpu/indexing_layout | `gather` | f64 | 8 | `262144` | 0.061 ± 0.003 | - | 3.068 ± 2.925 | 0.069 ± 0.004 | 0.165 ± 0.021 | 0.674 ± 0.290 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.607 ± 0.014 | - | 0.612 ± 0.014 | 0.970 ± 0.040 | 0.782 ± 0.106 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.522 ± 0.017 | - | 1.006 ± 0.021 | 0.594 ± 0.016 | 0.578 ± 0.060 | - | - |
| cpu/indexing_layout | `pad` | f64 | 8 | `2097152` | 0.712 ± 0.035 | - | 1.400 ± 0.191 | 0.868 ± 0.077 | 0.919 ± 0.248 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.690 ± 0.132 | - | 0.710 ± 0.031 | 0.640 ± 0.014 | 0.696 ± 0.090 | 0.977 ± 2.960 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.473 ± 0.029 | - | 1.074 ± 0.317 | 0.476 ± 0.015 | 0.607 ± 0.084 | 2.146 ± 3.205 | - |
| cpu/indexing_layout | `reverse` | f64 | 8 | `2097152` | 1.035 ± 0.340 | - | 1.369 ± 0.092 | 0.608 ± 0.013 | 0.912 ± 0.347 | 4.735 ± 2.153 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.387 ± 0.006 | - | 0.429 ± 0.016 | 0.176 ± 0.001 | 0.153 ± 0.005 | 0.552 ± 0.395 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.393 ± 0.006 | - | 1.804 ± 0.482 | 0.156 ± 0.006 | 0.148 ± 0.160 | 0.249 ± 0.327 | - |
| cpu/indexing_layout | `scatter` | f64 | 8 | `262144` | 0.616 ± 0.004 | - | 2.760 ± 2.086 | 0.247 ± 0.016 | 0.362 ± 0.183 | 0.387 ± 0.023 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.142 ± 0.015 | - | 0.001 ± 0.000 | 1.223 ± 0.020 | 1.072 ± 0.039 | 4.464 ± 1.152 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.825 ± 0.103 | - | 0.001 ± 0.000 | 0.825 ± 0.005 | 0.892 ± 0.099 | 4.094 ± 1.623 | - |
| cpu/indexing_layout | `slice` | f64 | 8 | `4194304 -> 2096128` | 1.427 ± 0.337 | - | 0.002 ± 0.000 | 0.981 ± 0.016 | 1.186 ± 0.047 | 5.459 ± 1.920 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.024 ± 0.000 | - | 0.026 ± 0.000 | 0.036 ± 0.000 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.048 ± 0.000 | - | 0.051 ± 0.000 | 0.051 ± 0.000 | 0.070 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.120 ± 0.001 | - | 0.130 ± 0.001 | 0.133 ± 0.000 | 0.149 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.036 ± 0.000 | - | 0.039 ± 0.001 | 0.086 ± 0.002 | 0.058 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.000 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.000 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.294 ± 0.003 | - | 0.304 ± 0.002 | 0.681 ± 0.003 | 0.392 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.024 ± 0.000 | - | 0.026 ± 0.000 | 0.021 ± 0.000 | 0.049 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.049 ± 0.000 | - | 0.510 ± 0.003 | 0.035 ± 0.000 | 0.067 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.127 ± 0.000 | - | 0.132 ± 0.001 | 0.106 ± 0.000 | 0.169 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.038 ± 0.000 | - | 0.042 ± 0.000 | 0.064 ± 0.000 | 0.074 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.297 ± 0.003 | - | 0.307 ± 0.002 | 0.469 ± 0.004 | 0.999 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1024x2x2` | 0.037 ± 0.000 | - | 0.040 ± 0.000 | 0.029 ± 0.000 | 0.074 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1024x4x4` | 0.072 ± 0.000 | - | 0.079 ± 0.001 | 0.045 ± 0.001 | 0.116 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1024x8x8` | 0.191 ± 0.004 | - | 0.210 ± 0.010 | 0.153 ± 0.000 | 0.280 ± 0.129 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1x128x128` | 0.058 ± 0.000 | - | 0.066 ± 0.001 | 0.089 ± 0.002 | 0.101 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1x32x32` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `1x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `3x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `4x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `8x128x128` | 0.477 ± 0.005 | - | 0.466 ± 0.002 | 0.665 ± 0.003 | 0.521 ± 0.144 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `8x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `8x32x32` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.025 ± 0.000 | 0.043 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `8x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 8 | `8x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.280 ± 0.002 | - | 0.283 ± 0.002 | 0.344 ± 0.002 | 0.405 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.092 ± 0.006 | - | 1.092 ± 0.002 | 1.155 ± 0.003 | 1.417 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 3.390 ± 0.006 | - | 3.403 ± 0.008 | 3.496 ± 0.003 | 4.816 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 1.894 ± 0.007 | - | 1.900 ± 0.006 | 1.700 ± 0.009 | 1.317 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.056 ± 0.000 | - | 0.058 ± 0.000 | 0.057 ± 0.000 | 0.069 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.014 ± 0.000 | - | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.028 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 15.102 ± 0.045 | - | 15.146 ± 0.065 | 13.404 ± 0.039 | 10.503 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.442 ± 0.003 | - | 0.443 ± 0.004 | 0.439 ± 0.002 | 0.488 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.027 ± 0.000 | - | 0.029 ± 0.000 | 0.030 ± 0.000 | 0.048 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.282 ± 0.002 | - | 0.286 ± 0.002 | 0.335 ± 0.004 | 0.445 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.116 ± 0.008 | - | 1.556 ± 0.004 | 1.177 ± 0.006 | 0.928 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 3.414 ± 0.009 | - | 3.542 ± 0.326 | 3.572 ± 0.005 | 2.936 ± 0.112 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 1.196 ± 0.012 | - | 1.416 ± 0.221 | 0.742 ± 0.006 | 1.263 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.059 ± 0.000 | - | 0.061 ± 0.000 | 0.060 ± 0.000 | 0.067 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.060 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.065 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 9.446 ± 0.011 | - | 9.721 ± 0.009 | 5.811 ± 0.011 | 10.546 ± 1.962 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.461 ± 0.005 | - | 0.461 ± 0.004 | 0.459 ± 0.003 | 0.544 ± 0.141 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.028 ± 0.000 | - | 0.029 ± 0.000 | 0.031 ± 0.000 | 0.043 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1024x2x2` | 0.437 ± 0.003 | - | 0.440 ± 0.012 | 0.514 ± 0.005 | 0.752 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1024x4x4` | 1.720 ± 0.009 | - | 1.733 ± 0.008 | 1.816 ± 0.007 | 1.514 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1024x8x8` | 5.342 ± 0.016 | - | 5.347 ± 0.050 | 5.501 ± 0.045 | 4.989 ± 0.099 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1x128x128` | 1.647 ± 0.015 | - | 1.997 ± 0.015 | 0.966 ± 0.010 | 1.962 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1x32x32` | 0.089 ± 0.000 | - | 0.092 ± 0.000 | 0.094 ± 0.001 | 0.115 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `1x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.025 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `3x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `3x8x8` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.021 ± 0.000 | 0.048 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `4x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `4x8x8` | 0.022 ± 0.000 | - | 0.025 ± 0.000 | 0.026 ± 0.000 | 0.043 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `8x128x128` | 12.987 ± 0.019 | - | 70.575 ± 0.389 | 7.453 ± 0.030 | 15.774 ± 4.442 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `8x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `8x32x32` | 0.698 ± 0.004 | - | 0.709 ± 0.006 | 0.718 ± 0.004 | 0.507 ± 0.101 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `8x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.019 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 8 | `8x8x8` | 0.043 ± 0.000 | - | 0.045 ± 0.000 | 0.048 ± 0.000 | 0.075 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.308 ± 0.002 | - | 0.310 ± 0.003 | 0.192 ± 0.002 | 0.171 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.135 ± 0.007 | - | 1.136 ± 0.003 | 0.993 ± 0.005 | 0.977 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 3.065 ± 0.002 | - | 3.081 ± 0.006 | 3.005 ± 0.005 | 3.363 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.687 ± 0.010 | - | 0.694 ± 0.006 | 0.711 ± 0.006 | 0.734 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.035 ± 0.000 | - | 0.037 ± 0.000 | 0.040 ± 0.000 | 0.055 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 5.427 ± 0.020 | - | 5.450 ± 0.010 | 5.612 ± 0.013 | 5.736 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.279 ± 0.002 | - | 0.281 ± 0.002 | 0.303 ± 0.002 | 0.372 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.024 ± 0.000 | - | 0.026 ± 0.000 | 0.026 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.308 ± 0.017 | - | 0.301 ± 0.022 | 0.198 ± 0.002 | 0.168 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.138 ± 0.013 | - | 1.561 ± 0.005 | 1.012 ± 0.005 | 0.643 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 3.104 ± 0.004 | - | 3.524 ± 0.273 | 3.064 ± 0.006 | 1.761 ± 0.781 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 0.422 ± 0.005 | - | 0.862 ± 0.005 | 0.424 ± 0.005 | 0.737 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.040 ± 0.000 | - | 0.041 ± 0.000 | 0.043 ± 0.000 | 0.050 ± 0.092 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.013 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 3.340 ± 0.013 | - | 16.832 ± 11.536 | 3.240 ± 0.008 | 5.956 ± 0.832 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.309 ± 0.003 | - | 0.311 ± 0.002 | 0.322 ± 0.002 | 0.203 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.024 ± 0.000 | - | 0.026 ± 0.000 | 0.027 ± 0.000 | 0.087 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1024x2x2` | 0.472 ± 0.006 | - | 0.478 ± 0.009 | 0.304 ± 0.003 | 0.303 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1024x4x4` | 1.771 ± 0.006 | - | 1.779 ± 0.166 | 1.555 ± 0.007 | 1.085 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1024x8x8` | 4.807 ± 0.006 | - | 4.819 ± 0.009 | 4.732 ± 0.008 | 2.202 ± 0.165 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1x128x128` | 0.621 ± 0.017 | - | 1.324 ± 0.013 | 0.639 ± 0.012 | 1.168 ± 0.845 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1x32x32` | 0.062 ± 0.001 | - | 0.065 ± 0.002 | 0.068 ± 0.001 | 0.112 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.021 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `3x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `3x8x8` | 0.015 ± 0.000 | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.024 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `4x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.022 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `4x8x8` | 0.020 ± 0.000 | - | 0.023 ± 0.000 | 0.023 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `8x128x128` | 4.920 ± 0.027 | - | 5.638 ± 0.709 | 4.856 ± 0.017 | 23.413 ± 29.703 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `8x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `8x32x32` | 0.484 ± 0.005 | - | 0.490 ± 0.005 | 0.509 ± 0.004 | 0.242 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `8x4x4` | 0.015 ± 0.000 | - | 0.018 ± 0.000 | 0.016 ± 0.000 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 8 | `8x8x8` | 0.038 ± 0.000 | - | 0.041 ± 0.000 | 0.041 ± 0.000 | 0.060 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.108 ± 0.003 | - | 0.112 ± 0.001 | 0.100 ± 0.000 | 0.171 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.582 ± 0.004 | - | 0.584 ± 0.004 | 0.506 ± 0.002 | 0.973 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 1.675 ± 0.003 | - | 1.690 ± 0.003 | 1.575 ± 0.004 | 3.376 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.275 ± 0.007 | - | 0.279 ± 0.002 | 0.305 ± 0.001 | 0.725 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.020 ± 0.000 | 0.054 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.001 | 0.004 ± 0.000 | 0.004 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 2.199 ± 0.005 | - | 2.219 ± 0.004 | 2.424 ± 0.002 | 5.640 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.123 ± 0.000 | - | 0.126 ± 0.001 | 0.128 ± 0.000 | 0.369 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.014 ± 0.000 | - | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.109 ± 0.001 | - | 0.110 ± 0.002 | 0.103 ± 0.000 | 0.168 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.579 ± 0.004 | - | 1.021 ± 0.004 | 0.524 ± 0.003 | 0.633 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 1.683 ± 0.005 | - | 2.129 ± 0.007 | 1.603 ± 0.014 | 1.743 ± 0.404 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.275 ± 0.003 | - | 0.717 ± 0.003 | 0.282 ± 0.004 | 0.999 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.020 ± 0.000 | 0.050 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.050 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 2.209 ± 0.004 | - | 2.677 ± 0.232 | 2.150 ± 0.004 | 5.717 ± 1.967 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.125 ± 0.000 | - | 0.127 ± 0.000 | 0.130 ± 0.000 | 0.146 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.014 ± 0.000 | - | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.086 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1024x2x2` | 0.170 ± 0.004 | - | 0.173 ± 0.002 | 0.158 ± 0.001 | 0.276 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1024x4x4` | 0.922 ± 0.008 | - | 1.618 ± 0.005 | 0.793 ± 0.001 | 1.081 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1024x8x8` | 2.663 ± 0.009 | - | 2.677 ± 0.008 | 2.484 ± 0.016 | 2.194 ± 0.268 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1x128x128` | 0.450 ± 0.004 | - | 1.148 ± 0.003 | 0.466 ± 0.004 | 1.134 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1x32x32` | 0.026 ± 0.000 | - | 0.029 ± 0.000 | 0.031 ± 0.000 | 0.080 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `3x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `3x8x8` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.021 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `4x8x8` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.027 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `8x128x128` | 3.556 ± 0.009 | - | 37.996 ± 1.569 | 3.592 ± 0.011 | 23.987 ± 30.644 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `8x32x32` | 0.197 ± 0.001 | - | 0.202 ± 0.001 | 0.204 ± 0.000 | 0.247 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `8x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 8 | `8x8x8` | 0.022 ± 0.000 | - | 0.025 ± 0.000 | 0.025 ± 0.000 | 0.065 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.060 ± 0.001 | - | 0.071 ± 0.001 | 0.080 ± 0.000 | 0.056 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.084 ± 0.001 | - | 0.092 ± 0.001 | 0.111 ± 0.001 | 0.100 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.236 ± 0.002 | - | 0.237 ± 0.003 | 0.251 ± 0.003 | 0.287 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.067 ± 0.003 | - | 0.072 ± 0.001 | 0.084 ± 0.000 | 0.098 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.534 ± 0.013 | - | 0.544 ± 0.011 | 0.636 ± 0.003 | 0.682 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.037 ± 0.000 | - | 0.039 ± 0.000 | 0.038 ± 0.000 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.068 ± 0.001 | - | 0.065 ± 0.001 | 0.043 ± 0.000 | 0.057 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.086 ± 0.001 | - | 0.530 ± 0.002 | 0.064 ± 0.000 | 0.107 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.223 ± 0.003 | - | 0.917 ± 0.008 | 0.104 ± 0.001 | 0.224 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.066 ± 0.003 | - | 0.510 ± 0.003 | 0.081 ± 0.001 | 0.096 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.020 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.548 ± 0.014 | - | 0.978 ± 0.905 | 0.174 ± 0.001 | 0.386 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.037 ± 0.000 | - | 0.039 ± 0.000 | 0.020 ± 0.000 | 0.073 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1024x2x2` | 0.110 ± 0.002 | - | 0.126 ± 0.001 | 0.058 ± 0.001 | 0.089 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1024x4x4` | 0.145 ± 0.004 | - | 0.859 ± 0.002 | 0.088 ± 0.001 | 0.191 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1024x8x8` | 0.365 ± 0.003 | - | 1.424 ± 0.113 | 0.130 ± 0.001 | 0.362 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x128x128` | 0.105 ± 0.003 | - | 0.806 ± 0.004 | 0.122 ± 0.004 | 0.178 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x32x32` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.000 | 0.021 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x64x24` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `3x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.020 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `8x128x128` | 0.844 ± 0.006 | - | 0.845 ± 0.020 | 0.189 ± 0.001 | 0.416 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `8x32x32` | 0.058 ± 0.000 | - | 0.062 ± 0.001 | 0.025 ± 0.000 | 0.107 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `8x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 8 | `8x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.018 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.107 ± 0.002 | - | 0.111 ± 0.001 | 0.093 ± 0.000 | 0.127 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.265 ± 0.002 | - | 0.265 ± 0.002 | 0.255 ± 0.002 | 0.364 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 0.647 ± 0.006 | - | 0.658 ± 0.005 | 0.662 ± 0.003 | 1.035 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.252 ± 0.003 | - | 0.255 ± 0.002 | 0.313 ± 0.003 | 0.289 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.016 ± 0.009 | - | 2.196 ± 0.009 | 2.476 ± 0.016 | 2.280 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.059 ± 0.000 | - | 0.062 ± 0.000 | 0.079 ± 0.000 | 0.091 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.108 ± 0.001 | - | 0.111 ± 0.001 | 0.081 ± 0.000 | 0.124 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.271 ± 0.004 | - | 0.275 ± 0.003 | 0.246 ± 0.002 | 0.357 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.659 ± 0.008 | - | 0.669 ± 0.005 | 0.648 ± 0.004 | 0.487 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.177 ± 0.004 | - | 0.651 ± 0.004 | 0.192 ± 0.003 | 0.999 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.038 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.000 | 0.015 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.028 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 1.381 ± 0.016 | - | 2.401 ± 0.088 | 1.400 ± 0.008 | 2.668 ± 4.243 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.061 ± 0.000 | - | 0.063 ± 0.000 | 0.084 ± 0.000 | 0.066 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1024x2x2` | 0.174 ± 0.003 | - | 0.179 ± 0.002 | 0.122 ± 0.001 | 0.224 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1024x4x4` | 0.415 ± 0.004 | - | 0.424 ± 0.003 | 0.369 ± 0.003 | 0.601 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1024x8x8` | 1.006 ± 0.004 | - | 1.015 ± 0.009 | 0.982 ± 0.009 | 0.583 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x128x128` | 0.301 ± 0.008 | - | 1.013 ± 0.006 | 0.259 ± 0.005 | 0.531 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x32x32` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.022 ± 0.000 | 0.036 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x64x24` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.024 ± 0.000 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `3x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `4x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `4x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.025 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `8x128x128` | 2.336 ± 0.017 | - | 4.790 ± 2.042 | 1.907 ± 0.009 | 4.805 ± 1.185 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `8x32x32` | 0.093 ± 0.000 | - | 0.099 ± 0.001 | 0.123 ± 0.000 | 0.107 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `8x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.021 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 8 | `8x8x8` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.087 ± 0.001 | - | 0.077 ± 0.001 | 0.040 ± 0.000 | 0.129 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.107 ± 0.001 | - | 0.091 ± 0.001 | 0.057 ± 0.000 | 0.185 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.247 ± 0.002 | - | 0.239 ± 0.014 | 0.205 ± 0.001 | 0.386 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.057 ± 0.000 | - | 0.060 ± 0.000 | 0.068 ± 0.000 | 0.080 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.004 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.001 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.450 ± 0.003 | - | 0.470 ± 0.005 | 0.507 ± 0.003 | 0.525 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.035 ± 0.000 | - | 0.038 ± 0.000 | 0.036 ± 0.000 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.091 ± 0.001 | - | 0.081 ± 0.000 | 0.033 ± 0.000 | 0.130 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.113 ± 0.001 | - | 0.553 ± 0.003 | 0.045 ± 0.000 | 0.186 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.264 ± 0.002 | - | 0.699 ± 0.151 | 0.129 ± 0.001 | 0.346 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.059 ± 0.000 | - | 0.525 ± 0.004 | 0.070 ± 0.000 | 0.074 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.471 ± 0.008 | - | 0.945 ± 0.015 | 0.156 ± 0.001 | 0.317 ± 0.085 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.037 ± 0.000 | - | 0.040 ± 0.000 | 0.019 ± 0.000 | 0.052 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1024x2x2,rhs=1` | 0.140 ± 0.002 | - | 0.124 ± 0.001 | 0.048 ± 0.001 | 0.246 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1024x4x4,rhs=1` | 0.172 ± 0.003 | - | 0.145 ± 0.001 | 0.066 ± 0.001 | 0.330 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1024x8x8,rhs=1` | 0.394 ± 0.002 | - | 2.761 ± 0.327 | 0.178 ± 0.001 | 0.591 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1x128x128,rhs=1` | 0.090 ± 0.001 | - | 0.788 ± 0.004 | 0.107 ± 0.000 | 0.119 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1x32x32,rhs=1` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `8x128x128,rhs=1` | 0.713 ± 0.005 | - | 0.732 ± 0.003 | 0.169 ± 0.001 | 0.364 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `8x32x32,rhs=1` | 0.056 ± 0.000 | - | 0.060 ± 0.000 | 0.025 ± 0.000 | 0.112 ± 0.071 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 8 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 0.904 ± 0.008 | - | 0.908 ± 0.005 | 0.756 ± 0.002 | 0.974 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 3.071 ± 0.009 | - | 3.076 ± 0.009 | 2.673 ± 0.003 | 2.608 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 9.130 ± 0.013 | - | 9.161 ± 0.035 | 8.388 ± 0.008 | 7.641 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 2.564 ± 0.022 | - | 2.556 ± 0.011 | 2.964 ± 0.008 | 2.245 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.097 ± 0.001 | - | 0.100 ± 0.001 | 0.111 ± 0.001 | 0.104 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.089 ± 0.001 | - | 0.091 ± 0.002 | 0.100 ± 0.001 | 0.102 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.010 ± 0.001 | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.028 ± 0.000 | - | 0.030 ± 0.000 | 0.028 ± 0.000 | 0.029 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.037 ± 0.000 | - | 0.039 ± 0.000 | 0.037 ± 0.000 | 0.037 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 20.268 ± 0.035 | - | 20.802 ± 0.022 | 23.697 ± 0.030 | 17.749 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 0.766 ± 0.006 | - | 0.769 ± 0.006 | 0.852 ± 0.001 | 0.762 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.025 ± 0.000 | - | 0.027 ± 0.000 | 0.024 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.072 ± 0.000 | - | 0.074 ± 0.000 | 0.069 ± 0.000 | 0.072 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 0.909 ± 0.010 | - | 0.917 ± 0.004 | 0.749 ± 0.003 | 0.997 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 3.137 ± 0.008 | - | 3.151 ± 0.007 | 2.710 ± 0.017 | 1.035 ± 0.210 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 9.160 ± 0.028 | - | 9.634 ± 0.058 | 8.602 ± 0.015 | 3.992 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 1.625 ± 0.011 | - | 1.850 ± 0.006 | 1.762 ± 0.006 | 1.999 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.107 ± 0.002 | - | 0.108 ± 0.001 | 0.116 ± 0.001 | 0.100 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.091 ± 0.000 | - | 0.093 ± 0.000 | 0.101 ± 0.001 | 0.099 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.029 ± 0.000 | - | 0.031 ± 0.000 | 0.029 ± 0.000 | 0.028 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.038 ± 0.000 | - | 0.040 ± 0.000 | 0.037 ± 0.000 | 0.037 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 12.961 ± 0.017 | - | 13.496 ± 0.445 | 13.963 ± 0.013 | 36.143 ± 15.196 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.041 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 0.823 ± 0.004 | - | 0.821 ± 0.006 | 0.890 ± 0.002 | 0.411 ± 0.176 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.025 ± 0.000 | - | 0.027 ± 0.000 | 0.025 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.074 ± 0.000 | - | 0.076 ± 0.000 | 0.070 ± 0.000 | 0.068 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1024x2x2` | 1.412 ± 0.006 | - | 1.417 ± 0.006 | 1.150 ± 0.005 | 1.571 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1024x4x4` | 4.787 ± 0.021 | - | 4.772 ± 0.034 | 4.168 ± 0.013 | 1.355 ± 0.300 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1024x8x8` | 14.446 ± 0.027 | - | 14.469 ± 0.102 | 13.291 ± 0.012 | 3.231 ± 0.092 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x128x128` | 2.240 ± 0.029 | - | 2.584 ± 0.028 | 2.482 ± 0.009 | 3.085 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x32x32` | 0.160 ± 0.002 | - | 0.164 ± 0.001 | 0.182 ± 0.002 | 0.166 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x64x24` | 0.195 ± 0.001 | - | 0.143 ± 0.002 | 0.155 ± 0.002 | 0.178 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `1x8x8` | 0.016 ± 0.000 | - | 0.019 ± 0.000 | 0.019 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `3x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `3x4x4` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.017 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `3x8x8` | 0.044 ± 0.000 | - | 0.047 ± 0.000 | 0.044 ± 0.000 | 0.046 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `4x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.026 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `4x4x4` | 0.021 ± 0.000 | - | 0.023 ± 0.000 | 0.021 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `4x8x8` | 0.059 ± 0.000 | - | 0.062 ± 0.000 | 0.058 ± 0.000 | 0.059 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `8x128x128` | 17.893 ± 0.034 | - | 84.309 ± 1.850 | 19.712 ± 0.028 | 113.990 ± 31.759 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `8x2x2` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.014 ± 0.000 | 0.021 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `8x32x32` | 3.018 ± 1.749 | - | 1.280 ± 0.008 | 1.403 ± 0.007 | 0.338 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `8x4x4` | 0.039 ± 0.000 | - | 0.042 ± 0.000 | 0.038 ± 0.000 | 0.042 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 8 | `8x8x8` | 0.114 ± 0.000 | - | 0.118 ± 0.000 | 0.109 ± 0.000 | 0.114 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.679 ± 0.005 | - | 0.682 ± 0.002 | 0.537 ± 0.002 | 0.611 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.585 ± 0.006 | - | 2.600 ± 0.004 | 2.241 ± 0.004 | 2.004 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 7.903 ± 0.006 | - | 7.917 ± 0.014 | 7.280 ± 0.006 | 6.302 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.196 ± 0.011 | - | 1.192 ± 0.005 | 1.299 ± 0.023 | 1.164 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.066 ± 0.001 | - | 0.068 ± 0.001 | 0.076 ± 0.001 | 0.081 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.054 ± 0.001 | - | 0.056 ± 0.000 | 0.060 ± 0.000 | 0.072 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.022 ± 0.000 | - | 0.024 ± 0.000 | 0.023 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.002 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.011 ± 0.001 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.030 ± 0.000 | - | 0.031 ± 0.000 | 0.030 ± 0.000 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 9.445 ± 0.041 | - | 9.461 ± 0.022 | 10.177 ± 0.008 | 9.267 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.521 ± 0.004 | - | 0.522 ± 0.003 | 0.587 ± 0.003 | 0.573 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.021 ± 0.000 | - | 0.023 ± 0.000 | 0.021 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.060 ± 0.000 | - | 0.061 ± 0.000 | 0.057 ± 0.000 | 0.058 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.696 ± 0.004 | - | 0.688 ± 0.007 | 0.520 ± 0.004 | 0.646 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.652 ± 0.008 | - | 2.655 ± 0.005 | 2.278 ± 0.005 | 0.805 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 7.974 ± 0.141 | - | 8.362 ± 0.029 | 7.481 ± 0.009 | 3.194 ± 0.778 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 0.979 ± 0.011 | - | 1.185 ± 0.012 | 1.069 ± 0.005 | 1.198 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.075 ± 0.001 | - | 0.076 ± 0.004 | 0.083 ± 0.006 | 0.078 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.057 ± 0.001 | - | 0.058 ± 0.000 | 0.062 ± 0.000 | 0.083 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.036 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.023 ± 0.000 | - | 0.024 ± 0.000 | 0.023 ± 0.000 | 0.024 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.041 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.030 ± 0.000 | - | 0.032 ± 0.000 | 0.030 ± 0.000 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 7.632 ± 0.027 | - | 8.058 ± 0.009 | 8.361 ± 0.015 | 9.594 ± 1.562 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.581 ± 0.003 | - | 0.580 ± 0.006 | 0.621 ± 0.002 | 0.184 ± 0.139 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.022 ± 0.000 | - | 0.023 ± 0.000 | 0.021 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.061 ± 0.000 | - | 0.062 ± 0.000 | 0.059 ± 0.000 | 0.054 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1024x2x2` | 1.037 ± 0.005 | - | 1.043 ± 0.005 | 0.808 ± 0.005 | 0.995 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1024x4x4` | 4.054 ± 0.008 | - | 4.064 ± 0.010 | 3.505 ± 0.005 | 1.432 ± 0.198 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1024x8x8` | 12.454 ± 0.028 | - | 12.486 ± 0.011 | 11.518 ± 0.011 | 2.606 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x128x128` | 1.459 ± 0.023 | - | 1.807 ± 0.012 | 1.598 ± 0.007 | 2.998 ± 1.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x32x32` | 0.113 ± 0.002 | - | 0.115 ± 0.001 | 0.130 ± 0.004 | 0.130 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x64x24` | 0.114 ± 0.001 | - | 0.123 ± 0.001 | 0.097 ± 0.001 | 0.127 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `1x8x8` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.019 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `3x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `3x4x4` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.018 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `3x8x8` | 0.035 ± 0.000 | - | 0.038 ± 0.000 | 0.036 ± 0.000 | 0.037 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `4x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `4x4x4` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.019 ± 0.000 | 0.037 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `4x8x8` | 0.047 ± 0.000 | - | 0.050 ± 0.000 | 0.047 ± 0.000 | 0.048 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `8x128x128` | 11.292 ± 0.080 | - | 21.235 ± 0.042 | 12.425 ± 0.035 | 38.872 ± 19.223 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `8x2x2` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `8x32x32` | 0.888 ± 0.003 | - | 0.897 ± 0.006 | 0.981 ± 0.002 | 0.276 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `8x4x4` | 0.034 ± 0.000 | - | 0.036 ± 0.000 | 0.032 ± 0.000 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 8 | `8x8x8` | 0.094 ± 0.000 | - | 0.097 ± 0.000 | 0.091 ± 0.000 | 0.142 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.210 ± 0.003 | - | unsupported | 0.146 ± 0.002 | 0.308 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.112 ± 0.003 | - | unsupported | 0.986 ± 0.002 | 1.265 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 2.711 ± 0.004 | - | unsupported | 2.651 ± 0.004 | 3.273 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.503 ± 0.004 | - | unsupported | 0.575 ± 0.010 | 0.576 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.003 ± 0.000 | 0.004 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.025 ± 0.000 | - | unsupported | 0.029 ± 0.000 | 0.033 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.004 ± 0.000 | 0.005 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.020 ± 0.000 | - | unsupported | 0.024 ± 0.000 | 0.028 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.004 ± 0.000 | 0.005 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.008 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.004 ± 0.000 | 0.005 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.011 ± 0.000 | - | unsupported | 0.013 ± 0.000 | 0.016 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 4.005 ± 0.022 | - | unsupported | 4.557 ± 0.010 | 4.478 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.196 ± 0.000 | - | unsupported | 0.214 ± 0.002 | 0.242 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.009 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.022 ± 0.000 | - | unsupported | 0.023 ± 0.000 | 0.029 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.239 ± 0.002 | - | unsupported | 0.152 ± 0.003 | 0.308 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.171 ± 0.002 | - | unsupported | 1.008 ± 0.004 | 0.508 ± 0.101 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 2.815 ± 0.005 | - | unsupported | 2.706 ± 0.003 | 1.677 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.510 ± 0.006 | - | unsupported | 0.581 ± 0.005 | 0.599 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.003 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.026 ± 0.000 | - | unsupported | 0.029 ± 0.000 | 0.033 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.004 ± 0.000 | 0.005 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.021 ± 0.000 | - | unsupported | 0.024 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.004 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.009 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.039 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.004 ± 0.000 | 0.005 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.009 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.012 ± 0.000 | - | unsupported | 0.014 ± 0.000 | 0.016 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 4.043 ± 0.007 | - | unsupported | 4.538 ± 0.009 | 4.998 ± 0.161 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.204 ± 0.003 | - | unsupported | 0.217 ± 0.001 | 0.099 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.010 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.022 ± 0.000 | - | unsupported | 0.024 ± 0.000 | 0.029 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1024x2x2` | 0.333 ± 0.006 | - | unsupported | 0.233 ± 0.002 | 0.519 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1024x4x4` | 1.757 ± 0.006 | - | unsupported | 1.552 ± 0.005 | 0.933 ± 0.375 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1024x8x8` | 4.292 ± 0.006 | - | unsupported | 4.186 ± 0.011 | 1.389 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x128x128` | 0.778 ± 0.009 | - | unsupported | 0.908 ± 0.015 | 0.978 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x32x32` | 0.040 ± 0.000 | - | unsupported | 0.046 ± 0.000 | 0.053 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x64x24` | 0.032 ± 0.000 | - | unsupported | 0.038 ± 0.000 | 0.045 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.011 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `3x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `3x8x8` | 0.013 ± 0.000 | - | unsupported | 0.017 ± 0.000 | 0.021 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.013 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `4x4x4` | 0.008 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.023 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `4x8x8` | 0.018 ± 0.000 | - | unsupported | 0.021 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `8x128x128` | 6.185 ± 0.014 | - | unsupported | 7.054 ± 0.019 | 23.455 ± 25.042 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `8x2x2` | 0.004 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `8x32x32` | 0.314 ± 0.002 | - | unsupported | 0.336 ± 0.005 | 0.124 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `8x4x4` | 0.015 ± 0.000 | - | unsupported | 0.017 ± 0.000 | 0.035 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 8 | `8x8x8` | 0.034 ± 0.000 | - | unsupported | 0.037 ± 0.000 | 0.047 ± 0.001 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.532 ± 0.004 | - | unsupported | 0.548 ± 0.003 | 1.232 ± 0.007 | 1.161 ± 0.125 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.103 ± 0.001 | - | unsupported | 0.025 ± 0.002 | 0.050 ± 0.001 | 0.070 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.072 ± 0.001 | - | unsupported | 0.036 ± 0.000 | 0.086 ± 0.001 | 0.110 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.152 ± 0.007 | - | unsupported | 0.120 ± 0.001 | 0.231 ± 0.002 | 0.337 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.557 ± 0.004 | - | unsupported | 0.144 ± 0.002 | 0.835 ± 0.015 | 1.192 ± 0.041 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.102 ± 0.002 | - | unsupported | 0.016 ± 0.001 | 0.057 ± 0.023 | 0.070 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.069 ± 0.001 | - | unsupported | 0.023 ± 0.001 | 0.083 ± 0.002 | 0.110 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.155 ± 0.002 | - | unsupported | 0.041 ± 0.001 | 0.186 ± 0.160 | 0.349 ± 0.095 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 8 | `1024x16x16,rhs=1` | 0.840 ± 0.007 | - | unsupported | 0.117 ± 0.001 | 0.539 ± 0.054 | 1.831 ± 0.406 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 8 | `1024x2x2,rhs=1` | 0.162 ± 0.002 | - | unsupported | 0.023 ± 0.001 | 0.084 ± 0.032 | 0.099 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 8 | `1024x4x4,rhs=1` | 0.109 ± 0.001 | - | unsupported | 0.031 ± 0.001 | 0.135 ± 0.023 | 0.154 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 8 | `1024x8x8,rhs=1` | 0.234 ± 0.002 | - | unsupported | 0.046 ± 0.000 | 0.348 ± 0.063 | 0.525 ± 0.029 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.151 ± 0.001 | - | unsupported | 0.193 ± 0.013 | 0.378 ± 0.004 | 0.198 ± 0.025 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.015 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.088 ± 0.001 | 0.043 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.032 ± 0.000 | - | unsupported | 0.024 ± 0.000 | 0.110 ± 0.001 | 0.063 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.101 ± 0.000 | - | unsupported | 0.088 ± 0.000 | 0.177 ± 0.002 | 0.099 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.162 ± 0.005 | - | unsupported | 0.195 ± 0.002 | 0.347 ± 0.015 | 0.218 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.016 ± 0.000 | - | unsupported | 0.019 ± 0.000 | 0.101 ± 0.050 | 0.045 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.033 ± 0.000 | - | unsupported | 0.024 ± 0.000 | 0.111 ± 0.002 | 0.065 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.106 ± 0.000 | - | unsupported | 0.088 ± 0.000 | 0.175 ± 0.031 | 0.110 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 8 | `1024x16x16,rhs=1` | 0.244 ± 0.004 | - | unsupported | 0.306 ± 0.002 | 0.570 ± 0.015 | 0.338 ± 0.028 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 8 | `1024x2x2,rhs=1` | 0.024 ± 0.000 | - | unsupported | 0.028 ± 0.000 | 0.160 ± 0.042 | 0.072 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 8 | `1024x4x4,rhs=1` | 0.051 ± 0.000 | - | unsupported | 0.037 ± 0.000 | 0.203 ± 0.030 | 0.104 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 8 | `1024x8x8,rhs=1` | 0.161 ± 0.001 | - | unsupported | 0.137 ± 0.001 | 0.321 ± 0.035 | 0.169 ± 0.011 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.100 ± 0.001 | - | 0.137 ± 0.001 | 0.100 ± 0.008 | 0.190 ± 0.003 | 0.122 ± 0.021 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.036 ± 0.000 | - | 0.038 ± 0.000 | 0.031 ± 0.000 | 0.041 ± 0.000 | 0.037 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.042 ± 0.000 | - | 0.045 ± 0.000 | 0.040 ± 0.001 | 0.053 ± 0.001 | 0.045 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.058 ± 0.000 | - | 0.069 ± 0.000 | 0.059 ± 0.001 | 0.085 ± 0.002 | 0.074 ± 0.011 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.103 ± 0.002 | - | 1.960 ± 1.379 | 0.103 ± 0.002 | 0.271 ± 0.080 | 0.121 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.038 ± 0.000 | - | 0.040 ± 0.000 | 0.032 ± 0.001 | 0.043 ± 0.086 | 0.037 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.044 ± 0.000 | - | 0.507 ± 0.003 | 0.040 ± 0.001 | 0.053 ± 0.000 | 0.043 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.061 ± 0.000 | - | 0.772 ± 0.678 | 0.060 ± 0.000 | 0.089 ± 0.003 | 0.067 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 8 | `1024x16x16,rhs=1` | 0.156 ± 0.001 | - | 1.954 ± 0.726 | 0.157 ± 0.002 | 0.320 ± 0.048 | 0.226 ± 0.018 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 8 | `1024x2x2,rhs=1` | 0.057 ± 0.000 | - | 0.061 ± 0.000 | 0.049 ± 0.001 | 0.072 ± 0.001 | 0.060 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 8 | `1024x4x4,rhs=1` | 0.067 ± 0.000 | - | 0.765 ± 0.003 | 0.061 ± 0.001 | 0.088 ± 0.001 | 0.071 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 8 | `1024x8x8,rhs=1` | 0.093 ± 0.000 | - | 0.830 ± 0.003 | 0.092 ± 0.001 | 0.145 ± 0.015 | 0.110 ± 0.015 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 24.749 ± 0.097 | - | 26.141 ± 0.204 | 38.689 ± 0.331 | 47.316 ± 0.136 | 33.736 ± 6.308 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 7.633 ± 2.066 | - | 9.066 ± 2.359 | 10.843 ± 0.077 | 32.624 ± 1.982 | 18.720 ± 6.116 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 8 | `1536x1536` | 7.347 ± 0.121 | - | 9.164 ± 0.111 | 8.433 ± 0.116 | 42.707 ± 22.659 | 27.125 ± 7.304 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 14.692 ± 0.170 | - | 14.511 ± 0.144 | 16.416 ± 0.043 | 14.797 ± 0.048 | 13.320 ± 0.265 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 5.036 ± 0.104 | - | 10.085 ± 8.197 | 3.723 ± 0.022 | 6.997 ± 0.241 | 5.271 ± 0.363 | - |
| cpu/linalg_uncovered | `det` | f64 | 8 | `1024x1024` | 5.203 ± 0.076 | - | 8.769 ± 5.910 | 3.493 ± 0.068 | 7.639 ± 0.103 | 4.580 ± 0.921 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 3.900 ± 0.005 | - | 3.903 ± 0.008 | 4.045 ± 0.024 | 3.924 ± 0.004 | 4.743 ± 0.020 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 3.734 ± 0.033 | - | 4.205 ± 0.035 | 3.718 ± 0.020 | 4.072 ± 0.014 | 4.703 ± 0.039 | - |
| cpu/linalg_uncovered | `eig` | f64 | 8 | `160x160` | 5.822 ± 0.028 | - | 6.258 ± 0.043 | 5.773 ± 0.031 | 6.380 ± 0.014 | 6.171 ± 0.091 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 4.574 ± 0.006 | - | 4.586 ± 0.004 | 4.482 ± 0.009 | 4.697 ± 0.007 | 6.093 ± 0.049 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 4.419 ± 0.007 | - | 4.854 ± 0.012 | 4.222 ± 0.010 | 5.004 ± 0.025 | 5.779 ± 0.011 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 8 | `192x192` | 6.676 ± 0.049 | - | 16.292 ± 0.017 | 6.457 ± 0.011 | 7.835 ± 0.019 | 7.584 ± 0.017 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 10.253 ± 0.030 | - | 10.295 ± 0.034 | 9.248 ± 0.005 | 19.774 ± 0.080 | 7.000 ± 0.308 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 5.727 ± 0.011 | - | 6.209 ± 0.032 | 5.172 ± 0.025 | 12.880 ± 0.039 | 4.800 ± 0.013 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 8 | `512x512` | 7.440 ± 0.024 | - | 45.806 ± 1.299 | 6.848 ± 0.014 | 18.876 ± 0.050 | 7.433 ± 0.026 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.257 ± 0.000 | 0.264 ± 0.000 | unsupported | 0.297 ± 0.001 | 0.345 ± 0.008 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 4.142 ± 0.005 | 4.206 ± 0.006 | unsupported | 4.500 ± 0.012 | 4.951 ± 0.018 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.162 ± 0.004 | 0.835 ± 0.237 | unsupported | 0.159 ± 0.001 | 0.434 ± 0.017 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 1.820 ± 0.005 | 2.179 ± 0.442 | unsupported | 1.928 ± 0.023 | 3.364 ± 0.035 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 8 | `256x128` | 0.146 ± 0.001 | 0.880 ± 0.659 | unsupported | 0.210 ± 0.005 | 0.674 ± 0.754 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 8 | `768x384` | 2.111 ± 0.016 | 2.647 ± 0.357 | unsupported | 1.715 ± 0.008 | 4.841 ± 0.033 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 22.769 ± 0.109 | - | 22.865 ± 0.038 | 23.517 ± 0.104 | 24.345 ± 0.088 | 17.161 ± 0.075 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 12.462 ± 0.024 | - | 12.974 ± 0.058 | 6.855 ± 0.038 | 8.416 ± 0.700 | 6.160 ± 0.066 | - |
| cpu/linalg_uncovered | `inv` | f64 | 8 | `768x768` | 11.777 ± 0.149 | - | 10.640 ± 1.280 | 6.557 ± 0.031 | 8.988 ± 0.386 | 6.154 ± 0.366 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 9.428 ± 0.016 | 9.282 ± 0.022 | 4.837 ± 0.077 | 24.388 ± 0.047 | 4.264 ± 0.026 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 4.310 ± 0.029 | 4.327 ± 0.018 | 2.074 ± 0.013 | 16.615 ± 0.656 | 2.404 ± 0.029 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 8 | `768x384,rhs=16` | - | failed | failed | failed | 23.090 ± 0.682 | 3.531 ± 0.475 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 14.664 ± 0.118 | - | 15.139 ± 0.119 | 21.053 ± 0.098 | 21.454 ± 0.115 | 21.780 ± 9.611 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 5.101 ± 0.102 | - | 5.888 ± 0.072 | 6.706 ± 0.029 | 13.596 ± 0.362 | 14.108 ± 9.839 | - |
| cpu/linalg_uncovered | `lu` | f64 | 8 | `1024x1024` | 5.198 ± 0.078 | - | 6.758 ± 20.727 | 6.826 ± 0.018 | 15.981 ± 3.629 | 17.400 ± 7.477 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.778 ± 0.185 | - | 0.652 ± 0.046 | 0.730 ± 0.024 | 9.357 ± 0.264 | 1.625 ± 0.010 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.975 ± 0.211 | - | 1.454 ± 0.042 | 0.733 ± 0.065 | 2.709 ± 0.163 | 1.629 ± 0.015 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 8 | `2048x2048` | 0.976 ± 0.284 | - | 2.680 ± 0.756 | 1.047 ± 0.009 | 7.328 ± 0.345 | 2.569 ± 0.010 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 10.396 ± 0.074 | - | 10.154 ± 0.039 | 10.934 ± 0.051 | 10.277 ± 0.013 | 10.305 ± 0.278 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 12.192 ± 0.045 | - | 10.807 ± 0.695 | 6.784 ± 0.030 | 8.095 ± 0.206 | 7.219 ± 0.149 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 8 | `512x256` | 11.764 ± 0.052 | - | 10.943 ± 0.883 | 9.050 ± 0.037 | 12.911 ± 0.569 | 11.523 ± 0.902 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 10.380 ± 0.053 | - | 10.142 ± 0.032 | 10.985 ± 0.098 | 10.253 ± 0.017 | 10.307 ± 0.266 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 8.889 ± 0.046 | - | 8.061 ± 0.065 | 6.801 ± 0.016 | 8.483 ± 0.613 | 7.244 ± 0.067 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 8 | `512x256` | 11.740 ± 0.080 | - | 11.487 ± 0.056 | 9.042 ± 0.025 | 13.324 ± 0.422 | 11.338 ± 0.305 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 0.598 ± 0.001 | 0.607 ± 0.001 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 11.935 ± 0.069 | 12.032 ± 0.052 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 0.507 ± 0.003 | 0.740 ± 0.356 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 8.269 ± 0.016 | 10.425 ± 0.369 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 8 | `256x128` | 0.820 ± 0.002 | 1.354 ± 0.316 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 8 | `768x384` | 12.068 ± 0.057 | 14.463 ± 0.318 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 0.560 ± 0.001 | 0.569 ± 0.001 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 11.367 ± 0.089 | 11.471 ± 0.091 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 0.472 ± 0.002 | 1.200 ± 0.214 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 7.790 ± 0.015 | 7.921 ± 0.040 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 8 | `256x128` | 0.757 ± 0.008 | 0.899 ± 0.008 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 8 | `768x384` | 11.289 ± 0.013 | 13.738 ± 0.316 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 14.669 ± 0.149 | - | 14.480 ± 0.115 | 16.438 ± 0.045 | 14.793 ± 0.038 | 13.374 ± 1.154 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 5.137 ± 0.062 | - | 43.917 ± 39.801 | 3.721 ± 0.008 | 7.202 ± 0.251 | 5.242 ± 0.055 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 8 | `1024x1024` | 5.188 ± 0.070 | - | 6.091 ± 2.996 | 3.491 ± 0.021 | 7.619 ± 0.144 | 4.574 ± 0.218 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 31.616 ± 0.115 | 31.543 ± 0.124 | 31.964 ± 0.069 | 33.643 ± 0.057 | 33.091 ± 0.203 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 13.233 ± 0.400 | 13.694 ± 0.194 | 13.712 ± 0.046 | 21.115 ± 0.063 | 20.243 ± 0.381 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 8 | `768x384` | - | 17.964 ± 0.068 | 18.164 ± 1.371 | 18.149 ± 0.081 | 29.622 ± 0.262 | 30.066 ± 2.782 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 20.660 ± 0.031 | - | 51.349 ± 0.194 | 18.401 ± 0.075 | 86.422 ± 0.357 | 20.044 ± 0.249 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 10.796 ± 0.046 | - | 32.756 ± 9.795 | 7.077 ± 0.048 | 74.609 ± 0.669 | 7.787 ± 0.019 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 8 | `4096x4096,rhs=64` | 15.512 ± 0.095 | - | 45.063 ± 6.000 | 9.502 ± 0.161 | 77.599 ± 4.238 | 8.853 ± 0.078 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.050 ± 0.000 | 0.050 ± 0.000 | unsupported | 0.070 ± 0.001 | 0.055 ± 0.002 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 0.799 ± 0.002 | 0.801 ± 0.004 | unsupported | 1.115 ± 0.016 | 0.395 ± 0.177 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.014 ± 0.000 | 0.263 ± 0.106 | unsupported | 0.071 ± 0.048 | 0.046 ± 0.024 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.207 ± 0.003 | 0.872 ± 0.001 | unsupported | 0.333 ± 0.019 | 0.360 ± 0.101 | - | - |
| cpu/neural_public_api | `erf` | f32 | 8 | `1024x64` | 0.013 ± 0.000 | 0.014 ± 0.696 | unsupported | 0.069 ± 0.047 | 0.060 ± 0.007 | - | - |
| cpu/neural_public_api | `erf` | f32 | 8 | `4096x256` | 0.164 ± 0.002 | 1.009 ± 0.003 | unsupported | 0.267 ± 0.009 | 0.376 ± 0.083 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.025 ± 0.000 | 0.025 ± 0.000 | unsupported | 0.042 ± 0.001 | 0.087 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 0.394 ± 0.003 | 0.401 ± 0.002 | unsupported | 0.596 ± 0.024 | 1.490 ± 0.012 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.007 ± 0.000 | 0.007 ± 0.457 | unsupported | 0.025 ± 0.003 | 0.039 ± 0.007 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.122 ± 0.004 | 0.124 ± 0.540 | unsupported | 0.225 ± 0.021 | 0.348 ± 0.042 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 8 | `1024x64` | 0.007 ± 0.002 | 0.346 ± 0.362 | unsupported | 0.032 ± 0.002 | 0.042 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 8 | `4096x256` | 0.126 ± 0.418 | 0.533 ± 0.005 | unsupported | 0.221 ± 0.016 | 0.300 ± 0.018 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.073 ± 0.000 | 0.073 ± 0.000 | unsupported | 0.098 ± 0.000 | 0.066 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 1.159 ± 0.004 | 1.168 ± 0.002 | unsupported | 1.611 ± 0.019 | 1.150 ± 0.006 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.019 ± 0.000 | 0.019 ± 0.464 | unsupported | 0.021 ± 0.000 | 0.042 ± 0.006 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 0.303 ± 0.003 | 1.063 ± 0.002 | unsupported | 0.330 ± 0.005 | 0.423 ± 0.083 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 8 | `1024x64` | 0.016 ± 0.000 | 0.199 ± 0.371 | unsupported | 0.028 ± 0.001 | 0.059 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 8 | `4096x256` | 0.233 ± 0.004 | 1.151 ± 0.001 | unsupported | 0.352 ± 0.007 | 0.461 ± 0.036 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.025 ± 0.002 | 0.038 ± 0.001 | unsupported | 0.070 ± 0.001 | 0.068 ± 0.001 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.362 ± 0.002 | 0.375 ± 0.003 | unsupported | 0.501 ± 0.002 | 1.209 ± 0.018 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.023 ± 0.000 | 0.036 ± 0.198 | unsupported | 0.015 ± 0.000 | 0.061 ± 0.004 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.200 ± 0.004 | 2.262 ± 2.010 | unsupported | 0.142 ± 0.002 | 0.526 ± 0.016 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 8 | `1024x64` | 0.035 ± 0.000 | 0.056 ± 0.001 | unsupported | 0.024 ± 0.001 | 0.092 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 8 | `4096x256` | 0.339 ± 0.011 | 0.419 ± 3.468 | unsupported | 0.124 ± 0.028 | 0.694 ± 0.168 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.042 ± 0.002 | 0.049 ± 0.000 | unsupported | 0.046 ± 0.000 | 0.107 ± 0.000 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 0.632 ± 0.009 | 0.673 ± 0.016 | unsupported | 0.443 ± 0.009 | 1.896 ± 0.016 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.028 ± 0.000 | 0.036 ± 0.000 | unsupported | 0.009 ± 0.000 | 0.095 ± 0.004 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.286 ± 0.003 | 0.602 ± 2.499 | unsupported | 0.126 ± 0.012 | 0.830 ± 0.093 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 8 | `1024x64` | 0.043 ± 0.001 | 0.056 ± 0.001 | unsupported | 0.020 ± 0.001 | 0.146 ± 0.003 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 8 | `4096x256` | 0.458 ± 0.014 | 0.569 ± 2.321 | unsupported | 0.122 ± 0.044 | 1.057 ± 0.262 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.059 ± 0.001 | 0.072 ± 0.001 | unsupported | unsupported | 0.117 ± 0.001 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 0.923 ± 0.009 | 0.932 ± 0.015 | unsupported | unsupported | 2.329 ± 0.009 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.038 ± 0.000 | 0.052 ± 0.001 | unsupported | unsupported | 0.101 ± 0.006 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 0.381 ± 0.007 | 0.439 ± 2.950 | unsupported | unsupported | 1.123 ± 0.026 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 8 | `1024x64` | 0.055 ± 0.000 | 0.075 ± 0.000 | unsupported | unsupported | 0.161 ± 0.002 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 8 | `4096x256` | 0.476 ± 0.004 | 0.588 ± 2.931 | unsupported | unsupported | 1.488 ± 0.064 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.004 ± 0.000 | 0.008 ± 0.000 | unsupported | 0.006 ± 0.000 | 0.016 ± 0.000 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.045 ± 0.001 | 0.048 ± 0.001 | unsupported | 0.030 ± 0.001 | 0.160 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.004 ± 0.000 | 0.242 ± 0.003 | unsupported | 0.004 ± 0.000 | 0.014 ± 0.000 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.018 ± 0.000 | 0.269 ± 0.003 | unsupported | 0.011 ± 0.013 | 0.085 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 8 | `1024x64` | 0.005 ± 0.000 | 0.011 ± 0.000 | unsupported | 0.006 ± 0.000 | 0.022 ± 0.000 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 8 | `4096x256` | 0.017 ± 0.002 | 0.723 ± 0.001 | unsupported | 0.011 ± 0.004 | 0.100 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.015 ± 0.000 | 0.023 ± 0.000 | unsupported | 0.036 ± 0.000 | 0.070 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.194 ± 0.003 | 0.205 ± 0.003 | unsupported | 0.318 ± 0.003 | 1.223 ± 0.007 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.013 ± 0.000 | 0.257 ± 0.238 | unsupported | 0.016 ± 0.001 | 0.051 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.129 ± 0.003 | 1.105 ± 0.005 | unsupported | 0.186 ± 0.003 | 0.497 ± 0.038 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 8 | `1024x64` | 0.021 ± 0.000 | 0.034 ± 0.694 | unsupported | 0.045 ± 0.003 | 0.082 ± 0.005 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 8 | `4096x256` | 0.165 ± 0.004 | 1.675 ± 0.010 | unsupported | 0.178 ± 0.002 | 0.538 ± 0.043 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.017 ± 0.000 | 0.017 ± 0.000 | unsupported | 0.036 ± 0.001 | 0.058 ± 0.001 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.268 ± 0.001 | 0.276 ± 0.003 | unsupported | 0.585 ± 0.031 | 1.028 ± 0.020 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.005 ± 0.000 | 0.243 ± 0.004 | unsupported | 0.027 ± 0.047 | 0.041 ± 0.003 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.112 ± 0.006 | 0.604 ± 0.006 | unsupported | 0.223 ± 0.020 | 0.368 ± 0.034 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 8 | `1024x64` | 0.005 ± 0.000 | 0.006 ± 0.690 | unsupported | 0.034 ± 0.001 | 0.060 ± 0.004 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 8 | `4096x256` | 0.122 ± 0.006 | 0.831 ± 0.003 | unsupported | 0.206 ± 0.018 | 0.394 ± 0.023 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.018 ± 0.000 | 0.018 ± 0.000 | unsupported | 0.036 ± 0.000 | 0.060 ± 0.001 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.283 ± 0.003 | 0.291 ± 0.003 | unsupported | 0.583 ± 0.028 | 1.065 ± 0.020 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.005 ± 0.000 | 0.005 ± 0.454 | unsupported | 0.025 ± 0.002 | 0.044 ± 0.004 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.114 ± 0.005 | 0.607 ± 0.002 | unsupported | 0.223 ± 0.020 | 0.387 ± 0.048 | - | - |
| cpu/neural_public_api | `silu` | f32 | 8 | `1024x64` | 0.005 ± 0.000 | 0.264 ± 0.360 | unsupported | 0.035 ± 0.002 | 0.061 ± 0.004 | - | - |
| cpu/neural_public_api | `silu` | f32 | 8 | `4096x256` | 0.122 ± 0.394 | 0.818 ± 0.004 | unsupported | 0.210 ± 0.016 | 0.422 ± 0.023 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.040 ± 0.000 | 0.049 ± 0.001 | unsupported | 0.050 ± 0.001 | 0.106 ± 0.001 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 0.628 ± 0.017 | 0.666 ± 0.024 | unsupported | 0.440 ± 0.003 | 1.839 ± 0.012 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.026 ± 0.000 | 0.036 ± 0.000 | unsupported | 0.010 ± 0.000 | 0.091 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.298 ± 0.047 | 0.382 ± 2.504 | unsupported | 0.138 ± 0.003 | 0.795 ± 0.070 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 8 | `1024x64` | 0.040 ± 0.000 | 0.057 ± 0.381 | unsupported | 0.019 ± 0.002 | 0.140 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 8 | `4096x256` | 0.462 ± 0.009 | 0.514 ± 0.059 | unsupported | 0.125 ± 0.031 | 0.890 ± 0.048 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.061 ± 0.000 | 0.061 ± 0.000 | unsupported | 0.083 ± 0.000 | 0.140 ± 0.002 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 0.970 ± 0.004 | 0.976 ± 0.002 | unsupported | 1.350 ± 0.023 | 2.415 ± 0.011 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.016 ± 0.000 | 0.172 ± 0.468 | unsupported | 0.048 ± 0.002 | 0.082 ± 0.009 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.253 ± 0.004 | 0.949 ± 0.005 | unsupported | 0.388 ± 0.015 | 1.061 ± 0.061 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 8 | `1024x64` | 0.013 ± 0.000 | 0.281 ± 0.366 | unsupported | 0.073 ± 0.003 | 0.125 ± 0.002 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 8 | `4096x256` | 0.196 ± 0.112 | 0.748 ± 0.004 | unsupported | 0.311 ± 0.009 | 1.142 ± 0.029 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.323 ± 0.001 | 0.522 ± 0.001 | unsupported | 0.129 ± 0.001 | 0.062 ± 0.001 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 6.719 ± 0.026 | 12.227 ± 0.034 | unsupported | 3.058 ± 0.101 | 1.175 ± 0.013 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.133 ± 0.001 | 0.343 ± 0.001 | unsupported | 0.079 ± 0.003 | 0.039 ± 0.003 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 3.835 ± 0.118 | 8.994 ± 0.072 | unsupported | 0.629 ± 0.012 | 0.412 ± 0.049 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 8 | `1024x64` | 0.149 ± 0.001 | 0.489 ± 0.003 | unsupported | 0.113 ± 0.002 | 0.053 ± 0.005 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 8 | `4096x256` | 3.893 ± 0.038 | 10.295 ± 0.088 | unsupported | 0.782 ± 0.024 | 0.485 ± 0.021 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 21.511 ± 0.097 | - | unsupported | 21.431 ± 0.077 | - | 21.347 ± 0.099 | 67.000 ± 0.314 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 17.325 ± 0.048 | - | unsupported | 17.363 ± 0.058 | - | 21.340 ± 0.070 | 22.973 ± 0.104 |
| cpu/output_reuse | `add_into` | f64 | 8 | `33554432` | 18.699 ± 0.102 | - | unsupported | 18.733 ± 0.101 | - | 22.035 ± 0.089 | 18.772 ± 0.119 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 15.040 ± 0.063 | - | unsupported | 15.103 ± 0.037 | - | 15.058 ± 0.060 | 14.665 ± 0.118 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 12.351 ± 0.053 | - | unsupported | 12.608 ± 0.067 | - | 15.082 ± 0.037 | 12.601 ± 0.080 |
| cpu/output_reuse | `conj_into` | c64 | 8 | `16777216` | 13.669 ± 0.077 | - | unsupported | 13.719 ± 0.135 | - | 15.585 ± 0.027 | 13.801 ± 0.202 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 12.387 ± 0.099 | - | unsupported | 15.252 ± 0.035 | - | 11.488 ± 0.056 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 12.382 ± 0.044 | - | unsupported | 12.676 ± 0.059 | - | 11.746 ± 0.065 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 8 | `33554432` | 13.650 ± 0.089 | - | unsupported | 13.750 ± 0.081 | - | 12.181 ± 0.099 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 21.628 ± 0.032 | - | unsupported | 21.409 ± 0.044 | - | 21.319 ± 0.049 | 68.032 ± 0.186 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 17.312 ± 0.043 | - | unsupported | 17.333 ± 0.076 | - | 21.303 ± 0.042 | 22.952 ± 0.126 |
| cpu/output_reuse | `div_into` | f64 | 8 | `33554432` | 18.656 ± 0.085 | - | unsupported | 18.725 ± 0.133 | - | 22.208 ± 0.065 | 18.759 ± 0.403 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 29.206 ± 0.063 | - | unsupported | 30.086 ± 0.124 | - | 31.015 ± 0.102 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 14.052 ± 7.913 | - | unsupported | 9.865 ± 0.082 | - | 8.194 ± 0.048 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 8 | `1024x1024` | 13.334 ± 5.033 | - | unsupported | 8.588 ± 0.059 | - | 6.629 ± 0.034 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 28.955 ± 0.040 | - | unsupported | 29.741 ± 0.070 | - | 30.750 ± 0.050 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 8.906 ± 0.037 | - | unsupported | 9.698 ± 0.027 | - | 8.163 ± 0.036 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 8 | `1024x1024` | 7.905 ± 0.048 | - | unsupported | 8.275 ± 0.042 | - | 6.495 ± 0.031 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 21.488 ± 0.040 | - | unsupported | 21.393 ± 0.045 | - | 21.379 ± 0.071 | 67.609 ± 0.176 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 17.334 ± 0.037 | - | unsupported | 17.312 ± 0.070 | - | 21.329 ± 0.078 | 22.898 ± 0.149 |
| cpu/output_reuse | `mul_into` | f64 | 8 | `33554432` | 18.674 ± 0.045 | - | unsupported | 18.712 ± 0.121 | - | 22.087 ± 0.037 | 18.842 ± 0.267 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 15.392 ± 0.029 | - | unsupported | 15.310 ± 0.057 | - | 15.268 ± 0.034 | 50.186 ± 0.365 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 12.389 ± 0.037 | - | unsupported | 12.689 ± 0.073 | - | 15.254 ± 0.069 | 17.661 ± 0.080 |
| cpu/output_reuse | `neg_into` | f64 | 8 | `33554432` | 13.649 ± 0.042 | - | unsupported | 13.736 ± 0.082 | - | 15.698 ± 0.028 | 14.228 ± 0.277 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.370 ± 0.000 | unsupported | unsupported | 0.444 ± 0.001 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 5.961 ± 0.012 | unsupported | unsupported | 6.802 ± 0.017 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.156 ± 0.001 | unsupported | unsupported | 0.188 ± 0.001 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 1.903 ± 0.004 | unsupported | unsupported | 2.200 ± 0.004 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 8 | `256x256,rhs=16` | 0.219 ± 0.001 | unsupported | unsupported | 0.279 ± 0.001 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 8 | `768x768,rhs=16` | 1.821 ± 0.024 | unsupported | unsupported | 1.943 ± 0.003 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 21.502 ± 0.092 | - | unsupported | 21.440 ± 0.121 | - | 21.347 ± 0.041 | 66.960 ± 0.221 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 17.321 ± 0.066 | - | unsupported | 17.333 ± 0.095 | - | 21.368 ± 0.077 | 22.974 ± 0.238 |
| cpu/output_reuse | `sub_into` | f64 | 8 | `33554432` | 18.655 ± 0.064 | - | unsupported | 18.763 ± 0.150 | - | 22.039 ± 0.065 | 18.722 ± 0.124 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 40.849 ± 1.231 | - | 0.001 ± 0.000 | 42.132 ± 0.782 | 42.904 ± 0.310 | 51.059 ± 1.552 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 15.660 ± 0.200 | - | 0.001 ± 0.000 | 14.047 ± 0.177 | 24.442 ± 0.152 | 51.398 ± 0.672 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 8 | `8192x1 -> 8192x4096` | 12.466 ± 0.396 | - | 0.002 ± 0.000 | 12.006 ± 0.302 | 23.025 ± 0.107 | 69.060 ± 2.725 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 27.035 ± 0.985 | - | 83.614 ± 0.615 | 26.963 ± 0.363 | 34.264 ± 0.860 | 28.218 ± 0.338 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 11.150 ± 0.256 | - | 38.960 ± 0.300 | 10.926 ± 0.282 | 18.815 ± 0.292 | 28.464 ± 0.417 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 8 | `33554432` | 10.092 ± 0.183 | - | 38.647 ± 0.184 | 10.066 ± 0.190 | 17.114 ± 0.341 | 38.929 ± 0.443 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 87.134 ± 2.367 | - | 87.123 ± 2.302 | 86.284 ± 2.037 | 113.951 ± 0.472 | 101.492 ± 3.644 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 27.314 ± 0.496 | - | 27.249 ± 0.092 | 27.438 ± 0.198 | 50.177 ± 0.313 | 100.576 ± 2.889 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 8 | `8192 -> 8192x8192` | 21.453 ± 0.099 | - | 21.533 ± 0.230 | 22.697 ± 0.211 | 44.115 ± 0.138 | 136.737 ± 3.544 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 24.989 ± 0.337 | - | 23.250 ± 0.258 | 31.133 ± 1.836 | 24.877 ± 0.226 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 14.322 ± 0.214 | - | 13.832 ± 0.376 | 12.039 ± 0.133 | 16.967 ± 0.263 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 8 | `8388608x2x2 -> 8388608x2` | 17.688 ± 0.515 | - | 17.127 ± 0.157 | 10.285 ± 0.119 | 14.916 ± 0.227 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 52.032 ± 0.926 | - | 0.001 ± 0.000 | 47.420 ± 1.056 | 52.552 ± 0.528 | 59.259 ± 0.864 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 18.013 ± 0.121 | - | 0.001 ± 0.000 | 17.564 ± 0.454 | 21.239 ± 1.228 | 58.762 ± 0.458 | - |
| cpu/structural_shape | `reshape` | f64 | 8 | `33554432 -> 8192x4096` | 15.186 ± 0.460 | - | 0.002 ± 0.000 | 15.209 ± 0.320 | 17.549 ± 0.282 | 75.658 ± 0.718 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 43.199 ± 0.140 | - | 0.001 ± 0.000 | 31.279 ± 0.105 | 39.196 ± 0.577 | 94.471 ± 0.579 | 60.623 ± 1.100 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 16.805 ± 0.386 | - | 0.001 ± 0.000 | 30.830 ± 0.117 | 24.009 ± 2.254 | 94.848 ± 1.758 | 23.688 ± 2.093 |
| cpu/structural_shape | `transpose` | f64 | 8 | `4096x4096` | 14.371 ± 0.210 | - | 0.002 ± 0.000 | 41.023 ± 0.390 | 21.776 ± 0.625 | 138.807 ± 0.432 | 26.127 ± 1.577 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 25.342 ± 0.524 | - | 25.122 ± 0.668 | 27.773 ± 0.732 | 28.062 ± 2.081 | 31.404 ± 0.947 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 8.410 ± 0.107 | - | 8.594 ± 5.654 | 10.313 ± 0.261 | 16.439 ± 0.201 | 30.546 ± 0.867 | - |
| cpu/structural_shape | `tril` | f64 | 8 | `4096x4096` | 6.886 ± 0.124 | - | 6.979 ± 3.131 | 8.149 ± 0.325 | 14.867 ± 0.285 | 39.403 ± 0.867 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 23.664 ± 0.690 | - | 23.317 ± 0.543 | 26.642 ± 0.602 | 27.840 ± 4.471 | 29.781 ± 0.119 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 8.491 ± 0.093 | - | 8.506 ± 4.638 | 8.756 ± 0.376 | 16.406 ± 0.223 | 30.626 ± 0.080 | - |
| cpu/structural_shape | `triu` | f64 | 8 | `4096x4096` | 6.875 ± 0.197 | - | 9.961 ± 3.334 | 7.554 ± 0.155 | 14.414 ± 0.900 | 43.530 ± 0.273 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 8 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 8 | `33554432 -> 8192x4096` | 0.001 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 8 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 8 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 2.858951 | 178684.44 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 1.687938 | 105496.12 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 1.891598 | 118224.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 1.128410 | 70525.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 0.795265 | 49704.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 1.044516 | 65282.25 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 16 | 2.625508 | 164094.25 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 8 | tenferro-rs direct API (ms) | 16 | 1.139845 | 71240.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 16 | 1.492149 | 93259.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 9.867374 | 616710.88 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 9.877001 | 617312.56 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 6.191999 | 386999.94 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 14.814998 | 925937.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 8 | tenferro-rs direct API (ms) | 16 | 8.693461 | 543341.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 8 | tenferro-rs eager mode (ms) | 16 | 14.204041 | 887752.56 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 5.518283 | 344892.69 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 4.747705 | 296731.56 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 4.106930 | 256683.12 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 4.217817 | 263613.56 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 79.219256 | 4951203.50 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 72.005288 | 4500330.50 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 66.272375 | 4142023.44 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 67.301454 | 4206340.88 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.938682 | 433667.62 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 2.536147 | 158509.19 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 2.591199 | 161949.94 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 13.354056 | 834628.50 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 53.816243 | 3363515.19 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 30.843910 | 1927744.38 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 29.126702 | 1820418.88 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 34.858606 | 2178662.88 |
| householder_qr | f64 | `256x128` | 8 | JAX Python (XLA CPU) (ms) | 16 | 10.782694 | 673918.38 |
| householder_qr | f64 | `256x128` | 8 | PyTorch Python (ms) | 16 | 3.366642 | 210415.12 |
| householder_qr | f64 | `256x128` | 8 | tenferro-rs direct API (ms) | 16 | 2.337501 | 146093.81 |
| householder_qr | f64 | `256x128` | 8 | tenferro-rs eager mode (ms) | 16 | 14.085025 | 880314.06 |
| householder_qr | f64 | `768x384` | 8 | JAX Python (XLA CPU) (ms) | 16 | 77.450819 | 4840676.19 |
| householder_qr | f64 | `768x384` | 8 | PyTorch Python (ms) | 16 | 27.447027 | 1715439.19 |
| householder_qr | f64 | `768x384` | 8 | tenferro-rs direct API (ms) | 16 | 33.768738 | 2110546.12 |
| householder_qr | f64 | `768x384` | 8 | tenferro-rs eager mode (ms) | 16 | 42.355986 | 2647249.12 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 9.568025 | 598001.56 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 9.707025 | 606689.06 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 190.963077 | 11935192.31 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 192.505955 | 12031622.19 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 8.112602 | 507037.63 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 11.846098 | 740381.12 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 132.300303 | 8268768.94 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 166.793410 | 10424588.12 |
| rank_revealing_qr | f64 | `256x128` | 8 | tenferro-rs direct API (ms) | 16 | 13.114519 | 819657.44 |
| rank_revealing_qr | f64 | `256x128` | 8 | tenferro-rs eager mode (ms) | 16 | 21.658486 | 1353655.38 |
| rank_revealing_qr | f64 | `768x384` | 8 | tenferro-rs direct API (ms) | 16 | 193.092324 | 12068270.25 |
| rank_revealing_qr | f64 | `768x384` | 8 | tenferro-rs eager mode (ms) | 16 | 231.406505 | 14462906.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 8.966633 | 560414.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 9.102447 | 568902.94 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 181.870668 | 11366916.75 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 183.537016 | 11471063.50 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 7.553329 | 472083.06 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 19.205056 | 1200316.00 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 124.636582 | 7789786.38 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 126.739229 | 7921201.81 |
| rank_revealing_qr_deficient | f64 | `256x128` | 8 | tenferro-rs direct API (ms) | 16 | 12.115219 | 757201.19 |
| rank_revealing_qr_deficient | f64 | `256x128` | 8 | tenferro-rs eager mode (ms) | 16 | 14.378273 | 898642.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 8 | tenferro-rs direct API (ms) | 16 | 180.626616 | 11289163.50 |
| rank_revealing_qr_deficient | f64 | `768x384` | 8 | tenferro-rs eager mode (ms) | 16 | 219.805959 | 13737872.44 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.510872 | 54857.38 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.507287 | 70426.36 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.202142 | 50033.47 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.208554 | 50133.66 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.156239 | 394529.88 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.919266 | 1114908.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.389547 | 798693.38 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.407371 | 800921.38 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.947947 | 46061.67 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.521987 | 70656.05 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.905090 | 14142.03 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 16.842845 | 263169.45 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.880413 | 360051.62 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.660042 | 332505.25 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.653946 | 206743.25 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.979452 | 872431.50 |
| erf | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 3.833093 | 59892.08 |
| erf | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 4.404746 | 68824.16 |
| erf | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 0.843515 | 13179.92 |
| erf | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 0.869283 | 13582.55 |
| erf | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 3.008608 | 376076.00 |
| erf | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 2.136557 | 267069.62 |
| erf | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 1.310862 | 163857.75 |
| erf | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 8.074578 | 1009322.25 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.589369 | 87333.89 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.699185 | 42174.77 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.577091 | 24642.05 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.597108 | 24954.81 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.923890 | 1490486.25 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.770167 | 596270.88 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.148592 | 393574.00 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.205929 | 400741.12 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.478912 | 38733.00 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 1.618885 | 25295.08 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.442362 | 6911.91 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 0.471164 | 7361.94 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.785922 | 348240.25 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.800675 | 225084.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.979564 | 122445.50 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.995784 | 124473.00 |
| gelu | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 2.691696 | 42057.75 |
| gelu | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 2.057392 | 32146.75 |
| gelu | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 0.434356 | 6786.81 |
| gelu | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 22.136746 | 345886.66 |
| gelu | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 2.403056 | 300382.00 |
| gelu | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 1.770827 | 221353.38 |
| gelu | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 1.006826 | 125853.25 |
| gelu | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 4.265738 | 533217.25 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.195072 | 65548.00 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.299889 | 98435.77 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.643080 | 72548.12 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.662627 | 72853.55 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.198168 | 1149771.00 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.890142 | 1611267.75 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.275099 | 1159387.38 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.344810 | 1168101.25 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.716676 | 42448.06 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 1.375744 | 21496.00 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.243222 | 19425.34 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.224958 | 19139.97 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.381416 | 422677.00 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.641659 | 330207.38 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.427087 | 303385.88 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.505006 | 1063125.75 |
| gelu_tanh | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 3.753346 | 58646.03 |
| gelu_tanh | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 1.786104 | 27907.88 |
| gelu_tanh | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 1.013778 | 15840.28 |
| gelu_tanh | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 12.748472 | 199194.88 |
| gelu_tanh | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 3.691854 | 461481.75 |
| gelu_tanh | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 2.816303 | 352037.88 |
| gelu_tanh | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 1.865909 | 233238.62 |
| gelu_tanh | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 9.204369 | 1150546.12 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.337697 | 67776.52 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.466059 | 69782.17 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.621695 | 25338.98 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.461591 | 38462.36 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.671635 | 1208954.38 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.009959 | 501244.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.894699 | 361837.38 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.001879 | 375234.88 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.877669 | 60588.58 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.962895 | 15045.23 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.483227 | 23175.42 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.295528 | 35867.62 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.211928 | 526491.00 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.134247 | 141780.88 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.600219 | 200027.38 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 18.095000 | 2261875.00 |
| layer_norm | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 5.904440 | 92256.88 |
| layer_norm | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 1.531547 | 23930.42 |
| layer_norm | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 2.271593 | 35493.64 |
| layer_norm | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 3.562579 | 55665.30 |
| layer_norm | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 5.548586 | 693573.25 |
| layer_norm | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 0.995104 | 124388.00 |
| layer_norm | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 2.710027 | 338753.38 |
| layer_norm | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 3.348998 | 418624.75 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.856820 | 107137.81 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.941366 | 45958.84 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.706388 | 42287.31 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.143603 | 49118.80 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.169141 | 1896142.62 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.546124 | 443265.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.055248 | 631906.00 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.386847 | 673355.88 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.106296 | 95410.88 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.559424 | 8741.00 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.823423 | 28490.98 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.316997 | 36203.08 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.641751 | 830218.88 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.005313 | 125664.12 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.287543 | 285942.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.819061 | 602382.62 |
| log_softmax | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 9.362453 | 146288.33 |
| log_softmax | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 1.292499 | 20195.30 |
| log_softmax | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 2.735072 | 42735.50 |
| log_softmax | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 3.604455 | 56319.61 |
| log_softmax | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 8.453323 | 1056665.38 |
| log_softmax | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 0.976299 | 122037.38 |
| log_softmax | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 3.664907 | 458113.38 |
| log_softmax | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 4.555589 | 569448.62 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.510616 | 117353.38 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.745947 | 58530.42 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.616571 | 72133.92 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.633251 | 2329156.38 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.385295 | 923161.88 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.452199 | 931524.88 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.441877 | 100654.33 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.419533 | 37805.20 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.298946 | 51546.03 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.984437 | 1123054.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.048212 | 381026.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.509868 | 438733.50 |
| masked_softmax | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 10.294175 | 160846.48 |
| masked_softmax | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 3.496948 | 54639.81 |
| masked_softmax | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 4.771995 | 74562.42 |
| masked_softmax | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 11.902462 | 1487807.75 |
| masked_softmax | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 3.805925 | 475740.62 |
| masked_softmax | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 4.701857 | 587732.12 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 1.001787 | 15652.92 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.368358 | 5755.59 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.271707 | 4245.42 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.500523 | 7820.67 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.280104 | 160013.00 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.241881 | 30235.12 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.357667 | 44708.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.386611 | 48326.38 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 0.913032 | 14266.12 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.245826 | 3841.03 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.226901 | 3545.33 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 15.511926 | 242373.84 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 0.678756 | 84844.50 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.087198 | 10899.75 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.145444 | 18180.50 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.149714 | 268714.25 |
| reduce_mean | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 1.432803 | 22387.55 |
| reduce_mean | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 0.371861 | 5810.33 |
| reduce_mean | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 0.347927 | 5436.36 |
| reduce_mean | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 0.688662 | 10760.34 |
| reduce_mean | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 0.797365 | 99670.62 |
| reduce_mean | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 0.086308 | 10788.50 |
| reduce_mean | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 0.135619 | 16952.38 |
| reduce_mean | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 5.783259 | 722907.38 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.468881 | 69826.27 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.286144 | 35721.00 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.949531 | 14836.42 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.488216 | 23253.38 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.787722 | 1223465.25 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 2.545046 | 318130.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 1.551614 | 193951.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.643104 | 205388.00 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.259173 | 50924.58 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 1.023796 | 15996.81 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.860789 | 13449.83 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 16.468961 | 257327.52 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.979214 | 497401.75 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.487346 | 185918.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.033373 | 129171.63 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.839373 | 1104921.62 |
| rms_norm | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 5.236432 | 81819.25 |
| rms_norm | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 2.849964 | 44530.69 |
| rms_norm | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 1.362146 | 21283.53 |
| rms_norm | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 2.202285 | 34410.70 |
| rms_norm | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 4.307536 | 538442.00 |
| rms_norm | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 1.421717 | 177714.62 |
| rms_norm | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 1.322254 | 165281.75 |
| rms_norm | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 13.400199 | 1675024.88 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.734500 | 58351.56 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.280123 | 35626.92 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.089001 | 17015.64 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.092718 | 17073.72 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.227224 | 1028403.00 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.677644 | 584705.50 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.141935 | 267741.88 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.209601 | 276200.12 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.614810 | 40856.41 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 1.728173 | 27002.70 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.319609 | 4993.89 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 15.525730 | 242589.53 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.944942 | 368117.75 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.786408 | 223301.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.895472 | 111934.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.831653 | 603956.62 |
| sigmoid | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 3.848231 | 60128.61 |
| sigmoid | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 2.175558 | 33993.09 |
| sigmoid | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 0.324815 | 5075.23 |
| sigmoid | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 0.355302 | 5551.59 |
| sigmoid | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 3.151611 | 393951.38 |
| sigmoid | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 1.651036 | 206379.50 |
| sigmoid | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 0.975588 | 121948.50 |
| sigmoid | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 6.649176 | 831147.00 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.855906 | 60248.53 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.323694 | 36307.72 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.134125 | 17720.70 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.152510 | 18007.97 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.517112 | 1064639.00 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.666263 | 583282.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.262470 | 282808.75 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.326649 | 290831.12 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.802953 | 43796.14 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 1.620066 | 25313.53 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.327994 | 5124.91 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 0.350475 | 5476.17 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.097659 | 387207.38 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.784555 | 223069.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.911552 | 113944.00 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.857299 | 607162.38 |
| silu | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 3.872075 | 60501.17 |
| silu | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 2.242982 | 35046.59 |
| silu | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 0.332038 | 5188.09 |
| silu | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 16.898446 | 264038.22 |
| silu | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 3.376495 | 422061.88 |
| silu | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 1.679599 | 209949.88 |
| silu | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 0.977101 | 122137.62 |
| silu | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 6.543593 | 817949.12 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.791422 | 106115.97 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.204837 | 50075.58 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.528706 | 39511.03 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.138254 | 49035.22 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.708477 | 1838559.62 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.522209 | 440276.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.027698 | 628462.25 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.330211 | 666276.38 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.794708 | 90542.31 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.617009 | 9640.77 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.637456 | 25585.25 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.281252 | 35644.56 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.363355 | 795419.38 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.107378 | 138422.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.384099 | 298012.38 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.052199 | 381524.88 |
| softmax | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 8.947270 | 139801.09 |
| softmax | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 1.203196 | 18799.94 |
| softmax | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 2.575630 | 40244.22 |
| softmax | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 3.621066 | 56579.16 |
| softmax | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 7.123280 | 890410.00 |
| softmax | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 0.996316 | 124539.50 |
| softmax | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 3.695684 | 461960.50 |
| softmax | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 4.112337 | 514042.12 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.928103 | 139501.61 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.329020 | 83265.94 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.877602 | 60587.53 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.900525 | 60945.70 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 19.319550 | 2414943.75 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.796286 | 1349535.75 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.758360 | 969795.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.807552 | 975944.00 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.276155 | 82439.92 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.100110 | 48439.22 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.027421 | 16053.45 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.983172 | 171612.06 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.488794 | 1061099.25 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.103426 | 387928.25 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.025048 | 253131.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.589509 | 948688.62 |
| softplus | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 8.000782 | 125012.22 |
| softplus | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 4.690378 | 73287.16 |
| softplus | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 0.857101 | 13392.20 |
| softplus | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 17.966835 | 280731.80 |
| softplus | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 9.139675 | 1142459.38 |
| softplus | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 2.485086 | 310635.75 |
| softplus | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 1.564558 | 195569.75 |
| softplus | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 5.987252 | 748406.50 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.993162 | 62393.16 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.238035 | 128719.30 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 20.703475 | 323491.80 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 33.385648 | 521650.75 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.397095 | 1174636.88 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 24.461793 | 3057724.12 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 53.752364 | 6719045.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 97.818152 | 12227269.00 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.514348 | 39286.69 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.042341 | 78786.58 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.498734 | 132792.72 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 21.959938 | 343124.03 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.298576 | 412322.00 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.032473 | 629059.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 30.681173 | 3835146.62 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 71.949502 | 8993687.75 |
| take_along_axis | f32 | `1024x64` | 8 | JAX Python (XLA CPU) (ms) | 64 | 3.361587 | 52524.80 |
| take_along_axis | f32 | `1024x64` | 8 | PyTorch Python (ms) | 64 | 7.211050 | 112672.66 |
| take_along_axis | f32 | `1024x64` | 8 | tenferro-rs direct API (ms) | 64 | 9.542327 | 149098.86 |
| take_along_axis | f32 | `1024x64` | 8 | tenferro-rs eager mode (ms) | 64 | 31.280305 | 488754.77 |
| take_along_axis | f32 | `4096x256` | 8 | JAX Python (XLA CPU) (ms) | 8 | 3.881062 | 485132.75 |
| take_along_axis | f32 | `4096x256` | 8 | PyTorch Python (ms) | 8 | 6.256321 | 782040.12 |
| take_along_axis | f32 | `4096x256` | 8 | tenferro-rs direct API (ms) | 8 | 31.144791 | 3893098.88 |
| take_along_axis | f32 | `4096x256` | 8 | tenferro-rs eager mode (ms) | 8 | 82.362975 | 10295371.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 7.110271 | 444391.94 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 5.915132 | 369695.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 108.835740 | 6802233.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 95.374995 | 5960937.19 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 3.001991 | 187624.44 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 2.494181 | 155886.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 35.198957 | 2199934.81 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 30.440285 | 1902517.81 |
| solve_read_into | f64 | `256x256,rhs=16` | 8 | PyTorch Python (ms) | 16 | 4.457432 | 278589.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 8 | tenferro-rs direct API (ms) | 16 | 3.496783 | 218548.94 |
| solve_read_into | f64 | `768x768,rhs=16` | 8 | PyTorch Python (ms) | 16 | 31.083665 | 1942729.06 |
| solve_read_into | f64 | `768x768,rhs=16` | 8 | tenferro-rs direct API (ms) | 16 | 29.130441 | 1820652.56 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.011372 | 710.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001072 | 67.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.011341 | 708.81 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001072 | 67.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 8 | PyTorch Python (ms) | 16 | 0.018033 | 1127.06 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 8 | tenferro-rs direct API (ms) | 16 | 0.001733 | 108.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000070 | 4.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.010469 | 654.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.006973 | 435.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000111 | 6.94 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.010448 | 653.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.007414 | 463.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 8 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000111 | 6.94 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 8 | PyTorch Python (ms) | 16 | 0.016670 | 1041.88 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 8 | tenferro-rs direct API (ms) | 16 | 0.008205 | 512.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.010920 | 682.50 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005090 | 318.12 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000170 | 10.62 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.010840 | 677.50 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005630 | 351.88 |
| slice_view | f64 | `4194304 -> 2096128` | 8 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000280 | 17.50 |
| slice_view | f64 | `4194304 -> 2096128` | 8 | PyTorch Python (ms) | 16 | 0.017241 | 1077.56 |
| slice_view | f64 | `4194304 -> 2096128` | 8 | tenferro-rs direct API (ms) | 16 | 0.005691 | 355.69 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000051 | 3.19 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.010420 | 651.25 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003456 | 216.00 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000091 | 5.69 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.010419 | 651.19 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003847 | 240.44 |
| transpose_view | f64 | `4096x4096` | 8 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000090 | 5.62 |
| transpose_view | f64 | `4096x4096` | 8 | PyTorch Python (ms) | 16 | 0.016610 | 1038.12 |
| transpose_view | f64 | `4096x4096` | 8 | tenferro-rs direct API (ms) | 16 | 0.004017 | 251.06 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/elementwise_reduction/cos` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`julia-base`, 32.882 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/cos` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 14.2x faster than the slowest successful cell (`julia-base`, 47.484 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`julia-base`, 40.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/expm1` (f64, threads=8, shape=`4194304`): `tenferro-direct` is 14.8x faster than the slowest successful cell (`julia-base`, 23.819 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`julia-base`, 33.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`julia-base`, 33.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 15.1x faster than the slowest successful cell (`julia-base`, 50.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`julia-base`, 18.648 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=8, shape=`4194304`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`julia-base`, 29.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=8, shape=`4194304`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`julia-base`, 29.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/pow` (f64, threads=4, shape=`4194304`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`julia-base`, 71.578 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/pow` (f64, threads=8, shape=`4194304`): `pytorch-cpu` is 23.2x faster than the slowest successful cell (`julia-base`, 112.591 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/pow` (f64, threads=8, shape=`4194304`): `strided-jl` is 10.6x faster than the slowest successful cell (`julia-base`, 112.591 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/pow` (f64, threads=8, shape=`4194304`): `tenferro-direct` is 19.9x faster than the slowest successful cell (`julia-base`, 112.591 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 17.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 17.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 17.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 17.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 4.974 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 18.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 18.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 18.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 18.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 5.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 5.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/rsqrt` (f64, threads=8, shape=`33554432`): `pytorch-cpu` is 10.8x faster than the slowest successful cell (`julia-base`, 176.818 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/rsqrt` (f64, threads=8, shape=`33554432`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`julia-base`, 176.818 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/rsqrt` (f64, threads=8, shape=`33554432`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`julia-base`, 176.818 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`julia-base`, 31.247 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`julia-base`, 31.247 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 13.6x faster than the slowest successful cell (`julia-base`, 45.487 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=8, shape=`8388608`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`julia-base`, 52.927 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=8, shape=`8388608`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`julia-base`, 52.927 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 14.4x faster than the slowest successful cell (`tenferro-trace`, 1.317 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 16.8x faster than the slowest successful cell (`tenferro-trace`, 1.317 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `tenferro-direct` is 18.5x faster than the slowest successful cell (`tenferro-trace`, 1.317 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=8, shape=`262144`): `jax-cpu` is 18.6x faster than the slowest successful cell (`tenferro-trace`, 3.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=8, shape=`262144`): `pytorch-cpu` is 44.5x faster than the slowest successful cell (`tenferro-trace`, 3.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=8, shape=`262144`): `tenferro-direct` is 50.0x faster than the slowest successful cell (`tenferro-trace`, 3.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 12.2x faster than the slowest successful cell (`tenferro-trace`, 1.804 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 11.6x faster than the slowest successful cell (`tenferro-trace`, 1.804 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=8, shape=`262144`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 2.760 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 3506.9x faster than the slowest successful cell (`julia-base`, 4.464 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 2983.7x faster than the slowest successful cell (`julia-base`, 4.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=8, shape=`4194304 -> 2096128`): `tenferro-trace` is 2607.1x faster than the slowest successful cell (`julia-base`, 5.459 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1024x4x4`): `pytorch-cpu` is 14.5x faster than the slowest successful cell (`tenferro-trace`, 0.510 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1024x4x4`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 0.510 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 16.4x faster than the slowest successful cell (`jax-cpu`, 0.010 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.010 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 15.1x faster than the slowest successful cell (`jax-cpu`, 0.010 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=8, shape=`8x128x128`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`tenferro-trace`, 37.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=8, shape=`8x128x128`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`tenferro-trace`, 37.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.017 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=8, shape=`1024x8x8`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`tenferro-trace`, 1.424 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `pytorch-cpu` is 12.2x faster than the slowest successful cell (`tenferro-trace`, 0.553 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x2x2,rhs=1`): `tenferro-direct` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x8x8,rhs=1`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x8x8,rhs=1`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`4x2x2,rhs=1`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=8, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 15.5x faster than the slowest successful cell (`tenferro-trace`, 2.761 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_lu_factor` (f64, threads=8, shape=`1024x16x16,rhs=1`): `pytorch-cpu` is 15.6x faster than the slowest successful cell (`julia-base`, 1.831 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_lu_factor` (f64, threads=8, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 11.5x faster than the slowest successful cell (`julia-base`, 0.525 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x16x16,rhs=1`): `julia-base` is 16.2x faster than the slowest successful cell (`tenferro-trace`, 1.960 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x16x16,rhs=1`): `pytorch-cpu` is 19.0x faster than the slowest successful cell (`tenferro-trace`, 1.960 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x16x16,rhs=1`): `tenferro-direct` is 19.1x faster than the slowest successful cell (`tenferro-trace`, 1.960 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `julia-base` is 11.7x faster than the slowest successful cell (`tenferro-trace`, 0.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`tenferro-trace`, 0.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`tenferro-trace`, 0.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `julia-base` is 11.6x faster than the slowest successful cell (`tenferro-trace`, 0.772 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 12.9x faster than the slowest successful cell (`tenferro-trace`, 0.772 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 12.7x faster than the slowest successful cell (`tenferro-trace`, 0.772 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=8, shape=`1024x16x16,rhs=1`): `pytorch-cpu` is 12.5x faster than the slowest successful cell (`tenferro-trace`, 1.954 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=8, shape=`1024x16x16,rhs=1`): `tenferro-direct` is 12.5x faster than the slowest successful cell (`tenferro-trace`, 1.954 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=8, shape=`1024x4x4,rhs=1`): `julia-base` is 10.8x faster than the slowest successful cell (`tenferro-trace`, 0.765 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=8, shape=`1024x4x4,rhs=1`): `pytorch-cpu` is 12.5x faster than the slowest successful cell (`tenferro-trace`, 0.765 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=8, shape=`1024x4x4,rhs=1`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`tenferro-trace`, 0.765 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `pytorch-cpu` is 12.8x faster than the slowest successful cell (`jax-cpu`, 9.357 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`jax-cpu`, 9.357 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`jax-cpu`, 9.357 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 43.917 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`jax-cpu`, 74.609 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 18.6x faster than the slowest successful cell (`tenferro-eager`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=8, shape=`1024x64`): `pytorch-cpu` is 10.8x faster than the slowest successful cell (`tenferro-eager`, 0.346 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 51.0x faster than the slowest successful cell (`tenferro-eager`, 0.346 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 12.6x faster than the slowest successful cell (`tenferro-eager`, 0.199 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 2.262 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`tenferro-eager`, 2.262 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 17.0x faster than the slowest successful cell (`tenferro-eager`, 0.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 63.1x faster than the slowest successful cell (`tenferro-eager`, 0.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 68.4x faster than the slowest successful cell (`tenferro-eager`, 0.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 24.7x faster than the slowest successful cell (`tenferro-eager`, 0.269 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 14.8x faster than the slowest successful cell (`tenferro-eager`, 0.269 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=8, shape=`4096x256`): `pytorch-cpu` is 67.0x faster than the slowest successful cell (`tenferro-eager`, 0.723 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=8, shape=`4096x256`): `tenferro-direct` is 42.6x faster than the slowest successful cell (`tenferro-eager`, 0.723 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 16.1x faster than the slowest successful cell (`tenferro-eager`, 0.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 19.1x faster than the slowest successful cell (`tenferro-eager`, 0.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=8, shape=`4096x256`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 1.675 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 48.6x faster than the slowest successful cell (`tenferro-eager`, 0.243 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=8, shape=`1024x64`): `tenferro-eager` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 50.9x faster than the slowest successful cell (`tenferro-eager`, 0.264 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=8, shape=`1024x64`): `tenferro-direct` is 21.0x faster than the slowest successful cell (`tenferro-eager`, 0.281 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 10.4x faster than the slowest successful cell (`tenferro-eager`, 12.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 21.8x faster than the slowest successful cell (`tenferro-eager`, 8.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 14.3x faster than the slowest successful cell (`tenferro-eager`, 8.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=8, shape=`4096x256`): `jax-cpu` is 21.2x faster than the slowest successful cell (`tenferro-eager`, 10.295 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=8, shape=`4096x256`): `pytorch-cpu` is 13.2x faster than the slowest successful cell (`tenferro-eager`, 10.295 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 43940.6x faster than the slowest successful cell (`julia-base`, 51.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 41718.9x faster than the slowest successful cell (`julia-base`, 51.398 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=8, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 35163.0x faster than the slowest successful cell (`julia-base`, 69.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 47331.4x faster than the slowest successful cell (`julia-base`, 59.259 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 44115.6x faster than the slowest successful cell (`julia-base`, 58.762 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=8, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 37196.5x faster than the slowest successful cell (`julia-base`, 75.658 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 80606.4x faster than the slowest successful cell (`julia-base`, 94.471 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 77617.1x faster than the slowest successful cell (`julia-base`, 94.848 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=8, shape=`4096x4096`): `tenferro-trace` is 72144.8x faster than the slowest successful cell (`julia-base`, 138.807 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=8, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 163.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 93.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=8, shape=`33554432 -> 8192x4096`): `julia-base` is 148.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 97.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 61.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=8, shape=`4194304 -> 2096128`): `julia-base` is 59.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 217.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 108.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=8, shape=`4096x4096`): `julia-base` is 173.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 13172.8 GB/s at 0.001 ms. The conservative physical lower bound is 0.168 ms, so the cell is 131.7x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 12222.3 GB/s at 0.001 ms. The conservative physical lower bound is 0.042 ms, so the cell is 30.6x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=8, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 8008.1 GB/s at 0.002 ms. The conservative physical lower bound is 0.021 ms, so the cell is 10.0x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 231011.6 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 2310.1x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 217885.9 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 544.7x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=8, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 136677.9 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 170.8x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 214405.3 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 2144.1x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 201528.1 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 503.8x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=8, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 131974.2 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 165.0x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 114520.2 GB/s at 0.001 ms. The conservative physical lower bound is 1.342 ms, so the cell is 1145.2x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 109834.5 GB/s at 0.001 ms. The conservative physical lower bound is 0.336 ms, so the cell is 274.6x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=8, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 69759.7 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 87.2x faster than the bound.

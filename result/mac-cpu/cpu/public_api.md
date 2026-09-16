# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260916_104531/run.yaml`
- Timestamp: `20260916_104531`

Latest run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260916_104531`.

- tenferro-rs commit: `bca2d54a5884586b1aeae8006caa9b8c0ba44a26`

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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260916_104531/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260916_104531/run_t4.yaml`
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
- `svd_full` remains in the table even when the selected tenferro-rs provider reports it as unsupported.
- Julia is column-major, like tenferro-rs, so the `julia-base`/`strided-jl` columns need no PyTorch/JAX-style layout reconstruction to keep the same logical fixture values.
- Julia warmup runs move JIT compilation outside the measured region, the same way PyTorch/JAX warmups do.
- Julia factorization rows materialize their factors inside the timed call (`cholesky` returns the factor matrix, `lu` returns P/L/U, `qr` returns Q/R). A Julia `Factorization` keeps its factors packed in LAPACK's working storage, so timing the compact object would compare strictly less work than the tenferro-rs and PyTorch columns, which return separate materialized tensors.
- Julia dense linalg rows run through Julia's own BLAS/LAPACK (libblastrampoline, by default OpenBLAS), recorded as `julia.blas_provider` in the run metadata. When that differs from the provider tenferro-rs and PyTorch link against (Accelerate on macOS), those rows partly compare BLAS implementations rather than framework overhead; read them together with the recorded providers.
- The Julia `lstsq` row uses `qr(a) \ rhs` rather than `a \ rhs`: the bare backslash runs a column-pivoted, rank-revealing QR (the LAPACK `gelsy` algorithm), while the PyTorch row selects the `gels` driver, so the unpivoted spelling is the like-for-like comparison.
- `julia-base` uses the natural Base/LinearAlgebra spelling and `strided-jl` the natural Strided.jl (`@strided`) spelling; each is populated only for rows where that spelling naturally applies (reductions and dense linalg have no natural Strided.jl spelling, so `strided-jl` covers elementwise/chain/transpose rows, the elementwise `cpu/output_reuse` `_into` rows, and the elementwise `cpu/complex` rows conj/mul/div/exp/log).
- Strided.jl (https://github.com/Jutho/Strided.jl) is prior art for tenferro-rs' strided-rs kernel layer; the `strided-jl` column credits that lineage directly in the report.

## Threads: 1 4

- CSV: `data/results/mac-cpu/cpu/public_api/20260916_104531/cpu_public_api_t1_20260916_104531.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260916_104531/cpu_public_api_t4_20260916_104531.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260916_104531/cpu_public_api_20260916_104531.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.467 ± 0.073 | - | 2.527 ± 0.075 | 2.841 ± 0.051 | 0.972 ± 0.016 | 2.932 ± 0.349 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.473 ± 0.078 | - | 2.527 ± 0.073 | 2.508 ± 0.084 | 0.764 ± 0.032 | 1.291 ± 0.509 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 11.522 ± 0.108 | - | 11.377 ± 0.570 | 4.588 ± 0.068 | 4.550 ± 0.128 | 4.726 ± 1.078 | 4.495 ± 0.083 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 2.811 ± 0.093 | - | 2.957 ± 0.072 | 1.616 ± 0.096 | 1.698 ± 0.160 | 4.574 ± 0.127 | 6.354 ± 7.186 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 4.525 ± 0.017 | - | 4.609 ± 0.066 | 8.610 ± 0.473 | 23.334 ± 0.455 | 18.820 ± 0.389 | 18.281 ± 0.054 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 1.589 ± 0.035 | - | 1.656 ± 0.102 | 2.860 ± 0.050 | 7.799 ± 0.074 | 19.298 ± 0.788 | 8.668 ± 9.032 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 6.084 ± 0.785 | - | 6.086 ± 0.276 | 5.688 ± 0.135 | 37.032 ± 0.100 | 40.692 ± 0.491 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 6.049 ± 0.992 | - | 6.086 ± 0.164 | 5.677 ± 0.073 | 9.679 ± 0.325 | 11.352 ± 0.533 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 6.333 ± 0.136 | - | unsupported | 5.857 ± 0.144 | 37.138 ± 0.071 | 40.434 ± 0.978 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 5.990 ± 0.091 | - | unsupported | 5.917 ± 0.162 | 10.136 ± 0.271 | 11.206 ± 0.359 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 5.909 ± 0.063 | - | 5.935 ± 0.044 | 5.927 ± 0.080 | 10.073 ± 0.057 | 5.196 ± 0.048 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 5.930 ± 0.060 | - | 5.953 ± 0.068 | 5.922 ± 0.017 | 10.062 ± 0.072 | 4.988 ± 0.167 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 20.839 ± 0.313 | - | 20.982 ± 0.127 | 20.332 ± 0.284 | 24.144 ± 0.492 | 25.846 ± 3.108 | 24.812 ± 0.101 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 5.481 ± 0.785 | - | 5.432 ± 0.098 | 5.963 ± 0.065 | 6.662 ± 0.164 | 25.737 ± 3.330 | 6.505 ± 3.533 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 24.300 ± 0.209 | - | 24.189 ± 0.278 | 25.705 ± 0.058 | 49.905 ± 0.124 | 35.546 ± 0.137 | 34.750 ± 0.332 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 6.661 ± 0.078 | - | 6.611 ± 0.135 | 7.094 ± 0.359 | 14.129 ± 0.447 | 35.511 ± 0.667 | 9.811 ± 2.300 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 4.113 ± 0.325 | - | 4.027 ± 0.065 | 4.259 ± 0.038 | 3.633 ± 0.117 | 3.715 ± 0.106 | 3.561 ± 0.112 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 1.380 ± 0.151 | - | 1.795 ± 0.093 | 1.461 ± 0.027 | 1.449 ± 0.168 | 3.744 ± 0.150 | 3.488 ± 8.145 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 4.425 ± 0.150 | - | 4.022 ± 0.084 | 3.683 ± 0.020 | 2.048 ± 0.045 | 5.718 ± 0.147 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 2.270 ± 0.115 | - | 1.431 ± 0.035 | 0.951 ± 0.016 | 0.751 ± 0.066 | 5.935 ± 0.142 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.737 ± 0.041 | - | 4.830 ± 0.018 | 5.045 ± 0.056 | 3.665 ± 0.030 | 6.172 ± 0.095 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.789 ± 0.042 | - | 4.847 ± 0.051 | 4.882 ± 0.030 | 3.633 ± 0.036 | 2.890 ± 0.329 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.311 ± 0.078 | - | 3.336 ± 0.110 | 3.136 ± 0.070 | 1.447 ± 0.056 | 3.698 ± 0.061 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.565 ± 0.130 | - | 3.577 ± 0.113 | 3.319 ± 0.127 | 1.361 ± 0.079 | 1.522 ± 0.092 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.916 ± 0.032 | - | 3.979 ± 0.050 | 3.874 ± 0.080 | 2.681 ± 0.065 | 4.455 ± 0.165 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.902 ± 0.043 | - | 4.045 ± 0.045 | 3.878 ± 0.037 | 2.764 ± 0.055 | 2.953 ± 0.314 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 5.822 ± 0.216 | - | 6.078 ± 0.125 | 5.660 ± 0.201 | 37.008 ± 0.184 | 40.380 ± 0.128 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 5.785 ± 0.263 | - | 5.940 ± 0.143 | 5.743 ± 0.136 | 9.974 ± 0.251 | 11.050 ± 0.212 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 5.043 ± 0.893 | - | unsupported | 5.140 ± 0.526 | 33.277 ± 0.169 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 5.421 ± 0.838 | - | unsupported | 4.649 ± 0.201 | 9.220 ± 0.251 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 4.532 ± 0.059 | - | 4.571 ± 0.040 | 4.430 ± 0.061 | 8.674 ± 0.751 | 4.538 ± 0.191 | 7.959 ± 0.088 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 1.594 ± 0.106 | - | 1.648 ± 0.114 | 1.579 ± 0.131 | 2.496 ± 0.050 | 4.490 ± 0.074 | 2.789 ± 2.858 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 6.815 ± 0.103 | - | 7.003 ± 0.248 | 6.754 ± 0.080 | 11.654 ± 0.103 | 6.747 ± 0.297 | 8.787 ± 0.218 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 2.574 ± 0.343 | - | 2.658 ± 0.165 | 2.649 ± 0.218 | 3.458 ± 0.080 | 6.929 ± 0.179 | 5.577 ± 11.435 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 16.768 ± 0.098 | - | 16.793 ± 0.102 | 18.013 ± 0.138 | 18.045 ± 0.069 | 20.966 ± 0.146 | 21.272 ± 0.262 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 4.978 ± 0.174 | - | 4.882 ± 0.091 | 5.490 ± 0.838 | 4.829 ± 0.198 | 21.004 ± 0.104 | 5.826 ± 0.298 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 2.802 ± 0.054 | - | 2.856 ± 0.036 | 2.250 ± 0.053 | 3.850 ± 0.066 | 2.259 ± 0.084 | 3.132 ± 0.262 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 1.601 ± 0.062 | - | 1.582 ± 0.151 | 0.924 ± 0.076 | 1.810 ± 0.105 | 2.300 ± 0.221 | 1.462 ± 3.635 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 5.700 ± 0.382 | - | 5.416 ± 0.268 | 8.601 ± 0.047 | 8.765 ± 0.019 | 5.834 ± 0.116 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 2.089 ± 0.168 | - | 2.083 ± 0.089 | 2.955 ± 0.094 | 2.906 ± 0.181 | 5.829 ± 0.124 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 16.451 ± 0.127 | - | 16.349 ± 0.421 | 17.069 ± 0.048 | 12.369 ± 0.568 | 20.435 ± 0.131 | 20.575 ± 0.107 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 4.300 ± 0.125 | - | 4.231 ± 0.151 | 4.650 ± 0.175 | 3.423 ± 0.261 | 20.513 ± 0.261 | 5.393 ± 0.359 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 6.847 ± 0.139 | - | 6.913 ± 0.208 | 6.695 ± 0.123 | 11.728 ± 0.084 | 6.576 ± 0.222 | 8.865 ± 0.404 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 2.554 ± 0.187 | - | 2.603 ± 0.157 | 2.518 ± 0.122 | 3.475 ± 0.093 | 6.894 ± 0.142 | 3.358 ± 2.357 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 15.957 ± 0.367 | - | 15.969 ± 0.801 | 12.291 ± 0.061 | 10.693 ± 0.708 | 16.956 ± 0.469 | 16.864 ± 1.599 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 4.095 ± 0.052 | - | 4.139 ± 0.141 | 4.439 ± 0.016 | 2.910 ± 0.341 | 16.735 ± 0.122 | 5.776 ± 0.886 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 8.274 ± 0.630 | - | 8.422 ± 0.198 | 14.746 ± 0.034 | 11.050 ± 0.081 | 12.725 ± 0.079 | 12.727 ± 0.084 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.173 ± 0.046 | - | 2.226 ± 0.098 | 4.019 ± 0.204 | 2.934 ± 0.123 | 12.737 ± 0.212 | 3.389 ± 0.959 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 15.365 ± 0.194 | - | 15.376 ± 0.047 | 20.898 ± 0.066 | 14.485 ± 0.228 | 22.570 ± 0.252 | 23.909 ± 0.328 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 4.129 ± 0.250 | - | 4.229 ± 0.439 | 5.560 ± 0.773 | 4.018 ± 0.277 | 22.667 ± 0.126 | 6.109 ± 0.437 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 10.239 ± 0.078 | - | 10.275 ± 0.068 | 11.028 ± 0.041 | 10.037 ± 0.076 | 12.282 ± 0.077 | 12.274 ± 0.084 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 2.776 ± 0.083 | - | 2.825 ± 0.076 | 2.919 ± 0.258 | 2.793 ± 0.352 | 12.258 ± 0.132 | 3.238 ± 0.624 |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 6.784 ± 0.260 | - | 6.990 ± 0.262 | 6.578 ± 0.057 | 11.541 ± 0.123 | 6.614 ± 0.065 | 8.690 ± 0.268 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 2.533 ± 0.170 | - | 2.544 ± 0.189 | 2.530 ± 0.094 | 3.466 ± 0.088 | 6.679 ± 0.138 | 3.342 ± 2.369 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 7.384 ± 0.482 | - | 7.141 ± 0.217 | 6.561 ± 0.029 | 11.582 ± 0.132 | 6.632 ± 0.132 | 8.513 ± 0.205 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 2.561 ± 0.140 | - | 2.631 ± 0.088 | 2.519 ± 0.071 | 3.470 ± 0.092 | 6.685 ± 0.154 | 3.336 ± 2.669 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 7.001 ± 0.128 | - | 6.987 ± 0.117 | 6.675 ± 0.027 | 11.583 ± 0.134 | 6.641 ± 0.137 | 8.583 ± 0.296 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 2.782 ± 0.118 | - | 2.813 ± 0.063 | 2.536 ± 0.143 | 3.568 ± 0.138 | 6.773 ± 0.263 | 3.638 ± 2.544 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 4.594 ± 0.159 | - | 4.605 ± 0.127 | 4.416 ± 0.062 | 8.757 ± 0.882 | 4.460 ± 0.100 | 7.931 ± 0.115 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 1.647 ± 0.089 | - | 1.628 ± 0.110 | 1.569 ± 0.116 | 2.510 ± 0.022 | 4.533 ± 0.095 | 2.732 ± 2.392 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 17.849 ± 0.329 | - | 17.775 ± 0.126 | 38.305 ± 0.100 | 17.732 ± 0.034 | 32.506 ± 1.015 | 31.604 ± 0.396 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 5.100 ± 0.721 | - | 4.994 ± 0.215 | 11.613 ± 0.180 | 5.065 ± 0.298 | 32.356 ± 2.142 | 8.658 ± 0.500 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 5.904 ± 0.177 | - | 5.955 ± 0.219 | 10.853 ± 0.382 | 1.098 ± 0.114 | 0.369 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 5.825 ± 0.096 | - | 5.784 ± 0.108 | 4.888 ± 0.313 | 0.459 ± 0.043 | 0.368 ± 0.027 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.259 ± 0.188 | - | 3.161 ± 0.053 | 3.978 ± 0.068 | 1.673 ± 0.079 | 1.953 ± 0.094 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.177 ± 0.048 | - | 3.199 ± 0.058 | 1.046 ± 0.026 | 0.591 ± 0.040 | 1.941 ± 0.028 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 3.419 ± 0.425 | - | 3.602 ± 0.284 | 3.240 ± 0.083 | 18.430 ± 0.403 | 3.139 ± 0.120 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.071 ± 0.098 | - | 6.101 ± 0.065 | 1.021 ± 0.079 | 5.432 ± 0.160 | 3.162 ± 0.118 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 3.538 ± 0.312 | - | 3.323 ± 0.130 | 2.984 ± 0.085 | 14.237 ± 0.175 | 3.120 ± 0.127 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 4.657 ± 0.438 | - | 4.521 ± 0.124 | 1.011 ± 0.105 | 4.600 ± 0.074 | 3.127 ± 0.115 | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 16.080 ± 0.518 | - | 15.970 ± 0.256 | 19.444 ± 0.082 | 15.034 ± 0.273 | 13.353 ± 2.737 | 13.425 ± 0.223 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 4.391 ± 0.514 | - | 4.259 ± 0.216 | 6.415 ± 0.142 | 4.468 ± 0.125 | 14.067 ± 1.674 | 4.056 ± 1.675 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 12.106 ± 0.247 | - | 12.089 ± 0.089 | 11.630 ± 0.179 | 24.323 ± 0.462 | 23.318 ± 0.287 | 23.657 ± 0.514 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 3.104 ± 0.042 | - | 3.133 ± 0.163 | 3.113 ± 0.053 | 6.110 ± 0.028 | 23.883 ± 0.197 | 6.476 ± 2.528 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 8.580 ± 0.451 | - | 8.299 ± 0.177 | 9.141 ± 0.109 | 14.175 ± 0.079 | 7.924 ± 0.108 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 3.375 ± 0.115 | - | 3.482 ± 0.162 | 3.634 ± 0.047 | 4.393 ± 0.327 | 8.026 ± 0.074 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 5.190 ± 0.054 | - | 5.209 ± 0.101 | 5.076 ± 0.045 | 9.994 ± 0.047 | 4.491 ± 0.137 | 11.974 ± 0.191 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 3.249 ± 0.075 | - | 3.086 ± 0.155 | 1.737 ± 0.015 | 3.339 ± 0.029 | 4.498 ± 0.081 | 3.117 ± 2.474 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 15.039 ± 0.223 | - | 15.371 ± 0.201 | 14.254 ± 0.057 | 11.318 ± 0.130 | 17.411 ± 0.399 | 17.412 ± 0.386 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 3.930 ± 0.129 | - | 3.931 ± 0.030 | 4.020 ± 0.238 | 3.185 ± 0.291 | 17.308 ± 0.215 | 4.546 ± 0.470 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 8.000 ± 0.104 | - | 7.973 ± 0.285 | 7.860 ± 0.184 | 8.747 ± 0.610 | 15.616 ± 0.154 | 15.799 ± 0.149 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 2.025 ± 0.041 | - | 2.031 ± 0.048 | 2.038 ± 0.050 | 2.516 ± 0.042 | 15.812 ± 0.218 | 4.211 ± 2.565 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 6.690 ± 0.076 | - | 6.752 ± 0.137 | 6.743 ± 0.095 | 11.709 ± 0.129 | 6.629 ± 0.041 | 8.528 ± 0.294 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 2.533 ± 0.135 | - | 2.592 ± 0.172 | 2.502 ± 0.094 | 3.526 ± 0.224 | 6.854 ± 0.094 | 3.438 ± 2.429 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 21.615 ± 0.703 | - | 21.628 ± 0.266 | 35.683 ± 0.075 | 8.464 ± 0.058 | 23.377 ± 0.446 | 23.678 ± 0.184 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 5.529 ± 0.282 | - | 6.460 ± 4.374 | 9.628 ± 0.184 | 2.331 ± 0.066 | 23.501 ± 0.171 | 6.324 ± 0.580 |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.269 ± 0.020 | - | 0.571 ± 0.019 | 0.227 ± 0.008 | 0.234 ± 0.004 | 0.280 ± 0.015 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.276 ± 0.012 | - | 0.431 ± 0.050 | 0.119 ± 0.006 | 0.236 ± 0.014 | 0.290 ± 0.011 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.238 ± 0.012 | - | 0.820 ± 0.055 | 0.271 ± 0.021 | 0.283 ± 0.007 | 0.275 ± 0.017 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.254 ± 0.005 | - | 0.606 ± 0.075 | 0.097 ± 0.006 | 0.128 ± 0.041 | 0.292 ± 0.014 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.392 ± 0.008 | - | unsupported | 0.422 ± 0.053 | 0.361 ± 0.025 | 0.400 ± 0.067 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.401 ± 0.033 | - | unsupported | 0.157 ± 0.011 | 0.309 ± 0.015 | 0.412 ± 0.081 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.428 ± 0.105 | - | 0.478 ± 0.136 | 0.163 ± 0.013 | 0.179 ± 0.014 | 0.300 ± 0.017 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.223 ± 0.047 | - | 0.292 ± 0.064 | 0.160 ± 0.007 | 0.084 ± 0.013 | 0.318 ± 0.063 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.350 ± 0.015 | - | 0.641 ± 0.053 | 0.407 ± 0.015 | 0.285 ± 0.001 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.346 ± 0.007 | - | 0.491 ± 0.037 | 0.172 ± 0.027 | 0.126 ± 0.016 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.499 ± 0.012 | - | 0.780 ± 0.033 | 0.290 ± 0.049 | 0.280 ± 0.008 | 0.573 ± 0.175 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.495 ± 0.016 | - | 0.690 ± 0.062 | 0.106 ± 0.017 | 0.117 ± 0.020 | 0.735 ± 0.240 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.469 ± 0.082 | - | 0.595 ± 0.055 | 0.244 ± 0.008 | 0.227 ± 0.012 | 0.242 ± 0.040 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.478 ± 0.098 | - | 0.540 ± 0.066 | 0.344 ± 0.026 | 0.216 ± 0.015 | 0.259 ± 0.092 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 0.659 ± 0.257 | - | 1.201 ± 0.191 | 0.436 ± 0.032 | 0.424 ± 0.028 | 0.733 ± 0.083 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.640 ± 0.085 | - | 0.968 ± 0.061 | 0.178 ± 0.048 | 0.280 ± 0.017 | 0.733 ± 0.040 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 4.716 ± 0.771 | - | 4.729 ± 0.120 | 15.522 ± 0.216 | 9.732 ± 0.357 | 26.447 ± 3.797 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 4.529 ± 0.648 | - | 4.711 ± 0.228 | 8.231 ± 0.577 | 6.368 ± 0.365 | 10.660 ± 12.055 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 4.421 ± 0.289 | - | 4.820 ± 0.117 | 4.071 ± 0.127 | 4.279 ± 0.123 | 13.709 ± 0.233 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 4.398 ± 0.184 | - | 4.652 ± 0.169 | 4.192 ± 0.189 | 3.929 ± 0.281 | 5.229 ± 0.326 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.448 ± 0.062 | - | 4.402 ± 0.040 | 4.486 ± 0.042 | 5.430 ± 0.075 | 4.080 ± 0.056 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.432 ± 0.055 | - | 4.450 ± 0.039 | 4.480 ± 0.069 | 5.477 ± 0.124 | 4.233 ± 0.170 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 4.659 ± 0.046 | - | 4.652 ± 0.041 | 4.651 ± 0.036 | 4.279 ± 0.083 | 4.420 ± 0.079 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 4.635 ± 0.046 | - | 4.661 ± 0.033 | 4.736 ± 0.129 | 4.454 ± 0.091 | 4.601 ± 0.078 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 15.103 ± 0.162 | - | 14.872 ± 0.372 | 15.234 ± 0.108 | 12.820 ± 0.067 | 10.991 ± 0.065 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 14.855 ± 0.141 | - | 15.035 ± 0.378 | 15.152 ± 0.402 | 12.639 ± 0.040 | 6.613 ± 0.185 | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 7.315 ± 0.132 | - | 6.830 ± 0.084 | 5.692 ± 0.073 | 5.670 ± 0.130 | 19.424 ± 0.228 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 7.328 ± 0.170 | - | 6.816 ± 0.057 | 5.473 ± 0.115 | 5.245 ± 0.077 | 6.644 ± 0.499 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 5.870 ± 0.054 | 5.577 ± 0.056 | 3.290 ± 0.049 | 13.821 ± 0.090 | 5.242 ± 0.052 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 5.792 ± 0.110 | 5.636 ± 0.110 | 3.194 ± 0.042 | 13.316 ± 0.110 | 3.151 ± 0.148 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 4.256 ± 0.178 | - | 4.523 ± 1.955 | 6.405 ± 0.243 | 6.877 ± 0.117 | 16.567 ± 3.508 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 4.232 ± 0.145 | - | 4.395 ± 0.149 | 5.726 ± 0.186 | 5.210 ± 0.304 | 7.295 ± 2.560 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.370 ± 0.016 | - | 0.385 ± 0.010 | 1.273 ± 0.023 | 2.705 ± 0.089 | 3.851 ± 0.140 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.571 ± 0.019 | - | 0.597 ± 0.044 | 1.279 ± 0.023 | 0.822 ± 0.036 | 4.029 ± 0.119 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.664 ± 0.036 | - | 6.632 ± 0.069 | 6.529 ± 0.054 | 7.428 ± 0.049 | 13.028 ± 0.120 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.637 ± 0.098 | - | 6.829 ± 0.106 | 6.610 ± 0.321 | 6.672 ± 0.122 | 9.658 ± 0.302 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 6.664 ± 0.085 | - | 6.618 ± 0.046 | 6.512 ± 0.055 | 7.411 ± 0.101 | 12.987 ± 0.198 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 6.622 ± 0.078 | - | 6.776 ± 0.063 | 6.579 ± 0.051 | 6.663 ± 0.076 | 9.765 ± 0.291 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 4.328 ± 0.290 | - | 4.876 ± 0.240 | 4.188 ± 0.209 | 4.286 ± 0.189 | 13.716 ± 0.196 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 4.269 ± 0.097 | - | 4.776 ± 0.120 | 4.203 ± 0.179 | 4.222 ± 0.197 | 5.164 ± 0.362 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | unsupported | unsupported | 18.774 ± 0.266 | 19.119 ± 0.183 | 44.867 ± 0.485 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | unsupported | unsupported | 19.168 ± 0.145 | 19.120 ± 0.113 | 30.069 ± 2.577 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 5.779 ± 0.269 | - | 8.213 ± 0.345 | 3.983 ± 0.065 | 25.904 ± 0.447 | 21.265 ± 0.170 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 5.517 ± 0.128 | - | 7.056 ± 0.231 | 3.884 ± 0.059 | 12.045 ± 1.063 | 7.400 ± 0.083 | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 23.564 ± 0.082 | - | unsupported | 6.643 ± 0.083 | - | 6.981 ± 0.802 | 8.584 ± 0.405 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 6.100 ± 0.202 | - | unsupported | 2.520 ± 0.060 | - | 6.650 ± 0.104 | 3.152 ± 0.274 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 8.362 ± 0.525 | - | unsupported | 4.583 ± 0.077 | - | 4.692 ± 0.290 | 4.632 ± 0.150 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 1.906 ± 0.111 | - | unsupported | 1.601 ± 0.063 | - | 4.479 ± 0.035 | 1.552 ± 0.268 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 3.929 ± 0.173 | - | unsupported | 4.401 ± 0.044 | - | 3.899 ± 0.114 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 1.550 ± 0.122 | - | unsupported | 1.616 ± 0.101 | - | 3.905 ± 0.076 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 20.029 ± 0.881 | - | unsupported | 6.541 ± 0.047 | - | 6.482 ± 0.121 | 8.635 ± 0.235 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 5.970 ± 0.525 | - | unsupported | 2.530 ± 0.109 | - | 6.566 ± 0.062 | 3.124 ± 0.184 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 6.845 ± 0.492 | - | unsupported | 5.050 ± 0.348 | - | 36.371 ± 0.077 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 4.737 ± 0.185 | - | unsupported | 5.669 ± 0.543 | - | 9.937 ± 0.326 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 5.273 ± 0.193 | - | unsupported | 5.435 ± 0.158 | - | 36.411 ± 0.516 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 5.923 ± 0.871 | - | unsupported | 5.215 ± 0.342 | - | 10.036 ± 0.114 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 23.581 ± 0.126 | - | unsupported | 6.567 ± 0.062 | - | 6.554 ± 0.043 | 8.477 ± 0.239 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 6.110 ± 0.108 | - | unsupported | 2.509 ± 0.052 | - | 6.644 ± 0.075 | 2.930 ± 0.326 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 11.542 ± 1.661 | - | unsupported | 4.390 ± 0.075 | - | 4.384 ± 0.046 | 8.106 ± 0.371 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 2.929 ± 0.197 | - | unsupported | 1.561 ± 0.030 | - | 4.383 ± 0.079 | 2.541 ± 0.543 |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 23.560 ± 0.099 | - | unsupported | 6.627 ± 0.047 | - | 6.527 ± 0.079 | 8.609 ± 0.308 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 6.065 ± 0.127 | - | unsupported | 2.534 ± 0.094 | - | 6.645 ± 0.051 | 2.995 ± 0.403 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 3.410 ± 0.063 | - | 2.037 ± 0.029 | 2.064 ± 0.024 | 1.929 ± 0.077 | 8.682 ± 0.084 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 1.922 ± 0.057 | - | 1.132 ± 0.033 | 0.979 ± 0.157 | 0.975 ± 0.133 | 8.715 ± 0.228 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 3.545 ± 0.075 | - | 7.985 ± 0.077 | 3.584 ± 0.074 | 4.522 ± 0.440 | 3.669 ± 0.084 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 1.374 ± 0.165 | - | 2.978 ± 0.124 | 1.344 ± 0.090 | 1.719 ± 0.042 | 3.696 ± 0.090 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 2.077 ± 0.102 | - | 2.067 ± 0.072 | 4.101 ± 0.060 | 6.140 ± 0.049 | 2.218 ± 0.173 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 2.034 ± 0.062 | - | 2.062 ± 0.107 | 2.029 ± 0.292 | 2.118 ± 0.223 | 2.208 ± 0.086 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 2.234 ± 0.042 | - | 6.676 ± 0.125 | 4.468 ± 0.792 | 3.534 ± 0.063 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 2.208 ± 0.065 | - | 3.884 ± 0.106 | 1.831 ± 0.026 | 1.301 ± 0.063 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 4.185 ± 0.372 | - | 4.455 ± 0.096 | 4.373 ± 0.037 | 3.665 ± 0.133 | 4.121 ± 0.614 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 3.901 ± 0.048 | - | 1.630 ± 0.087 | 1.588 ± 0.094 | 1.715 ± 0.076 | 4.042 ± 0.044 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 29.666 ± 0.726 | - | 23.894 ± 0.242 | 18.994 ± 0.099 | 13.402 ± 0.197 | 52.558 ± 5.353 | 31.531 ± 0.227 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 11.756 ± 0.825 | - | 7.386 ± 0.103 | 19.103 ± 1.336 | 6.159 ± 0.781 | 52.075 ± 1.444 | 22.429 ± 7.967 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 3.392 ± 0.079 | - | 5.504 ± 0.117 | 3.532 ± 0.121 | 2.250 ± 0.054 | 4.248 ± 0.165 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 3.411 ± 0.226 | - | 4.796 ± 0.142 | 1.302 ± 0.022 | 0.868 ± 0.083 | 4.297 ± 0.112 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 3.205 ± 0.223 | - | 5.674 ± 0.116 | 3.247 ± 0.125 | 2.228 ± 0.030 | 3.993 ± 0.133 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 3.334 ± 0.284 | - | 4.860 ± 0.208 | 1.191 ± 0.026 | 0.836 ± 0.068 | 4.113 ± 0.215 | - |
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
| broadcast_in_dim_view | 1 | PyTorch Python (ms) | 16 | 0.005583 | 348.94 |
| broadcast_in_dim_view | 1 | tenferro-rs direct API (ms) | 16 | 0.001959 | 122.44 |
| broadcast_in_dim_view | 4 | PyTorch Python (ms) | 16 | 0.005958 | 372.38 |
| broadcast_in_dim_view | 4 | tenferro-rs direct API (ms) | 16 | 0.001875 | 117.19 |
| reshape_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| reshape_view | 1 | PyTorch Python (ms) | 16 | 0.005791 | 361.94 |
| reshape_view | 1 | tenferro-rs direct API (ms) | 16 | 0.003750 | 234.38 |
| reshape_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| reshape_view | 4 | PyTorch Python (ms) | 16 | 0.005375 | 335.94 |
| reshape_view | 4 | tenferro-rs direct API (ms) | 16 | 0.003708 | 231.75 |
| slice_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| slice_view | 1 | PyTorch Python (ms) | 16 | 0.004667 | 291.69 |
| slice_view | 1 | tenferro-rs direct API (ms) | 16 | 0.003250 | 203.12 |
| slice_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| slice_view | 4 | PyTorch Python (ms) | 16 | 0.004875 | 304.69 |
| slice_view | 4 | tenferro-rs direct API (ms) | 16 | 0.003375 | 210.94 |
| transpose_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| transpose_view | 1 | PyTorch Python (ms) | 16 | 0.005250 | 328.12 |
| transpose_view | 1 | tenferro-rs direct API (ms) | 16 | 0.002958 | 184.88 |
| transpose_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| transpose_view | 4 | PyTorch Python (ms) | 16 | 0.004709 | 294.31 |
| transpose_view | 4 | tenferro-rs direct API (ms) | 16 | 0.003042 | 190.12 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-direct` is 12.1x faster than the slowest successful cell (`julia-base`, 19.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`julia-base`, 19.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 29.4x faster than the slowest successful cell (`pytorch-cpu`, 10.853 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-direct`, 5.825 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 15.8x faster than the slowest successful cell (`tenferro-direct`, 5.825 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=4, shape=`8388608`): `jax-cpu` is 10.1x faster than the slowest successful cell (`julia-base`, 23.501 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`julia-base`, 3.851 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 120.7x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 112.0x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 97.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 61.0x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 109.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 98.0x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

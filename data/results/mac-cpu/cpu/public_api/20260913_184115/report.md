# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260913_184115/run.yaml`
- Timestamp: `20260913_184115`

Latest run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260913_184115`.

- tenferro-rs commit: `28dfc7e383a653e1461d38b73ced3d563b4efdad`

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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260913_184115/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260913_184115/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/public_api/20260913_184115/cpu_public_api_t1_20260913_184115.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260913_184115/cpu_public_api_t4_20260913_184115.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260913_184115/cpu_public_api_20260913_184115.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.425 ± 0.036 | - | 2.492 ± 0.052 | 2.757 ± 0.050 | 0.964 ± 0.024 | 2.834 ± 0.294 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.389 ± 0.066 | - | 2.406 ± 0.041 | 2.482 ± 0.040 | 0.764 ± 0.016 | 1.197 ± 0.388 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 8.086 ± 0.446 | - | 7.797 ± 0.516 | 4.545 ± 0.042 | 4.535 ± 0.109 | 4.575 ± 0.249 | 4.409 ± 0.069 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 2.960 ± 0.152 | - | 2.934 ± 0.048 | 1.587 ± 0.043 | 1.582 ± 0.093 | 4.635 ± 0.272 | 6.677 ± 6.888 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 4.488 ± 0.026 | - | 4.618 ± 0.087 | 8.131 ± 0.438 | 23.100 ± 0.038 | 18.682 ± 0.079 | 18.190 ± 0.044 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 1.569 ± 0.036 | - | 1.601 ± 0.053 | 2.818 ± 0.047 | 7.727 ± 0.087 | 19.249 ± 0.691 | 7.271 ± 7.390 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 6.021 ± 0.616 | - | 5.896 ± 0.167 | 5.516 ± 0.213 | 36.660 ± 0.095 | 40.084 ± 0.515 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 5.733 ± 1.046 | - | 5.828 ± 0.227 | 5.612 ± 0.190 | 11.544 ± 0.547 | 10.654 ± 0.269 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 5.922 ± 0.150 | - | unsupported | 5.688 ± 0.204 | 36.765 ± 0.067 | 40.003 ± 0.683 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 5.772 ± 0.160 | - | unsupported | 5.643 ± 0.186 | 10.084 ± 0.137 | 11.002 ± 0.488 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 5.907 ± 0.042 | - | 5.794 ± 0.031 | 5.840 ± 0.070 | 9.953 ± 0.037 | 5.148 ± 0.029 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 5.896 ± 0.053 | - | 5.898 ± 0.028 | 5.884 ± 0.078 | 9.962 ± 0.069 | 4.916 ± 0.058 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 20.401 ± 0.330 | - | 20.469 ± 0.390 | 19.923 ± 0.264 | 23.977 ± 0.026 | 24.764 ± 2.996 | 24.605 ± 0.091 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 5.488 ± 0.827 | - | 5.325 ± 0.031 | 5.973 ± 0.037 | 6.435 ± 0.228 | 24.797 ± 3.640 | 6.656 ± 2.260 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 23.685 ± 0.088 | - | 23.655 ± 0.060 | 25.670 ± 0.214 | 49.640 ± 0.058 | 35.355 ± 0.108 | 34.506 ± 0.153 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 6.506 ± 0.222 | - | 6.561 ± 0.173 | 8.356 ± 0.574 | 13.810 ± 0.483 | 35.357 ± 0.072 | 10.354 ± 1.441 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 4.086 ± 0.274 | - | 4.055 ± 0.037 | 4.221 ± 0.029 | 3.605 ± 0.111 | 3.722 ± 0.218 | 3.467 ± 0.029 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 1.364 ± 0.081 | - | 1.388 ± 0.055 | 1.462 ± 0.051 | 1.376 ± 0.124 | 3.777 ± 0.215 | 3.984 ± 8.588 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 4.534 ± 0.458 | - | 3.798 ± 0.472 | 3.652 ± 0.012 | 1.970 ± 0.088 | 5.695 ± 0.317 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 2.194 ± 0.101 | - | 1.403 ± 0.022 | 0.929 ± 0.006 | 0.758 ± 0.052 | 5.894 ± 0.019 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.725 ± 0.042 | - | 4.789 ± 0.013 | 5.019 ± 0.027 | 3.629 ± 0.073 | 5.992 ± 0.094 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.773 ± 0.043 | - | 4.790 ± 0.025 | 4.840 ± 0.030 | 3.633 ± 0.031 | 2.801 ± 0.166 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.243 ± 0.044 | - | 3.303 ± 0.065 | 3.153 ± 0.051 | 1.429 ± 0.026 | 3.649 ± 0.041 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.475 ± 0.096 | - | 3.509 ± 0.091 | 3.300 ± 0.114 | 1.323 ± 0.029 | 1.504 ± 0.042 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.956 ± 0.029 | - | 3.880 ± 0.040 | 3.907 ± 0.045 | 2.721 ± 0.074 | 4.454 ± 0.048 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.885 ± 0.039 | - | 3.980 ± 0.018 | 3.904 ± 0.028 | 2.721 ± 0.043 | 2.900 ± 0.099 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 5.564 ± 0.215 | - | 5.801 ± 0.194 | 5.597 ± 0.164 | 36.653 ± 0.080 | 39.748 ± 0.042 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 5.657 ± 0.258 | - | 5.793 ± 0.165 | 5.518 ± 0.162 | 9.967 ± 0.301 | 10.731 ± 0.299 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 4.931 ± 0.984 | - | unsupported | 4.497 ± 0.084 | 32.823 ± 0.121 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 4.872 ± 1.107 | - | unsupported | 4.497 ± 0.058 | 8.877 ± 0.292 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 4.352 ± 0.024 | - | 4.346 ± 0.055 | 4.346 ± 0.047 | 8.663 ± 1.497 | 4.373 ± 0.153 | 7.902 ± 0.095 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 1.564 ± 0.027 | - | 1.563 ± 0.025 | 1.542 ± 0.018 | 2.496 ± 0.042 | 4.427 ± 0.116 | 3.144 ± 2.233 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 6.540 ± 0.199 | - | 6.610 ± 0.213 | 6.614 ± 0.204 | 11.644 ± 0.299 | 6.770 ± 0.430 | 8.740 ± 0.543 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 2.586 ± 0.299 | - | 2.591 ± 0.195 | 4.720 ± 0.157 | 3.481 ± 0.162 | 6.772 ± 0.306 | 5.572 ± 9.183 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 16.926 ± 0.660 | - | 16.836 ± 0.466 | 17.914 ± 0.061 | 17.881 ± 0.051 | 20.889 ± 0.123 | 21.189 ± 0.339 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 4.774 ± 0.114 | - | 4.845 ± 0.114 | 5.088 ± 0.229 | 4.797 ± 0.127 | 20.900 ± 0.162 | 5.774 ± 0.418 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 2.784 ± 0.020 | - | 2.896 ± 0.106 | 2.228 ± 0.025 | 3.844 ± 0.066 | 2.251 ± 0.101 | 3.102 ± 0.088 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 1.563 ± 0.054 | - | 1.565 ± 0.011 | 1.012 ± 0.034 | 1.775 ± 0.023 | 2.188 ± 0.128 | 1.461 ± 3.477 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 5.156 ± 0.023 | - | 5.157 ± 0.026 | 8.550 ± 0.132 | 8.682 ± 0.043 | 5.792 ± 0.075 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 2.067 ± 0.038 | - | 2.082 ± 0.028 | 2.920 ± 0.046 | 2.771 ± 0.030 | 5.830 ± 0.091 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 15.678 ± 1.521 | - | 15.920 ± 0.743 | 16.958 ± 0.040 | 12.439 ± 0.229 | 20.409 ± 0.195 | 20.579 ± 0.161 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 4.116 ± 0.135 | - | 4.112 ± 0.017 | 5.981 ± 0.131 | 3.323 ± 0.114 | 20.288 ± 0.156 | 5.262 ± 0.207 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 6.479 ± 0.123 | - | 6.604 ± 0.134 | 6.519 ± 0.195 | 11.586 ± 0.193 | 6.613 ± 0.313 | 8.540 ± 0.361 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 2.449 ± 0.121 | - | 2.476 ± 0.112 | 4.641 ± 0.028 | 3.518 ± 0.075 | 6.606 ± 0.145 | 4.613 ± 9.140 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 15.770 ± 1.100 | - | 16.771 ± 0.347 | 12.206 ± 0.025 | 10.343 ± 0.033 | 17.696 ± 4.080 | 16.955 ± 0.965 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 4.132 ± 0.039 | - | 4.123 ± 0.028 | 4.410 ± 0.063 | 2.646 ± 0.307 | 16.669 ± 0.476 | 5.807 ± 0.753 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 8.629 ± 0.089 | - | 8.653 ± 0.092 | 14.646 ± 0.041 | 10.938 ± 0.024 | 12.635 ± 0.054 | 12.709 ± 0.082 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.153 ± 0.025 | - | 2.174 ± 0.017 | 3.869 ± 0.186 | 2.898 ± 0.096 | 12.716 ± 0.581 | 3.230 ± 0.688 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 15.246 ± 0.052 | - | 15.214 ± 0.120 | 20.718 ± 0.028 | 14.121 ± 0.044 | 22.443 ± 0.101 | 22.902 ± 0.452 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 4.071 ± 0.116 | - | 4.073 ± 0.148 | 6.949 ± 0.033 | 3.860 ± 0.276 | 22.557 ± 0.164 | 6.049 ± 0.245 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 10.222 ± 0.038 | - | 10.197 ± 0.242 | 10.935 ± 0.020 | 9.924 ± 0.034 | 12.199 ± 0.065 | 12.194 ± 0.068 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 2.632 ± 0.096 | - | 2.735 ± 0.020 | 2.904 ± 0.041 | 2.665 ± 0.103 | 12.206 ± 0.092 | 3.106 ± 0.458 |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 6.521 ± 0.026 | - | 6.542 ± 0.027 | 6.511 ± 0.099 | 11.328 ± 0.178 | 6.436 ± 0.174 | 8.427 ± 0.377 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 2.516 ± 0.084 | - | 2.478 ± 0.124 | 2.499 ± 0.035 | 3.466 ± 0.176 | 6.570 ± 0.090 | 3.169 ± 2.234 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 6.546 ± 0.024 | - | 6.560 ± 0.049 | 6.467 ± 0.079 | 11.409 ± 0.175 | 6.475 ± 0.138 | 8.565 ± 0.267 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 2.543 ± 0.043 | - | 2.544 ± 0.026 | 2.493 ± 0.024 | 3.427 ± 0.136 | 6.595 ± 0.097 | 3.191 ± 2.318 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 6.633 ± 0.061 | - | 6.671 ± 0.233 | 6.672 ± 0.076 | 11.383 ± 0.161 | 6.532 ± 0.122 | 8.645 ± 0.568 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 2.799 ± 0.043 | - | 2.859 ± 0.076 | 4.643 ± 0.057 | 3.505 ± 0.120 | 6.647 ± 0.073 | 3.248 ± 8.875 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 4.391 ± 0.058 | - | 4.381 ± 0.088 | 4.334 ± 0.085 | 8.740 ± 0.900 | 4.400 ± 0.079 | 7.927 ± 0.234 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 1.574 ± 0.031 | - | 1.577 ± 0.026 | 1.541 ± 0.020 | 2.512 ± 0.031 | 4.442 ± 0.107 | 3.262 ± 2.754 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 17.614 ± 0.045 | - | 17.647 ± 0.085 | 38.085 ± 0.055 | 17.599 ± 0.032 | 31.874 ± 0.351 | 30.975 ± 0.187 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 4.945 ± 0.692 | - | 4.954 ± 0.222 | 11.516 ± 1.840 | 4.930 ± 0.123 | 31.527 ± 1.151 | 8.554 ± 0.435 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 5.650 ± 0.234 | - | 5.544 ± 0.092 | 9.434 ± 0.143 | 1.027 ± 0.090 | 0.371 ± 0.012 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 5.633 ± 0.092 | - | 5.636 ± 0.160 | 4.607 ± 0.059 | 0.463 ± 0.070 | 0.373 ± 0.007 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.178 ± 0.094 | - | 3.158 ± 0.130 | 3.850 ± 0.077 | 1.667 ± 0.120 | 1.914 ± 0.016 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.159 ± 0.106 | - | 3.215 ± 0.067 | 1.045 ± 0.008 | 0.686 ± 0.046 | 1.918 ± 0.042 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 3.289 ± 0.132 | - | 3.349 ± 0.087 | 3.171 ± 0.028 | 17.587 ± 0.420 | 3.058 ± 0.067 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 5.933 ± 0.020 | - | 6.109 ± 0.110 | 0.981 ± 0.024 | 5.330 ± 0.029 | 3.049 ± 0.050 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 3.325 ± 0.188 | - | 3.215 ± 0.281 | 2.905 ± 0.140 | 13.969 ± 0.220 | 3.016 ± 0.108 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 4.648 ± 0.424 | - | 4.428 ± 0.198 | 0.988 ± 0.020 | 4.545 ± 0.026 | 3.046 ± 0.094 | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 15.835 ± 0.639 | - | 15.822 ± 0.728 | 19.297 ± 0.082 | 15.327 ± 0.091 | 14.220 ± 3.087 | 13.931 ± 1.237 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 4.485 ± 0.615 | - | 4.272 ± 0.026 | 5.188 ± 0.139 | 4.424 ± 0.068 | 13.995 ± 1.291 | 3.905 ± 2.061 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 11.387 ± 0.102 | - | 11.362 ± 0.050 | 11.385 ± 0.050 | 22.869 ± 0.367 | 24.085 ± 0.132 | 23.383 ± 0.552 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 3.073 ± 0.035 | - | 3.080 ± 0.010 | 3.151 ± 0.079 | 6.069 ± 0.019 | 23.347 ± 0.488 | 6.300 ± 2.496 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 7.785 ± 0.037 | - | 7.789 ± 0.012 | 9.081 ± 0.160 | 14.074 ± 0.137 | 7.911 ± 0.132 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 3.311 ± 0.066 | - | 3.380 ± 0.051 | 3.712 ± 0.165 | 6.007 ± 0.660 | 7.851 ± 0.030 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 5.067 ± 0.030 | - | 5.074 ± 0.017 | 5.061 ± 0.086 | 9.868 ± 0.017 | 4.436 ± 0.085 | 11.745 ± 0.126 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 3.173 ± 0.088 | - | 3.174 ± 0.055 | 1.709 ± 0.016 | 3.306 ± 0.043 | 4.466 ± 0.138 | 3.359 ± 2.240 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 14.294 ± 0.533 | - | 15.043 ± 0.336 | 14.156 ± 0.029 | 11.276 ± 0.032 | 17.281 ± 0.317 | 17.509 ± 0.268 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 3.804 ± 0.166 | - | 3.857 ± 0.125 | 5.023 ± 0.016 | 3.131 ± 0.377 | 17.325 ± 0.149 | 4.484 ± 0.266 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 7.617 ± 0.256 | - | 7.502 ± 0.157 | 7.565 ± 0.226 | 8.363 ± 0.832 | 15.357 ± 0.278 | 15.896 ± 0.159 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 2.026 ± 0.045 | - | 2.038 ± 0.030 | 2.086 ± 0.108 | 2.481 ± 0.019 | 15.581 ± 0.079 | 4.339 ± 2.554 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 6.459 ± 0.135 | - | 6.562 ± 0.203 | 6.640 ± 0.232 | 11.381 ± 0.157 | 6.542 ± 0.113 | 8.558 ± 0.414 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 2.436 ± 0.128 | - | 2.560 ± 0.146 | 4.630 ± 0.081 | 3.496 ± 0.201 | 6.584 ± 0.047 | 3.190 ± 2.572 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 21.193 ± 0.047 | - | 21.227 ± 0.030 | 35.470 ± 0.063 | 8.404 ± 0.028 | 23.380 ± 0.211 | 23.863 ± 0.314 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 5.423 ± 0.106 | - | 5.433 ± 0.163 | 12.193 ± 0.133 | 2.258 ± 0.009 | 23.384 ± 0.098 | 5.934 ± 0.215 |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.264 ± 0.012 | - | 0.531 ± 0.015 | 0.215 ± 0.010 | 0.223 ± 0.005 | 0.288 ± 0.012 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.275 ± 0.007 | - | 0.410 ± 0.052 | 0.115 ± 0.006 | 0.239 ± 0.009 | 0.288 ± 0.008 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.238 ± 0.003 | - | 0.780 ± 0.019 | 0.275 ± 0.020 | 0.262 ± 0.005 | 0.280 ± 0.013 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.252 ± 0.005 | - | 0.589 ± 0.047 | 0.099 ± 0.015 | 0.113 ± 0.004 | 0.292 ± 0.003 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.376 ± 0.030 | - | unsupported | 0.414 ± 0.008 | 0.361 ± 0.012 | 0.386 ± 0.053 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.388 ± 0.004 | - | unsupported | 0.176 ± 0.014 | 0.331 ± 0.022 | 0.406 ± 0.043 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.343 ± 0.164 | - | 0.435 ± 0.111 | 0.167 ± 0.002 | 0.183 ± 0.010 | 0.284 ± 0.037 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.247 ± 0.109 | - | 0.314 ± 0.116 | 0.162 ± 0.010 | 0.077 ± 0.013 | 0.301 ± 0.043 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.334 ± 0.012 | - | 0.600 ± 0.021 | 0.398 ± 0.004 | 0.284 ± 0.004 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.352 ± 0.011 | - | 0.465 ± 0.012 | 0.178 ± 0.008 | 0.139 ± 0.010 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.704 ± 0.212 | - | 0.769 ± 0.049 | 0.326 ± 0.059 | 0.276 ± 0.007 | 0.684 ± 0.170 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.484 ± 0.008 | - | 0.733 ± 0.200 | 0.109 ± 0.021 | 0.109 ± 0.010 | 0.610 ± 0.083 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.462 ± 0.037 | - | 0.574 ± 0.024 | 0.243 ± 0.005 | 0.226 ± 0.004 | 0.266 ± 0.048 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.479 ± 0.074 | - | 0.529 ± 0.054 | 0.348 ± 0.018 | 0.219 ± 0.004 | 0.251 ± 0.100 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 0.628 ± 0.102 | - | 1.152 ± 0.066 | 0.435 ± 0.031 | 0.425 ± 0.044 | 0.713 ± 0.077 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.646 ± 0.321 | - | 0.932 ± 0.040 | 0.148 ± 0.023 | 0.291 ± 0.030 | 0.679 ± 0.236 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 4.479 ± 0.865 | - | 6.791 ± 2.329 | 14.944 ± 0.107 | 9.088 ± 0.220 | 26.087 ± 3.752 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 4.305 ± 0.715 | - | 4.639 ± 0.121 | 7.993 ± 0.190 | 5.958 ± 0.114 | 10.418 ± 11.791 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 4.185 ± 0.048 | - | 4.498 ± 0.175 | 4.037 ± 0.069 | 4.126 ± 0.043 | 13.613 ± 0.479 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 4.192 ± 0.224 | - | 4.294 ± 0.056 | 4.014 ± 0.030 | 3.831 ± 0.031 | 5.135 ± 0.516 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.416 ± 0.057 | - | 4.396 ± 0.021 | 4.437 ± 0.086 | 5.363 ± 0.103 | 4.015 ± 0.066 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.428 ± 0.064 | - | 4.442 ± 0.031 | 4.437 ± 0.054 | 5.360 ± 0.081 | 4.153 ± 0.055 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 4.612 ± 0.034 | - | 4.607 ± 0.025 | 4.629 ± 0.032 | 4.274 ± 0.068 | 4.402 ± 0.148 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 4.607 ± 0.037 | - | 4.624 ± 0.029 | 4.692 ± 0.099 | 4.393 ± 0.085 | 4.512 ± 0.065 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 13.982 ± 0.292 | - | 14.002 ± 1.061 | 15.100 ± 0.084 | 12.588 ± 0.089 | 10.841 ± 0.117 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 14.739 ± 0.085 | - | 15.043 ± 0.083 | 15.104 ± 0.091 | 12.480 ± 0.039 | 6.508 ± 0.131 | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 7.065 ± 0.047 | - | 6.678 ± 0.073 | 5.532 ± 0.029 | 5.516 ± 0.028 | 19.169 ± 0.153 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 7.078 ± 0.028 | - | 6.791 ± 0.214 | 5.308 ± 0.019 | 5.094 ± 0.027 | 6.278 ± 0.411 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 5.777 ± 0.030 | 5.509 ± 0.059 | 3.231 ± 0.022 | 13.594 ± 0.066 | 5.203 ± 0.066 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 5.712 ± 0.038 | 5.552 ± 0.014 | 3.165 ± 0.039 | 13.071 ± 0.038 | 3.125 ± 0.062 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 4.061 ± 0.030 | - | 4.258 ± 0.047 | 6.059 ± 0.196 | 6.698 ± 0.011 | 15.996 ± 3.213 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 4.058 ± 0.048 | - | 4.256 ± 0.060 | 5.447 ± 0.168 | 5.007 ± 0.103 | 7.435 ± 2.199 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.369 ± 0.005 | - | 0.372 ± 0.032 | 1.226 ± 0.038 | 2.606 ± 0.123 | 3.683 ± 0.064 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.555 ± 0.035 | - | 0.588 ± 0.057 | 1.270 ± 0.046 | 0.886 ± 0.173 | 4.012 ± 0.021 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.576 ± 0.031 | - | 6.502 ± 0.029 | 6.413 ± 0.021 | 7.309 ± 0.076 | 12.842 ± 0.129 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.544 ± 0.028 | - | 6.743 ± 0.051 | 6.479 ± 0.014 | 6.557 ± 0.078 | 9.422 ± 0.089 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 6.550 ± 0.015 | - | 6.482 ± 0.039 | 6.398 ± 0.026 | 7.291 ± 0.027 | 12.821 ± 0.163 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 6.534 ± 0.028 | - | 6.691 ± 0.033 | 6.478 ± 0.046 | 6.527 ± 0.084 | 9.393 ± 0.160 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 4.172 ± 0.036 | - | 4.640 ± 0.049 | 4.013 ± 0.033 | 4.175 ± 0.288 | 13.551 ± 0.142 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 4.142 ± 0.046 | - | 4.470 ± 0.066 | 3.989 ± 0.043 | 3.808 ± 0.073 | 5.069 ± 0.282 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | unsupported | unsupported | 18.477 ± 0.073 | 18.869 ± 0.109 | 44.235 ± 0.443 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | unsupported | unsupported | 18.864 ± 0.163 | 18.689 ± 0.187 | 29.217 ± 0.245 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 5.411 ± 0.607 | - | 7.861 ± 0.403 | 3.918 ± 0.025 | 25.179 ± 0.230 | 20.911 ± 0.157 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 5.413 ± 0.089 | - | 7.069 ± 0.099 | 3.779 ± 0.038 | 11.962 ± 1.398 | 7.280 ± 0.110 | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 17.373 ± 0.038 | - | unsupported | 6.684 ± 0.196 | - | 6.433 ± 0.149 | 8.443 ± 0.484 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 6.051 ± 0.084 | - | unsupported | 2.511 ± 0.041 | - | 6.709 ± 0.072 | 3.070 ± 0.166 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 8.131 ± 0.078 | - | unsupported | 4.522 ± 0.030 | - | 4.359 ± 0.041 | 4.340 ± 0.039 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 1.890 ± 0.061 | - | unsupported | 1.567 ± 0.041 | - | 4.377 ± 0.042 | 1.558 ± 0.241 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 3.846 ± 0.039 | - | unsupported | 4.337 ± 0.072 | - | 3.833 ± 0.054 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 1.649 ± 0.077 | - | unsupported | 1.554 ± 0.013 | - | 3.850 ± 0.049 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 18.262 ± 0.101 | - | unsupported | 6.460 ± 0.074 | - | 6.392 ± 0.107 | 8.442 ± 0.507 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 5.945 ± 1.299 | - | unsupported | 2.494 ± 0.039 | - | 6.500 ± 0.038 | 3.088 ± 0.131 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 4.470 ± 0.029 | - | unsupported | 4.457 ± 0.046 | - | 36.044 ± 0.063 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 4.462 ± 0.029 | - | unsupported | 4.436 ± 0.029 | - | 9.625 ± 0.126 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 4.974 ± 0.060 | - | unsupported | 4.976 ± 0.052 | - | 35.948 ± 0.047 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 4.975 ± 0.046 | - | unsupported | 4.942 ± 0.052 | - | 9.618 ± 0.144 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 17.367 ± 0.036 | - | unsupported | 6.480 ± 0.073 | - | 6.431 ± 0.100 | 8.417 ± 0.310 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 5.947 ± 0.028 | - | unsupported | 2.647 ± 0.162 | - | 6.500 ± 0.030 | 3.135 ± 0.245 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 11.513 ± 0.445 | - | unsupported | 4.348 ± 0.043 | - | 4.329 ± 0.038 | 7.814 ± 0.178 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 2.757 ± 0.077 | - | unsupported | 1.561 ± 0.022 | - | 4.349 ± 0.027 | 2.437 ± 0.207 |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 17.517 ± 0.189 | - | unsupported | 6.530 ± 0.065 | - | 6.516 ± 0.190 | 8.510 ± 0.496 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 6.063 ± 0.112 | - | unsupported | 2.641 ± 0.053 | - | 6.501 ± 0.045 | 3.037 ± 0.315 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 2.035 ± 1.373 | - | 2.020 ± 0.067 | 2.059 ± 0.020 | 1.888 ± 0.030 | 8.442 ± 0.307 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 1.861 ± 0.060 | - | 1.123 ± 0.014 | 1.080 ± 0.176 | 1.093 ± 0.017 | 8.569 ± 0.034 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 3.494 ± 0.056 | - | 7.832 ± 0.148 | 3.545 ± 0.035 | 4.589 ± 0.370 | 3.630 ± 0.064 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 1.383 ± 0.112 | - | 2.914 ± 0.066 | 1.327 ± 0.058 | 1.418 ± 0.140 | 3.639 ± 0.043 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 2.029 ± 0.057 | - | 1.987 ± 0.074 | 4.090 ± 0.083 | 6.263 ± 0.187 | 2.016 ± 0.070 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 2.060 ± 0.067 | - | 2.050 ± 0.046 | 2.195 ± 0.086 | 2.244 ± 0.156 | 2.111 ± 0.121 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 2.113 ± 0.059 | - | 6.567 ± 0.115 | 5.026 ± 1.357 | 3.511 ± 0.068 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 2.134 ± 0.067 | - | 3.816 ± 0.131 | 1.817 ± 0.036 | 1.362 ± 0.031 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 3.911 ± 0.066 | - | 4.350 ± 0.134 | 4.351 ± 0.063 | 3.628 ± 0.100 | 3.905 ± 0.067 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 3.972 ± 0.088 | - | 1.625 ± 0.094 | 1.541 ± 0.026 | 1.683 ± 0.047 | 3.930 ± 0.096 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 28.323 ± 0.239 | - | 23.386 ± 0.162 | 19.238 ± 0.192 | 12.946 ± 0.362 | 50.989 ± 4.119 | 30.854 ± 0.076 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 11.940 ± 1.391 | - | 7.277 ± 0.210 | 19.273 ± 0.258 | 5.253 ± 0.709 | 51.136 ± 1.749 | 22.437 ± 7.867 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 3.296 ± 0.120 | - | 5.576 ± 0.174 | 3.433 ± 0.124 | 2.218 ± 0.026 | 4.238 ± 0.110 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 3.313 ± 0.082 | - | 4.745 ± 0.146 | 1.347 ± 0.030 | 0.850 ± 0.076 | 4.304 ± 0.124 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 3.144 ± 0.097 | - | 5.455 ± 0.085 | 3.179 ± 0.047 | 2.206 ± 0.025 | 3.949 ± 0.163 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 3.118 ± 0.101 | - | 4.418 ± 0.082 | 1.181 ± 0.021 | 0.822 ± 0.048 | 4.211 ± 0.272 | - |
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
| broadcast_in_dim_view | 1 | PyTorch Python (ms) | 16 | 0.005542 | 346.38 |
| broadcast_in_dim_view | 1 | tenferro-rs direct API (ms) | 16 | 0.001875 | 117.19 |
| broadcast_in_dim_view | 4 | PyTorch Python (ms) | 16 | 0.007125 | 445.31 |
| broadcast_in_dim_view | 4 | tenferro-rs direct API (ms) | 16 | 0.001750 | 109.38 |
| reshape_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| reshape_view | 1 | PyTorch Python (ms) | 16 | 0.005334 | 333.38 |
| reshape_view | 1 | tenferro-rs direct API (ms) | 16 | 0.003166 | 197.88 |
| reshape_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| reshape_view | 4 | PyTorch Python (ms) | 16 | 0.006917 | 432.31 |
| reshape_view | 4 | tenferro-rs direct API (ms) | 16 | 0.003292 | 205.75 |
| slice_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| slice_view | 1 | PyTorch Python (ms) | 16 | 0.004750 | 296.88 |
| slice_view | 1 | tenferro-rs direct API (ms) | 16 | 0.002792 | 174.50 |
| slice_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| slice_view | 4 | PyTorch Python (ms) | 16 | 0.005959 | 372.44 |
| slice_view | 4 | tenferro-rs direct API (ms) | 16 | 0.002916 | 182.25 |
| transpose_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| transpose_view | 1 | PyTorch Python (ms) | 16 | 0.004666 | 291.62 |
| transpose_view | 1 | tenferro-rs direct API (ms) | 16 | 0.002542 | 158.88 |
| transpose_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| transpose_view | 4 | PyTorch Python (ms) | 16 | 0.006166 | 385.38 |
| transpose_view | 4 | tenferro-rs direct API (ms) | 16 | 0.002542 | 158.88 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`julia-base`, 19.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`julia-base`, 19.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 25.4x faster than the slowest successful cell (`pytorch-cpu`, 9.434 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 12.2x faster than the slowest successful cell (`tenferro-trace`, 5.636 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 15.1x faster than the slowest successful cell (`tenferro-trace`, 5.636 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=4, shape=`8388608`): `jax-cpu` is 10.4x faster than the slowest successful cell (`julia-base`, 23.384 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 111.0x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 144.0x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 99.0x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 124.0x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 97.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 128.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

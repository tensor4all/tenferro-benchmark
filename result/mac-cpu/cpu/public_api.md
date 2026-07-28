# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_195329/run.yaml`
- Timestamp: `20260728_195329`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260728_195329`.

- tenferro-rs commit: `4ff6b8e1d56fc8d3f6b772f6053c09c745429585`

## CPU Information

- Model: `Apple M4`
- Vendor: `Apple`
- Logical CPUs: `10`
- Physical CPUs: `10`
- Sockets: `1`
- Cores per socket: `10`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Performance: 4 physical / 4 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 4 CPUs/L2); Efficiency: 6 physical / 6 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.2-arm64-arm-64bit`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_195329/run_t1.yaml`
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

### Threads: 4

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_195329/run_t4.yaml`
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

## Timing Discipline

- Input fixture tensors are created during warmup and outside the measured region for tenferro-rs, PyTorch, and JAX.
- tenferro-rs trace graphs are constructed and compiled outside the measured region; each compiled graph is reused for every warmup and timed run.
- JAX functions are compiled with `jax.jit` during warmup, outside the measured region; timed calls include dispatch through `jax.block_until_ready`.
- Allocation-returning API rows create their output tensor inside each timed call.
- `cpu/output_reuse` rows allocate the destination during warmup and reuse it; tenferro-rs `*_into` is compared with PyTorch `out=`/`copy_`.
- Trace mode is `unsupported` for caller-output rows because compiled tenferro-rs graphs own their output tensors; JAX is shown as missing because it has no equivalent mutable `out=` API.
- PyTorch view-producing indexing operations are cloned inside the timed region to match tenferro-rs owned, materialized outputs.
- `reshape` is not force-materialized: each framework's public reshape storage semantics are part of the API behavior being compared.
- PyTorch complex conjugation uses `torch.conj_physical` to match tenferro-rs physical output rather than the lazy conjugate view from `torch.conj`.
- `dynamic_update_slice` reports trace mode as `unsupported` because tenferro-rs does not currently expose a corresponding `TracedTensor` API.
- `full_piv_lu` and `full_piv_lu_solve` are excluded because PyTorch has no direct public full-pivot equivalent; substituting `torch.linalg.solve` would compare different algorithms.
- `svd_full` remains in the table even when the selected tenferro-rs provider reports it as unsupported.

## Threads: 1 4

- CSV: `data/results/mac-cpu/cpu/public_api/20260728_195329/cpu_public_api_t1_20260728_195329.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260728_195329/cpu_public_api_t4_20260728_195329.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260728_195329/cpu_public_api_20260728_195329.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.742 ± 0.066 | 2.679 ± 0.045 | 3.077 ± 0.187 | 0.798 ± 0.038 |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.671 ± 0.061 | 2.690 ± 0.060 | 2.730 ± 0.065 | 0.803 ± 0.018 |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 7.295 ± 0.483 | 6.715 ± 1.033 | 11.027 ± 2.837 | 5.541 ± 0.485 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 5.705 ± 1.076 | 5.976 ± 0.435 | 5.825 ± 0.704 | 5.560 ± 0.076 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 7.742 ± 0.317 | 7.728 ± 0.255 | 10.485 ± 0.129 | 6.537 ± 0.591 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 4.245 ± 0.116 | 4.343 ± 0.178 | 4.453 ± 0.315 | 6.666 ± 1.889 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 6.640 ± 0.578 | 6.508 ± 0.133 | 6.303 ± 0.244 | 7.106 ± 1.101 |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 6.225 ± 0.603 | 5.675 ± 0.174 | 6.241 ± 0.134 | 7.152 ± 1.124 |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 6.920 ± 0.052 | 6.901 ± 0.205 | 6.601 ± 0.087 | 7.456 ± 0.588 |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 6.349 ± 0.067 | 6.217 ± 0.275 | 6.516 ± 0.235 | 7.892 ± 0.823 |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 6.075 ± 0.139 | 6.131 ± 0.104 | 6.916 ± 0.073 | 13.419 ± 0.695 |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 6.211 ± 0.097 | 6.176 ± 0.097 | 6.901 ± 0.117 | 13.358 ± 1.510 |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 24.313 ± 0.383 | 24.433 ± 0.333 | 23.035 ± 0.416 | 8.308 ± 2.136 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 7.598 ± 1.299 | 6.602 ± 0.141 | 7.604 ± 0.181 | 8.560 ± 2.515 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 26.322 ± 0.125 | 26.293 ± 0.114 | 28.456 ± 0.108 | 12.497 ± 0.192 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 8.804 ± 2.442 | 8.284 ± 1.920 | 9.412 ± 1.991 | 12.898 ± 0.996 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 4.898 ± 0.286 | 4.717 ± 0.145 | 4.627 ± 0.091 | 4.266 ± 0.098 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 4.422 ± 0.268 | 4.284 ± 0.455 | 4.412 ± 0.202 | 4.305 ± 0.185 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 21.205 ± 0.155 | 21.922 ± 0.240 | 4.065 ± 0.066 | 1.222 ± 0.017 |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 7.484 ± 1.091 | 6.958 ± 0.506 | 1.222 ± 0.431 | 1.374 ± 0.277 |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.509 ± 0.106 | 4.589 ± 0.123 | 4.761 ± 0.106 | 3.794 ± 0.018 |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.582 ± 0.102 | 4.638 ± 0.118 | 4.748 ± 0.087 | 3.772 ± 0.035 |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.146 ± 0.084 | 3.433 ± 0.159 | 3.202 ± 0.063 | 1.299 ± 0.012 |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.295 ± 0.064 | 3.615 ± 0.075 | 3.363 ± 0.080 | 1.312 ± 0.064 |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.663 ± 0.190 | 3.740 ± 0.267 | 3.787 ± 0.288 | 2.799 ± 0.048 |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.903 ± 0.216 | 3.913 ± 0.066 | 3.857 ± 0.155 | 2.780 ± 0.143 |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 6.290 ± 0.236 | 6.369 ± 0.113 | 6.518 ± 0.321 | 7.612 ± 0.922 |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 5.793 ± 0.212 | 5.699 ± 0.142 | 6.181 ± 0.224 | 7.873 ± 0.932 |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 6.385 ± 0.885 | unsupported | 6.919 ± 0.169 | 7.942 ± 0.626 |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 5.258 ± 1.231 | unsupported | 5.206 ± 1.033 | 8.403 ± 1.680 |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 6.336 ± 0.211 | 6.350 ± 0.792 | 6.423 ± 0.378 | 5.943 ± 0.374 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 5.689 ± 0.164 | 5.839 ± 0.794 | 5.474 ± 0.446 | 5.612 ± 0.542 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 9.000 ± 0.413 | 8.798 ± 0.271 | 8.958 ± 0.398 | 11.329 ± 0.633 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 8.759 ± 1.099 | 8.641 ± 0.197 | 9.114 ± 0.853 | 11.382 ± 0.336 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 18.465 ± 0.196 | 18.525 ± 0.172 | 22.432 ± 0.258 | 5.343 ± 1.060 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 6.793 ± 1.842 | 6.745 ± 1.227 | 7.846 ± 3.436 | 5.247 ± 0.214 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 3.055 ± 0.324 | 3.064 ± 0.068 | 3.061 ± 0.148 | 3.705 ± 0.331 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 3.056 ± 0.211 | 3.051 ± 0.351 | 2.995 ± 0.078 | 3.658 ± 0.238 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 43.494 ± 5.545 | 43.658 ± 11.996 | 8.683 ± 0.047 | 6.575 ± 0.443 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 9.756 ± 3.039 | 9.740 ± 3.085 | 7.295 ± 0.765 | 7.727 ± 0.445 |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 17.613 ± 0.267 | 17.563 ± 0.156 | 21.266 ± 0.578 | 4.660 ± 2.086 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 5.997 ± 0.708 | 5.660 ± 0.896 | 6.428 ± 1.138 | 4.236 ± 1.055 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 8.921 ± 0.402 | 9.079 ± 0.853 | 8.762 ± 0.389 | 11.424 ± 0.181 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 8.508 ± 0.155 | 8.561 ± 0.113 | 8.846 ± 0.882 | 11.486 ± 0.996 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 17.994 ± 0.205 | 17.975 ± 0.077 | 14.997 ± 0.079 | 3.126 ± 1.045 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 5.537 ± 0.578 | 5.565 ± 1.768 | 5.015 ± 0.593 | 3.276 ± 0.756 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 9.019 ± 0.169 | 9.008 ± 0.171 | 17.945 ± 0.097 | 3.854 ± 1.446 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.728 ± 0.920 | 2.894 ± 0.413 | 6.414 ± 1.339 | 3.647 ± 0.757 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 16.559 ± 0.141 | 16.723 ± 0.615 | 26.542 ± 0.090 | 5.667 ± 1.564 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 5.309 ± 1.107 | 5.188 ± 0.724 | 8.131 ± 2.250 | 5.637 ± 2.625 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 10.925 ± 0.099 | 10.947 ± 0.117 | 13.730 ± 0.126 | 3.641 ± 1.117 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.728 ± 0.999 | 3.515 ± 1.613 | 4.803 ± 1.384 | 3.916 ± 1.653 |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 8.844 ± 1.302 | 10.577 ± 3.505 | 9.679 ± 0.593 | 11.410 ± 0.123 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 8.655 ± 0.928 | 8.606 ± 1.013 | 8.949 ± 0.932 | 11.208 ± 0.343 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 10.544 ± 2.741 | 9.779 ± 3.351 | 8.953 ± 0.290 | 11.210 ± 0.251 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 8.608 ± 0.439 | 9.023 ± 1.882 | 8.650 ± 0.228 | 11.340 ± 0.803 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 9.470 ± 0.900 | 9.202 ± 0.380 | 9.027 ± 1.268 | 11.446 ± 0.533 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 8.629 ± 0.230 | 8.502 ± 0.268 | 8.706 ± 0.319 | 11.225 ± 1.139 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 6.350 ± 0.063 | 6.286 ± 0.281 | 6.433 ± 0.340 | 5.729 ± 0.252 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 5.608 ± 0.287 | 5.774 ± 0.488 | 5.858 ± 0.692 | 5.529 ± 0.176 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 20.356 ± 0.099 | 20.417 ± 0.164 | 45.463 ± 0.148 | 6.967 ± 2.189 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 6.762 ± 0.914 | 7.201 ± 2.164 | 13.932 ± 2.135 | 6.316 ± 0.985 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 6.195 ± 0.038 | 6.190 ± 0.059 | 9.754 ± 1.501 | 0.714 ± 0.052 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 6.146 ± 0.051 | 6.169 ± 0.088 | 6.286 ± 0.800 | 0.705 ± 0.303 |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.784 ± 0.021 | 3.780 ± 0.206 | 4.815 ± 0.070 | 1.833 ± 0.323 |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.751 ± 0.040 | 3.766 ± 0.035 | 1.852 ± 0.319 | 1.890 ± 0.370 |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 28.961 ± 1.809 | 29.095 ± 0.811 | 4.061 ± 0.716 | 5.423 ± 1.783 |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.569 ± 0.811 | 7.533 ± 0.816 | 4.063 ± 0.739 | 5.308 ± 2.829 |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 18.613 ± 1.816 | 18.818 ± 0.410 | 4.013 ± 0.174 | 4.940 ± 1.406 |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.030 ± 0.627 | 5.043 ± 0.299 | 3.729 ± 4.593 | 4.265 ± 1.008 |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 19.823 ± 0.685 | 19.496 ± 0.201 | 22.730 ± 0.149 | 5.823 ± 0.600 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 6.492 ± 1.923 | 5.175 ± 0.350 | 5.853 ± 1.371 | 5.641 ± 0.356 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 13.136 ± 0.684 | 12.858 ± 0.485 | 12.896 ± 0.378 | 7.498 ± 0.687 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 6.011 ± 0.997 | 6.136 ± 0.938 | 6.039 ± 0.687 | 7.115 ± 1.387 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 13.900 ± 4.760 | 15.194 ± 1.174 | 9.727 ± 1.414 | 12.031 ± 0.538 |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 11.938 ± 0.312 | 11.868 ± 0.890 | 8.869 ± 0.062 | 12.037 ± 0.509 |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 6.543 ± 0.473 | 6.506 ± 0.307 | 7.119 ± 0.692 | 5.814 ± 0.305 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 5.982 ± 0.751 | 6.630 ± 0.894 | 5.924 ± 0.461 | 5.501 ± 0.763 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 16.894 ± 0.120 | 16.317 ± 0.164 | 18.305 ± 0.243 | 4.630 ± 0.826 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 5.655 ± 1.635 | 5.170 ± 0.971 | 5.856 ± 1.095 | 4.270 ± 0.818 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 9.659 ± 1.257 | 9.484 ± 0.470 | 8.418 ± 0.019 | 6.123 ± 0.452 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 5.639 ± 0.534 | 5.733 ± 1.395 | 5.949 ± 0.609 | 5.823 ± 1.297 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 10.131 ± 0.535 | 10.205 ± 0.646 | 9.588 ± 0.836 | 11.666 ± 0.619 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 8.696 ± 0.809 | 8.701 ± 0.871 | 8.819 ± 1.139 | 11.341 ± 0.523 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 23.007 ± 0.109 | 22.959 ± 0.129 | 43.054 ± 0.293 | 4.136 ± 0.403 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 7.230 ± 1.467 | 7.702 ± 3.381 | 13.836 ± 3.064 | 3.151 ± 1.456 |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 10.547 ± 1.118 | 10.494 ± 0.168 | 0.334 ± 0.012 | 0.343 ± 0.020 |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 10.776 ± 0.331 | 10.547 ± 0.167 | 0.273 ± 0.005 | 0.344 ± 0.048 |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304` | 11.063 ± 0.130 | 11.025 ± 0.069 | 0.315 ± 0.014 | 0.266 ± 0.012 |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304` | 1.504 ± 0.326 | 1.793 ± 0.531 | 0.254 ± 0.018 | 0.259 ± 0.020 |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 5.892 ± 0.045 | unsupported | 0.568 ± 0.064 | 0.467 ± 0.031 |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.123 ± 0.059 | unsupported | 0.507 ± 0.035 | 0.451 ± 0.022 |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 3.627 ± 0.598 | 2.847 ± 0.173 | 0.179 ± 0.001 | 0.130 ± 0.018 |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.966 ± 0.666 | 1.004 ± 0.100 | 0.181 ± 0.000 | 0.083 ± 0.008 |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 7.611 ± 0.098 | 7.640 ± 0.052 | 0.561 ± 0.015 | 0.295 ± 0.043 |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 7.606 ± 0.034 | 7.590 ± 0.046 | 0.499 ± 0.007 | 0.257 ± 0.012 |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 5.319 ± 0.065 | 5.267 ± 0.073 | 0.420 ± 0.083 | 0.267 ± 0.015 |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 5.316 ± 0.482 | 5.326 ± 0.060 | 0.277 ± 0.029 | 0.282 ± 0.010 |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 4.425 ± 0.058 | 4.226 ± 0.065 | 0.289 ± 0.008 | 0.258 ± 0.011 |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 4.365 ± 0.395 | 4.271 ± 0.101 | 0.294 ± 0.009 | 0.275 ± 0.047 |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304` | 7.408 ± 0.102 | 7.419 ± 0.073 | 0.637 ± 0.036 | 0.507 ± 0.013 |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304` | 7.436 ± 0.062 | 7.417 ± 0.023 | 0.511 ± 0.052 | 0.552 ± 0.066 |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 5.023 ± 0.314 | 5.112 ± 0.182 | 14.682 ± 0.380 | 7.072 ± 0.676 |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 5.812 ± 0.774 | 5.054 ± 0.553 | 7.794 ± 0.511 | 6.418 ± 0.128 |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 5.172 ± 0.254 | 5.508 ± 0.318 | 4.284 ± 0.350 | 5.293 ± 0.219 |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 5.243 ± 0.144 | 5.562 ± 0.083 | 4.172 ± 0.216 | 5.225 ± 0.301 |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.602 ± 0.180 | 4.728 ± 0.213 | 5.298 ± 0.212 | 5.751 ± 0.058 |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.779 ± 0.214 | 4.753 ± 0.124 | 5.262 ± 0.125 | 5.755 ± 0.080 |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 6.985 ± 0.134 | 4.885 ± 0.100 | 4.699 ± 0.012 | 4.138 ± 0.071 |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 7.078 ± 0.099 | 4.961 ± 0.068 | 4.723 ± 0.113 | 4.215 ± 0.063 |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 7.771 ± 0.117 | 0.228 ± 0.001 | 0.287 ± 0.002 | 8.040 ± 0.107 |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 7.860 ± 0.164 | 0.244 ± 0.005 | 0.287 ± 0.008 | 8.060 ± 0.110 |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 5.214 ± 0.359 | 6.559 ± 0.082 | 5.349 ± 0.116 | 5.071 ± 0.098 |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 4.868 ± 0.105 | 6.252 ± 0.139 | 4.797 ± 0.172 | 4.618 ± 0.081 |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | 5.211 ± 0.073 | 5.351 ± 0.113 | 3.069 ± 0.124 | 13.141 ± 0.406 |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | 5.410 ± 0.151 | 5.467 ± 0.122 | 2.984 ± 0.064 | 13.355 ± 0.266 |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 4.788 ± 0.212 | 5.282 ± 0.210 | 7.298 ± 0.231 | 6.209 ± 0.309 |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 5.066 ± 0.110 | 5.414 ± 0.488 | 6.451 ± 0.022 | 6.317 ± 0.286 |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 23.441 ± 0.088 | 24.698 ± 0.209 | 1.411 ± 0.020 | 1.178 ± 0.350 |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 8.937 ± 1.236 | 8.215 ± 1.087 | 1.375 ± 0.029 | 1.198 ± 0.123 |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.292 ± 0.293 | 6.485 ± 0.190 | 6.317 ± 0.295 | 6.919 ± 0.250 |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.604 ± 0.277 | 6.626 ± 0.204 | 6.391 ± 0.131 | 7.080 ± 0.313 |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 6.423 ± 0.204 | 6.434 ± 0.274 | 6.312 ± 0.164 | 6.947 ± 0.134 |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 6.541 ± 0.194 | 6.505 ± 0.161 | 6.397 ± 0.183 | 6.988 ± 0.075 |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 5.150 ± 0.227 | 5.590 ± 0.128 | 4.202 ± 0.301 | 5.282 ± 0.407 |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 5.252 ± 0.114 | 5.476 ± 0.251 | 4.103 ± 0.150 | 5.224 ± 0.203 |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | unsupported | unsupported | 18.366 ± 3.656 | 18.951 ± 0.397 |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | unsupported | unsupported | 19.116 ± 0.607 | 20.360 ± 1.021 |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 5.925 ± 0.258 | 6.005 ± 0.348 | 5.756 ± 1.376 | 15.974 ± 1.775 |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 6.944 ± 0.130 | 6.923 ± 0.190 | 5.243 ± 0.432 | 16.994 ± 1.729 |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 15.283 ± 1.258 | unsupported | 9.110 ± 0.363 | - |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 14.487 ± 0.277 | unsupported | 9.139 ± 0.547 | - |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 12.350 ± 0.524 | unsupported | 6.909 ± 3.534 | - |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 11.205 ± 0.603 | unsupported | 5.724 ± 0.062 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 6.029 ± 0.119 | unsupported | 6.909 ± 0.731 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 5.606 ± 0.587 | unsupported | 5.747 ± 0.056 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 15.064 ± 0.738 | unsupported | 9.000 ± 0.881 | - |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 14.279 ± 0.964 | unsupported | 9.133 ± 0.428 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 4.963 ± 0.186 | unsupported | 6.915 ± 0.296 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 4.617 ± 0.146 | unsupported | 4.713 ± 0.532 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 5.678 ± 0.058 | unsupported | 7.386 ± 0.373 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 5.367 ± 0.176 | unsupported | 6.217 ± 3.088 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 15.200 ± 0.687 | unsupported | 8.740 ± 0.177 | - |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 14.466 ± 1.134 | unsupported | 9.698 ± 4.220 | - |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 12.461 ± 0.436 | unsupported | 6.278 ± 0.133 | - |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 11.005 ± 0.508 | unsupported | 5.844 ± 0.172 | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 15.251 ± 0.756 | unsupported | 8.776 ± 0.233 | - |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 14.544 ± 0.955 | unsupported | 9.357 ± 0.495 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 5.009 ± 1.008 | 3.675 ± 0.085 | 3.765 ± 0.218 | 3.671 ± 0.294 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 3.595 ± 0.330 | 3.632 ± 0.210 | 3.623 ± 0.189 | 3.434 ± 0.360 |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 5.078 ± 0.233 | 4.983 ± 0.175 | 5.262 ± 0.318 | 4.510 ± 0.161 |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 4.255 ± 0.135 | 4.290 ± 0.186 | 4.484 ± 0.266 | 4.344 ± 0.421 |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 4.848 ± 0.853 | 4.928 ± 0.732 | 7.425 ± 0.241 | 7.101 ± 0.434 |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 4.847 ± 0.058 | 4.654 ± 0.121 | 7.090 ± 0.461 | 7.074 ± 0.326 |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 3.114 ± 0.161 | 3.068 ± 0.178 | 6.410 ± 0.146 | 4.437 ± 0.245 |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 3.200 ± 0.223 | 3.106 ± 0.044 | 4.471 ± 0.481 | 4.334 ± 0.166 |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 6.604 ± 0.525 | 6.003 ± 0.077 | 0.001 ± 0.000 | 5.359 ± 0.366 |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 6.085 ± 0.286 | 6.461 ± 0.639 | 0.001 ± 0.000 | 5.184 ± 0.838 |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 21.473 ± 0.312 | 21.663 ± 0.933 | 19.856 ± 0.229 | 7.776 ± 0.433 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 9.985 ± 1.136 | 10.313 ± 1.039 | 19.785 ± 0.940 | 7.151 ± 0.544 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 22.073 ± 0.151 | 22.017 ± 0.313 | 5.019 ± 0.275 | 3.130 ± 0.158 |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 21.846 ± 0.130 | 21.956 ± 0.220 | 3.215 ± 0.094 | 2.757 ± 0.103 |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 21.745 ± 0.252 | 21.744 ± 0.235 | 4.697 ± 0.106 | 2.991 ± 0.204 |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 21.760 ± 0.391 | 21.631 ± 0.216 | 3.148 ± 0.275 | 2.873 ± 0.255 |

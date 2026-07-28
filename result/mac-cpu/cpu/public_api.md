# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_181002/run.yaml`
- Timestamp: `20260728_181002`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260728_181002`.

- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_181002/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_181002/run_t4.yaml`
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

- Input fixture tensors are created during warmup and outside the measured region for both tenferro-rs and PyTorch.
- tenferro-rs trace graphs are constructed and compiled outside the measured region; each compiled graph is reused for every warmup and timed run.
- Each timed call creates the output tensor.
- PyTorch view-producing indexing operations are cloned inside the timed region to match tenferro-rs owned, materialized outputs.
- PyTorch complex conjugation uses `torch.conj_physical` to match tenferro-rs physical output rather than the lazy conjugate view from `torch.conj`.
- `dynamic_update_slice` reports trace mode as `unsupported` because tenferro-rs does not currently expose a corresponding `TracedTensor` API.
- `full_piv_lu` and `full_piv_lu_solve` are excluded because PyTorch has no direct public full-pivot equivalent; substituting `torch.linalg.solve` would compare different algorithms.

## Threads: 1 4

- CSV: `data/results/mac-cpu/cpu/public_api/20260728_181002/cpu_public_api_t1_20260728_181002.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260728_181002/cpu_public_api_t4_20260728_181002.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260728_181002/cpu_public_api_20260728_181002.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.804 ± 0.033 | 2.709 ± 0.024 | 3.117 ± 0.046 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.787 ± 0.075 | 2.893 ± 0.214 | 2.735 ± 0.046 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 8.779 ± 1.450 | 6.431 ± 1.464 | 6.262 ± 0.117 | - |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 5.708 ± 0.253 | 5.854 ± 0.487 | 5.592 ± 0.412 | - |
| cpu/complex | `div` | c64 | 1 | `8388608` | 7.593 ± 0.041 | 9.553 ± 7.273 | 11.385 ± 0.369 | - |
| cpu/complex | `div` | c64 | 4 | `8388608` | 4.463 ± 0.340 | 4.293 ± 0.247 | 4.385 ± 0.285 | - |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 6.885 ± 1.022 | 6.292 ± 0.183 | 6.093 ± 0.170 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 6.081 ± 0.746 | 5.782 ± 0.294 | 5.897 ± 0.113 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 6.058 ± 0.012 | 6.106 ± 0.038 | 6.893 ± 0.017 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 6.222 ± 0.029 | 6.190 ± 0.041 | 6.902 ± 0.095 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 24.294 ± 0.803 | 24.431 ± 0.158 | 22.656 ± 0.797 | - |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 7.570 ± 0.783 | 7.431 ± 0.556 | 6.683 ± 0.555 | - |
| cpu/complex | `log` | c64 | 1 | `4194304` | 26.203 ± 0.111 | 26.009 ± 0.020 | 28.145 ± 0.030 | - |
| cpu/complex | `log` | c64 | 4 | `4194304` | 7.779 ± 0.787 | 8.045 ± 0.813 | 9.456 ± 1.334 | - |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 4.945 ± 1.138 | 4.536 ± 0.756 | 4.565 ± 0.074 | - |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 4.436 ± 0.078 | 4.390 ± 0.179 | 4.402 ± 0.350 | - |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 20.744 ± 0.084 | 21.544 ± 0.029 | 3.995 ± 0.023 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 8.247 ± 0.676 | 7.452 ± 0.572 | 1.098 ± 0.178 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.583 ± 0.037 | 4.569 ± 0.012 | 4.818 ± 0.023 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.588 ± 0.052 | 4.619 ± 0.065 | 4.688 ± 0.067 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.109 ± 0.037 | 3.368 ± 0.023 | 3.210 ± 0.189 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.607 ± 0.669 | 3.733 ± 0.029 | 3.304 ± 0.040 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.799 ± 0.008 | 3.862 ± 0.043 | 3.823 ± 0.019 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.894 ± 0.058 | 3.867 ± 0.221 | 3.678 ± 0.286 | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 6.341 ± 0.271 | 6.320 ± 0.178 | 6.180 ± 0.334 | - |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 5.352 ± 0.059 | 5.409 ± 0.724 | 5.710 ± 0.478 | - |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 11.688 ± 7.571 | 9.155 ± 0.357 | 8.614 ± 0.273 | - |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 9.271 ± 2.018 | 9.077 ± 0.687 | 8.861 ± 0.319 | - |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 18.394 ± 0.130 | 18.329 ± 0.167 | 22.128 ± 0.021 | - |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 6.843 ± 0.442 | 6.979 ± 1.049 | 7.535 ± 1.188 | - |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 3.054 ± 0.144 | 3.131 ± 0.146 | 2.949 ± 0.069 | - |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 3.080 ± 0.162 | 2.974 ± 0.193 | 2.920 ± 0.068 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 38.927 ± 0.375 | 38.878 ± 0.186 | 8.562 ± 0.061 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 9.742 ± 2.052 | 9.903 ± 1.221 | 6.989 ± 0.257 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 17.535 ± 0.350 | 17.670 ± 0.257 | 20.872 ± 0.039 | - |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 5.520 ± 1.654 | 5.040 ± 0.410 | 6.255 ± 1.184 | - |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 9.306 ± 1.038 | 9.106 ± 0.708 | 8.612 ± 0.482 | - |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 8.384 ± 0.039 | 8.400 ± 0.450 | 8.706 ± 0.401 | - |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 17.812 ± 0.159 | 17.853 ± 0.143 | 14.833 ± 0.014 | - |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 5.524 ± 1.475 | 5.260 ± 0.935 | 5.333 ± 1.240 | - |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 8.995 ± 0.142 | 8.908 ± 0.304 | 17.918 ± 0.028 | - |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.664 ± 0.281 | 2.558 ± 0.559 | 5.156 ± 0.833 | - |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 16.462 ± 0.165 | 16.316 ± 0.181 | 26.134 ± 0.016 | - |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 5.615 ± 0.482 | 5.345 ± 0.977 | 7.487 ± 0.699 | - |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 10.876 ± 0.059 | 10.962 ± 0.208 | 13.670 ± 0.077 | - |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 4.225 ± 1.148 | 3.140 ± 0.926 | 3.898 ± 1.196 | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 9.075 ± 0.224 | 9.093 ± 0.308 | 8.661 ± 0.154 | - |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 8.463 ± 0.280 | 8.542 ± 0.126 | 8.581 ± 0.481 | - |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 9.053 ± 0.314 | 9.372 ± 0.552 | 8.661 ± 0.205 | - |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 8.393 ± 0.142 | 8.535 ± 1.398 | 8.763 ± 0.234 | - |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 10.238 ± 1.064 | 9.617 ± 1.001 | 8.599 ± 0.167 | - |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 8.755 ± 0.285 | 8.510 ± 0.842 | 8.880 ± 0.729 | - |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 6.549 ± 0.237 | 6.678 ± 0.687 | 6.193 ± 0.117 | - |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 5.347 ± 0.060 | 5.329 ± 0.082 | 5.555 ± 0.311 | - |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 20.324 ± 0.189 | 20.348 ± 0.099 | 44.887 ± 0.133 | - |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 7.128 ± 1.049 | 6.088 ± 1.173 | 12.530 ± 1.521 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 6.040 ± 0.075 | 6.001 ± 0.406 | 8.838 ± 0.130 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 6.190 ± 0.497 | 6.167 ± 0.052 | 6.159 ± 0.588 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.776 ± 0.047 | 3.767 ± 0.072 | 4.771 ± 0.017 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.828 ± 0.045 | 3.796 ± 0.233 | 1.831 ± 0.046 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 28.614 ± 0.502 | 27.673 ± 0.423 | 3.909 ± 0.174 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.661 ± 0.847 | 7.534 ± 0.432 | 3.630 ± 0.341 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 17.741 ± 0.360 | 17.719 ± 0.590 | 3.840 ± 0.080 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.367 ± 0.325 | 5.853 ± 1.766 | 3.367 ± 0.530 | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 19.419 ± 0.122 | 19.407 ± 0.222 | 22.426 ± 0.029 | - |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 6.506 ± 1.193 | 6.436 ± 1.630 | 6.636 ± 0.801 | - |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 12.829 ± 0.137 | 12.751 ± 0.124 | 12.797 ± 0.135 | - |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 5.684 ± 0.652 | 5.678 ± 0.303 | 5.821 ± 0.697 | - |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 12.319 ± 0.263 | 12.249 ± 0.293 | 9.407 ± 0.122 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 11.721 ± 0.153 | 11.720 ± 0.123 | 9.232 ± 0.320 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 6.492 ± 0.224 | 6.522 ± 0.184 | 6.172 ± 0.046 | - |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 5.656 ± 0.122 | 5.645 ± 0.051 | 5.551 ± 0.495 | - |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 16.273 ± 0.195 | 16.461 ± 0.109 | 17.954 ± 0.025 | - |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 5.295 ± 0.796 | 5.142 ± 1.608 | 5.359 ± 1.363 | - |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 8.444 ± 0.109 | 8.401 ± 0.203 | 8.445 ± 0.150 | - |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 5.947 ± 0.386 | 5.564 ± 0.261 | 5.722 ± 0.255 | - |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 10.974 ± 2.314 | 11.977 ± 1.674 | 8.638 ± 0.194 | - |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 9.108 ± 0.625 | 9.378 ± 0.675 | 8.758 ± 0.249 | - |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 22.968 ± 0.317 | 22.889 ± 0.088 | 42.636 ± 0.781 | - |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 10.251 ± 1.760 | 6.711 ± 0.393 | 12.683 ± 1.438 | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 11.538 ± 0.453 | 11.519 ± 0.075 | 0.327 ± 0.020 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 11.636 ± 0.144 | 11.623 ± 0.075 | 0.269 ± 0.005 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304` | 10.484 ± 0.058 | 9.947 ± 0.148 | 0.335 ± 0.029 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304` | 10.034 ± 0.081 | 10.001 ± 0.073 | 0.252 ± 0.005 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 5.539 ± 0.273 | unsupported | 0.520 ± 0.017 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 5.354 ± 0.088 | unsupported | 0.523 ± 0.101 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 4.102 ± 0.964 | 2.953 ± 0.082 | 0.179 ± 0.001 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 3.784 ± 0.671 | 3.055 ± 0.243 | 0.179 ± 0.003 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 7.622 ± 0.067 | 7.652 ± 0.467 | 0.542 ± 0.013 | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 7.593 ± 0.110 | 7.637 ± 0.039 | 0.507 ± 0.005 | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 5.255 ± 0.050 | 5.278 ± 0.023 | 0.328 ± 0.131 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 5.334 ± 0.044 | 5.356 ± 0.078 | 0.252 ± 0.015 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 4.041 ± 0.058 | 4.216 ± 0.196 | 0.285 ± 0.004 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 4.203 ± 0.235 | 4.194 ± 0.088 | 0.283 ± 0.010 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304` | 7.921 ± 0.101 | 7.917 ± 0.073 | 0.585 ± 0.020 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304` | 8.836 ± 2.032 | 8.165 ± 0.228 | 0.544 ± 0.075 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 6.037 ± 1.551 | 5.209 ± 0.443 | 14.056 ± 0.047 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 5.597 ± 0.550 | 4.926 ± 0.032 | 7.644 ± 0.446 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 5.229 ± 0.067 | 5.507 ± 0.126 | 3.953 ± 0.044 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 5.149 ± 0.130 | 5.483 ± 0.087 | 4.091 ± 0.392 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.704 ± 0.051 | 7.061 ± 2.213 | 5.268 ± 0.030 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.737 ± 0.105 | 4.773 ± 0.076 | 5.270 ± 0.065 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.195 ± 0.244 | 4.995 ± 0.080 | 4.688 ± 0.029 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 7.116 ± 0.020 | 4.928 ± 0.026 | 4.711 ± 0.045 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 8.013 ± 0.253 | 0.246 ± 0.012 | 0.283 ± 0.002 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 7.764 ± 0.117 | 0.242 ± 0.007 | 0.290 ± 0.011 | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 5.174 ± 0.142 | 6.590 ± 0.302 | 5.253 ± 0.054 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 4.885 ± 0.059 | 6.340 ± 0.154 | 4.818 ± 0.092 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 23.695 ± 0.376 | 24.684 ± 0.145 | 1.361 ± 0.001 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 9.020 ± 1.304 | 8.441 ± 0.692 | 1.369 ± 0.039 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.493 ± 0.549 | 6.504 ± 0.194 | 6.341 ± 0.021 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.553 ± 0.093 | 6.618 ± 0.072 | 6.358 ± 0.123 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 5.217 ± 0.145 | 5.552 ± 0.203 | 3.961 ± 0.043 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 5.256 ± 0.114 | 5.442 ± 0.105 | 4.106 ± 0.352 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 5.933 ± 0.115 | 5.963 ± 0.087 | 4.499 ± 0.029 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 6.726 ± 0.088 | 6.815 ± 0.051 | 4.958 ± 0.184 | - |

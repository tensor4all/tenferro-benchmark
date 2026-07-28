# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_182203/run.yaml`
- Timestamp: `20260728_182203`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260728_182203`.

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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_182203/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_182203/run_t4.yaml`
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
- Each timed call creates the output tensor.
- PyTorch view-producing indexing operations are cloned inside the timed region to match tenferro-rs owned, materialized outputs.
- PyTorch complex conjugation uses `torch.conj_physical` to match tenferro-rs physical output rather than the lazy conjugate view from `torch.conj`.
- `dynamic_update_slice` reports trace mode as `unsupported` because tenferro-rs does not currently expose a corresponding `TracedTensor` API.
- `full_piv_lu` and `full_piv_lu_solve` are excluded because PyTorch has no direct public full-pivot equivalent; substituting `torch.linalg.solve` would compare different algorithms.

## Threads: 1 4

- CSV: `data/results/mac-cpu/cpu/public_api/20260728_182203/cpu_public_api_t1_20260728_182203.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260728_182203/cpu_public_api_t4_20260728_182203.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260728_182203/cpu_public_api_20260728_182203.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.136 ± 0.815 | 3.653 ± 0.181 | 3.277 ± 0.101 | 0.810 ± 0.020 |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.777 ± 0.014 | 2.626 ± 0.039 | 3.387 ± 0.072 | 0.808 ± 0.013 |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 8.766 ± 1.125 | 32.050 ± 8.457 | 10.659 ± 2.555 | 6.184 ± 0.644 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 5.992 ± 0.495 | 5.718 ± 0.192 | 5.726 ± 0.160 | 5.623 ± 0.242 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 15.625 ± 13.056 | 13.498 ± 0.191 | 11.174 ± 1.256 | 6.680 ± 0.850 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 4.607 ± 0.051 | 4.738 ± 0.498 | 4.850 ± 0.437 | 6.398 ± 0.117 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 9.595 ± 4.841 | 10.435 ± 0.970 | 6.715 ± 0.059 | 8.156 ± 0.687 |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 6.350 ± 1.371 | 5.990 ± 0.317 | 8.977 ± 0.274 | 7.532 ± 1.265 |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 8.327 ± 0.160 | 8.363 ± 0.090 | 7.482 ± 0.060 | 12.801 ± 0.022 |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 6.378 ± 0.029 | 5.912 ± 0.016 | 9.350 ± 0.081 | 13.887 ± 0.008 |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 37.606 ± 1.160 | 37.687 ± 0.295 | 24.257 ± 0.335 | 8.648 ± 1.185 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 9.272 ± 1.080 | 10.445 ± 1.409 | 9.756 ± 0.171 | 7.629 ± 0.443 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 40.980 ± 0.376 | 40.727 ± 0.492 | 31.302 ± 0.302 | 13.158 ± 0.854 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 8.642 ± 2.561 | 10.284 ± 1.622 | 13.086 ± 1.559 | 13.052 ± 1.774 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 11.455 ± 1.650 | 9.060 ± 1.189 | 5.844 ± 0.837 | 4.644 ± 0.342 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 4.570 ± 0.182 | 4.540 ± 0.174 | 4.581 ± 0.290 | 4.340 ± 0.104 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 28.955 ± 2.071 | 26.435 ± 1.307 | 4.466 ± 0.039 | 1.354 ± 0.184 |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 6.532 ± 0.216 | 7.320 ± 1.209 | 1.345 ± 0.012 | 1.280 ± 0.080 |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 6.090 ± 0.308 | 6.086 ± 0.199 | 5.090 ± 0.026 | 3.791 ± 0.060 |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.645 ± 0.041 | 4.670 ± 0.039 | 6.733 ± 0.054 | 3.987 ± 0.030 |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 4.328 ± 0.109 | 5.222 ± 0.478 | 3.524 ± 0.062 | 1.310 ± 0.041 |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.412 ± 0.026 | 3.755 ± 0.011 | 4.246 ± 0.295 | 1.293 ± 0.023 |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 5.408 ± 0.110 | 5.486 ± 0.095 | 3.882 ± 0.030 | 2.833 ± 0.058 |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.860 ± 0.138 | 3.800 ± 0.024 | 5.875 ± 0.011 | 3.013 ± 0.041 |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 10.409 ± 1.017 | 10.964 ± 1.961 | 6.530 ± 0.324 | 6.588 ± 0.933 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 5.824 ± 0.282 | 5.976 ± 0.287 | 5.837 ± 0.297 | 5.891 ± 1.262 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 9.862 ± 1.252 | 13.000 ± 2.290 | 10.165 ± 0.544 | 11.523 ± 0.238 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 8.889 ± 0.511 | 8.864 ± 0.536 | 8.908 ± 0.229 | 11.809 ± 0.445 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 19.076 ± 0.427 | 20.223 ± 1.112 | 23.177 ± 0.127 | 7.816 ± 1.444 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 8.935 ± 1.035 | 9.384 ± 0.905 | 13.607 ± 1.093 | 7.575 ± 0.776 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 3.427 ± 0.205 | 3.564 ± 0.362 | 3.355 ± 0.221 | 4.047 ± 0.342 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 3.299 ± 0.144 | 3.427 ± 0.172 | 3.575 ± 0.126 | 4.057 ± 0.307 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 62.460 ± 5.175 | 60.124 ± 0.228 | 9.714 ± 0.167 | 7.747 ± 0.514 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 14.581 ± 1.481 | 16.119 ± 1.459 | 7.572 ± 0.207 | 7.591 ± 0.999 |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 19.359 ± 0.707 | 18.700 ± 0.060 | 23.404 ± 0.097 | 7.430 ± 1.627 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 10.299 ± 0.482 | 10.185 ± 0.383 | 14.169 ± 2.008 | 6.746 ± 0.368 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 11.673 ± 1.707 | 11.998 ± 1.406 | 9.869 ± 0.377 | 11.700 ± 0.421 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 8.891 ± 0.529 | 8.796 ± 0.397 | 9.219 ± 0.257 | 11.979 ± 0.355 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 19.860 ± 0.175 | 19.816 ± 0.134 | 16.704 ± 0.104 | 5.852 ± 0.914 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 7.161 ± 0.206 | 8.223 ± 0.375 | 8.358 ± 0.031 | 4.374 ± 0.177 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 8.955 ± 0.167 | 9.107 ± 0.055 | 18.244 ± 0.067 | 4.581 ± 0.997 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.823 ± 0.592 | 3.068 ± 0.386 | 10.133 ± 0.348 | 4.336 ± 0.120 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 18.189 ± 0.074 | 18.351 ± 0.091 | 29.291 ± 0.126 | 7.536 ± 1.068 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 7.385 ± 1.068 | 8.227 ± 1.442 | 14.821 ± 2.759 | 6.702 ± 0.229 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 10.961 ± 0.067 | 10.953 ± 0.213 | 14.221 ± 0.056 | 4.192 ± 0.302 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 4.216 ± 0.929 | 4.344 ± 0.455 | 7.899 ± 1.464 | 4.107 ± 0.377 |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 13.518 ± 1.400 | 12.671 ± 1.568 | 10.065 ± 0.785 | 11.832 ± 0.404 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 9.080 ± 0.274 | 9.350 ± 0.795 | 9.790 ± 0.443 | 12.155 ± 1.008 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 13.189 ± 0.676 | 13.456 ± 0.899 | 10.094 ± 0.127 | 12.108 ± 0.559 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 9.319 ± 0.541 | 9.245 ± 0.255 | 9.914 ± 1.000 | 11.908 ± 0.371 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 12.396 ± 0.804 | 12.235 ± 0.518 | 10.043 ± 0.320 | 12.098 ± 0.688 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 8.949 ± 0.627 | 8.917 ± 0.493 | 8.782 ± 0.372 | 11.918 ± 0.302 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 10.777 ± 3.772 | 11.793 ± 1.614 | 6.544 ± 0.746 | 6.944 ± 0.901 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 5.822 ± 0.306 | 5.788 ± 0.709 | 5.757 ± 0.670 | 5.821 ± 0.228 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 20.412 ± 0.088 | 21.121 ± 0.924 | 46.501 ± 0.751 | 7.180 ± 1.488 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 8.918 ± 1.588 | 6.790 ± 1.069 | 25.322 ± 0.170 | 6.562 ± 0.589 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 9.611 ± 0.186 | 9.577 ± 0.188 | 10.654 ± 0.894 | 0.895 ± 0.111 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 6.256 ± 0.116 | 6.195 ± 0.120 | 12.698 ± 0.563 | 0.683 ± 0.037 |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 5.828 ± 0.086 | 5.796 ± 0.098 | 4.781 ± 0.022 | 2.015 ± 0.093 |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.752 ± 0.103 | 3.766 ± 0.019 | 2.364 ± 0.040 | 2.014 ± 0.076 |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 35.694 ± 0.258 | 41.842 ± 2.036 | 4.989 ± 0.417 | 6.792 ± 0.601 |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.971 ± 0.571 | 7.715 ± 0.470 | 3.813 ± 0.087 | 6.159 ± 0.168 |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 20.647 ± 0.907 | 22.360 ± 1.230 | 4.249 ± 0.217 | 5.272 ± 0.730 |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.080 ± 0.549 | 5.137 ± 0.764 | 3.892 ± 0.162 | 5.729 ± 0.447 |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 22.170 ± 0.825 | 21.418 ± 0.089 | 23.079 ± 0.060 | 7.913 ± 0.826 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 6.095 ± 1.109 | 6.412 ± 0.952 | 8.897 ± 0.920 | 6.672 ± 0.425 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 20.269 ± 1.343 | 20.197 ± 1.173 | 14.622 ± 0.762 | 15.040 ± 0.251 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 8.912 ± 0.716 | 9.388 ± 0.922 | 8.632 ± 0.043 | 12.557 ± 0.193 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 16.014 ± 1.426 | 18.609 ± 4.173 | 12.295 ± 0.969 | 13.179 ± 0.967 |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 12.152 ± 0.249 | 12.705 ± 0.782 | 10.641 ± 0.512 | 12.686 ± 0.378 |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 11.271 ± 3.323 | 12.048 ± 0.758 | 6.713 ± 0.452 | 7.250 ± 1.150 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 7.300 ± 0.786 | 7.383 ± 1.285 | 6.263 ± 0.514 | 5.922 ± 0.308 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 18.251 ± 0.071 | 18.259 ± 0.043 | 20.131 ± 0.100 | 6.925 ± 1.114 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 8.613 ± 1.568 | 9.971 ± 1.059 | 12.016 ± 0.025 | 5.946 ± 0.507 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 16.061 ± 1.228 | 14.550 ± 1.905 | 10.948 ± 0.576 | 8.493 ± 1.293 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 7.226 ± 0.375 | 7.019 ± 0.856 | 6.434 ± 0.291 | 6.484 ± 0.730 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 11.923 ± 2.840 | 10.680 ± 1.375 | 9.898 ± 1.019 | 11.471 ± 0.527 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 8.804 ± 0.337 | 8.954 ± 0.599 | 8.767 ± 0.165 | 11.942 ± 0.476 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 24.538 ± 0.078 | 24.582 ± 0.074 | 47.786 ± 0.379 | 5.207 ± 1.113 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 13.565 ± 0.862 | 13.843 ± 0.661 | 28.544 ± 0.031 | 4.546 ± 0.260 |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 26.619 ± 10.193 | 26.304 ± 1.435 | 0.363 ± 0.049 | 0.353 ± 0.032 |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 11.772 ± 1.014 | 11.459 ± 1.036 | 0.423 ± 0.052 | 0.341 ± 0.024 |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304` | 17.352 ± 1.350 | 19.101 ± 0.879 | 0.335 ± 0.008 | 0.271 ± 0.025 |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304` | 10.011 ± 0.059 | 10.011 ± 0.101 | 0.396 ± 0.015 | 0.265 ± 0.010 |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 10.800 ± 1.580 | unsupported | 0.589 ± 0.115 | 0.469 ± 0.024 |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 5.705 ± 0.154 | unsupported | 0.863 ± 0.112 | 0.479 ± 0.030 |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 3.700 ± 0.124 | 3.673 ± 0.121 | 0.184 ± 0.002 | 0.086 ± 0.015 |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 3.850 ± 0.730 | 2.951 ± 0.071 | 0.348 ± 0.001 | 0.084 ± 0.009 |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 16.899 ± 0.486 | 16.983 ± 2.940 | 0.566 ± 0.100 | 0.261 ± 0.024 |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 7.515 ± 0.064 | 7.665 ± 0.707 | 0.691 ± 0.012 | 0.257 ± 0.006 |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 12.607 ± 2.178 | 12.221 ± 1.174 | 0.389 ± 0.157 | 0.264 ± 0.012 |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 5.262 ± 0.015 | 5.269 ± 0.017 | 0.354 ± 0.018 | 0.265 ± 0.017 |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 5.575 ± 0.151 | 5.412 ± 0.065 | 0.290 ± 0.005 | 0.249 ± 0.018 |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 4.212 ± 0.090 | 4.303 ± 0.189 | 0.501 ± 0.003 | 0.249 ± 0.008 |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304` | 10.370 ± 0.142 | 11.246 ± 0.242 | 0.638 ± 0.070 | 0.516 ± 0.037 |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304` | 7.938 ± 0.050 | 7.970 ± 0.074 | 0.852 ± 0.031 | 0.523 ± 0.023 |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 9.473 ± 1.952 | 7.865 ± 0.284 | 14.700 ± 0.232 | 6.973 ± 0.657 |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 5.723 ± 1.214 | 5.018 ± 0.119 | 12.500 ± 0.278 | 7.355 ± 0.229 |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 6.838 ± 0.220 | 7.529 ± 0.218 | 3.759 ± 0.183 | 5.420 ± 0.459 |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 5.307 ± 0.067 | 5.566 ± 0.105 | 5.912 ± 0.081 | 5.211 ± 0.255 |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.296 ± 0.844 | 7.358 ± 0.662 | 5.340 ± 0.056 | 5.812 ± 0.098 |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.763 ± 0.038 | 4.731 ± 0.062 | 8.171 ± 0.017 | 5.778 ± 0.049 |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 10.702 ± 1.317 | 7.433 ± 0.148 | 4.770 ± 0.060 | 4.201 ± 0.052 |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 7.092 ± 0.020 | 4.919 ± 0.029 | 6.836 ± 0.035 | 4.173 ± 0.075 |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 12.466 ± 0.964 | 0.336 ± 0.001 | 0.294 ± 0.006 | 8.152 ± 0.188 |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 7.891 ± 0.125 | 0.255 ± 0.017 | 0.435 ± 0.006 | 7.889 ± 0.133 |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 7.923 ± 1.142 | 9.873 ± 0.624 | 5.375 ± 0.145 | 5.180 ± 0.086 |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 4.893 ± 0.119 | 6.230 ± 0.081 | 6.262 ± 0.093 | 4.594 ± 0.025 |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 37.961 ± 2.091 | 40.767 ± 1.001 | 1.463 ± 0.017 | 1.198 ± 0.152 |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 7.925 ± 0.752 | 7.989 ± 0.663 | 1.776 ± 0.139 | 1.207 ± 0.142 |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 9.667 ± 0.506 | 9.466 ± 0.656 | 6.347 ± 0.146 | 7.077 ± 0.130 |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.596 ± 0.042 | 6.620 ± 0.095 | 8.280 ± 0.249 | 7.016 ± 0.081 |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 6.953 ± 0.366 | 7.290 ± 0.141 | 3.925 ± 0.408 | 5.314 ± 0.471 |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 5.246 ± 0.076 | 5.557 ± 0.306 | 5.878 ± 0.152 | 5.198 ± 0.118 |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 8.734 ± 0.127 | 8.708 ± 0.148 | 4.681 ± 0.096 | 15.553 ± 0.568 |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 6.911 ± 0.149 | 6.916 ± 0.104 | 7.192 ± 0.385 | 16.277 ± 0.486 |

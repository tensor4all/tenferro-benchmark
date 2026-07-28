# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_175226/run.yaml`
- Timestamp: `20260728_175226`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260728_175226`.

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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_175226/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_175226/run_t4.yaml`
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
- Each timed call creates the output tensor.
- PyTorch view-producing indexing operations are cloned inside the timed region to match tenferro-rs owned, materialized outputs.
- PyTorch complex conjugation uses `torch.conj_physical` to match tenferro-rs physical output rather than the lazy conjugate view from `torch.conj`.
- `full_piv_lu` and `full_piv_lu_solve` are excluded because PyTorch has no direct public full-pivot equivalent; substituting `torch.linalg.solve` would compare different algorithms.

## Threads: 1 4

- CSV: `data/results/mac-cpu/cpu/public_api/20260728_175226/cpu_public_api_t1_20260728_175226.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260728_175226/cpu_public_api_t4_20260728_175226.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260728_175226/cpu_public_api_20260728_175226.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.798 ± 0.060 | - | 3.062 ± 0.097 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.759 ± 0.215 | - | 2.753 ± 0.038 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 8.925 ± 1.862 | - | 6.758 ± 0.250 | - |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 6.361 ± 0.374 | - | 5.588 ± 0.626 | - |
| cpu/complex | `div` | c64 | 1 | `8388608` | 7.591 ± 0.107 | - | 10.720 ± 0.263 | - |
| cpu/complex | `div` | c64 | 4 | `8388608` | 6.108 ± 0.725 | - | 4.424 ± 0.280 | - |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 6.565 ± 1.051 | - | 6.617 ± 0.607 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 6.175 ± 0.040 | - | 6.076 ± 0.321 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 6.122 ± 0.035 | - | 6.893 ± 0.157 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 6.204 ± 0.294 | - | 6.853 ± 0.301 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 24.521 ± 0.371 | - | 23.011 ± 0.574 | - |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 8.996 ± 0.815 | - | 7.025 ± 0.041 | - |
| cpu/complex | `log` | c64 | 1 | `4194304` | 26.210 ± 0.034 | - | 29.095 ± 0.563 | - |
| cpu/complex | `log` | c64 | 4 | `4194304` | 10.718 ± 1.415 | - | 9.300 ± 1.198 | - |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 5.149 ± 0.419 | - | 5.146 ± 0.607 | - |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 5.498 ± 0.309 | - | 4.412 ± 0.223 | - |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 20.960 ± 0.065 | - | 4.716 ± 0.410 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 9.341 ± 0.878 | - | 1.116 ± 0.014 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.566 ± 0.042 | - | 4.915 ± 0.201 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.475 ± 0.150 | - | 4.851 ± 0.146 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.123 ± 0.061 | - | 3.253 ± 0.105 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.408 ± 0.651 | - | 3.328 ± 0.081 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.744 ± 0.125 | - | 3.889 ± 0.368 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.626 ± 0.048 | - | 3.871 ± 0.034 | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 6.451 ± 1.062 | - | 6.646 ± 0.374 | - |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 5.896 ± 0.258 | - | 6.068 ± 1.140 | - |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 8.819 ± 1.149 | - | 8.809 ± 0.384 | - |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 9.200 ± 0.370 | - | 10.055 ± 1.124 | - |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 18.836 ± 1.100 | - | 22.758 ± 0.164 | - |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 9.483 ± 0.617 | - | 7.570 ± 1.449 | - |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 3.454 ± 0.524 | - | 2.962 ± 0.084 | - |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 3.715 ± 0.684 | - | 3.065 ± 0.279 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 39.189 ± 1.599 | - | 8.695 ± 0.144 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 10.217 ± 0.655 | - | 6.009 ± 0.186 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 18.095 ± 0.313 | - | 21.069 ± 0.135 | - |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 7.355 ± 1.139 | - | 8.529 ± 1.201 | - |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 8.985 ± 1.551 | - | 9.047 ± 0.597 | - |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 9.070 ± 0.449 | - | 8.908 ± 0.455 | - |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 18.040 ± 0.655 | - | 14.941 ± 0.063 | - |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 6.821 ± 1.020 | - | 6.945 ± 0.951 | - |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 9.077 ± 0.054 | - | 18.122 ± 0.119 | - |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 3.685 ± 0.466 | - | 6.006 ± 1.416 | - |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 16.641 ± 0.324 | - | 26.391 ± 0.128 | - |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 7.286 ± 0.864 | - | 11.240 ± 2.700 | - |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 11.077 ± 0.163 | - | 13.891 ± 0.073 | - |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 5.627 ± 6.038 | - | 5.319 ± 0.963 | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 9.361 ± 0.947 | - | 8.884 ± 0.262 | - |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 9.047 ± 0.560 | - | 8.777 ± 0.509 | - |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 9.146 ± 0.430 | - | 9.012 ± 0.325 | - |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 9.361 ± 0.953 | - | 10.615 ± 0.482 | - |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 8.919 ± 0.206 | - | 8.682 ± 0.391 | - |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 9.070 ± 0.580 | - | 8.880 ± 0.453 | - |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 6.930 ± 1.136 | - | 6.663 ± 0.386 | - |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 6.003 ± 0.566 | - | 6.955 ± 0.326 | - |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 20.462 ± 0.069 | - | 45.970 ± 0.528 | - |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 8.859 ± 1.316 | - | 16.711 ± 1.256 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 5.891 ± 0.461 | - | 12.287 ± 0.220 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 6.251 ± 0.057 | - | 6.259 ± 0.568 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.759 ± 0.030 | - | 4.888 ± 0.486 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.821 ± 0.152 | - | 1.815 ± 0.033 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 28.393 ± 0.301 | - | 4.333 ± 0.367 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 9.807 ± 2.515 | - | 3.631 ± 0.085 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 17.765 ± 0.271 | - | 4.102 ± 0.098 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.219 ± 0.732 | - | 3.603 ± 0.241 | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 19.170 ± 0.119 | - | 22.793 ± 1.214 | - |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 6.470 ± 1.203 | - | 9.185 ± 1.453 | - |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 12.802 ± 0.250 | - | 13.016 ± 0.148 | - |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 6.529 ± 0.487 | - | 6.612 ± 0.567 | - |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 12.236 ± 0.358 | - | 9.537 ± 0.236 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 11.919 ± 0.503 | - | 9.209 ± 0.217 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 7.058 ± 1.301 | - | 6.876 ± 0.458 | - |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 6.388 ± 0.663 | - | 5.838 ± 0.424 | - |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 16.260 ± 0.128 | - | 18.107 ± 0.659 | - |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 6.246 ± 0.984 | - | 8.557 ± 1.270 | - |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 8.492 ± 0.153 | - | 8.719 ± 0.361 | - |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 6.429 ± 3.202 | - | 7.380 ± 0.579 | - |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 8.929 ± 1.085 | - | 8.923 ± 0.253 | - |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 8.879 ± 0.634 | - | 8.774 ± 0.485 | - |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 22.919 ± 0.066 | - | 43.004 ± 0.200 | - |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 8.873 ± 0.373 | - | 17.113 ± 1.604 | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 10.434 ± 0.084 | - | 0.378 ± 0.020 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 10.802 ± 0.799 | - | 0.268 ± 0.006 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304` | 10.177 ± 1.001 | - | 0.349 ± 0.009 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304` | 10.486 ± 0.177 | - | 0.265 ± 0.022 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 5.653 ± 0.152 | - | 0.640 ± 0.054 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 5.816 ± 0.879 | - | 0.466 ± 0.040 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 4.150 ± 0.970 | - | 0.182 ± 0.001 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 2.937 ± 0.185 | - | 0.185 ± 0.006 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 7.658 ± 0.086 | - | 0.655 ± 0.077 | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 8.394 ± 0.196 | - | 0.494 ± 0.007 | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 5.295 ± 0.094 | - | 0.466 ± 0.120 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 6.011 ± 1.067 | - | 0.252 ± 0.008 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 4.058 ± 0.024 | - | 0.288 ± 0.008 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 4.233 ± 0.191 | - | 0.288 ± 0.020 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304` | 7.352 ± 0.076 | - | 0.655 ± 0.066 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304` | 7.488 ± 0.029 | - | 0.506 ± 0.009 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 5.767 ± 1.295 | - | 14.985 ± 0.523 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 5.805 ± 0.656 | - | 7.569 ± 0.545 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 5.139 ± 0.026 | - | 4.077 ± 0.250 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 6.003 ± 0.435 | - | 3.981 ± 0.072 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.738 ± 0.114 | - | 5.242 ± 0.423 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.493 ± 0.045 | - | 5.275 ± 0.050 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.112 ± 0.036 | - | 4.630 ± 0.560 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 6.845 ± 0.021 | - | 4.744 ± 0.095 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 7.753 ± 0.145 | - | 0.290 ± 0.004 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 7.983 ± 0.283 | - | 0.289 ± 0.020 | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 5.087 ± 0.075 | - | 5.375 ± 0.149 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 5.056 ± 0.308 | - | 4.728 ± 0.169 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 23.386 ± 0.133 | - | 1.389 ± 0.009 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 11.869 ± 9.341 | - | 1.386 ± 0.030 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.431 ± 0.055 | - | 6.117 ± 0.132 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 7.058 ± 0.342 | - | 6.416 ± 0.110 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 5.145 ± 0.216 | - | 3.838 ± 0.228 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 5.595 ± 0.688 | - | 4.116 ± 0.107 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 5.913 ± 0.299 | - | 4.703 ± 0.221 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 7.709 ± 0.467 | - | 4.932 ± 0.124 | - |

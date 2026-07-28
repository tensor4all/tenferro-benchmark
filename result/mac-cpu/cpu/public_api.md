# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_174441/run.yaml`
- Timestamp: `20260728_174441`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260728_174441`.

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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_174441/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_174441/run_t4.yaml`
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
- `full_piv_lu` is intentionally excluded from this speed table because PyTorch has no direct public equivalent selected for this suite.
- PyTorch `full_piv_lu_solve` uses `torch.linalg.solve` as the closest solve-level comparison.

## Threads: 1 4

- CSV: `data/results/mac-cpu/cpu/public_api/20260728_174441/cpu_public_api_t1_20260728_174441.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260728_174441/cpu_public_api_t4_20260728_174441.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260728_174441/cpu_public_api_20260728_174441.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.767 ± 0.020 | - | 3.138 ± 0.037 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.802 ± 0.106 | - | 2.723 ± 0.081 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 8.403 ± 0.107 | - | 0.002 ± 0.000 | - |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 5.818 ± 0.516 | - | 0.002 ± 0.000 | - |
| cpu/complex | `div` | c64 | 1 | `8388608` | 7.603 ± 0.049 | - | 9.523 ± 0.867 | - |
| cpu/complex | `div` | c64 | 4 | `8388608` | 4.658 ± 0.371 | - | 4.456 ± 0.340 | - |
| cpu/complex | `dot_general_conj` | c64 | 1 | `640x640` | 6.521 ± 1.447 | - | 6.264 ± 0.097 | - |
| cpu/complex | `dot_general_conj` | c64 | 4 | `640x640` | 9.344 ± 0.915 | - | 5.831 ± 0.282 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 6.093 ± 0.031 | - | 6.903 ± 0.046 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 6.425 ± 0.081 | - | 6.925 ± 0.088 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 24.065 ± 0.651 | - | 22.443 ± 0.162 | - |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 7.431 ± 1.446 | - | 6.374 ± 0.343 | - |
| cpu/complex | `log` | c64 | 1 | `4194304` | 26.182 ± 0.290 | - | 28.298 ± 0.074 | - |
| cpu/complex | `log` | c64 | 4 | `4194304` | 8.427 ± 0.906 | - | 9.153 ± 0.709 | - |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 5.016 ± 0.370 | - | 4.617 ± 2.123 | - |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 4.503 ± 0.202 | - | 4.278 ± 0.238 | - |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 20.893 ± 0.099 | - | 4.047 ± 0.083 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 9.733 ± 0.594 | - | 1.107 ± 0.055 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.613 ± 0.055 | - | 4.823 ± 0.030 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.676 ± 0.108 | - | 4.691 ± 0.059 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.111 ± 0.047 | - | 3.197 ± 0.246 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.400 ± 0.011 | - | 3.334 ± 0.066 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.756 ± 0.051 | - | 3.839 ± 0.081 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.816 ± 0.049 | - | 3.880 ± 0.065 | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 6.306 ± 0.216 | - | 6.342 ± 0.370 | - |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 5.570 ± 0.412 | - | 5.487 ± 0.331 | - |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 9.763 ± 0.799 | - | 8.945 ± 0.679 | - |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 8.842 ± 0.350 | - | 8.777 ± 0.168 | - |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 18.427 ± 0.130 | - | 22.379 ± 0.207 | - |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 6.935 ± 0.803 | - | 7.404 ± 2.171 | - |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 3.087 ± 0.245 | - | 4.429 ± 0.273 | - |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 2.948 ± 0.045 | - | 4.317 ± 0.112 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 38.848 ± 0.673 | - | 8.644 ± 0.118 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 8.880 ± 1.622 | - | 7.137 ± 0.674 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 17.487 ± 0.177 | - | 20.994 ± 0.087 | - |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 5.252 ± 0.949 | - | 6.504 ± 0.332 | - |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 8.956 ± 0.534 | - | 8.742 ± 0.626 | - |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 8.553 ± 0.441 | - | 8.671 ± 0.082 | - |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 17.901 ± 0.039 | - | 14.931 ± 0.044 | - |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 5.554 ± 0.849 | - | 4.991 ± 0.300 | - |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 8.958 ± 0.195 | - | 17.882 ± 0.452 | - |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.915 ± 0.557 | - | 5.381 ± 0.208 | - |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 16.388 ± 0.231 | - | 26.354 ± 0.111 | - |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 5.078 ± 0.149 | - | 7.465 ± 0.347 | - |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 10.918 ± 0.380 | - | 13.783 ± 0.887 | - |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.945 ± 0.954 | - | 4.468 ± 0.840 | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 9.179 ± 0.258 | - | 8.838 ± 0.272 | - |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 8.665 ± 0.522 | - | 8.544 ± 0.464 | - |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 8.919 ± 0.386 | - | 8.987 ± 0.350 | - |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 8.622 ± 0.281 | - | 8.611 ± 0.328 | - |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 9.186 ± 0.299 | - | 8.869 ± 0.602 | - |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 8.906 ± 0.536 | - | 8.695 ± 0.572 | - |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 6.275 ± 0.214 | - | 6.477 ± 0.240 | - |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 5.755 ± 0.253 | - | 5.461 ± 0.089 | - |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 20.369 ± 0.229 | - | 44.608 ± 0.091 | - |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 7.205 ± 0.885 | - | 12.448 ± 0.952 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 6.154 ± 0.052 | - | 9.483 ± 0.504 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 6.391 ± 0.207 | - | 6.273 ± 0.720 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.781 ± 0.087 | - | 4.795 ± 0.147 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.811 ± 2.110 | - | 1.813 ± 0.025 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 28.500 ± 0.635 | - | 4.138 ± 0.246 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.586 ± 0.438 | - | 3.441 ± 0.332 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 18.478 ± 0.197 | - | 4.006 ± 0.164 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 4.942 ± 0.904 | - | 3.704 ± 0.342 | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 19.489 ± 0.260 | - | 22.650 ± 0.120 | - |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 5.504 ± 0.783 | - | 6.175 ± 0.275 | - |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 12.869 ± 0.441 | - | 12.786 ± 0.071 | - |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 5.717 ± 0.662 | - | 5.704 ± 0.588 | - |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 12.225 ± 0.665 | - | 9.414 ± 0.424 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 11.845 ± 0.867 | - | 8.993 ± 0.511 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 6.546 ± 0.495 | - | 6.507 ± 0.562 | - |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 5.874 ± 0.339 | - | 5.516 ± 0.521 | - |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 16.399 ± 0.037 | - | 18.066 ± 0.055 | - |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 5.146 ± 0.964 | - | 5.557 ± 0.573 | - |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 8.453 ± 0.047 | - | 8.442 ± 0.131 | - |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 5.679 ± 0.154 | - | 5.650 ± 0.328 | - |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 8.959 ± 0.388 | - | 8.878 ± 0.102 | - |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 9.031 ± 0.469 | - | 8.641 ± 0.466 | - |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 23.018 ± 0.148 | - | 42.987 ± 0.279 | - |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 6.341 ± 1.097 | - | 12.751 ± 1.175 | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 10.477 ± 0.092 | - | 0.353 ± 0.096 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 10.476 ± 0.087 | - | 0.283 ± 0.008 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304` | 10.989 ± 0.109 | - | 0.001 ± 0.000 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304` | 9.989 ± 0.454 | - | 0.001 ± 0.000 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 5.316 ± 0.030 | - | 0.530 ± 0.036 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 5.432 ± 0.150 | - | 0.496 ± 0.029 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 4.423 ± 1.097 | - | 0.181 ± 0.003 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 4.290 ± 1.111 | - | 0.180 ± 0.001 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 7.610 ± 0.107 | - | 0.566 ± 0.019 | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 7.567 ± 0.092 | - | 0.497 ± 0.008 | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 5.305 ± 0.070 | - | 0.415 ± 0.097 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 5.276 ± 0.108 | - | 0.253 ± 0.010 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 4.063 ± 0.061 | - | 0.279 ± 0.002 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 4.042 ± 0.077 | - | 0.283 ± 0.023 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304` | 7.385 ± 0.067 | - | 0.001 ± 0.000 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304` | 7.435 ± 0.104 | - | 0.001 ± 0.000 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 5.994 ± 1.433 | - | 14.393 ± 0.272 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 6.209 ± 1.277 | - | 7.510 ± 0.117 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 5.246 ± 0.167 | - | 4.046 ± 0.277 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 5.294 ± 0.117 | - | 3.957 ± 0.103 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.718 ± 0.097 | - | 5.253 ± 0.056 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.727 ± 0.087 | - | 5.288 ± 0.066 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.095 ± 0.119 | - | 4.693 ± 0.056 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 7.150 ± 0.099 | - | 4.721 ± 0.069 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 7.863 ± 0.063 | - | 0.283 ± 0.013 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 7.884 ± 0.066 | - | 0.282 ± 0.003 | - |
| cpu/linalg_uncovered | `full_piv_lu_solve` | f64 | 1 | `256x256,rhs=16` | 8.290 ± 0.053 | - | 0.197 ± 0.003 | - |
| cpu/linalg_uncovered | `full_piv_lu_solve` | f64 | 4 | `256x256,rhs=16` | 8.827 ± 0.299 | - | 0.193 ± 0.002 | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 5.094 ± 0.059 | - | 5.293 ± 0.107 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 4.933 ± 0.184 | - | 4.695 ± 0.085 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 23.335 ± 0.139 | - | 1.374 ± 0.013 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 8.383 ± 1.022 | - | 1.386 ± 0.036 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.451 ± 0.063 | - | 6.352 ± 0.105 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.617 ± 0.067 | - | 6.420 ± 0.132 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 5.212 ± 0.159 | - | 4.005 ± 0.154 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 5.323 ± 0.156 | - | 4.029 ± 0.097 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 6.187 ± 0.315 | - | 4.567 ± 0.118 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 6.868 ± 0.310 | - | 4.929 ± 0.215 | - |

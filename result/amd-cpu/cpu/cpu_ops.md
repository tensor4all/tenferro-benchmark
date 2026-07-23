# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Timestamp: `20260723_051422`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20260723_051422`.

- tenferro-rs commit: `68855c2b65b5adc42dccca9bac04fd136a8f14c8`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`

## Thread Environment

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

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.0`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- CSV: `data/results/amd-cpu/cpu/einsum/20260723_051422/cpu_ops_t4_20260723_051422.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20260723_051422/cpu_ops_t4_20260723_051422.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.427 ± 0.030 | 0.074 ± 0.001 | 0.562 ± 0.040 | 1.534 ± 0.022 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.539 ± 0.021 | 0.272 ± 0.006 | 1.941 ± 0.238 | 4.722 ± 0.391 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.487 ± 0.043 | 0.163 ± 0.018 | 0.723 ± 0.119 | 1.660 ± 0.030 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.744 ± 0.041 | 0.634 ± 0.038 | 2.474 ± 0.110 | 6.311 ± 0.536 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.393 ± 0.035 | 0.017 ± 0.000 | 0.188 ± 0.035 | 1.561 ± 0.030 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.389 ± 0.022 | 0.046 ± 0.001 | 0.315 ± 0.028 | 1.845 ± 0.365 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.389 ± 0.033 | 0.018 ± 0.001 | 0.374 ± 0.052 | 2.037 ± 0.147 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.399 ± 0.054 | 0.048 ± 0.001 | 0.888 ± 0.155 | 2.223 ± 0.167 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.401 ± 0.037 | 0.055 ± 0.001 | 0.357 ± 0.049 | 0.990 ± 0.015 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.477 ± 0.016 | 0.193 ± 0.003 | 1.186 ± 0.058 | 1.694 ± 0.020 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.428 ± 0.020 | 0.074 ± 0.004 | 0.489 ± 0.024 | 1.083 ± 0.029 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.530 ± 0.041 | 0.242 ± 0.007 | 1.629 ± 0.056 | 3.985 ± 0.027 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.473 ± 0.024 | 0.070 ± 0.002 | 0.681 ± 0.033 | 1.771 ± 0.046 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.550 ± 0.058 | 0.215 ± 0.003 | 2.140 ± 0.100 | 5.164 ± 0.382 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.460 ± 0.048 | 0.074 ± 0.002 | 0.841 ± 0.070 | 1.818 ± 0.029 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.589 ± 0.040 | 0.221 ± 0.004 | 2.757 ± 0.040 | 6.449 ± 0.902 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.460 ± 0.018 | 0.115 ± 0.002 | 0.362 ± 0.032 | 1.006 ± 0.022 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.629 ± 0.037 | 0.449 ± 0.018 | 1.158 ± 0.055 | 2.128 ± 0.099 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.552 ± 0.028 | 0.313 ± 0.035 | 0.617 ± 0.076 | 1.180 ± 0.026 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 1.068 ± 0.085 | 1.153 ± 0.007 | 1.878 ± 0.106 | 4.849 ± 0.054 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.657 ± 0.034 | 0.172 ± 0.012 | 0.387 ± 0.015 | 4.043 ± 2.158 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.702 ± 0.037 | 0.247 ± 0.014 | 0.580 ± 0.029 | 4.033 ± 2.000 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.682 ± 0.052 | 0.169 ± 0.050 | 0.596 ± 0.096 | 3.962 ± 1.579 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.734 ± 0.038 | 0.273 ± 0.026 | 1.302 ± 0.134 | 5.033 ± 0.786 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.877 ± 0.040 | 0.454 ± 0.052 | 0.884 ± 0.015 | 4.551 ± 2.641 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.985 ± 0.014 | 0.801 ± 0.430 | 2.423 ± 0.073 | 6.369 ± 0.837 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.862 ± 0.040 | 0.438 ± 0.048 | 1.035 ± 0.021 | 4.572 ± 2.453 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 1.029 ± 0.029 | 0.898 ± 1.408 | 2.963 ± 0.155 | 7.853 ± 1.428 |
| large | `eigh` | f64 | 4 | `64x64` | 1.276 ± 0.109 | 0.412 ± 0.012 | 2.524 ± 0.050 | 5.920 ± 0.185 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 11.336 ± 0.120 | 7.837 ± 0.846 | 18.600 ± 0.648 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 48.518 ± 2.604 | 29.920 ± 0.636 | 87.844 ± 2.279 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 10.801 ± 3.070 | 6.676 ± 0.137 | 18.249 ± 0.853 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 47.011 ± 3.105 | 22.631 ± 0.654 | 88.731 ± 1.645 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 8.054 ± 0.223 | 4.366 ± 0.261 | 6.904 ± 2.481 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 28.615 ± 9.108 | 27.721 ± 0.900 | 24.639 ± 3.808 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 7.233 ± 0.126 | 2.758 ± 0.119 | 8.158 ± 0.506 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 31.109 ± 4.542 | 12.960 ± 0.267 | 30.187 ± 8.425 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.440 ± 0.023 | 0.039 ± 0.004 | 3.746 ± 0.092 | 8.193 ± 0.310 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.731 ± 0.019 | 0.106 ± 0.005 | 3.897 ± 0.105 | 12.160 ± 0.455 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 10.232 ± 2.514 | 4.604 ± 0.028 | 10.496 ± 1.133 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 31.151 ± 4.750 | 20.848 ± 0.951 | 44.076 ± 1.611 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 10.596 ± 2.667 | 4.448 ± 0.054 | 11.245 ± 1.693 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 30.298 ± 1.471 | 20.361 ± 0.738 | 60.387 ± 4.090 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 1.282 ± 0.066 | 0.231 ± 0.007 | 2.187 ± 0.096 | 5.102 ± 2.469 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 2.180 ± 0.091 | 1.572 ± 0.032 | 3.128 ± 0.631 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 9.088 ± 0.705 | 3.417 ± 0.195 | 10.290 ± 0.657 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 1.327 ± 0.048 | 1.671 ± 0.017 | 4.978 ± 0.299 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 5.849 ± 1.381 | 4.933 ± 0.279 | 10.948 ± 0.121 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 2.489 ± 0.097 | 1.130 ± 0.051 | 3.156 ± 0.043 | 8.398 ± 2.372 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 24.284 ± 5.945 | 19.567 ± 0.228 | 39.784 ± 1.925 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 106.834 ± 0.890 | 85.594 ± 2.051 | 163.130 ± 6.305 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 23.724 ± 0.184 | 17.360 ± 0.168 | 37.190 ± 4.759 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 103.813 ± 1.362 | 70.554 ± 1.030 | 160.055 ± 5.213 |
| large | `matmul` | f64 | 4 | `128x128` | 0.637 ± 0.027 | 0.148 ± 0.017 | 13.791 ± 0.595 | 26.928 ± 1.639 |
| large | `matmul` | f64 | 4 | `256x256` | 1.874 ± 0.058 | 0.665 ± 0.017 | 56.079 ± 1.303 | 117.860 ± 3.023 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 4.589 ± 0.094 | 2.710 ± 0.145 | 210.709 ± 4.034 | 459.750 ± 9.754 |
| large | `qr` | f64 | 4 | `64x64` | 0.596 ± 0.057 | 0.155 ± 0.005 | 2.184 ± 0.032 | 4.749 ± 0.263 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.958 ± 0.043 | 0.065 ± 0.005 | 2.093 ± 0.087 | 4.992 ± 0.099 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.961 ± 0.035 | 0.091 ± 0.005 | 2.698 ± 0.046 | 5.458 ± 0.156 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.904 ± 0.036 | 0.143 ± 0.017 | 4.136 ± 0.094 | 7.963 ± 0.188 |
| large | `svd` | f64 | 4 | `64x64` | 1.583 ± 0.657 | 0.985 ± 0.017 | 3.047 ± 0.051 | 6.854 ± 2.026 |
| small | `eigh` | f64 | 4 | `2x2` | 0.435 ± 0.028 | 0.009 ± 0.000 | 0.112 ± 0.003 | 0.318 ± 0.016 |
| small | `eigh` | f64 | 4 | `4x4` | 0.401 ± 0.030 | 0.013 ± 0.000 | 0.129 ± 0.003 | 0.468 ± 0.014 |
| small | `eigh` | f64 | 4 | `8x8` | 0.408 ± 0.023 | 0.019 ± 0.000 | 0.163 ± 0.003 | 0.516 ± 0.033 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.434 ± 0.040 | 0.005 ± 0.001 | 0.127 ± 0.010 | 0.793 ± 0.072 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.394 ± 0.027 | 0.005 ± 0.001 | 0.135 ± 0.018 | 1.163 ± 0.026 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.376 ± 0.036 | 0.005 ± 0.001 | 0.185 ± 0.014 | 1.410 ± 0.091 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.112 ± 0.007 | 0.387 ± 0.038 | 1.703 ± 0.108 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.123 ± 0.007 | 0.387 ± 0.060 | 1.900 ± 0.215 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.131 ± 0.011 | 0.414 ± 0.013 | 1.635 ± 0.160 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.123 ± 0.013 | 0.374 ± 0.023 | 4.923 ± 1.305 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.135 ± 0.011 | 0.366 ± 0.019 | 4.174 ± 0.348 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.143 ± 0.008 | 0.396 ± 0.010 | 4.691 ± 0.564 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.115 ± 0.016 | 0.566 ± 0.011 | 2.408 ± 0.277 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.117 ± 0.009 | 0.562 ± 0.012 | 2.537 ± 0.177 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.122 ± 0.008 | 0.570 ± 0.008 | 2.721 ± 0.048 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.124 ± 0.005 | 0.501 ± 0.011 | 6.673 ± 0.454 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.124 ± 0.006 | 0.496 ± 0.009 | 7.560 ± 0.779 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.128 ± 0.003 | 0.505 ± 0.009 | 6.219 ± 0.199 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.649 ± 0.046 | 0.047 ± 0.005 | 0.256 ± 0.009 | 3.689 ± 1.695 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.648 ± 0.033 | 0.049 ± 0.003 | 0.272 ± 0.018 | 5.544 ± 0.981 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.660 ± 0.036 | 0.050 ± 0.005 | 0.318 ± 0.022 | 6.076 ± 1.049 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.284 ± 0.012 | 0.445 ± 0.012 | 3.412 ± 0.211 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.285 ± 0.012 | 0.454 ± 0.016 | 2.856 ± 0.352 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.278 ± 0.015 | 0.446 ± 0.011 | 2.464 ± 0.072 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.592 ± 0.027 | 0.473 ± 0.012 | 7.218 ± 0.584 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.612 ± 0.051 | 0.492 ± 0.012 | 6.738 ± 1.233 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.598 ± 0.037 | 0.485 ± 0.013 | 6.661 ± 1.051 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.830 ± 0.015 | 0.149 ± 0.008 | 0.368 ± 0.009 | 6.397 ± 0.632 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.817 ± 0.050 | 0.148 ± 0.006 | 0.373 ± 0.030 | 6.437 ± 1.328 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.854 ± 0.019 | 0.144 ± 0.009 | 0.404 ± 0.018 | 5.671 ± 0.659 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.073 ± 0.007 | 0.781 ± 0.043 | 2.392 ± 0.468 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.074 ± 0.006 | 0.780 ± 0.039 | 2.195 ± 0.380 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.075 ± 0.003 | 0.830 ± 0.037 | 1.752 ± 0.096 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.104 ± 0.003 | 0.507 ± 0.042 | 5.858 ± 0.334 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.106 ± 0.004 | 0.532 ± 0.018 | 5.438 ± 0.732 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.107 ± 0.008 | 0.533 ± 0.023 | 3.730 ± 0.214 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 1.234 ± 0.040 | 0.082 ± 0.006 | 0.270 ± 0.007 | 4.894 ± 0.094 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 1.267 ± 0.033 | 0.091 ± 0.005 | 0.275 ± 0.027 | 6.794 ± 0.921 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 1.272 ± 0.040 | 0.113 ± 0.010 | 0.333 ± 0.023 | 6.095 ± 0.603 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.069 ± 0.002 | 0.525 ± 0.014 | 1.950 ± 0.145 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.079 ± 0.003 | 0.522 ± 0.031 | 1.478 ± 0.096 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.097 ± 0.011 | 0.536 ± 0.024 | 1.462 ± 0.092 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.084 ± 0.002 | 0.408 ± 0.013 | 6.237 ± 1.208 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.091 ± 0.004 | 0.417 ± 0.016 | 4.825 ± 0.512 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.109 ± 0.013 | 0.443 ± 0.027 | 4.552 ± 0.238 |
| small | `matmul` | f64 | 4 | `2x2` | 0.440 ± 0.041 | 0.005 ± 0.000 | 0.091 ± 0.005 | 0.433 ± 0.041 |
| small | `matmul` | f64 | 4 | `4x4` | 0.380 ± 0.028 | 0.005 ± 0.001 | 0.094 ± 0.005 | 0.677 ± 0.015 |
| small | `matmul` | f64 | 4 | `8x8` | 0.375 ± 0.027 | 0.005 ± 0.000 | 0.142 ± 0.015 | 0.789 ± 0.016 |
| small | `qr` | f64 | 4 | `2x2` | 0.429 ± 0.017 | 0.008 ± 0.000 | 0.086 ± 0.004 | 0.230 ± 0.013 |
| small | `qr` | f64 | 4 | `4x4` | 0.399 ± 0.024 | 0.009 ± 0.005 | 0.095 ± 0.002 | 0.422 ± 0.009 |
| small | `qr` | f64 | 4 | `8x8` | 0.391 ± 0.031 | 0.010 ± 0.000 | 0.119 ± 0.002 | 0.486 ± 0.016 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.471 ± 0.060 | 0.018 ± 0.001 | 0.176 ± 0.006 | 0.523 ± 0.017 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.458 ± 0.029 | 0.019 ± 0.001 | 0.176 ± 0.003 | 0.546 ± 0.028 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.436 ± 0.031 | 0.018 ± 0.001 | 0.172 ± 0.020 | 0.731 ± 0.017 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.455 ± 0.032 | 0.020 ± 0.001 | 0.182 ± 0.020 | 0.709 ± 0.030 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.471 ± 0.077 | 0.019 ± 0.001 | 0.206 ± 0.016 | 0.743 ± 0.102 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.506 ± 0.037 | 0.020 ± 0.001 | 0.227 ± 0.027 | 0.742 ± 0.092 |
| small | `svd` | f64 | 4 | `2x2` | 0.432 ± 0.028 | 0.011 ± 0.000 | 0.105 ± 0.003 | 0.226 ± 0.031 |
| small | `svd` | f64 | 4 | `4x4` | 0.417 ± 0.014 | 0.018 ± 0.000 | 0.117 ± 0.003 | 0.443 ± 0.017 |
| small | `svd` | f64 | 4 | `8x8` | 0.440 ± 0.044 | 0.033 ± 0.001 | 0.153 ± 0.018 | 0.530 ± 0.015 |

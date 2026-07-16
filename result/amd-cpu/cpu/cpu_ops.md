# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Timestamp: `20260716_081717`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20260716_081717`.

- tenferro-rs commit: `8c267c636537d3d407925b61a2000f9bbe07e6a1`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-134-generic-x86_64-with-glibc2.39`

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

- tenferro-rs features: `system-openblas`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `openblas`
- BLAS version: `OpenBLAS 0.3.26`
- BLAS root: `/opt/openblas`
- BLAS library: `/opt/openblas/lib/libopenblas.so`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- CSV: `data/results/amd-cpu/cpu/einsum/20260716_081717/cpu_ops_t4_20260716_081717.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20260716_081717/cpu_ops_t4_20260716_081717.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.199 ± 0.018 | 0.025 ± 0.000 | 0.565 ± 0.013 | 1.865 ± 0.019 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.280 ± 0.016 | 0.091 ± 0.011 | 1.898 ± 0.116 | 5.353 ± 0.025 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.225 ± 0.010 | 0.067 ± 0.017 | 0.706 ± 0.020 | 2.224 ± 0.023 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.402 ± 0.008 | 0.241 ± 0.043 | 2.442 ± 0.022 | 7.040 ± 0.121 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.192 ± 0.030 | 0.022 ± 0.006 | 0.175 ± 0.020 | 1.549 ± 0.024 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.191 ± 0.026 | 0.030 ± 0.004 | 0.306 ± 0.022 | 1.770 ± 0.037 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.180 ± 0.031 | 0.017 ± 0.011 | 0.309 ± 0.015 | 2.166 ± 0.245 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.198 ± 0.017 | 0.053 ± 0.006 | 0.854 ± 0.032 | 3.288 ± 0.046 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.186 ± 0.010 | 0.028 ± 0.003 | 0.351 ± 0.017 | 1.288 ± 0.027 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.278 ± 0.007 | 0.106 ± 0.011 | 1.202 ± 0.045 | 2.426 ± 0.044 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.191 ± 0.011 | 0.048 ± 0.003 | 0.466 ± 0.025 | 1.399 ± 0.028 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.298 ± 0.010 | 0.163 ± 0.015 | 1.616 ± 0.026 | 4.595 ± 0.089 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.211 ± 0.011 | 0.031 ± 0.011 | 0.684 ± 0.021 | 1.804 ± 0.037 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.298 ± 0.012 | 0.086 ± 0.019 | 2.187 ± 0.141 | 6.043 ± 0.299 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.189 ± 0.015 | 0.036 ± 0.002 | 0.802 ± 0.035 | 2.060 ± 0.034 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.277 ± 0.006 | 0.129 ± 0.036 | 2.822 ± 0.164 | 7.150 ± 0.323 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.242 ± 0.012 | 0.064 ± 0.010 | 0.372 ± 0.014 | 1.158 ± 0.015 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.435 ± 0.006 | 0.229 ± 0.037 | 1.186 ± 0.017 | 2.876 ± 0.058 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.288 ± 0.006 | 0.137 ± 0.008 | 0.545 ± 0.014 | 1.512 ± 0.028 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.659 ± 0.024 | 0.509 ± 0.024 | 1.818 ± 0.012 | 5.378 ± 0.160 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.470 ± 0.066 | 0.049 ± 0.005 | 0.378 ± 0.011 | 6.210 ± 1.394 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.570 ± 0.019 | 0.069 ± 0.006 | 0.541 ± 0.013 | 6.487 ± 1.836 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.489 ± 0.029 | 0.051 ± 0.006 | 0.541 ± 0.009 | 6.015 ± 1.991 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.543 ± 0.031 | 0.089 ± 0.005 | 1.133 ± 0.014 | 6.421 ± 2.646 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.612 ± 0.014 | 0.276 ± 0.054 | 0.880 ± 0.020 | 6.836 ± 1.290 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.807 ± 0.072 | 0.348 ± 0.006 | 2.450 ± 0.072 | 8.284 ± 3.935 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.650 ± 0.032 | 0.175 ± 0.009 | 1.019 ± 0.033 | 7.132 ± 1.546 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.774 ± 0.117 | 0.376 ± 0.005 | 2.921 ± 0.046 | 8.186 ± 2.801 |
| large | `eigh` | f64 | 4 | `64x64` | 1.386 ± 0.039 | 0.764 ± 0.011 | 2.293 ± 0.117 | 6.753 ± 0.082 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 14.349 ± 0.314 | 7.679 ± 0.228 | 29.508 ± 0.288 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 54.466 ± 0.378 | 27.081 ± 0.256 | 87.934 ± 0.916 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 13.529 ± 0.174 | 6.663 ± 0.112 | 28.565 ± 0.634 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 53.019 ± 0.855 | 20.258 ± 0.277 | 87.883 ± 2.625 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 6.396 ± 0.425 | 4.456 ± 0.006 | 7.057 ± 2.102 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 25.893 ± 0.493 | 23.894 ± 0.414 | 23.766 ± 2.318 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 5.733 ± 0.119 | 2.919 ± 0.106 | 10.291 ± 1.760 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 24.485 ± 0.034 | 11.530 ± 0.047 | 27.751 ± 5.654 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.280 ± 0.044 | 0.054 ± 0.036 | 3.414 ± 0.044 | 9.013 ± 1.380 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.644 ± 0.009 | 0.121 ± 0.017 | 3.505 ± 0.105 | 7.775 ± 2.881 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 9.085 ± 0.635 | 4.366 ± 0.020 | 12.854 ± 0.993 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 36.163 ± 0.597 | 18.448 ± 0.393 | 43.595 ± 0.802 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 9.266 ± 0.140 | 4.231 ± 0.015 | 13.872 ± 1.407 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 34.074 ± 0.485 | 18.301 ± 0.204 | 56.605 ± 10.465 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 1.068 ± 0.024 | 0.201 ± 0.012 | 1.942 ± 0.017 | 6.607 ± 2.116 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 1.594 ± 0.021 | 1.597 ± 0.030 | 4.157 ± 0.090 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 6.061 ± 0.049 | 3.159 ± 0.180 | 10.320 ± 0.164 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 1.154 ± 0.056 | 1.696 ± 0.040 | 5.579 ± 0.187 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 4.334 ± 0.061 | 4.531 ± 0.121 | 11.176 ± 1.702 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 2.771 ± 0.020 | 1.399 ± 0.041 | 2.829 ± 0.051 | 8.949 ± 1.772 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 22.354 ± 0.325 | 17.764 ± 0.317 | 52.812 ± 1.935 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 96.461 ± 0.638 | 80.634 ± 6.353 | 164.388 ± 1.464 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 21.732 ± 0.359 | 15.844 ± 0.093 | 50.359 ± 2.140 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 94.478 ± 0.455 | 63.366 ± 0.479 | 158.732 ± 6.097 |
| large | `matmul` | f64 | 4 | `128x128` | 0.309 ± 0.056 | 0.187 ± 0.011 | 11.659 ± 0.104 | 18.166 ± 0.294 |
| large | `matmul` | f64 | 4 | `256x256` | 1.420 ± 0.026 | 0.828 ± 0.049 | 45.151 ± 0.735 | 69.739 ± 2.115 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 5.045 ± 0.043 | 3.378 ± 0.063 | 178.060 ± 4.801 | 180.622 ± 2.512 |
| large | `qr` | f64 | 4 | `64x64` | 0.358 ± 0.026 | 0.169 ± 0.013 | 1.953 ± 0.051 | 5.244 ± 0.063 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.607 ± 0.015 | 0.074 ± 0.002 | 1.867 ± 0.213 | 5.643 ± 0.066 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.622 ± 0.007 | 0.078 ± 0.009 | 2.439 ± 0.034 | 6.502 ± 0.135 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.657 ± 0.019 | 0.150 ± 0.045 | 3.751 ± 0.074 | 9.422 ± 1.303 |
| large | `svd` | f64 | 4 | `64x64` | 1.549 ± 0.036 | 1.330 ± 0.078 | 2.736 ± 0.033 | 6.918 ± 1.414 |
| small | `eigh` | f64 | 4 | `2x2` | 0.158 ± 0.010 | 0.004 ± 0.000 | 0.114 ± 0.020 | 0.419 ± 0.012 |
| small | `eigh` | f64 | 4 | `4x4` | 0.167 ± 0.014 | 0.008 ± 0.000 | 0.123 ± 0.002 | 0.689 ± 0.023 |
| small | `eigh` | f64 | 4 | `8x8` | 0.169 ± 0.012 | 0.016 ± 0.000 | 0.152 ± 0.002 | 0.779 ± 0.008 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.183 ± 0.038 | 0.011 ± 0.002 | 0.128 ± 0.005 | 0.585 ± 0.032 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.186 ± 0.012 | 0.013 ± 0.024 | 0.129 ± 0.014 | 1.360 ± 0.015 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.194 ± 0.030 | 0.013 ± 0.009 | 0.171 ± 0.001 | 1.793 ± 0.026 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.064 ± 0.009 | 0.336 ± 0.028 | 1.377 ± 0.026 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.075 ± 0.005 | 0.357 ± 0.018 | 1.480 ± 0.026 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.083 ± 0.006 | 0.368 ± 0.010 | 1.508 ± 0.029 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.071 ± 0.017 | 0.321 ± 0.014 | 3.395 ± 0.096 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.076 ± 0.007 | 0.333 ± 0.011 | 3.414 ± 0.100 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.091 ± 0.019 | 0.343 ± 0.009 | 4.013 ± 0.954 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.080 ± 0.012 | 0.508 ± 0.011 | 1.925 ± 0.054 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.085 ± 0.017 | 0.510 ± 0.013 | 2.097 ± 0.049 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.090 ± 0.005 | 0.514 ± 0.017 | 3.026 ± 0.037 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.080 ± 0.008 | 0.463 ± 0.024 | 5.216 ± 0.159 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.080 ± 0.006 | 0.469 ± 0.026 | 5.593 ± 0.327 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.084 ± 0.002 | 0.462 ± 0.021 | 6.917 ± 1.631 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.485 ± 0.025 | 0.042 ± 0.009 | 0.229 ± 0.013 | 5.400 ± 0.669 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.481 ± 0.043 | 0.042 ± 0.003 | 0.241 ± 0.023 | 5.230 ± 0.745 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.473 ± 0.026 | 0.046 ± 0.011 | 0.278 ± 0.012 | 5.277 ± 0.910 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.187 ± 0.015 | 0.398 ± 0.016 | 3.070 ± 0.020 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.208 ± 0.011 | 0.399 ± 0.009 | 2.548 ± 0.054 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.215 ± 0.011 | 0.407 ± 0.015 | 2.255 ± 0.137 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.480 ± 0.006 | 0.443 ± 0.043 | 6.795 ± 1.329 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.475 ± 0.015 | 0.442 ± 0.026 | 6.472 ± 0.999 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.483 ± 0.029 | 0.446 ± 0.023 | 4.715 ± 1.290 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.525 ± 0.031 | 0.094 ± 0.008 | 0.332 ± 0.037 | 5.069 ± 1.261 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.574 ± 0.016 | 0.100 ± 0.012 | 0.336 ± 0.013 | 5.893 ± 0.673 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.588 ± 0.066 | 0.103 ± 0.009 | 0.362 ± 0.012 | 5.816 ± 0.493 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.050 ± 0.007 | 0.694 ± 0.010 | 2.275 ± 0.052 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.052 ± 0.006 | 0.696 ± 0.091 | 2.176 ± 0.130 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.057 ± 0.003 | 0.697 ± 0.058 | 2.308 ± 0.053 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.068 ± 0.007 | 0.453 ± 0.009 | 4.833 ± 0.802 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.074 ± 0.013 | 0.458 ± 0.015 | 4.971 ± 0.734 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.080 ± 0.011 | 0.459 ± 0.013 | 5.022 ± 0.101 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.963 ± 0.041 | 0.053 ± 0.008 | 0.243 ± 0.032 | 5.519 ± 0.557 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 1.034 ± 0.060 | 0.063 ± 0.002 | 0.252 ± 0.016 | 5.467 ± 0.533 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.990 ± 0.015 | 0.079 ± 0.003 | 0.290 ± 0.010 | 5.582 ± 0.561 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.042 ± 0.003 | 0.428 ± 0.011 | 2.059 ± 0.023 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.058 ± 0.001 | 0.436 ± 0.011 | 1.942 ± 0.016 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.078 ± 0.008 | 0.454 ± 0.009 | 1.861 ± 0.015 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.053 ± 0.003 | 0.361 ± 0.015 | 4.868 ± 0.210 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.063 ± 0.003 | 0.354 ± 0.007 | 4.950 ± 0.339 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.079 ± 0.010 | 0.382 ± 0.015 | 4.959 ± 0.181 |
| small | `matmul` | f64 | 4 | `2x2` | 0.187 ± 0.022 | 0.018 ± 0.022 | 0.091 ± 0.003 | 0.329 ± 0.020 |
| small | `matmul` | f64 | 4 | `4x4` | 0.165 ± 0.006 | 0.013 ± 0.006 | 0.095 ± 0.002 | 0.741 ± 0.020 |
| small | `matmul` | f64 | 4 | `8x8` | 0.186 ± 0.009 | 0.017 ± 0.020 | 0.129 ± 0.004 | 1.143 ± 0.026 |
| small | `qr` | f64 | 4 | `2x2` | 0.154 ± 0.004 | 0.005 ± 0.000 | 0.090 ± 0.004 | 0.276 ± 0.051 |
| small | `qr` | f64 | 4 | `4x4` | 0.159 ± 0.009 | 0.006 ± 0.001 | 0.092 ± 0.001 | 0.581 ± 0.026 |
| small | `qr` | f64 | 4 | `8x8` | 0.167 ± 0.011 | 0.008 ± 0.000 | 0.116 ± 0.011 | 0.677 ± 0.019 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.170 ± 0.015 | 0.008 ± 0.001 | 0.180 ± 0.022 | 0.659 ± 0.021 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.171 ± 0.011 | 0.013 ± 0.001 | 0.168 ± 0.014 | 0.660 ± 0.024 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.165 ± 0.003 | 0.009 ± 0.001 | 0.187 ± 0.014 | 0.802 ± 0.020 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.178 ± 0.007 | 0.014 ± 0.000 | 0.189 ± 0.039 | 0.753 ± 0.022 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.173 ± 0.021 | 0.010 ± 0.001 | 0.207 ± 0.024 | 0.857 ± 0.011 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.173 ± 0.021 | 0.016 ± 0.001 | 0.201 ± 0.009 | 0.783 ± 0.020 |
| small | `svd` | f64 | 4 | `2x2` | 0.163 ± 0.007 | 0.008 ± 0.001 | 0.101 ± 0.008 | 0.268 ± 0.038 |
| small | `svd` | f64 | 4 | `4x4` | 0.172 ± 0.007 | 0.012 ± 0.001 | 0.108 ± 0.002 | 0.556 ± 0.022 |
| small | `svd` | f64 | 4 | `8x8` | 0.190 ± 0.011 | 0.026 ± 0.001 | 0.142 ± 0.001 | 0.678 ± 0.024 |

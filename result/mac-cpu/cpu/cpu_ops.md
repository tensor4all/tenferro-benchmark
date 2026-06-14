# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260614_142926`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260614_142926`.

- tenferro-rs commit: `db1990549801351308b631aef1bbca292d11a457`

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
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

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

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 4

- CSV: `data/results/mac-cpu/cpu/einsum/20260614_142926/cpu_ops_t4_20260614_142926.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260614_142926/cpu_ops_t4_20260614_142926.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.040 ± 0.004 | 0.006 ± 0.000 | 0.088 ± 0.002 | 0.117 ± 0.011 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.052 ± 0.001 | 0.021 ± 0.000 | 0.329 ± 0.004 | 0.374 ± 0.006 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.048 ± 0.008 | 0.017 ± 0.000 | 0.128 ± 0.001 | 0.158 ± 0.006 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.104 ± 0.002 | 0.063 ± 0.001 | 0.490 ± 0.007 | 0.580 ± 0.747 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.041 ± 0.005 | 0.007 ± 0.000 | 0.034 ± 0.009 | 0.138 ± 0.007 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.037 ± 0.003 | 0.009 ± 0.008 | 0.083 ± 0.007 | 0.198 ± 0.017 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.034 ± 0.011 | 0.007 ± 0.003 | 0.083 ± 0.002 | 0.197 ± 0.008 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.039 ± 0.007 | 0.009 ± 0.013 | 0.278 ± 0.009 | 0.412 ± 0.006 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.037 ± 0.003 | 0.008 ± 0.000 | 0.081 ± 0.006 | 0.087 ± 0.005 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.059 ± 0.002 | 0.028 ± 0.000 | 0.237 ± 0.008 | 0.245 ± 0.008 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.042 ± 0.005 | 0.011 ± 0.000 | 0.109 ± 0.004 | 0.127 ± 0.060 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.077 ± 0.011 | 0.041 ± 0.000 | 0.360 ± 0.006 | 0.412 ± 0.041 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.043 ± 0.002 | 0.010 ± 0.000 | 0.117 ± 0.003 | 0.147 ± 0.022 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.070 ± 0.001 | 0.032 ± 0.000 | 0.362 ± 0.011 | 0.417 ± 0.019 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.042 ± 0.002 | 0.011 ± 0.000 | 0.149 ± 0.005 | 0.178 ± 0.011 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.075 ± 0.005 | 0.035 ± 0.006 | 0.501 ± 0.010 | 0.535 ± 0.010 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.044 ± 0.003 | 0.011 ± 0.000 | 0.068 ± 0.002 | 0.105 ± 0.005 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.075 ± 0.004 | 0.043 ± 0.002 | 0.232 ± 0.002 | 0.338 ± 0.088 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.064 ± 0.002 | 0.031 ± 0.000 | 0.116 ± 0.002 | 0.158 ± 0.004 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.158 ± 0.001 | 0.119 ± 0.000 | 0.427 ± 0.014 | 0.551 ± 0.114 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.113 ± 0.020 | 0.014 ± 0.003 | 0.063 ± 0.002 | 0.606 ± 0.110 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.112 ± 0.009 | 0.022 ± 0.004 | 0.112 ± 0.003 | 0.620 ± 0.031 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.106 ± 0.029 | 0.014 ± 0.002 | 0.114 ± 0.002 | 0.649 ± 0.139 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.117 ± 0.016 | 0.026 ± 0.001 | 0.309 ± 0.004 | 0.817 ± 0.031 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.165 ± 0.116 | 0.119 ± 0.028 | 0.143 ± 0.006 | 0.606 ± 0.225 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.212 ± 0.086 | 0.180 ± 0.011 | 0.402 ± 0.034 | 0.825 ± 0.233 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.133 ± 0.017 | 0.114 ± 0.045 | 0.175 ± 0.007 | 0.650 ± 0.167 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.206 ± 0.025 | 0.231 ± 0.095 | 0.523 ± 0.008 | 1.143 ± 0.505 |
| large | `eigh` | f64 | 4 | `64x64` | 0.423 ± 0.012 | 0.309 ± 0.004 | 0.787 ± 0.022 | 0.840 ± 0.109 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 10.928 ± 0.313 | 3.316 ± 2.032 | 3.778 ± 0.230 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 69.533 ± 1.380 | 15.116 ± 0.281 | 17.923 ± 0.593 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 11.128 ± 0.310 | 3.410 ± 0.588 | 3.570 ± 0.194 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 72.390 ± 2.796 | 13.694 ± 0.160 | 17.156 ± 0.189 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 1.163 ± 0.076 | 1.885 ± 0.864 | 1.135 ± 0.115 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 6.191 ± 0.512 | 8.421 ± 0.449 | 5.632 ± 1.038 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 1.923 ± 0.141 | 0.805 ± 0.070 | 1.376 ± 0.212 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 10.868 ± 2.034 | 4.746 ± 0.447 | 5.766 ± 0.452 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.056 ± 0.002 | 0.013 ± 0.002 | 1.098 ± 0.015 | 1.177 ± 0.084 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.125 ± 0.005 | 0.025 ± 0.000 | 1.126 ± 0.016 | 1.499 ± 0.036 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 1.870 ± 0.072 | 1.874 ± 0.022 | 2.325 ± 0.070 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 9.016 ± 0.402 | 9.387 ± 0.752 | 9.923 ± 0.271 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 2.543 ± 0.112 | 1.805 ± 0.087 | 2.289 ± 0.168 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 12.691 ± 5.814 | 8.740 ± 0.253 | 10.767 ± 0.365 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.218 ± 0.143 | 0.108 ± 0.046 | 0.628 ± 0.016 | 1.056 ± 0.032 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.286 ± 0.031 | 0.440 ± 0.057 | 0.448 ± 0.052 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 1.381 ± 0.037 | 1.317 ± 0.012 | 1.405 ± 0.208 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.317 ± 0.042 | 0.468 ± 0.013 | 0.565 ± 0.070 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 1.444 ± 0.055 | 2.037 ± 0.095 | 1.415 ± 0.268 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | panic | 0.647 ± 0.002 | 0.897 ± 0.006 | 1.087 ± 0.053 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 63.845 ± 0.613 | 5.777 ± 0.061 | 6.275 ± 0.199 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 613.462 ± 1.333 | 41.603 ± 29.579 | 28.888 ± 0.553 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 63.992 ± 0.916 | 5.196 ± 0.108 | 5.952 ± 0.066 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 614.640 ± 2.705 | 31.808 ± 0.382 | 34.753 ± 0.270 |
| large | `matmul` | f64 | 4 | `128x128` | 0.084 ± 0.019 | 0.023 ± 0.001 | 4.328 ± 0.069 | 4.437 ± 0.097 |
| large | `matmul` | f64 | 4 | `256x256` | 0.201 ± 0.001 | 0.127 ± 0.005 | 17.604 ± 0.263 | 17.619 ± 0.790 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 0.648 ± 0.019 | 0.516 ± 0.004 | 68.750 ± 0.861 | 69.939 ± 3.639 |
| large | `qr` | f64 | 4 | `64x64` | 0.121 ± 0.006 | 0.083 ± 0.001 | 0.667 ± 0.029 | 0.695 ± 0.045 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.117 ± 0.005 | 0.020 ± 0.002 | 0.597 ± 0.006 | 0.639 ± 0.015 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.118 ± 0.000 | 0.021 ± 0.001 | 0.722 ± 0.052 | 0.778 ± 0.150 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.130 ± 0.004 | 0.034 ± 0.005 | 1.141 ± 0.011 | 1.192 ± 0.042 |
| large | `svd` | f64 | 4 | `64x64` | 0.737 ± 0.081 | 0.605 ± 0.028 | 0.869 ± 0.017 | 0.949 ± 0.093 |
| small | `eigh` | f64 | 4 | `2x2` | 0.051 ± 0.020 | 0.001 ± 0.000 | 0.012 ± 0.002 | 0.038 ± 0.009 |
| small | `eigh` | f64 | 4 | `4x4` | 0.047 ± 0.003 | 0.002 ± 0.000 | 0.015 ± 0.002 | 0.038 ± 0.013 |
| small | `eigh` | f64 | 4 | `8x8` | 0.045 ± 0.005 | 0.003 ± 0.000 | 0.022 ± 0.001 | 0.047 ± 0.008 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.066 ± 0.003 | 0.006 ± 0.002 | 0.015 ± 0.001 | 0.121 ± 0.010 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.074 ± 0.008 | 0.005 ± 0.001 | 0.018 ± 0.001 | 0.122 ± 0.002 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.058 ± 0.006 | 0.005 ± 0.001 | 0.030 ± 0.001 | 0.137 ± 0.006 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.036 ± 0.015 | 0.046 ± 0.004 | 0.162 ± 0.007 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.025 ± 0.032 | 0.046 ± 0.003 | 0.170 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.029 ± 0.009 | 0.053 ± 0.002 | 0.164 ± 0.006 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.043 ± 0.004 | 0.041 ± 0.001 | 0.380 ± 0.008 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.042 ± 0.005 | 0.040 ± 0.002 | 0.409 ± 0.098 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.050 ± 0.004 | 0.049 ± 0.019 | 0.379 ± 0.010 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.043 ± 0.010 | 0.116 ± 0.009 | 0.236 ± 0.014 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.046 ± 0.017 | 0.135 ± 0.009 | 0.230 ± 0.005 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.042 ± 0.008 | 0.139 ± 0.015 | 0.237 ± 0.007 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.056 ± 0.012 | 0.092 ± 0.008 | 0.606 ± 0.017 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.056 ± 0.005 | 0.105 ± 0.000 | 0.617 ± 0.285 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.061 ± 0.007 | 0.115 ± 0.009 | 0.628 ± 0.019 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.280 ± 0.050 | 0.012 ± 0.004 | 0.026 ± 0.003 | 0.435 ± 0.017 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.183 ± 0.008 | 0.010 ± 0.003 | 0.029 ± 0.001 | 0.428 ± 0.016 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.155 ± 0.054 | 0.013 ± 0.001 | 0.047 ± 0.002 | 0.478 ± 0.273 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.072 ± 0.039 | 0.079 ± 0.029 | 0.252 ± 0.061 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.061 ± 0.013 | 0.086 ± 0.020 | 0.239 ± 0.006 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.066 ± 0.025 | 0.088 ± 0.011 | 0.248 ± 0.051 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.114 ± 0.014 | 0.082 ± 0.021 | 0.594 ± 0.096 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.123 ± 0.015 | 0.087 ± 0.008 | 0.579 ± 0.024 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.122 ± 0.015 | 0.086 ± 0.007 | 0.574 ± 0.016 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.251 ± 0.027 | 0.060 ± 0.005 | 0.038 ± 0.001 | 0.466 ± 0.021 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.200 ± 0.045 | 0.051 ± 0.012 | 0.045 ± 0.004 | 0.492 ± 0.038 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.172 ± 0.015 | 0.052 ± 0.008 | 0.047 ± 0.001 | 0.590 ± 0.380 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.022 ± 0.010 | 0.121 ± 0.005 | 0.198 ± 0.007 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.034 ± 0.001 | 0.118 ± 0.004 | 0.193 ± 0.019 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.032 ± 0.036 | 0.118 ± 0.003 | 0.190 ± 0.008 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.037 ± 0.009 | 0.056 ± 0.003 | 0.407 ± 0.036 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.038 ± 0.003 | 0.056 ± 0.003 | 0.416 ± 0.013 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.038 ± 0.010 | 0.058 ± 0.003 | 0.422 ± 0.037 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | panic | 0.032 ± 0.002 | 0.025 ± 0.001 | 0.478 ± 0.060 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | panic | 0.031 ± 0.007 | 0.032 ± 0.019 | 0.512 ± 0.180 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | panic | 0.039 ± 0.003 | 0.039 ± 0.001 | 0.489 ± 0.031 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.014 ± 0.013 | 0.066 ± 0.006 | 0.173 ± 0.011 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.017 ± 0.012 | 0.066 ± 0.003 | 0.170 ± 0.011 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.023 ± 0.001 | 0.072 ± 0.002 | 0.170 ± 0.005 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.028 ± 0.002 | 0.046 ± 0.007 | 0.436 ± 0.115 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.029 ± 0.006 | 0.045 ± 0.002 | 0.457 ± 0.244 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.039 ± 0.024 | 0.048 ± 0.003 | 0.498 ± 0.247 |
| small | `matmul` | f64 | 4 | `2x2` | 0.090 ± 0.016 | 0.007 ± 0.002 | 0.011 ± 0.002 | 0.049 ± 0.007 |
| small | `matmul` | f64 | 4 | `4x4` | 0.067 ± 0.006 | 0.006 ± 0.001 | 0.013 ± 0.001 | 0.051 ± 0.006 |
| small | `matmul` | f64 | 4 | `8x8` | 0.055 ± 0.009 | 0.006 ± 0.001 | 0.025 ± 0.000 | 0.074 ± 0.005 |
| small | `qr` | f64 | 4 | `2x2` | 0.049 ± 0.008 | 0.001 ± 0.000 | 0.029 ± 0.007 | 0.035 ± 0.006 |
| small | `qr` | f64 | 4 | `4x4` | 0.045 ± 0.009 | 0.001 ± 0.000 | 0.027 ± 0.007 | 0.035 ± 0.008 |
| small | `qr` | f64 | 4 | `8x8` | 0.043 ± 0.004 | 0.002 ± 0.000 | 0.032 ± 0.002 | 0.041 ± 0.005 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.058 ± 0.020 | 0.002 ± 0.000 | 0.019 ± 0.001 | 0.056 ± 0.007 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.068 ± 0.002 | 0.002 ± 0.000 | 0.019 ± 0.000 | 0.055 ± 0.005 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.054 ± 0.003 | 0.003 ± 0.000 | 0.022 ± 0.001 | 0.058 ± 0.008 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.049 ± 0.004 | 0.002 ± 0.000 | 0.022 ± 0.000 | 0.061 ± 0.008 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.045 ± 0.001 | 0.003 ± 0.000 | 0.028 ± 0.001 | 0.065 ± 0.007 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.050 ± 0.009 | 0.003 ± 0.000 | 0.031 ± 0.000 | 0.069 ± 0.006 |
| small | `svd` | f64 | 4 | `2x2` | 0.058 ± 0.005 | 0.001 ± 0.000 | 0.011 ± 0.002 | 0.042 ± 0.034 |
| small | `svd` | f64 | 4 | `4x4` | 0.051 ± 0.003 | 0.003 ± 0.000 | 0.013 ± 0.000 | 0.039 ± 0.008 |
| small | `svd` | f64 | 4 | `8x8` | 0.052 ± 0.005 | 0.007 ± 0.000 | 0.023 ± 0.000 | 0.051 ± 0.010 |

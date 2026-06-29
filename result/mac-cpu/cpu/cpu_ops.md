# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260630_071006`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260630_071006`.

- tenferro-rs commit: `3b4136ca3a3d53cbbdd5096954a470d9407ad25e`

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
- Python platform: `macOS-26.5.1-arm64-arm-64bit`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260630_071006/cpu_ops_t4_20260630_071006.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260630_071006/cpu_ops_t4_20260630_071006.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.047 ± 0.009 | 0.006 ± 0.000 | 0.073 ± 0.004 | 0.095 ± 0.008 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.062 ± 0.005 | 0.020 ± 0.000 | 0.269 ± 0.004 | 0.334 ± 0.018 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.054 ± 0.004 | 0.016 ± 0.001 | 0.106 ± 0.005 | 0.147 ± 0.026 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.111 ± 0.013 | 0.056 ± 0.000 | 0.416 ± 0.009 | 0.466 ± 0.024 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.053 ± 0.005 | 0.009 ± 0.002 | 0.025 ± 0.001 | 0.104 ± 0.003 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.055 ± 0.004 | 0.012 ± 0.003 | 0.061 ± 0.002 | 0.159 ± 0.010 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.049 ± 0.004 | 0.010 ± 0.001 | 0.063 ± 0.003 | 0.160 ± 0.006 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.051 ± 0.010 | 0.011 ± 0.003 | 0.220 ± 0.004 | 0.348 ± 0.033 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.048 ± 0.011 | 0.007 ± 0.001 | 0.069 ± 0.009 | 0.094 ± 0.026 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.069 ± 0.008 | 0.026 ± 0.000 | 0.190 ± 0.003 | 0.214 ± 0.023 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.050 ± 0.006 | 0.009 ± 0.000 | 0.100 ± 0.007 | 0.104 ± 0.018 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.085 ± 0.006 | 0.037 ± 0.000 | 0.296 ± 0.026 | 0.345 ± 0.053 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.053 ± 0.004 | 0.010 ± 0.000 | 0.100 ± 0.003 | 0.115 ± 0.005 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.077 ± 0.004 | 0.035 ± 0.000 | 0.305 ± 0.007 | 0.347 ± 0.011 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.058 ± 0.002 | 0.011 ± 0.000 | 0.130 ± 0.005 | 0.146 ± 0.007 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.083 ± 0.002 | 0.038 ± 0.001 | 0.417 ± 0.012 | 0.471 ± 0.009 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.055 ± 0.001 | 0.010 ± 0.000 | 0.055 ± 0.004 | 0.089 ± 0.003 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.078 ± 0.002 | 0.039 ± 0.000 | 0.195 ± 0.006 | 0.263 ± 0.003 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.069 ± 0.002 | 0.026 ± 0.001 | 0.095 ± 0.001 | 0.128 ± 0.014 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.157 ± 0.006 | 0.106 ± 0.001 | 0.359 ± 0.010 | 0.463 ± 0.022 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.136 ± 0.009 | 0.016 ± 0.002 | 0.048 ± 0.002 | 0.473 ± 0.063 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.134 ± 0.005 | 0.019 ± 0.001 | 0.095 ± 0.012 | 0.527 ± 0.037 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.134 ± 0.013 | 0.016 ± 0.002 | 0.090 ± 0.001 | 0.567 ± 0.147 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.139 ± 0.007 | 0.023 ± 0.001 | 0.253 ± 0.016 | 0.737 ± 0.111 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.174 ± 0.006 | 0.113 ± 0.012 | 0.126 ± 0.008 | 0.490 ± 0.047 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.199 ± 0.011 | 0.189 ± 0.007 | 0.326 ± 0.015 | 0.695 ± 0.048 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.186 ± 0.010 | 0.116 ± 0.008 | 0.156 ± 0.007 | 0.503 ± 0.027 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.207 ± 0.006 | 0.209 ± 0.011 | 0.448 ± 0.009 | 1.310 ± 0.385 |
| large | `eigh` | f64 | 4 | `64x64` | 0.418 ± 0.028 | 0.267 ± 0.003 | 0.649 ± 0.011 | 0.722 ± 0.039 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 9.800 ± 0.628 | 3.272 ± 0.214 | 3.604 ± 0.233 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 64.006 ± 0.664 | 14.406 ± 0.087 | 14.345 ± 0.232 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 9.811 ± 0.058 | 2.905 ± 0.104 | 3.469 ± 0.113 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 65.847 ± 0.191 | 13.007 ± 0.160 | 14.127 ± 0.151 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 1.250 ± 0.044 | 1.423 ± 0.064 | 1.035 ± 0.102 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 6.097 ± 0.059 | 8.239 ± 0.371 | 3.550 ± 0.097 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 2.056 ± 0.147 | 0.923 ± 0.083 | 1.182 ± 0.057 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 10.382 ± 0.062 | 4.515 ± 0.145 | 3.919 ± 0.212 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.075 ± 0.004 | 0.016 ± 0.002 | 0.896 ± 0.031 | 0.945 ± 0.071 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.162 ± 0.015 | 0.026 ± 0.002 | 0.888 ± 0.042 | 1.261 ± 0.098 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 1.891 ± 0.050 | 1.913 ± 0.097 | 2.053 ± 0.120 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 9.086 ± 0.712 | 8.725 ± 0.150 | 7.645 ± 0.233 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 2.700 ± 0.303 | 1.824 ± 0.071 | 2.218 ± 0.068 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 18.121 ± 7.595 | 8.314 ± 0.098 | 8.289 ± 0.137 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.236 ± 0.011 | 0.105 ± 0.004 | 0.488 ± 0.004 | 0.907 ± 0.063 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.254 ± 0.016 | 0.430 ± 0.113 | 0.417 ± 0.109 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 1.120 ± 0.008 | 1.180 ± 0.085 | 1.321 ± 0.137 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.312 ± 0.008 | 0.488 ± 0.050 | 0.531 ± 0.152 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 1.236 ± 0.014 | 1.832 ± 0.048 | 1.238 ± 0.103 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 0.815 ± 0.070 | 0.573 ± 0.005 | 0.754 ± 0.048 | 0.968 ± 0.060 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 60.062 ± 1.138 | 5.630 ± 0.061 | 5.877 ± 0.025 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 595.751 ± 14.842 | 25.700 ± 0.346 | 25.707 ± 0.619 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 58.916 ± 0.996 | 5.039 ± 0.084 | 6.094 ± 1.177 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 591.349 ± 4.332 | 22.702 ± 0.041 | 24.110 ± 0.144 |
| large | `matmul` | f64 | 4 | `128x128` | 0.087 ± 0.007 | 0.021 ± 0.001 | 3.524 ± 0.099 | 3.609 ± 0.073 |
| large | `matmul` | f64 | 4 | `256x256` | 0.220 ± 0.034 | 0.130 ± 0.004 | 13.727 ± 0.086 | 14.542 ± 0.396 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 0.560 ± 0.010 | 0.508 ± 0.023 | 57.997 ± 1.398 | 57.166 ± 3.136 |
| large | `qr` | f64 | 4 | `64x64` | 0.118 ± 0.002 | 0.067 ± 0.002 | 0.544 ± 0.022 | 0.566 ± 0.025 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.122 ± 0.003 | 0.016 ± 0.002 | 0.484 ± 0.013 | 0.572 ± 0.056 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.124 ± 0.005 | 0.020 ± 0.003 | 0.586 ± 0.012 | 0.669 ± 0.316 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.134 ± 0.007 | 0.032 ± 0.003 | 0.917 ± 0.013 | 0.978 ± 0.028 |
| large | `svd` | f64 | 4 | `64x64` | 0.665 ± 0.032 | 0.524 ± 0.002 | 0.726 ± 0.014 | 0.788 ± 0.041 |
| small | `eigh` | f64 | 4 | `2x2` | 0.043 ± 0.003 | 0.001 ± 0.000 | 0.009 ± 0.001 | 0.025 ± 0.005 |
| small | `eigh` | f64 | 4 | `4x4` | 0.054 ± 0.011 | 0.002 ± 0.000 | 0.012 ± 0.001 | 0.031 ± 0.007 |
| small | `eigh` | f64 | 4 | `8x8` | 0.056 ± 0.003 | 0.003 ± 0.000 | 0.018 ± 0.000 | 0.036 ± 0.004 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.068 ± 0.039 | 0.009 ± 0.001 | 0.012 ± 0.001 | 0.085 ± 0.009 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.071 ± 0.010 | 0.009 ± 0.001 | 0.015 ± 0.001 | 0.101 ± 0.005 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.065 ± 0.013 | 0.007 ± 0.003 | 0.025 ± 0.001 | 0.118 ± 0.014 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.027 ± 0.002 | 0.038 ± 0.004 | 0.130 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.029 ± 0.002 | 0.036 ± 0.004 | 0.127 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.029 ± 0.001 | 0.045 ± 0.001 | 0.129 ± 0.009 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.051 ± 0.003 | 0.034 ± 0.011 | 0.331 ± 0.092 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.052 ± 0.003 | 0.042 ± 0.031 | 0.317 ± 0.024 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.064 ± 0.005 | 0.040 ± 0.004 | 0.360 ± 0.182 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.051 ± 0.005 | 0.135 ± 0.002 | 0.191 ± 0.030 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.049 ± 0.005 | 0.136 ± 0.005 | 0.187 ± 0.010 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.050 ± 0.006 | 0.145 ± 0.010 | 0.189 ± 0.009 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.072 ± 0.007 | 0.106 ± 0.010 | 0.535 ± 0.083 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.069 ± 0.007 | 0.104 ± 0.006 | 0.534 ± 0.034 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.078 ± 0.005 | 0.104 ± 0.010 | 0.684 ± 0.239 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.174 ± 0.024 | 0.013 ± 0.003 | 0.020 ± 0.003 | 0.355 ± 0.012 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.193 ± 0.021 | 0.018 ± 0.004 | 0.023 ± 0.001 | 0.358 ± 0.021 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.194 ± 0.040 | 0.015 ± 0.001 | 0.039 ± 0.001 | 0.392 ± 0.029 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.077 ± 0.003 | 0.082 ± 0.008 | 0.192 ± 0.076 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.073 ± 0.004 | 0.093 ± 0.023 | 0.187 ± 0.008 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.076 ± 0.001 | 0.082 ± 0.011 | 0.185 ± 0.008 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.132 ± 0.003 | 0.082 ± 0.021 | 0.473 ± 0.029 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.130 ± 0.011 | 0.081 ± 0.012 | 0.481 ± 0.085 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.135 ± 0.014 | 0.079 ± 0.009 | 0.482 ± 0.029 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.265 ± 0.023 | 0.070 ± 0.002 | 0.029 ± 0.001 | 0.408 ± 0.082 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.207 ± 0.006 | 0.073 ± 0.001 | 0.033 ± 0.001 | 0.435 ± 0.089 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.191 ± 0.015 | 0.083 ± 0.008 | 0.038 ± 0.001 | 0.435 ± 0.062 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.023 ± 0.004 | 0.093 ± 0.002 | 0.155 ± 0.020 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.025 ± 0.008 | 0.095 ± 0.004 | 0.156 ± 0.008 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.024 ± 0.003 | 0.091 ± 0.004 | 0.169 ± 0.087 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.046 ± 0.006 | 0.046 ± 0.003 | 0.371 ± 0.180 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.045 ± 0.006 | 0.047 ± 0.011 | 0.376 ± 0.207 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.047 ± 0.003 | 0.048 ± 0.005 | 0.357 ± 0.023 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.323 ± 0.051 | 0.032 ± 0.004 | 0.019 ± 0.001 | 0.403 ± 0.088 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.260 ± 0.079 | 0.033 ± 0.002 | 0.023 ± 0.001 | 0.393 ± 0.021 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.239 ± 0.012 | 0.038 ± 0.005 | 0.032 ± 0.003 | 0.410 ± 0.034 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.035 ± 0.023 | 0.053 ± 0.004 | 0.134 ± 0.014 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.019 ± 0.001 | 0.053 ± 0.002 | 0.134 ± 0.005 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.020 ± 0.001 | 0.066 ± 0.025 | 0.136 ± 0.035 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.032 ± 0.003 | 0.045 ± 0.022 | 0.341 ± 0.016 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.032 ± 0.002 | 0.039 ± 0.008 | 0.346 ± 0.017 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.038 ± 0.001 | 0.040 ± 0.002 | 0.356 ± 0.020 |
| small | `matmul` | f64 | 4 | `2x2` | 0.105 ± 0.053 | 0.008 ± 0.002 | 0.008 ± 0.001 | 0.037 ± 0.006 |
| small | `matmul` | f64 | 4 | `4x4` | 0.055 ± 0.012 | 0.009 ± 0.002 | 0.011 ± 0.013 | 0.041 ± 0.004 |
| small | `matmul` | f64 | 4 | `8x8` | 0.060 ± 0.007 | 0.007 ± 0.001 | 0.020 ± 0.001 | 0.064 ± 0.007 |
| small | `qr` | f64 | 4 | `2x2` | 0.053 ± 0.021 | 0.001 ± 0.000 | 0.033 ± 0.021 | 0.025 ± 0.004 |
| small | `qr` | f64 | 4 | `4x4` | 0.054 ± 0.009 | 0.001 ± 0.000 | 0.022 ± 0.005 | 0.028 ± 0.009 |
| small | `qr` | f64 | 4 | `8x8` | 0.051 ± 0.003 | 0.002 ± 0.000 | 0.029 ± 0.006 | 0.032 ± 0.008 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.043 ± 0.007 | 0.002 ± 0.000 | 0.015 ± 0.001 | 0.044 ± 0.003 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.048 ± 0.006 | 0.002 ± 0.000 | 0.016 ± 0.002 | 0.041 ± 0.004 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.044 ± 0.007 | 0.002 ± 0.000 | 0.017 ± 0.000 | 0.044 ± 0.004 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.058 ± 0.019 | 0.002 ± 0.000 | 0.018 ± 0.001 | 0.046 ± 0.007 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.048 ± 0.002 | 0.003 ± 0.000 | 0.022 ± 0.000 | 0.051 ± 0.008 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.054 ± 0.006 | 0.003 ± 0.000 | 0.024 ± 0.000 | 0.053 ± 0.005 |
| small | `svd` | f64 | 4 | `2x2` | 0.046 ± 0.011 | 0.001 ± 0.000 | 0.009 ± 0.001 | 0.030 ± 0.006 |
| small | `svd` | f64 | 4 | `4x4` | 0.057 ± 0.012 | 0.002 ± 0.000 | 0.011 ± 0.001 | 0.031 ± 0.006 |
| small | `svd` | f64 | 4 | `8x8` | 0.061 ± 0.005 | 0.006 ± 0.000 | 0.020 ± 0.001 | 0.042 ± 0.007 |

# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260806_150841`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260806_150841`.

- tenferro-rs commit: `11a5b5a3c30b6258919557b4b69b429b6e686d75`

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

- tenferro-rs features: `system-accelerate`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 4

- CSV: `data/results/mac-cpu/cpu/einsum/20260806_150841/cpu_ops_t4_20260806_150841.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260806_150841/cpu_ops_t4_20260806_150841.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.143 ± 0.021 | 0.051 ± 0.003 | 0.085 ± 0.002 | 0.112 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.205 ± 0.020 | 0.102 ± 0.002 | 0.314 ± 0.001 | 0.367 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.155 ± 0.018 | 0.059 ± 0.002 | 0.126 ± 0.001 | 0.150 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.250 ± 0.009 | 0.143 ± 0.003 | 0.469 ± 0.009 | 0.528 ± 0.012 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.149 ± 0.018 | 0.039 ± 0.003 | 0.029 ± 0.001 | 0.127 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.136 ± 0.011 | 0.044 ± 0.042 | 0.076 ± 0.001 | 0.175 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.142 ± 0.018 | 0.041 ± 0.009 | 0.077 ± 0.001 | 0.180 ± 0.012 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.139 ± 0.021 | 0.040 ± 0.003 | 0.276 ± 0.003 | 0.389 ± 0.011 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.149 ± 0.014 | 0.055 ± 0.003 | 0.075 ± 0.004 | 0.083 ± 0.008 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.200 ± 0.009 | 0.109 ± 0.002 | 0.230 ± 0.016 | 0.237 ± 0.013 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.150 ± 0.024 | 0.055 ± 0.003 | 0.107 ± 0.007 | 0.114 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.229 ± 0.007 | 0.128 ± 0.007 | 0.349 ± 0.007 | 0.373 ± 0.011 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.250 ± 0.013 | 0.123 ± 0.005 | 0.123 ± 0.011 | 0.132 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.400 ± 0.086 | 0.240 ± 0.079 | 0.345 ± 0.004 | 0.395 ± 0.011 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.241 ± 0.016 | 0.117 ± 0.012 | 0.141 ± 0.005 | 0.171 ± 0.010 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.354 ± 0.030 | 0.242 ± 0.010 | 0.482 ± 0.006 | 0.524 ± 0.012 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.164 ± 0.015 | 0.067 ± 0.001 | 0.065 ± 0.002 | 0.104 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.244 ± 0.050 | 0.151 ± 0.004 | 0.232 ± 0.005 | 0.312 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.183 ± 0.006 | 0.086 ± 0.006 | 0.111 ± 0.001 | 0.152 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.332 ± 0.035 | 0.234 ± 0.005 | 0.413 ± 0.019 | 0.509 ± 0.017 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.355 ± 0.038 | 0.092 ± 0.005 | 0.060 ± 0.002 | 0.516 ± 0.087 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.400 ± 0.193 | 0.094 ± 0.011 | 0.106 ± 0.001 | 0.568 ± 0.062 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.361 ± 0.033 | 0.089 ± 0.008 | 0.108 ± 0.002 | 0.583 ± 0.053 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.405 ± 0.071 | 0.099 ± 0.003 | 0.318 ± 0.022 | 0.773 ± 0.045 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.773 ± 0.085 | 0.333 ± 0.014 | 0.142 ± 0.011 | 0.533 ± 0.027 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 1.189 ± 0.104 | 0.540 ± 0.021 | 0.374 ± 0.005 | 0.763 ± 0.023 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.788 ± 0.041 | 0.336 ± 0.027 | 0.164 ± 0.003 | 0.568 ± 0.034 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 1.194 ± 0.051 | 0.518 ± 0.034 | 0.541 ± 0.011 | 0.909 ± 0.024 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | 0.413 ± 0.036 | 0.247 ± 0.006 | 0.748 ± 0.015 | 0.782 ± 0.007 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 3.157 ± 0.027 | 3.218 ± 0.095 | 3.645 ± 0.131 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 14.624 ± 0.401 | 15.274 ± 0.097 | 17.907 ± 0.191 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 3.115 ± 0.023 | 2.927 ± 0.103 | 3.512 ± 0.123 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 14.252 ± 0.092 | 13.945 ± 3.638 | 16.637 ± 0.220 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 0.856 ± 0.054 | 1.326 ± 0.101 | 1.130 ± 0.100 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 4.724 ± 0.101 | 9.497 ± 1.240 | 5.015 ± 0.160 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 0.825 ± 0.039 | 0.847 ± 0.017 | 1.369 ± 0.207 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 4.746 ± 0.082 | 4.933 ± 0.445 | 5.622 ± 0.190 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.157 ± 0.014 | 0.054 ± 0.007 | 1.026 ± 0.028 | 1.065 ± 0.035 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.379 ± 0.045 | 0.110 ± 0.008 | 1.048 ± 0.030 | 1.372 ± 0.073 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 2.079 ± 0.059 | 1.869 ± 0.021 | 2.111 ± 0.163 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 9.190 ± 0.028 | 9.185 ± 0.170 | 9.434 ± 0.157 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 2.067 ± 0.055 | 1.795 ± 0.045 | 2.253 ± 0.061 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 9.065 ± 0.065 | 8.765 ± 0.219 | 10.477 ± 0.223 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.775 ± 0.009 | 0.280 ± 0.018 | 0.594 ± 0.026 | 0.950 ± 0.032 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.488 ± 0.071 | 0.392 ± 0.017 | 0.443 ± 0.072 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 1.469 ± 0.022 | 1.294 ± 0.079 | 1.367 ± 0.081 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.446 ± 0.041 | 0.469 ± 0.032 | 0.532 ± 0.079 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 1.421 ± 0.028 | 2.116 ± 0.323 | 1.366 ± 0.056 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 0.991 ± 0.034 | 0.428 ± 0.013 | 0.856 ± 0.035 | 0.980 ± 0.051 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 5.264 ± 0.056 | 5.755 ± 0.162 | 6.084 ± 0.095 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 24.093 ± 0.140 | 26.569 ± 0.287 | 27.961 ± 0.178 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 5.227 ± 0.026 | 5.154 ± 0.073 | 5.751 ± 0.061 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 23.923 ± 0.129 | 32.055 ± 0.908 | 34.231 ± 0.337 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | 0.195 ± 0.022 | 0.058 ± 0.017 | 4.027 ± 0.016 | 3.957 ± 0.163 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | 0.294 ± 0.038 | 0.134 ± 0.007 | 16.261 ± 0.158 | 15.381 ± 0.300 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 0.768 ± 0.016 | 0.429 ± 0.049 | 67.247 ± 1.598 | 61.349 ± 0.412 | - | - |
| large | `qr` | f64 | 4 | `64x64` | 0.208 ± 0.014 | 0.109 ± 0.008 | 0.630 ± 0.024 | 0.640 ± 0.009 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.273 ± 0.091 | 0.097 ± 0.016 | 0.569 ± 0.010 | 0.620 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.267 ± 0.011 | 0.099 ± 0.017 | 0.678 ± 0.015 | 0.736 ± 0.011 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.277 ± 0.024 | 0.108 ± 0.009 | 1.058 ± 0.034 | 1.087 ± 0.043 | - | - |
| large | `svd` | f64 | 4 | `64x64` | 0.445 ± 0.038 | 0.342 ± 0.006 | 0.833 ± 0.011 | 0.833 ± 0.020 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | 0.231 ± 0.009 | 0.032 ± 0.004 | 0.011 ± 0.001 | 0.038 ± 0.006 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | 0.148 ± 0.015 | 0.029 ± 0.002 | 0.014 ± 0.002 | 0.038 ± 0.009 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | 0.128 ± 0.008 | 0.033 ± 0.027 | 0.021 ± 0.001 | 0.048 ± 0.008 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.425 ± 0.160 | 0.029 ± 0.006 | 0.014 ± 0.001 | 0.133 ± 0.008 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.156 ± 0.020 | 0.029 ± 0.001 | 0.017 ± 0.001 | 0.117 ± 0.005 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.127 ± 0.020 | 0.030 ± 0.001 | 0.029 ± 0.001 | 0.131 ± 0.006 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.148 ± 0.019 | 0.044 ± 0.004 | 0.149 ± 0.009 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.140 ± 0.028 | 0.044 ± 0.002 | 0.147 ± 0.008 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.159 ± 0.026 | 0.051 ± 0.001 | 0.156 ± 0.014 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.152 ± 0.035 | 0.041 ± 0.005 | 0.352 ± 0.022 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.124 ± 0.005 | 0.037 ± 0.003 | 0.363 ± 0.024 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.151 ± 0.026 | 0.046 ± 0.002 | 0.353 ± 0.024 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.161 ± 0.022 | 0.116 ± 0.022 | 0.217 ± 0.006 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.173 ± 0.037 | 0.135 ± 0.012 | 0.221 ± 0.009 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.155 ± 0.017 | 0.136 ± 0.003 | 0.218 ± 0.015 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.156 ± 0.035 | 0.090 ± 0.030 | 0.589 ± 0.030 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.156 ± 0.025 | 0.105 ± 0.005 | 0.561 ± 0.026 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.181 ± 0.008 | 0.104 ± 0.013 | 0.584 ± 0.042 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.477 ± 0.058 | 0.091 ± 0.005 | 0.025 ± 0.004 | 0.390 ± 0.022 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.359 ± 0.029 | 0.103 ± 0.004 | 0.028 ± 0.001 | 0.411 ± 0.024 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.360 ± 0.020 | 0.097 ± 0.002 | 0.045 ± 0.001 | 0.415 ± 0.023 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.324 ± 0.032 | 0.083 ± 0.018 | 0.212 ± 0.013 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.275 ± 0.028 | 0.068 ± 0.016 | 0.210 ± 0.006 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.285 ± 0.031 | 0.081 ± 0.007 | 0.210 ± 0.008 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.335 ± 0.037 | 0.075 ± 0.011 | 0.541 ± 0.043 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.328 ± 0.034 | 0.079 ± 0.004 | 0.541 ± 0.031 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.319 ± 0.014 | 0.083 ± 0.003 | 0.523 ± 0.026 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.791 ± 0.120 | 0.234 ± 0.040 | 0.036 ± 0.001 | 0.447 ± 0.041 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.647 ± 0.040 | 0.227 ± 0.035 | 0.040 ± 0.001 | 0.433 ± 0.024 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.663 ± 0.044 | 0.212 ± 0.014 | 0.046 ± 0.001 | 0.470 ± 0.038 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.176 ± 0.028 | 0.114 ± 0.003 | 0.186 ± 0.012 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.172 ± 0.023 | 0.111 ± 0.006 | 0.180 ± 0.016 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.178 ± 0.026 | 0.110 ± 0.004 | 0.175 ± 0.016 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.181 ± 0.026 | 0.057 ± 0.004 | 0.398 ± 0.045 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.196 ± 0.021 | 0.054 ± 0.002 | 0.381 ± 0.039 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.167 ± 0.017 | 0.055 ± 0.005 | 0.378 ± 0.020 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.496 ± 0.041 | 0.108 ± 0.004 | 0.024 ± 0.001 | 0.441 ± 0.031 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.335 ± 0.024 | 0.111 ± 0.017 | 0.028 ± 0.001 | 0.449 ± 0.063 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.342 ± 0.018 | 0.104 ± 0.022 | 0.038 ± 0.001 | 0.451 ± 0.043 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.120 ± 0.021 | 0.065 ± 0.001 | 0.159 ± 0.011 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.116 ± 0.029 | 0.062 ± 0.002 | 0.163 ± 0.013 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.121 ± 0.006 | 0.072 ± 0.001 | 0.158 ± 0.011 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.092 ± 0.008 | 0.051 ± 0.009 | 0.385 ± 0.021 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.101 ± 0.020 | 0.042 ± 0.003 | 0.405 ± 0.018 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.128 ± 0.047 | 0.045 ± 0.003 | 0.411 ± 0.029 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | 0.296 ± 0.081 | 0.038 ± 0.002 | 0.010 ± 0.001 | 0.065 ± 0.010 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | 0.170 ± 0.022 | 0.055 ± 0.022 | 0.012 ± 0.001 | 0.063 ± 0.034 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | 0.129 ± 0.035 | 0.037 ± 0.028 | 0.023 ± 0.000 | 0.075 ± 0.008 | - | - |
| small | `qr` | f64 | 4 | `2x2` | 0.224 ± 0.015 | 0.032 ± 0.014 | 0.021 ± 0.024 | 0.034 ± 0.009 | - | - |
| small | `qr` | f64 | 4 | `4x4` | 0.138 ± 0.025 | 0.031 ± 0.012 | 0.020 ± 0.011 | 0.035 ± 0.007 | - | - |
| small | `qr` | f64 | 4 | `8x8` | 0.124 ± 0.027 | 0.031 ± 0.001 | 0.032 ± 0.002 | 0.041 ± 0.010 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.332 ± 0.066 | 0.081 ± 0.013 | 0.018 ± 0.001 | 0.051 ± 0.006 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.292 ± 0.072 | 0.077 ± 0.039 | 0.018 ± 0.000 | 0.054 ± 0.008 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.213 ± 0.018 | 0.078 ± 0.056 | 0.021 ± 0.001 | 0.053 ± 0.004 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.205 ± 0.012 | 0.107 ± 0.010 | 0.021 ± 0.000 | 0.059 ± 0.011 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.218 ± 0.066 | 0.085 ± 0.009 | 0.026 ± 0.000 | 0.062 ± 0.006 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.188 ± 0.040 | 0.083 ± 0.016 | 0.029 ± 0.000 | 0.064 ± 0.008 | - | - |
| small | `svd` | f64 | 4 | `2x2` | 0.276 ± 0.022 | 0.037 ± 0.029 | 0.010 ± 0.001 | 0.038 ± 0.011 | - | - |
| small | `svd` | f64 | 4 | `4x4` | 0.141 ± 0.015 | 0.035 ± 0.011 | 0.013 ± 0.001 | 0.036 ± 0.010 | - | - |
| small | `svd` | f64 | 4 | `8x8` | 0.139 ± 0.023 | 0.039 ± 0.003 | 0.023 ± 0.001 | 0.049 ± 0.008 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `large/grad_sum_matmul` (f64, threads=4, shape=`64x64`): `tenferro-trace` is 19.6x faster than the slowest successful cell (`jax-cpu`, 1.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=4, shape=`64x64`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`jax-cpu`, 1.372 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`128x128`): `tenferro-eager` is 20.6x faster than the slowest successful cell (`pytorch-cpu`, 4.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`128x128`): `tenferro-trace` is 68.9x faster than the slowest successful cell (`pytorch-cpu`, 4.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`256x256`): `tenferro-eager` is 55.4x faster than the slowest successful cell (`pytorch-cpu`, 16.261 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`256x256`): `tenferro-trace` is 121.6x faster than the slowest successful cell (`pytorch-cpu`, 16.261 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=4, shape=`256x1024 * 1024x256`): `tenferro-eager` is 87.5x faster than the slowest successful cell (`pytorch-cpu`, 67.247 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=4, shape=`256x1024 * 1024x256`): `tenferro-trace` is 156.8x faster than the slowest successful cell (`pytorch-cpu`, 67.247 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`64x64,rhs=64`): `tenferro-trace` is 10.0x faster than the slowest successful cell (`jax-cpu`, 1.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/eigh` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 20.4x faster than the slowest successful cell (`tenferro-eager`, 0.231 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/eigh` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`tenferro-eager`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 29.9x faster than the slowest successful cell (`tenferro-eager`, 0.425 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`tenferro-eager`, 0.425 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 19.0x faster than the slowest successful cell (`tenferro-eager`, 0.477 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`2x2,rhs=1`): `pytorch-cpu` is 21.8x faster than the slowest successful cell (`tenferro-eager`, 0.791 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`4x4,rhs=1`): `pytorch-cpu` is 16.4x faster than the slowest successful cell (`tenferro-eager`, 0.647 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`8x8,rhs=1`): `pytorch-cpu` is 14.5x faster than the slowest successful cell (`tenferro-eager`, 0.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 20.5x faster than the slowest successful cell (`tenferro-eager`, 0.496 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 16.2x faster than the slowest successful cell (`jax-cpu`, 0.449 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.451 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 30.9x faster than the slowest successful cell (`tenferro-eager`, 0.296 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 14.0x faster than the slowest successful cell (`tenferro-eager`, 0.170 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`tenferro-eager`, 0.224 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/solve` (f64, threads=4, shape=`2x2,rhs=1`): `pytorch-cpu` is 18.4x faster than the slowest successful cell (`tenferro-eager`, 0.332 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/solve` (f64, threads=4, shape=`2x2,rhs=4`): `pytorch-cpu` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 0.292 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/solve` (f64, threads=4, shape=`4x4,rhs=1`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`tenferro-eager`, 0.213 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/svd` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 27.0x faster than the slowest successful cell (`tenferro-eager`, 0.276 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/svd` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

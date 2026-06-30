# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260630_183200`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260630_183200`.

- tenferro-rs commit: `38b1c5f2a0f0229336dda751d1033cae3cfc106a`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260630_183200/cpu_ops_t4_20260630_183200.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260630_183200/cpu_ops_t4_20260630_183200.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.040 ± 0.005 | 0.008 ± 0.000 | 0.089 ± 0.002 | 0.122 ± 0.007 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.066 ± 0.004 | 0.030 ± 0.033 | 0.322 ± 0.003 | 0.385 ± 0.008 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.053 ± 0.007 | 0.018 ± 0.000 | 0.129 ± 0.008 | 0.166 ± 0.004 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.115 ± 0.005 | 0.068 ± 0.000 | 0.491 ± 0.084 | 0.562 ± 0.011 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.046 ± 0.030 | 0.009 ± 0.003 | 0.031 ± 0.001 | 0.139 ± 0.003 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.041 ± 0.008 | 0.010 ± 0.004 | 0.082 ± 0.001 | 0.199 ± 0.010 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.042 ± 0.010 | 0.006 ± 0.003 | 0.083 ± 0.003 | 0.204 ± 0.007 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.041 ± 0.005 | 0.010 ± 0.005 | 0.288 ± 0.010 | 0.421 ± 0.012 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.038 ± 0.006 | 0.009 ± 0.000 | 0.089 ± 0.023 | 0.092 ± 0.005 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.067 ± 0.001 | 0.031 ± 0.000 | 0.232 ± 0.004 | 0.263 ± 0.008 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.044 ± 0.015 | 0.012 ± 0.000 | 0.109 ± 0.005 | 0.122 ± 0.005 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.083 ± 0.006 | 0.044 ± 0.000 | 0.366 ± 0.013 | 0.410 ± 0.010 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.046 ± 0.003 | 0.012 ± 0.000 | 0.116 ± 0.003 | 0.146 ± 0.009 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.079 ± 0.006 | 0.043 ± 0.009 | 0.359 ± 0.030 | 0.489 ± 0.102 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.048 ± 0.008 | 0.013 ± 0.000 | 0.148 ± 0.003 | 0.183 ± 0.005 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.083 ± 0.003 | 0.043 ± 0.000 | 0.509 ± 0.054 | 0.569 ± 0.026 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.060 ± 0.044 | 0.013 ± 0.000 | 0.069 ± 0.002 | 0.112 ± 0.005 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.087 ± 0.003 | 0.046 ± 0.000 | 0.231 ± 0.001 | 0.333 ± 0.004 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.067 ± 0.004 | 0.032 ± 0.000 | 0.117 ± 0.009 | 0.159 ± 0.005 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.163 ± 0.004 | 0.123 ± 0.000 | 0.425 ± 0.005 | 0.544 ± 0.013 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.114 ± 0.029 | 0.029 ± 0.015 | 0.064 ± 0.003 | 0.569 ± 0.033 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.172 ± 0.046 | 0.037 ± 0.039 | 0.116 ± 0.002 | 0.671 ± 0.190 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.115 ± 0.029 | 0.032 ± 0.009 | 0.116 ± 0.002 | 0.639 ± 0.016 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.145 ± 0.030 | 0.039 ± 0.013 | 0.324 ± 0.007 | 0.844 ± 0.011 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.186 ± 0.072 | 0.144 ± 0.035 | 0.147 ± 0.017 | 0.573 ± 0.028 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.196 ± 0.015 | 0.195 ± 0.006 | 0.404 ± 0.018 | 0.836 ± 0.025 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.151 ± 0.029 | 0.125 ± 0.037 | 0.191 ± 0.025 | 0.620 ± 0.024 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.282 ± 0.063 | 0.209 ± 0.025 | 0.528 ± 0.007 | 1.012 ± 0.084 |
| large | `eigh` | f64 | 4 | `64x64` | 0.327 ± 0.018 | 0.220 ± 0.001 | 0.801 ± 0.020 | 0.853 ± 0.079 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 3.209 ± 0.038 | 3.267 ± 0.084 | 3.793 ± 0.040 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 14.829 ± 0.230 | 15.406 ± 0.213 | 17.876 ± 0.437 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 3.184 ± 0.073 | 2.927 ± 0.054 | 3.570 ± 0.072 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 14.939 ± 0.831 | 13.883 ± 0.237 | 16.742 ± 0.178 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 1.283 ± 0.025 | 1.336 ± 0.068 | 1.106 ± 0.095 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 6.589 ± 0.222 | 8.505 ± 0.257 | 5.047 ± 0.190 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 2.125 ± 0.249 | 0.857 ± 0.022 | 1.335 ± 0.098 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 11.199 ± 0.535 | 4.736 ± 0.349 | 5.549 ± 0.078 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.066 ± 0.004 | 0.013 ± 0.002 | 1.128 ± 0.029 | 1.173 ± 0.034 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.130 ± 0.010 | 0.043 ± 0.007 | 1.135 ± 0.032 | 1.546 ± 0.191 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 1.875 ± 0.068 | 1.880 ± 0.023 | 2.364 ± 0.258 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 9.106 ± 0.238 | 8.971 ± 0.084 | 9.523 ± 0.071 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 2.663 ± 0.143 | 1.820 ± 0.127 | 2.280 ± 0.044 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 12.559 ± 0.555 | 8.776 ± 0.271 | 10.270 ± 0.156 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.199 ± 0.002 | 0.098 ± 0.031 | 0.648 ± 0.021 | 1.124 ± 0.175 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.262 ± 0.019 | 0.413 ± 0.036 | 0.455 ± 0.085 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 1.402 ± 0.035 | 1.335 ± 0.038 | 1.233 ± 0.125 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.310 ± 0.010 | 0.454 ± 0.004 | 0.585 ± 0.103 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 1.417 ± 0.099 | 2.066 ± 0.048 | 1.296 ± 0.054 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 0.745 ± 0.018 | 0.633 ± 0.023 | 0.904 ± 0.089 | 1.097 ± 0.029 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 64.036 ± 0.175 | 5.752 ± 0.139 | 6.182 ± 0.160 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 617.695 ± 1.424 | 26.915 ± 0.368 | 28.095 ± 1.708 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 63.950 ± 0.763 | 5.163 ± 0.137 | 5.836 ± 0.037 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 624.223 ± 8.006 | 24.878 ± 7.273 | 34.196 ± 0.271 |
| large | `matmul` | f64 | 4 | `128x128` | 0.078 ± 0.006 | 0.024 ± 0.002 | 4.539 ± 0.222 | 4.445 ± 0.071 |
| large | `matmul` | f64 | 4 | `256x256` | 0.194 ± 0.011 | 0.125 ± 0.002 | 17.403 ± 0.403 | 17.399 ± 0.297 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 0.607 ± 0.009 | 0.544 ± 0.059 | 71.004 ± 0.429 | 69.809 ± 2.224 |
| large | `qr` | f64 | 4 | `64x64` | 0.119 ± 0.029 | 0.082 ± 0.000 | 0.685 ± 0.051 | 0.683 ± 0.046 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.116 ± 0.005 | 0.020 ± 0.005 | 0.606 ± 0.052 | 0.667 ± 0.069 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.120 ± 0.003 | 0.021 ± 0.004 | 0.719 ± 0.014 | 0.791 ± 0.073 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.130 ± 0.006 | 0.034 ± 0.006 | 1.128 ± 0.040 | 1.181 ± 0.022 |
| large | `svd` | f64 | 4 | `64x64` | 0.686 ± 0.037 | 0.601 ± 0.013 | 0.874 ± 0.018 | 0.908 ± 0.017 |
| small | `eigh` | f64 | 4 | `2x2` | 0.076 ± 0.007 | 0.001 ± 0.000 | 0.011 ± 0.001 | 0.040 ± 0.012 |
| small | `eigh` | f64 | 4 | `4x4` | 0.051 ± 0.008 | 0.002 ± 0.000 | 0.014 ± 0.001 | 0.044 ± 0.008 |
| small | `eigh` | f64 | 4 | `8x8` | 0.046 ± 0.006 | 0.004 ± 0.000 | 0.024 ± 0.000 | 0.050 ± 0.007 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.070 ± 0.009 | 0.007 ± 0.002 | 0.017 ± 0.002 | 0.120 ± 0.005 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.061 ± 0.005 | 0.006 ± 0.001 | 0.017 ± 0.001 | 0.124 ± 0.003 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.047 ± 0.005 | 0.005 ± 0.002 | 0.033 ± 0.001 | 0.142 ± 0.010 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.028 ± 0.010 | 0.047 ± 0.004 | 0.215 ± 0.154 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.029 ± 0.013 | 0.049 ± 0.005 | 0.162 ± 0.012 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.080 ± 0.075 | 0.067 ± 0.009 | 0.173 ± 0.008 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.036 ± 0.003 | 0.040 ± 0.001 | 0.380 ± 0.026 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.032 ± 0.014 | 0.042 ± 0.002 | 0.386 ± 0.019 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.029 ± 0.041 | 0.050 ± 0.007 | 0.386 ± 0.018 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.045 ± 0.014 | 0.121 ± 0.015 | 0.240 ± 0.116 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.043 ± 0.001 | 0.110 ± 0.007 | 0.238 ± 0.010 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.049 ± 0.029 | 0.148 ± 0.018 | 0.234 ± 0.004 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.060 ± 0.008 | 0.090 ± 0.032 | 0.667 ± 0.109 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.056 ± 0.004 | 0.090 ± 0.013 | 0.633 ± 0.027 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.065 ± 0.007 | 0.117 ± 0.008 | 0.633 ± 0.021 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.235 ± 0.061 | 0.028 ± 0.018 | 0.028 ± 0.005 | 0.444 ± 0.238 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.173 ± 0.030 | 0.022 ± 0.078 | 0.029 ± 0.001 | 0.450 ± 0.015 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.161 ± 0.043 | 0.033 ± 0.010 | 0.048 ± 0.001 | 0.461 ± 0.055 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.072 ± 0.010 | 0.085 ± 0.025 | 0.238 ± 0.007 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.069 ± 0.021 | 0.072 ± 0.007 | 0.240 ± 0.011 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.067 ± 0.008 | 0.086 ± 0.009 | 0.234 ± 0.005 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.128 ± 0.033 | 0.084 ± 0.011 | 0.564 ± 0.025 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.120 ± 0.006 | 0.077 ± 0.008 | 0.683 ± 0.169 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.118 ± 0.019 | 0.092 ± 0.023 | 0.579 ± 0.025 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.219 ± 0.013 | 0.093 ± 0.025 | 0.041 ± 0.003 | 0.477 ± 0.011 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.177 ± 0.013 | 0.063 ± 0.031 | 0.039 ± 0.001 | 0.494 ± 0.012 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.154 ± 0.012 | 0.058 ± 0.018 | 0.048 ± 0.001 | 0.535 ± 0.232 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.021 ± 0.005 | 0.117 ± 0.010 | 0.188 ± 0.017 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.037 ± 0.027 | 0.120 ± 0.003 | 0.190 ± 0.012 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.021 ± 0.004 | 0.118 ± 0.003 | 0.197 ± 0.008 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.060 ± 0.029 | 0.057 ± 0.003 | 0.408 ± 0.022 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.073 ± 0.039 | 0.060 ± 0.004 | 0.438 ± 0.080 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.040 ± 0.007 | 0.058 ± 0.002 | 0.456 ± 0.159 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.348 ± 0.032 | 0.037 ± 0.007 | 0.027 ± 0.001 | 0.488 ± 0.128 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.234 ± 0.010 | 0.030 ± 0.002 | 0.028 ± 0.001 | 0.480 ± 0.032 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.201 ± 0.019 | 0.038 ± 0.001 | 0.040 ± 0.001 | 0.481 ± 0.022 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.013 ± 0.006 | 0.065 ± 0.010 | 0.168 ± 0.012 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.017 ± 0.002 | 0.065 ± 0.001 | 0.175 ± 0.017 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.025 ± 0.001 | 0.075 ± 0.002 | 0.172 ± 0.007 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.045 ± 0.044 | 0.048 ± 0.006 | 0.415 ± 0.048 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.028 ± 0.004 | 0.043 ± 0.001 | 0.442 ± 0.178 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.035 ± 0.003 | 0.050 ± 0.002 | 0.418 ± 0.030 |
| small | `matmul` | f64 | 4 | `2x2` | 0.077 ± 0.006 | 0.008 ± 0.001 | 0.011 ± 0.001 | 0.048 ± 0.007 |
| small | `matmul` | f64 | 4 | `4x4` | 0.064 ± 0.009 | 0.007 ± 0.001 | 0.013 ± 0.000 | 0.053 ± 0.004 |
| small | `matmul` | f64 | 4 | `8x8` | 0.049 ± 0.007 | 0.008 ± 0.004 | 0.026 ± 0.001 | 0.076 ± 0.006 |
| small | `qr` | f64 | 4 | `2x2` | 0.051 ± 0.005 | 0.001 ± 0.000 | 0.019 ± 0.005 | 0.035 ± 0.008 |
| small | `qr` | f64 | 4 | `4x4` | 0.049 ± 0.011 | 0.001 ± 0.000 | 0.022 ± 0.003 | 0.036 ± 0.008 |
| small | `qr` | f64 | 4 | `8x8` | 0.042 ± 0.002 | 0.002 ± 0.000 | 0.029 ± 0.007 | 0.043 ± 0.009 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.079 ± 0.004 | 0.003 ± 0.000 | 0.019 ± 0.001 | 0.056 ± 0.006 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.067 ± 0.012 | 0.003 ± 0.000 | 0.020 ± 0.000 | 0.054 ± 0.005 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.057 ± 0.014 | 0.003 ± 0.000 | 0.021 ± 0.000 | 0.056 ± 0.005 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.058 ± 0.009 | 0.003 ± 0.000 | 0.022 ± 0.003 | 0.061 ± 0.008 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.048 ± 0.010 | 0.003 ± 0.000 | 0.029 ± 0.001 | 0.067 ± 0.008 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.047 ± 0.004 | 0.003 ± 0.000 | 0.032 ± 0.000 | 0.070 ± 0.006 |
| small | `svd` | f64 | 4 | `2x2` | 0.056 ± 0.009 | 0.002 ± 0.000 | 0.012 ± 0.002 | 0.039 ± 0.012 |
| small | `svd` | f64 | 4 | `4x4` | 0.052 ± 0.004 | 0.003 ± 0.000 | 0.014 ± 0.003 | 0.044 ± 0.004 |
| small | `svd` | f64 | 4 | `8x8` | 0.051 ± 0.005 | 0.007 ± 0.000 | 0.024 ± 0.001 | 0.053 ± 0.007 |

# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260723_121605`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260723_121605`.

- tenferro-rs commit: `68855c2b65b5adc42dccca9bac04fd136a8f14c8`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260723_121605/cpu_ops_t4_20260723_121605.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260723_121605/cpu_ops_t4_20260723_121605.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.103 ± 0.017 | 0.007 ± 0.000 | 0.085 ± 0.001 | 0.114 ± 0.007 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.116 ± 0.020 | 0.027 ± 0.000 | 0.324 ± 0.018 | 0.368 ± 0.010 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.097 ± 0.021 | 0.018 ± 0.000 | 0.124 ± 0.009 | 0.153 ± 0.011 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.150 ± 0.013 | 0.069 ± 0.001 | 0.457 ± 0.007 | 0.515 ± 0.008 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.097 ± 0.018 | 0.001 ± 0.000 | 0.029 ± 0.001 | 0.124 ± 0.005 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.098 ± 0.012 | 0.003 ± 0.000 | 0.075 ± 0.001 | 0.179 ± 0.013 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.086 ± 0.003 | 0.002 ± 0.000 | 0.074 ± 0.001 | 0.182 ± 0.005 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.119 ± 0.019 | 0.004 ± 0.001 | 0.270 ± 0.006 | 0.381 ± 0.009 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.098 ± 0.036 | 0.008 ± 0.000 | 0.082 ± 0.011 | 0.081 ± 0.006 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.123 ± 0.059 | 0.028 ± 0.000 | 0.224 ± 0.010 | 0.246 ± 0.018 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.090 ± 0.010 | 0.011 ± 0.003 | 0.112 ± 0.009 | 0.113 ± 0.010 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.127 ± 0.011 | 0.042 ± 0.000 | 0.351 ± 0.005 | 0.399 ± 0.015 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.121 ± 0.013 | 0.010 ± 0.000 | 0.111 ± 0.005 | 0.136 ± 0.006 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.152 ± 0.013 | 0.033 ± 0.000 | 0.350 ± 0.016 | 0.413 ± 0.026 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.136 ± 0.011 | 0.012 ± 0.002 | 0.149 ± 0.016 | 0.174 ± 0.011 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.157 ± 0.012 | 0.037 ± 0.000 | 0.480 ± 0.013 | 0.567 ± 0.086 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.108 ± 0.017 | 0.014 ± 0.000 | 0.064 ± 0.002 | 0.104 ± 0.005 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.135 ± 0.030 | 0.050 ± 0.001 | 0.224 ± 0.007 | 0.309 ± 0.016 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.113 ± 0.005 | 0.033 ± 0.000 | 0.119 ± 0.009 | 0.154 ± 0.021 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.234 ± 0.015 | 0.127 ± 0.000 | 0.417 ± 0.006 | 0.506 ± 0.012 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.140 ± 0.008 | 0.032 ± 0.009 | 0.059 ± 0.003 | 0.548 ± 0.035 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.162 ± 0.021 | 0.033 ± 0.024 | 0.109 ± 0.006 | 0.572 ± 0.060 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.145 ± 0.017 | 0.031 ± 0.008 | 0.111 ± 0.006 | 0.607 ± 0.062 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.163 ± 0.018 | 0.042 ± 0.033 | 0.300 ± 0.007 | 0.815 ± 0.076 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.272 ± 0.012 | 0.098 ± 0.046 | 0.134 ± 0.008 | 0.537 ± 0.027 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.290 ± 0.012 | 0.134 ± 0.008 | 0.379 ± 0.009 | 0.775 ± 0.043 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.239 ± 0.026 | 0.081 ± 0.014 | 0.174 ± 0.005 | 0.590 ± 0.045 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.312 ± 0.055 | 0.143 ± 0.007 | 0.527 ± 0.044 | 0.922 ± 0.031 |
| large | `eigh` | f64 | 4 | `64x64` | 0.363 ± 0.023 | 0.226 ± 0.008 | 0.739 ± 0.023 | 0.795 ± 0.038 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 3.163 ± 0.007 | 3.204 ± 0.027 | 3.785 ± 0.089 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 14.669 ± 0.172 | 15.374 ± 0.224 | 18.053 ± 0.968 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 3.141 ± 0.074 | 2.905 ± 0.080 | 3.599 ± 0.069 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 15.348 ± 3.016 | 13.739 ± 0.306 | 16.932 ± 0.196 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 0.863 ± 0.038 | 1.343 ± 0.126 | 1.151 ± 0.082 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 5.591 ± 0.128 | 8.610 ± 0.484 | 5.475 ± 0.686 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 0.872 ± 0.025 | 0.884 ± 0.071 | 1.406 ± 0.185 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 5.119 ± 0.154 | 4.967 ± 0.185 | 5.754 ± 0.328 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.103 ± 0.007 | 0.015 ± 0.052 | 0.999 ± 0.016 | 1.078 ± 0.043 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.167 ± 0.030 | 0.040 ± 0.009 | 1.044 ± 0.013 | 1.382 ± 0.020 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 1.896 ± 0.022 | 1.870 ± 0.070 | 2.221 ± 0.072 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 9.142 ± 0.189 | 9.381 ± 0.522 | 11.073 ± 0.341 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 1.993 ± 0.053 | 1.797 ± 0.077 | 2.320 ± 0.135 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 9.294 ± 0.179 | 8.677 ± 0.127 | 11.262 ± 0.632 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.361 ± 0.018 | 0.105 ± 0.015 | 0.587 ± 0.011 | 0.975 ± 0.054 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.247 ± 0.004 | 0.414 ± 0.061 | 0.425 ± 0.053 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 1.361 ± 0.013 | 1.366 ± 0.127 | 1.376 ± 0.100 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.254 ± 0.014 | 0.474 ± 0.013 | 0.672 ± 0.102 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 1.178 ± 0.054 | 2.080 ± 0.102 | 1.421 ± 0.065 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 0.690 ± 0.044 | 0.345 ± 0.004 | 0.852 ± 0.021 | 1.097 ± 0.070 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 5.207 ± 0.050 | 5.731 ± 0.117 | 6.151 ± 0.130 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 32.166 ± 0.151 | 26.624 ± 0.451 | 28.284 ± 0.860 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 5.188 ± 0.007 | 5.129 ± 0.057 | 5.824 ± 0.179 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 31.612 ± 0.283 | 31.523 ± 0.175 | 34.434 ± 0.543 |
| large | `matmul` | f64 | 4 | `128x128` | 0.111 ± 0.015 | 0.017 ± 0.000 | 3.961 ± 0.041 | 4.004 ± 0.192 |
| large | `matmul` | f64 | 4 | `256x256` | 0.179 ± 0.018 | 0.110 ± 0.010 | 16.009 ± 0.119 | 15.592 ± 0.521 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 0.528 ± 0.017 | 0.518 ± 0.029 | 60.670 ± 0.482 | 63.062 ± 1.819 |
| large | `qr` | f64 | 4 | `64x64` | 0.176 ± 0.023 | 0.083 ± 0.001 | 0.626 ± 0.018 | 0.646 ± 0.024 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.183 ± 0.002 | 0.017 ± 0.001 | 0.574 ± 0.019 | 0.634 ± 0.027 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.184 ± 0.020 | 0.020 ± 0.001 | 0.667 ± 0.025 | 0.691 ± 0.037 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.202 ± 0.023 | 0.030 ± 0.002 | 1.050 ± 0.037 | 1.061 ± 0.023 |
| large | `svd` | f64 | 4 | `64x64` | 0.424 ± 0.037 | 0.316 ± 0.006 | 0.824 ± 0.032 | 0.857 ± 0.037 |
| small | `eigh` | f64 | 4 | `2x2` | 0.198 ± 0.028 | 0.001 ± 0.000 | 0.012 ± 0.002 | 0.037 ± 0.009 |
| small | `eigh` | f64 | 4 | `4x4` | 0.121 ± 0.068 | 0.002 ± 0.000 | 0.014 ± 0.001 | 0.037 ± 0.010 |
| small | `eigh` | f64 | 4 | `8x8` | 0.129 ± 0.046 | 0.004 ± 0.001 | 0.021 ± 0.001 | 0.046 ± 0.011 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.142 ± 0.011 | 0.001 ± 0.000 | 0.015 ± 0.001 | 0.125 ± 0.006 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.148 ± 0.030 | 0.001 ± 0.000 | 0.017 ± 0.001 | 0.110 ± 0.007 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.099 ± 0.005 | 0.001 ± 0.000 | 0.028 ± 0.000 | 0.137 ± 0.010 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.051 ± 0.011 | 0.043 ± 0.003 | 0.161 ± 0.012 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.059 ± 0.006 | 0.045 ± 0.003 | 0.156 ± 0.012 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.056 ± 0.025 | 0.050 ± 0.002 | 0.157 ± 0.014 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.058 ± 0.024 | 0.041 ± 0.008 | 0.374 ± 0.020 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.083 ± 0.022 | 0.040 ± 0.007 | 0.351 ± 0.020 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.091 ± 0.018 | 0.045 ± 0.002 | 0.375 ± 0.025 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.039 ± 0.012 | 0.113 ± 0.005 | 0.227 ± 0.017 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.039 ± 0.008 | 0.132 ± 0.010 | 0.208 ± 0.005 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.039 ± 0.023 | 0.130 ± 0.004 | 0.226 ± 0.013 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.038 ± 0.003 | 0.085 ± 0.013 | 0.592 ± 0.046 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.035 ± 0.007 | 0.111 ± 0.017 | 0.573 ± 0.029 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.043 ± 0.003 | 0.105 ± 0.009 | 0.574 ± 0.023 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.286 ± 0.023 | 0.027 ± 0.010 | 0.026 ± 0.004 | 0.409 ± 0.038 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.174 ± 0.017 | 0.026 ± 0.003 | 0.028 ± 0.002 | 0.397 ± 0.019 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.161 ± 0.013 | 0.032 ± 0.006 | 0.045 ± 0.001 | 0.431 ± 0.039 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.113 ± 0.010 | 0.080 ± 0.036 | 0.211 ± 0.008 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.107 ± 0.017 | 0.079 ± 0.011 | 0.232 ± 0.111 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.107 ± 0.012 | 0.079 ± 0.014 | 0.216 ± 0.011 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.164 ± 0.012 | 0.086 ± 0.023 | 0.540 ± 0.038 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.227 ± 0.034 | 0.082 ± 0.013 | 0.525 ± 0.038 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.200 ± 0.018 | 0.090 ± 0.015 | 0.527 ± 0.023 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.335 ± 0.045 | 0.055 ± 0.012 | 0.037 ± 0.002 | 0.446 ± 0.034 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.256 ± 0.024 | 0.059 ± 0.010 | 0.038 ± 0.001 | 0.456 ± 0.038 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.263 ± 0.039 | 0.062 ± 0.013 | 0.045 ± 0.001 | 0.459 ± 0.034 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.031 ± 0.012 | 0.111 ± 0.007 | 0.180 ± 0.014 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.022 ± 0.015 | 0.108 ± 0.003 | 0.186 ± 0.019 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.034 ± 0.006 | 0.108 ± 0.003 | 0.176 ± 0.016 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.040 ± 0.014 | 0.055 ± 0.004 | 0.383 ± 0.029 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.035 ± 0.015 | 0.055 ± 0.010 | 0.384 ± 0.031 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.039 ± 0.005 | 0.060 ± 0.006 | 0.391 ± 0.020 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.605 ± 0.036 | 0.030 ± 0.007 | 0.025 ± 0.002 | 0.440 ± 0.033 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.426 ± 0.025 | 0.027 ± 0.014 | 0.027 ± 0.001 | 0.450 ± 0.050 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.366 ± 0.011 | 0.032 ± 0.004 | 0.038 ± 0.001 | 0.460 ± 0.037 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.031 ± 0.007 | 0.064 ± 0.004 | 0.152 ± 0.010 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.022 ± 0.009 | 0.061 ± 0.003 | 0.159 ± 0.008 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.035 ± 0.033 | 0.071 ± 0.003 | 0.157 ± 0.009 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.026 ± 0.013 | 0.050 ± 0.009 | 0.406 ± 0.021 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.026 ± 0.026 | 0.041 ± 0.003 | 0.410 ± 0.055 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.033 ± 0.003 | 0.046 ± 0.002 | 0.415 ± 0.015 |
| small | `matmul` | f64 | 4 | `2x2` | 0.167 ± 0.019 | 0.001 ± 0.000 | 0.010 ± 0.002 | 0.055 ± 0.010 |
| small | `matmul` | f64 | 4 | `4x4` | 0.126 ± 0.005 | 0.001 ± 0.000 | 0.012 ± 0.000 | 0.052 ± 0.017 |
| small | `matmul` | f64 | 4 | `8x8` | 0.106 ± 0.017 | 0.001 ± 0.000 | 0.023 ± 0.006 | 0.072 ± 0.003 |
| small | `qr` | f64 | 4 | `2x2` | 0.210 ± 0.039 | 0.001 ± 0.000 | 0.022 ± 0.028 | 0.040 ± 0.009 |
| small | `qr` | f64 | 4 | `4x4` | 0.118 ± 0.015 | 0.002 ± 0.000 | 0.021 ± 0.003 | 0.032 ± 0.007 |
| small | `qr` | f64 | 4 | `8x8` | 0.111 ± 0.027 | 0.002 ± 0.000 | 0.031 ± 0.001 | 0.040 ± 0.007 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.275 ± 0.017 | 0.003 ± 0.001 | 0.019 ± 0.000 | 0.049 ± 0.010 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.228 ± 0.006 | 0.003 ± 0.000 | 0.019 ± 0.000 | 0.054 ± 0.014 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.157 ± 0.005 | 0.003 ± 0.000 | 0.020 ± 0.001 | 0.053 ± 0.006 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.175 ± 0.022 | 0.003 ± 0.000 | 0.021 ± 0.000 | 0.056 ± 0.008 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.139 ± 0.010 | 0.003 ± 0.000 | 0.026 ± 0.000 | 0.058 ± 0.004 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.128 ± 0.005 | 0.003 ± 0.000 | 0.028 ± 0.000 | 0.064 ± 0.011 |
| small | `svd` | f64 | 4 | `2x2` | 0.160 ± 0.088 | 0.002 ± 0.000 | 0.011 ± 0.001 | 0.054 ± 0.013 |
| small | `svd` | f64 | 4 | `4x4` | 0.128 ± 0.018 | 0.003 ± 0.000 | 0.013 ± 0.001 | 0.036 ± 0.008 |
| small | `svd` | f64 | 4 | `8x8` | 0.118 ± 0.008 | 0.007 ± 0.000 | 0.022 ± 0.001 | 0.050 ± 0.011 |

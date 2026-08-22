# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260822_090304`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260822_090304`.

- tenferro-rs commit: `a21a4c602fc6700b9bc0c3f1b14ebd19b9d7ec45`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260822_090304/cpu_ops_t4_20260822_090304.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260822_090304/cpu_ops_t4_20260822_090304.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.136 ± 0.015 | 0.051 ± 0.003 | 0.086 ± 0.001 | 0.114 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.190 ± 0.010 | 0.110 ± 0.007 | 0.311 ± 0.002 | 0.384 ± 0.011 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.144 ± 0.007 | 0.059 ± 0.003 | 0.123 ± 0.001 | 0.159 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.246 ± 0.026 | 0.145 ± 0.003 | 0.468 ± 0.008 | 0.519 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.152 ± 0.012 | 0.040 ± 0.009 | 0.029 ± 0.005 | 0.124 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.126 ± 0.041 | 0.040 ± 0.010 | 0.075 ± 0.003 | 0.190 ± 0.023 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.144 ± 0.010 | 0.038 ± 0.013 | 0.075 ± 0.001 | 0.187 ± 0.015 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.137 ± 0.025 | 0.042 ± 0.005 | 0.268 ± 0.007 | 0.404 ± 0.038 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.144 ± 0.012 | 0.053 ± 0.001 | 0.075 ± 0.006 | 0.083 ± 0.015 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.193 ± 0.012 | 0.111 ± 0.028 | 0.223 ± 0.005 | 0.246 ± 0.016 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.145 ± 0.009 | 0.054 ± 0.002 | 0.105 ± 0.003 | 0.115 ± 0.012 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.212 ± 0.014 | 0.125 ± 0.002 | 0.355 ± 0.023 | 0.379 ± 0.016 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.140 ± 0.007 | 0.125 ± 0.023 | 0.112 ± 0.003 | 0.145 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.204 ± 0.030 | 0.237 ± 0.030 | 0.344 ± 0.003 | 0.399 ± 0.013 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.147 ± 0.002 | 0.128 ± 0.020 | 0.144 ± 0.002 | 0.189 ± 0.160 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.224 ± 0.005 | 0.247 ± 0.028 | 0.475 ± 0.011 | 0.540 ± 0.027 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.149 ± 0.022 | 0.070 ± 0.003 | 0.065 ± 0.004 | 0.106 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.237 ± 0.017 | 0.150 ± 0.007 | 0.224 ± 0.001 | 0.319 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.166 ± 0.011 | 0.091 ± 0.006 | 0.110 ± 0.001 | 0.154 ± 0.009 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.310 ± 0.010 | 0.226 ± 0.008 | 0.410 ± 0.006 | 0.516 ± 0.029 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.319 ± 0.029 | 0.118 ± 0.018 | 0.059 ± 0.002 | 0.541 ± 0.048 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.398 ± 0.051 | 0.100 ± 0.016 | 0.106 ± 0.003 | 0.610 ± 0.059 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.328 ± 0.046 | 0.092 ± 0.014 | 0.106 ± 0.001 | 0.933 ± 0.536 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.334 ± 0.018 | 0.098 ± 0.006 | 0.299 ± 0.007 | 0.878 ± 0.143 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.597 ± 0.056 | 0.311 ± 0.096 | 0.137 ± 0.008 | 0.541 ± 0.041 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.804 ± 0.070 | 0.527 ± 0.015 | 0.367 ± 0.003 | 0.800 ± 0.042 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.630 ± 0.070 | 0.305 ± 0.034 | 0.167 ± 0.003 | 0.697 ± 0.686 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.744 ± 0.059 | 0.538 ± 0.030 | 0.497 ± 0.004 | 0.939 ± 0.080 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | 0.393 ± 0.028 | 0.240 ± 0.014 | 0.701 ± 0.005 | 0.771 ± 0.076 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 3.162 ± 0.036 | 3.149 ± 0.091 | 4.088 ± 0.190 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 14.466 ± 0.242 | 15.122 ± 0.456 | 17.799 ± 0.496 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 3.161 ± 0.102 | 2.927 ± 0.083 | 7.303 ± 0.498 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 14.745 ± 0.149 | 13.565 ± 0.259 | 17.001 ± 0.413 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 0.879 ± 0.013 | 1.312 ± 0.146 | 2.170 ± 0.705 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 5.046 ± 0.191 | 8.455 ± 0.437 | 5.131 ± 0.676 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 0.835 ± 0.015 | 0.848 ± 0.027 | 1.415 ± 0.284 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 5.067 ± 0.235 | 4.749 ± 0.304 | 6.342 ± 0.390 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.136 ± 0.032 | 0.059 ± 0.005 | 0.938 ± 0.011 | 1.071 ± 0.068 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.390 ± 0.049 | 0.108 ± 0.012 | 1.081 ± 0.043 | 1.584 ± 0.138 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 2.062 ± 0.051 | 1.868 ± 0.047 | 2.263 ± 0.106 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 9.074 ± 0.397 | 9.057 ± 0.183 | 9.422 ± 0.200 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 2.080 ± 0.046 | 1.808 ± 0.023 | 2.463 ± 0.220 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 9.043 ± 0.059 | 8.531 ± 0.108 | 10.542 ± 0.418 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.656 ± 0.047 | 0.268 ± 0.019 | 0.593 ± 0.016 | 0.938 ± 0.033 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.421 ± 0.031 | 0.381 ± 0.009 | 0.452 ± 0.084 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 1.467 ± 0.059 | 1.307 ± 0.045 | 1.362 ± 0.081 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.440 ± 0.067 | 0.453 ± 0.029 | 0.588 ± 0.175 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 1.487 ± 0.057 | 2.080 ± 0.087 | 1.405 ± 0.063 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 0.938 ± 0.029 | 0.429 ± 0.020 | 0.859 ± 0.021 | 1.020 ± 0.187 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 5.267 ± 0.026 | 5.701 ± 0.096 | 6.094 ± 0.063 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 24.303 ± 0.146 | 26.499 ± 0.151 | 28.140 ± 0.618 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 5.242 ± 0.035 | 5.134 ± 0.040 | 5.855 ± 0.193 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 24.612 ± 7.793 | 31.200 ± 0.399 | 33.712 ± 0.397 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | 0.176 ± 0.028 | 0.058 ± 0.004 | 3.683 ± 0.047 | 4.086 ± 0.360 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | 0.272 ± 0.010 | 0.140 ± 0.009 | 14.783 ± 0.132 | 15.906 ± 0.740 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 0.723 ± 0.024 | 0.429 ± 0.011 | 66.000 ± 0.758 | 62.245 ± 1.097 | - | - |
| large | `qr` | f64 | 4 | `64x64` | 0.212 ± 0.016 | 0.105 ± 0.010 | 0.582 ± 0.036 | 0.630 ± 0.013 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.202 ± 0.004 | 0.126 ± 0.020 | 0.516 ± 0.007 | 0.604 ± 0.032 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.199 ± 0.007 | 0.105 ± 0.009 | 0.632 ± 0.007 | 0.750 ± 0.025 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.212 ± 0.007 | 0.111 ± 0.021 | 0.979 ± 0.017 | 1.056 ± 0.043 | - | - |
| large | `svd` | f64 | 4 | `64x64` | 0.434 ± 0.043 | 0.339 ± 0.018 | 0.789 ± 0.002 | 0.849 ± 0.031 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | 0.211 ± 0.046 | 0.038 ± 0.008 | 0.012 ± 0.001 | 0.033 ± 0.008 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | 0.163 ± 0.007 | 0.039 ± 0.015 | 0.014 ± 0.002 | 0.040 ± 0.010 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | 0.148 ± 0.042 | 0.031 ± 0.002 | 0.021 ± 0.001 | 0.046 ± 0.008 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.300 ± 0.153 | 0.042 ± 0.032 | 0.014 ± 0.001 | 0.114 ± 0.007 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.146 ± 0.021 | 0.032 ± 0.006 | 0.017 ± 0.001 | 0.113 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.121 ± 0.044 | 0.028 ± 0.002 | 0.029 ± 0.000 | 0.134 ± 0.006 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.184 ± 0.026 | 0.044 ± 0.002 | 0.149 ± 0.007 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.177 ± 0.014 | 0.044 ± 0.002 | 0.153 ± 0.007 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.176 ± 0.020 | 0.051 ± 0.001 | 0.156 ± 0.011 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.152 ± 0.013 | 0.040 ± 0.004 | 0.346 ± 0.017 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.159 ± 0.015 | 0.039 ± 0.001 | 0.365 ± 0.019 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.181 ± 0.015 | 0.045 ± 0.001 | 0.363 ± 0.028 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.165 ± 0.018 | 0.116 ± 0.033 | 0.208 ± 0.007 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.160 ± 0.013 | 0.132 ± 0.008 | 0.218 ± 0.006 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.180 ± 0.022 | 0.137 ± 0.011 | 0.220 ± 0.009 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.172 ± 0.034 | 0.088 ± 0.027 | 0.601 ± 0.047 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.176 ± 0.012 | 0.107 ± 0.029 | 0.618 ± 0.081 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.198 ± 0.041 | 0.110 ± 0.007 | 0.630 ± 0.021 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.525 ± 0.041 | 0.095 ± 0.017 | 0.026 ± 0.005 | 0.412 ± 0.023 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.411 ± 0.096 | 0.106 ± 0.013 | 0.027 ± 0.001 | 0.400 ± 0.018 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.351 ± 0.074 | 0.100 ± 0.006 | 0.045 ± 0.001 | 0.427 ± 0.023 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.303 ± 0.015 | 0.078 ± 0.028 | 0.213 ± 0.012 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.323 ± 0.023 | 0.078 ± 0.007 | 0.216 ± 0.011 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.309 ± 0.021 | 0.081 ± 0.003 | 0.219 ± 0.020 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.398 ± 0.096 | 0.075 ± 0.027 | 0.537 ± 0.056 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.355 ± 0.070 | 0.095 ± 0.033 | 0.530 ± 0.024 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.314 ± 0.022 | 0.087 ± 0.011 | 0.533 ± 0.027 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.558 ± 0.011 | 0.214 ± 0.019 | 0.037 ± 0.002 | 0.443 ± 0.018 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.484 ± 0.009 | 0.226 ± 0.033 | 0.038 ± 0.001 | 0.451 ± 0.027 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.487 ± 0.035 | 0.273 ± 0.026 | 0.045 ± 0.000 | 0.512 ± 0.131 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.167 ± 0.009 | 0.110 ± 0.008 | 0.176 ± 0.012 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.157 ± 0.036 | 0.109 ± 0.003 | 0.187 ± 0.012 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.156 ± 0.021 | 0.112 ± 0.003 | 0.183 ± 0.015 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.173 ± 0.031 | 0.054 ± 0.003 | 0.392 ± 0.029 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.157 ± 0.021 | 0.054 ± 0.003 | 0.396 ± 0.029 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.169 ± 0.005 | 0.056 ± 0.007 | 0.407 ± 0.022 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.439 ± 0.036 | 0.102 ± 0.005 | 0.025 ± 0.002 | 0.445 ± 0.019 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.364 ± 0.032 | 0.107 ± 0.009 | 0.027 ± 0.001 | 0.455 ± 0.058 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.320 ± 0.019 | 0.115 ± 0.011 | 0.038 ± 0.001 | 0.456 ± 0.028 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.110 ± 0.007 | 0.064 ± 0.006 | 0.165 ± 0.015 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.108 ± 0.018 | 0.060 ± 0.002 | 0.161 ± 0.018 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.122 ± 0.006 | 0.070 ± 0.002 | 0.159 ± 0.006 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.100 ± 0.004 | 0.050 ± 0.008 | 0.400 ± 0.014 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.107 ± 0.012 | 0.041 ± 0.003 | 0.413 ± 0.020 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.104 ± 0.016 | 0.046 ± 0.002 | 0.419 ± 0.019 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | 0.244 ± 0.039 | 0.038 ± 0.030 | 0.010 ± 0.001 | 0.109 ± 0.007 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | 0.153 ± 0.008 | 0.064 ± 0.045 | 0.012 ± 0.000 | 0.054 ± 0.006 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | 0.117 ± 0.005 | 0.034 ± 0.015 | 0.022 ± 0.000 | 0.071 ± 0.008 | - | - |
| small | `qr` | f64 | 4 | `2x2` | 0.222 ± 0.015 | 0.036 ± 0.013 | 0.022 ± 0.010 | 0.041 ± 0.010 | - | - |
| small | `qr` | f64 | 4 | `4x4` | 0.130 ± 0.023 | 0.035 ± 0.017 | 0.020 ± 0.015 | 0.036 ± 0.009 | - | - |
| small | `qr` | f64 | 4 | `8x8` | 0.117 ± 0.019 | 0.032 ± 0.013 | 0.034 ± 0.004 | 0.042 ± 0.010 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.206 ± 0.031 | 0.085 ± 0.045 | 0.018 ± 0.001 | 0.053 ± 0.008 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.208 ± 0.023 | 0.114 ± 0.033 | 0.018 ± 0.000 | 0.052 ± 0.004 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.136 ± 0.009 | 0.080 ± 0.009 | 0.020 ± 0.001 | 0.055 ± 0.006 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.131 ± 0.021 | 0.075 ± 0.022 | 0.021 ± 0.000 | 0.055 ± 0.007 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.131 ± 0.017 | 0.091 ± 0.014 | 0.027 ± 0.001 | 0.062 ± 0.014 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.128 ± 0.022 | 0.067 ± 0.007 | 0.029 ± 0.000 | 0.067 ± 0.005 | - | - |
| small | `svd` | f64 | 4 | `2x2` | 0.239 ± 0.021 | 0.033 ± 0.009 | 0.010 ± 0.001 | 0.051 ± 0.016 | - | - |
| small | `svd` | f64 | 4 | `4x4` | 0.138 ± 0.031 | 0.052 ± 0.011 | 0.013 ± 0.001 | 0.040 ± 0.006 | - | - |
| small | `svd` | f64 | 4 | `8x8` | 0.120 ± 0.028 | 0.042 ± 0.002 | 0.023 ± 0.001 | 0.049 ± 0.008 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.933 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=4, shape=`64x64`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`jax-cpu`, 1.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=4, shape=`64x64`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`jax-cpu`, 1.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`128x128`): `tenferro-eager` is 23.2x faster than the slowest successful cell (`jax-cpu`, 4.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`128x128`): `tenferro-trace` is 70.9x faster than the slowest successful cell (`jax-cpu`, 4.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`256x256`): `tenferro-eager` is 58.4x faster than the slowest successful cell (`jax-cpu`, 15.906 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`256x256`): `tenferro-trace` is 113.5x faster than the slowest successful cell (`jax-cpu`, 15.906 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=4, shape=`256x1024 * 1024x256`): `tenferro-eager` is 91.3x faster than the slowest successful cell (`pytorch-cpu`, 66.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=4, shape=`256x1024 * 1024x256`): `tenferro-trace` is 153.7x faster than the slowest successful cell (`pytorch-cpu`, 66.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/eigh` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 18.1x faster than the slowest successful cell (`tenferro-eager`, 0.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/eigh` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 11.7x faster than the slowest successful cell (`tenferro-eager`, 0.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 20.8x faster than the slowest successful cell (`tenferro-eager`, 0.300 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 19.9x faster than the slowest successful cell (`tenferro-eager`, 0.525 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 15.2x faster than the slowest successful cell (`tenferro-eager`, 0.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`2x2,rhs=1`): `pytorch-cpu` is 15.1x faster than the slowest successful cell (`tenferro-eager`, 0.558 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`4x4,rhs=1`): `pytorch-cpu` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 0.484 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`8x8,rhs=1`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.512 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 18.0x faster than the slowest successful cell (`jax-cpu`, 0.445 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 17.1x faster than the slowest successful cell (`jax-cpu`, 0.455 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.456 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.413 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 24.0x faster than the slowest successful cell (`tenferro-eager`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`tenferro-eager`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`tenferro-eager`, 0.222 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/solve` (f64, threads=4, shape=`2x2,rhs=1`): `pytorch-cpu` is 11.3x faster than the slowest successful cell (`tenferro-eager`, 0.206 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/solve` (f64, threads=4, shape=`2x2,rhs=4`): `pytorch-cpu` is 11.3x faster than the slowest successful cell (`tenferro-eager`, 0.208 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/svd` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 23.9x faster than the slowest successful cell (`tenferro-eager`, 0.239 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/svd` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`tenferro-eager`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

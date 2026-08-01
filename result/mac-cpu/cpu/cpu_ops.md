# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260801_221400`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260801_221400`.

- tenferro-rs commit: `0ee2d0dc2f8d21ff62ea682f90f34e4319108ace`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260801_221400/cpu_ops_t4_20260801_221400.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260801_221400/cpu_ops_t4_20260801_221400.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.112 ± 0.028 | 0.030 ± 0.003 | 0.084 ± 0.002 | 0.200 ± 0.018 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.139 ± 0.018 | 0.052 ± 0.001 | 0.308 ± 0.002 | 0.494 ± 0.023 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.117 ± 0.013 | 0.043 ± 0.001 | 0.121 ± 0.001 | 0.231 ± 0.014 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.182 ± 0.011 | 0.094 ± 0.001 | 0.456 ± 0.004 | 0.661 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.092 ± 0.040 | 0.021 ± 0.005 | 0.028 ± 0.001 | 0.226 ± 0.021 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.098 ± 0.058 | 0.024 ± 0.001 | 0.074 ± 0.001 | 0.249 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.094 ± 0.053 | 0.022 ± 0.002 | 0.075 ± 0.001 | 0.274 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.092 ± 0.015 | 0.022 ± 0.002 | 0.263 ± 0.010 | 0.476 ± 0.052 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.112 ± 0.024 | 0.034 ± 0.002 | 0.074 ± 0.003 | 0.146 ± 0.028 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.128 ± 0.018 | 0.058 ± 0.003 | 0.220 ± 0.004 | 0.336 ± 0.012 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.113 ± 0.018 | 0.035 ± 0.000 | 0.102 ± 0.003 | 0.176 ± 0.009 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.151 ± 0.015 | 0.068 ± 0.001 | 0.335 ± 0.008 | 0.479 ± 0.013 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.171 ± 0.009 | 0.111 ± 0.022 | 0.107 ± 0.005 | 0.240 ± 0.019 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.191 ± 0.025 | 0.135 ± 0.011 | 0.338 ± 0.005 | 0.524 ± 0.018 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.169 ± 0.022 | 0.121 ± 0.018 | 0.137 ± 0.006 | 0.261 ± 0.016 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.207 ± 0.021 | 0.132 ± 0.010 | 0.461 ± 0.009 | 0.657 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.130 ± 0.016 | 0.043 ± 0.004 | 0.066 ± 0.002 | 0.196 ± 0.031 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.169 ± 0.008 | 0.081 ± 0.001 | 0.225 ± 0.001 | 0.434 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.136 ± 0.029 | 0.062 ± 0.003 | 0.109 ± 0.001 | 0.230 ± 0.013 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.239 ± 0.015 | 0.158 ± 0.002 | 0.399 ± 0.004 | 0.648 ± 0.010 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.254 ± 0.045 | 0.101 ± 0.009 | 0.058 ± 0.003 | 1.016 ± 0.247 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.245 ± 0.088 | 0.102 ± 0.008 | 0.105 ± 0.002 | 0.814 ± 0.093 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.260 ± 0.028 | 0.101 ± 0.007 | 0.104 ± 0.001 | 0.873 ± 0.103 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.229 ± 0.058 | 0.102 ± 0.046 | 0.295 ± 0.006 | 0.987 ± 0.071 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.500 ± 0.038 | 0.191 ± 0.011 | 0.135 ± 0.003 | 0.898 ± 0.215 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.624 ± 0.035 | 0.252 ± 0.005 | 0.364 ± 0.006 | 1.009 ± 0.039 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.502 ± 0.026 | 0.191 ± 0.019 | 0.162 ± 0.004 | 0.874 ± 0.033 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.635 ± 0.030 | 0.258 ± 0.011 | 0.486 ± 0.010 | 1.159 ± 0.049 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | 0.381 ± 0.015 | 0.247 ± 0.011 | 0.747 ± 0.004 | 0.933 ± 0.061 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 3.153 ± 0.038 | 3.098 ± 0.012 | 11.009 ± 0.974 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 14.304 ± 0.111 | 14.620 ± 0.044 | 49.045 ± 4.577 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 3.119 ± 0.032 | 2.848 ± 0.031 | 17.440 ± 7.800 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 14.220 ± 0.023 | 13.056 ± 0.170 | 38.447 ± 0.768 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 0.836 ± 0.059 | 1.295 ± 0.013 | 5.133 ± 1.114 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 4.828 ± 0.300 | 7.979 ± 0.042 | 13.408 ± 0.568 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 0.773 ± 0.011 | 0.831 ± 0.013 | 4.217 ± 0.154 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 4.541 ± 0.053 | 4.517 ± 0.017 | 14.684 ± 1.175 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.116 ± 0.017 | 0.109 ± 0.009 | 1.003 ± 0.005 | 121.595 ± 110.293 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.250 ± 0.001 | 0.111 ± 0.003 | 1.014 ± 0.026 | 107.705 ± 39.009 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 1.982 ± 0.035 | 1.848 ± 0.021 | 88.564 ± 46.677 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 8.968 ± 0.194 | 8.738 ± 0.063 | 13.000 ± 0.883 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 1.966 ± 0.042 | 1.771 ± 0.035 | 8.878 ± 1.713 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 8.727 ± 0.028 | 8.363 ± 0.061 | 13.830 ± 0.193 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.615 ± 0.036 | 0.204 ± 0.011 | 0.585 ± 0.007 | 71.179 ± 53.924 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.458 ± 0.068 | 0.377 ± 0.007 | 1.023 ± 0.103 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 1.378 ± 0.007 | 1.286 ± 0.044 | 3.197 ± 0.185 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.449 ± 0.078 | 0.447 ± 0.006 | 1.299 ± 0.067 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 1.389 ± 0.017 | 1.995 ± 0.024 | 2.573 ± 0.132 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 0.880 ± 0.035 | 0.421 ± 0.014 | 0.847 ± 0.012 | 142.496 ± 118.907 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 5.231 ± 0.034 | 5.667 ± 0.041 | 162.323 ± 153.446 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 24.032 ± 0.032 | 26.169 ± 0.066 | 44.218 ± 2.311 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 5.218 ± 0.006 | 5.138 ± 0.024 | 302.928 ± 147.727 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 31.223 ± 7.563 | 30.670 ± 0.069 | 44.899 ± 3.259 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | 0.113 ± 0.016 | 0.031 ± 0.000 | 3.983 ± 0.043 | 12.555 ± 3.882 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | 0.201 ± 0.003 | 0.095 ± 0.000 | 15.991 ± 0.206 | 55.318 ± 35.681 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 0.531 ± 0.007 | 0.347 ± 0.003 | 64.602 ± 0.764 | 80.405 ± 5.829 | - | - |
| large | `qr` | f64 | 4 | `64x64` | 0.192 ± 0.019 | 0.102 ± 0.003 | 0.615 ± 0.013 | 0.764 ± 0.014 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.211 ± 0.004 | 0.115 ± 0.006 | 0.558 ± 0.007 | 86.211 ± 74.617 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.203 ± 0.003 | 0.113 ± 0.019 | 0.671 ± 0.005 | 44.810 ± 47.552 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.214 ± 0.011 | 0.098 ± 0.107 | 1.037 ± 0.009 | 69.359 ± 114.285 | - | - |
| large | `svd` | f64 | 4 | `64x64` | 0.416 ± 0.035 | 0.339 ± 0.015 | 0.826 ± 0.004 | 1.023 ± 0.028 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | 0.197 ± 0.019 | 0.093 ± 0.044 | 0.011 ± 0.001 | 0.041 ± 0.010 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | 0.135 ± 0.010 | 0.041 ± 0.058 | 0.014 ± 0.001 | 0.041 ± 0.012 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | 0.103 ± 0.012 | 0.088 ± 0.077 | 0.021 ± 0.000 | 0.047 ± 0.007 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.149 ± 0.003 | 0.025 ± 0.003 | 0.015 ± 0.001 | 0.135 ± 0.009 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.131 ± 0.025 | 0.025 ± 0.001 | 0.016 ± 0.001 | 0.112 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.100 ± 0.007 | 0.025 ± 0.022 | 0.028 ± 0.000 | 0.131 ± 0.004 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.108 ± 0.004 | 0.045 ± 0.003 | 0.151 ± 0.007 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.102 ± 0.007 | 0.044 ± 0.001 | 0.147 ± 0.010 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.112 ± 0.001 | 0.051 ± 0.001 | 0.499 ± 0.410 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.106 ± 0.002 | 0.040 ± 0.005 | 0.363 ± 0.020 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.106 ± 0.005 | 0.038 ± 0.002 | 0.362 ± 0.022 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.111 ± 0.002 | 0.045 ± 0.002 | 2.277 ± 1.698 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.105 ± 0.003 | 0.112 ± 0.010 | 0.217 ± 0.012 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.100 ± 0.004 | 0.107 ± 0.007 | 0.216 ± 0.007 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.098 ± 0.013 | 0.138 ± 0.006 | 3.080 ± 3.954 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.120 ± 0.004 | 0.083 ± 0.012 | 0.589 ± 0.038 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.119 ± 0.003 | 0.095 ± 0.018 | 0.586 ± 0.026 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.130 ± 0.005 | 0.112 ± 0.016 | 1.037 ± 0.291 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.389 ± 0.019 | 0.100 ± 0.009 | 0.025 ± 0.004 | 0.391 ± 0.024 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.283 ± 0.063 | 0.057 ± 0.057 | 0.028 ± 0.001 | 0.397 ± 0.022 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.277 ± 0.033 | 0.102 ± 0.010 | 0.043 ± 0.001 | 0.419 ± 0.013 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.224 ± 0.018 | 0.080 ± 0.017 | 0.218 ± 0.012 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.219 ± 0.014 | 0.069 ± 0.006 | 0.221 ± 0.014 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.217 ± 0.006 | 0.079 ± 0.002 | 70.858 ± 135.975 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.240 ± 0.006 | 0.077 ± 0.018 | 0.555 ± 0.041 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.238 ± 0.009 | 0.069 ± 0.005 | 0.534 ± 0.026 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.246 ± 0.004 | 0.079 ± 0.011 | 6.297 ± 3.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.568 ± 0.051 | 0.182 ± 0.022 | 0.037 ± 0.001 | 0.434 ± 0.023 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.492 ± 0.030 | 0.180 ± 0.028 | 0.039 ± 0.001 | 0.440 ± 0.033 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.443 ± 0.070 | 0.173 ± 0.012 | 0.045 ± 0.001 | 6.421 ± 67.557 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.103 ± 0.001 | 0.107 ± 0.005 | 0.178 ± 0.018 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.103 ± 0.007 | 0.106 ± 0.003 | 0.178 ± 0.010 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.102 ± 0.013 | 0.105 ± 0.003 | 0.316 ± 0.062 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.112 ± 0.001 | 0.053 ± 0.002 | 0.380 ± 0.034 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.115 ± 0.006 | 0.053 ± 0.002 | 0.389 ± 0.023 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.113 ± 0.009 | 0.054 ± 0.005 | 1.253 ± 1.043 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.377 ± 0.037 | 0.089 ± 0.019 | 0.024 ± 0.001 | 0.427 ± 0.031 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.277 ± 0.020 | 0.098 ± 0.010 | 0.028 ± 0.002 | 0.448 ± 0.045 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.253 ± 0.025 | 0.112 ± 0.003 | 0.037 ± 0.001 | 0.477 ± 0.048 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.104 ± 0.006 | 0.064 ± 0.005 | 0.161 ± 0.013 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.102 ± 0.001 | 0.063 ± 0.001 | 0.159 ± 0.006 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.125 ± 0.008 | 0.070 ± 0.001 | 44.674 ± 47.250 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.096 ± 0.011 | 0.048 ± 0.008 | 0.389 ± 0.023 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.101 ± 0.022 | 0.042 ± 0.003 | 0.399 ± 0.027 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.112 ± 0.007 | 0.045 ± 0.002 | 2.797 ± 0.761 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | 0.182 ± 0.030 | 0.025 ± 0.002 | 0.009 ± 0.001 | 0.062 ± 0.008 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | 0.115 ± 0.022 | 0.021 ± 0.003 | 0.012 ± 0.000 | 0.048 ± 0.005 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | 0.114 ± 0.047 | 0.023 ± 0.001 | 0.022 ± 0.000 | 0.072 ± 0.005 | - | - |
| small | `qr` | f64 | 4 | `2x2` | 0.193 ± 0.044 | 0.046 ± 0.021 | 0.026 ± 0.015 | 0.041 ± 0.006 | - | - |
| small | `qr` | f64 | 4 | `4x4` | 0.130 ± 0.005 | 0.022 ± 0.083 | 0.019 ± 0.005 | 0.035 ± 0.009 | - | - |
| small | `qr` | f64 | 4 | `8x8` | 0.123 ± 0.035 | 0.051 ± 0.069 | 0.030 ± 0.003 | 0.039 ± 0.009 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.252 ± 0.025 | 0.095 ± 0.012 | 0.018 ± 0.000 | 0.052 ± 0.005 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.252 ± 0.031 | 0.098 ± 0.054 | 0.018 ± 0.000 | 0.054 ± 0.006 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.173 ± 0.017 | 0.103 ± 0.009 | 0.020 ± 0.000 | 0.055 ± 0.007 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.183 ± 0.014 | 0.102 ± 0.008 | 0.021 ± 0.000 | 0.057 ± 0.007 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.155 ± 0.009 | 0.107 ± 0.002 | 0.026 ± 0.000 | 0.061 ± 0.005 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.159 ± 0.014 | 0.094 ± 0.044 | 0.028 ± 0.000 | 0.062 ± 0.006 | - | - |
| small | `svd` | f64 | 4 | `2x2` | 0.276 ± 0.043 | 0.044 ± 0.013 | 0.010 ± 0.001 | 0.053 ± 0.015 | - | - |
| small | `svd` | f64 | 4 | `4x4` | 0.149 ± 0.024 | 0.100 ± 0.009 | 0.012 ± 0.001 | 0.039 ± 0.009 | - | - |
| small | `svd` | f64 | 4 | `8x8` | 0.122 ± 0.021 | 0.033 ± 0.072 | 0.022 ± 0.001 | 0.049 ± 0.010 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.274 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`jax-cpu`, 0.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 17.5x faster than the slowest successful cell (`jax-cpu`, 1.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 1.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=4, shape=`64x64`): `pytorch-cpu` is 121.2x faster than the slowest successful cell (`jax-cpu`, 121.595 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=4, shape=`64x64`): `tenferro-eager` is 1046.7x faster than the slowest successful cell (`jax-cpu`, 121.595 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=4, shape=`64x64`): `tenferro-trace` is 1113.4x faster than the slowest successful cell (`jax-cpu`, 121.595 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=4, shape=`64x64`): `pytorch-cpu` is 106.2x faster than the slowest successful cell (`jax-cpu`, 107.705 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=4, shape=`64x64`): `tenferro-eager` is 430.6x faster than the slowest successful cell (`jax-cpu`, 107.705 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=4, shape=`64x64`): `tenferro-trace` is 971.4x faster than the slowest successful cell (`jax-cpu`, 107.705 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 47.9x faster than the slowest successful cell (`jax-cpu`, 88.564 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `tenferro-trace` is 44.7x faster than the slowest successful cell (`jax-cpu`, 88.564 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`64x64,rhs=1`): `pytorch-cpu` is 121.6x faster than the slowest successful cell (`jax-cpu`, 71.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`64x64,rhs=1`): `tenferro-eager` is 115.7x faster than the slowest successful cell (`jax-cpu`, 71.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`64x64,rhs=1`): `tenferro-trace` is 348.8x faster than the slowest successful cell (`jax-cpu`, 71.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_svd_s_backward` (f64, threads=4, shape=`64x64`): `pytorch-cpu` is 168.2x faster than the slowest successful cell (`jax-cpu`, 142.496 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_svd_s_backward` (f64, threads=4, shape=`64x64`): `tenferro-eager` is 161.9x faster than the slowest successful cell (`jax-cpu`, 142.496 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_svd_s_backward` (f64, threads=4, shape=`64x64`): `tenferro-trace` is 338.5x faster than the slowest successful cell (`jax-cpu`, 142.496 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_svd_s_jvp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 28.6x faster than the slowest successful cell (`jax-cpu`, 162.323 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_svd_s_jvp` (f64, threads=4, shape=`256x256`): `tenferro-trace` is 31.0x faster than the slowest successful cell (`jax-cpu`, 162.323 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_svd_s_vjp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 59.0x faster than the slowest successful cell (`jax-cpu`, 302.928 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_svd_s_vjp` (f64, threads=4, shape=`256x256`): `tenferro-trace` is 58.1x faster than the slowest successful cell (`jax-cpu`, 302.928 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`128x128`): `tenferro-eager` is 111.4x faster than the slowest successful cell (`jax-cpu`, 12.555 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`128x128`): `tenferro-trace` is 409.4x faster than the slowest successful cell (`jax-cpu`, 12.555 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`256x256`): `tenferro-eager` is 274.8x faster than the slowest successful cell (`jax-cpu`, 55.318 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`256x256`): `tenferro-trace` is 580.5x faster than the slowest successful cell (`jax-cpu`, 55.318 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=4, shape=`256x1024 * 1024x256`): `tenferro-eager` is 151.5x faster than the slowest successful cell (`jax-cpu`, 80.405 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=4, shape=`256x1024 * 1024x256`): `tenferro-trace` is 231.6x faster than the slowest successful cell (`jax-cpu`, 80.405 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`64x64,rhs=1`): `pytorch-cpu` is 154.6x faster than the slowest successful cell (`jax-cpu`, 86.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`64x64,rhs=1`): `tenferro-eager` is 409.2x faster than the slowest successful cell (`jax-cpu`, 86.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`64x64,rhs=1`): `tenferro-trace` is 748.8x faster than the slowest successful cell (`jax-cpu`, 86.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`64x64,rhs=16`): `pytorch-cpu` is 66.7x faster than the slowest successful cell (`jax-cpu`, 44.810 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`64x64,rhs=16`): `tenferro-eager` is 220.4x faster than the slowest successful cell (`jax-cpu`, 44.810 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`64x64,rhs=16`): `tenferro-trace` is 397.9x faster than the slowest successful cell (`jax-cpu`, 44.810 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`64x64,rhs=64`): `pytorch-cpu` is 66.9x faster than the slowest successful cell (`jax-cpu`, 69.359 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`64x64,rhs=64`): `tenferro-eager` is 324.2x faster than the slowest successful cell (`jax-cpu`, 69.359 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`64x64,rhs=64`): `tenferro-trace` is 708.3x faster than the slowest successful cell (`jax-cpu`, 69.359 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/eigh` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 17.9x faster than the slowest successful cell (`tenferro-eager`, 0.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`tenferro-eager`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 50.5x faster than the slowest successful cell (`jax-cpu`, 2.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 20.4x faster than the slowest successful cell (`jax-cpu`, 2.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 22.3x faster than the slowest successful cell (`jax-cpu`, 3.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 31.4x faster than the slowest successful cell (`jax-cpu`, 3.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.391 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.397 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 891.8x faster than the slowest successful cell (`jax-cpu`, 70.858 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 326.2x faster than the slowest successful cell (`jax-cpu`, 70.858 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 79.5x faster than the slowest successful cell (`jax-cpu`, 6.297 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 25.6x faster than the slowest successful cell (`jax-cpu`, 6.297 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`2x2,rhs=1`): `pytorch-cpu` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 0.568 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`4x4,rhs=1`): `pytorch-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 0.492 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`8x8,rhs=1`): `pytorch-cpu` is 144.2x faster than the slowest successful cell (`jax-cpu`, 6.421 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-eager` is 14.5x faster than the slowest successful cell (`jax-cpu`, 6.421 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 37.0x faster than the slowest successful cell (`jax-cpu`, 6.421 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `pytorch-cpu` is 23.3x faster than the slowest successful cell (`jax-cpu`, 1.253 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`jax-cpu`, 1.253 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 17.9x faster than the slowest successful cell (`jax-cpu`, 0.427 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 16.1x faster than the slowest successful cell (`jax-cpu`, 0.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 13.0x faster than the slowest successful cell (`jax-cpu`, 0.477 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 633.7x faster than the slowest successful cell (`jax-cpu`, 44.674 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 356.3x faster than the slowest successful cell (`jax-cpu`, 44.674 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 62.4x faster than the slowest successful cell (`jax-cpu`, 2.797 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 24.9x faster than the slowest successful cell (`jax-cpu`, 2.797 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 19.2x faster than the slowest successful cell (`tenferro-eager`, 0.182 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/solve` (f64, threads=4, shape=`2x2,rhs=1`): `pytorch-cpu` is 14.1x faster than the slowest successful cell (`tenferro-eager`, 0.252 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/solve` (f64, threads=4, shape=`2x2,rhs=4`): `pytorch-cpu` is 13.7x faster than the slowest successful cell (`tenferro-eager`, 0.252 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/svd` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 27.2x faster than the slowest successful cell (`tenferro-eager`, 0.276 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/svd` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`tenferro-eager`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

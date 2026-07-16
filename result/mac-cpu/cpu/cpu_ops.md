# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260716_171807`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260716_171807`.

- tenferro-rs commit: `3abc9108e4f0500f4e75519711cdb4e21b9625df`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260716_171807/cpu_ops_t4_20260716_171807.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260716_171807/cpu_ops_t4_20260716_171807.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.077 ± 0.003 | 0.008 ± 0.001 | 0.086 ± 0.001 | 0.123 ± 0.020 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.099 ± 0.008 | 0.028 ± 0.000 | 0.330 ± 0.043 | 0.365 ± 0.008 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.083 ± 0.003 | 0.018 ± 0.000 | 0.123 ± 0.000 | 0.150 ± 0.006 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.148 ± 0.008 | 0.073 ± 0.002 | 0.468 ± 0.013 | 0.526 ± 0.019 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.075 ± 0.010 | 0.001 ± 0.000 | 0.029 ± 0.001 | 0.127 ± 0.006 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.071 ± 0.001 | 0.003 ± 0.000 | 0.076 ± 0.003 | 0.180 ± 0.006 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.070 ± 0.004 | 0.002 ± 0.000 | 0.078 ± 0.003 | 0.190 ± 0.035 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.075 ± 0.010 | 0.004 ± 0.001 | 0.270 ± 0.007 | 0.374 ± 0.022 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.077 ± 0.003 | 0.009 ± 0.000 | 0.078 ± 0.005 | 0.087 ± 0.008 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.098 ± 0.004 | 0.031 ± 0.000 | 0.223 ± 0.003 | 0.238 ± 0.017 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.078 ± 0.004 | 0.012 ± 0.000 | 0.102 ± 0.002 | 0.114 ± 0.005 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.118 ± 0.005 | 0.045 ± 0.000 | 0.347 ± 0.014 | 0.377 ± 0.010 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.080 ± 0.002 | 0.012 ± 0.000 | 0.113 ± 0.004 | 0.138 ± 0.008 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.104 ± 0.006 | 0.036 ± 0.000 | 0.346 ± 0.002 | 0.454 ± 0.104 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.081 ± 0.006 | 0.012 ± 0.000 | 0.140 ± 0.004 | 0.169 ± 0.002 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.111 ± 0.013 | 0.040 ± 0.006 | 0.476 ± 0.006 | 0.535 ± 0.017 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.080 ± 0.003 | 0.014 ± 0.000 | 0.065 ± 0.001 | 0.122 ± 0.009 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.121 ± 0.006 | 0.051 ± 0.000 | 0.227 ± 0.002 | 0.320 ± 0.013 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.100 ± 0.004 | 0.033 ± 0.001 | 0.110 ± 0.001 | 0.148 ± 0.004 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.203 ± 0.007 | 0.160 ± 0.082 | 0.414 ± 0.035 | 0.519 ± 0.081 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.129 ± 0.013 | 0.041 ± 0.010 | 0.060 ± 0.002 | 0.541 ± 0.073 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.147 ± 0.031 | 0.046 ± 0.007 | 0.108 ± 0.003 | 0.576 ± 0.052 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.140 ± 0.007 | 0.024 ± 0.008 | 0.109 ± 0.003 | 0.639 ± 0.125 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.139 ± 0.045 | 0.054 ± 0.014 | 0.305 ± 0.003 | 0.830 ± 0.079 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.236 ± 0.087 | 0.141 ± 0.021 | 0.160 ± 0.026 | 0.550 ± 0.041 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.249 ± 0.025 | 0.207 ± 0.044 | 0.377 ± 0.008 | 0.815 ± 0.166 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.275 ± 0.112 | 0.167 ± 0.039 | 0.168 ± 0.012 | 0.618 ± 0.235 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.262 ± 0.025 | 0.208 ± 0.013 | 0.501 ± 0.015 | 0.972 ± 0.269 |
| large | `eigh` | f64 | 4 | `64x64` | 0.317 ± 0.022 | 0.218 ± 0.005 | 0.748 ± 0.078 | 0.780 ± 0.035 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 3.409 ± 0.301 | 3.244 ± 0.052 | 3.822 ± 0.455 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 15.022 ± 0.113 | 15.449 ± 0.184 | 17.894 ± 0.370 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 3.194 ± 0.276 | 2.957 ± 0.078 | 3.673 ± 0.086 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 14.898 ± 0.052 | 13.948 ± 0.213 | 16.951 ± 0.189 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 0.872 ± 0.035 | 1.360 ± 0.048 | 1.136 ± 0.039 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 9.339 ± 7.509 | 8.630 ± 0.934 | 5.317 ± 0.369 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 0.921 ± 0.108 | 0.853 ± 0.026 | 1.340 ± 0.580 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 5.198 ± 0.618 | 4.998 ± 0.301 | 5.846 ± 0.686 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.085 ± 0.006 | 0.017 ± 0.013 | 1.017 ± 0.076 | 1.068 ± 0.067 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.147 ± 0.017 | 0.047 ± 0.011 | 1.043 ± 0.045 | 1.488 ± 0.335 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 1.945 ± 0.004 | 1.893 ± 0.087 | 2.329 ± 0.305 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 9.403 ± 0.121 | 9.211 ± 0.204 | 10.523 ± 1.451 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 2.039 ± 0.048 | 1.792 ± 0.054 | 4.552 ± 1.869 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 9.273 ± 0.063 | 8.773 ± 0.356 | 10.837 ± 0.608 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.366 ± 0.116 | 0.162 ± 0.004 | 0.585 ± 0.027 | 0.981 ± 0.071 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.262 ± 0.008 | 0.400 ± 0.050 | 0.468 ± 0.062 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 1.426 ± 0.221 | 1.341 ± 0.149 | 1.378 ± 0.120 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.301 ± 0.044 | 0.459 ± 0.018 | 0.501 ± 0.022 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 1.385 ± 0.138 | 2.123 ± 0.180 | 1.399 ± 0.188 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 0.674 ± 0.010 | 0.355 ± 0.027 | 0.859 ± 0.058 | 1.034 ± 0.152 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 5.259 ± 0.039 | 5.792 ± 0.063 | 6.251 ± 0.104 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 25.045 ± 0.521 | 26.915 ± 0.273 | 28.483 ± 0.397 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 5.205 ± 0.037 | 5.221 ± 0.081 | 5.965 ± 0.169 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 32.078 ± 0.235 | 31.634 ± 0.575 | 34.934 ± 0.923 |
| large | `matmul` | f64 | 4 | `128x128` | 0.100 ± 0.005 | 0.017 ± 0.001 | 4.009 ± 0.033 | 3.975 ± 0.079 |
| large | `matmul` | f64 | 4 | `256x256` | 0.179 ± 0.002 | 0.117 ± 0.006 | 16.280 ± 0.303 | 15.943 ± 1.683 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 0.536 ± 0.008 | 0.502 ± 0.010 | 64.959 ± 0.409 | 63.739 ± 1.335 |
| large | `qr` | f64 | 4 | `64x64` | 0.129 ± 0.004 | 0.083 ± 0.006 | 0.630 ± 0.014 | 0.689 ± 0.093 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.134 ± 0.002 | 0.018 ± 0.001 | 0.566 ± 0.024 | 0.582 ± 0.011 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.136 ± 0.007 | 0.021 ± 0.002 | 0.679 ± 0.022 | 0.723 ± 0.027 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.148 ± 0.006 | 0.031 ± 0.002 | 1.054 ± 0.024 | 1.051 ± 0.055 |
| large | `svd` | f64 | 4 | `64x64` | 0.371 ± 0.007 | 0.304 ± 0.012 | 0.832 ± 0.018 | 0.892 ± 0.057 |
| small | `eigh` | f64 | 4 | `2x2` | 0.137 ± 0.009 | 0.001 ± 0.000 | 0.012 ± 0.002 | 0.042 ± 0.009 |
| small | `eigh` | f64 | 4 | `4x4` | 0.092 ± 0.012 | 0.002 ± 0.000 | 0.014 ± 0.002 | 0.038 ± 0.007 |
| small | `eigh` | f64 | 4 | `8x8` | 0.082 ± 0.011 | 0.004 ± 0.000 | 0.021 ± 0.000 | 0.049 ± 0.014 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.109 ± 0.016 | 0.001 ± 0.000 | 0.015 ± 0.001 | 0.126 ± 0.008 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.094 ± 0.005 | 0.001 ± 0.000 | 0.017 ± 0.001 | 0.120 ± 0.009 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.083 ± 0.011 | 0.001 ± 0.000 | 0.029 ± 0.001 | 0.133 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.058 ± 0.013 | 0.043 ± 0.003 | 0.153 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.054 ± 0.018 | 0.045 ± 0.003 | 0.172 ± 0.053 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.074 ± 0.038 | 0.051 ± 0.002 | 0.158 ± 0.008 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.092 ± 0.032 | 0.040 ± 0.008 | 0.353 ± 0.020 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.092 ± 0.017 | 0.040 ± 0.009 | 0.353 ± 0.026 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.065 ± 0.049 | 0.045 ± 0.002 | 0.394 ± 0.086 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.044 ± 0.018 | 0.110 ± 0.004 | 0.220 ± 0.013 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.035 ± 0.006 | 0.132 ± 0.006 | 0.215 ± 0.005 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.086 ± 0.048 | 0.136 ± 0.006 | 0.212 ± 0.015 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.048 ± 0.034 | 0.087 ± 0.015 | 0.571 ± 0.011 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.041 ± 0.010 | 0.108 ± 0.011 | 0.584 ± 0.036 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.103 ± 0.016 | 0.108 ± 0.005 | 0.629 ± 0.100 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.259 ± 0.016 | 0.033 ± 0.011 | 0.027 ± 0.004 | 0.456 ± 0.145 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.171 ± 0.006 | 0.026 ± 0.014 | 0.028 ± 0.001 | 0.418 ± 0.024 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.145 ± 0.016 | 0.053 ± 0.023 | 0.045 ± 0.002 | 0.428 ± 0.027 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.113 ± 0.021 | 0.086 ± 0.013 | 0.214 ± 0.008 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.098 ± 0.039 | 0.070 ± 0.008 | 0.226 ± 0.026 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.153 ± 0.057 | 0.082 ± 0.025 | 0.216 ± 0.010 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.203 ± 0.007 | 0.076 ± 0.026 | 0.521 ± 0.041 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.202 ± 0.004 | 0.082 ± 0.044 | 0.534 ± 0.023 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.183 ± 0.008 | 0.088 ± 0.020 | 0.667 ± 0.246 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.291 ± 0.044 | 0.256 ± 0.089 | 0.039 ± 0.003 | 0.437 ± 0.025 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.226 ± 0.017 | 0.097 ± 0.015 | 0.039 ± 0.001 | 0.434 ± 0.031 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.215 ± 0.208 | 0.108 ± 0.030 | 0.046 ± 0.001 | 0.475 ± 0.207 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.025 ± 0.011 | 0.123 ± 0.045 | 0.177 ± 0.012 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.020 ± 0.005 | 0.110 ± 0.005 | 0.174 ± 0.014 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.044 ± 0.046 | 0.111 ± 0.004 | 0.193 ± 0.013 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.073 ± 0.005 | 0.060 ± 0.009 | 0.376 ± 0.039 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.066 ± 0.024 | 0.055 ± 0.003 | 0.499 ± 0.171 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.199 ± 0.168 | 0.058 ± 0.003 | 0.429 ± 0.144 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.654 ± 0.196 | 0.030 ± 0.006 | 0.025 ± 0.001 | 0.441 ± 0.029 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.436 ± 0.302 | 0.025 ± 0.005 | 0.028 ± 0.001 | 0.448 ± 0.046 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.380 ± 0.014 | 0.037 ± 0.016 | 0.038 ± 0.001 | 0.459 ± 0.020 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.023 ± 0.002 | 0.063 ± 0.005 | 0.165 ± 0.013 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.026 ± 0.010 | 0.064 ± 0.004 | 0.162 ± 0.029 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.040 ± 0.013 | 0.071 ± 0.002 | 0.160 ± 0.014 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.025 ± 0.008 | 0.051 ± 0.011 | 0.381 ± 0.025 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.025 ± 0.005 | 0.043 ± 0.002 | 0.401 ± 0.029 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.051 ± 0.015 | 0.046 ± 0.003 | 0.401 ± 0.026 |
| small | `matmul` | f64 | 4 | `2x2` | 0.118 ± 0.012 | 0.001 ± 0.002 | 0.009 ± 0.002 | 0.058 ± 0.008 |
| small | `matmul` | f64 | 4 | `4x4` | 0.097 ± 0.008 | 0.001 ± 0.000 | 0.012 ± 0.000 | 0.051 ± 0.005 |
| small | `matmul` | f64 | 4 | `8x8` | 0.085 ± 0.012 | 0.001 ± 0.000 | 0.023 ± 0.000 | 0.076 ± 0.009 |
| small | `qr` | f64 | 4 | `2x2` | 0.138 ± 0.046 | 0.001 ± 0.000 | 0.022 ± 0.008 | 0.043 ± 0.007 |
| small | `qr` | f64 | 4 | `4x4` | 0.086 ± 0.006 | 0.001 ± 0.000 | 0.023 ± 0.010 | 0.033 ± 0.008 |
| small | `qr` | f64 | 4 | `8x8` | 0.079 ± 0.007 | 0.002 ± 0.000 | 0.031 ± 0.003 | 0.040 ± 0.007 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.127 ± 0.022 | 0.003 ± 0.000 | 0.018 ± 0.001 | 0.051 ± 0.007 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.121 ± 0.006 | 0.003 ± 0.000 | 0.019 ± 0.000 | 0.054 ± 0.004 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.090 ± 0.004 | 0.003 ± 0.000 | 0.020 ± 0.001 | 0.052 ± 0.005 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.086 ± 0.007 | 0.003 ± 0.000 | 0.021 ± 0.000 | 0.059 ± 0.009 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.083 ± 0.013 | 0.003 ± 0.000 | 0.026 ± 0.000 | 0.063 ± 0.005 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.084 ± 0.007 | 0.003 ± 0.000 | 0.029 ± 0.001 | 0.064 ± 0.003 |
| small | `svd` | f64 | 4 | `2x2` | 0.109 ± 0.003 | 0.002 ± 0.000 | 0.010 ± 0.002 | 0.053 ± 0.012 |
| small | `svd` | f64 | 4 | `4x4` | 0.088 ± 0.006 | 0.003 ± 0.000 | 0.013 ± 0.001 | 0.037 ± 0.007 |
| small | `svd` | f64 | 4 | `8x8` | 0.084 ± 0.004 | 0.007 ± 0.000 | 0.022 ± 0.001 | 0.046 ± 0.009 |

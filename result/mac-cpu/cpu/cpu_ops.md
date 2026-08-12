# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260812_150722`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260812_150722`.

- tenferro-rs commit: `7f91eb7dadca2227a3043dc9e97997a2d287237c`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260812_150722/cpu_ops_t4_20260812_150722.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260812_150722/cpu_ops_t4_20260812_150722.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.019 ± 0.007 | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.045 ± 0.018 | 0.023 ± 0.000 | 0.012 ± 0.000 | 0.038 ± 0.005 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.029 ± 0.001 | 0.016 ± 0.000 | 0.014 ± 0.001 | 0.019 ± 0.001 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.083 ± 0.022 | 0.059 ± 0.001 | 0.046 ± 0.000 | 0.074 ± 0.004 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | 0.051 ± 0.004 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.003 ± 0.000 | 0.003 ± 0.002 | 0.005 ± 0.000 | 0.062 ± 0.008 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.068 ± 0.006 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.073 ± 0.007 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.019 ± 0.001 | 0.007 ± 0.000 | 0.019 ± 0.004 | 0.008 ± 0.001 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.040 ± 0.003 | 0.024 ± 0.000 | 0.025 ± 0.003 | 0.016 ± 0.002 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.022 ± 0.002 | 0.011 ± 0.001 | 0.022 ± 0.003 | 0.011 ± 0.001 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.052 ± 0.008 | 0.037 ± 0.000 | 0.033 ± 0.002 | 0.042 ± 0.003 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.097 ± 0.023 | 0.009 ± 0.000 | 0.022 ± 0.003 | 0.007 ± 0.001 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.090 ± 0.018 | 0.029 ± 0.001 | 0.024 ± 0.001 | 0.027 ± 0.001 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.051 ± 0.007 | 0.010 ± 0.000 | 0.024 ± 0.006 | 0.008 ± 0.001 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.123 ± 0.029 | 0.032 ± 0.000 | 0.026 ± 0.004 | 0.030 ± 0.003 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.028 ± 0.040 | 0.012 ± 0.000 | 0.009 ± 0.000 | 0.024 ± 0.001 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.056 ± 0.005 | 0.043 ± 0.000 | 0.023 ± 0.000 | 0.073 ± 0.002 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.041 ± 0.002 | 0.029 ± 0.001 | 0.026 ± 0.000 | 0.039 ± 0.001 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.158 ± 0.032 | 0.109 ± 0.003 | 0.091 ± 0.000 | 0.159 ± 0.009 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.114 ± 0.032 | 0.037 ± 0.010 | 0.029 ± 0.002 | 0.380 ± 0.035 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.084 ± 0.015 | 0.056 ± 0.025 | 0.030 ± 0.001 | 0.393 ± 0.037 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.077 ± 0.003 | 0.039 ± 0.009 | 0.030 ± 0.004 | 0.388 ± 0.046 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.083 ± 0.014 | 0.038 ± 0.007 | 0.031 ± 0.001 | 0.398 ± 0.057 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.208 ± 0.072 | 0.097 ± 0.012 | 0.041 ± 0.007 | 0.319 ± 0.011 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.197 ± 0.019 | 0.141 ± 0.032 | 0.047 ± 0.004 | 0.341 ± 0.034 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.154 ± 0.013 | 0.153 ± 0.017 | 0.041 ± 0.007 | 0.325 ± 0.014 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.200 ± 0.027 | 0.139 ± 0.008 | 0.049 ± 0.004 | 0.330 ± 0.016 |
| large | `eigh` | f64 | 4 | `64x64` | 0.264 ± 0.032 | 0.207 ± 0.003 | 0.200 ± 0.003 | 0.234 ± 0.016 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.023 ± 0.002 | 0.017 ± 0.002 | 0.006 ± 0.001 | 0.038 ± 0.011 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.089 ± 0.009 | 0.044 ± 0.004 | 0.026 ± 0.008 | 0.335 ± 0.055 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.222 ± 0.038 | 0.124 ± 0.014 | 0.039 ± 0.002 | 0.356 ± 0.046 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 0.664 ± 0.055 | 0.346 ± 0.018 | 0.297 ± 0.004 | 0.471 ± 0.104 |
| large | `matmul` | f64 | 4 | `128x128` | 0.018 ± 0.001 | 0.015 ± 0.002 | 0.017 ± 0.007 | 0.100 ± 0.016 |
| large | `matmul` | f64 | 4 | `256x256` | 0.111 ± 0.008 | 0.094 ± 0.001 | 0.095 ± 0.013 | 0.184 ± 0.013 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 0.469 ± 0.120 | 0.455 ± 0.018 | 0.366 ± 0.015 | 0.418 ± 0.012 |
| large | `qr` | f64 | 4 | `64x64` | 0.086 ± 0.007 | 0.068 ± 0.000 | 0.089 ± 0.013 | 0.091 ± 0.010 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.118 ± 0.030 | 0.016 ± 0.000 | 0.019 ± 0.007 | 0.043 ± 0.006 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.115 ± 0.007 | 0.018 ± 0.001 | 0.020 ± 0.000 | 0.045 ± 0.007 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.117 ± 0.012 | 0.031 ± 0.005 | 0.026 ± 0.004 | 0.053 ± 0.011 |
| large | `svd` | f64 | 4 | `64x64` | 0.298 ± 0.008 | 0.286 ± 0.003 | 0.280 ± 0.005 | 0.313 ± 0.021 |
| small | `eigh` | f64 | 4 | `2x2` | 0.013 ± 0.003 | 0.001 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.002 |
| small | `eigh` | f64 | 4 | `4x4` | 0.019 ± 0.004 | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 |
| small | `eigh` | f64 | 4 | `8x8` | 0.017 ± 0.002 | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.002 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.007 ± 0.004 | 0.001 ± 0.000 | 0.005 ± 0.000 | 0.049 ± 0.005 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.001 | 0.052 ± 0.004 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.000 | 0.059 ± 0.005 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.128 ± 0.020 | 0.039 ± 0.005 | 0.013 ± 0.002 | 0.290 ± 0.024 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.099 ± 0.012 | 0.038 ± 0.011 | 0.013 ± 0.001 | 0.299 ± 0.026 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.092 ± 0.060 | 0.036 ± 0.005 | 0.015 ± 0.002 | 0.296 ± 0.020 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.197 ± 0.021 | 0.069 ± 0.007 | 0.021 ± 0.001 | 0.320 ± 0.012 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.191 ± 0.057 | 0.072 ± 0.002 | 0.020 ± 0.002 | 0.411 ± 0.121 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.188 ± 0.067 | 0.071 ± 0.003 | 0.021 ± 0.001 | 0.325 ± 0.019 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.529 ± 0.144 | 0.033 ± 0.007 | 0.016 ± 0.001 | 0.407 ± 0.043 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.404 ± 0.022 | 0.033 ± 0.003 | 0.017 ± 0.001 | 0.352 ± 0.018 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.550 ± 0.091 | 0.040 ± 0.008 | 0.021 ± 0.001 | 0.369 ± 0.016 |
| small | `matmul` | f64 | 4 | `2x2` | 0.006 ± 0.003 | 0.001 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.001 |
| small | `matmul` | f64 | 4 | `4x4` | 0.003 ± 0.002 | 0.001 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.001 |
| small | `matmul` | f64 | 4 | `8x8` | 0.002 ± 0.001 | 0.001 ± 0.000 | 0.001 ± 0.000 | 0.018 ± 0.012 |
| small | `qr` | f64 | 4 | `2x2` | 0.014 ± 0.008 | 0.001 ± 0.000 | 0.018 ± 0.008 | 0.006 ± 0.001 |
| small | `qr` | f64 | 4 | `4x4` | 0.014 ± 0.001 | 0.001 ± 0.000 | 0.016 ± 0.003 | 0.006 ± 0.001 |
| small | `qr` | f64 | 4 | `8x8` | 0.016 ± 0.003 | 0.002 ± 0.000 | 0.013 ± 0.001 | 0.007 ± 0.002 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.064 ± 0.010 | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.052 ± 0.007 | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.054 ± 0.002 | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.001 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.051 ± 0.008 | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.077 ± 0.025 | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.001 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.065 ± 0.017 | 0.003 ± 0.001 | 0.006 ± 0.000 | 0.006 ± 0.001 |
| small | `svd` | f64 | 4 | `2x2` | 0.026 ± 0.004 | 0.001 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 |
| small | `svd` | f64 | 4 | `4x4` | 0.023 ± 0.007 | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 |
| small | `svd` | f64 | 4 | `8x8` | 0.026 ± 0.016 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 |

# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Timestamp: `20260606_101358`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/cpu/einsum/20260606_101358`.

- tenferro-rs commit: `b17186e352184fd4244be604d6482b82a1264aef`

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
- Python platform: `macOS-26.5-arm64-arm-64bit`

## Thread Environment

- OMP_NUM_THREADS: `1`
- OMP_THREAD_LIMIT: `1`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `1`
- OPENBLAS_NUM_THREADS: `1`
- GOTO_NUM_THREADS: `1`
- MKL_NUM_THREADS: `1`
- VECLIB_MAXIMUM_THREADS: `1`
- VECLIB_NUM_THREADS: `1`
- NUMEXPR_NUM_THREADS: `1`
- BLIS_NUM_THREADS: `1`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 1

- CSV: `data/results/cpu/einsum/20260606_101358/cpu_ops_t1_20260606_101358.csv`
- Source table: `data/results/cpu/einsum/20260606_101358/cpu_ops_t1_20260606_101358.md`

## PR884 CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.009 ± 0.000 | 0.006 ± 0.000 | - | 0.075 ± 0.002 | 0.097 ± 0.004 |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.029 ± 0.000 | 0.020 ± 0.000 | - | 0.276 ± 0.007 | 0.329 ± 0.049 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.022 ± 0.000 | 0.015 ± 0.000 | - | 0.110 ± 0.003 | 0.129 ± 0.006 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.075 ± 0.000 | 0.056 ± 0.001 | - | 0.416 ± 0.010 | 0.467 ± 0.019 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.025 ± 0.002 | 0.107 ± 0.004 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.004 ± 0.000 | 0.002 ± 0.000 | - | 0.066 ± 0.007 | 0.159 ± 0.020 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.064 ± 0.003 | 0.162 ± 0.021 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.005 ± 0.000 | 0.003 ± 0.000 | - | 0.223 ± 0.007 | 0.327 ± 0.008 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.011 ± 0.000 | 0.008 ± 0.000 | - | 0.052 ± 0.003 | 0.073 ± 0.005 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.035 ± 0.000 | 0.026 ± 0.000 | - | 0.194 ± 0.033 | 0.208 ± 0.006 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.014 ± 0.000 | 0.010 ± 0.000 | - | 0.077 ± 0.001 | 0.096 ± 0.008 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.047 ± 0.000 | 0.037 ± 0.000 | - | 0.294 ± 0.006 | 0.366 ± 0.062 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.016 ± 0.000 | 0.011 ± 0.000 | - | 0.084 ± 0.011 | 0.115 ± 0.011 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.049 ± 0.000 | 0.037 ± 0.009 | - | 0.293 ± 0.007 | 0.343 ± 0.009 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.018 ± 0.000 | 0.012 ± 0.001 | - | 0.121 ± 0.014 | 0.145 ± 0.005 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.058 ± 0.005 | 0.040 ± 0.001 | - | 0.409 ± 0.019 | 0.464 ± 0.009 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.015 ± 0.000 | 0.011 ± 0.000 | - | 0.058 ± 0.006 | 0.087 ± 0.008 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.051 ± 0.000 | 0.038 ± 0.000 | - | 0.199 ± 0.004 | 0.265 ± 0.015 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.036 ± 0.000 | 0.028 ± 0.000 | - | 0.098 ± 0.003 | 0.129 ± 0.007 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.128 ± 0.014 | 0.104 ± 0.000 | - | 0.372 ± 0.026 | 0.446 ± 0.010 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.032 ± 0.000 | 0.005 ± 0.000 | - | 0.050 ± 0.005 | 0.476 ± 0.074 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.034 ± 0.000 | 0.010 ± 0.000 | - | 0.090 ± 0.003 | 0.533 ± 0.058 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.032 ± 0.000 | 0.006 ± 0.002 | - | 0.091 ± 0.007 | 0.508 ± 0.052 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.037 ± 0.002 | 0.014 ± 0.000 | - | 0.253 ± 0.003 | 0.763 ± 0.195 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.063 ± 0.001 | 0.044 ± 0.000 | - | 0.103 ± 0.003 | 0.472 ± 0.018 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.115 ± 0.003 | 0.134 ± 0.001 | - | 0.308 ± 0.009 | 0.723 ± 0.135 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.064 ± 0.001 | 0.046 ± 0.000 | - | 0.132 ± 0.002 | 0.494 ± 0.025 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.123 ± 0.001 | 0.145 ± 0.000 | - | 0.442 ± 0.020 | 0.803 ± 0.012 |
| large | `eigh` | f64 | 1 | `64x64` | 0.439 ± 0.015 | 0.276 ± 0.004 | - | 0.656 ± 0.035 | 0.717 ± 0.038 |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | 0.015 ± 0.000 | 0.007 ± 0.001 | - | 0.863 ± 0.038 | 0.975 ± 0.074 |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | 0.052 ± 0.008 | 0.018 ± 0.000 | - | 0.872 ± 0.069 | 1.260 ± 0.087 |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | 0.126 ± 0.004 | 0.038 ± 0.001 | - | 0.505 ± 0.026 | 0.865 ± 0.026 |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | 0.719 ± 0.054 | 0.574 ± 0.019 | - | 0.738 ± 0.035 | 0.950 ± 0.133 |
| large | `matmul` | f64 | 1 | `128x128` | 0.035 ± 0.001 | 0.017 ± 0.001 | - | 3.352 ± 0.065 | 3.795 ± 0.083 |
| large | `matmul` | f64 | 1 | `256x256` | 0.200 ± 0.005 | 0.129 ± 0.028 | - | 13.278 ± 0.186 | 14.314 ± 0.294 |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | 0.648 ± 0.003 | 0.518 ± 0.024 | - | 54.597 ± 1.223 | 56.727 ± 1.619 |
| large | `qr` | f64 | 1 | `64x64` | 0.103 ± 0.001 | 0.069 ± 0.001 | - | 0.519 ± 0.035 | 0.566 ± 0.014 |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | 0.088 ± 0.002 | 0.018 ± 0.001 | - | 0.463 ± 0.012 | 0.549 ± 0.015 |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | 0.090 ± 0.001 | 0.020 ± 0.002 | - | 0.579 ± 0.022 | 0.638 ± 0.008 |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | 0.104 ± 0.004 | 0.030 ± 0.003 | - | 0.885 ± 0.046 | 0.978 ± 0.041 |
| large | `svd` | f64 | 1 | `64x64` | 0.785 ± 0.015 | 0.565 ± 0.012 | - | 0.722 ± 0.027 | 0.784 ± 0.018 |
| small | `eigh` | f64 | 1 | `2x2` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.010 ± 0.003 | 0.027 ± 0.007 |
| small | `eigh` | f64 | 1 | `4x4` | 0.005 ± 0.001 | 0.002 ± 0.000 | - | 0.011 ± 0.000 | 0.033 ± 0.007 |
| small | `eigh` | f64 | 1 | `8x8` | 0.009 ± 0.000 | 0.003 ± 0.000 | - | 0.018 ± 0.000 | 0.040 ± 0.009 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | 0.006 ± 0.004 | 0.001 ± 0.000 | - | 0.012 ± 0.001 | 0.091 ± 0.004 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | 0.005 ± 0.000 | 0.001 ± 0.000 | - | 0.014 ± 0.001 | 0.089 ± 0.008 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | 0.005 ± 0.000 | 0.001 ± 0.000 | - | 0.024 ± 0.000 | 0.132 ± 0.041 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | 0.051 ± 0.011 | 0.004 ± 0.000 | - | 0.022 ± 0.003 | 0.349 ± 0.020 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | 0.048 ± 0.000 | 0.004 ± 0.000 | - | 0.023 ± 0.001 | 0.371 ± 0.029 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | 0.053 ± 0.000 | 0.006 ± 0.000 | - | 0.039 ± 0.001 | 0.400 ± 0.050 |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | 0.070 ± 0.002 | 0.011 ± 0.000 | - | 0.032 ± 0.001 | 0.390 ± 0.034 |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | 0.068 ± 0.003 | 0.012 ± 0.002 | - | 0.033 ± 0.001 | 0.422 ± 0.076 |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | 0.058 ± 0.002 | 0.012 ± 0.000 | - | 0.038 ± 0.000 | 0.404 ± 0.034 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | 0.080 ± 0.004 | 0.008 ± 0.000 | - | 0.021 ± 0.001 | 0.395 ± 0.128 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | 0.079 ± 0.003 | 0.009 ± 0.000 | - | 0.023 ± 0.000 | 0.419 ± 0.118 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | 0.092 ± 0.012 | 0.017 ± 0.002 | - | 0.033 ± 0.001 | 0.441 ± 0.250 |
| small | `matmul` | f64 | 1 | `2x2` | 0.006 ± 0.002 | 0.001 ± 0.001 | - | 0.008 ± 0.002 | 0.036 ± 0.005 |
| small | `matmul` | f64 | 1 | `4x4` | 0.005 ± 0.001 | 0.001 ± 0.000 | - | 0.009 ± 0.001 | 0.042 ± 0.007 |
| small | `matmul` | f64 | 1 | `8x8` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.019 ± 0.000 | 0.077 ± 0.049 |
| small | `qr` | f64 | 1 | `2x2` | 0.005 ± 0.003 | 0.001 ± 0.000 | - | 0.008 ± 0.002 | 0.025 ± 0.004 |
| small | `qr` | f64 | 1 | `4x4` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.008 ± 0.000 | 0.027 ± 0.005 |
| small | `qr` | f64 | 1 | `8x8` | 0.005 ± 0.000 | 0.002 ± 0.000 | - | 0.014 ± 0.000 | 0.033 ± 0.003 |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.016 ± 0.001 | 0.043 ± 0.006 |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.016 ± 0.001 | 0.041 ± 0.003 |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.017 ± 0.000 | 0.043 ± 0.005 |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.018 ± 0.000 | 0.047 ± 0.003 |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | 0.008 ± 0.000 | 0.002 ± 0.000 | - | 0.022 ± 0.001 | 0.052 ± 0.004 |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | 0.008 ± 0.002 | 0.002 ± 0.000 | - | 0.024 ± 0.000 | 0.054 ± 0.011 |
| small | `svd` | f64 | 1 | `2x2` | 0.007 ± 0.002 | 0.001 ± 0.000 | - | 0.009 ± 0.001 | 0.026 ± 0.004 |
| small | `svd` | f64 | 1 | `4x4` | 0.007 ± 0.000 | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.031 ± 0.006 |
| small | `svd` | f64 | 1 | `8x8` | 0.014 ± 0.000 | 0.006 ± 0.000 | - | 0.019 ± 0.000 | 0.040 ± 0.004 |

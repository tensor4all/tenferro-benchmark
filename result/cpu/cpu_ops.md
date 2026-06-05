# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Timestamp: `20260605_195601`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/cpu/einsum/20260605_195601`.

- tenferro-rs commit: `f28016c6dca2b1c69f95330e17025d59de714a2f`

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

- CSV: `data/results/cpu/einsum/20260605_195601/cpu_ops_t1_20260605_195601.csv`
- Source table: `data/results/cpu/einsum/20260605_195601/cpu_ops_t1_20260605_195601.md`

## PR884 CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.009 ± 0.000 | 0.006 ± 0.000 | - | 0.075 ± 0.001 | 0.098 ± 0.009 |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.029 ± 0.000 | 0.020 ± 0.000 | - | 0.272 ± 0.005 | 0.330 ± 0.026 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.022 ± 0.000 | 0.015 ± 0.000 | - | 0.109 ± 0.001 | 0.133 ± 0.010 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.077 ± 0.004 | 0.055 ± 0.007 | - | 0.407 ± 0.001 | 0.473 ± 0.070 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.025 ± 0.001 | 0.102 ± 0.004 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.004 ± 0.000 | 0.002 ± 0.000 | - | 0.064 ± 0.001 | 0.156 ± 0.005 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.063 ± 0.004 | 0.155 ± 0.003 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.005 ± 0.000 | 0.003 ± 0.000 | - | 0.238 ± 0.009 | 0.328 ± 0.012 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.010 ± 0.000 | 0.007 ± 0.000 | - | 0.051 ± 0.001 | 0.069 ± 0.006 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.034 ± 0.000 | 0.026 ± 0.000 | - | 0.190 ± 0.010 | 0.204 ± 0.009 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.015 ± 0.000 | 0.010 ± 0.000 | - | 0.080 ± 0.003 | 0.103 ± 0.006 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.048 ± 0.000 | 0.036 ± 0.001 | - | 0.291 ± 0.009 | 0.327 ± 0.013 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.016 ± 0.000 | 0.011 ± 0.000 | - | 0.083 ± 0.003 | 0.116 ± 0.007 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.048 ± 0.001 | 0.037 ± 0.000 | - | 0.287 ± 0.001 | 0.341 ± 0.008 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.018 ± 0.000 | 0.011 ± 0.000 | - | 0.112 ± 0.001 | 0.144 ± 0.005 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.055 ± 0.000 | 0.040 ± 0.002 | - | 0.407 ± 0.030 | 0.474 ± 0.004 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.015 ± 0.000 | 0.011 ± 0.000 | - | 0.057 ± 0.004 | 0.082 ± 0.008 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.048 ± 0.000 | 0.038 ± 0.004 | - | 0.207 ± 0.009 | 0.272 ± 0.010 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.036 ± 0.001 | 0.027 ± 0.000 | - | 0.097 ± 0.001 | 0.133 ± 0.004 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.127 ± 0.001 | 0.103 ± 0.002 | - | 0.362 ± 0.003 | 0.446 ± 0.032 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.032 ± 0.001 | 0.005 ± 0.000 | - | 0.051 ± 0.004 | 0.460 ± 0.023 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.033 ± 0.001 | 0.010 ± 0.000 | - | 0.093 ± 0.005 | 0.508 ± 0.025 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.032 ± 0.002 | 0.006 ± 0.000 | - | 0.090 ± 0.002 | 0.514 ± 0.061 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.036 ± 0.000 | 0.014 ± 0.001 | - | 0.249 ± 0.011 | 0.684 ± 0.038 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.062 ± 0.002 | 0.042 ± 0.002 | - | 0.106 ± 0.001 | 0.487 ± 0.042 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.112 ± 0.000 | 0.142 ± 0.008 | - | 0.310 ± 0.004 | 0.689 ± 0.034 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.060 ± 0.000 | 0.044 ± 0.000 | - | 0.131 ± 0.004 | 0.504 ± 0.043 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.116 ± 0.001 | 0.153 ± 0.010 | - | 0.445 ± 0.060 | 0.816 ± 0.065 |
| large | `eigh` | f64 | 1 | `64x64` | 0.422 ± 0.003 | 0.276 ± 0.003 | - | 0.629 ± 0.008 | 0.715 ± 0.014 |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | 0.015 ± 0.000 | 0.006 ± 0.000 | - | 0.840 ± 0.012 | 0.948 ± 0.028 |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | 0.046 ± 0.004 | 0.018 ± 0.000 | - | 0.858 ± 0.040 | 1.233 ± 0.093 |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | 0.125 ± 0.004 | 0.037 ± 0.001 | - | 0.485 ± 0.012 | 0.889 ± 0.202 |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | 0.716 ± 0.039 | 0.558 ± 0.002 | - | 0.729 ± 0.007 | 0.955 ± 0.136 |
| large | `matmul` | f64 | 1 | `128x128` | 0.035 ± 0.000 | 0.024 ± 0.007 | - | 3.293 ± 0.039 | 3.604 ± 0.136 |
| large | `matmul` | f64 | 1 | `256x256` | 0.194 ± 0.005 | 0.130 ± 0.006 | - | 13.139 ± 0.088 | 13.995 ± 0.235 |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | 0.647 ± 0.003 | 0.520 ± 0.007 | - | 53.797 ± 0.937 | 57.147 ± 1.704 |
| large | `qr` | f64 | 1 | `64x64` | 0.102 ± 0.002 | 0.069 ± 0.000 | - | 0.491 ± 0.004 | 0.558 ± 0.009 |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | 0.088 ± 0.000 | 0.020 ± 0.000 | - | 0.464 ± 0.012 | 0.560 ± 0.039 |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | 0.090 ± 0.001 | 0.021 ± 0.004 | - | 0.559 ± 0.005 | 0.658 ± 0.019 |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | 0.102 ± 0.003 | 0.032 ± 0.003 | - | 0.892 ± 0.033 | 0.985 ± 0.042 |
| large | `svd` | f64 | 1 | `64x64` | 0.779 ± 0.013 | 0.556 ± 0.029 | - | 0.709 ± 0.018 | 0.797 ± 0.055 |
| small | `eigh` | f64 | 1 | `2x2` | 0.004 ± 0.001 | 0.001 ± 0.000 | - | 0.008 ± 0.001 | 0.028 ± 0.007 |
| small | `eigh` | f64 | 1 | `4x4` | 0.005 ± 0.000 | 0.002 ± 0.000 | - | 0.010 ± 0.000 | 0.029 ± 0.005 |
| small | `eigh` | f64 | 1 | `8x8` | 0.008 ± 0.000 | 0.003 ± 0.000 | - | 0.018 ± 0.000 | 0.038 ± 0.006 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | 0.006 ± 0.002 | 0.001 ± 0.000 | - | 0.012 ± 0.001 | 0.086 ± 0.002 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | 0.005 ± 0.002 | 0.001 ± 0.000 | - | 0.014 ± 0.000 | 0.091 ± 0.006 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.025 ± 0.000 | 0.122 ± 0.016 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | 0.057 ± 0.007 | 0.004 ± 0.000 | - | 0.020 ± 0.006 | 0.350 ± 0.023 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | 0.047 ± 0.000 | 0.004 ± 0.000 | - | 0.022 ± 0.002 | 0.349 ± 0.013 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | 0.048 ± 0.001 | 0.006 ± 0.000 | - | 0.038 ± 0.002 | 0.432 ± 0.236 |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | 0.069 ± 0.005 | 0.011 ± 0.000 | - | 0.033 ± 0.002 | 0.389 ± 0.024 |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | 0.059 ± 0.001 | 0.012 ± 0.000 | - | 0.032 ± 0.001 | 0.397 ± 0.021 |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | 0.060 ± 0.009 | 0.012 ± 0.000 | - | 0.039 ± 0.002 | 0.403 ± 0.026 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | 0.083 ± 0.010 | 0.008 ± 0.000 | - | 0.021 ± 0.004 | 0.372 ± 0.005 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | 0.070 ± 0.012 | 0.009 ± 0.000 | - | 0.023 ± 0.002 | 0.386 ± 0.023 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | 0.083 ± 0.001 | 0.019 ± 0.011 | - | 0.032 ± 0.001 | 0.399 ± 0.021 |
| small | `matmul` | f64 | 1 | `2x2` | 0.007 ± 0.001 | 0.001 ± 0.000 | - | 0.008 ± 0.002 | 0.038 ± 0.003 |
| small | `matmul` | f64 | 1 | `4x4` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.010 ± 0.000 | 0.041 ± 0.010 |
| small | `matmul` | f64 | 1 | `8x8` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.019 ± 0.000 | 0.064 ± 0.010 |
| small | `qr` | f64 | 1 | `2x2` | 0.004 ± 0.001 | 0.001 ± 0.000 | - | 0.006 ± 0.001 | 0.026 ± 0.006 |
| small | `qr` | f64 | 1 | `4x4` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.008 ± 0.000 | 0.026 ± 0.005 |
| small | `qr` | f64 | 1 | `8x8` | 0.004 ± 0.000 | 0.002 ± 0.000 | - | 0.014 ± 0.000 | 0.036 ± 0.005 |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | 0.003 ± 0.001 | 0.002 ± 0.000 | - | 0.017 ± 0.004 | 0.040 ± 0.006 |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | 0.003 ± 0.000 | 0.002 ± 0.000 | - | 0.015 ± 0.001 | 0.043 ± 0.004 |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.016 ± 0.001 | 0.044 ± 0.002 |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.017 ± 0.000 | 0.046 ± 0.005 |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.022 ± 0.000 | 0.048 ± 0.005 |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.025 ± 0.000 | 0.051 ± 0.003 |
| small | `svd` | f64 | 1 | `2x2` | 0.007 ± 0.003 | 0.001 ± 0.000 | - | 0.008 ± 0.001 | 0.027 ± 0.005 |
| small | `svd` | f64 | 1 | `4x4` | 0.007 ± 0.000 | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.030 ± 0.007 |
| small | `svd` | f64 | 1 | `8x8` | 0.012 ± 0.001 | 0.006 ± 0.000 | - | 0.019 ± 0.000 | 0.040 ± 0.004 |

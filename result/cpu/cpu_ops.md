# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Timestamp: `20260605_225341`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/cpu/einsum/20260605_225341`.

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

- CSV: `data/results/cpu/einsum/20260605_225341/cpu_ops_t4_20260605_225341.csv`
- Source table: `data/results/cpu/einsum/20260605_225341/cpu_ops_t4_20260605_225341.md`

## PR884 CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.009 ± 0.001 | 0.006 ± 0.000 | - | 0.073 ± 0.001 | 0.096 ± 0.011 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.029 ± 0.000 | 0.020 ± 0.000 | - | 0.270 ± 0.006 | 0.341 ± 0.024 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.023 ± 0.000 | 0.016 ± 0.000 | - | 0.105 ± 0.004 | 0.126 ± 0.005 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.077 ± 0.000 | 0.056 ± 0.002 | - | 0.406 ± 0.004 | 0.464 ± 0.008 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.024 ± 0.001 | 0.105 ± 0.008 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.004 ± 0.000 | 0.002 ± 0.000 | - | 0.062 ± 0.003 | 0.148 ± 0.016 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.061 ± 0.003 | 0.158 ± 0.005 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.005 ± 0.000 | 0.003 ± 0.000 | - | 0.219 ± 0.002 | 0.336 ± 0.017 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.011 ± 0.000 | 0.007 ± 0.000 | - | 0.062 ± 0.001 | 0.072 ± 0.004 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.034 ± 0.000 | 0.025 ± 0.000 | - | 0.196 ± 0.002 | 0.201 ± 0.002 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.014 ± 0.000 | 0.010 ± 0.000 | - | 0.094 ± 0.008 | 0.094 ± 0.007 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.048 ± 0.000 | 0.037 ± 0.000 | - | 0.304 ± 0.010 | 0.333 ± 0.010 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.016 ± 0.000 | 0.011 ± 0.000 | - | 0.100 ± 0.006 | 0.113 ± 0.009 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.049 ± 0.000 | 0.036 ± 0.000 | - | 0.305 ± 0.010 | 0.340 ± 0.007 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.018 ± 0.000 | 0.011 ± 0.000 | - | 0.128 ± 0.013 | 0.139 ± 0.005 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.055 ± 0.000 | 0.040 ± 0.001 | - | 0.403 ± 0.018 | 0.464 ± 0.007 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.015 ± 0.000 | 0.011 ± 0.000 | - | 0.056 ± 0.002 | 0.086 ± 0.008 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.048 ± 0.000 | 0.038 ± 0.000 | - | 0.191 ± 0.003 | 0.264 ± 0.002 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.036 ± 0.000 | 0.029 ± 0.000 | - | 0.095 ± 0.002 | 0.126 ± 0.008 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.130 ± 0.000 | 0.106 ± 0.000 | - | 0.353 ± 0.004 | 0.442 ± 0.013 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.032 ± 0.000 | 0.005 ± 0.000 | - | 0.047 ± 0.002 | 0.463 ± 0.043 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.034 ± 0.000 | 0.010 ± 0.000 | - | 0.086 ± 0.005 | 0.501 ± 0.019 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.032 ± 0.000 | 0.006 ± 0.000 | - | 0.086 ± 0.004 | 0.519 ± 0.024 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.037 ± 0.001 | 0.014 ± 0.001 | - | 0.245 ± 0.003 | 0.697 ± 0.053 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.061 ± 0.001 | 0.044 ± 0.001 | - | 0.116 ± 0.014 | 0.461 ± 0.016 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.120 ± 0.028 | 0.134 ± 0.001 | - | 0.318 ± 0.012 | 0.676 ± 0.041 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.060 ± 0.004 | 0.044 ± 0.001 | - | 0.147 ± 0.017 | 0.493 ± 0.019 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.119 ± 0.006 | 0.141 ± 0.001 | - | 0.425 ± 0.009 | 0.804 ± 0.019 |
| large | `eigh` | f64 | 4 | `64x64` | 0.453 ± 0.027 | 0.278 ± 0.003 | - | 0.629 ± 0.006 | 0.704 ± 0.032 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.015 ± 0.000 | 0.006 ± 0.001 | - | 0.833 ± 0.020 | 0.954 ± 0.038 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.046 ± 0.006 | 0.018 ± 0.000 | - | 0.844 ± 0.008 | 1.298 ± 0.059 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.124 ± 0.002 | 0.039 ± 0.001 | - | 0.482 ± 0.009 | 0.872 ± 0.008 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 0.713 ± 0.022 | 0.557 ± 0.004 | - | 0.725 ± 0.003 | 0.953 ± 0.135 |
| large | `matmul` | f64 | 4 | `128x128` | 0.035 ± 0.000 | 0.017 ± 0.001 | - | 3.302 ± 0.036 | 3.426 ± 0.042 |
| large | `matmul` | f64 | 4 | `256x256` | 0.193 ± 0.001 | 0.130 ± 0.007 | - | 13.056 ± 0.048 | 13.926 ± 0.116 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 0.649 ± 0.003 | 0.504 ± 0.024 | - | 53.677 ± 0.839 | 56.454 ± 2.502 |
| large | `qr` | f64 | 4 | `64x64` | 0.104 ± 0.004 | 0.069 ± 0.000 | - | 0.507 ± 0.018 | 0.567 ± 0.032 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.087 ± 0.002 | 0.018 ± 0.002 | - | 0.460 ± 0.010 | 0.545 ± 0.026 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.089 ± 0.001 | 0.020 ± 0.001 | - | 0.556 ± 0.005 | 0.629 ± 0.013 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.102 ± 0.003 | 0.030 ± 0.006 | - | 0.866 ± 0.032 | 0.988 ± 0.042 |
| large | `svd` | f64 | 4 | `64x64` | 0.770 ± 0.013 | 0.558 ± 0.002 | - | 0.701 ± 0.006 | 0.756 ± 0.024 |
| small | `eigh` | f64 | 4 | `2x2` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.010 ± 0.001 | 0.027 ± 0.004 |
| small | `eigh` | f64 | 4 | `4x4` | 0.005 ± 0.000 | 0.002 ± 0.000 | - | 0.012 ± 0.001 | 0.028 ± 0.005 |
| small | `eigh` | f64 | 4 | `8x8` | 0.008 ± 0.000 | 0.003 ± 0.000 | - | 0.019 ± 0.001 | 0.037 ± 0.006 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.006 ± 0.002 | 0.001 ± 0.000 | - | 0.013 ± 0.001 | 0.089 ± 0.005 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.005 ± 0.000 | 0.001 ± 0.000 | - | 0.015 ± 0.001 | 0.095 ± 0.005 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.025 ± 0.001 | 0.112 ± 0.005 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.051 ± 0.005 | 0.004 ± 0.000 | - | 0.022 ± 0.003 | 0.349 ± 0.010 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.048 ± 0.000 | 0.004 ± 0.000 | - | 0.024 ± 0.003 | 0.352 ± 0.023 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.046 ± 0.001 | 0.006 ± 0.000 | - | 0.035 ± 0.002 | 0.377 ± 0.011 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.032 ± 0.001 | 0.011 ± 0.000 | - | 0.032 ± 0.001 | 0.388 ± 0.017 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.057 ± 0.003 | 0.012 ± 0.000 | - | 0.033 ± 0.001 | 0.391 ± 0.026 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.058 ± 0.000 | 0.012 ± 0.000 | - | 0.039 ± 0.001 | 0.393 ± 0.017 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.079 ± 0.001 | 0.008 ± 0.001 | - | 0.021 ± 0.001 | 0.377 ± 0.008 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.081 ± 0.002 | 0.009 ± 0.000 | - | 0.024 ± 0.003 | 0.381 ± 0.006 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.081 ± 0.001 | 0.017 ± 0.002 | - | 0.033 ± 0.001 | 0.402 ± 0.012 |
| small | `matmul` | f64 | 4 | `2x2` | 0.005 ± 0.002 | 0.001 ± 0.000 | - | 0.008 ± 0.001 | 0.036 ± 0.005 |
| small | `matmul` | f64 | 4 | `4x4` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.010 ± 0.000 | 0.039 ± 0.002 |
| small | `matmul` | f64 | 4 | `8x8` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.020 ± 0.000 | 0.062 ± 0.005 |
| small | `qr` | f64 | 4 | `2x2` | 0.006 ± 0.002 | 0.001 ± 0.000 | - | 0.032 ± 0.018 | 0.025 ± 0.004 |
| small | `qr` | f64 | 4 | `4x4` | 0.004 ± 0.001 | 0.001 ± 0.000 | - | 0.021 ± 0.006 | 0.027 ± 0.005 |
| small | `qr` | f64 | 4 | `8x8` | 0.004 ± 0.000 | 0.002 ± 0.000 | - | 0.034 ± 0.012 | 0.033 ± 0.005 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.015 ± 0.001 | 0.040 ± 0.003 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.016 ± 0.000 | 0.042 ± 0.005 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.017 ± 0.001 | 0.043 ± 0.004 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.018 ± 0.001 | 0.043 ± 0.004 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.023 ± 0.000 | 0.051 ± 0.003 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.025 ± 0.001 | 0.054 ± 0.004 |
| small | `svd` | f64 | 4 | `2x2` | 0.005 ± 0.002 | 0.001 ± 0.000 | - | 0.009 ± 0.001 | 0.028 ± 0.007 |
| small | `svd` | f64 | 4 | `4x4` | 0.007 ± 0.000 | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.032 ± 0.006 |
| small | `svd` | f64 | 4 | `8x8` | 0.012 ± 0.000 | 0.006 ± 0.000 | - | 0.020 ± 0.001 | 0.040 ± 0.005 |

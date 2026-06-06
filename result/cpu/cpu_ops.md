# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Timestamp: `20260606_234410`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/cpu/einsum/20260606_234410`.

- tenferro-rs commit: `3bd07ac07e5520e33fa6babd1ad537341fd71d50`

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

- CSV: `data/results/cpu/einsum/20260606_234410/cpu_ops_t1_20260606_234410.csv`
- Source table: `data/results/cpu/einsum/20260606_234410/cpu_ops_t1_20260606_234410.md`

## PR884 CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.009 ± 0.000 | 0.006 ± 0.000 | - | 0.073 ± 0.001 | 0.095 ± 0.009 |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.028 ± 0.000 | 0.020 ± 0.000 | - | 0.267 ± 0.002 | 0.337 ± 0.029 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.022 ± 0.000 | 0.015 ± 0.000 | - | 0.108 ± 0.005 | 0.127 ± 0.006 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.076 ± 0.001 | 0.056 ± 0.000 | - | 0.401 ± 0.002 | 0.461 ± 0.019 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.025 ± 0.001 | 0.100 ± 0.005 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.004 ± 0.000 | 0.002 ± 0.000 | - | 0.062 ± 0.001 | 0.160 ± 0.020 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.061 ± 0.002 | 0.150 ± 0.008 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.005 ± 0.000 | 0.003 ± 0.000 | - | 0.219 ± 0.006 | 0.324 ± 0.011 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.010 ± 0.000 | 0.007 ± 0.000 | - | 0.050 ± 0.001 | 0.069 ± 0.009 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.033 ± 0.000 | 0.025 ± 0.000 | - | 0.181 ± 0.005 | 0.203 ± 0.007 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.014 ± 0.000 | 0.010 ± 0.000 | - | 0.078 ± 0.001 | 0.099 ± 0.006 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.046 ± 0.000 | 0.036 ± 0.000 | - | 0.284 ± 0.016 | 0.333 ± 0.011 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.016 ± 0.000 | 0.011 ± 0.000 | - | 0.082 ± 0.023 | 0.108 ± 0.002 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.048 ± 0.000 | 0.036 ± 0.000 | - | 0.285 ± 0.003 | 0.345 ± 0.004 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.018 ± 0.002 | 0.012 ± 0.000 | - | 0.111 ± 0.001 | 0.140 ± 0.004 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.054 ± 0.000 | 0.040 ± 0.002 | - | 0.395 ± 0.005 | 0.468 ± 0.012 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.014 ± 0.000 | 0.011 ± 0.000 | - | 0.056 ± 0.001 | 0.085 ± 0.004 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.047 ± 0.001 | 0.036 ± 0.000 | - | 0.191 ± 0.014 | 0.262 ± 0.003 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.035 ± 0.001 | 0.028 ± 0.000 | - | 0.097 ± 0.001 | 0.131 ± 0.005 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.126 ± 0.001 | 0.100 ± 0.000 | - | 0.358 ± 0.002 | 0.445 ± 0.017 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.031 ± 0.001 | 0.005 ± 0.000 | - | 0.048 ± 0.003 | 0.448 ± 0.026 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.033 ± 0.000 | 0.010 ± 0.000 | - | 0.091 ± 0.006 | 0.514 ± 0.033 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.032 ± 0.001 | 0.006 ± 0.000 | - | 0.088 ± 0.002 | 0.521 ± 0.044 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.036 ± 0.000 | 0.013 ± 0.001 | - | 0.244 ± 0.003 | 0.688 ± 0.046 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.060 ± 0.001 | 0.043 ± 0.000 | - | 0.098 ± 0.002 | 0.472 ± 0.009 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.110 ± 0.001 | 0.130 ± 0.008 | - | 0.307 ± 0.005 | 0.682 ± 0.024 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.062 ± 0.000 | 0.044 ± 0.001 | - | 0.129 ± 0.001 | 0.493 ± 0.020 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.120 ± 0.001 | 0.144 ± 0.002 | - | 0.426 ± 0.004 | 0.800 ± 0.031 |
| large | `eigh` | f64 | 1 | `64x64` | 0.421 ± 0.002 | 0.279 ± 0.004 | - | 0.626 ± 0.004 | 0.697 ± 0.025 |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | 0.014 ± 0.001 | 0.006 ± 0.001 | - | 0.826 ± 0.009 | 0.933 ± 0.015 |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | 0.045 ± 0.001 | 0.018 ± 0.001 | - | 0.846 ± 0.019 | 1.281 ± 0.185 |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | 0.123 ± 0.001 | 0.038 ± 0.001 | - | 0.480 ± 0.006 | 0.870 ± 0.023 |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | 0.708 ± 0.027 | 0.556 ± 0.004 | - | 0.727 ± 0.010 | 0.924 ± 0.040 |
| large | `matmul` | f64 | 1 | `128x128` | 0.035 ± 0.000 | 0.017 ± 0.001 | - | 3.239 ± 0.051 | 3.588 ± 0.042 |
| large | `matmul` | f64 | 1 | `256x256` | 0.193 ± 0.001 | 0.126 ± 0.008 | - | 12.960 ± 0.032 | 13.857 ± 0.328 |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | 0.647 ± 0.008 | 0.502 ± 0.026 | - | 52.373 ± 0.546 | 55.825 ± 1.089 |
| large | `qr` | f64 | 1 | `64x64` | 0.105 ± 0.001 | 0.073 ± 0.002 | - | 0.489 ± 0.016 | 0.565 ± 0.016 |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | 0.087 ± 0.001 | 0.018 ± 0.003 | - | 0.455 ± 0.012 | 0.537 ± 0.023 |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | 0.089 ± 0.001 | 0.021 ± 0.001 | - | 0.554 ± 0.010 | 0.650 ± 0.016 |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | 0.102 ± 0.001 | 0.030 ± 0.001 | - | 0.859 ± 0.008 | 0.973 ± 0.034 |
| large | `svd` | f64 | 1 | `64x64` | 0.783 ± 0.007 | 0.561 ± 0.002 | - | 0.699 ± 0.004 | 0.741 ± 0.019 |
| small | `eigh` | f64 | 1 | `2x2` | 0.004 ± 0.002 | 0.001 ± 0.000 | - | 0.008 ± 0.001 | 0.026 ± 0.006 |
| small | `eigh` | f64 | 1 | `4x4` | 0.005 ± 0.002 | 0.002 ± 0.000 | - | 0.010 ± 0.000 | 0.030 ± 0.006 |
| small | `eigh` | f64 | 1 | `8x8` | 0.008 ± 0.000 | 0.003 ± 0.000 | - | 0.018 ± 0.000 | 0.038 ± 0.007 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | 0.006 ± 0.002 | 0.001 ± 0.000 | - | 0.012 ± 0.001 | 0.084 ± 0.007 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | 0.005 ± 0.000 | 0.001 ± 0.000 | - | 0.014 ± 0.001 | 0.090 ± 0.008 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.025 ± 0.001 | 0.116 ± 0.006 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | 0.052 ± 0.006 | 0.004 ± 0.001 | - | 0.021 ± 0.004 | 0.336 ± 0.011 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | 0.048 ± 0.001 | 0.004 ± 0.000 | - | 0.022 ± 0.001 | 0.363 ± 0.033 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | 0.047 ± 0.000 | 0.006 ± 0.000 | - | 0.039 ± 0.001 | 0.373 ± 0.016 |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | 0.067 ± 0.002 | 0.011 ± 0.000 | - | 0.034 ± 0.003 | 0.380 ± 0.015 |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | 0.066 ± 0.001 | 0.012 ± 0.000 | - | 0.032 ± 0.004 | 0.385 ± 0.010 |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | 0.058 ± 0.001 | 0.012 ± 0.000 | - | 0.038 ± 0.000 | 0.386 ± 0.014 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | 0.079 ± 0.003 | 0.008 ± 0.001 | - | 0.020 ± 0.001 | 0.365 ± 0.014 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | 0.079 ± 0.001 | 0.009 ± 0.000 | - | 0.022 ± 0.000 | 0.385 ± 0.022 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | 0.082 ± 0.001 | 0.017 ± 0.002 | - | 0.032 ± 0.001 | 0.399 ± 0.013 |
| small | `matmul` | f64 | 1 | `2x2` | 0.007 ± 0.004 | 0.001 ± 0.000 | - | 0.008 ± 0.001 | 0.036 ± 0.004 |
| small | `matmul` | f64 | 1 | `4x4` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.010 ± 0.001 | 0.038 ± 0.004 |
| small | `matmul` | f64 | 1 | `8x8` | 0.004 ± 0.002 | 0.001 ± 0.000 | - | 0.019 ± 0.001 | 0.065 ± 0.003 |
| small | `qr` | f64 | 1 | `2x2` | 0.004 ± 0.002 | 0.001 ± 0.000 | - | 0.006 ± 0.001 | 0.026 ± 0.005 |
| small | `qr` | f64 | 1 | `4x4` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.008 ± 0.000 | 0.026 ± 0.004 |
| small | `qr` | f64 | 1 | `8x8` | 0.005 ± 0.000 | 0.002 ± 0.000 | - | 0.014 ± 0.000 | 0.036 ± 0.004 |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.014 ± 0.001 | 0.040 ± 0.003 |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.015 ± 0.000 | 0.042 ± 0.005 |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.016 ± 0.000 | 0.045 ± 0.004 |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.017 ± 0.000 | 0.045 ± 0.008 |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.022 ± 0.000 | 0.047 ± 0.006 |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | 0.007 ± 0.000 | 0.002 ± 0.000 | - | 0.025 ± 0.001 | 0.057 ± 0.005 |
| small | `svd` | f64 | 1 | `2x2` | 0.007 ± 0.002 | 0.001 ± 0.000 | - | 0.008 ± 0.001 | 0.028 ± 0.003 |
| small | `svd` | f64 | 1 | `4x4` | 0.007 ± 0.000 | 0.003 ± 0.000 | - | 0.010 ± 0.001 | 0.029 ± 0.004 |
| small | `svd` | f64 | 1 | `8x8` | 0.013 ± 0.001 | 0.006 ± 0.000 | - | 0.019 ± 0.001 | 0.039 ± 0.005 |

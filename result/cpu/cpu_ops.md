# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Timestamp: `20260605_142400`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/cpu/einsum/20260605_142400`.

- tenferro-rs commit: `3c3cc0b1079ca84e6b700ce93c77606e2110f946`

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

- CSV: `data/results/cpu/einsum/20260605_142400/cpu_ops_t4_20260605_142400.csv`
- Source table: `data/results/cpu/einsum/20260605_142400/cpu_ops_t4_20260605_142400.md`

## PR884 CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.012 ± 0.001 | 0.006 ± 0.002 | - | 0.072 ± 0.018 | 0.099 ± 0.006 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.026 ± 0.002 | 0.020 ± 0.000 | - | 0.270 ± 0.007 | 0.326 ± 0.023 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.019 ± 0.001 | 0.017 ± 0.001 | - | 0.110 ± 0.008 | 0.135 ± 0.010 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.063 ± 0.001 | 0.060 ± 0.008 | - | 0.408 ± 0.002 | 0.487 ± 0.165 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.024 ± 0.001 | 0.110 ± 0.008 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.004 ± 0.000 | 0.003 ± 0.000 | - | 0.065 ± 0.002 | 0.172 ± 0.031 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.063 ± 0.002 | 0.168 ± 0.011 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.004 ± 0.000 | 0.003 ± 0.000 | - | 0.222 ± 0.012 | 0.356 ± 0.042 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.011 ± 0.000 | 0.007 ± 0.001 | - | 0.069 ± 0.004 | 0.076 ± 0.004 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.031 ± 0.000 | 0.025 ± 0.003 | - | 0.199 ± 0.006 | 0.204 ± 0.012 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.014 ± 0.000 | 0.011 ± 0.002 | - | 0.098 ± 0.010 | 0.104 ± 0.019 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.056 ± 0.008 | 0.040 ± 0.004 | - | 0.315 ± 0.015 | 0.334 ± 0.007 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.008 ± 0.000 | 0.005 ± 0.001 | - | 0.097 ± 0.005 | 0.117 ± 0.007 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.019 ± 0.000 | 0.014 ± 0.001 | - | 0.317 ± 0.008 | 0.369 ± 0.120 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.009 ± 0.000 | 0.006 ± 0.001 | - | 0.133 ± 0.005 | 0.145 ± 0.005 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.024 ± 0.000 | 0.018 ± 0.000 | - | 0.424 ± 0.011 | 0.499 ± 0.058 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.014 ± 0.000 | 0.011 ± 0.001 | - | 0.056 ± 0.001 | 0.090 ± 0.012 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.043 ± 0.000 | 0.038 ± 0.001 | - | 0.198 ± 0.002 | 0.280 ± 0.082 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.031 ± 0.000 | 0.031 ± 0.005 | - | 0.096 ± 0.003 | 0.136 ± 0.018 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.110 ± 0.002 | 0.105 ± 0.002 | - | 0.356 ± 0.002 | 0.476 ± 0.058 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.032 ± 0.001 | 0.006 ± 0.001 | - | 0.049 ± 0.002 | 0.467 ± 0.037 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.030 ± 0.001 | 0.010 ± 0.002 | - | 0.086 ± 0.002 | 0.518 ± 0.087 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.030 ± 0.001 | 0.006 ± 0.003 | - | 0.088 ± 0.005 | 0.535 ± 0.048 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.032 ± 0.001 | 0.014 ± 0.002 | - | 0.248 ± 0.009 | 0.748 ± 0.248 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.043 ± 0.003 | 0.031 ± 0.002 | - | 0.121 ± 0.010 | 0.475 ± 0.013 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.061 ± 0.001 | 0.081 ± 0.002 | - | 0.332 ± 0.017 | 0.742 ± 0.136 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.044 ± 0.003 | 0.038 ± 0.001 | - | 0.159 ± 0.021 | 0.512 ± 0.057 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.087 ± 0.001 | 0.101 ± 0.002 | - | 0.456 ± 0.015 | 1.085 ± 0.309 |
| large | `eigh` | f64 | 4 | `64x64` | 0.367 ± 0.048 | 0.265 ± 0.007 | - | 0.635 ± 0.018 | 0.726 ± 0.023 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.017 ± 0.001 | 0.006 ± 0.001 | - | 0.834 ± 0.023 | 0.979 ± 0.087 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.054 ± 0.002 | 0.018 ± 0.005 | - | 0.866 ± 0.028 | 1.323 ± 0.168 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.105 ± 0.002 | 0.068 ± 0.002 | - | 0.486 ± 0.006 | 0.915 ± 0.209 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 0.591 ± 0.116 | 0.579 ± 0.031 | - | 0.728 ± 0.058 | 1.012 ± 0.130 |
| large | `matmul` | f64 | 4 | `128x128` | 0.019 ± 0.000 | 0.015 ± 0.000 | - | 3.313 ± 0.064 | 3.579 ± 0.129 |
| large | `matmul` | f64 | 4 | `256x256` | 0.119 ± 0.020 | 0.143 ± 0.035 | - | 13.166 ± 0.148 | 13.934 ± 0.099 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 0.451 ± 0.197 | 0.482 ± 0.061 | - | 53.416 ± 0.592 | 56.864 ± 1.880 |
| large | `qr` | f64 | 4 | `64x64` | 0.070 ± 0.001 | 0.070 ± 0.009 | - | 0.512 ± 0.010 | 0.554 ± 0.027 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.071 ± 0.029 | 0.012 ± 0.000 | - | 0.464 ± 0.004 | 0.533 ± 0.018 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.090 ± 0.001 | 0.014 ± 0.001 | - | 0.566 ± 0.012 | 0.643 ± 0.037 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.096 ± 0.000 | 0.021 ± 0.011 | - | 0.872 ± 0.039 | 0.983 ± 0.083 |
| large | `svd` | f64 | 4 | `64x64` | 0.533 ± 0.042 | 0.536 ± 0.013 | - | 0.740 ± 0.057 | 0.830 ± 0.069 |
| small | `eigh` | f64 | 4 | `2x2` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.009 ± 0.001 | 0.029 ± 0.009 |
| small | `eigh` | f64 | 4 | `4x4` | 0.002 ± 0.000 | 0.002 ± 0.000 | - | 0.012 ± 0.000 | 0.034 ± 0.005 |
| small | `eigh` | f64 | 4 | `8x8` | 0.004 ± 0.000 | 0.003 ± 0.000 | - | 0.019 ± 0.003 | 0.040 ± 0.006 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.003 ± 0.001 | 0.001 ± 0.000 | - | 0.013 ± 0.003 | 0.090 ± 0.003 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.015 ± 0.000 | 0.097 ± 0.003 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.002 ± 0.001 | 0.001 ± 0.000 | - | 0.025 ± 0.001 | 0.129 ± 0.022 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.024 ± 0.004 | 0.004 ± 0.001 | - | 0.025 ± 0.003 | 0.367 ± 0.034 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.022 ± 0.001 | 0.005 ± 0.000 | - | 0.024 ± 0.001 | 0.359 ± 0.035 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.023 ± 0.000 | 0.005 ± 0.000 | - | 0.039 ± 0.006 | 0.428 ± 0.138 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.024 ± 0.001 | 0.009 ± 0.000 | - | 0.031 ± 0.001 | 0.402 ± 0.024 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.024 ± 0.000 | 0.009 ± 0.002 | - | 0.034 ± 0.001 | 0.417 ± 0.008 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.025 ± 0.001 | 0.009 ± 0.000 | - | 0.042 ± 0.003 | 0.410 ± 0.052 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.036 ± 0.002 | 0.007 ± 0.000 | - | 0.021 ± 0.001 | 0.388 ± 0.043 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.036 ± 0.000 | 0.009 ± 0.000 | - | 0.024 ± 0.001 | 0.418 ± 0.103 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.043 ± 0.003 | 0.014 ± 0.000 | - | 0.034 ± 0.005 | 0.405 ± 0.019 |
| small | `matmul` | f64 | 4 | `2x2` | 0.004 ± 0.002 | 0.001 ± 0.000 | - | 0.009 ± 0.002 | 0.040 ± 0.005 |
| small | `matmul` | f64 | 4 | `4x4` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.010 ± 0.000 | 0.040 ± 0.003 |
| small | `matmul` | f64 | 4 | `8x8` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.020 ± 0.001 | 0.123 ± 0.062 |
| small | `qr` | f64 | 4 | `2x2` | 0.002 ± 0.001 | 0.001 ± 0.000 | - | 0.021 ± 0.007 | 0.027 ± 0.008 |
| small | `qr` | f64 | 4 | `4x4` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.028 ± 0.008 | 0.029 ± 0.007 |
| small | `qr` | f64 | 4 | `8x8` | 0.003 ± 0.000 | 0.002 ± 0.000 | - | 0.035 ± 0.015 | 0.033 ± 0.005 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.002 ± 0.001 | 0.001 ± 0.000 | - | 0.015 ± 0.000 | 0.042 ± 0.005 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.005 ± 0.000 | 0.001 ± 0.000 | - | 0.016 ± 0.000 | 0.044 ± 0.004 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.018 ± 0.003 | 0.047 ± 0.005 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.018 ± 0.000 | 0.048 ± 0.009 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.023 ± 0.004 | 0.051 ± 0.004 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.002 ± 0.001 | 0.001 ± 0.000 | - | 0.027 ± 0.003 | 0.054 ± 0.007 |
| small | `svd` | f64 | 4 | `2x2` | 0.003 ± 0.001 | 0.001 ± 0.000 | - | 0.009 ± 0.001 | 0.029 ± 0.007 |
| small | `svd` | f64 | 4 | `4x4` | 0.003 ± 0.000 | 0.002 ± 0.000 | - | 0.011 ± 0.000 | 0.030 ± 0.008 |
| small | `svd` | f64 | 4 | `8x8` | 0.007 ± 0.000 | 0.006 ± 0.000 | - | 0.020 ± 0.002 | 0.042 ± 0.004 |

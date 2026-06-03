# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Timestamp: `20260603_184022`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/cpu/einsum/20260603_184022`.

- tenferro-rs commit: `cc46577429a8cc5a91cb0ed86910b47b2055b420`

## CPU Information

- Model: `arm`
- Vendor: `unknown`
- Logical CPUs: `18`
- Sockets: `unknown`
- Cores per socket: `unknown`
- Threads per core: `unknown`
- NUMA nodes: `unknown`
- Python platform: `macOS-26.5-arm64-arm-64bit`

## Threads: 1

- CSV: `data/results/cpu/einsum/20260603_184022/cpu_ops_t1_20260603_184022.csv`
- Source table: `data/results/cpu/einsum/20260603_184022/cpu_ops_t1_20260603_184022.md`

## PR884 CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.009 ± 0.000 | 0.007 ± 0.000 | - | 0.074 ± 0.002 | 0.099 ± 0.009 |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.030 ± 0.000 | 0.025 ± 0.000 | - | 0.271 ± 0.002 | 0.338 ± 0.035 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.021 ± 0.000 | 0.017 ± 0.000 | - | 0.108 ± 0.001 | 0.135 ± 0.006 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.076 ± 0.001 | 0.071 ± 0.009 | - | 0.417 ± 0.009 | 0.465 ± 0.011 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.024 ± 0.001 | 0.111 ± 0.008 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.064 ± 0.003 | 0.165 ± 0.021 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.066 ± 0.002 | 0.167 ± 0.013 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.003 ± 0.000 | 0.002 ± 0.001 | - | 0.226 ± 0.008 | 0.342 ± 0.013 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.008 ± 0.000 | 0.006 ± 0.000 | - | 0.051 ± 0.001 | 0.079 ± 0.014 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.026 ± 0.000 | 0.022 ± 0.000 | - | 0.183 ± 0.003 | 0.209 ± 0.006 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.011 ± 0.003 | 0.009 ± 0.000 | - | 0.077 ± 0.001 | 0.107 ± 0.008 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.036 ± 0.000 | 0.032 ± 0.011 | - | 0.298 ± 0.003 | 0.343 ± 0.040 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.007 ± 0.000 | 0.005 ± 0.000 | - | 0.087 ± 0.005 | 0.123 ± 0.036 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.025 ± 0.001 | 0.019 ± 0.000 | - | 0.284 ± 0.007 | 0.343 ± 0.021 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.009 ± 0.000 | 0.006 ± 0.000 | - | 0.113 ± 0.004 | 0.147 ± 0.008 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.030 ± 0.000 | 0.022 ± 0.003 | - | 0.411 ± 0.006 | 0.463 ± 0.017 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.013 ± 0.000 | 0.011 ± 0.000 | - | 0.056 ± 0.002 | 0.088 ± 0.006 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.044 ± 0.001 | 0.040 ± 0.000 | - | 0.200 ± 0.002 | 0.269 ± 0.017 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.031 ± 0.000 | 0.027 ± 0.000 | - | 0.101 ± 0.004 | 0.132 ± 0.004 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.119 ± 0.004 | 0.108 ± 0.001 | - | 0.368 ± 0.009 | 0.466 ± 0.032 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.026 ± 0.001 | 0.004 ± 0.000 | - | 0.052 ± 0.003 | 0.534 ± 0.107 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.027 ± 0.000 | 0.007 ± 0.000 | - | 0.095 ± 0.014 | 0.515 ± 0.042 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.027 ± 0.001 | 0.005 ± 0.000 | - | 0.091 ± 0.002 | 0.513 ± 0.074 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.029 ± 0.000 | 0.010 ± 0.001 | - | 0.253 ± 0.006 | 0.732 ± 0.117 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.039 ± 0.001 | 0.032 ± 0.001 | - | 0.104 ± 0.003 | 0.483 ± 0.018 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.070 ± 0.001 | 0.105 ± 0.012 | - | 0.307 ± 0.002 | 0.698 ± 0.034 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.043 ± 0.001 | 0.037 ± 0.000 | - | 0.132 ± 0.002 | 0.504 ± 0.027 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.081 ± 0.001 | 0.121 ± 0.001 | - | 0.439 ± 0.035 | 1.177 ± 0.810 |
| large | `eigh` | f64 | 1 | `64x64` | 0.235 ± 0.006 | 0.155 ± 0.005 | - | 0.636 ± 0.010 | 0.732 ± 0.051 |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | 0.022 ± 0.006 | 0.016 ± 0.004 | - | 0.867 ± 0.052 | 0.980 ± 0.066 |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | 0.072 ± 0.002 | 0.051 ± 0.003 | - | 0.989 ± 0.190 | 1.279 ± 0.215 |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | 0.103 ± 0.000 | 0.065 ± 0.000 | - | 0.498 ± 0.017 | 0.941 ± 0.058 |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | 0.493 ± 0.015 | 0.434 ± 0.016 | - | 0.760 ± 0.009 | 0.970 ± 0.148 |
| large | `matmul` | f64 | 1 | `128x128` | 0.153 ± 0.007 | 0.082 ± 0.000 | - | 3.383 ± 0.124 | 3.869 ± 0.134 |
| large | `matmul` | f64 | 1 | `256x256` | 0.989 ± 0.135 | 0.633 ± 0.013 | - | 13.594 ± 0.310 | 14.404 ± 0.349 |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | 3.074 ± 0.166 | 2.562 ± 0.081 | - | 54.782 ± 0.604 | 58.327 ± 0.798 |
| large | `qr` | f64 | 1 | `64x64` | 0.066 ± 0.001 | 0.055 ± 0.003 | - | 0.496 ± 0.019 | 0.579 ± 0.034 |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | 0.068 ± 0.004 | 0.013 ± 0.003 | - | 0.474 ± 0.029 | 0.571 ± 0.075 |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | 0.084 ± 0.012 | 0.017 ± 0.002 | - | 0.580 ± 0.014 | 0.668 ± 0.023 |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | 0.092 ± 0.005 | 0.028 ± 0.002 | - | 0.885 ± 0.052 | 1.014 ± 0.088 |
| large | `svd` | f64 | 1 | `64x64` | 0.448 ± 0.009 | 0.375 ± 0.008 | - | 0.723 ± 0.020 | 0.806 ± 0.066 |
| small | `eigh` | f64 | 1 | `2x2` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.010 ± 0.002 | 0.033 ± 0.009 |
| small | `eigh` | f64 | 1 | `4x4` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.011 ± 0.000 | 0.031 ± 0.006 |
| small | `eigh` | f64 | 1 | `8x8` | 0.009 ± 0.000 | 0.003 ± 0.000 | - | 0.018 ± 0.000 | 0.101 ± 0.085 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | 0.037 ± 0.008 | 0.005 ± 0.000 | - | 0.012 ± 0.001 | 0.095 ± 0.007 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | 0.016 ± 0.001 | 0.005 ± 0.000 | - | 0.015 ± 0.000 | 0.097 ± 0.009 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | 0.026 ± 0.000 | 0.005 ± 0.000 | - | 0.025 ± 0.000 | 0.447 ± 0.115 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | 0.029 ± 0.003 | 0.003 ± 0.000 | - | 0.021 ± 0.004 | 0.352 ± 0.015 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | 0.046 ± 0.001 | 0.003 ± 0.000 | - | 0.023 ± 0.000 | 0.360 ± 0.028 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | 0.040 ± 0.001 | 0.004 ± 0.000 | - | 0.035 ± 0.001 | 0.484 ± 0.097 |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | 0.030 ± 0.001 | 0.009 ± 0.000 | - | 0.031 ± 0.001 | 0.426 ± 0.048 |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | 0.031 ± 0.000 | 0.009 ± 0.000 | - | 0.032 ± 0.001 | 0.766 ± 0.446 |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | 0.031 ± 0.002 | 0.010 ± 0.000 | - | 0.039 ± 0.001 | 0.496 ± 0.194 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | 0.043 ± 0.003 | 0.007 ± 0.000 | - | 0.021 ± 0.001 | 0.384 ± 0.024 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | 0.045 ± 0.000 | 0.009 ± 0.000 | - | 0.023 ± 0.001 | 0.400 ± 0.016 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | 0.073 ± 0.000 | 0.012 ± 0.000 | - | 0.033 ± 0.001 | 0.481 ± 0.088 |
| small | `matmul` | f64 | 1 | `2x2` | 0.002 ± 0.001 | 0.001 ± 0.000 | - | 0.009 ± 0.001 | 0.039 ± 0.005 |
| small | `matmul` | f64 | 1 | `4x4` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.010 ± 0.000 | 0.042 ± 0.005 |
| small | `matmul` | f64 | 1 | `8x8` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.020 ± 0.000 | 0.311 ± 0.102 |
| small | `qr` | f64 | 1 | `2x2` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.007 ± 0.001 | 0.035 ± 0.012 |
| small | `qr` | f64 | 1 | `4x4` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.008 ± 0.000 | 0.028 ± 0.008 |
| small | `qr` | f64 | 1 | `8x8` | 0.005 ± 0.000 | 0.001 ± 0.000 | - | 0.014 ± 0.000 | 0.038 ± 0.010 |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.015 ± 0.001 | 0.042 ± 0.013 |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.021 ± 0.025 | 0.045 ± 0.050 |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.017 ± 0.000 | 0.047 ± 0.006 |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.018 ± 0.000 | 0.046 ± 0.004 |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | 0.005 ± 0.000 | 0.001 ± 0.000 | - | 0.023 ± 0.000 | 0.070 ± 0.091 |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | 0.005 ± 0.000 | 0.001 ± 0.000 | - | 0.026 ± 0.001 | 0.055 ± 0.004 |
| small | `svd` | f64 | 1 | `2x2` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.009 ± 0.002 | 0.032 ± 0.004 |
| small | `svd` | f64 | 1 | `4x4` | 0.003 ± 0.000 | 0.002 ± 0.000 | - | 0.011 ± 0.000 | 0.032 ± 0.005 |
| small | `svd` | f64 | 1 | `8x8` | 0.013 ± 0.000 | 0.006 ± 0.000 | - | 0.020 ± 0.001 | 0.074 ± 0.046 |

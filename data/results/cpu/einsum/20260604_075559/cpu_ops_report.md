# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Timestamp: `20260604_075559`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/cpu/einsum/20260604_075559`.

- tenferro-rs commit: `cc46577429a8cc5a91cb0ed86910b47b2055b420`

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
- NUMEXPR_NUM_THREADS: `1`
- BLIS_NUM_THREADS: `1`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`

## Threads: 1

- CSV: `data/results/cpu/einsum/20260604_075559/cpu_ops_t1_20260604_075559.csv`
- Source table: `data/results/cpu/einsum/20260604_075559/cpu_ops_t1_20260604_075559.md`

## PR884 CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.010 ± 0.000 | 0.007 ± 0.000 | - | 0.089 ± 0.015 | 0.103 ± 0.004 |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.033 ± 0.001 | 0.025 ± 0.000 | - | 0.280 ± 0.013 | 0.350 ± 0.040 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.021 ± 0.000 | 0.017 ± 0.000 | - | 0.108 ± 0.004 | 0.133 ± 0.004 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.077 ± 0.002 | 0.066 ± 0.002 | - | 0.424 ± 0.009 | 0.487 ± 0.033 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.025 ± 0.002 | 0.109 ± 0.009 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.069 ± 0.004 | 0.163 ± 0.011 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.062 ± 0.002 | 0.157 ± 0.007 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.003 ± 0.000 | 0.002 ± 0.001 | - | 0.223 ± 0.006 | 0.344 ± 0.035 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.009 ± 0.000 | 0.006 ± 0.000 | - | 0.051 ± 0.000 | 0.074 ± 0.003 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.027 ± 0.003 | 0.023 ± 0.000 | - | 0.183 ± 0.001 | 0.217 ± 0.015 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.012 ± 0.006 | 0.009 ± 0.000 | - | 0.075 ± 0.001 | 0.112 ± 0.012 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.036 ± 0.000 | 0.031 ± 0.000 | - | 0.297 ± 0.034 | 0.374 ± 0.105 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.008 ± 0.000 | 0.005 ± 0.000 | - | 0.086 ± 0.007 | 0.124 ± 0.007 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.024 ± 0.003 | 0.019 ± 0.000 | - | 0.293 ± 0.009 | 0.398 ± 0.060 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.010 ± 0.000 | 0.007 ± 0.000 | - | 0.109 ± 0.001 | 0.148 ± 0.013 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.032 ± 0.000 | 0.022 ± 0.000 | - | 0.412 ± 0.019 | 0.487 ± 0.021 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.013 ± 0.001 | 0.011 ± 0.000 | - | 0.055 ± 0.002 | 0.099 ± 0.040 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.045 ± 0.003 | 0.041 ± 0.001 | - | 0.196 ± 0.004 | 0.280 ± 0.010 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.032 ± 0.002 | 0.027 ± 0.000 | - | 0.095 ± 0.001 | 0.130 ± 0.012 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.115 ± 0.001 | 0.113 ± 0.021 | - | 0.361 ± 0.015 | 0.465 ± 0.055 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.027 ± 0.000 | 0.004 ± 0.000 | - | 0.049 ± 0.003 | 0.486 ± 0.051 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.028 ± 0.000 | 0.007 ± 0.000 | - | 0.092 ± 0.007 | 0.533 ± 0.084 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.026 ± 0.002 | 0.005 ± 0.001 | - | 0.087 ± 0.014 | 0.533 ± 0.154 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.030 ± 0.000 | 0.011 ± 0.001 | - | 0.248 ± 0.006 | 0.741 ± 0.107 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.048 ± 0.007 | 0.033 ± 0.000 | - | 0.100 ± 0.001 | 0.513 ± 0.096 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.071 ± 0.002 | 0.102 ± 0.001 | - | 0.318 ± 0.024 | 0.686 ± 0.024 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.042 ± 0.002 | 0.041 ± 0.005 | - | 0.132 ± 0.008 | 0.513 ± 0.026 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.082 ± 0.001 | 0.121 ± 0.002 | - | 0.437 ± 0.014 | 0.839 ± 0.090 |
| large | `eigh` | f64 | 1 | `64x64` | 0.239 ± 0.029 | 0.155 ± 0.002 | - | 0.651 ± 0.027 | 0.730 ± 0.090 |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | 0.020 ± 0.000 | 0.017 ± 0.004 | - | 0.864 ± 0.043 | 0.986 ± 0.039 |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | 0.064 ± 0.001 | 0.056 ± 0.002 | - | 0.880 ± 0.027 | 1.330 ± 0.193 |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | 0.111 ± 0.005 | 0.066 ± 0.001 | - | 0.489 ± 0.006 | 0.905 ± 0.089 |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | 0.480 ± 0.008 | 0.431 ± 0.026 | - | 0.738 ± 0.011 | 1.004 ± 0.102 |
| large | `matmul` | f64 | 1 | `128x128` | 0.154 ± 0.002 | 0.085 ± 0.003 | - | 3.436 ± 0.100 | 3.566 ± 0.124 |
| large | `matmul` | f64 | 1 | `256x256` | 0.902 ± 0.154 | 0.618 ± 0.020 | - | 13.636 ± 0.126 | 14.201 ± 0.266 |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | 3.123 ± 0.174 | 2.577 ± 0.021 | - | 55.181 ± 1.119 | 56.976 ± 1.684 |
| large | `qr` | f64 | 1 | `64x64` | 0.067 ± 0.001 | 0.054 ± 0.002 | - | 0.505 ± 0.034 | 0.565 ± 0.028 |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | 0.069 ± 0.002 | 0.013 ± 0.000 | - | 0.471 ± 0.018 | 0.600 ± 0.096 |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | 0.075 ± 0.003 | 0.015 ± 0.000 | - | 0.579 ± 0.015 | 0.704 ± 0.090 |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | 0.087 ± 0.003 | 0.027 ± 0.000 | - | 0.885 ± 0.012 | 1.010 ± 0.107 |
| large | `svd` | f64 | 1 | `64x64` | 0.443 ± 0.003 | 0.379 ± 0.005 | - | 0.742 ± 0.066 | 0.819 ± 0.053 |
| small | `eigh` | f64 | 1 | `2x2` | 0.004 ± 0.001 | 0.001 ± 0.000 | - | 0.009 ± 0.001 | 0.029 ± 0.006 |
| small | `eigh` | f64 | 1 | `4x4` | 0.005 ± 0.000 | 0.001 ± 0.000 | - | 0.011 ± 0.000 | 0.033 ± 0.005 |
| small | `eigh` | f64 | 1 | `8x8` | 0.008 ± 0.000 | 0.003 ± 0.000 | - | 0.018 ± 0.001 | 0.038 ± 0.006 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | 0.046 ± 0.009 | 0.005 ± 0.000 | - | 0.013 ± 0.001 | 0.093 ± 0.004 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | 0.028 ± 0.001 | 0.005 ± 0.000 | - | 0.014 ± 0.000 | 0.097 ± 0.002 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | 0.023 ± 0.001 | 0.005 ± 0.001 | - | 0.025 ± 0.000 | 0.118 ± 0.005 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | 0.032 ± 0.002 | 0.004 ± 0.000 | - | 0.022 ± 0.003 | 0.383 ± 0.104 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | 0.037 ± 0.019 | 0.003 ± 0.000 | - | 0.024 ± 0.001 | 0.368 ± 0.059 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | 0.041 ± 0.003 | 0.004 ± 0.000 | - | 0.039 ± 0.002 | 0.386 ± 0.032 |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | 0.054 ± 0.001 | 0.009 ± 0.000 | - | 0.030 ± 0.001 | 0.404 ± 0.022 |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | 0.046 ± 0.008 | 0.009 ± 0.000 | - | 0.033 ± 0.004 | 0.403 ± 0.030 |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | 0.031 ± 0.000 | 0.010 ± 0.001 | - | 0.038 ± 0.003 | 0.408 ± 0.037 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | 0.080 ± 0.006 | 0.008 ± 0.000 | - | 0.021 ± 0.001 | 0.384 ± 0.030 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | 0.045 ± 0.006 | 0.009 ± 0.000 | - | 0.025 ± 0.008 | 0.440 ± 0.104 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | 0.050 ± 0.025 | 0.013 ± 0.000 | - | 0.031 ± 0.000 | 0.410 ± 0.029 |
| small | `matmul` | f64 | 1 | `2x2` | 0.006 ± 0.003 | 0.001 ± 0.000 | - | 0.009 ± 0.001 | 0.042 ± 0.006 |
| small | `matmul` | f64 | 1 | `4x4` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.010 ± 0.000 | 0.041 ± 0.004 |
| small | `matmul` | f64 | 1 | `8x8` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.020 ± 0.001 | 0.065 ± 0.007 |
| small | `qr` | f64 | 1 | `2x2` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.007 ± 0.000 | 0.028 ± 0.006 |
| small | `qr` | f64 | 1 | `4x4` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.008 ± 0.000 | 0.029 ± 0.006 |
| small | `qr` | f64 | 1 | `8x8` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.014 ± 0.000 | 0.035 ± 0.007 |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | 0.004 ± 0.001 | 0.001 ± 0.000 | - | 0.015 ± 0.001 | 0.043 ± 0.004 |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.015 ± 0.000 | 0.048 ± 0.009 |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.016 ± 0.000 | 0.048 ± 0.008 |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.017 ± 0.000 | 0.045 ± 0.006 |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.023 ± 0.001 | 0.052 ± 0.005 |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.024 ± 0.000 | 0.054 ± 0.010 |
| small | `svd` | f64 | 1 | `2x2` | 0.004 ± 0.001 | 0.001 ± 0.000 | - | 0.012 ± 0.025 | 0.030 ± 0.010 |
| small | `svd` | f64 | 1 | `4x4` | 0.006 ± 0.000 | 0.002 ± 0.000 | - | 0.010 ± 0.001 | 0.033 ± 0.007 |
| small | `svd` | f64 | 1 | `8x8` | 0.012 ± 0.001 | 0.006 ± 0.000 | - | 0.019 ± 0.001 | 0.039 ± 0.005 |

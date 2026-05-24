# CPU Benchmark Results

Latest run: `./scripts/run_all.sh 4`.

This file collects the latest PR884 CPU benchmark item comparison for each thread count under `data/results/`.

- tenferro-rs commit: `70dd331362dc02c87727ddee5d034da59e410ad5`

## Threads: 1

- Timestamp: `20260524_162107`
- Source table: `data/results/cpu_ops_t1_20260524_162107.md`

Logs:

- `data/results/cpu_ops_t1_20260524_162107.csv`

## PR884 CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.010 ± 0.000 | 0.007 ± 0.000 | 0.140 ± 0.007 | 0.115 ± 0.006 | 0.143 ± 0.011 |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.032 ± 0.003 | 0.024 ± 0.000 | 0.557 ± 0.022 | 0.417 ± 0.015 | 0.473 ± 0.016 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.027 ± 0.006 | 0.020 ± 0.000 | 0.238 ± 0.008 | 0.166 ± 0.010 | 0.197 ± 0.009 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.091 ± 0.011 | 0.077 ± 0.000 | 0.916 ± 0.048 | 0.637 ± 0.029 | 0.671 ± 0.033 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.007 ± 0.000 | 0.038 ± 0.004 | 0.161 ± 0.039 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.096 ± 0.013 | 0.220 ± 0.013 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.003 ± 0.001 | 0.002 ± 0.000 | 0.007 ± 0.000 | 0.096 ± 0.015 | 0.248 ± 0.028 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.007 ± 0.000 | 0.006 ± 0.003 | 0.009 ± 0.002 | 0.355 ± 0.090 | 0.510 ± 0.043 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.011 ± 0.000 | 0.009 ± 0.000 | 0.112 ± 0.010 | 0.076 ± 0.004 | 0.106 ± 0.007 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.039 ± 0.000 | 0.032 ± 0.000 | 0.450 ± 0.021 | 0.294 ± 0.014 | 0.307 ± 0.007 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.017 ± 0.000 | 0.014 ± 0.000 | 0.209 ± 0.008 | 0.114 ± 0.001 | 0.149 ± 0.010 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.067 ± 0.010 | 0.054 ± 0.007 | 0.778 ± 0.021 | 0.455 ± 0.047 | 0.525 ± 0.061 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.008 ± 0.000 | 0.005 ± 0.000 | 0.143 ± 0.008 | 0.124 ± 0.008 | 0.187 ± 0.019 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.023 ± 0.000 | 0.016 ± 0.002 | 0.536 ± 0.024 | 0.453 ± 0.124 | 0.554 ± 0.051 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.011 ± 0.000 | 0.006 ± 0.000 | 0.238 ± 0.022 | 0.165 ± 0.002 | 0.224 ± 0.038 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.035 ± 0.002 | 0.024 ± 0.000 | 0.938 ± 0.178 | 0.595 ± 0.040 | 0.771 ± 0.125 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.020 ± 0.000 | 0.017 ± 0.000 | 0.121 ± 0.013 | 0.086 ± 0.008 | 0.134 ± 0.016 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.075 ± 0.006 | 0.066 ± 0.005 | 0.465 ± 0.022 | 0.293 ± 0.011 | 0.395 ± 0.009 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.048 ± 0.010 | 0.041 ± 0.002 | 0.226 ± 0.015 | 0.148 ± 0.011 | 0.202 ± 0.012 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.172 ± 0.003 | 0.182 ± 0.015 | 0.909 ± 0.036 | 0.555 ± 0.071 | 0.664 ± 0.021 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.036 ± 0.000 | 0.007 ± 0.000 | 0.038 ± 0.004 | 0.082 ± 0.010 | 0.650 ± 0.108 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.046 ± 0.000 | 0.017 ± 0.000 | 0.038 ± 0.008 | 0.141 ± 0.005 | 0.811 ± 0.079 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.039 ± 0.002 | 0.008 ± 0.000 | 0.037 ± 0.002 | 0.140 ± 0.001 | 0.753 ± 0.035 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.048 ± 0.002 | 0.021 ± 0.000 | 0.043 ± 0.005 | 0.389 ± 0.017 | 0.999 ± 0.042 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | panic | error: extension: backend failure: family_id="tenferro-linalg.linalg.v1": transpose: rank mismatch expected 3, actual 2 | 0.173 ± 0.019 | 0.156 ± 0.005 | 0.665 ± 0.051 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | panic | error: extension: backend failure: family_id="tenferro-linalg.linalg.v1": transpose: rank mismatch expected 3, actual 2 | 0.587 ± 0.013 | 0.461 ± 0.004 | 1.036 ± 0.087 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | panic | error: extension: backend failure: family_id="tenferro-linalg.linalg.v1": transpose: rank mismatch expected 3, actual 2 | 0.262 ± 0.007 | 0.194 ± 0.007 | 0.721 ± 0.038 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | panic | error: extension: backend failure: family_id="tenferro-linalg.linalg.v1": transpose: rank mismatch expected 3, actual 2 | 0.927 ± 0.057 | 0.641 ± 0.056 | 1.146 ± 0.063 |
| large | `eigh` | f64 | 1 | `64x64` | 0.437 ± 0.002 | 0.330 ± 0.005 | 0.432 ± 0.021 | 0.909 ± 0.019 | 1.004 ± 0.041 |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | 0.023 ± 0.000 | 0.016 ± 0.000 | 0.021 ± 0.000 | 1.338 ± 0.031 | 1.386 ± 0.131 |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | 0.072 ± 0.004 | 0.058 ± 0.006 | 0.055 ± 0.003 | 1.349 ± 0.078 | 1.845 ± 0.017 |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | 0.186 ± 0.000 | 0.161 ± 0.030 | 0.235 ± 0.003 | 0.768 ± 0.033 | 1.270 ± 0.018 |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | 0.702 ± 0.011 | 0.703 ± 0.007 | 0.590 ± 0.050 | 1.087 ± 0.031 | 1.282 ± 0.029 |
| large | `matmul` | f64 | 1 | `128x128` | 0.279 ± 0.007 | 0.093 ± 0.001 | 0.133 ± 0.011 | 5.133 ± 0.251 | 5.389 ± 0.173 |
| large | `matmul` | f64 | 1 | `256x256` | 1.599 ± 0.239 | 0.702 ± 0.008 | 0.870 ± 0.035 | 21.015 ± 0.296 | 21.058 ± 0.628 |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | 3.402 ± 0.545 | 3.175 ± 0.118 | 3.156 ± 0.057 | 84.267 ± 1.502 | 85.138 ± 1.925 |
| large | `qr` | f64 | 1 | `64x64` | 0.095 ± 0.004 | 0.084 ± 0.004 | 0.266 ± 0.015 | 0.747 ± 0.032 | 0.752 ± 0.023 |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | 0.133 ± 0.002 | 0.028 ± 0.004 | 0.220 ± 0.020 | 0.724 ± 0.014 | 0.916 ± 0.095 |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | 0.140 ± 0.008 | 0.033 ± 0.001 | 0.222 ± 0.005 | 0.883 ± 0.090 | 0.944 ± 0.108 |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | 0.165 ± 0.004 | 0.062 ± 0.007 | 0.245 ± 0.002 | 1.416 ± 0.094 | 1.434 ± 0.047 |
| large | `svd` | f64 | 1 | `64x64` | 0.640 ± 0.008 | 0.624 ± 0.025 | 0.554 ± 0.028 | 1.026 ± 0.037 | 1.103 ± 0.037 |
| small | `eigh` | f64 | 1 | `2x2` | 0.007 ± 0.003 | 0.001 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.001 | 0.043 ± 0.011 |
| small | `eigh` | f64 | 1 | `4x4` | 0.008 ± 0.002 | 0.002 ± 0.000 | 0.019 ± 0.000 | 0.018 ± 0.002 | 0.046 ± 0.006 |
| small | `eigh` | f64 | 1 | `8x8` | 0.017 ± 0.000 | 0.004 ± 0.000 | 0.032 ± 0.000 | 0.027 ± 0.000 | 0.061 ± 0.013 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | 0.014 ± 0.005 | 0.008 ± 0.002 | 0.007 ± 0.000 | 0.021 ± 0.001 | 0.251 ± 0.139 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | 0.009 ± 0.002 | 0.007 ± 0.003 | 0.007 ± 0.000 | 0.023 ± 0.001 | 0.148 ± 0.012 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | 0.010 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.038 ± 0.001 | 0.157 ± 0.007 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | 0.111 ± 0.013 | 0.004 ± 0.000 | 0.014 ± 0.001 | 0.036 ± 0.006 | 0.492 ± 0.045 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | 0.075 ± 0.001 | 0.003 ± 0.000 | 0.014 ± 0.000 | 0.038 ± 0.001 | 0.504 ± 0.029 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | 0.089 ± 0.002 | 0.004 ± 0.000 | 0.014 ± 0.000 | 0.060 ± 0.001 | 0.521 ± 0.018 |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | 0.091 ± 0.006 | 0.010 ± 0.000 | 0.035 ± 0.000 | 0.051 ± 0.002 | 0.557 ± 0.052 |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | 0.100 ± 0.009 | 0.009 ± 0.000 | 0.040 ± 0.000 | 0.052 ± 0.002 | 0.618 ± 0.081 |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | 0.085 ± 0.011 | 0.013 ± 0.000 | 0.052 ± 0.008 | 0.059 ± 0.001 | 0.589 ± 0.034 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | 0.139 ± 0.008 | 0.010 ± 0.001 | 0.025 ± 0.001 | 0.034 ± 0.003 | 0.555 ± 0.057 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | 0.151 ± 0.007 | 0.009 ± 0.000 | 0.032 ± 0.000 | 0.037 ± 0.001 | 0.562 ± 0.013 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | 0.165 ± 0.001 | 0.016 ± 0.004 | 0.048 ± 0.000 | 0.050 ± 0.002 | 0.596 ± 0.023 |
| small | `matmul` | f64 | 1 | `2x2` | 0.009 ± 0.013 | 0.001 ± 0.001 | 0.002 ± 0.000 | 0.017 ± 0.001 | 0.071 ± 0.022 |
| small | `matmul` | f64 | 1 | `4x4` | 0.005 ± 0.000 | 0.001 ± 0.000 | 0.002 ± 0.000 | 0.016 ± 0.000 | 0.061 ± 0.007 |
| small | `matmul` | f64 | 1 | `8x8` | 0.007 ± 0.007 | 0.001 ± 0.000 | 0.002 ± 0.000 | 0.029 ± 0.000 | 0.101 ± 0.009 |
| small | `qr` | f64 | 1 | `2x2` | 0.008 ± 0.004 | 0.001 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.002 | 0.045 ± 0.007 |
| small | `qr` | f64 | 1 | `4x4` | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.015 ± 0.000 | 0.015 ± 0.003 | 0.042 ± 0.008 |
| small | `qr` | f64 | 1 | `8x8` | 0.010 ± 0.000 | 0.002 ± 0.000 | 0.027 ± 0.000 | 0.021 ± 0.000 | 0.049 ± 0.013 |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | 0.008 ± 0.000 | 0.001 ± 0.000 | 0.018 ± 0.001 | 0.025 ± 0.001 | 0.070 ± 0.013 |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | 0.008 ± 0.000 | 0.001 ± 0.000 | 0.017 ± 0.000 | 0.026 ± 0.000 | 0.066 ± 0.010 |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.023 ± 0.000 | 0.027 ± 0.001 | 0.073 ± 0.015 |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | 0.017 ± 0.040 | 0.001 ± 0.000 | 0.023 ± 0.000 | 0.028 ± 0.000 | 0.073 ± 0.014 |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | 0.010 ± 0.000 | 0.002 ± 0.000 | 0.034 ± 0.000 | 0.034 ± 0.000 | 0.074 ± 0.007 |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | 0.010 ± 0.000 | 0.002 ± 0.000 | 0.034 ± 0.000 | 0.038 ± 0.000 | 0.084 ± 0.012 |
| small | `svd` | f64 | 1 | `2x2` | 0.009 ± 0.000 | 0.002 ± 0.000 | 0.012 ± 0.001 | 0.014 ± 0.001 | 0.043 ± 0.008 |
| small | `svd` | f64 | 1 | `4x4` | 0.011 ± 0.000 | 0.003 ± 0.000 | 0.019 ± 0.001 | 0.017 ± 0.000 | 0.045 ± 0.006 |
| small | `svd` | f64 | 1 | `8x8` | 0.026 ± 0.001 | 0.008 ± 0.000 | 0.035 ± 0.000 | 0.029 ± 0.000 | 0.064 ± 0.006 |

## Threads: 4

- Timestamp: `20260524_163132`
- Source table: `data/results/cpu_ops_t4_20260524_163132.md`

Logs:

- `data/results/cpu_ops_t4_20260524_163132.csv`

## PR884 CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.010 ± 0.000 | 0.007 ± 0.000 | 0.140 ± 0.009 | 0.114 ± 0.020 | 0.145 ± 0.014 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.031 ± 0.000 | 0.025 ± 0.000 | 0.547 ± 0.023 | 0.409 ± 0.012 | 0.523 ± 0.073 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.024 ± 0.000 | 0.021 ± 0.000 | 0.249 ± 0.026 | 0.158 ± 0.014 | 0.196 ± 0.016 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.089 ± 0.000 | 0.078 ± 0.000 | 0.914 ± 0.015 | 0.599 ± 0.036 | 0.668 ± 0.009 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.007 ± 0.000 | 0.038 ± 0.008 | 0.150 ± 0.013 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.103 ± 0.009 | 0.221 ± 0.022 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.007 ± 0.004 | 0.099 ± 0.008 | 0.217 ± 0.011 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.002 | 0.342 ± 0.005 | 0.484 ± 0.015 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.011 ± 0.000 | 0.009 ± 0.001 | 0.127 ± 0.001 | 0.103 ± 0.018 | 0.108 ± 0.005 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.037 ± 0.004 | 0.032 ± 0.001 | 0.468 ± 0.013 | 0.295 ± 0.025 | 0.309 ± 0.004 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.017 ± 0.000 | 0.014 ± 0.000 | 0.219 ± 0.022 | 0.137 ± 0.008 | 0.143 ± 0.006 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.060 ± 0.007 | 0.054 ± 0.000 | 0.809 ± 0.045 | 0.450 ± 0.015 | 0.483 ± 0.018 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.008 ± 0.000 | 0.005 ± 0.000 | 0.163 ± 0.009 | 0.151 ± 0.015 | 0.167 ± 0.007 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.025 ± 0.001 | 0.018 ± 0.000 | 0.548 ± 0.020 | 0.441 ± 0.012 | 0.519 ± 0.021 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.011 ± 0.000 | 0.007 ± 0.000 | 0.240 ± 0.005 | 0.181 ± 0.003 | 0.207 ± 0.010 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.036 ± 0.004 | 0.025 ± 0.000 | 0.905 ± 0.038 | 0.624 ± 0.029 | 0.667 ± 0.008 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.020 ± 0.000 | 0.017 ± 0.000 | 0.119 ± 0.000 | 0.086 ± 0.005 | 0.131 ± 0.016 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.071 ± 0.003 | 0.063 ± 0.000 | 0.463 ± 0.019 | 0.299 ± 0.013 | 0.391 ± 0.007 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.043 ± 0.001 | 0.041 ± 0.000 | 0.228 ± 0.009 | 0.142 ± 0.008 | 0.194 ± 0.007 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.168 ± 0.005 | 0.160 ± 0.013 | 0.888 ± 0.019 | 0.531 ± 0.007 | 0.663 ± 0.016 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.036 ± 0.001 | 0.007 ± 0.000 | 0.037 ± 0.003 | 0.081 ± 0.003 | 0.687 ± 0.041 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.044 ± 0.000 | 0.018 ± 0.000 | 0.042 ± 0.007 | 0.143 ± 0.009 | 0.728 ± 0.036 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.037 ± 0.002 | 0.008 ± 0.002 | 0.038 ± 0.001 | 0.141 ± 0.008 | 0.730 ± 0.030 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.045 ± 0.002 | 0.022 ± 0.001 | 0.046 ± 0.008 | 0.381 ± 0.012 | 0.987 ± 0.046 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | panic | error: extension: backend failure: family_id="tenferro-linalg.linalg.v1": transpose: rank mismatch expected 3, actual 2 | 0.184 ± 0.031 | 0.180 ± 0.003 | 0.649 ± 0.034 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | panic | error: extension: backend failure: family_id="tenferro-linalg.linalg.v1": transpose: rank mismatch expected 3, actual 2 | 0.590 ± 0.010 | 0.496 ± 0.050 | 0.966 ± 0.042 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | panic | error: extension: backend failure: family_id="tenferro-linalg.linalg.v1": transpose: rank mismatch expected 3, actual 2 | 0.275 ± 0.017 | 0.215 ± 0.012 | 0.693 ± 0.052 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | panic | error: extension: backend failure: family_id="tenferro-linalg.linalg.v1": transpose: rank mismatch expected 3, actual 2 | 0.969 ± 0.077 | 0.713 ± 0.082 | 1.142 ± 0.116 |
| large | `eigh` | f64 | 4 | `64x64` | 0.526 ± 0.002 | 0.339 ± 0.003 | 0.481 ± 0.027 | 0.913 ± 0.027 | 1.032 ± 0.069 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.024 ± 0.001 | 0.019 ± 0.004 | 0.021 ± 0.000 | 1.309 ± 0.023 | 1.472 ± 0.014 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.079 ± 0.003 | 0.064 ± 0.004 | 0.055 ± 0.007 | 1.340 ± 0.038 | 1.817 ± 0.047 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.187 ± 0.002 | 0.135 ± 0.000 | 0.234 ± 0.039 | 0.749 ± 0.015 | 1.257 ± 0.008 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 0.730 ± 0.005 | 0.680 ± 0.007 | 0.789 ± 0.199 | 1.064 ± 0.032 | 1.271 ± 0.046 |
| large | `matmul` | f64 | 4 | `128x128` | 0.106 ± 0.001 | 0.049 ± 0.003 | 0.071 ± 0.008 | 5.263 ± 0.239 | 5.226 ± 0.180 |
| large | `matmul` | f64 | 4 | `256x256` | 0.503 ± 0.004 | 0.264 ± 0.017 | 0.327 ± 0.010 | 20.385 ± 0.320 | 20.560 ± 0.420 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 1.382 ± 0.146 | 1.081 ± 0.046 | 1.155 ± 0.076 | 82.880 ± 1.377 | 83.342 ± 0.710 |
| large | `qr` | f64 | 4 | `64x64` | 0.110 ± 0.005 | 0.083 ± 0.000 | 0.282 ± 0.026 | 0.735 ± 0.033 | 0.752 ± 0.018 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.144 ± 0.003 | 0.025 ± 0.000 | 0.217 ± 0.031 | 0.712 ± 0.026 | 0.854 ± 0.074 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.167 ± 0.001 | 0.044 ± 0.004 | 0.234 ± 0.011 | 0.867 ± 0.035 | 0.929 ± 0.053 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.172 ± 0.006 | 0.068 ± 0.025 | 0.258 ± 0.084 | 1.357 ± 0.063 | 1.448 ± 0.048 |
| large | `svd` | f64 | 4 | `64x64` | 0.827 ± 0.029 | 0.625 ± 0.002 | 0.696 ± 0.037 | 1.017 ± 0.086 | 1.073 ± 0.032 |
| small | `eigh` | f64 | 4 | `2x2` | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.013 ± 0.001 | 0.016 ± 0.003 | 0.041 ± 0.009 |
| small | `eigh` | f64 | 4 | `4x4` | 0.007 ± 0.000 | 0.002 ± 0.000 | 0.019 ± 0.000 | 0.021 ± 0.001 | 0.047 ± 0.008 |
| small | `eigh` | f64 | 4 | `8x8` | 0.014 ± 0.000 | 0.004 ± 0.000 | 0.032 ± 0.000 | 0.032 ± 0.007 | 0.067 ± 0.032 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.013 ± 0.003 | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.004 | 0.131 ± 0.010 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.023 ± 0.001 | 0.132 ± 0.007 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.040 ± 0.002 | 0.158 ± 0.004 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.083 ± 0.031 | 0.004 ± 0.001 | 0.014 ± 0.001 | 0.038 ± 0.042 | 0.480 ± 0.023 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.072 ± 0.003 | 0.004 ± 0.000 | 0.014 ± 0.001 | 0.040 ± 0.001 | 0.509 ± 0.066 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.068 ± 0.014 | 0.004 ± 0.000 | 0.014 ± 0.000 | 0.062 ± 0.008 | 0.531 ± 0.037 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.080 ± 0.002 | 0.010 ± 0.000 | 0.035 ± 0.001 | 0.050 ± 0.001 | 0.555 ± 0.017 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.080 ± 0.001 | 0.010 ± 0.002 | 0.040 ± 0.001 | 0.055 ± 0.001 | 0.574 ± 0.050 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.071 ± 0.001 | 0.013 ± 0.000 | 0.056 ± 0.011 | 0.062 ± 0.007 | 0.562 ± 0.042 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.123 ± 0.004 | 0.009 ± 0.000 | 0.025 ± 0.001 | 0.033 ± 0.002 | 0.563 ± 0.049 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.122 ± 0.005 | 0.011 ± 0.000 | 0.032 ± 0.001 | 0.039 ± 0.001 | 0.554 ± 0.013 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.114 ± 0.003 | 0.016 ± 0.000 | 0.051 ± 0.009 | 0.051 ± 0.001 | 0.571 ± 0.051 |
| small | `matmul` | f64 | 4 | `2x2` | 0.009 ± 0.003 | 0.001 ± 0.000 | 0.002 ± 0.000 | 0.014 ± 0.002 | 0.062 ± 0.014 |
| small | `matmul` | f64 | 4 | `4x4` | 0.005 ± 0.000 | 0.001 ± 0.000 | 0.002 ± 0.000 | 0.016 ± 0.000 | 0.063 ± 0.006 |
| small | `matmul` | f64 | 4 | `8x8` | 0.005 ± 0.001 | 0.001 ± 0.000 | 0.002 ± 0.000 | 0.031 ± 0.000 | 0.121 ± 0.052 |
| small | `qr` | f64 | 4 | `2x2` | 0.007 ± 0.003 | 0.001 ± 0.000 | 0.030 ± 0.006 | 0.041 ± 0.013 | 0.041 ± 0.007 |
| small | `qr` | f64 | 4 | `4x4` | 0.006 ± 0.002 | 0.001 ± 0.000 | 0.031 ± 0.001 | 0.030 ± 0.017 | 0.051 ± 0.037 |
| small | `qr` | f64 | 4 | `8x8` | 0.008 ± 0.000 | 0.002 ± 0.000 | 0.044 ± 0.001 | 0.053 ± 0.011 | 0.050 ± 0.013 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.007 ± 0.003 | 0.001 ± 0.000 | 0.017 ± 0.000 | 0.026 ± 0.002 | 0.063 ± 0.013 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.033 ± 0.006 | 0.013 ± 0.001 | 0.035 ± 0.002 | 0.026 ± 0.000 | 0.066 ± 0.006 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.022 ± 0.001 | 0.030 ± 0.001 | 0.067 ± 0.006 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.028 ± 0.021 | 0.015 ± 0.005 | 0.042 ± 0.001 | 0.030 ± 0.000 | 0.072 ± 0.009 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.008 ± 0.000 | 0.002 ± 0.000 | 0.034 ± 0.003 | 0.042 ± 0.006 | 0.076 ± 0.007 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.033 ± 0.002 | 0.015 ± 0.039 | 0.054 ± 0.005 | 0.039 ± 0.004 | 0.079 ± 0.010 |
| small | `svd` | f64 | 4 | `2x2` | 0.008 ± 0.001 | 0.002 ± 0.000 | 0.012 ± 0.001 | 0.014 ± 0.001 | 0.042 ± 0.010 |
| small | `svd` | f64 | 4 | `4x4` | 0.010 ± 0.000 | 0.003 ± 0.000 | 0.019 ± 0.000 | 0.017 ± 0.000 | 0.046 ± 0.007 |
| small | `svd` | f64 | 4 | `8x8` | 0.021 ± 0.000 | 0.008 ± 0.000 | 0.041 ± 0.007 | 0.030 ± 0.000 | 0.060 ± 0.009 |

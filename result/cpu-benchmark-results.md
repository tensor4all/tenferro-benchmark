# CPU Benchmark Results

Latest run: `./scripts/run_all.sh 4`.

This file collects the latest PR884 CPU benchmark item comparison for each thread count under `data/results/`.

- tenferro-rs commit: `fa722375c8662b5532a6f875a6bef3494ace40b5`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-124-generic-x86_64-with-glibc2.39`

## Threads: 1

- Timestamp: `20260531_023422`
- Source table: `data/results/cpu_ops_t1_20260531_023422.md`

Logs:

- `data/results/cpu_ops_t1_20260531_023422.csv`

## PR884 CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.022 ± 0.000 | 0.019 ± 0.000 | 0.589 ± 0.013 | 0.551 ± 0.024 | 1.722 ± 0.044 |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.085 ± 0.000 | 0.069 ± 0.000 | 2.286 ± 0.027 | 1.824 ± 0.040 | 6.148 ± 0.060 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.057 ± 0.001 | 0.051 ± 0.001 | 0.887 ± 0.006 | 0.697 ± 0.054 | 2.257 ± 0.060 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.222 ± 0.004 | 0.200 ± 0.004 | 3.492 ± 0.084 | 2.355 ± 0.038 | 7.189 ± 0.118 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.020 ± 0.000 | 0.170 ± 0.013 | 1.087 ± 0.003 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.023 ± 0.001 | 0.300 ± 0.011 | 1.738 ± 0.035 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.024 ± 0.001 | 0.300 ± 0.009 | 2.175 ± 0.187 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.036 ± 0.000 | 0.812 ± 0.012 | 3.062 ± 0.129 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.028 ± 0.002 | 0.020 ± 0.000 | 0.357 ± 0.007 | 0.318 ± 0.008 | 1.021 ± 0.011 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.097 ± 0.000 | 0.063 ± 0.000 | 1.377 ± 0.064 | 1.022 ± 0.015 | 2.581 ± 0.059 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.039 ± 0.000 | 0.035 ± 0.000 | 0.659 ± 0.011 | 0.412 ± 0.017 | 1.441 ± 0.014 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.145 ± 0.001 | 0.135 ± 0.001 | 2.565 ± 0.039 | 1.395 ± 0.017 | 4.679 ± 0.039 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.018 ± 0.000 | 0.015 ± 0.001 | 0.601 ± 0.005 | 0.638 ± 0.065 | 1.793 ± 0.047 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.065 ± 0.000 | 0.050 ± 0.000 | 2.271 ± 0.017 | 1.899 ± 0.008 | 5.983 ± 0.068 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.023 ± 0.000 | 0.017 ± 0.000 | 0.876 ± 0.094 | 0.729 ± 0.018 | 2.447 ± 0.068 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.083 ± 0.002 | 0.062 ± 0.000 | 3.303 ± 0.032 | 2.343 ± 0.038 | 7.838 ± 0.371 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.036 ± 0.013 | 0.030 ± 0.000 | 0.399 ± 0.039 | 0.367 ± 0.012 | 1.030 ± 0.016 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.145 ± 0.011 | 0.114 ± 0.000 | 1.477 ± 0.104 | 1.136 ± 0.105 | 3.060 ± 0.078 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.079 ± 0.001 | 0.076 ± 0.000 | 0.751 ± 0.064 | 0.535 ± 0.013 | 1.553 ± 0.019 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.322 ± 0.004 | 0.298 ± 0.044 | 2.816 ± 0.020 | 1.809 ± 0.123 | 5.450 ± 0.088 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.104 ± 0.001 | 0.014 ± 0.006 | 0.153 ± 0.005 | 0.392 ± 0.011 | 6.036 ± 1.469 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.118 ± 0.003 | 0.025 ± 0.001 | 0.159 ± 0.003 | 0.555 ± 0.009 | 6.264 ± 1.721 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.113 ± 0.005 | 0.014 ± 0.000 | 0.162 ± 0.001 | 0.560 ± 0.011 | 5.879 ± 1.961 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.130 ± 0.003 | 0.031 ± 0.001 | 0.191 ± 0.007 | 1.142 ± 0.016 | 6.494 ± 2.811 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.154 ± 0.001 | 0.087 ± 0.004 | 0.816 ± 0.106 | 0.823 ± 0.013 | 6.995 ± 1.369 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.261 ± 0.012 | 0.277 ± 0.004 | 2.408 ± 0.027 | 2.124 ± 0.063 | 8.269 ± 3.284 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.173 ± 0.001 | 0.105 ± 0.001 | 1.015 ± 0.008 | 0.944 ± 0.006 | 7.087 ± 1.623 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.354 ± 0.002 | 0.340 ± 0.005 | 3.475 ± 0.116 | 2.586 ± 0.080 | 9.452 ± 1.824 |
| large | `eigh` | f64 | 1 | `64x64` | 0.716 ± 0.002 | 0.468 ± 0.055 | 1.207 ± 0.004 | 2.103 ± 0.027 | 4.282 ± 0.181 |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | 0.043 ± 0.001 | 0.030 ± 0.029 | 0.060 ± 0.001 | 2.967 ± 0.029 | 4.920 ± 0.023 |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | 0.174 ± 0.005 | 0.099 ± 0.030 | 0.167 ± 0.003 | 3.071 ± 0.089 | 7.349 ± 2.113 |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | 0.456 ± 0.121 | 0.216 ± 0.007 | 0.836 ± 0.160 | 1.938 ± 0.019 | 7.682 ± 3.146 |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | 1.125 ± 0.003 | 0.997 ± 0.001 | 1.584 ± 0.008 | 2.561 ± 0.010 | 8.472 ± 0.166 |
| large | `matmul` | f64 | 1 | `128x128` | 0.224 ± 0.007 | 0.186 ± 0.003 | 0.214 ± 0.004 | 10.781 ± 0.220 | 17.816 ± 0.151 |
| large | `matmul` | f64 | 1 | `256x256` | 1.532 ± 0.004 | 1.307 ± 0.025 | 1.942 ± 0.019 | 44.401 ± 0.692 | 68.850 ± 0.593 |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | 6.473 ± 0.052 | 5.713 ± 0.496 | 6.579 ± 0.352 | 181.318 ± 1.976 | 181.403 ± 5.278 |
| large | `qr` | f64 | 1 | `64x64` | 0.204 ± 0.013 | 0.199 ± 0.018 | 0.683 ± 0.009 | 1.702 ± 0.027 | 1.968 ± 0.300 |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | 0.279 ± 0.005 | 0.034 ± 0.012 | 0.671 ± 0.006 | 1.871 ± 0.089 | 5.431 ± 0.371 |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | 0.287 ± 0.007 | 0.041 ± 0.001 | 0.683 ± 0.077 | 2.151 ± 0.135 | 6.487 ± 0.105 |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | 0.319 ± 0.005 | 0.085 ± 0.003 | 0.723 ± 0.008 | 3.254 ± 0.015 | 8.851 ± 0.103 |
| large | `svd` | f64 | 1 | `64x64` | 0.874 ± 0.004 | 0.937 ± 0.029 | 1.478 ± 0.058 | 2.409 ± 0.012 | 5.547 ± 0.091 |
| small | `eigh` | f64 | 1 | `2x2` | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.056 ± 0.002 | 0.107 ± 0.005 | 0.577 ± 0.014 |
| small | `eigh` | f64 | 1 | `4x4` | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.080 ± 0.001 | 0.120 ± 0.004 | 0.717 ± 0.029 |
| small | `eigh` | f64 | 1 | `8x8` | 0.011 ± 0.000 | 0.010 ± 0.001 | 0.116 ± 0.001 | 0.147 ± 0.002 | 0.790 ± 0.021 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | 0.071 ± 0.008 | 0.026 ± 0.001 | 0.019 ± 0.000 | 0.125 ± 0.012 | 0.565 ± 0.042 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | 0.071 ± 0.010 | 0.028 ± 0.002 | 0.019 ± 0.000 | 0.132 ± 0.005 | 1.404 ± 0.024 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | 0.068 ± 0.004 | 0.023 ± 0.001 | 0.023 ± 0.000 | 0.172 ± 0.011 | 1.791 ± 0.029 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | 0.097 ± 0.009 | 0.011 ± 0.000 | 0.073 ± 0.001 | 0.255 ± 0.010 | 5.144 ± 0.608 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | 0.097 ± 0.005 | 0.010 ± 0.002 | 0.074 ± 0.002 | 0.258 ± 0.008 | 5.407 ± 0.893 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | 0.093 ± 0.001 | 0.010 ± 0.000 | 0.075 ± 0.001 | 0.297 ± 0.006 | 5.378 ± 0.907 |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | 0.118 ± 0.002 | 0.029 ± 0.002 | 0.188 ± 0.005 | 0.355 ± 0.009 | 5.829 ± 0.517 |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | 0.127 ± 0.005 | 0.026 ± 0.003 | 0.205 ± 0.003 | 0.359 ± 0.010 | 5.861 ± 0.629 |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | 0.125 ± 0.004 | 0.029 ± 0.001 | 0.270 ± 0.063 | 0.385 ± 0.012 | 5.943 ± 0.614 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | 0.163 ± 0.011 | 0.031 ± 0.002 | 0.142 ± 0.004 | 0.258 ± 0.007 | 5.530 ± 0.776 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | 0.170 ± 0.006 | 0.040 ± 0.008 | 0.162 ± 0.002 | 0.270 ± 0.008 | 5.616 ± 0.546 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | 0.184 ± 0.005 | 0.041 ± 0.001 | 0.207 ± 0.008 | 0.310 ± 0.007 | 5.696 ± 0.614 |
| small | `matmul` | f64 | 1 | `2x2` | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.091 ± 0.005 | 0.378 ± 0.054 |
| small | `matmul` | f64 | 1 | `4x4` | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.096 ± 0.001 | 0.745 ± 0.020 |
| small | `matmul` | f64 | 1 | `8x8` | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.132 ± 0.003 | 1.155 ± 0.009 |
| small | `qr` | f64 | 1 | `2x2` | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.034 ± 0.012 | 0.078 ± 0.003 | 0.359 ± 0.019 |
| small | `qr` | f64 | 1 | `4x4` | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.051 ± 0.000 | 0.084 ± 0.002 | 0.511 ± 0.031 |
| small | `qr` | f64 | 1 | `8x8` | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.085 ± 0.002 | 0.106 ± 0.002 | 0.670 ± 0.027 |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.075 ± 0.004 | 0.169 ± 0.011 | 0.787 ± 0.017 |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.076 ± 0.001 | 0.168 ± 0.004 | 0.668 ± 0.034 |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.095 ± 0.001 | 0.173 ± 0.012 | 0.803 ± 0.008 |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.094 ± 0.002 | 0.182 ± 0.027 | 0.788 ± 0.015 |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.126 ± 0.003 | 0.195 ± 0.013 | 0.845 ± 0.025 |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.128 ± 0.001 | 0.203 ± 0.010 | 0.812 ± 0.014 |
| small | `svd` | f64 | 1 | `2x2` | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.048 ± 0.001 | 0.104 ± 0.005 | 0.251 ± 0.027 |
| small | `svd` | f64 | 1 | `4x4` | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.073 ± 0.002 | 0.114 ± 0.002 | 0.556 ± 0.033 |
| small | `svd` | f64 | 1 | `8x8` | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.115 ± 0.002 | 0.147 ± 0.003 | 0.699 ± 0.026 |

## Threads: 4

- Timestamp: `20260531_031122`
- Source table: `data/results/cpu_ops_t4_20260531_031122.md`

Logs:

- `data/results/cpu_ops_t4_20260531_031122.csv`

## PR884 CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.026 ± 0.000 | 0.029 ± 0.000 | 0.683 ± 0.060 | 0.611 ± 0.107 | 1.557 ± 0.014 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.095 ± 0.002 | 0.097 ± 0.001 | 2.272 ± 0.081 | 1.994 ± 0.041 | 5.118 ± 0.060 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.065 ± 0.000 | 0.074 ± 0.004 | 1.040 ± 0.018 | 0.731 ± 0.014 | 2.325 ± 0.040 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.223 ± 0.005 | 0.256 ± 0.022 | 3.469 ± 0.020 | 2.510 ± 0.021 | 6.722 ± 0.102 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.023 ± 0.001 | 0.173 ± 0.014 | 0.941 ± 0.022 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.029 ± 0.001 | 0.330 ± 0.024 | 1.721 ± 0.023 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.029 ± 0.003 | 0.315 ± 0.024 | 1.967 ± 0.151 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.011 ± 0.000 | 0.008 ± 0.000 | 0.042 ± 0.012 | 0.821 ± 0.108 | 3.121 ± 0.185 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.033 ± 0.000 | 0.035 ± 0.000 | 0.471 ± 0.034 | 0.418 ± 0.025 | 0.961 ± 0.004 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.122 ± 0.006 | 0.090 ± 0.001 | 1.530 ± 0.032 | 1.267 ± 0.043 | 2.347 ± 0.057 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.042 ± 0.000 | 0.045 ± 0.000 | 0.847 ± 0.023 | 0.491 ± 0.036 | 1.420 ± 0.031 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.140 ± 0.014 | 0.160 ± 0.004 | 2.880 ± 0.053 | 1.694 ± 0.083 | 4.273 ± 0.063 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.021 ± 0.000 | 0.022 ± 0.000 | 0.774 ± 0.022 | 0.731 ± 0.038 | 1.878 ± 0.044 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.076 ± 0.000 | 0.072 ± 0.000 | 2.566 ± 0.046 | 2.342 ± 0.080 | 5.543 ± 0.228 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.027 ± 0.000 | 0.026 ± 0.000 | 1.120 ± 0.035 | 0.860 ± 0.027 | 2.149 ± 0.031 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.087 ± 0.020 | 0.082 ± 0.000 | 3.813 ± 0.137 | 2.751 ± 0.058 | 6.833 ± 0.236 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.051 ± 0.002 | 0.055 ± 0.000 | 0.448 ± 0.023 | 0.427 ± 0.091 | 0.809 ± 0.016 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.195 ± 0.011 | 0.195 ± 0.003 | 1.618 ± 0.079 | 1.273 ± 0.139 | 2.700 ± 0.073 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.102 ± 0.003 | 0.121 ± 0.001 | 0.889 ± 0.059 | 0.567 ± 0.023 | 1.523 ± 0.011 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.361 ± 0.007 | 0.433 ± 0.005 | 2.824 ± 0.015 | 1.860 ± 0.018 | 5.071 ± 0.074 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.123 ± 0.003 | 0.020 ± 0.001 | 0.209 ± 0.017 | 0.418 ± 0.047 | 5.983 ± 1.249 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.129 ± 0.014 | 0.036 ± 0.000 | 0.214 ± 0.030 | 0.599 ± 0.154 | 6.160 ± 1.669 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.125 ± 0.003 | 0.021 ± 0.002 | 0.223 ± 0.027 | 0.564 ± 0.016 | 6.077 ± 1.854 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.130 ± 0.023 | 0.040 ± 0.001 | 0.224 ± 0.016 | 1.178 ± 0.064 | 6.408 ± 2.798 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.361 ± 0.009 | 0.315 ± 0.006 | 0.952 ± 0.016 | 0.922 ± 0.019 | 6.739 ± 1.160 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.935 ± 0.029 | 1.046 ± 0.019 | 2.690 ± 0.030 | 2.509 ± 0.047 | 7.896 ± 2.929 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.365 ± 0.013 | 0.329 ± 0.006 | 1.298 ± 0.018 | 1.125 ± 0.066 | 7.039 ± 1.248 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 1.069 ± 0.008 | 1.092 ± 0.034 | 3.971 ± 0.057 | 3.015 ± 0.027 | 9.244 ± 2.174 |
| large | `eigh` | f64 | 4 | `64x64` | 1.125 ± 0.010 | 0.842 ± 0.040 | 1.387 ± 0.164 | 2.327 ± 0.076 | 6.801 ± 0.069 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.079 ± 0.006 | 0.065 ± 0.063 | 0.064 ± 0.001 | 3.399 ± 0.090 | 4.819 ± 0.083 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.298 ± 0.015 | 0.212 ± 0.065 | 0.193 ± 0.016 | 3.464 ± 0.072 | 7.352 ± 2.992 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.709 ± 0.033 | 0.516 ± 0.027 | 0.937 ± 0.055 | 1.958 ± 0.054 | 7.622 ± 2.790 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 1.719 ± 0.053 | 1.881 ± 0.061 | 1.939 ± 0.031 | 2.846 ± 0.051 | 8.463 ± 0.152 |
| large | `matmul` | f64 | 4 | `128x128` | 0.310 ± 0.037 | 0.229 ± 0.008 | 0.394 ± 0.033 | 11.180 ± 0.307 | 18.045 ± 0.251 |
| large | `matmul` | f64 | 4 | `256x256` | 1.622 ± 0.023 | 1.218 ± 0.013 | 1.657 ± 0.037 | 44.123 ± 0.324 | 68.839 ± 0.916 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 5.518 ± 0.243 | 4.439 ± 0.111 | 4.498 ± 0.087 | 178.108 ± 3.048 | 181.607 ± 3.283 |
| large | `qr` | f64 | 4 | `64x64` | 1.032 ± 0.008 | 1.122 ± 0.012 | 0.860 ± 0.033 | 1.995 ± 0.052 | 3.978 ± 0.215 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.429 ± 0.055 | 0.102 ± 0.025 | 0.756 ± 0.031 | 1.745 ± 0.029 | 5.659 ± 0.084 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.469 ± 0.048 | 0.112 ± 0.017 | 0.762 ± 0.029 | 2.359 ± 0.085 | 6.557 ± 0.097 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.588 ± 0.010 | 0.225 ± 0.017 | 0.780 ± 0.037 | 3.820 ± 0.176 | 8.887 ± 0.130 |
| large | `svd` | f64 | 4 | `64x64` | 1.234 ± 0.072 | 1.559 ± 0.029 | 1.815 ± 0.018 | 2.778 ± 0.072 | 5.744 ± 2.766 |
| small | `eigh` | f64 | 4 | `2x2` | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.063 ± 0.001 | 0.117 ± 0.007 | 0.345 ± 0.017 |
| small | `eigh` | f64 | 4 | `4x4` | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.098 ± 0.007 | 0.139 ± 0.033 | 0.718 ± 0.029 |
| small | `eigh` | f64 | 4 | `8x8` | 0.014 ± 0.000 | 0.010 ± 0.000 | 0.144 ± 0.014 | 0.158 ± 0.021 | 0.762 ± 0.018 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.081 ± 0.022 | 0.023 ± 0.012 | 0.019 ± 0.001 | 0.126 ± 0.025 | 0.537 ± 0.044 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.081 ± 0.004 | 0.024 ± 0.001 | 0.021 ± 0.012 | 0.136 ± 0.034 | 1.381 ± 0.021 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.084 ± 0.009 | 0.024 ± 0.001 | 0.026 ± 0.001 | 0.176 ± 0.059 | 1.759 ± 0.004 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.112 ± 0.019 | 0.010 ± 0.000 | 0.099 ± 0.023 | 0.294 ± 0.072 | 5.165 ± 0.683 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.112 ± 0.036 | 0.010 ± 0.001 | 0.085 ± 0.010 | 0.263 ± 0.051 | 5.313 ± 0.962 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.099 ± 0.022 | 0.010 ± 0.000 | 0.088 ± 0.010 | 0.296 ± 0.009 | 5.214 ± 0.909 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.166 ± 0.007 | 0.049 ± 0.006 | 0.216 ± 0.036 | 0.374 ± 0.029 | 5.739 ± 0.433 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.167 ± 0.003 | 0.049 ± 0.003 | 0.233 ± 0.017 | 0.402 ± 0.038 | 5.642 ± 0.562 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.176 ± 0.008 | 0.060 ± 0.012 | 0.291 ± 0.033 | 0.383 ± 0.010 | 5.717 ± 0.498 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.203 ± 0.006 | 0.030 ± 0.001 | 0.162 ± 0.030 | 0.256 ± 0.018 | 5.196 ± 0.487 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.182 ± 0.025 | 0.034 ± 0.001 | 0.197 ± 0.025 | 0.313 ± 0.039 | 5.542 ± 0.620 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.225 ± 0.028 | 0.047 ± 0.003 | 0.235 ± 0.014 | 0.307 ± 0.011 | 5.479 ± 0.597 |
| small | `matmul` | f64 | 4 | `2x2` | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.107 ± 0.023 | 0.308 ± 0.106 |
| small | `matmul` | f64 | 4 | `4x4` | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.097 ± 0.003 | 0.729 ± 0.008 |
| small | `matmul` | f64 | 4 | `8x8` | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.134 ± 0.027 | 1.120 ± 0.044 |
| small | `qr` | f64 | 4 | `2x2` | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.037 ± 0.001 | 0.101 ± 0.015 | 0.245 ± 0.023 |
| small | `qr` | f64 | 4 | `4x4` | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.064 ± 0.001 | 0.095 ± 0.007 | 0.554 ± 0.015 |
| small | `qr` | f64 | 4 | `8x8` | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.111 ± 0.001 | 0.153 ± 0.023 | 0.669 ± 0.019 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.085 ± 0.001 | 0.182 ± 0.022 | 0.492 ± 0.019 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.097 ± 0.013 | 0.170 ± 0.010 | 0.574 ± 0.015 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.115 ± 0.030 | 0.195 ± 0.032 | 0.770 ± 0.014 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.111 ± 0.012 | 0.193 ± 0.038 | 0.774 ± 0.025 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.007 ± 0.000 | 0.003 ± 0.000 | 0.146 ± 0.019 | 0.198 ± 0.035 | 0.871 ± 0.013 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.007 ± 0.000 | 0.004 ± 0.000 | 0.146 ± 0.014 | 0.208 ± 0.012 | 0.802 ± 0.015 |
| small | `svd` | f64 | 4 | `2x2` | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.056 ± 0.006 | 0.109 ± 0.003 | 0.245 ± 0.021 |
| small | `svd` | f64 | 4 | `4x4` | 0.010 ± 0.009 | 0.007 ± 0.000 | 0.086 ± 0.025 | 0.114 ± 0.003 | 0.556 ± 0.033 |
| small | `svd` | f64 | 4 | `8x8` | 0.019 ± 0.000 | 0.016 ± 0.000 | 0.133 ± 0.011 | 0.157 ± 0.015 | 0.654 ± 0.019 |

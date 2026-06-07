# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260607_080940`
- Source run metadata: `data/results/cpu/einsum/20260607_080940/run.yaml`
- tenferro-rs commit: `801d4345213800eceb25d0000e4b562344e83682`
- tenferro-rs dirty: not recorded in this pre-target-profile raw run
- tenferro-rs features: `system-accelerate`
- tenferro BLAS: `accelerate`
- PyTorch: BLAS provider `accelerate`, version `2.12.0`
- JAX: dot backend `xla_cpu`, version `0.10.1`

Latest run: `./scripts/run_all.sh 1`.

This latest report was regenerated from the existing macOS CPU raw run after the target-profile layout cleanup. New runs write raw data under `data/results/mac-cpu/cpu/einsum/<timestamp>/`.

## Threads: 1

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.078 ± 0.003 | 0.101 ± 0.005 |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.028 ± 0.000 | 0.020 ± 0.000 | 0.279 ± 0.014 | 0.320 ± 0.003 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.022 ± 0.000 | 0.016 ± 0.000 | 0.111 ± 0.005 | 0.130 ± 0.003 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.076 ± 0.001 | 0.055 ± 0.000 | 0.426 ± 0.030 | 0.454 ± 0.011 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.026 ± 0.001 | 0.101 ± 0.006 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.064 ± 0.002 | 0.154 ± 0.009 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.063 ± 0.004 | 0.148 ± 0.004 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.223 ± 0.009 | 0.322 ± 0.017 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.010 ± 0.000 | 0.007 ± 0.000 | 0.054 ± 0.005 | 0.077 ± 0.010 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.033 ± 0.000 | 0.025 ± 0.000 | 0.182 ± 0.002 | 0.207 ± 0.009 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.014 ± 0.000 | 0.011 ± 0.000 | 0.076 ± 0.001 | 0.099 ± 0.004 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.047 ± 0.000 | 0.037 ± 0.000 | 0.282 ± 0.002 | 0.330 ± 0.006 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.016 ± 0.001 | 0.011 ± 0.000 | 0.082 ± 0.002 | 0.115 ± 0.004 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.048 ± 0.000 | 0.035 ± 0.000 | 0.291 ± 0.002 | 0.343 ± 0.011 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.017 ± 0.000 | 0.012 ± 0.000 | 0.118 ± 0.013 | 0.141 ± 0.011 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.053 ± 0.000 | 0.040 ± 0.004 | 0.405 ± 0.005 | 0.467 ± 0.011 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.014 ± 0.000 | 0.010 ± 0.000 | 0.058 ± 0.005 | 0.083 ± 0.008 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.046 ± 0.001 | 0.035 ± 0.000 | 0.199 ± 0.003 | 0.268 ± 0.010 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.035 ± 0.000 | 0.028 ± 0.000 | 0.097 ± 0.004 | 0.120 ± 0.004 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.127 ± 0.000 | 0.103 ± 0.000 | 0.354 ± 0.002 | 0.441 ± 0.011 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.031 ± 0.000 | 0.005 ± 0.000 | 0.051 ± 0.008 | 0.451 ± 0.024 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.033 ± 0.000 | 0.010 ± 0.000 | 0.091 ± 0.003 | 0.524 ± 0.046 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.031 ± 0.001 | 0.006 ± 0.000 | 0.092 ± 0.002 | 0.506 ± 0.028 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.036 ± 0.000 | 0.014 ± 0.001 | 0.251 ± 0.005 | 0.689 ± 0.049 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.060 ± 0.002 | 0.043 ± 0.001 | 0.097 ± 0.007 | 0.455 ± 0.019 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.111 ± 0.018 | 0.129 ± 0.002 | 0.302 ± 0.001 | 0.673 ± 0.025 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.061 ± 0.000 | 0.045 ± 0.000 | 0.134 ± 0.002 | 0.503 ± 0.022 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.119 ± 0.001 | 0.135 ± 0.003 | 0.428 ± 0.029 | 0.791 ± 0.068 |
| large | `eigh` | f64 | 1 | `64x64` | 0.429 ± 0.018 | 0.275 ± 0.002 | 0.636 ± 0.017 | 0.705 ± 0.017 |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | 0.015 ± 0.000 | 0.006 ± 0.000 | 0.850 ± 0.018 | 0.931 ± 0.020 |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | 0.046 ± 0.003 | 0.018 ± 0.004 | 0.851 ± 0.012 | 1.307 ± 0.119 |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | 0.123 ± 0.001 | 0.038 ± 0.000 | 0.494 ± 0.022 | 0.859 ± 0.036 |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | 0.702 ± 0.022 | 0.559 ± 0.018 | 0.756 ± 0.044 | 0.971 ± 0.107 |
| large | `matmul` | f64 | 1 | `128x128` | 0.035 ± 0.000 | 0.017 ± 0.001 | 3.292 ± 0.053 | 3.619 ± 0.046 |
| large | `matmul` | f64 | 1 | `256x256` | 0.196 ± 0.001 | 0.130 ± 0.009 | 13.094 ± 0.097 | 14.074 ± 0.313 |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | 0.648 ± 0.005 | 0.520 ± 0.018 | 54.180 ± 0.726 | 55.695 ± 0.817 |
| large | `qr` | f64 | 1 | `64x64` | 0.104 ± 0.003 | 0.069 ± 0.002 | 0.504 ± 0.008 | 0.559 ± 0.014 |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | 0.088 ± 0.000 | 0.015 ± 0.002 | 0.466 ± 0.006 | 0.537 ± 0.014 |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | 0.090 ± 0.003 | 0.020 ± 0.001 | 0.568 ± 0.009 | 0.648 ± 0.026 |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | 0.103 ± 0.004 | 0.031 ± 0.005 | 0.862 ± 0.007 | 0.969 ± 0.025 |
| large | `svd` | f64 | 1 | `64x64` | 0.772 ± 0.013 | 0.562 ± 0.014 | 0.704 ± 0.006 | 0.773 ± 0.032 |
| small | `eigh` | f64 | 1 | `2x2` | 0.004 ± 0.001 | 0.001 ± 0.000 | 0.008 ± 0.001 | 0.027 ± 0.007 |
| small | `eigh` | f64 | 1 | `4x4` | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.000 | 0.028 ± 0.005 |
| small | `eigh` | f64 | 1 | `8x8` | 0.007 ± 0.000 | 0.003 ± 0.000 | 0.018 ± 0.000 | 0.037 ± 0.005 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | 0.006 ± 0.001 | 0.001 ± 0.000 | 0.012 ± 0.001 | 0.085 ± 0.007 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | 0.005 ± 0.000 | 0.001 ± 0.000 | 0.014 ± 0.000 | 0.090 ± 0.006 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.024 ± 0.001 | 0.116 ± 0.008 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | 0.051 ± 0.007 | 0.004 ± 0.000 | 0.021 ± 0.003 | 0.338 ± 0.003 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | 0.047 ± 0.000 | 0.004 ± 0.000 | 0.022 ± 0.000 | 0.358 ± 0.015 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | 0.046 ± 0.000 | 0.006 ± 0.000 | 0.038 ± 0.001 | 0.372 ± 0.016 |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | 0.067 ± 0.002 | 0.011 ± 0.000 | 0.030 ± 0.003 | 0.389 ± 0.012 |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | 0.058 ± 0.001 | 0.011 ± 0.000 | 0.031 ± 0.002 | 0.383 ± 0.009 |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | 0.058 ± 0.001 | 0.012 ± 0.000 | 0.037 ± 0.001 | 0.392 ± 0.006 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | 0.078 ± 0.003 | 0.008 ± 0.000 | 0.019 ± 0.002 | 0.381 ± 0.019 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | 0.077 ± 0.002 | 0.009 ± 0.000 | 0.021 ± 0.000 | 0.389 ± 0.020 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | 0.080 ± 0.001 | 0.017 ± 0.002 | 0.032 ± 0.000 | 0.399 ± 0.012 |
| small | `matmul` | f64 | 1 | `2x2` | 0.007 ± 0.001 | 0.001 ± 0.000 | 0.008 ± 0.001 | 0.039 ± 0.005 |
| small | `matmul` | f64 | 1 | `4x4` | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.010 ± 0.000 | 0.037 ± 0.001 |
| small | `matmul` | f64 | 1 | `8x8` | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.019 ± 0.002 | 0.066 ± 0.005 |
| small | `qr` | f64 | 1 | `2x2` | 0.004 ± 0.002 | 0.001 ± 0.000 | 0.008 ± 0.002 | 0.025 ± 0.005 |
| small | `qr` | f64 | 1 | `4x4` | 0.004 ± 0.001 | 0.001 ± 0.000 | 0.007 ± 0.000 | 0.025 ± 0.005 |
| small | `qr` | f64 | 1 | `8x8` | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.013 ± 0.000 | 0.035 ± 0.004 |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | 0.007 ± 0.002 | 0.002 ± 0.000 | 0.014 ± 0.001 | 0.042 ± 0.006 |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | 0.007 ± 0.000 | 0.002 ± 0.000 | 0.014 ± 0.000 | 0.042 ± 0.004 |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | 0.007 ± 0.000 | 0.002 ± 0.000 | 0.015 ± 0.000 | 0.044 ± 0.010 |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | 0.007 ± 0.000 | 0.002 ± 0.000 | 0.017 ± 0.000 | 0.042 ± 0.003 |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | 0.007 ± 0.000 | 0.002 ± 0.000 | 0.022 ± 0.000 | 0.050 ± 0.002 |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | 0.007 ± 0.000 | 0.003 ± 0.000 | 0.024 ± 0.000 | 0.050 ± 0.007 |
| small | `svd` | f64 | 1 | `2x2` | 0.007 ± 0.002 | 0.001 ± 0.000 | 0.008 ± 0.001 | 0.028 ± 0.007 |
| small | `svd` | f64 | 1 | `4x4` | 0.007 ± 0.000 | 0.003 ± 0.000 | 0.010 ± 0.000 | 0.029 ± 0.006 |
| small | `svd` | f64 | 1 | `8x8` | 0.012 ± 0.000 | 0.006 ± 0.000 | 0.019 ± 0.000 | 0.039 ± 0.006 |

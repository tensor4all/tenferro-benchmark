# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260614_135639`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260614_135639`.

- tenferro-rs commit: `db1990549801351308b631aef1bbca292d11a457`

## CPU Information

- Model: `Apple M4`
- Vendor: `Apple`
- Logical CPUs: `10`
- Physical CPUs: `10`
- Sockets: `1`
- Cores per socket: `10`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Performance: 4 physical / 4 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 4 CPUs/L2); Efficiency: 6 physical / 6 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260614_135639/cpu_ops_t4_20260614_135639.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260614_135639/cpu_ops_t4_20260614_135639.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.038 ± 0.005 | 0.006 ± 0.000 | 0.094 ± 0.001 | 0.139 ± 0.032 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.057 ± 0.003 | 0.021 ± 0.000 | 0.345 ± 0.011 | 0.378 ± 0.015 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.052 ± 0.003 | 0.017 ± 0.000 | 0.137 ± 0.001 | 0.156 ± 0.006 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.105 ± 0.007 | 0.063 ± 0.000 | 0.499 ± 0.031 | 0.543 ± 0.024 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.043 ± 0.011 | 0.007 ± 0.003 | 0.032 ± 0.000 | 0.137 ± 0.014 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.037 ± 0.006 | 0.006 ± 0.001 | 0.089 ± 0.014 | 0.193 ± 0.013 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.038 ± 0.005 | 0.008 ± 0.003 | 0.086 ± 0.002 | 0.195 ± 0.003 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.037 ± 0.006 | 0.008 ± 0.001 | 0.286 ± 0.005 | 0.412 ± 0.014 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.039 ± 0.001 | 0.008 ± 0.000 | 0.079 ± 0.008 | 0.089 ± 0.006 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.060 ± 0.004 | 0.028 ± 0.000 | 0.238 ± 0.007 | 0.243 ± 0.002 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.043 ± 0.004 | 0.011 ± 0.000 | 0.108 ± 0.005 | 0.119 ± 0.006 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.073 ± 0.008 | 0.040 ± 0.000 | 0.371 ± 0.006 | 0.392 ± 0.004 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.042 ± 0.007 | 0.010 ± 0.000 | 0.118 ± 0.011 | 0.143 ± 0.009 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.068 ± 0.008 | 0.031 ± 0.000 | 0.375 ± 0.014 | 0.395 ± 0.006 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.051 ± 0.010 | 0.011 ± 0.000 | 0.155 ± 0.006 | 0.169 ± 0.007 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.075 ± 0.008 | 0.035 ± 0.000 | 0.502 ± 0.005 | 0.578 ± 0.061 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.041 ± 0.002 | 0.012 ± 0.000 | 0.070 ± 0.001 | 0.106 ± 0.005 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.077 ± 0.002 | 0.042 ± 0.000 | 0.248 ± 0.003 | 0.324 ± 0.007 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.063 ± 0.003 | 0.032 ± 0.000 | 0.123 ± 0.001 | 0.175 ± 0.024 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.156 ± 0.003 | 0.118 ± 0.000 | 0.442 ± 0.014 | 0.527 ± 0.015 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.109 ± 0.058 | 0.012 ± 0.003 | 0.065 ± 0.002 | 0.552 ± 0.036 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.112 ± 0.018 | 0.021 ± 0.001 | 0.119 ± 0.002 | 0.610 ± 0.037 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.109 ± 0.002 | 0.017 ± 0.016 | 0.116 ± 0.004 | 0.608 ± 0.016 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.115 ± 0.005 | 0.024 ± 0.001 | 0.314 ± 0.005 | 0.830 ± 0.033 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.151 ± 0.036 | 0.126 ± 0.032 | 0.153 ± 0.030 | 0.604 ± 0.133 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.190 ± 0.002 | 0.190 ± 0.017 | 0.410 ± 0.017 | 0.788 ± 0.020 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.142 ± 0.057 | 0.147 ± 0.036 | 0.184 ± 0.014 | 0.607 ± 0.054 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.198 ± 0.006 | 0.203 ± 0.017 | 0.529 ± 0.008 | 0.950 ± 0.018 |
| large | `eigh` | f64 | 4 | `64x64` | 0.426 ± 0.014 | 0.310 ± 0.003 | 0.763 ± 0.009 | 0.833 ± 0.028 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.057 ± 0.003 | 0.011 ± 0.003 | 1.159 ± 0.099 | 2.294 ± 1.282 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.135 ± 0.007 | 0.025 ± 0.001 | 1.220 ± 0.014 | 1.532 ± 0.212 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.219 ± 0.029 | 0.194 ± 0.129 | 0.665 ± 0.010 | 1.024 ± 0.029 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | panic | 0.641 ± 0.006 | 0.912 ± 0.026 | 1.081 ± 0.208 |
| large | `matmul` | f64 | 4 | `128x128` | 0.084 ± 0.004 | 0.023 ± 0.000 | 4.500 ± 0.621 | 4.424 ± 0.122 |
| large | `matmul` | f64 | 4 | `256x256` | 0.202 ± 0.001 | 0.123 ± 0.003 | 17.715 ± 0.956 | 19.463 ± 0.571 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 0.642 ± 0.072 | 0.515 ± 0.011 | 75.630 ± 3.684 | 74.653 ± 3.511 |
| large | `qr` | f64 | 4 | `64x64` | 0.113 ± 0.007 | 0.083 ± 0.001 | 0.694 ± 0.040 | 0.668 ± 0.035 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.117 ± 0.004 | 0.019 ± 0.003 | 0.646 ± 0.077 | 0.654 ± 0.130 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.118 ± 0.005 | 0.020 ± 0.001 | 0.772 ± 0.014 | 0.766 ± 0.017 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.129 ± 0.006 | 0.030 ± 0.002 | 1.234 ± 0.095 | 1.202 ± 0.438 |
| large | `svd` | f64 | 4 | `64x64` | 0.708 ± 0.049 | 0.605 ± 0.030 | 0.899 ± 0.011 | 0.907 ± 0.042 |
| small | `eigh` | f64 | 4 | `2x2` | 0.058 ± 0.010 | 0.001 ± 0.000 | 0.012 ± 0.001 | 0.034 ± 0.006 |
| small | `eigh` | f64 | 4 | `4x4` | 0.054 ± 0.008 | 0.002 ± 0.000 | 0.014 ± 0.000 | 0.041 ± 0.010 |
| small | `eigh` | f64 | 4 | `8x8` | 0.045 ± 0.003 | 0.003 ± 0.000 | 0.023 ± 0.001 | 0.048 ± 0.008 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.071 ± 0.010 | 0.006 ± 0.002 | 0.016 ± 0.002 | 0.121 ± 0.006 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.061 ± 0.005 | 0.005 ± 0.001 | 0.018 ± 0.000 | 0.120 ± 0.005 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.052 ± 0.003 | 0.005 ± 0.001 | 0.032 ± 0.001 | 0.143 ± 0.011 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.238 ± 0.036 | 0.016 ± 0.024 | 0.028 ± 0.006 | 0.423 ± 0.017 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.179 ± 0.009 | 0.010 ± 0.001 | 0.030 ± 0.001 | 0.483 ± 0.153 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.144 ± 0.006 | 0.013 ± 0.001 | 0.048 ± 0.001 | 0.443 ± 0.012 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.264 ± 0.051 | 0.053 ± 0.011 | 0.038 ± 0.002 | 0.494 ± 0.083 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.194 ± 0.059 | 0.063 ± 0.009 | 0.041 ± 0.001 | 0.475 ± 0.074 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.178 ± 0.020 | 0.061 ± 0.030 | 0.048 ± 0.001 | 0.467 ± 0.018 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | panic | 0.035 ± 0.005 | 0.026 ± 0.001 | 0.465 ± 0.022 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | panic | 0.031 ± 0.004 | 0.029 ± 0.001 | 0.478 ± 0.024 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | panic | 0.033 ± 0.003 | 0.040 ± 0.000 | 0.477 ± 0.034 |
| small | `matmul` | f64 | 4 | `2x2` | 0.083 ± 0.029 | 0.006 ± 0.001 | 0.011 ± 0.002 | 0.050 ± 0.008 |
| small | `matmul` | f64 | 4 | `4x4` | 0.070 ± 0.003 | 0.004 ± 0.001 | 0.013 ± 0.000 | 0.049 ± 0.005 |
| small | `matmul` | f64 | 4 | `8x8` | 0.053 ± 0.003 | 0.006 ± 0.002 | 0.025 ± 0.000 | 0.075 ± 0.006 |
| small | `qr` | f64 | 4 | `2x2` | 0.052 ± 0.012 | 0.001 ± 0.000 | 0.018 ± 0.003 | 0.034 ± 0.005 |
| small | `qr` | f64 | 4 | `4x4` | 0.047 ± 0.001 | 0.001 ± 0.000 | 0.019 ± 0.003 | 0.035 ± 0.007 |
| small | `qr` | f64 | 4 | `8x8` | 0.042 ± 0.008 | 0.002 ± 0.000 | 0.026 ± 0.002 | 0.044 ± 0.008 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.083 ± 0.007 | 0.003 ± 0.000 | 0.020 ± 0.003 | 0.052 ± 0.009 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.085 ± 0.036 | 0.002 ± 0.000 | 0.019 ± 0.000 | 0.059 ± 0.007 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.051 ± 0.003 | 0.002 ± 0.000 | 0.021 ± 0.000 | 0.056 ± 0.006 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.055 ± 0.014 | 0.002 ± 0.000 | 0.022 ± 0.000 | 0.056 ± 0.006 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.046 ± 0.003 | 0.003 ± 0.000 | 0.029 ± 0.000 | 0.066 ± 0.015 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.049 ± 0.001 | 0.003 ± 0.000 | 0.032 ± 0.000 | 0.065 ± 0.008 |
| small | `svd` | f64 | 4 | `2x2` | 0.056 ± 0.004 | 0.001 ± 0.000 | 0.011 ± 0.002 | 0.036 ± 0.008 |
| small | `svd` | f64 | 4 | `4x4` | 0.054 ± 0.008 | 0.003 ± 0.000 | 0.013 ± 0.000 | 0.040 ± 0.011 |
| small | `svd` | f64 | 4 | `8x8` | 0.051 ± 0.006 | 0.006 ± 0.000 | 0.024 ± 0.000 | 0.049 ± 0.007 |

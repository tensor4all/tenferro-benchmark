# Linux CPU Ops Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `linux-cpu`
- Thread runs: `1 4`

Mirrored from `amd-cpu` runs (`BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh <THREADS>`);
regenerate with `./scripts/mirror_linux_cpu_cpu_reports.sh 1:20260728_013419 4:20260728_015909 permutation:20260728_021651`.

This report is generated from sequential CPU runs. Do not compare it with
measurements collected while another CPU benchmark process was running.

## Run Inputs

- Threads 1: timestamp `20260728_013419`, raw run `data/results/linux-cpu/cpu/einsum/20260728_013419`
- Threads 4: timestamp `20260728_015909`, raw run `data/results/linux-cpu/cpu/einsum/20260728_015909`

## Threads: 1

### CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `linux-cpu`
- Timestamp: `20260728_013419`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/linux-cpu/cpu/einsum/20260728_013419`.

- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

#### CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

#### Thread Environment

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

#### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 1

- CSV: `data/results/linux-cpu/cpu/einsum/20260728_013419/cpu_ops_t1_20260728_013419.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260728_013419/cpu_ops_t1_20260728_013419.md`

#### CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.319 ± 0.015 | 0.053 ± 0.003 | 0.320 ± 0.015 | 0.925 ± 0.030 |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.298 ± 0.011 | 0.127 ± 0.019 | 1.058 ± 0.013 | 2.527 ± 0.052 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.273 ± 0.006 | 0.078 ± 0.002 | 0.418 ± 0.012 | 1.121 ± 0.021 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.370 ± 0.006 | 0.226 ± 0.002 | 1.437 ± 0.007 | 3.279 ± 0.033 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.220 ± 0.004 | 0.026 ± 0.000 | 0.099 ± 0.005 | 0.965 ± 0.033 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.178 ± 0.011 | 0.032 ± 0.003 | 0.201 ± 0.006 | 1.241 ± 0.062 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.220 ± 0.016 | 0.027 ± 0.000 | 0.198 ± 0.006 | 1.250 ± 0.038 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.189 ± 0.009 | 0.034 ± 0.001 | 0.623 ± 0.012 | 1.870 ± 0.046 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.264 ± 0.009 | 0.046 ± 0.002 | 0.194 ± 0.004 | 0.667 ± 0.022 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.254 ± 0.014 | 0.093 ± 0.002 | 0.650 ± 0.010 | 1.433 ± 0.022 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.256 ± 0.011 | 0.052 ± 0.001 | 0.267 ± 0.014 | 0.842 ± 0.009 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.286 ± 0.012 | 0.117 ± 0.010 | 0.920 ± 0.010 | 2.298 ± 0.015 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.381 ± 0.005 | 0.114 ± 0.009 | 0.350 ± 0.014 | 0.548 ± 0.016 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.369 ± 0.015 | 0.173 ± 0.011 | 1.083 ± 0.011 | 2.762 ± 1.386 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.358 ± 0.008 | 0.118 ± 0.002 | 0.433 ± 0.011 | 0.645 ± 0.041 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.441 ± 0.057 | 0.204 ± 0.009 | 1.413 ± 0.006 | 1.691 ± 0.024 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.289 ± 0.016 | 0.069 ± 0.006 | 0.231 ± 0.006 | 0.724 ± 0.032 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.328 ± 0.007 | 0.175 ± 0.003 | 0.715 ± 0.011 | 1.555 ± 0.026 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.325 ± 0.011 | 0.125 ± 0.007 | 0.353 ± 0.005 | 0.860 ± 0.029 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.522 ± 0.004 | 0.396 ± 0.005 | 1.164 ± 0.009 | 2.490 ± 0.030 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.623 ± 0.012 | 0.092 ± 0.004 | 0.254 ± 0.013 | 3.832 ± 2.102 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.622 ± 0.024 | 0.105 ± 0.002 | 0.360 ± 0.007 | 3.720 ± 2.191 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.598 ± 0.056 | 0.092 ± 0.004 | 0.364 ± 0.010 | 3.681 ± 2.138 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.548 ± 0.006 | 0.115 ± 0.003 | 0.789 ± 0.004 | 2.491 ± 1.544 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 1.088 ± 0.008 | 0.368 ± 0.006 | 0.486 ± 0.015 | 4.056 ± 0.173 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 1.278 ± 0.112 | 0.498 ± 0.008 | 1.242 ± 0.012 | 5.004 ± 2.925 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 1.077 ± 0.006 | 0.389 ± 0.010 | 0.583 ± 0.008 | 4.233 ± 0.182 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 1.349 ± 0.023 | 0.556 ± 0.018 | 1.566 ± 0.017 | 4.557 ± 3.273 |
| large | `eigh` | f64 | 1 | `64x64` | 0.751 ± 0.049 | 0.289 ± 0.006 | 1.504 ± 0.014 | 3.190 ± 0.074 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 9.064 ± 0.040 | 10.574 ± 0.027 | 17.013 ± 0.539 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 55.894 ± 2.239 | 74.686 ± 0.140 | 64.454 ± 1.766 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 8.763 ± 0.074 | 8.698 ± 0.010 | 16.673 ± 1.240 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 63.834 ± 0.594 | 52.969 ± 0.092 | 66.897 ± 5.224 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 6.668 ± 0.178 | 12.679 ± 0.011 | 6.192 ± 0.547 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 44.815 ± 0.384 | 86.364 ± 0.191 | 33.107 ± 4.292 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 6.377 ± 0.481 | 5.582 ± 0.008 | 6.278 ± 0.470 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 43.135 ± 0.334 | 47.325 ± 0.290 | 36.848 ± 8.133 |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | 0.224 ± 0.002 | 0.065 ± 0.001 | 2.217 ± 0.006 | 2.526 ± 0.089 |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | 0.627 ± 0.056 | 0.156 ± 0.001 | 2.306 ± 0.017 | 3.823 ± 0.137 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 8.529 ± 1.176 | 7.710 ± 0.023 | 7.877 ± 3.073 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 54.923 ± 2.992 | 61.072 ± 0.196 | 46.338 ± 1.181 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 7.916 ± 0.623 | 7.535 ± 0.014 | 11.114 ± 2.604 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 48.624 ± 0.539 | 60.115 ± 0.158 | 70.849 ± 12.535 |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | 1.642 ± 0.046 | 0.535 ± 0.019 | 1.378 ± 0.004 | 4.640 ± 2.968 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.422 ± 0.007 | 1.464 ± 0.006 | 2.130 ± 0.205 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 6.125 ± 0.039 | 6.015 ± 0.007 | 9.472 ± 0.841 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 1.378 ± 0.012 | 1.937 ± 0.019 | 2.644 ± 0.088 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 6.061 ± 0.024 | 9.949 ± 0.036 | 10.133 ± 0.290 |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | 2.066 ± 0.091 | 0.766 ± 0.125 | 1.928 ± 0.007 | 4.402 ± 2.791 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 14.250 ± 0.366 | 19.466 ± 0.081 | 30.285 ± 12.269 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 91.832 ± 1.144 | 141.249 ± 0.712 | 120.480 ± 3.988 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 13.945 ± 0.237 | 16.504 ± 0.018 | 31.983 ± 13.253 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 90.469 ± 3.817 | 115.386 ± 0.356 | 121.976 ± 16.647 |
| large | `matmul` | f64 | 1 | `128x128` | 0.455 ± 0.043 | 0.167 ± 0.040 | 8.498 ± 0.020 | 9.193 ± 0.352 |
| large | `matmul` | f64 | 1 | `256x256` | 2.301 ± 0.292 | 0.831 ± 0.013 | 34.144 ± 0.066 | 37.183 ± 0.680 |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | 6.614 ± 0.132 | 3.246 ± 0.575 | 144.529 ± 3.588 | 150.554 ± 7.163 |
| large | `qr` | f64 | 1 | `64x64` | 0.327 ± 0.070 | 0.114 ± 0.004 | 1.255 ± 0.008 | 2.735 ± 0.046 |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | 0.520 ± 0.017 | 0.124 ± 0.000 | 1.261 ± 0.013 | 1.500 ± 0.025 |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | 0.546 ± 0.006 | 0.158 ± 0.006 | 1.534 ± 0.010 | 1.835 ± 0.010 |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | 0.611 ± 0.007 | 0.218 ± 0.006 | 2.369 ± 0.006 | 5.374 ± 1.607 |
| large | `svd` | f64 | 1 | `64x64` | 0.800 ± 0.038 | 0.546 ± 0.009 | 1.809 ± 0.008 | 2.049 ± 0.042 |
| small | `eigh` | f64 | 1 | `2x2` | 0.220 ± 0.006 | 0.027 ± 0.001 | 0.062 ± 0.002 | 0.369 ± 0.073 |
| small | `eigh` | f64 | 1 | `4x4` | 0.216 ± 0.015 | 0.029 ± 0.001 | 0.074 ± 0.006 | 0.395 ± 0.014 |
| small | `eigh` | f64 | 1 | `8x8` | 0.216 ± 0.045 | 0.032 ± 0.000 | 0.090 ± 0.002 | 0.440 ± 0.028 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | 0.217 ± 0.006 | 0.023 ± 0.000 | 0.063 ± 0.002 | 0.437 ± 0.024 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | 0.195 ± 0.010 | 0.023 ± 0.001 | 0.071 ± 0.002 | 0.437 ± 0.014 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | 0.199 ± 0.010 | 0.026 ± 0.001 | 0.100 ± 0.002 | 0.878 ± 0.027 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.181 ± 0.001 | 0.227 ± 0.011 | 0.579 ± 0.014 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.184 ± 0.003 | 0.225 ± 0.005 | 0.584 ± 0.015 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.188 ± 0.003 | 0.234 ± 0.005 | 0.589 ± 0.021 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.183 ± 0.004 | 0.225 ± 0.008 | 1.318 ± 0.010 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.187 ± 0.004 | 0.228 ± 0.012 | 1.326 ± 0.010 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.188 ± 0.004 | 0.233 ± 0.007 | 2.279 ± 0.752 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.180 ± 0.006 | 0.308 ± 0.011 | 0.829 ± 0.025 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.179 ± 0.002 | 0.305 ± 0.008 | 0.834 ± 0.016 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.201 ± 0.004 | 0.308 ± 0.008 | 0.816 ± 0.030 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.192 ± 0.002 | 0.296 ± 0.016 | 2.103 ± 0.023 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.216 ± 0.003 | 0.287 ± 0.012 | 2.090 ± 0.019 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.192 ± 0.027 | 0.289 ± 0.013 | 2.085 ± 0.009 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | 0.584 ± 0.065 | 0.077 ± 0.001 | 0.143 ± 0.009 | 3.063 ± 1.381 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | 0.577 ± 0.003 | 0.077 ± 0.001 | 0.151 ± 0.005 | 3.172 ± 0.398 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | 0.566 ± 0.078 | 0.077 ± 0.001 | 0.180 ± 0.008 | 3.151 ± 1.115 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.323 ± 0.007 | 0.246 ± 0.014 | 1.545 ± 0.252 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.323 ± 0.008 | 0.250 ± 0.013 | 1.700 ± 0.040 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.333 ± 0.004 | 0.250 ± 0.007 | 1.743 ± 0.043 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.329 ± 0.002 | 0.273 ± 0.023 | 1.950 ± 0.029 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.339 ± 0.042 | 0.273 ± 0.019 | 4.057 ± 0.531 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.336 ± 0.010 | 0.280 ± 0.010 | 4.161 ± 0.274 |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | 1.123 ± 0.058 | 0.305 ± 0.032 | 0.219 ± 0.005 | 3.595 ± 0.037 |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | 1.001 ± 0.130 | 0.331 ± 0.026 | 0.225 ± 0.009 | 3.436 ± 0.012 |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | 1.062 ± 0.114 | 0.301 ± 0.020 | 0.244 ± 0.006 | 2.709 ± 0.903 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.198 ± 0.003 | 0.496 ± 0.017 | 0.689 ± 0.025 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.225 ± 0.009 | 0.486 ± 0.013 | 0.695 ± 0.017 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.227 ± 0.009 | 0.498 ± 0.009 | 0.705 ± 0.036 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.219 ± 0.004 | 0.316 ± 0.011 | 2.653 ± 0.027 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.220 ± 0.028 | 0.318 ± 0.015 | 2.668 ± 0.026 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.225 ± 0.001 | 0.319 ± 0.015 | 1.440 ± 0.025 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | 0.613 ± 0.026 | 0.107 ± 0.001 | 0.167 ± 0.003 | 1.582 ± 1.570 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | 0.629 ± 0.024 | 0.125 ± 0.008 | 0.180 ± 0.008 | 3.250 ± 0.028 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | 0.565 ± 0.004 | 0.120 ± 0.002 | 0.202 ± 0.012 | 3.385 ± 0.395 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.118 ± 0.002 | 0.301 ± 0.009 | 1.102 ± 0.043 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.124 ± 0.003 | 0.301 ± 0.007 | 1.113 ± 0.029 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.135 ± 0.011 | 0.310 ± 0.010 | 1.075 ± 0.038 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.106 ± 0.029 | 0.261 ± 0.020 | 2.944 ± 0.023 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.125 ± 0.004 | 0.248 ± 0.010 | 2.988 ± 0.039 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.120 ± 0.002 | 0.265 ± 0.013 | 3.057 ± 0.023 |
| small | `matmul` | f64 | 1 | `2x2` | 0.215 ± 0.007 | 0.021 ± 0.000 | 0.039 ± 0.005 | 0.247 ± 0.016 |
| small | `matmul` | f64 | 1 | `4x4` | 0.192 ± 0.005 | 0.021 ± 0.000 | 0.046 ± 0.001 | 0.249 ± 0.052 |
| small | `matmul` | f64 | 1 | `8x8` | 0.214 ± 0.014 | 0.026 ± 0.007 | 0.072 ± 0.002 | 0.516 ± 0.024 |
| small | `qr` | f64 | 1 | `2x2` | 0.229 ± 0.011 | 0.027 ± 0.001 | 0.040 ± 0.001 | 0.163 ± 0.009 |
| small | `qr` | f64 | 1 | `4x4` | 0.230 ± 0.031 | 0.028 ± 0.000 | 0.045 ± 0.001 | 0.308 ± 0.013 |
| small | `qr` | f64 | 1 | `8x8` | 0.229 ± 0.009 | 0.032 ± 0.002 | 0.060 ± 0.001 | 0.338 ± 0.013 |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | 0.346 ± 0.006 | 0.087 ± 0.004 | 0.095 ± 0.006 | 0.294 ± 0.004 |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | 0.338 ± 0.007 | 0.085 ± 0.007 | 0.096 ± 0.002 | 0.278 ± 0.013 |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | 0.300 ± 0.007 | 0.088 ± 0.001 | 0.104 ± 0.007 | 0.287 ± 0.020 |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | 0.333 ± 0.045 | 0.088 ± 0.001 | 0.103 ± 0.001 | 0.286 ± 0.017 |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | 0.292 ± 0.005 | 0.088 ± 0.002 | 0.117 ± 0.001 | 0.303 ± 0.012 |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | 0.294 ± 0.002 | 0.088 ± 0.002 | 0.126 ± 0.004 | 0.312 ± 0.020 |
| small | `svd` | f64 | 1 | `2x2` | 0.231 ± 0.013 | 0.029 ± 0.001 | 0.058 ± 0.005 | 0.172 ± 0.007 |
| small | `svd` | f64 | 1 | `4x4` | 0.225 ± 0.006 | 0.032 ± 0.001 | 0.067 ± 0.002 | 0.179 ± 0.010 |
| small | `svd` | f64 | 1 | `8x8` | 0.204 ± 0.009 | 0.044 ± 0.001 | 0.092 ± 0.002 | 0.204 ± 0.007 |

## Threads: 4

### CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `linux-cpu`
- Timestamp: `20260728_015909`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/linux-cpu/cpu/einsum/20260728_015909`.

- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

#### CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

#### Thread Environment

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

#### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 4

- CSV: `data/results/linux-cpu/cpu/einsum/20260728_015909/cpu_ops_t4_20260728_015909.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260728_015909/cpu_ops_t4_20260728_015909.md`

#### CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 1.141 ± 0.622 | 0.065 ± 0.010 | 0.318 ± 0.014 | 0.928 ± 0.038 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.625 ± 0.032 | 0.141 ± 0.043 | 1.064 ± 0.010 | 1.731 ± 0.045 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.729 ± 0.140 | 0.092 ± 0.010 | 0.422 ± 0.013 | 1.012 ± 0.023 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.761 ± 0.072 | 0.239 ± 0.003 | 1.426 ± 0.013 | 3.435 ± 0.531 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.799 ± 0.697 | 0.072 ± 0.037 | 0.096 ± 0.003 | 0.577 ± 0.015 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.477 ± 0.099 | 0.130 ± 0.010 | 0.204 ± 0.003 | 0.738 ± 0.296 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.535 ± 0.348 | 0.034 ± 0.002 | 0.201 ± 0.003 | 0.936 ± 0.176 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.505 ± 0.048 | 0.134 ± 0.003 | 0.602 ± 0.007 | 1.101 ± 0.030 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 1.002 ± 0.656 | 0.058 ± 0.002 | 0.202 ± 0.003 | 0.519 ± 0.020 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.576 ± 0.034 | 0.111 ± 0.003 | 0.652 ± 0.010 | 0.755 ± 0.141 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.702 ± 0.050 | 0.067 ± 0.003 | 0.273 ± 0.010 | 0.407 ± 0.010 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.639 ± 0.023 | 0.142 ± 0.006 | 0.920 ± 0.008 | 2.354 ± 0.022 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.971 ± 0.147 | 0.149 ± 0.014 | 0.359 ± 0.009 | 0.555 ± 0.020 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.731 ± 0.010 | 0.215 ± 0.028 | 1.100 ± 0.004 | 1.402 ± 0.004 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.680 ± 0.126 | 0.151 ± 0.009 | 0.432 ± 0.015 | 0.640 ± 0.011 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.774 ± 0.010 | 0.242 ± 0.343 | 1.409 ± 0.009 | 3.448 ± 0.055 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.790 ± 0.534 | 0.082 ± 0.013 | 0.230 ± 0.013 | 0.347 ± 0.011 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.658 ± 0.041 | 0.190 ± 0.008 | 0.713 ± 0.011 | 0.855 ± 0.013 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.644 ± 0.086 | 0.144 ± 0.144 | 0.347 ± 0.010 | 0.465 ± 0.015 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.939 ± 0.032 | 0.413 ± 0.011 | 1.145 ± 0.004 | 2.872 ± 0.056 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 1.719 ± 0.711 | 0.121 ± 0.008 | 0.258 ± 0.009 | 3.749 ± 1.893 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.953 ± 0.084 | 0.132 ± 0.010 | 0.362 ± 0.009 | 4.162 ± 1.714 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 1.236 ± 0.469 | 0.118 ± 0.005 | 0.373 ± 0.022 | 3.758 ± 2.275 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 1.016 ± 0.029 | 0.136 ± 0.007 | 0.790 ± 0.016 | 3.526 ± 2.972 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 1.972 ± 0.312 | 0.491 ± 0.062 | 0.502 ± 0.007 | 3.117 ± 0.219 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 1.895 ± 0.030 | 0.649 ± 0.219 | 1.249 ± 0.004 | 5.015 ± 2.788 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 1.856 ± 0.156 | 0.538 ± 0.020 | 0.585 ± 0.011 | 3.236 ± 0.252 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 2.007 ± 0.021 | 0.671 ± 0.023 | 1.578 ± 0.012 | 4.979 ± 2.961 |
| large | `eigh` | f64 | 4 | `64x64` | 22.052 ± 3.126 | 7.777 ± 1.261 | 1.305 ± 0.044 | 3.256 ± 1.402 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 62.996 ± 14.374 | 6.796 ± 0.222 | 14.493 ± 3.775 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 172.693 ± 8.039 | 33.858 ± 0.860 | 51.672 ± 11.005 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 70.433 ± 14.696 | 5.895 ± 0.079 | 14.597 ± 4.601 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 97.448 ± 1.793 | 22.678 ± 2.169 | 52.944 ± 12.229 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 42.591 ± 2.101 | 7.554 ± 1.166 | 4.014 ± 0.520 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 159.262 ± 8.610 | 32.516 ± 1.149 | 14.129 ± 1.227 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 43.311 ± 4.269 | 3.249 ± 0.031 | 7.244 ± 0.793 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 178.901 ± 11.545 | 18.069 ± 0.922 | 15.600 ± 1.149 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 14.985 ± 1.397 | 1.071 ± 0.031 | 2.247 ± 0.009 | 2.602 ± 0.049 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 32.565 ± 9.031 | 1.500 ± 0.118 | 2.405 ± 0.177 | 3.881 ± 0.149 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 56.400 ± 6.916 | 4.448 ± 0.420 | 7.295 ± 2.581 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 236.788 ± 4121.962 | 23.965 ± 1.402 | 36.223 ± 9.099 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 53.952 ± 1.687 | 5.001 ± 0.653 | 10.108 ± 1.445 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 228.478 ± 36.682 | 25.298 ± 0.863 | 56.087 ± 10.404 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 2.781 ± 0.684 | 0.557 ± 0.016 | 1.174 ± 0.210 | 5.603 ± 3.000 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 16.333 ± 8.153 | 1.240 ± 0.013 | 2.643 ± 0.266 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 100.197 ± 1.012 | 2.995 ± 0.029 | 6.948 ± 0.305 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 14.536 ± 12.643 | 1.545 ± 0.108 | 2.233 ± 0.363 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 80.277 ± 18.023 | 4.581 ± 0.035 | 6.799 ± 0.244 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 70.791 ± 2.997 | 11.169 ± 7.303 | 1.906 ± 0.291 | 4.152 ± 2.863 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 100.653 ± 22.089 | 14.089 ± 0.274 | 28.923 ± 11.240 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 190.369 ± 76.573 | 81.753 ± 7.483 | 104.181 ± 16.088 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 91.903 ± 6.826 | 13.093 ± 0.282 | 29.012 ± 11.053 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 186.015 ± 3.519 | 74.330 ± 2.723 | 100.685 ± 10.115 |
| large | `matmul` | f64 | 4 | `128x128` | 10.965 ± 1.882 | 0.268 ± 0.016 | 7.354 ± 0.313 | 9.594 ± 1.467 |
| large | `matmul` | f64 | 4 | `256x256` | 23.727 ± 5.526 | 1.838 ± 0.027 | 30.058 ± 1.724 | 32.839 ± 1.503 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 36.058 ± 1.596 | 9.703 ± 0.729 | 138.308 ± 7.351 | 130.606 ± 2.963 |
| large | `qr` | f64 | 4 | `64x64` | 18.857 ± 4.207 | 3.246 ± 0.783 | 1.340 ± 0.195 | 2.836 ± 0.059 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 1.452 ± 0.628 | 0.318 ± 0.047 | 1.057 ± 0.006 | 3.043 ± 0.367 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 15.019 ± 7.171 | 1.239 ± 0.383 | 1.284 ± 0.159 | 3.519 ± 0.026 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 16.729 ± 1.008 | 2.225 ± 0.045 | 1.988 ± 0.016 | 5.270 ± 0.591 |
| large | `svd` | f64 | 4 | `64x64` | 27.919 ± 3.291 | 9.376 ± 2.425 | 1.871 ± 0.010 | 2.233 ± 0.048 |
| small | `eigh` | f64 | 4 | `2x2` | 0.503 ± 0.037 | 0.038 ± 0.003 | 0.061 ± 0.007 | 0.176 ± 0.009 |
| small | `eigh` | f64 | 4 | `4x4` | 0.486 ± 0.023 | 0.041 ± 0.002 | 0.060 ± 0.001 | 0.346 ± 0.009 |
| small | `eigh` | f64 | 4 | `8x8` | 0.468 ± 0.131 | 0.046 ± 0.003 | 0.081 ± 0.015 | 0.384 ± 0.019 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.452 ± 0.034 | 0.031 ± 0.048 | 0.063 ± 0.005 | 0.361 ± 0.015 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.440 ± 0.022 | 0.030 ± 0.003 | 0.072 ± 0.002 | 0.437 ± 0.019 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.425 ± 0.037 | 0.031 ± 0.029 | 0.084 ± 0.002 | 0.471 ± 0.015 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.234 ± 0.040 | 0.193 ± 0.051 | 0.573 ± 0.015 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.244 ± 0.007 | 0.193 ± 0.005 | 0.587 ± 0.021 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.243 ± 0.003 | 0.196 ± 0.008 | 1.061 ± 0.036 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.238 ± 0.004 | 0.192 ± 0.006 | 1.568 ± 1.074 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.238 ± 0.006 | 0.194 ± 0.009 | 1.314 ± 0.042 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.245 ± 0.003 | 0.236 ± 0.041 | 2.761 ± 0.031 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.225 ± 0.006 | 0.276 ± 0.015 | 0.818 ± 0.019 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.224 ± 0.015 | 0.279 ± 0.009 | 0.825 ± 0.024 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.237 ± 0.006 | 0.331 ± 0.070 | 0.830 ± 0.023 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.238 ± 0.012 | 0.306 ± 0.049 | 2.054 ± 0.023 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.245 ± 0.007 | 0.261 ± 0.009 | 2.079 ± 0.017 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.245 ± 0.006 | 0.261 ± 0.010 | 2.111 ± 0.046 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.867 ± 0.174 | 0.098 ± 0.004 | 0.146 ± 0.007 | 3.057 ± 0.032 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.841 ± 0.032 | 0.100 ± 0.004 | 0.128 ± 0.005 | 3.101 ± 0.387 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.908 ± 0.035 | 0.102 ± 0.003 | 0.150 ± 0.005 | 3.277 ± 0.432 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.420 ± 0.012 | 0.262 ± 0.015 | 1.720 ± 0.040 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.427 ± 0.012 | 0.219 ± 0.014 | 1.537 ± 0.034 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.435 ± 0.006 | 0.266 ± 0.008 | 1.625 ± 0.407 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.421 ± 0.009 | 0.241 ± 0.015 | 3.931 ± 0.071 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.422 ± 0.016 | 0.244 ± 0.086 | 4.026 ± 0.444 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.434 ± 0.005 | 0.299 ± 0.039 | 4.173 ± 0.423 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 1.443 ± 0.011 | 0.381 ± 0.005 | 0.186 ± 0.020 | 3.314 ± 0.424 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 1.436 ± 0.029 | 0.387 ± 0.010 | 0.191 ± 0.004 | 3.429 ± 0.039 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 1.520 ± 0.087 | 0.391 ± 0.040 | 0.244 ± 0.013 | 3.481 ± 0.018 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.261 ± 0.031 | 0.420 ± 0.013 | 0.681 ± 0.024 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.255 ± 0.006 | 0.484 ± 0.068 | 0.700 ± 0.023 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.256 ± 0.003 | 0.419 ± 0.113 | 0.694 ± 0.017 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.279 ± 0.008 | 0.312 ± 0.056 | 1.425 ± 0.030 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.277 ± 0.011 | 0.290 ± 0.091 | 1.430 ± 0.012 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.286 ± 0.024 | 0.279 ± 0.030 | 1.426 ± 0.024 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.906 ± 0.031 | 0.146 ± 0.009 | 0.168 ± 0.009 | 3.187 ± 0.052 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.931 ± 0.139 | 0.150 ± 0.021 | 0.151 ± 0.002 | 3.307 ± 0.227 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.995 ± 0.040 | 0.158 ± 0.003 | 0.174 ± 0.048 | 2.999 ± 0.875 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.134 ± 0.023 | 0.251 ± 0.008 | 1.220 ± 0.051 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.143 ± 0.021 | 0.254 ± 0.008 | 1.121 ± 0.048 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.153 ± 0.020 | 0.263 ± 0.010 | 1.216 ± 0.066 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.138 ± 0.019 | 0.221 ± 0.008 | 2.878 ± 0.031 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.145 ± 0.005 | 0.211 ± 0.010 | 2.978 ± 0.032 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.157 ± 0.009 | 0.223 ± 0.017 | 2.996 ± 0.020 |
| small | `matmul` | f64 | 4 | `2x2` | 0.492 ± 0.050 | 0.028 ± 0.024 | 0.032 ± 0.001 | 0.244 ± 0.008 |
| small | `matmul` | f64 | 4 | `4x4` | 0.460 ± 0.030 | 0.030 ± 0.023 | 0.048 ± 0.004 | 0.244 ± 0.029 |
| small | `matmul` | f64 | 4 | `8x8` | 0.435 ± 0.036 | 0.029 ± 0.031 | 0.062 ± 0.003 | 0.296 ± 0.018 |
| small | `qr` | f64 | 4 | `2x2` | 0.473 ± 0.006 | 0.037 ± 0.027 | 0.046 ± 0.002 | 0.162 ± 0.007 |
| small | `qr` | f64 | 4 | `4x4` | 0.470 ± 0.016 | 0.039 ± 0.033 | 0.045 ± 0.002 | 0.175 ± 0.019 |
| small | `qr` | f64 | 4 | `8x8` | 0.471 ± 0.014 | 0.040 ± 0.039 | 0.057 ± 0.001 | 0.192 ± 0.012 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.562 ± 0.003 | 0.114 ± 0.010 | 0.098 ± 0.007 | 0.283 ± 0.011 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.605 ± 0.029 | 0.117 ± 0.011 | 0.096 ± 0.006 | 0.280 ± 0.019 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.598 ± 0.098 | 0.110 ± 0.006 | 0.086 ± 0.002 | 0.283 ± 0.009 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.609 ± 0.026 | 0.117 ± 0.003 | 0.087 ± 0.002 | 0.280 ± 0.014 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.584 ± 0.029 | 0.126 ± 0.020 | 0.117 ± 0.004 | 0.302 ± 0.009 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.657 ± 0.051 | 0.118 ± 0.005 | 0.104 ± 0.032 | 0.308 ± 0.016 |
| small | `svd` | f64 | 4 | `2x2` | 0.511 ± 0.138 | 0.040 ± 0.002 | 0.059 ± 0.005 | 0.171 ± 0.007 |
| small | `svd` | f64 | 4 | `4x4` | 0.497 ± 0.091 | 0.043 ± 0.005 | 0.057 ± 0.001 | 0.175 ± 0.015 |
| small | `svd` | f64 | 4 | `8x8` | 0.484 ± 0.004 | 0.054 ± 0.011 | 0.077 ± 0.001 | 0.168 ± 0.013 |

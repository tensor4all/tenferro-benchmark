# Linux CPU Ops Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `linux-cpu`
- Thread runs: `1 4`

Mirrored from `amd-cpu` runs (`BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh <THREADS>`);
regenerate with `./scripts/mirror_linux_cpu_cpu_reports.sh 1:20260723_035222 4:20260723_042457 permutation:20260723_051703`.

This report is generated from sequential CPU runs. Do not compare it with
measurements collected while another CPU benchmark process was running.

## Run Inputs

- Threads 1: timestamp `20260723_035222`, raw run `data/results/linux-cpu/cpu/einsum/20260723_035222`
- Threads 4: timestamp `20260723_042457`, raw run `data/results/linux-cpu/cpu/einsum/20260723_042457`

## Threads: 1

### CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `linux-cpu`
- Timestamp: `20260723_035222`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/linux-cpu/cpu/einsum/20260723_035222`.

- tenferro-rs commit: `68855c2b65b5adc42dccca9bac04fd136a8f14c8`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`

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
- BLAS version: `2026.0.0`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 1

- CSV: `data/results/linux-cpu/cpu/einsum/20260723_035222/cpu_ops_t1_20260723_035222.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260723_035222/cpu_ops_t1_20260723_035222.md`

#### CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.182 ± 0.010 | 0.036 ± 0.001 | 0.576 ± 0.042 | 1.770 ± 0.021 |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.276 ± 0.017 | 0.110 ± 0.010 | 1.954 ± 0.011 | 5.918 ± 0.277 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.225 ± 0.009 | 0.076 ± 0.001 | 0.709 ± 0.039 | 2.108 ± 0.055 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.460 ± 0.047 | 0.275 ± 0.026 | 2.601 ± 0.125 | 3.718 ± 0.070 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.140 ± 0.006 | 0.008 ± 0.000 | 0.167 ± 0.018 | 1.576 ± 0.040 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.149 ± 0.010 | 0.020 ± 0.001 | 0.303 ± 0.017 | 1.302 ± 0.136 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.139 ± 0.007 | 0.008 ± 0.000 | 0.313 ± 0.036 | 1.873 ± 0.035 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.152 ± 0.002 | 0.023 ± 0.000 | 0.852 ± 0.040 | 2.920 ± 0.152 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.164 ± 0.003 | 0.027 ± 0.010 | 0.317 ± 0.024 | 1.225 ± 0.024 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.222 ± 0.008 | 0.084 ± 0.017 | 1.095 ± 0.040 | 3.044 ± 0.094 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.174 ± 0.014 | 0.034 ± 0.000 | 0.425 ± 0.032 | 1.554 ± 0.057 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.265 ± 0.003 | 0.103 ± 0.011 | 1.514 ± 0.035 | 2.298 ± 0.039 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.207 ± 0.006 | 0.035 ± 0.000 | 0.644 ± 0.033 | 1.437 ± 0.017 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.274 ± 0.043 | 0.092 ± 0.007 | 2.056 ± 0.046 | 6.029 ± 0.091 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.208 ± 0.002 | 0.037 ± 0.000 | 0.768 ± 0.086 | 2.349 ± 0.060 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.295 ± 0.038 | 0.115 ± 0.004 | 2.560 ± 0.071 | 3.714 ± 0.091 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.204 ± 0.008 | 0.053 ± 0.001 | 0.391 ± 0.033 | 1.313 ± 0.018 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.350 ± 0.036 | 0.208 ± 0.023 | 1.206 ± 0.047 | 2.670 ± 0.040 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.325 ± 0.075 | 0.139 ± 0.022 | 0.570 ± 0.038 | 1.617 ± 0.015 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.750 ± 0.027 | 0.520 ± 0.011 | 1.838 ± 0.156 | 5.059 ± 0.243 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.354 ± 0.011 | 0.050 ± 0.007 | 0.379 ± 0.041 | 5.656 ± 1.246 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.433 ± 0.096 | 0.080 ± 0.002 | 0.563 ± 0.026 | 5.531 ± 0.309 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.366 ± 0.022 | 0.051 ± 0.002 | 0.559 ± 0.022 | 5.758 ± 0.077 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.415 ± 0.049 | 0.084 ± 0.008 | 1.185 ± 0.067 | 5.301 ± 1.830 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.555 ± 0.030 | 0.201 ± 0.017 | 0.857 ± 0.063 | 6.408 ± 1.375 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.681 ± 0.018 | 0.322 ± 0.018 | 2.274 ± 0.072 | 9.996 ± 0.199 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.572 ± 0.043 | 0.180 ± 0.019 | 0.967 ± 0.030 | 6.201 ± 0.810 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.710 ± 0.052 | 0.323 ± 0.041 | 2.756 ± 0.032 | 7.185 ± 0.731 |
| large | `eigh` | f64 | 1 | `64x64` | 0.985 ± 0.031 | 0.496 ± 0.078 | 2.192 ± 0.026 | 6.810 ± 0.081 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 12.684 ± 0.622 | 12.219 ± 0.245 | 33.582 ± 0.244 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 70.333 ± 0.486 | 80.834 ± 1.245 | 112.353 ± 0.657 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 12.562 ± 0.339 | 10.203 ± 0.259 | 32.987 ± 0.269 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 68.780 ± 0.747 | 59.669 ± 1.486 | 114.540 ± 1.363 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 7.788 ± 0.297 | 9.464 ± 0.261 | 11.413 ± 1.823 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 43.059 ± 2.141 | 72.511 ± 0.900 | 43.025 ± 3.329 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 7.168 ± 0.523 | 5.460 ± 0.025 | 14.104 ± 0.907 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 42.527 ± 0.279 | 40.201 ± 2.188 | 51.466 ± 1.233 |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | 0.203 ± 0.010 | 0.045 ± 0.004 | 3.147 ± 0.031 | 9.011 ± 1.168 |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | 0.482 ± 0.022 | 0.108 ± 0.008 | 3.213 ± 0.067 | 7.582 ± 2.283 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 10.369 ± 0.314 | 8.054 ± 0.112 | 17.175 ± 1.551 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 57.296 ± 2.150 | 52.903 ± 0.487 | 75.656 ± 2.339 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 10.792 ± 0.602 | 7.689 ± 0.233 | 18.757 ± 1.389 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 57.963 ± 0.628 | 51.758 ± 1.590 | 97.342 ± 4.293 |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | 0.903 ± 0.030 | 0.210 ± 0.030 | 2.099 ± 0.068 | 6.300 ± 1.218 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.512 ± 0.044 | 1.780 ± 0.016 | 4.591 ± 0.212 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 7.629 ± 0.185 | 5.794 ± 0.318 | 14.915 ± 0.282 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 1.223 ± 0.081 | 2.158 ± 0.021 | 4.359 ± 0.067 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 5.975 ± 0.102 | 9.391 ± 0.268 | 15.763 ± 0.478 |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | 2.015 ± 0.043 | 0.977 ± 0.013 | 2.672 ± 0.020 | 9.237 ± 1.055 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 20.292 ± 0.128 | 23.061 ± 0.210 | 58.292 ± 2.898 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 123.906 ± 0.528 | 147.386 ± 1.131 | 217.923 ± 27.369 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 19.885 ± 0.147 | 19.124 ± 0.258 | 56.446 ± 16.246 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 122.615 ± 1.553 | 119.296 ± 1.170 | 178.635 ± 21.268 |
| large | `matmul` | f64 | 1 | `128x128` | 0.499 ± 0.018 | 0.199 ± 0.016 | 11.759 ± 0.243 | 18.069 ± 0.407 |
| large | `matmul` | f64 | 1 | `256x256` | 2.125 ± 0.041 | 1.150 ± 0.026 | 46.613 ± 0.217 | 70.161 ± 1.952 |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | 6.161 ± 0.170 | 4.602 ± 0.190 | 188.652 ± 3.741 | 188.915 ± 3.222 |
| large | `qr` | f64 | 1 | `64x64` | 0.284 ± 0.019 | 0.113 ± 0.035 | 1.798 ± 0.037 | 5.318 ± 0.052 |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | 0.566 ± 0.046 | 0.055 ± 0.001 | 1.889 ± 0.052 | 5.503 ± 0.098 |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | 0.596 ± 0.064 | 0.067 ± 0.012 | 2.273 ± 0.058 | 6.517 ± 0.049 |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | 0.613 ± 0.023 | 0.126 ± 0.022 | 3.448 ± 0.028 | 9.631 ± 1.303 |
| large | `svd` | f64 | 1 | `64x64` | 1.108 ± 0.019 | 0.869 ± 0.111 | 2.565 ± 0.091 | 6.683 ± 1.766 |
| small | `eigh` | f64 | 1 | `2x2` | 0.183 ± 0.008 | 0.008 ± 0.000 | 0.104 ± 0.024 | 0.306 ± 0.009 |
| small | `eigh` | f64 | 1 | `4x4` | 0.175 ± 0.009 | 0.013 ± 0.000 | 0.115 ± 0.011 | 0.602 ± 0.035 |
| small | `eigh` | f64 | 1 | `8x8` | 0.162 ± 0.003 | 0.019 ± 0.006 | 0.157 ± 0.026 | 0.687 ± 0.016 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | 0.175 ± 0.005 | 0.004 ± 0.000 | 0.124 ± 0.033 | 0.570 ± 0.044 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | 0.134 ± 0.003 | 0.004 ± 0.000 | 0.133 ± 0.011 | 1.435 ± 0.028 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | 0.134 ± 0.002 | 0.004 ± 0.000 | 0.169 ± 0.018 | 1.568 ± 0.011 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.090 ± 0.001 | 0.360 ± 0.041 | 1.242 ± 0.011 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.099 ± 0.011 | 0.355 ± 0.028 | 1.255 ± 0.013 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.105 ± 0.009 | 0.356 ± 0.035 | 1.632 ± 0.024 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.100 ± 0.008 | 0.330 ± 0.032 | 3.536 ± 0.017 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.107 ± 0.001 | 0.345 ± 0.029 | 3.562 ± 0.026 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.114 ± 0.004 | 0.359 ± 0.031 | 4.148 ± 0.035 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.094 ± 0.011 | 0.464 ± 0.042 | 1.710 ± 0.042 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.091 ± 0.013 | 0.468 ± 0.015 | 1.794 ± 0.012 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.093 ± 0.007 | 0.484 ± 0.016 | 2.563 ± 0.026 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.097 ± 0.011 | 0.433 ± 0.023 | 5.209 ± 0.045 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.105 ± 0.021 | 0.429 ± 0.045 | 5.651 ± 0.515 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.106 ± 0.019 | 0.439 ± 0.028 | 6.515 ± 1.208 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | 0.449 ± 0.042 | 0.041 ± 0.001 | 0.227 ± 0.012 | 4.625 ± 0.323 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | 0.374 ± 0.025 | 0.041 ± 0.009 | 0.238 ± 0.025 | 4.687 ± 0.616 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | 0.346 ± 0.296 | 0.040 ± 0.001 | 0.290 ± 0.019 | 4.812 ± 0.593 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.209 ± 0.018 | 0.373 ± 0.035 | 2.677 ± 0.033 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.221 ± 0.027 | 0.381 ± 0.022 | 2.363 ± 0.025 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.228 ± 0.016 | 0.376 ± 0.037 | 2.548 ± 0.040 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.506 ± 0.027 | 0.405 ± 0.032 | 6.064 ± 0.995 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.502 ± 0.046 | 0.413 ± 0.035 | 6.049 ± 0.741 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.490 ± 0.006 | 0.429 ± 0.053 | 6.190 ± 0.981 |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | 0.536 ± 0.032 | 0.116 ± 0.010 | 0.351 ± 0.030 | 5.067 ± 0.274 |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | 0.517 ± 0.051 | 0.116 ± 0.008 | 0.342 ± 0.036 | 5.138 ± 0.334 |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | 0.508 ± 0.026 | 0.118 ± 0.004 | 0.376 ± 0.031 | 5.268 ± 0.226 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.088 ± 0.029 | 0.714 ± 0.029 | 2.098 ± 0.009 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.058 ± 0.003 | 0.713 ± 0.022 | 2.242 ± 0.018 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.059 ± 0.008 | 0.724 ± 0.068 | 2.061 ± 0.027 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.083 ± 0.012 | 0.465 ± 0.016 | 4.430 ± 0.022 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.082 ± 0.002 | 0.473 ± 0.083 | 4.450 ± 0.048 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.085 ± 0.003 | 0.475 ± 0.028 | 4.627 ± 0.069 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | 0.974 ± 0.013 | 0.073 ± 0.005 | 0.250 ± 0.021 | 4.818 ± 0.308 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | 0.832 ± 0.024 | 0.073 ± 0.004 | 0.262 ± 0.030 | 4.891 ± 0.346 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | 0.894 ± 0.136 | 0.094 ± 0.005 | 0.297 ± 0.019 | 5.052 ± 0.529 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.057 ± 0.005 | 0.452 ± 0.011 | 1.780 ± 0.006 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.063 ± 0.002 | 0.456 ± 0.030 | 1.961 ± 0.019 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.079 ± 0.004 | 0.482 ± 0.045 | 1.660 ± 0.023 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.067 ± 0.001 | 0.396 ± 0.116 | 4.325 ± 0.069 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.075 ± 0.004 | 0.372 ± 0.035 | 4.395 ± 0.108 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.090 ± 0.008 | 0.413 ± 0.095 | 4.503 ± 0.143 |
| small | `matmul` | f64 | 1 | `2x2` | 0.175 ± 0.030 | 0.004 ± 0.001 | 0.084 ± 0.004 | 0.375 ± 0.064 |
| small | `matmul` | f64 | 1 | `4x4` | 0.134 ± 0.011 | 0.004 ± 0.001 | 0.093 ± 0.004 | 0.699 ± 0.041 |
| small | `matmul` | f64 | 1 | `8x8` | 0.133 ± 0.003 | 0.004 ± 0.000 | 0.128 ± 0.015 | 1.004 ± 0.027 |
| small | `qr` | f64 | 1 | `2x2` | 0.182 ± 0.004 | 0.008 ± 0.001 | 0.075 ± 0.001 | 0.242 ± 0.032 |
| small | `qr` | f64 | 1 | `4x4` | 0.163 ± 0.018 | 0.008 ± 0.000 | 0.084 ± 0.005 | 0.524 ± 0.029 |
| small | `qr` | f64 | 1 | `8x8` | 0.146 ± 0.015 | 0.010 ± 0.000 | 0.111 ± 0.024 | 0.601 ± 0.021 |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | 0.229 ± 0.008 | 0.016 ± 0.001 | 0.159 ± 0.006 | 0.473 ± 0.021 |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | 0.233 ± 0.011 | 0.017 ± 0.000 | 0.158 ± 0.013 | 0.530 ± 0.022 |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | 0.212 ± 0.039 | 0.016 ± 0.000 | 0.174 ± 0.014 | 0.698 ± 0.032 |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | 0.212 ± 0.012 | 0.018 ± 0.000 | 0.171 ± 0.010 | 0.761 ± 0.038 |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | 0.181 ± 0.003 | 0.017 ± 0.000 | 0.190 ± 0.015 | 0.767 ± 0.020 |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | 0.188 ± 0.007 | 0.018 ± 0.000 | 0.217 ± 0.030 | 0.790 ± 0.013 |
| small | `svd` | f64 | 1 | `2x2` | 0.203 ± 0.013 | 0.010 ± 0.000 | 0.098 ± 0.003 | 0.262 ± 0.073 |
| small | `svd` | f64 | 1 | `4x4` | 0.165 ± 0.005 | 0.016 ± 0.000 | 0.113 ± 0.012 | 0.569 ± 0.033 |
| small | `svd` | f64 | 1 | `8x8` | 0.177 ± 0.005 | 0.030 ± 0.012 | 0.148 ± 0.009 | 0.625 ± 0.016 |

## Threads: 4

### CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `linux-cpu`
- Timestamp: `20260723_042457`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/linux-cpu/cpu/einsum/20260723_042457`.

- tenferro-rs commit: `68855c2b65b5adc42dccca9bac04fd136a8f14c8`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`

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
- BLAS version: `2026.0.0`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 4

- CSV: `data/results/linux-cpu/cpu/einsum/20260723_042457/cpu_ops_t4_20260723_042457.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260723_042457/cpu_ops_t4_20260723_042457.md`

#### CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.469 ± 0.034 | 0.046 ± 0.009 | 0.622 ± 0.090 | 1.738 ± 0.026 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.591 ± 0.016 | 0.116 ± 0.020 | 2.027 ± 0.064 | 5.277 ± 0.138 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.500 ± 0.012 | 0.083 ± 0.023 | 0.723 ± 0.021 | 2.104 ± 0.046 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.769 ± 0.038 | 0.291 ± 0.060 | 2.612 ± 0.972 | 6.653 ± 0.157 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.400 ± 0.022 | 0.010 ± 0.001 | 0.208 ± 0.038 | 1.565 ± 0.016 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.405 ± 0.011 | 0.023 ± 0.001 | 0.311 ± 0.013 | 1.339 ± 0.023 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.400 ± 0.008 | 0.009 ± 0.001 | 0.309 ± 0.011 | 1.827 ± 0.138 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.443 ± 0.128 | 0.025 ± 0.001 | 0.898 ± 0.070 | 3.024 ± 0.214 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.430 ± 0.013 | 0.031 ± 0.008 | 0.408 ± 0.037 | 1.180 ± 0.018 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.502 ± 0.016 | 0.095 ± 0.001 | 1.293 ± 0.056 | 2.287 ± 0.034 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.449 ± 0.036 | 0.040 ± 0.000 | 0.496 ± 0.048 | 1.265 ± 0.025 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.577 ± 0.040 | 0.117 ± 0.019 | 1.826 ± 0.260 | 4.239 ± 0.045 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.485 ± 0.027 | 0.043 ± 0.000 | 0.735 ± 0.259 | 1.454 ± 0.029 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.642 ± 0.067 | 0.119 ± 0.010 | 2.371 ± 0.133 | 5.935 ± 2.656 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.485 ± 0.017 | 0.046 ± 0.007 | 0.924 ± 0.202 | 1.808 ± 0.906 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.630 ± 0.046 | 0.106 ± 0.035 | 2.853 ± 0.056 | 6.866 ± 0.099 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.458 ± 0.032 | 0.059 ± 0.010 | 0.404 ± 0.020 | 1.181 ± 0.022 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.659 ± 0.042 | 0.233 ± 0.005 | 1.266 ± 0.215 | 2.446 ± 0.033 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.594 ± 0.027 | 0.164 ± 0.020 | 0.582 ± 0.180 | 1.362 ± 0.048 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 1.295 ± 0.647 | 0.635 ± 0.017 | 1.968 ± 0.139 | 5.094 ± 0.021 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.679 ± 0.051 | 0.063 ± 0.006 | 0.400 ± 0.022 | 5.738 ± 1.276 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.815 ± 0.029 | 0.099 ± 0.005 | 0.577 ± 0.028 | 5.948 ± 1.590 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.695 ± 0.032 | 0.067 ± 0.006 | 0.549 ± 0.014 | 4.215 ± 0.898 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.754 ± 0.030 | 0.102 ± 0.007 | 1.142 ± 0.010 | 6.110 ± 2.456 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.962 ± 0.034 | 0.219 ± 0.023 | 0.936 ± 0.073 | 6.384 ± 1.055 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 1.305 ± 0.179 | 0.380 ± 0.024 | 2.637 ± 0.068 | 7.337 ± 3.644 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.938 ± 0.186 | 0.223 ± 0.005 | 1.069 ± 0.055 | 6.716 ± 1.642 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 1.078 ± 0.012 | 0.419 ± 0.012 | 3.078 ± 0.200 | 7.672 ± 1.105 |
| large | `eigh` | f64 | 4 | `64x64` | 1.196 ± 0.012 | 0.410 ± 0.025 | 3.070 ± 0.081 | 6.762 ± 0.147 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 8.173 ± 0.073 | 7.977 ± 0.397 | 30.567 ± 0.522 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 30.303 ± 0.384 | 28.180 ± 0.272 | 88.076 ± 2.704 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 7.758 ± 0.259 | 6.779 ± 0.129 | 29.475 ± 0.470 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 29.052 ± 0.862 | 23.267 ± 0.574 | 89.556 ± 3.272 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 5.701 ± 0.527 | 4.548 ± 0.144 | 8.193 ± 1.669 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 24.414 ± 0.485 | 24.724 ± 1.446 | 25.470 ± 0.793 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 4.877 ± 0.553 | 2.962 ± 0.170 | 10.804 ± 1.812 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 22.778 ± 0.554 | 12.157 ± 0.396 | 35.680 ± 6.376 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.471 ± 0.055 | 0.041 ± 0.003 | 4.082 ± 0.133 | 8.531 ± 1.494 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.735 ± 0.038 | 0.116 ± 0.007 | 4.181 ± 0.044 | 7.799 ± 1.942 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 7.520 ± 0.588 | 4.514 ± 0.063 | 14.114 ± 0.886 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 29.933 ± 1.123 | 19.717 ± 0.543 | 44.192 ± 1.527 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 7.130 ± 0.979 | 4.384 ± 0.154 | 14.752 ± 2.000 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 28.096 ± 0.366 | 19.489 ± 0.918 | 61.908 ± 2.113 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 1.235 ± 0.021 | 0.237 ± 0.020 | 1.971 ± 0.043 | 6.646 ± 3.195 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 1.450 ± 0.062 | 1.664 ± 0.031 | 4.117 ± 0.073 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 5.517 ± 0.283 | 3.288 ± 0.217 | 10.569 ± 0.736 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.897 ± 0.024 | 1.773 ± 0.023 | 5.198 ± 0.149 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 3.525 ± 0.203 | 4.664 ± 0.224 | 11.329 ± 0.821 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 2.491 ± 0.095 | 1.130 ± 0.027 | 3.806 ± 0.213 | 8.740 ± 1.442 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 17.542 ± 0.224 | 21.703 ± 3.839 | 53.130 ± 0.596 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 72.295 ± 0.196 | 82.333 ± 6.689 | 162.393 ± 2.778 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 16.955 ± 0.036 | 16.514 ± 0.439 | 51.212 ± 1.665 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 70.770 ± 1.352 | 65.032 ± 1.764 | 163.880 ± 4.562 |
| large | `matmul` | f64 | 4 | `128x128` | 0.694 ± 0.034 | 0.143 ± 0.009 | 12.355 ± 0.851 | 18.160 ± 0.270 |
| large | `matmul` | f64 | 4 | `256x256` | 2.063 ± 0.048 | 0.650 ± 0.034 | 46.001 ± 0.662 | 71.836 ± 1.896 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 4.603 ± 0.085 | 2.683 ± 0.073 | 186.742 ± 2.633 | 211.755 ± 4.199 |
| large | `qr` | f64 | 4 | `64x64` | 0.616 ± 0.017 | 0.146 ± 0.017 | 2.706 ± 0.087 | 5.303 ± 0.113 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.864 ± 0.031 | 0.067 ± 0.009 | 1.898 ± 0.281 | 5.477 ± 0.170 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.901 ± 0.073 | 0.086 ± 0.015 | 3.240 ± 0.389 | 6.488 ± 0.504 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.953 ± 0.029 | 0.148 ± 0.030 | 4.351 ± 0.168 | 9.485 ± 1.489 |
| large | `svd` | f64 | 4 | `64x64` | 1.486 ± 0.039 | 0.953 ± 0.031 | 3.571 ± 0.046 | 6.545 ± 1.532 |
| small | `eigh` | f64 | 4 | `2x2` | 0.423 ± 0.032 | 0.010 ± 0.002 | 0.111 ± 0.013 | 0.420 ± 0.015 |
| small | `eigh` | f64 | 4 | `4x4` | 0.420 ± 0.025 | 0.014 ± 0.001 | 0.121 ± 0.005 | 0.702 ± 0.035 |
| small | `eigh` | f64 | 4 | `8x8` | 0.439 ± 0.070 | 0.021 ± 0.001 | 0.153 ± 0.017 | 0.748 ± 0.023 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.437 ± 0.029 | 0.006 ± 0.004 | 0.125 ± 0.011 | 0.580 ± 0.045 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.409 ± 0.017 | 0.005 ± 0.001 | 0.125 ± 0.003 | 1.451 ± 0.017 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.398 ± 0.033 | 0.006 ± 0.001 | 0.169 ± 0.030 | 1.797 ± 0.013 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.114 ± 0.011 | 0.339 ± 0.028 | 1.470 ± 0.033 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.124 ± 0.029 | 0.353 ± 0.011 | 1.201 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.128 ± 0.022 | 0.408 ± 0.018 | 1.721 ± 0.021 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.123 ± 0.031 | 0.320 ± 0.009 | 3.827 ± 0.155 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.130 ± 0.013 | 0.327 ± 0.013 | 3.503 ± 0.060 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.134 ± 0.028 | 0.343 ± 0.035 | 4.578 ± 0.042 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.123 ± 0.020 | 0.522 ± 0.012 | 1.906 ± 0.051 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.128 ± 0.008 | 0.508 ± 0.032 | 1.837 ± 0.016 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.119 ± 0.006 | 0.524 ± 0.013 | 2.832 ± 0.102 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.126 ± 0.022 | 0.477 ± 0.014 | 5.502 ± 0.124 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.121 ± 0.024 | 0.478 ± 0.025 | 5.966 ± 0.564 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.131 ± 0.013 | 0.474 ± 0.077 | 6.469 ± 1.336 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.652 ± 0.077 | 0.053 ± 0.006 | 0.248 ± 0.036 | 5.418 ± 0.824 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.666 ± 0.016 | 0.049 ± 0.004 | 0.244 ± 0.036 | 4.887 ± 0.695 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.709 ± 0.098 | 0.048 ± 0.011 | 0.310 ± 0.046 | 5.428 ± 0.770 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.365 ± 0.104 | 0.408 ± 0.019 | 3.181 ± 0.028 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.295 ± 0.053 | 0.410 ± 0.031 | 2.966 ± 0.032 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.307 ± 0.046 | 0.425 ± 0.052 | 2.812 ± 0.036 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.644 ± 0.050 | 0.483 ± 0.042 | 7.013 ± 1.481 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.612 ± 0.016 | 0.438 ± 0.033 | 6.283 ± 0.939 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.576 ± 0.085 | 0.439 ± 0.060 | 6.738 ± 1.232 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.862 ± 0.060 | 0.143 ± 0.011 | 0.340 ± 0.051 | 5.874 ± 0.580 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.832 ± 0.043 | 0.138 ± 0.023 | 0.342 ± 0.044 | 5.760 ± 0.673 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.891 ± 0.030 | 0.142 ± 0.024 | 0.373 ± 0.033 | 5.825 ± 0.575 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.081 ± 0.017 | 0.689 ± 0.006 | 2.289 ± 0.033 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.076 ± 0.017 | 0.703 ± 0.104 | 2.465 ± 0.078 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.076 ± 0.011 | 0.758 ± 0.160 | 2.311 ± 0.037 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.103 ± 0.003 | 0.451 ± 0.052 | 4.618 ± 0.486 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.105 ± 0.011 | 0.457 ± 0.011 | 5.062 ± 0.181 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.104 ± 0.013 | 0.462 ± 0.054 | 4.527 ± 0.045 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 1.251 ± 0.040 | 0.088 ± 0.012 | 0.241 ± 0.038 | 5.536 ± 0.761 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 1.283 ± 0.069 | 0.100 ± 0.005 | 0.269 ± 0.027 | 5.472 ± 0.580 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 1.392 ± 0.034 | 0.112 ± 0.013 | 0.301 ± 0.040 | 5.335 ± 0.462 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.074 ± 0.002 | 0.431 ± 0.010 | 1.899 ± 0.078 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.081 ± 0.008 | 0.470 ± 0.021 | 2.062 ± 0.012 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.092 ± 0.004 | 0.587 ± 0.087 | 1.791 ± 0.049 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.080 ± 0.008 | 0.393 ± 0.065 | 5.046 ± 0.317 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.094 ± 0.018 | 0.365 ± 0.032 | 4.968 ± 0.391 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.109 ± 0.013 | 0.389 ± 0.054 | 4.995 ± 0.270 |
| small | `matmul` | f64 | 4 | `2x2` | 0.448 ± 0.082 | 0.005 ± 0.003 | 0.087 ± 0.004 | 0.353 ± 0.037 |
| small | `matmul` | f64 | 4 | `4x4` | 0.405 ± 0.026 | 0.005 ± 0.004 | 0.090 ± 0.001 | 0.765 ± 0.042 |
| small | `matmul` | f64 | 4 | `8x8` | 0.401 ± 0.044 | 0.008 ± 0.008 | 0.128 ± 0.001 | 1.166 ± 0.030 |
| small | `qr` | f64 | 4 | `2x2` | 0.439 ± 0.028 | 0.008 ± 0.000 | 0.091 ± 0.005 | 0.329 ± 0.072 |
| small | `qr` | f64 | 4 | `4x4` | 0.397 ± 0.037 | 0.013 ± 0.004 | 0.095 ± 0.024 | 0.590 ± 0.024 |
| small | `qr` | f64 | 4 | `8x8` | 0.410 ± 0.020 | 0.010 ± 0.001 | 0.115 ± 0.003 | 0.674 ± 0.019 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.479 ± 0.096 | 0.018 ± 0.000 | 0.173 ± 0.024 | 0.631 ± 0.012 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.452 ± 0.029 | 0.021 ± 0.001 | 0.168 ± 0.038 | 0.686 ± 0.017 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.481 ± 0.037 | 0.019 ± 0.001 | 0.176 ± 0.015 | 0.790 ± 0.033 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.469 ± 0.019 | 0.021 ± 0.009 | 0.171 ± 0.028 | 0.783 ± 0.022 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.472 ± 0.047 | 0.019 ± 0.001 | 0.221 ± 0.017 | 0.831 ± 0.009 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.493 ± 0.016 | 0.021 ± 0.001 | 0.229 ± 0.037 | 0.845 ± 0.011 |
| small | `svd` | f64 | 4 | `2x2` | 0.450 ± 0.011 | 0.013 ± 0.001 | 0.101 ± 0.014 | 0.279 ± 0.026 |
| small | `svd` | f64 | 4 | `4x4` | 0.421 ± 0.004 | 0.020 ± 0.001 | 0.108 ± 0.002 | 0.543 ± 0.019 |
| small | `svd` | f64 | 4 | `8x8` | 0.441 ± 0.020 | 0.034 ± 0.001 | 0.141 ± 0.002 | 0.686 ± 0.021 |

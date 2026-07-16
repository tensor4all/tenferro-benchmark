# Linux CPU Ops Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `linux-cpu`
- Thread runs: `1 4`

Mirrored from `amd-cpu` runs (`BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh <THREADS>`);
regenerate with `./scripts/mirror_linux_cpu_cpu_reports.sh 1:20260716_094347 4:20260716_100934`.

This report is generated from sequential CPU runs. Do not compare it with
measurements collected while another CPU benchmark process was running.

## Run Inputs

- Threads 1: timestamp `20260716_094347`, raw run `data/results/linux-cpu/cpu/einsum/20260716_094347`
- Threads 4: timestamp `20260716_100934`, raw run `data/results/linux-cpu/cpu/einsum/20260716_100934`

## Threads: 1

### CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `linux-cpu`
- Timestamp: `20260716_094347`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/linux-cpu/cpu/einsum/20260716_094347`.

- tenferro-rs commit: `3abc9108e4f0500f4e75519711cdb4e21b9625df`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-134-generic-x86_64-with-glibc2.39`

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

- tenferro-rs features: `system-openblas`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `openblas`
- BLAS version: `OpenBLAS 0.3.26`
- BLAS root: `/opt/openblas`
- BLAS library: `/opt/openblas/lib/libopenblas.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 1

- CSV: `data/results/linux-cpu/cpu/einsum/20260716_094347/cpu_ops_t1_20260716_094347.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260716_094347/cpu_ops_t1_20260716_094347.md`

#### CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.152 ± 0.011 | 0.025 ± 0.001 | 0.569 ± 0.009 | 1.972 ± 0.025 |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.218 ± 0.014 | 0.081 ± 0.007 | 1.914 ± 0.011 | 5.794 ± 0.052 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.188 ± 0.023 | 0.053 ± 0.000 | 0.711 ± 0.011 | 2.222 ± 0.053 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.357 ± 0.006 | 0.196 ± 0.017 | 2.466 ± 0.032 | 7.268 ± 0.099 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.132 ± 0.007 | 0.007 ± 0.000 | 0.154 ± 0.012 | 1.809 ± 0.010 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.147 ± 0.005 | 0.015 ± 0.000 | 0.297 ± 0.011 | 1.680 ± 0.031 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.136 ± 0.012 | 0.007 ± 0.000 | 0.295 ± 0.009 | 1.750 ± 0.021 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.146 ± 0.002 | 0.018 ± 0.000 | 0.827 ± 0.005 | 2.829 ± 0.719 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.143 ± 0.003 | 0.026 ± 0.007 | 0.312 ± 0.018 | 1.376 ± 0.018 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.202 ± 0.022 | 0.079 ± 0.011 | 1.030 ± 0.011 | 2.446 ± 0.039 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.167 ± 0.032 | 0.038 ± 0.000 | 0.416 ± 0.020 | 1.496 ± 0.029 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.253 ± 0.004 | 0.128 ± 0.006 | 1.424 ± 0.035 | 4.708 ± 0.099 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.165 ± 0.031 | 0.033 ± 0.000 | 0.624 ± 0.016 | 1.614 ± 0.018 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.234 ± 0.022 | 0.089 ± 0.024 | 1.975 ± 0.024 | 6.258 ± 0.299 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.169 ± 0.008 | 0.036 ± 0.002 | 0.756 ± 0.009 | 2.124 ± 0.011 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.244 ± 0.021 | 0.095 ± 0.001 | 2.503 ± 0.024 | 6.942 ± 0.139 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.184 ± 0.002 | 0.053 ± 0.000 | 0.362 ± 0.017 | 1.436 ± 0.136 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.318 ± 0.013 | 0.209 ± 0.022 | 1.144 ± 0.075 | 2.968 ± 0.094 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.247 ± 0.022 | 0.107 ± 0.012 | 0.536 ± 0.014 | 1.554 ± 0.048 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.604 ± 0.035 | 0.415 ± 0.021 | 1.750 ± 0.008 | 5.475 ± 0.094 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.388 ± 0.030 | 0.046 ± 0.002 | 0.368 ± 0.014 | 6.211 ± 1.755 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.405 ± 0.039 | 0.064 ± 0.001 | 0.537 ± 0.012 | 6.409 ± 1.910 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.381 ± 0.036 | 0.048 ± 0.005 | 0.541 ± 0.011 | 6.167 ± 2.107 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.420 ± 0.089 | 0.075 ± 0.010 | 1.130 ± 0.003 | 6.326 ± 2.759 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.542 ± 0.035 | 0.198 ± 0.018 | 0.813 ± 0.006 | 7.024 ± 1.602 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.644 ± 0.040 | 0.329 ± 0.040 | 2.204 ± 0.024 | 8.274 ± 3.863 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.557 ± 0.071 | 0.214 ± 0.010 | 0.935 ± 0.005 | 7.174 ± 1.862 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.669 ± 0.090 | 0.359 ± 0.056 | 2.702 ± 0.051 | 8.467 ± 3.783 |
| large | `eigh` | f64 | 1 | `64x64` | 1.121 ± 0.052 | 0.630 ± 0.022 | 2.103 ± 0.009 | 6.867 ± 0.098 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 14.280 ± 0.274 | 11.573 ± 0.397 | 35.877 ± 1.236 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 76.226 ± 0.436 | 69.828 ± 0.203 | 115.983 ± 1.476 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 13.937 ± 0.130 | 9.530 ± 0.007 | 35.141 ± 1.892 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 74.037 ± 1.242 | 52.394 ± 0.560 | 116.085 ± 3.599 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 8.419 ± 0.251 | 8.941 ± 0.009 | 12.832 ± 2.321 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 44.926 ± 0.911 | 65.959 ± 0.179 | 49.387 ± 1.920 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 8.097 ± 0.383 | 5.337 ± 0.172 | 15.129 ± 1.257 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 45.299 ± 0.705 | 37.280 ± 0.303 | 61.839 ± 4.102 |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | 0.205 ± 0.010 | 0.047 ± 0.009 | 3.000 ± 0.010 | 9.220 ± 1.295 |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | 0.453 ± 0.028 | 0.107 ± 0.006 | 3.074 ± 0.021 | 8.130 ± 2.239 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 11.081 ± 0.282 | 7.576 ± 0.010 | 19.334 ± 1.263 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 62.227 ± 1.787 | 50.236 ± 0.385 | 70.143 ± 12.857 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 11.533 ± 0.188 | 7.333 ± 0.115 | 20.592 ± 2.605 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 62.243 ± 0.452 | 49.184 ± 0.537 | 103.090 ± 2.237 |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | 0.839 ± 0.040 | 0.231 ± 0.029 | 1.932 ± 0.019 | 7.871 ± 3.953 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.542 ± 0.040 | 1.682 ± 0.008 | 5.131 ± 0.076 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 7.047 ± 0.182 | 5.558 ± 0.010 | 15.813 ± 0.649 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 1.194 ± 0.106 | 2.042 ± 0.048 | 6.203 ± 0.119 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 5.609 ± 0.047 | 8.945 ± 0.071 | 16.636 ± 0.288 |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | 2.026 ± 0.046 | 1.079 ± 0.053 | 2.554 ± 0.009 | 7.665 ± 2.458 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 22.532 ± 0.158 | 21.772 ± 0.061 | 61.665 ± 1.940 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 130.338 ± 0.497 | 138.929 ± 1.648 | 205.127 ± 2.514 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 22.353 ± 0.088 | 18.067 ± 0.345 | 61.415 ± 2.363 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 128.295 ± 1.332 | 112.433 ± 0.225 | 205.256 ± 4.399 |
| large | `matmul` | f64 | 1 | `128x128` | 0.596 ± 0.039 | 0.204 ± 0.030 | 11.245 ± 0.019 | 18.183 ± 0.687 |
| large | `matmul` | f64 | 1 | `256x256` | 2.091 ± 0.036 | 1.170 ± 0.017 | 44.438 ± 0.302 | 69.248 ± 1.365 |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | 6.053 ± 0.020 | 4.501 ± 0.013 | 180.056 ± 2.736 | 177.866 ± 4.210 |
| large | `qr` | f64 | 1 | `64x64` | 0.279 ± 0.017 | 0.136 ± 0.020 | 1.728 ± 0.008 | 5.363 ± 0.047 |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | 0.485 ± 0.038 | 0.064 ± 0.001 | 1.791 ± 0.017 | 5.744 ± 0.135 |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | 0.487 ± 0.019 | 0.076 ± 0.006 | 2.165 ± 0.009 | 6.743 ± 0.078 |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | 0.581 ± 0.008 | 0.153 ± 0.029 | 3.286 ± 0.012 | 9.927 ± 0.816 |
| large | `svd` | f64 | 1 | `64x64` | 1.175 ± 0.061 | 0.942 ± 0.023 | 2.442 ± 0.031 | 6.581 ± 1.245 |
| small | `eigh` | f64 | 1 | `2x2` | 0.132 ± 0.009 | 0.006 ± 0.000 | 0.106 ± 0.010 | 0.333 ± 0.012 |
| small | `eigh` | f64 | 1 | `4x4` | 0.138 ± 0.026 | 0.009 ± 0.000 | 0.114 ± 0.003 | 0.702 ± 0.028 |
| small | `eigh` | f64 | 1 | `8x8` | 0.169 ± 0.031 | 0.015 ± 0.000 | 0.141 ± 0.023 | 0.783 ± 0.012 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | 0.133 ± 0.002 | 0.003 ± 0.000 | 0.127 ± 0.017 | 0.558 ± 0.028 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | 0.149 ± 0.014 | 0.004 ± 0.000 | 0.124 ± 0.002 | 1.557 ± 0.016 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | 0.134 ± 0.016 | 0.004 ± 0.000 | 0.164 ± 0.014 | 1.801 ± 0.025 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.083 ± 0.003 | 0.329 ± 0.008 | 1.485 ± 0.027 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.092 ± 0.006 | 0.333 ± 0.008 | 1.600 ± 0.011 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.097 ± 0.002 | 0.344 ± 0.010 | 1.632 ± 0.030 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.094 ± 0.001 | 0.319 ± 0.012 | 3.674 ± 0.104 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.104 ± 0.011 | 0.329 ± 0.015 | 3.919 ± 0.191 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.112 ± 0.010 | 0.341 ± 0.013 | 4.887 ± 0.079 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.089 ± 0.019 | 0.454 ± 0.012 | 1.965 ± 0.067 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.086 ± 0.000 | 0.456 ± 0.012 | 2.065 ± 0.052 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.096 ± 0.008 | 0.456 ± 0.016 | 3.172 ± 0.031 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.091 ± 0.002 | 0.409 ± 0.012 | 5.331 ± 0.201 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.090 ± 0.010 | 0.411 ± 0.017 | 5.681 ± 0.286 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.100 ± 0.003 | 0.420 ± 0.012 | 7.069 ± 1.865 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | 0.391 ± 0.062 | 0.041 ± 0.001 | 0.252 ± 0.030 | 5.323 ± 0.576 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | 0.323 ± 0.012 | 0.043 ± 0.001 | 0.234 ± 0.009 | 5.197 ± 0.964 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | 0.412 ± 0.037 | 0.042 ± 0.001 | 0.294 ± 0.039 | 5.378 ± 0.919 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.218 ± 0.025 | 0.363 ± 0.012 | 3.150 ± 0.011 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.216 ± 0.033 | 0.360 ± 0.012 | 2.636 ± 0.069 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.218 ± 0.017 | 0.365 ± 0.008 | 2.857 ± 0.057 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.479 ± 0.043 | 0.391 ± 0.013 | 6.948 ± 1.388 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.490 ± 0.033 | 0.394 ± 0.012 | 6.547 ± 1.149 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.495 ± 0.037 | 0.396 ± 0.014 | 6.986 ± 1.350 |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | 0.521 ± 0.024 | 0.136 ± 0.007 | 0.347 ± 0.034 | 5.777 ± 0.543 |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | 0.490 ± 0.026 | 0.135 ± 0.013 | 0.331 ± 0.011 | 5.781 ± 0.582 |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | 0.567 ± 0.065 | 0.138 ± 0.013 | 0.392 ± 0.058 | 6.133 ± 0.703 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.057 ± 0.003 | 0.689 ± 0.007 | 2.298 ± 0.025 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.052 ± 0.001 | 0.689 ± 0.014 | 2.300 ± 0.024 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.063 ± 0.003 | 0.690 ± 0.010 | 2.644 ± 0.038 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.093 ± 0.011 | 0.450 ± 0.010 | 5.102 ± 0.148 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.094 ± 0.008 | 0.446 ± 0.012 | 5.099 ± 0.099 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.106 ± 0.010 | 0.456 ± 0.014 | 5.353 ± 0.245 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | 0.863 ± 0.013 | 0.064 ± 0.007 | 0.244 ± 0.020 | 5.513 ± 0.600 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | 0.813 ± 0.074 | 0.069 ± 0.007 | 0.247 ± 0.005 | 5.526 ± 0.650 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | 0.998 ± 0.035 | 0.078 ± 0.005 | 0.296 ± 0.025 | 5.676 ± 0.863 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.056 ± 0.010 | 0.472 ± 0.042 | 1.981 ± 0.063 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.056 ± 0.002 | 0.443 ± 0.008 | 2.003 ± 0.004 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.071 ± 0.009 | 0.483 ± 0.014 | 2.116 ± 0.064 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.061 ± 0.005 | 0.350 ± 0.021 | 4.960 ± 0.205 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.065 ± 0.001 | 0.351 ± 0.008 | 5.034 ± 0.350 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.078 ± 0.001 | 0.375 ± 0.076 | 5.229 ± 0.310 |
| small | `matmul` | f64 | 1 | `2x2` | 0.142 ± 0.022 | 0.004 ± 0.000 | 0.088 ± 0.034 | 0.346 ± 0.045 |
| small | `matmul` | f64 | 1 | `4x4` | 0.153 ± 0.013 | 0.003 ± 0.000 | 0.086 ± 0.002 | 0.765 ± 0.022 |
| small | `matmul` | f64 | 1 | `8x8` | 0.131 ± 0.007 | 0.004 ± 0.000 | 0.124 ± 0.013 | 1.150 ± 0.020 |
| small | `qr` | f64 | 1 | `2x2` | 0.123 ± 0.004 | 0.006 ± 0.000 | 0.076 ± 0.010 | 0.299 ± 0.042 |
| small | `qr` | f64 | 1 | `4x4` | 0.130 ± 0.009 | 0.007 ± 0.002 | 0.078 ± 0.004 | 0.619 ± 0.032 |
| small | `qr` | f64 | 1 | `8x8` | 0.153 ± 0.022 | 0.008 ± 0.000 | 0.100 ± 0.066 | 0.698 ± 0.031 |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | 0.136 ± 0.001 | 0.013 ± 0.001 | 0.168 ± 0.023 | 0.561 ± 0.014 |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | 0.136 ± 0.005 | 0.014 ± 0.002 | 0.165 ± 0.023 | 0.681 ± 0.022 |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | 0.137 ± 0.009 | 0.014 ± 0.002 | 0.161 ± 0.003 | 0.758 ± 0.019 |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | 0.140 ± 0.002 | 0.014 ± 0.000 | 0.165 ± 0.003 | 0.741 ± 0.018 |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | 0.182 ± 0.016 | 0.016 ± 0.000 | 0.201 ± 0.042 | 0.834 ± 0.014 |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | 0.171 ± 0.002 | 0.017 ± 0.000 | 0.194 ± 0.032 | 0.867 ± 0.032 |
| small | `svd` | f64 | 1 | `2x2` | 0.140 ± 0.007 | 0.008 ± 0.001 | 0.104 ± 0.022 | 0.264 ± 0.032 |
| small | `svd` | f64 | 1 | `4x4` | 0.149 ± 0.022 | 0.012 ± 0.000 | 0.106 ± 0.003 | 0.662 ± 0.033 |
| small | `svd` | f64 | 1 | `8x8` | 0.183 ± 0.029 | 0.022 ± 0.001 | 0.143 ± 0.011 | 0.683 ± 0.018 |

## Threads: 4

### CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `linux-cpu`
- Timestamp: `20260716_100934`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/linux-cpu/cpu/einsum/20260716_100934`.

- tenferro-rs commit: `3abc9108e4f0500f4e75519711cdb4e21b9625df`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-134-generic-x86_64-with-glibc2.39`

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

- tenferro-rs features: `system-openblas`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `openblas`
- BLAS version: `OpenBLAS 0.3.26`
- BLAS root: `/opt/openblas`
- BLAS library: `/opt/openblas/lib/libopenblas.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 4

- CSV: `data/results/linux-cpu/cpu/einsum/20260716_100934/cpu_ops_t4_20260716_100934.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260716_100934/cpu_ops_t4_20260716_100934.md`

#### CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.387 ± 0.006 | 0.030 ± 0.000 | 0.628 ± 0.018 | 1.938 ± 0.021 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.518 ± 0.026 | 0.092 ± 0.012 | 2.174 ± 0.012 | 5.696 ± 0.053 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.479 ± 0.031 | 0.076 ± 0.009 | 0.766 ± 0.011 | 2.140 ± 0.053 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.698 ± 0.021 | 0.218 ± 0.037 | 2.720 ± 0.009 | 7.226 ± 0.119 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.383 ± 0.030 | 0.008 ± 0.003 | 0.167 ± 0.031 | 1.552 ± 0.024 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.429 ± 0.004 | 0.018 ± 0.000 | 0.301 ± 0.025 | 2.074 ± 0.507 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.457 ± 0.040 | 0.008 ± 0.001 | 0.305 ± 0.028 | 2.236 ± 0.187 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.438 ± 0.010 | 0.021 ± 0.000 | 0.818 ± 0.029 | 3.333 ± 0.036 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.382 ± 0.047 | 0.030 ± 0.000 | 0.387 ± 0.021 | 1.301 ± 0.033 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.505 ± 0.023 | 0.082 ± 0.016 | 1.263 ± 0.040 | 2.597 ± 0.101 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.509 ± 0.021 | 0.047 ± 0.001 | 0.500 ± 0.053 | 1.372 ± 0.034 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.579 ± 0.069 | 0.130 ± 0.023 | 1.702 ± 0.044 | 4.525 ± 0.032 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.407 ± 0.042 | 0.040 ± 0.001 | 0.764 ± 0.036 | 2.061 ± 0.035 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.521 ± 0.013 | 0.095 ± 0.015 | 2.506 ± 0.088 | 5.925 ± 0.184 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.458 ± 0.010 | 0.044 ± 0.000 | 0.874 ± 0.008 | 2.291 ± 0.037 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.624 ± 0.073 | 0.108 ± 0.017 | 2.979 ± 0.066 | 7.161 ± 0.305 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.424 ± 0.014 | 0.063 ± 0.006 | 0.397 ± 0.014 | 1.186 ± 0.027 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.654 ± 0.020 | 0.230 ± 0.017 | 1.259 ± 0.015 | 2.895 ± 0.066 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.652 ± 0.020 | 0.135 ± 0.003 | 0.565 ± 0.020 | 1.531 ± 0.049 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 1.008 ± 0.033 | 0.486 ± 0.018 | 1.896 ± 0.018 | 5.373 ± 0.151 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.638 ± 0.020 | 0.057 ± 0.004 | 0.397 ± 0.067 | 6.211 ± 1.507 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.791 ± 0.030 | 0.092 ± 0.011 | 0.545 ± 0.032 | 6.365 ± 1.832 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.727 ± 0.025 | 0.060 ± 0.001 | 0.545 ± 0.010 | 6.071 ± 2.036 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.786 ± 0.066 | 0.095 ± 0.003 | 1.132 ± 0.015 | 6.471 ± 2.654 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.842 ± 0.015 | 0.245 ± 0.008 | 0.972 ± 0.031 | 7.124 ± 1.377 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 1.141 ± 0.091 | 0.423 ± 0.014 | 2.788 ± 0.113 | 8.484 ± 3.880 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.966 ± 0.036 | 0.257 ± 0.011 | 1.094 ± 0.043 | 7.163 ± 1.755 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 1.179 ± 0.029 | 0.456 ± 0.009 | 3.243 ± 0.241 | 8.603 ± 3.783 |
| large | `eigh` | f64 | 4 | `64x64` | 1.488 ± 0.069 | 0.727 ± 0.027 | 2.254 ± 0.022 | 6.839 ± 0.020 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 13.875 ± 0.207 | 9.330 ± 0.425 | 29.603 ± 0.340 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 55.186 ± 0.238 | 27.444 ± 0.319 | 88.491 ± 1.155 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 13.857 ± 0.361 | 6.539 ± 0.329 | 28.735 ± 0.296 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 53.473 ± 0.418 | 20.419 ± 0.131 | 87.502 ± 1.230 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 6.181 ± 0.285 | 4.432 ± 0.038 | 8.390 ± 0.735 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 25.755 ± 0.780 | 24.032 ± 0.094 | 21.056 ± 2.355 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 5.773 ± 0.110 | 2.834 ± 0.045 | 10.279 ± 1.208 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 24.647 ± 0.290 | 11.612 ± 0.425 | 26.781 ± 1.371 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.464 ± 0.009 | 0.059 ± 0.004 | 3.357 ± 0.067 | 8.719 ± 1.558 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.749 ± 0.021 | 0.134 ± 0.004 | 3.454 ± 0.060 | 7.916 ± 2.799 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 9.070 ± 0.300 | 4.484 ± 0.032 | 13.436 ± 1.250 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 35.758 ± 1.085 | 18.545 ± 0.116 | 45.213 ± 2.752 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 9.125 ± 0.117 | 4.313 ± 0.028 | 14.295 ± 1.386 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 34.135 ± 1.079 | 18.294 ± 0.110 | 59.749 ± 4.136 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 1.172 ± 0.044 | 0.278 ± 0.004 | 1.936 ± 0.022 | 7.076 ± 3.168 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 1.692 ± 0.022 | 1.578 ± 0.032 | 4.147 ± 0.269 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 6.231 ± 0.087 | 3.268 ± 0.135 | 10.265 ± 0.117 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 1.234 ± 0.040 | 1.678 ± 0.027 | 5.500 ± 0.159 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 4.431 ± 0.077 | 4.675 ± 0.098 | 10.894 ± 0.218 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 2.764 ± 0.221 | 1.417 ± 0.032 | 2.801 ± 0.024 | 9.372 ± 1.697 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 20.825 ± 0.049 | 17.897 ± 0.271 | 52.259 ± 0.724 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 96.337 ± 0.696 | 76.321 ± 0.480 | 162.473 ± 4.459 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 21.611 ± 0.133 | 16.049 ± 0.158 | 50.482 ± 1.542 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 95.077 ± 0.239 | 62.869 ± 0.170 | 161.060 ± 4.606 |
| large | `matmul` | f64 | 4 | `128x128` | 0.473 ± 0.002 | 0.158 ± 0.006 | 11.649 ± 0.050 | 18.034 ± 0.134 |
| large | `matmul` | f64 | 4 | `256x256` | 1.930 ± 0.037 | 0.681 ± 0.036 | 45.179 ± 0.200 | 67.827 ± 0.498 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 4.522 ± 0.018 | 2.644 ± 0.052 | 178.669 ± 2.793 | 178.058 ± 2.753 |
| large | `qr` | f64 | 4 | `64x64` | 0.579 ± 0.026 | 0.167 ± 0.024 | 1.953 ± 0.020 | 5.253 ± 0.083 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.786 ± 0.012 | 0.069 ± 0.018 | 1.791 ± 0.006 | 5.639 ± 0.135 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.801 ± 0.015 | 0.085 ± 0.002 | 2.395 ± 0.016 | 6.490 ± 0.080 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.830 ± 0.013 | 0.144 ± 0.032 | 3.841 ± 0.162 | 8.930 ± 1.845 |
| large | `svd` | f64 | 4 | `64x64` | 1.678 ± 0.007 | 1.147 ± 0.037 | 2.711 ± 0.041 | 6.904 ± 1.428 |
| small | `eigh` | f64 | 4 | `2x2` | 0.368 ± 0.042 | 0.007 ± 0.000 | 0.111 ± 0.024 | 0.369 ± 0.015 |
| small | `eigh` | f64 | 4 | `4x4` | 0.368 ± 0.050 | 0.011 ± 0.000 | 0.124 ± 0.025 | 0.720 ± 0.014 |
| small | `eigh` | f64 | 4 | `8x8` | 0.402 ± 0.034 | 0.018 ± 0.000 | 0.153 ± 0.003 | 0.798 ± 0.007 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.391 ± 0.016 | 0.004 ± 0.000 | 0.128 ± 0.018 | 0.597 ± 0.024 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.377 ± 0.019 | 0.004 ± 0.002 | 0.128 ± 0.014 | 1.532 ± 0.032 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.375 ± 0.046 | 0.005 ± 0.004 | 0.169 ± 0.004 | 1.813 ± 0.037 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.112 ± 0.003 | 0.334 ± 0.026 | 1.477 ± 0.027 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.116 ± 0.005 | 0.360 ± 0.013 | 1.355 ± 0.018 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.125 ± 0.007 | 0.371 ± 0.013 | 1.542 ± 0.018 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.121 ± 0.003 | 0.323 ± 0.015 | 3.551 ± 0.065 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.127 ± 0.007 | 0.334 ± 0.012 | 3.903 ± 0.094 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.135 ± 0.009 | 0.342 ± 0.015 | 4.721 ± 0.133 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.117 ± 0.004 | 0.524 ± 0.025 | 2.015 ± 0.009 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.117 ± 0.003 | 0.517 ± 0.028 | 2.056 ± 0.057 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.122 ± 0.007 | 0.522 ± 0.014 | 3.105 ± 0.097 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.120 ± 0.003 | 0.498 ± 0.077 | 5.795 ± 0.400 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.121 ± 0.004 | 0.464 ± 0.031 | 5.730 ± 0.345 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.128 ± 0.002 | 0.464 ± 0.020 | 6.297 ± 2.059 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.622 ± 0.061 | 0.073 ± 0.012 | 0.229 ± 0.011 | 5.404 ± 0.709 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.617 ± 0.027 | 0.048 ± 0.001 | 0.251 ± 0.037 | 5.265 ± 0.830 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.629 ± 0.019 | 0.050 ± 0.004 | 0.279 ± 0.040 | 4.268 ± 1.043 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.316 ± 0.126 | 0.401 ± 0.014 | 3.176 ± 0.037 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.280 ± 0.009 | 0.397 ± 0.012 | 2.581 ± 0.107 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.283 ± 0.013 | 0.406 ± 0.011 | 2.578 ± 0.033 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.594 ± 0.012 | 0.437 ± 0.018 | 6.989 ± 1.332 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.531 ± 0.061 | 0.436 ± 0.021 | 6.452 ± 1.245 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.592 ± 0.019 | 0.438 ± 0.042 | 5.256 ± 1.604 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.759 ± 0.022 | 0.246 ± 0.047 | 0.335 ± 0.057 | 5.865 ± 0.591 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.774 ± 0.018 | 0.168 ± 0.008 | 0.355 ± 0.034 | 5.913 ± 0.665 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.753 ± 0.020 | 0.173 ± 0.003 | 0.360 ± 0.012 | 5.670 ± 0.890 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.073 ± 0.006 | 0.722 ± 0.049 | 2.386 ± 0.065 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.073 ± 0.006 | 0.688 ± 0.083 | 2.341 ± 0.038 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.073 ± 0.007 | 0.696 ± 0.060 | 2.584 ± 0.031 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.116 ± 0.008 | 0.464 ± 0.052 | 5.132 ± 0.122 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.116 ± 0.009 | 0.454 ± 0.011 | 4.288 ± 0.855 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.116 ± 0.004 | 0.458 ± 0.014 | 5.177 ± 0.154 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 1.179 ± 0.050 | 0.101 ± 0.004 | 0.236 ± 0.016 | 5.533 ± 0.723 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 1.171 ± 0.057 | 0.083 ± 0.006 | 0.254 ± 0.028 | 5.596 ± 0.635 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 1.201 ± 0.022 | 0.097 ± 0.003 | 0.289 ± 0.011 | 5.709 ± 0.670 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.093 ± 0.006 | 0.433 ± 0.011 | 1.970 ± 0.021 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.073 ± 0.002 | 0.443 ± 0.021 | 1.899 ± 0.031 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.089 ± 0.003 | 0.457 ± 0.012 | 1.800 ± 0.020 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.105 ± 0.004 | 0.349 ± 0.029 | 4.992 ± 0.233 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.083 ± 0.005 | 0.351 ± 0.010 | 5.081 ± 0.210 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.098 ± 0.003 | 0.379 ± 0.015 | 4.721 ± 0.896 |
| small | `matmul` | f64 | 4 | `2x2` | 0.413 ± 0.059 | 0.004 ± 0.000 | 0.085 ± 0.003 | 0.361 ± 0.042 |
| small | `matmul` | f64 | 4 | `4x4` | 0.392 ± 0.006 | 0.004 ± 0.001 | 0.089 ± 0.010 | 0.817 ± 0.028 |
| small | `matmul` | f64 | 4 | `8x8` | 0.384 ± 0.018 | 0.005 ± 0.001 | 0.127 ± 0.003 | 1.139 ± 0.033 |
| small | `qr` | f64 | 4 | `2x2` | 0.362 ± 0.019 | 0.007 ± 0.000 | 0.090 ± 0.003 | 0.248 ± 0.017 |
| small | `qr` | f64 | 4 | `4x4` | 0.376 ± 0.021 | 0.008 ± 0.001 | 0.096 ± 0.035 | 0.610 ± 0.017 |
| small | `qr` | f64 | 4 | `8x8` | 0.386 ± 0.029 | 0.010 ± 0.001 | 0.116 ± 0.008 | 0.585 ± 0.023 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.370 ± 0.032 | 0.016 ± 0.001 | 0.168 ± 0.029 | 0.683 ± 0.012 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.385 ± 0.042 | 0.020 ± 0.000 | 0.158 ± 0.012 | 0.678 ± 0.018 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.373 ± 0.021 | 0.018 ± 0.001 | 0.183 ± 0.024 | 0.874 ± 0.013 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.390 ± 0.045 | 0.020 ± 0.001 | 0.171 ± 0.033 | 0.870 ± 0.030 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.399 ± 0.005 | 0.019 ± 0.000 | 0.203 ± 0.020 | 0.762 ± 0.015 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.388 ± 0.003 | 0.023 ± 0.000 | 0.198 ± 0.012 | 0.781 ± 0.011 |
| small | `svd` | f64 | 4 | `2x2` | 0.399 ± 0.024 | 0.010 ± 0.000 | 0.100 ± 0.003 | 0.231 ± 0.009 |
| small | `svd` | f64 | 4 | `4x4` | 0.400 ± 0.013 | 0.015 ± 0.001 | 0.110 ± 0.025 | 0.570 ± 0.024 |
| small | `svd` | f64 | 4 | `8x8` | 0.407 ± 0.006 | 0.027 ± 0.000 | 0.144 ± 0.002 | 0.701 ± 0.023 |

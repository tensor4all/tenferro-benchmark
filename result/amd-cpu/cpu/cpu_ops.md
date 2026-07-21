# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Timestamp: `20260721_071519`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20260721_071519`.

- tenferro-rs commit: `85855e272b1495611deb601a9ee06f3546772c3c`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`

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

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.0`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- CSV: `data/results/amd-cpu/cpu/einsum/20260721_071519/cpu_ops_t4_20260721_071519.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20260721_071519/cpu_ops_t4_20260721_071519.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.428 ± 0.039 | 0.043 ± 0.000 | 0.586 ± 0.017 | 1.901 ± 0.023 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.569 ± 0.020 | 0.138 ± 0.025 | 1.990 ± 0.672 | 5.360 ± 0.089 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.479 ± 0.026 | 0.084 ± 0.000 | 0.723 ± 0.012 | 2.100 ± 0.051 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 1.000 ± 0.111 | 0.302 ± 0.054 | 2.531 ± 0.011 | 6.666 ± 0.129 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.424 ± 0.023 | 0.010 ± 0.003 | 0.182 ± 0.026 | 1.566 ± 0.021 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.371 ± 0.035 | 0.025 ± 0.000 | 0.324 ± 0.052 | 1.930 ± 0.219 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.383 ± 0.051 | 0.010 ± 0.000 | 0.339 ± 0.065 | 2.232 ± 0.276 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.415 ± 0.218 | 0.026 ± 0.000 | 0.827 ± 0.003 | 2.928 ± 0.068 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.414 ± 0.030 | 0.030 ± 0.000 | 0.375 ± 0.068 | 1.309 ± 0.018 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.480 ± 0.011 | 0.090 ± 0.000 | 1.221 ± 0.166 | 2.421 ± 0.058 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.436 ± 0.032 | 0.040 ± 0.000 | 0.486 ± 0.031 | 1.373 ± 0.020 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.691 ± 0.144 | 0.132 ± 0.010 | 1.645 ± 0.030 | 4.405 ± 0.102 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.455 ± 0.011 | 0.041 ± 0.001 | 0.716 ± 0.032 | 1.923 ± 0.009 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.538 ± 0.010 | 0.108 ± 0.009 | 2.269 ± 0.076 | 5.734 ± 0.229 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.503 ± 0.032 | 0.044 ± 0.000 | 0.829 ± 0.029 | 2.252 ± 0.021 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.587 ± 0.016 | 0.112 ± 0.026 | 2.795 ± 0.157 | 6.778 ± 0.237 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.450 ± 0.034 | 0.067 ± 0.003 | 0.429 ± 0.086 | 1.171 ± 0.041 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.622 ± 0.023 | 0.233 ± 0.015 | 1.209 ± 0.024 | 2.688 ± 0.050 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.565 ± 0.042 | 0.163 ± 0.004 | 0.547 ± 0.011 | 1.473 ± 0.013 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 1.726 ± 0.451 | 0.573 ± 0.068 | 1.821 ± 0.056 | 5.159 ± 0.051 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.653 ± 0.091 | 0.063 ± 0.012 | 0.376 ± 0.009 | 6.245 ± 1.589 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.750 ± 0.038 | 0.105 ± 0.003 | 0.546 ± 0.014 | 6.378 ± 1.837 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.664 ± 0.023 | 0.064 ± 0.008 | 0.549 ± 0.064 | 5.996 ± 1.883 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.706 ± 0.028 | 0.107 ± 0.011 | 1.129 ± 0.025 | 6.418 ± 2.670 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.922 ± 0.045 | 0.221 ± 0.008 | 0.932 ± 0.147 | 6.934 ± 1.407 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 1.030 ± 0.038 | 0.381 ± 0.005 | 2.583 ± 0.063 | 7.923 ± 3.629 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.896 ± 0.059 | 0.220 ± 0.007 | 1.079 ± 0.033 | 6.037 ± 2.174 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.995 ± 0.047 | 0.402 ± 0.002 | 3.065 ± 0.063 | 8.337 ± 2.966 |
| large | `eigh` | f64 | 4 | `64x64` | 1.279 ± 0.034 | 0.410 ± 0.017 | 2.365 ± 0.092 | 6.792 ± 0.058 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 8.282 ± 0.160 | 8.182 ± 0.193 | 29.690 ± 0.475 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 30.525 ± 0.865 | 28.267 ± 0.312 | 88.789 ± 5.303 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 7.637 ± 0.146 | 6.521 ± 0.412 | 28.563 ± 1.523 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 29.283 ± 0.301 | 21.345 ± 0.344 | 88.646 ± 3.305 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 5.272 ± 0.285 | 4.531 ± 0.157 | 7.324 ± 1.681 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 23.892 ± 0.734 | 24.787 ± 0.528 | 23.799 ± 3.082 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 4.864 ± 0.192 | 2.831 ± 0.135 | 10.714 ± 1.535 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 22.522 ± 0.251 | 11.949 ± 0.296 | 30.937 ± 5.709 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.484 ± 0.017 | 0.038 ± 0.012 | 3.428 ± 0.042 | 6.150 ± 0.381 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.815 ± 0.115 | 0.110 ± 0.003 | 3.477 ± 0.155 | 7.982 ± 2.240 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 7.036 ± 0.282 | 4.608 ± 0.148 | 13.969 ± 0.535 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 30.797 ± 1.169 | 19.190 ± 0.252 | 45.298 ± 2.553 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 6.922 ± 0.567 | 4.363 ± 0.144 | 14.030 ± 0.576 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 28.912 ± 0.887 | 18.894 ± 0.343 | 60.881 ± 2.649 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 1.338 ± 0.025 | 0.229 ± 0.007 | 1.958 ± 0.022 | 7.650 ± 2.463 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 1.431 ± 0.081 | 1.591 ± 0.059 | 4.110 ± 0.063 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 5.439 ± 0.142 | 3.263 ± 0.152 | 10.390 ± 0.165 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.943 ± 0.082 | 1.724 ± 0.101 | 5.608 ± 0.187 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 3.246 ± 0.078 | 4.654 ± 0.188 | 11.262 ± 0.227 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 2.704 ± 0.173 | 1.110 ± 0.035 | 2.804 ± 0.058 | 7.359 ± 0.529 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 17.299 ± 0.165 | 18.330 ± 0.410 | 52.039 ± 0.829 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 70.598 ± 0.539 | 81.509 ± 4.477 | 163.776 ± 3.663 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 16.870 ± 0.145 | 16.529 ± 0.290 | 50.427 ± 0.733 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 69.497 ± 0.399 | 64.075 ± 0.696 | 161.626 ± 2.779 |
| large | `matmul` | f64 | 4 | `128x128` | 0.625 ± 0.026 | 0.189 ± 0.020 | 12.134 ± 0.844 | 18.217 ± 0.657 |
| large | `matmul` | f64 | 4 | `256x256` | 1.873 ± 0.028 | 0.650 ± 0.025 | 46.294 ± 0.732 | 68.003 ± 23.556 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 4.546 ± 0.096 | 2.598 ± 0.155 | 183.358 ± 2.994 | 184.613 ± 2.776 |
| large | `qr` | f64 | 4 | `64x64` | 0.628 ± 0.026 | 0.147 ± 0.008 | 2.071 ± 0.064 | 5.250 ± 0.111 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.959 ± 0.014 | 0.058 ± 0.003 | 1.800 ± 0.041 | 5.540 ± 0.099 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.956 ± 0.035 | 0.083 ± 0.003 | 2.425 ± 0.047 | 4.733 ± 0.073 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.986 ± 0.019 | 0.142 ± 0.005 | 3.796 ± 0.046 | 6.679 ± 0.352 |
| large | `svd` | f64 | 4 | `64x64` | 1.440 ± 0.060 | 1.002 ± 0.026 | 2.784 ± 0.058 | 6.843 ± 1.586 |
| small | `eigh` | f64 | 4 | `2x2` | 0.403 ± 0.021 | 0.009 ± 0.001 | 0.121 ± 0.033 | 0.390 ± 0.018 |
| small | `eigh` | f64 | 4 | `4x4` | 0.409 ± 0.047 | 0.014 ± 0.000 | 0.123 ± 0.003 | 0.717 ± 0.034 |
| small | `eigh` | f64 | 4 | `8x8` | 0.416 ± 0.021 | 0.020 ± 0.001 | 0.153 ± 0.001 | 0.801 ± 0.017 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.413 ± 0.018 | 0.005 ± 0.001 | 0.145 ± 0.026 | 0.560 ± 0.026 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.381 ± 0.015 | 0.005 ± 0.001 | 0.126 ± 0.005 | 1.417 ± 0.015 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.369 ± 0.031 | 0.005 ± 0.005 | 0.171 ± 0.010 | 1.812 ± 0.006 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.105 ± 0.021 | 0.338 ± 0.055 | 1.549 ± 0.023 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.125 ± 0.020 | 0.359 ± 0.020 | 1.591 ± 0.125 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.144 ± 0.008 | 0.378 ± 0.012 | 1.801 ± 0.049 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.124 ± 0.018 | 0.325 ± 0.014 | 3.505 ± 0.097 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.132 ± 0.018 | 0.331 ± 0.012 | 3.566 ± 0.358 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.144 ± 0.030 | 0.343 ± 0.019 | 4.753 ± 0.162 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.113 ± 0.016 | 0.523 ± 0.013 | 2.024 ± 0.055 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.111 ± 0.012 | 0.520 ± 0.014 | 1.829 ± 0.020 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.131 ± 0.023 | 0.541 ± 0.118 | 3.108 ± 0.051 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.126 ± 0.015 | 0.495 ± 0.049 | 5.214 ± 1.037 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.121 ± 0.007 | 0.485 ± 0.039 | 5.200 ± 0.996 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.128 ± 0.011 | 0.471 ± 0.060 | 5.607 ± 1.994 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.697 ± 0.047 | 0.053 ± 0.008 | 0.232 ± 0.017 | 5.252 ± 0.657 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.640 ± 0.009 | 0.048 ± 0.016 | 0.256 ± 0.034 | 5.305 ± 1.194 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.639 ± 0.036 | 0.054 ± 0.021 | 0.281 ± 0.008 | 5.289 ± 0.976 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.296 ± 0.032 | 0.426 ± 0.030 | 3.184 ± 0.026 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.293 ± 0.014 | 0.416 ± 0.026 | 2.714 ± 0.083 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.292 ± 0.039 | 0.408 ± 0.020 | 2.686 ± 0.073 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.582 ± 0.025 | 0.467 ± 0.048 | 6.398 ± 1.807 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.579 ± 0.014 | 0.432 ± 0.019 | 6.596 ± 1.134 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.579 ± 0.034 | 0.440 ± 0.012 | 6.953 ± 1.089 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.847 ± 0.087 | 0.141 ± 0.033 | 0.336 ± 0.048 | 5.873 ± 0.639 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.837 ± 0.083 | 0.145 ± 0.016 | 0.380 ± 0.113 | 4.363 ± 1.030 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.855 ± 0.079 | 0.141 ± 0.012 | 0.367 ± 0.011 | 5.442 ± 1.327 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.078 ± 0.005 | 0.696 ± 0.019 | 2.348 ± 0.068 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.076 ± 0.006 | 0.697 ± 0.058 | 2.269 ± 0.071 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.071 ± 0.007 | 0.758 ± 0.075 | 2.368 ± 0.023 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.103 ± 0.012 | 0.451 ± 0.014 | 4.571 ± 1.193 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.100 ± 0.012 | 0.453 ± 0.011 | 5.176 ± 0.575 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.103 ± 0.003 | 0.491 ± 0.017 | 5.206 ± 0.289 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 1.236 ± 0.051 | 0.087 ± 0.018 | 0.236 ± 0.031 | 5.525 ± 0.673 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 1.279 ± 0.171 | 0.098 ± 0.008 | 0.265 ± 0.024 | 5.624 ± 0.793 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 1.278 ± 0.042 | 0.118 ± 0.016 | 0.303 ± 0.030 | 5.957 ± 0.882 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.069 ± 0.003 | 0.438 ± 0.010 | 1.871 ± 0.064 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.085 ± 0.008 | 0.490 ± 0.052 | 1.884 ± 0.061 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.097 ± 0.011 | 0.466 ± 0.014 | 1.909 ± 0.055 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.087 ± 0.018 | 0.360 ± 0.061 | 4.036 ± 0.937 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.092 ± 0.006 | 0.350 ± 0.013 | 5.083 ± 0.392 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.109 ± 0.013 | 0.374 ± 0.013 | 4.634 ± 0.959 |
| small | `matmul` | f64 | 4 | `2x2` | 0.464 ± 0.091 | 0.005 ± 0.001 | 0.096 ± 0.024 | 0.349 ± 0.036 |
| small | `matmul` | f64 | 4 | `4x4` | 0.393 ± 0.061 | 0.005 ± 0.003 | 0.092 ± 0.001 | 0.758 ± 0.012 |
| small | `matmul` | f64 | 4 | `8x8` | 0.369 ± 0.036 | 0.005 ± 0.001 | 0.131 ± 0.003 | 1.181 ± 0.020 |
| small | `qr` | f64 | 4 | `2x2` | 0.436 ± 0.026 | 0.008 ± 0.000 | 0.089 ± 0.015 | 0.255 ± 0.004 |
| small | `qr` | f64 | 4 | `4x4` | 0.399 ± 0.038 | 0.009 ± 0.001 | 0.095 ± 0.020 | 0.568 ± 0.016 |
| small | `qr` | f64 | 4 | `8x8` | 0.385 ± 0.017 | 0.011 ± 0.001 | 0.116 ± 0.007 | 0.674 ± 0.022 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.482 ± 0.136 | 0.018 ± 0.001 | 0.168 ± 0.013 | 0.630 ± 0.017 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.499 ± 0.042 | 0.020 ± 0.001 | 0.163 ± 0.021 | 0.651 ± 0.028 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.436 ± 0.029 | 0.019 ± 0.001 | 0.182 ± 0.024 | 0.838 ± 0.020 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.439 ± 0.009 | 0.021 ± 0.004 | 0.175 ± 0.012 | 0.844 ± 0.013 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.464 ± 0.043 | 0.019 ± 0.001 | 0.207 ± 0.021 | 0.938 ± 0.011 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.438 ± 0.052 | 0.021 ± 0.000 | 0.202 ± 0.003 | 0.872 ± 0.007 |
| small | `svd` | f64 | 4 | `2x2` | 0.418 ± 0.030 | 0.013 ± 0.001 | 0.103 ± 0.002 | 0.280 ± 0.031 |
| small | `svd` | f64 | 4 | `4x4` | 0.411 ± 0.019 | 0.019 ± 0.001 | 0.111 ± 0.003 | 0.543 ± 0.018 |
| small | `svd` | f64 | 4 | `8x8` | 0.421 ± 0.021 | 0.034 ± 0.006 | 0.144 ± 0.004 | 0.679 ± 0.021 |

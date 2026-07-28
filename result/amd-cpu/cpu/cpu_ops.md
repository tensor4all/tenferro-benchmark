# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Timestamp: `20260728_023431`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20260728_023431`.

- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

## CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

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
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- CSV: `data/results/amd-cpu/cpu/einsum/20260728_023431/cpu_ops_t4_20260728_023431.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20260728_023431/cpu_ops_t4_20260728_023431.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.671 ± 0.274 | 0.063 ± 0.004 | 0.314 ± 0.009 | 0.810 ± 0.022 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.584 ± 0.023 | 0.131 ± 0.005 | 1.004 ± 0.168 | 1.122 ± 0.126 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.596 ± 0.036 | 0.086 ± 0.006 | 0.415 ± 0.013 | 0.544 ± 0.016 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.743 ± 0.047 | 0.232 ± 0.012 | 1.197 ± 0.009 | 1.511 ± 0.266 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.760 ± 0.439 | 0.034 ± 0.013 | 0.096 ± 0.003 | 0.945 ± 0.013 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.465 ± 0.033 | 0.096 ± 0.086 | 0.200 ± 0.006 | 0.608 ± 0.151 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.492 ± 0.045 | 0.060 ± 0.036 | 0.205 ± 0.006 | 0.598 ± 0.042 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.514 ± 0.027 | 0.128 ± 0.001 | 0.609 ± 0.099 | 0.898 ± 0.106 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.618 ± 0.271 | 0.055 ± 0.004 | 0.196 ± 0.008 | 0.588 ± 0.022 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.568 ± 0.018 | 0.101 ± 0.004 | 0.647 ± 0.011 | 0.653 ± 0.010 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.584 ± 0.038 | 0.061 ± 0.008 | 0.276 ± 0.007 | 0.399 ± 0.010 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.663 ± 0.033 | 0.129 ± 0.011 | 0.793 ± 0.077 | 1.262 ± 0.851 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.727 ± 0.236 | 0.129 ± 0.007 | 0.352 ± 0.012 | 0.547 ± 0.011 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.745 ± 0.027 | 0.178 ± 0.018 | 0.915 ± 0.009 | 1.588 ± 0.274 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.693 ± 0.089 | 0.132 ± 0.049 | 0.438 ± 0.011 | 0.639 ± 0.015 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.795 ± 0.032 | 0.218 ± 0.027 | 1.195 ± 0.012 | 1.715 ± 0.752 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.768 ± 0.384 | 0.069 ± 0.007 | 0.226 ± 0.008 | 0.342 ± 0.012 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.670 ± 0.049 | 0.183 ± 0.006 | 0.711 ± 0.007 | 0.713 ± 0.067 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.665 ± 0.039 | 0.132 ± 0.005 | 0.348 ± 0.007 | 0.459 ± 0.016 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.971 ± 0.103 | 0.399 ± 0.015 | 0.971 ± 0.002 | 1.600 ± 0.071 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 1.230 ± 0.197 | 0.104 ± 0.010 | 0.255 ± 0.005 | 1.569 ± 0.059 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.990 ± 0.023 | 0.117 ± 0.005 | 0.303 ± 0.007 | 1.829 ± 2.340 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.906 ± 0.037 | 0.103 ± 0.012 | 0.364 ± 0.008 | 2.022 ± 0.279 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 1.023 ± 0.084 | 0.128 ± 0.020 | 0.660 ± 0.008 | 2.198 ± 0.414 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 1.715 ± 0.091 | 0.400 ± 0.012 | 0.491 ± 0.010 | 1.951 ± 2.117 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 1.955 ± 0.083 | 0.542 ± 0.031 | 1.055 ± 0.190 | 2.807 ± 0.569 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 1.622 ± 0.068 | 0.404 ± 0.031 | 0.578 ± 0.011 | 2.036 ± 0.020 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 2.016 ± 0.050 | 0.565 ± 0.067 | 1.330 ± 0.017 | 3.143 ± 1.310 |
| large | `eigh` | f64 | 4 | `64x64` | 23.868 ± 2.243 | 7.216 ± 2.806 | 1.257 ± 0.141 | 3.155 ± 0.318 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 66.107 ± 16.591 | 6.816 ± 0.072 | 14.292 ± 4.508 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 174.251 ± 12.171 | 30.463 ± 3.707 | 50.981 ± 9.068 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 55.058 ± 14.406 | 5.805 ± 0.053 | 14.608 ± 3.192 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 97.587 ± 2.734 | 21.206 ± 0.398 | 51.834 ± 12.916 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 32.090 ± 15.525 | 6.614 ± 1.063 | 3.556 ± 0.317 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 91.550 ± 39.987 | 31.477 ± 1.005 | 14.738 ± 2.094 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 32.631 ± 3.838 | 2.935 ± 0.386 | 6.916 ± 0.608 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 184.738 ± 22.504 | 17.800 ± 0.790 | 18.979 ± 4.993 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 15.155 ± 1.146 | 0.834 ± 0.062 | 1.950 ± 0.004 | 2.574 ± 0.075 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 30.942 ± 4.661 | 1.607 ± 0.150 | 2.055 ± 0.036 | 3.831 ± 0.036 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 43.370 ± 5.736 | 4.160 ± 0.184 | 8.069 ± 1.883 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 197.710 ± 49.207 | 23.685 ± 2.281 | 36.082 ± 5.066 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 41.036 ± 16.969 | 5.393 ± 0.651 | 9.665 ± 1.530 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 225.868 ± 28.018 | 24.974 ± 0.759 | 54.273 ± 9.482 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 3.743 ± 3.971 | 0.547 ± 0.035 | 1.447 ± 0.164 | 2.815 ± 2.363 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 11.780 ± 3.452 | 1.233 ± 0.021 | 2.037 ± 0.247 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 72.287 ± 3.812 | 2.977 ± 0.029 | 6.507 ± 0.542 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 12.900 ± 2.603 | 1.514 ± 0.016 | 1.839 ± 0.097 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 84.474 ± 27.820 | 4.569 ± 0.032 | 7.075 ± 0.161 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 68.023 ± 5.763 | 9.352 ± 8.565 | 1.873 ± 0.008 | 3.267 ± 0.652 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 96.675 ± 8.376 | 15.876 ± 1.131 | 28.610 ± 10.040 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 197.245 ± 90.302 | 77.337 ± 3.369 | 106.981 ± 10.443 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 91.287 ± 24.502 | 13.555 ± 1.276 | 28.353 ± 9.763 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 186.419 ± 3.642 | 72.322 ± 2.823 | 107.394 ± 13.006 |
| large | `matmul` | f64 | 4 | `128x128` | 17.211 ± 8.680 | 0.246 ± 0.009 | 7.440 ± 0.463 | 8.660 ± 1.098 |
| large | `matmul` | f64 | 4 | `256x256` | 26.281 ± 8.860 | 1.893 ± 0.052 | 31.929 ± 1.204 | 32.077 ± 1.292 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 37.494 ± 5.204 | 9.822 ± 0.316 | 132.693 ± 5.942 | 127.010 ± 7.432 |
| large | `qr` | f64 | 4 | `64x64` | 18.418 ± 1.494 | 2.353 ± 0.873 | 1.145 ± 0.004 | 1.777 ± 0.827 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 1.507 ± 0.464 | 0.286 ± 0.054 | 1.065 ± 0.206 | 2.388 ± 0.888 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 16.050 ± 3.013 | 0.962 ± 0.115 | 1.294 ± 0.007 | 1.774 ± 0.007 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 16.287 ± 2.935 | 1.468 ± 0.053 | 1.992 ± 0.011 | 2.737 ± 2.526 |
| large | `svd` | f64 | 4 | `64x64` | 27.368 ± 1.682 | 7.361 ± 0.229 | 1.868 ± 0.017 | 2.203 ± 0.036 |
| small | `eigh` | f64 | 4 | `2x2` | 0.461 ± 0.015 | 0.037 ± 0.004 | 0.061 ± 0.003 | 0.363 ± 0.023 |
| small | `eigh` | f64 | 4 | `4x4` | 0.503 ± 0.019 | 0.040 ± 0.003 | 0.072 ± 0.002 | 0.150 ± 0.009 |
| small | `eigh` | f64 | 4 | `8x8` | 0.474 ± 0.025 | 0.044 ± 0.003 | 0.074 ± 0.002 | 0.203 ± 0.008 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.441 ± 0.021 | 0.029 ± 0.003 | 0.052 ± 0.001 | 0.424 ± 0.014 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.482 ± 0.037 | 0.028 ± 0.003 | 0.058 ± 0.002 | 0.426 ± 0.015 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.440 ± 0.014 | 0.029 ± 0.002 | 0.082 ± 0.003 | 0.466 ± 0.026 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.230 ± 0.006 | 0.188 ± 0.008 | 0.552 ± 0.014 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.236 ± 0.012 | 0.190 ± 0.008 | 0.565 ± 0.015 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.242 ± 0.007 | 0.234 ± 0.006 | 0.563 ± 0.016 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.235 ± 0.008 | 0.183 ± 0.011 | 1.277 ± 0.018 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.238 ± 0.010 | 0.187 ± 0.008 | 1.298 ± 0.017 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.247 ± 0.011 | 0.229 ± 0.005 | 2.714 ± 1.418 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.221 ± 0.008 | 0.274 ± 0.007 | 0.796 ± 0.020 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.226 ± 0.004 | 0.278 ± 0.009 | 0.801 ± 0.021 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.231 ± 0.010 | 0.330 ± 0.014 | 0.817 ± 0.016 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.242 ± 0.007 | 0.264 ± 0.047 | 2.034 ± 0.028 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.244 ± 0.011 | 0.253 ± 0.005 | 2.063 ± 0.018 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.249 ± 0.018 | 0.304 ± 0.009 | 2.091 ± 0.023 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.883 ± 0.029 | 0.096 ± 0.003 | 0.141 ± 0.021 | 3.000 ± 1.814 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.882 ± 0.049 | 0.097 ± 0.006 | 0.126 ± 0.003 | 1.350 ± 1.871 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.850 ± 0.018 | 0.098 ± 0.002 | 0.172 ± 0.031 | 1.300 ± 0.292 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.409 ± 0.007 | 0.220 ± 0.054 | 1.645 ± 0.058 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.415 ± 0.012 | 0.252 ± 0.056 | 0.811 ± 0.017 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.423 ± 0.017 | 0.255 ± 0.010 | 1.674 ± 0.033 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.419 ± 0.011 | 0.251 ± 0.043 | 1.931 ± 0.842 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.421 ± 0.012 | 0.240 ± 0.006 | 1.926 ± 0.024 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.435 ± 0.014 | 0.286 ± 0.011 | 2.913 ± 2.176 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 1.426 ± 0.051 | 0.376 ± 0.005 | 0.216 ± 0.029 | 3.312 ± 1.703 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 1.422 ± 0.049 | 0.380 ± 0.004 | 0.186 ± 0.003 | 3.000 ± 1.798 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 1.458 ± 0.023 | 0.381 ± 0.009 | 0.239 ± 0.005 | 3.366 ± 1.729 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.251 ± 0.008 | 0.409 ± 0.017 | 0.667 ± 0.021 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.255 ± 0.010 | 0.409 ± 0.013 | 0.658 ± 0.013 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.252 ± 0.008 | 0.408 ± 0.095 | 0.679 ± 0.031 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.271 ± 0.007 | 0.263 ± 0.058 | 1.415 ± 0.023 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.280 ± 0.007 | 0.271 ± 0.055 | 1.409 ± 0.012 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.273 ± 0.006 | 0.266 ± 0.007 | 1.426 ± 0.011 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.946 ± 0.056 | 0.137 ± 0.010 | 0.137 ± 0.004 | 1.818 ± 1.632 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.943 ± 0.031 | 0.140 ± 0.005 | 0.147 ± 0.003 | 1.912 ± 1.992 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.943 ± 0.033 | 0.155 ± 0.013 | 0.201 ± 0.006 | 1.590 ± 2.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.133 ± 0.006 | 0.249 ± 0.010 | 0.574 ± 0.022 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.141 ± 0.011 | 0.251 ± 0.006 | 0.599 ± 0.017 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.153 ± 0.005 | 0.309 ± 0.013 | 0.578 ± 0.008 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.138 ± 0.006 | 0.206 ± 0.016 | 2.497 ± 1.749 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.141 ± 0.006 | 0.204 ± 0.008 | 1.749 ± 1.429 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.158 ± 0.013 | 0.257 ± 0.009 | 2.611 ± 1.616 |
| small | `matmul` | f64 | 4 | `2x2` | 0.450 ± 0.039 | 0.028 ± 0.003 | 0.031 ± 0.001 | 0.195 ± 0.009 |
| small | `matmul` | f64 | 4 | `4x4` | 0.490 ± 0.028 | 0.028 ± 0.003 | 0.036 ± 0.003 | 0.232 ± 0.010 |
| small | `matmul` | f64 | 4 | `8x8` | 0.438 ± 0.018 | 0.029 ± 0.002 | 0.060 ± 0.002 | 0.283 ± 0.018 |
| small | `qr` | f64 | 4 | `2x2` | 0.477 ± 0.021 | 0.038 ± 0.002 | 0.044 ± 0.001 | 0.129 ± 0.008 |
| small | `qr` | f64 | 4 | `4x4` | 0.513 ± 0.040 | 0.038 ± 0.003 | 0.051 ± 0.003 | 0.134 ± 0.005 |
| small | `qr` | f64 | 4 | `8x8` | 0.469 ± 0.059 | 0.040 ± 0.002 | 0.056 ± 0.002 | 0.178 ± 0.011 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.583 ± 0.017 | 0.112 ± 0.005 | 0.094 ± 0.003 | 0.264 ± 0.013 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.593 ± 0.050 | 0.111 ± 0.004 | 0.095 ± 0.003 | 0.267 ± 0.013 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.612 ± 0.016 | 0.111 ± 0.005 | 0.085 ± 0.003 | 0.272 ± 0.012 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.634 ± 0.032 | 0.112 ± 0.005 | 0.086 ± 0.002 | 0.275 ± 0.011 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.572 ± 0.033 | 0.112 ± 0.006 | 0.095 ± 0.002 | 0.292 ± 0.012 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.593 ± 0.043 | 0.114 ± 0.005 | 0.100 ± 0.003 | 0.299 ± 0.011 |
| small | `svd` | f64 | 4 | `2x2` | 0.483 ± 0.017 | 0.039 ± 0.003 | 0.047 ± 0.003 | 0.140 ± 0.006 |
| small | `svd` | f64 | 4 | `4x4` | 0.512 ± 0.018 | 0.045 ± 0.007 | 0.056 ± 0.002 | 0.145 ± 0.039 |
| small | `svd` | f64 | 4 | `8x8` | 0.489 ± 0.025 | 0.053 ± 0.006 | 0.074 ± 0.002 | 0.195 ± 0.008 |

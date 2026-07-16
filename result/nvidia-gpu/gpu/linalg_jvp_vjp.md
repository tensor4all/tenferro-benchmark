# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/workspaces/tenferro-benchmark/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-07-16T11:06:54.527544+00:00`
- tenferro-rs commit: `3abc9108e4f0500f4e75519711cdb4e21b9625df`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA GeForce RTX 3060`
- UUID: `GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`
- Memory: `12 GiB`
- Driver version: `580.159.03`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92400`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-134-generic-x86_64-with-glibc2.39`

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext` on CUDA; PyTorch uses `torch.func.jvp` / `vjp` on CUDA; JAX uses `jax.jvp` / `jax.vjp` on CUDA.
Inputs are uploaded to the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization without downloading AD outputs in the timed region.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA | JAX CUDA |
|---|---|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 9.641 ± 0.182 | 9.566 ± 0.117 | 10.623 ± 0.128 |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | 24.810 ± 0.151 | 25.950 ± 0.141 | 27.292 ± 1.537 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 9.759 ± 0.072 | 9.398 ± 0.240 | 10.717 ± 0.354 |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | 24.859 ± 0.208 | 23.034 ± 0.104 | 26.773 ± 0.325 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 2.717 ± 0.126 | 3.945 ± 0.160 | 2.507 ± 0.198 |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | 10.992 ± 0.016 | 18.919 ± 0.104 | 9.836 ± 0.199 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 2.513 ± 0.112 | 2.204 ± 0.079 | 5.824 ± 0.659 |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | 11.201 ± 0.169 | 10.441 ± 0.007 | 11.449 ± 1.585 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 5.677 ± 0.177 | 5.031 ± 0.096 | 6.087 ± 0.117 |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | 17.589 ± 0.052 | 16.655 ± 0.009 | 18.000 ± 0.170 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 6.164 ± 0.223 | 5.054 ± 0.219 | 6.295 ± 0.686 |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | 18.202 ± 0.368 | 17.000 ± 0.017 | 17.345 ± 0.198 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 1.754 ± 0.060 | 1.190 ± 0.012 | 1.981 ± 0.011 |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | 3.625 ± 0.072 | 3.092 ± 0.020 | 3.899 ± 0.210 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 1.725 ± 0.029 | 2.462 ± 0.062 | 2.740 ± 1.428 |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | 3.632 ± 0.049 | 5.984 ± 0.115 | 4.378 ± 0.313 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 81.819 ± 0.376 | 85.769 ± 0.077 | 83.133 ± 0.040 |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | 267.602 ± 0.090 | 278.066 ± 0.048 | 270.542 ± 0.053 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 81.936 ± 0.200 | 85.333 ± 0.113 | 84.310 ± 0.041 |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | 267.602 ± 0.070 | 273.570 ± 0.060 | 269.296 ± 0.042 |
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 1.069 ± 0.159 | 0.676 ± 0.013 | 1.138 ± 0.022 |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.787 ± 0.088 | 0.750 ± 0.012 | 1.486 ± 0.222 |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 0.878 ± 0.024 | 0.848 ± 0.014 | 1.574 ± 0.013 |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 0.724 ± 0.025 | 0.949 ± 0.117 | 2.705 ± 0.023 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.789 ± 0.047 | 0.892 ± 0.056 | 2.983 ± 0.018 |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 0.896 ± 0.059 | 0.969 ± 0.052 | 3.065 ± 0.038 |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 1.061 ± 0.451 | 1.056 ± 0.005 | 1.777 ± 0.012 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 0.988 ± 0.019 | 1.155 ± 0.027 | 1.759 ± 0.061 |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 0.998 ± 0.022 | 1.139 ± 0.053 | 1.889 ± 0.172 |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 0.967 ± 0.240 | 1.620 ± 0.117 | 4.372 ± 0.093 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 0.922 ± 0.043 | 1.577 ± 0.113 | 4.425 ± 0.132 |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 0.946 ± 0.029 | 1.567 ± 0.031 | 4.465 ± 0.425 |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | 3.173 ± 0.092 | 0.808 ± 0.006 | 2.574 ± 0.014 |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | 1.447 ± 0.058 | 0.860 ± 0.008 | 2.117 ± 0.233 |
| small | `grad_sum_qr_jvp` | f64 | `8x8` | 1.435 ± 0.051 | 0.859 ± 0.036 | 2.199 ± 0.264 |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | 2.846 ± 0.319 | 1.519 ± 0.206 | 4.214 ± 0.329 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 2.375 ± 0.273 | 1.602 ± 0.075 | 4.299 ± 0.179 |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | 2.492 ± 0.121 | 1.549 ± 0.118 | 4.442 ± 0.252 |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 0.978 ± 0.263 | 0.925 ± 0.006 | 1.716 ± 0.023 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 1.091 ± 0.035 | 1.040 ± 0.009 | 1.629 ± 0.021 |
| small | `grad_sum_solve_jvp` | f64 | `8x8,rhs=1` | 1.118 ± 0.042 | 1.034 ± 0.021 | 1.838 ± 0.026 |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 1.067 ± 0.017 | 1.412 ± 0.077 | 4.122 ± 0.679 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 1.014 ± 0.030 | 1.362 ± 0.095 | 3.760 ± 0.625 |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 1.068 ± 0.026 | 1.368 ± 0.193 | 4.096 ± 0.069 |
| small | `grad_sum_svd_s_jvp` | f64 | `2x2` | 1.187 ± 0.039 | 0.977 ± 0.008 | 1.104 ± 0.067 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 1.453 ± 0.052 | 1.237 ± 0.017 | 2.161 ± 0.030 |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 2.116 ± 0.146 | 1.409 ± 0.042 | 2.485 ± 0.027 |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 1.650 ± 0.276 | 0.785 ± 0.146 | 2.383 ± 0.106 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 1.460 ± 0.064 | 1.094 ± 0.041 | 3.472 ± 0.014 |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 1.671 ± 0.034 | 1.291 ± 0.076 | 3.795 ± 0.011 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

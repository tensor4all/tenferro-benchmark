# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/workspaces/tenferro-benchmark/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-07-11T06:17:40.618558+00:00`
- tenferro-rs commit: `d5c768c7eb58f252e7855fea80bb6af5bb7ddb40`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA GeForce RTX 3060`
- UUID: `GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`
- Memory: `12 GiB`
- Driver version: `580.159.03`
- CUDA version: `13.0`
- CUDA runtime: `12.6`
- cuDNN version: `92000`

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
tenferro-rs CUDA trace linalg JVP/VJP is partially unsupported: `grad_sum_qr` hits scalar-vector `mul` shape mismatches on CUDA because the backend does not yet implement CPU-style 0-D scalar broadcast.
Inputs are uploaded to the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization without downloading AD outputs in the timed region.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA | JAX CUDA |
|---|---|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 9.722 ± 0.126 | 9.649 ± 0.071 | not configured |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | 24.804 ± 0.136 | 26.310 ± 0.215 | not configured |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 9.671 ± 0.209 | 0.676 ± 0.013 | not configured |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | 24.966 ± 0.215 | 1.852 ± 0.119 | not configured |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 2.595 ± 0.220 | 4.042 ± 0.061 | not configured |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | 11.001 ± 0.009 | 19.036 ± 0.147 | not configured |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 2.621 ± 0.234 | 1.839 ± 0.026 | not configured |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | 11.264 ± 0.118 | 8.105 ± 0.067 | not configured |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | unsupported | 5.116 ± 0.038 | not configured |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | unsupported | 16.771 ± 0.009 | not configured |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | unsupported | 1.641 ± 0.077 | not configured |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | unsupported | 7.253 ± 0.090 | not configured |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 1.746 ± 0.019 | 1.184 ± 0.034 | not configured |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | 3.590 ± 0.070 | 3.100 ± 0.035 | not configured |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 1.754 ± 0.015 | 1.378 ± 0.042 | not configured |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | 3.587 ± 0.033 | 3.212 ± 0.065 | not configured |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 82.133 ± 0.199 | 86.385 ± 0.069 | not configured |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | 270.605 ± 0.084 | 280.139 ± 0.104 | not configured |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 82.322 ± 0.148 | 0.759 ± 0.013 | not configured |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | 270.656 ± 0.161 | 2.036 ± 0.014 | not configured |
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 1.302 ± 0.105 | 0.681 ± 0.020 | not configured |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.765 ± 0.026 | 0.729 ± 0.006 | not configured |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 0.870 ± 0.049 | 0.857 ± 0.025 | not configured |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 0.764 ± 0.034 | 0.377 ± 0.025 | not configured |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.816 ± 0.094 | 0.389 ± 0.035 | not configured |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 0.891 ± 0.039 | 0.391 ± 0.016 | not configured |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 0.987 ± 0.031 | 1.061 ± 0.031 | not configured |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 0.990 ± 0.036 | 1.147 ± 0.041 | not configured |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 0.999 ± 0.024 | 1.154 ± 0.030 | not configured |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 0.896 ± 0.017 | 1.001 ± 0.019 | not configured |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 0.914 ± 0.019 | 0.997 ± 0.024 | not configured |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 0.927 ± 0.029 | 1.000 ± 0.027 | not configured |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | unsupported | 0.995 ± 0.094 | not configured |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | unsupported | 0.840 ± 0.023 | not configured |
| small | `grad_sum_qr_jvp` | f64 | `8x8` | unsupported | 0.857 ± 0.020 | not configured |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | unsupported | 0.791 ± 0.109 | not configured |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | unsupported | 0.965 ± 0.028 | not configured |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | unsupported | 0.973 ± 0.022 | not configured |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 0.822 ± 0.294 | 0.938 ± 0.020 | not configured |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 1.096 ± 0.026 | 1.026 ± 0.035 | not configured |
| small | `grad_sum_solve_jvp` | f64 | `8x8,rhs=1` | 1.100 ± 0.039 | 1.037 ± 0.030 | not configured |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 1.146 ± 0.098 | 0.819 ± 0.034 | not configured |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 1.095 ± 0.025 | 0.847 ± 0.069 | not configured |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 1.115 ± 0.024 | 0.853 ± 0.056 | not configured |
| small | `grad_sum_svd_s_jvp` | f64 | `2x2` | 1.743 ± 0.040 | 0.987 ± 0.018 | not configured |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 1.452 ± 0.047 | 1.225 ± 0.013 | not configured |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 1.659 ± 0.027 | 1.390 ± 0.055 | not configured |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 1.535 ± 0.038 | 0.303 ± 0.041 | not configured |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 1.459 ± 0.049 | 0.390 ± 0.023 | not configured |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 1.695 ± 0.022 | 0.390 ± 0.018 | not configured |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

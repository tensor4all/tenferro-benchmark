# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/workspaces/tenferro-benchmark/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-07-11T06:29:55.094745+00:00`
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
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 9.765 ± 0.059 | 9.538 ± 0.125 | not configured |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | 25.040 ± 0.189 | 26.176 ± 0.038 | not configured |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 9.748 ± 0.205 | 9.504 ± 0.117 | not configured |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | 25.000 ± 0.062 | 23.366 ± 0.162 | not configured |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 2.825 ± 0.197 | 3.890 ± 0.165 | not configured |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | 10.999 ± 0.018 | 19.118 ± 0.032 | not configured |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 2.713 ± 0.218 | 2.296 ± 0.101 | not configured |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | 11.271 ± 0.133 | 10.515 ± 0.009 | not configured |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | unsupported | 5.106 ± 0.068 | not configured |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | unsupported | 16.763 ± 0.007 | not configured |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | unsupported | 5.145 ± 0.168 | not configured |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | unsupported | 17.110 ± 0.006 | not configured |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 1.667 ± 0.085 | 1.211 ± 0.040 | not configured |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | 3.606 ± 0.054 | 3.114 ± 0.052 | not configured |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 1.939 ± 0.187 | 2.569 ± 0.156 | not configured |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | 3.578 ± 0.070 | 6.129 ± 0.080 | not configured |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 82.803 ± 0.130 | 86.411 ± 0.105 | not configured |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | 271.795 ± 0.059 | 280.096 ± 0.051 | not configured |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 82.927 ± 0.242 | 85.939 ± 0.178 | not configured |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | 271.799 ± 0.068 | 275.630 ± 0.235 | not configured |
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 1.390 ± 0.173 | 0.680 ± 0.012 | not configured |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.731 ± 0.042 | 0.760 ± 0.013 | not configured |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 0.860 ± 0.025 | 0.858 ± 0.013 | not configured |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 0.773 ± 0.082 | 0.922 ± 0.169 | not configured |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.756 ± 0.063 | 0.890 ± 0.027 | not configured |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 1.013 ± 0.033 | 0.967 ± 0.065 | not configured |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 0.980 ± 0.040 | 1.098 ± 0.007 | not configured |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 0.869 ± 0.025 | 1.202 ± 0.028 | not configured |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 1.001 ± 0.044 | 1.191 ± 0.054 | not configured |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 0.887 ± 0.048 | 1.639 ± 0.102 | not configured |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 0.891 ± 0.035 | 1.594 ± 0.121 | not configured |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 0.998 ± 0.082 | 1.656 ± 0.120 | not configured |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | unsupported | 0.818 ± 0.027 | not configured |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | unsupported | 0.878 ± 0.024 | not configured |
| small | `grad_sum_qr_jvp` | f64 | `8x8` | unsupported | 0.876 ± 0.029 | not configured |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | unsupported | 1.573 ± 0.201 | not configured |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | unsupported | 1.638 ± 0.172 | not configured |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | unsupported | 1.549 ± 0.073 | not configured |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 1.139 ± 0.030 | 0.929 ± 0.006 | not configured |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 1.138 ± 0.054 | 1.047 ± 0.009 | not configured |
| small | `grad_sum_solve_jvp` | f64 | `8x8,rhs=1` | 1.123 ± 0.030 | 1.063 ± 0.043 | not configured |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 1.069 ± 0.021 | 1.358 ± 0.040 | not configured |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 1.096 ± 0.033 | 1.395 ± 0.099 | not configured |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 1.082 ± 0.018 | 1.474 ± 0.117 | not configured |
| small | `grad_sum_svd_s_jvp` | f64 | `2x2` | 1.619 ± 0.248 | 1.005 ± 0.013 | not configured |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 1.439 ± 0.033 | 1.259 ± 0.013 | not configured |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 1.648 ± 0.033 | 1.449 ± 0.043 | not configured |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 1.758 ± 0.034 | 0.765 ± 0.139 | not configured |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 1.494 ± 0.084 | 1.149 ± 0.148 | not configured |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 1.661 ± 0.012 | 1.326 ± 0.071 | not configured |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

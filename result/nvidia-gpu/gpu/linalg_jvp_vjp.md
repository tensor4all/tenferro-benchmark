# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/workspaces/tenferro-benchmark/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-08-22T03:08:44.884643+00:00`
- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA GeForce RTX 3060`
- UUID: `GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`
- Memory: `12 GiB`
- Driver version: `580.173.02`
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
- Python platform: `Linux-6.8.0-138-generic-x86_64-with-glibc2.39`

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext` on CUDA; PyTorch uses `torch.func.jvp` / `vjp` on CUDA.
Inputs are uploaded to the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization without downloading AD outputs in the timed region.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA |
|---|---|---:|---|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 9.794 ± 0.165 | 9.672 ± 0.122 |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | 25.235 ± 0.176 | 26.389 ± 0.093 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 9.872 ± 0.180 | 9.446 ± 0.149 |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | 25.376 ± 0.367 | 23.525 ± 0.121 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 2.677 ± 0.043 | 4.074 ± 0.160 |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | 11.228 ± 0.096 | 19.199 ± 0.096 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 2.533 ± 0.040 | 2.392 ± 0.080 |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | 11.484 ± 0.255 | 10.601 ± 0.006 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 5.696 ± 0.169 | 5.144 ± 0.090 |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | 18.005 ± 0.242 | 16.902 ± 0.020 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 5.556 ± 0.271 | 5.210 ± 0.189 |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | 17.987 ± 0.072 | 17.178 ± 0.135 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 1.627 ± 0.046 | 1.203 ± 0.014 |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | 4.007 ± 0.094 | 3.132 ± 0.007 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 1.645 ± 0.026 | 2.427 ± 0.090 |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | 4.018 ± 0.041 | 6.104 ± 0.129 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 82.891 ± 0.036 | 87.107 ± 0.100 |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | 271.404 ± 0.200 | 282.143 ± 0.026 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 83.024 ± 0.158 | 86.439 ± 0.137 |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | 271.438 ± 0.193 | 277.402 ± 0.069 |
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 0.467 ± 0.035 | 0.689 ± 0.036 |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.521 ± 0.051 | 0.720 ± 0.007 |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 0.778 ± 0.036 | 0.830 ± 0.038 |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 0.489 ± 0.089 | 0.842 ± 0.119 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.502 ± 0.034 | 0.654 ± 0.212 |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 0.777 ± 0.050 | 0.913 ± 0.220 |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 0.663 ± 0.021 | 1.088 ± 0.006 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 0.679 ± 0.014 | 1.107 ± 0.024 |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 0.711 ± 0.026 | 1.095 ± 0.026 |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 0.645 ± 0.047 | 1.527 ± 0.126 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 0.652 ± 0.033 | 1.497 ± 0.323 |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 0.668 ± 0.045 | 1.537 ± 0.062 |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | 2.627 ± 0.108 | 0.827 ± 0.012 |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | 1.126 ± 0.056 | 0.844 ± 0.062 |
| small | `grad_sum_qr_jvp` | f64 | `8x8` | 1.190 ± 0.153 | 0.819 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | 1.065 ± 0.042 | 1.190 ± 0.213 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 1.051 ± 0.032 | 1.300 ± 0.543 |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | 1.116 ± 0.060 | 1.474 ± 0.220 |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 0.886 ± 0.038 | 0.934 ± 0.005 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 0.916 ± 0.114 | 0.986 ± 0.067 |
| small | `grad_sum_solve_jvp` | f64 | `8x8,rhs=1` | 0.925 ± 0.112 | 0.986 ± 0.010 |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 0.859 ± 0.025 | 1.316 ± 0.212 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 0.892 ± 0.023 | 1.143 ± 0.237 |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 0.890 ± 0.135 | 1.309 ± 0.375 |
| small | `grad_sum_svd_s_jvp` | f64 | `2x2` | 1.417 ± 0.275 | 1.075 ± 0.096 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 1.238 ± 0.198 | 1.239 ± 0.032 |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 1.480 ± 0.060 | 1.368 ± 0.032 |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 0.644 ± 0.012 | 0.718 ± 0.069 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 1.059 ± 0.202 | 1.005 ± 0.226 |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 1.465 ± 0.061 | 1.067 ± 0.122 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

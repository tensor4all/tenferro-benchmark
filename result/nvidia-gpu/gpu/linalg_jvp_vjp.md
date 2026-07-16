# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/workspaces/tenferro-benchmark/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-07-16T09:12:48.166681+00:00`
- tenferro-rs commit: `8c267c636537d3d407925b61a2000f9bbe07e6a1`

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
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 9.727 ± 0.230 | 9.634 ± 0.134 | 10.601 ± 0.227 |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | 24.838 ± 0.198 | 25.981 ± 0.055 | 26.367 ± 0.293 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 9.712 ± 0.182 | 9.460 ± 0.108 | 10.515 ± 0.226 |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | 24.988 ± 0.340 | 23.178 ± 0.089 | 26.788 ± 0.355 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 2.646 ± 0.140 | 3.951 ± 0.235 | 2.484 ± 0.145 |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | 11.000 ± 0.034 | 18.971 ± 0.047 | 9.915 ± 0.243 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 2.561 ± 0.089 | 2.288 ± 0.078 | 5.462 ± 1.278 |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | 11.204 ± 0.154 | 10.448 ± 0.014 | 11.889 ± 1.485 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 5.599 ± 0.195 | 5.095 ± 0.027 | 6.018 ± 0.066 |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | 17.604 ± 0.036 | 16.660 ± 0.020 | 17.735 ± 0.095 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 6.050 ± 0.087 | 5.146 ± 0.128 | 5.998 ± 0.552 |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | 18.234 ± 0.367 | 16.999 ± 0.015 | 17.310 ± 0.216 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 1.759 ± 0.021 | 1.199 ± 0.019 | 1.931 ± 0.020 |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | 3.646 ± 0.048 | 3.096 ± 0.006 | 3.879 ± 0.040 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 1.750 ± 0.027 | 2.521 ± 0.161 | 3.603 ± 0.192 |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | 3.586 ± 0.031 | 5.961 ± 0.018 | 4.616 ± 0.849 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 81.828 ± 0.212 | 85.852 ± 0.091 | 83.062 ± 0.036 |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | 267.583 ± 0.096 | 278.058 ± 0.080 | 270.578 ± 0.130 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 81.980 ± 0.238 | 85.362 ± 0.077 | 84.440 ± 0.292 |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | 267.593 ± 0.045 | 273.555 ± 0.107 | 269.309 ± 0.054 |
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 0.821 ± 0.124 | 0.737 ± 0.036 | 1.270 ± 0.722 |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.743 ± 0.042 | 0.791 ± 0.004 | 1.412 ± 0.045 |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 0.878 ± 0.040 | 0.886 ± 0.013 | 1.731 ± 0.250 |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 0.897 ± 0.052 | 0.870 ± 0.083 | 2.703 ± 0.014 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.761 ± 0.047 | 0.953 ± 0.145 | 2.937 ± 0.010 |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 0.869 ± 0.064 | 1.026 ± 0.061 | 3.121 ± 0.020 |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 1.145 ± 0.204 | 1.120 ± 0.031 | 2.086 ± 0.020 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 1.036 ± 0.050 | 1.222 ± 0.053 | 1.758 ± 0.036 |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 0.991 ± 0.019 | 1.192 ± 0.045 | 1.928 ± 0.035 |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 0.879 ± 0.011 | 1.717 ± 0.130 | 4.284 ± 0.074 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 0.890 ± 0.037 | 1.664 ± 0.111 | 4.400 ± 0.153 |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 0.915 ± 0.122 | 1.647 ± 0.115 | 4.422 ± 0.025 |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | 2.645 ± 0.197 | 0.848 ± 0.006 | 2.383 ± 0.038 |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | 1.403 ± 0.067 | 0.905 ± 0.017 | 2.102 ± 0.025 |
| small | `grad_sum_qr_jvp` | f64 | `8x8` | 1.435 ± 0.073 | 0.902 ± 0.021 | 2.572 ± 0.039 |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | 3.440 ± 0.105 | 1.486 ± 0.098 | 4.120 ± 0.084 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 2.156 ± 0.250 | 1.652 ± 0.150 | 4.303 ± 0.334 |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | 2.396 ± 0.204 | 1.610 ± 0.105 | 4.123 ± 0.305 |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 0.819 ± 0.017 | 0.953 ± 0.004 | 1.397 ± 0.013 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 1.104 ± 0.017 | 1.088 ± 0.035 | 1.606 ± 0.034 |
| small | `grad_sum_solve_jvp` | f64 | `8x8,rhs=1` | 1.100 ± 0.038 | 1.073 ± 0.002 | 1.550 ± 0.014 |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 1.090 ± 0.039 | 1.414 ± 0.131 | 3.368 ± 0.028 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 1.073 ± 0.021 | 1.445 ± 0.107 | 3.360 ± 0.022 |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 1.096 ± 0.020 | 1.430 ± 0.114 | 3.289 ± 0.034 |
| small | `grad_sum_svd_s_jvp` | f64 | `2x2` | 1.790 ± 0.304 | 1.036 ± 0.087 | 1.116 ± 0.029 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 1.437 ± 0.217 | 1.307 ± 0.050 | 2.012 ± 0.322 |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 1.652 ± 0.019 | 1.477 ± 0.038 | 2.239 ± 0.010 |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 1.336 ± 0.138 | 0.795 ± 0.090 | 2.990 ± 0.035 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 1.441 ± 0.095 | 1.178 ± 0.189 | 3.520 ± 0.100 |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 1.655 ± 0.024 | 1.346 ± 0.122 | 3.754 ± 0.005 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

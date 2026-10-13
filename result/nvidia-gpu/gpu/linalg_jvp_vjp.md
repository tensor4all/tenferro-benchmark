# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/workspaces/tenferro-benchmark/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-10-13T14:29:37.083456+00:00`
- tenferro-rs commit: `dfbdf6b66d40162f4731da116ec42ed034da0dce`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA GeForce RTX 3060`
- UUID: `GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`
- Memory: `12 GiB`
- Driver version: `580.178.04`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92700`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-146-generic-x86_64-with-glibc2.39`

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext` on CUDA; PyTorch uses `torch.func.jvp` / `vjp` on CUDA.
JAX uses an explicitly lowered/compiled whole-operation JIT; compilation and lazy initialization occur before sampling. Primal loss and first-order derivatives with respect to A are both returned and checked against CPU F64 AD, whose directional derivative is checked by central differences. RHS and derivative seeds are prepared before timing.
Standard tenferro trace rows are unsupported when run_prepared cannot borrow an entered session. Small trace rows are separately labelled public-path latency diagnostics including internal session entry. Each unsupported row retains its actual reason in the raw records.
Inputs are uploaded to the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization without downloading AD outputs in the timed region.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | unsupported | 9.587 ± 0.068 | 10.153 ± 0.244 |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | unsupported | 26.226 ± 0.078 | 24.549 ± 0.198 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | unsupported | 9.469 ± 0.100 | 10.154 ± 0.134 |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | unsupported | 23.384 ± 0.077 | 24.466 ± 0.176 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | unsupported | 3.800 ± 0.171 | 2.106 ± 0.037 |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | unsupported | 19.069 ± 0.153 | 8.743 ± 0.090 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | unsupported | 2.292 ± 0.142 | 2.267 ± 0.090 |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | unsupported | 10.549 ± 0.011 | 9.523 ± 0.329 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | unsupported | 5.108 ± 0.025 | 5.416 ± 0.080 |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | unsupported | 16.819 ± 0.031 | 16.370 ± 0.147 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | unsupported | 5.049 ± 0.173 | 5.424 ± 0.166 |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | unsupported | 17.116 ± 0.164 | 16.684 ± 0.649 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | unsupported | 1.218 ± 0.054 | 1.138 ± 0.013 |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | unsupported | 3.114 ± 0.016 | 3.247 ± 0.049 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | unsupported | 2.489 ± 0.116 | 1.173 ± 0.036 |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | unsupported | 6.170 ± 0.097 | 3.241 ± 0.035 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | unsupported | 84.360 ± 0.094 | 82.279 ± 0.127 |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | unsupported | 279.844 ± 0.044 | 269.415 ± 0.069 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | unsupported | 83.885 ± 0.091 | 82.202 ± 0.064 |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | unsupported | 275.270 ± 0.025 | 269.448 ± 0.095 |

## Sampling and unsupported routes

| Case | Backend | Ops/sample | Samples | Scope / reason |
| --- | --- | ---: | ---: | --- |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_256 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_256 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_qr_jvp_f64_256 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_qr_vjp_f64_256 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_eigh_jvp_f64_256 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_eigh_vjp_f64_256 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_lu_jvp_f64_256 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_lu_vjp_f64_256 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_solve_jvp_f64_256 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_solve_vjp_f64_256 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_512 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_512 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_qr_jvp_f64_512 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_qr_vjp_f64_512 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_eigh_jvp_f64_512 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_eigh_vjp_f64_512 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_lu_jvp_f64_512 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_lu_vjp_f64_512 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_solve_jvp_f64_512 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_solve_vjp_f64_512 | tenferro-cuda-trace | — | — | run_prepared cannot borrow an entered session; standard operation timing would include internal setup |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_256 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_512 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_512 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_512 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_512 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_512 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_512 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_512 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_512 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_512 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_512 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_256 | jax-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_256 | jax-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_512 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_512 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_512 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_512 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_512 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_512 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_512 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_512 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_512 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_512 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-10-17T00:05:44.550151+00:00`
- tenferro-rs commit: `2754c8c85dcc35c1b63cb7bbdd94ddfd7ea5746e`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.126.09`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92700`

## CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext` on CUDA; PyTorch uses `torch.func.jvp` / `vjp` on CUDA.
JAX uses an explicitly lowered/compiled whole-operation JIT; compilation and lazy initialization occur before sampling. Primal loss and first-order derivatives with respect to A are both returned and checked against CPU F64 AD, whose directional derivative is checked by central differences. RHS and derivative seeds are prepared before timing.
Standard tenferro trace rows are unsupported when run_prepared cannot borrow an entered session. Small trace rows are separately labelled public-path latency diagnostics including internal session entry. Each unsupported row retains its actual reason in the raw records.
Inputs are uploaded to the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization without downloading AD outputs in the timed region.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | unsupported | 3.030 ± 0.014 | 3.229 ± 0.336 |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | unsupported | 6.377 ± 0.070 | 6.357 ± 0.032 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | unsupported | 3.224 ± 0.032 | 3.037 ± 0.193 |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | unsupported | 6.408 ± 0.010 | 6.386 ± 0.023 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | unsupported | 1.103 ± 0.010 | 0.816 ± 0.011 |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | unsupported | 2.419 ± 0.020 | 2.032 ± 0.047 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | unsupported | 0.860 ± 0.028 | 0.816 ± 0.004 |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | unsupported | 1.887 ± 0.009 | 1.933 ± 0.035 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | unsupported | 1.440 ± 0.014 | 1.418 ± 0.030 |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | unsupported | 3.249 ± 0.018 | 3.391 ± 0.035 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | unsupported | 1.414 ± 0.015 | 1.458 ± 0.027 |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | unsupported | 3.214 ± 0.008 | 3.402 ± 0.026 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | unsupported | 0.926 ± 0.015 | 0.735 ± 0.005 |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | unsupported | 1.818 ± 0.007 | 1.703 ± 0.035 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | unsupported | 1.609 ± 0.061 | 0.744 ± 0.008 |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | unsupported | 3.276 ± 0.099 | 1.728 ± 0.061 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | unsupported | 12.466 ± 0.023 | 11.761 ± 0.083 |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | unsupported | 33.429 ± 0.017 | 32.534 ± 0.055 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | unsupported | 12.508 ± 0.048 | 11.920 ± 0.350 |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | unsupported | 33.375 ± 0.045 | 32.553 ± 0.031 |

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
| linalg_ad_large_grad_sum_svd_s_jvp_f64_256 | pytorch-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_256 | pytorch-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_256 | pytorch-cuda | 2 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_256 | pytorch-cuda | 2 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_256 | pytorch-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_256 | pytorch-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_256 | pytorch-cuda | 2 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_256 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_256 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_256 | pytorch-cuda | 2 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_512 | pytorch-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_512 | pytorch-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_512 | pytorch-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_512 | pytorch-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_512 | pytorch-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_512 | pytorch-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_512 | pytorch-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_512 | pytorch-cuda | 2 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_512 | pytorch-cuda | 2 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_512 | pytorch-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_256 | jax-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_256 | jax-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_256 | jax-cuda | 2 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_256 | jax-cuda | 2 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_256 | jax-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_256 | jax-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_256 | jax-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_256 | jax-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_256 | jax-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_256 | jax-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_512 | jax-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_512 | jax-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_512 | jax-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_512 | jax-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_512 | jax-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_512 | jax-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_512 | jax-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_512 | jax-cuda | 2 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_512 | jax-cuda | 2 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_512 | jax-cuda | 2 | 7 | steady_state_host_api_plus_device_sync |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

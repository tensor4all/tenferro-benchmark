# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-10-17T10:53:37.162246+00:00`
- tenferro-rs commit: `88daa885a5bb09e07f3031814bbb1a21f7a74e62`

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
Standard tenferro trace rows run the prepared graph in one caller-held CUDA session (Runtime::run_prepared_in); each timed call includes the run's own stream-retirement barrier. Small trace rows are separately labelled public-path latency diagnostics including internal session entry. Any unsupported row retains its actual reason in the raw records.
Inputs are uploaded to the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization without downloading AD outputs in the timed region.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 3.084 ± 0.019 | 3.031 ± 0.013 | 3.103 ± 0.174 |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | 6.419 ± 0.048 | 6.375 ± 0.019 | 6.411 ± 0.033 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 3.135 ± 0.082 | 3.215 ± 0.020 | 3.022 ± 0.025 |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | 6.415 ± 0.056 | 6.540 ± 0.020 | 6.414 ± 0.039 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 1.201 ± 0.015 | 1.103 ± 0.003 | 0.812 ± 0.003 |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | 2.083 ± 0.091 | 2.418 ± 0.023 | 2.007 ± 0.014 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 1.140 ± 0.010 | 0.868 ± 0.045 | 0.820 ± 0.003 |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | 2.409 ± 0.069 | 1.890 ± 0.007 | 2.050 ± 0.010 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 2.047 ± 0.105 | 1.448 ± 0.015 | 1.417 ± 0.032 |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | 3.595 ± 0.005 | 3.247 ± 0.017 | 3.304 ± 0.062 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 2.071 ± 0.028 | 1.412 ± 0.017 | 1.434 ± 0.041 |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | 3.680 ± 0.024 | 3.216 ± 0.010 | 3.451 ± 0.195 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 1.409 ± 0.214 | 0.924 ± 0.005 | 0.724 ± 0.011 |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | 2.357 ± 0.095 | 1.823 ± 0.024 | 1.695 ± 0.027 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 1.318 ± 0.017 | 1.731 ± 0.010 | 0.735 ± 0.003 |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | 2.336 ± 0.012 | 3.471 ± 0.061 | 1.738 ± 0.032 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 11.883 ± 0.075 | 12.463 ± 0.024 | 11.733 ± 0.238 |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | 32.532 ± 0.033 | 33.428 ± 0.066 | 32.512 ± 0.034 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 11.854 ± 0.044 | 12.519 ± 0.016 | 11.700 ± 0.044 |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | 32.552 ± 0.012 | 33.465 ± 0.160 | 32.514 ± 0.124 |

## Sampling and unsupported routes

| Case | Backend | Ops/sample | Samples | Scope / reason |
| --- | --- | ---: | ---: | --- |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_256 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_256 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_256 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_256 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_256 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_256 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_256 | tenferro-cuda-trace | 2 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_256 | tenferro-cuda-trace | 2 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_256 | tenferro-cuda-trace | 2 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_256 | tenferro-cuda-trace | 2 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_512 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_512 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_512 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_512 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_512 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_512 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_512 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_512 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_512 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_512 | tenferro-cuda-trace | 1 | 7 | steady_state_host_api_plus_device_sync |
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
| linalg_ad_large_grad_sum_lu_vjp_f64_512 | jax-cuda | 1 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_512 | jax-cuda | 2 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_512 | jax-cuda | 2 | 7 | steady_state_host_api_plus_device_sync |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261114_082852/gpu_linalg_jvp_vjp.yaml`
- Timestamp: `2026-11-14T08:28:52+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-c70851ec-1874-5bb5-8a0a-85e4eed17212`
- Memory: `80 GiB`
- Driver version: `595.91.07`
- CUDA version: `13.2`
- CUDA runtime: `12.8`
- cuDNN version: `91900`
- Container: `devcontainer`
- CUDA toolkit: `unknown`
- CUDA toolkit root: `/opt/tenferro-benchmark-ci/cuda`
- cuTENSOR version: `2.2.0`
- cuSOLVER version: `11.7.3`

## CPU Information

- Model: `AMD EPYC-Genoa Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `116`
- Sockets: `1`
- Cores per socket: `58`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-147-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-115`
- Container: `devcontainer`

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext` on CUDA; PyTorch uses `torch.func.jvp` / `vjp` on CUDA.
JAX uses an explicitly lowered/compiled whole-operation JIT; compilation and lazy initialization occur before sampling. Primal loss and first-order derivatives with respect to A are both returned; the check that backs each row is stated below, and a row that carries none is marked as not checked. RHS and derivative seeds are prepared before timing.
The `tenferro-rs CUDA eager` column runs the same logical case through the eager API on one caller-held CUDA engine. Unlike the trace row, the eager API exposes no separate preparation step, so its own AD preparation is inside the sample together with the forward pass; the forward and derivative scope matches the trace column.
Standard and latency AD rows time one prepared operation plus device completion per call on every backend, repeated in bounded batches. tenferro uses Runtime::run on a reusable caller-held CUDA engine. Fixtures, transfers, graph construction, compilation, preparation and initialization are untimed; output conversion and verification follow the stopped clock. Small rows measure API latency, not kernel throughput. Legacy session-entry measurements require recollection (issue #309). Any unsupported row retains its actual reason in the raw records.
Inputs are uploaded to the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization without downloading AD outputs in the timed region.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | tenferro-rs CUDA eager, per call, incl. AD prepare | tenferro-rs CUDA trace (deferred batch throughput) | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 6.171 ± 1.565 | 6.300 ± 0.248 | - | 3.102 ± 0.020 | 4.032 ± 0.751 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 5.846 ± 0.650 | 3.388 ± 0.183 | - | 3.644 ± 0.282 | 3.969 ± 0.488 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 1.898 ± 0.456 | 3.248 ± 0.841 | - | 1.187 ± 0.063 | 1.596 ± 0.160 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 1.682 ± 0.480 | 2.065 ± 0.107 | - | 1.408 ± 0.198 | 2.026 ± 0.354 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 3.066 ± 0.584 | 4.311 ± 0.201 | - | 1.525 ± 0.043 | 2.542 ± 0.496 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 2.961 ± 0.063 | 2.639 ± 0.190 | - | 1.510 ± 0.072 | 2.505 ± 0.564 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 1.908 ± 0.182 | 2.136 ± 0.339 | - | 0.976 ± 0.309 | 1.569 ± 0.124 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 1.695 ± 0.189 | 1.486 ± 0.058 | - | 3.804 ± 3.005 | 1.787 ± 0.281 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 25.844 ± 10.021 | 25.051 ± 1.027 | - | 12.937 ± 0.456 | 12.685 ± 0.369 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 23.992 ± 0.879 | 12.837 ± 3.205 | - | 13.503 ± 0.362 | 12.773 ± 6.137 |

## Sampling and unsupported routes

| Case | Backend | Ops/sample | Samples | Scope / reason |
| --- | --- | ---: | ---: | --- |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_256 | tenferro-cuda-trace | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_256 | tenferro-cuda-trace | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_256 | tenferro-cuda-trace | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_256 | tenferro-cuda-trace | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_256 | tenferro-cuda-trace | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_256 | tenferro-cuda-trace | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_256 | tenferro-cuda-trace | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_256 | tenferro-cuda-trace | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_256 | tenferro-cuda-trace | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_256 | tenferro-cuda-trace | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_256 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_256 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_256 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_256 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_256 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_256 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_256 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_256 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_256 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_256 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_256 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_256 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_256 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_256 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_256 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_256 | jax-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_256 | jax-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_256 | jax-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_256 | jax-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

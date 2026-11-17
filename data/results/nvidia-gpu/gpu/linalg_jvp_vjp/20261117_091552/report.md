# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261117_091552/gpu_linalg_jvp_vjp.yaml`
- Timestamp: `2026-11-17T09:15:52+00:00`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100-SXM4-80GB`
- UUID: `GPU-704abe6c-29a2-539e-7ccd-01f9de4eed3b`
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

- Model: `AMD EPYC 7713 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `128`
- Sockets: `2`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `2`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-147-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-7; 8-15; 16-23; 24-31; 32-39; 40-47; 48-55; 56-63; 64-71; 72-79; 80-87; 88-95; 96-103; 104-111; 112-119; 120-127`
- CPU boost: `enabled`
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
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 5.576 ± 0.096 | 6.082 ± 0.044 | - | 3.119 ± 0.029 | 3.497 ± 0.045 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 5.612 ± 0.029 | 3.194 ± 0.012 | - | 3.132 ± 0.085 | 3.529 ± 0.107 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 1.619 ± 0.012 | 2.925 ± 0.014 | - | 1.160 ± 0.003 | 1.065 ± 0.030 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 1.629 ± 0.068 | 1.767 ± 0.007 | - | 0.917 ± 0.004 | 1.107 ± 0.022 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 2.845 ± 0.022 | 4.157 ± 0.037 | - | 1.500 ± 0.010 | 1.473 ± 0.908 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 2.812 ± 0.015 | 2.297 ± 0.023 | - | 1.475 ± 0.008 | 1.826 ± 0.059 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 1.526 ± 0.005 | 1.897 ± 0.012 | - | 0.961 ± 0.009 | 0.988 ± 0.045 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 1.514 ± 0.011 | 1.207 ± 0.006 | - | 1.635 ± 0.030 | 1.020 ± 0.039 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 23.602 ± 0.105 | 23.828 ± 0.137 | - | 15.479 ± 2.799 | 14.500 ± 2.643 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 23.485 ± 0.029 | 12.037 ± 0.017 | - | 12.542 ± 0.480 | 14.510 ± 2.087 |

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
| linalg_ad_large_grad_sum_lu_vjp_f64_256 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_256 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_256 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_256 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_256 | jax-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_256 | jax-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
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

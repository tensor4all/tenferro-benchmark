# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_ad_latency`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261117_091552/gpu_linalg_ad_latency.yaml`
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
Standard and latency AD rows time one prepared operation plus device completion per call on every backend, repeated in bounded batches. tenferro uses Runtime::run on a reusable caller-held CUDA engine. Fixtures, transfers, graph construction, compilation, preparation and initialization are untimed; output conversion and verification follow the stopped clock. Small rows measure API latency, not kernel throughput. Legacy session-entry measurements require recollection (issue #309). Any unsupported row retains its actual reason in the raw records.
Inputs are uploaded to the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization without downloading AD outputs in the timed region.
Small cases (n=2, 4, 8) use bounded batches of repeated declared operations with retained outputs. They compare steady-state API latency, not isolated GPU throughput. Native synchronization completes each operation; numerical downloads are outside timing.
This suite exists to measure single-call latency and per-op overhead: the device kernels are a small fraction of each row, so the numbers are not GPU throughput. GPU-sized AD results live in `result/nvidia-gpu/gpu/linalg_jvp_vjp.md`.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | tenferro-rs CUDA eager, per call, incl. AD prepare | tenferro-rs CUDA trace (deferred batch throughput) | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|---:|---:|
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.433 ± 0.009 | - | - | 0.504 ± 0.024 | 0.401 ± 0.042 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.442 ± 0.048 | - | - | 0.516 ± 0.071 | 0.355 ± 0.003 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 0.480 ± 0.008 | - | - | 0.703 ± 0.012 | 0.222 ± 0.046 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 0.480 ± 0.021 | - | - | 0.703 ± 0.191 | 0.206 ± 0.014 |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | 0.612 ± 0.013 | - | - | 0.517 ± 0.027 | 0.249 ± 0.027 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 0.606 ± 0.005 | - | - | 0.619 ± 0.016 | 0.225 ± 0.012 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 0.511 ± 0.010 | - | - | 0.648 ± 0.022 | 0.219 ± 0.045 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 0.513 ± 0.006 | - | - | 0.663 ± 0.036 | 0.205 ± 0.037 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 0.854 ± 0.003 | - | - | 0.698 ± 0.025 | 0.522 ± 0.020 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 0.844 ± 0.002 | - | - | 0.507 ± 0.024 | 0.499 ± 0.015 |

## Sampling and unsupported routes

| Case | Backend | Ops/sample | Samples | Scope / reason |
| --- | --- | ---: | ---: | --- |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_4 | tenferro-cuda-trace | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_4 | tenferro-cuda-trace | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_4 | tenferro-cuda-trace | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_4 | tenferro-cuda-trace | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_4 | tenferro-cuda-trace | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_4 | tenferro-cuda-trace | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_4 | tenferro-cuda-trace | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_4 | tenferro-cuda-trace | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_4 | tenferro-cuda-trace | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_4 | tenferro-cuda-trace | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_4 | pytorch-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_4 | jax-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_4 | jax-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_4 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_4 | jax-cuda | 16 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_4 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_4 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_4 | jax-cuda | 16 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_4 | jax-cuda | 16 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_4 | jax-cuda | 16 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_4 | jax-cuda | 16 | 15 | steady_state_host_api_plus_device_sync |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

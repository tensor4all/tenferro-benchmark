# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_ad_latency`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261126_062950/gpu_linalg_ad_latency.yaml`
- Timestamp: `2026-11-26T06:29:50+00:00`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd`
- Memory: `80 GiB`
- Driver version: `595.112.07`
- CUDA version: `13.2`
- CUDA runtime: `12.8`
- cuDNN version: `91900`
- Container: `devcontainer`
- CUDA toolkit: `unknown`
- CUDA toolkit root: `/opt/tenferro-benchmark-ci/cuda`
- cuTENSOR version: `2.2.0`
- cuSOLVER version: `11.7.3`

## CPU Information

- Model: `AMD EPYC 7543 32-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `128`
- Sockets: `2`
- Cores per socket: `32`
- Threads per core: `2`
- NUMA nodes: `2`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-138-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,64-67; 4-7,68-71; 8-11,72-75; 12-15,76-79; 16-19,80-83; 20-23,84-87; 24-27,88-91; 28-31,92-95; 32-35,96-99; 36-39,100-103; 40-43,104-107; 44-47,108-111; 48-51,112-115; 52-55,116-119; 56-59,120-123; 60-63,124-127`
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
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.366 ± 0.005 | - | - | 0.435 ± 0.024 | 0.328 ± 0.007 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.369 ± 0.004 | - | - | 0.385 ± 0.025 | 0.356 ± 0.010 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 0.386 ± 0.003 | - | - | 0.654 ± 0.025 | 0.223 ± 0.006 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 0.380 ± 0.003 | - | - | 0.563 ± 0.020 | 0.205 ± 0.023 |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | 0.491 ± 0.004 | - | - | 0.459 ± 0.016 | 0.239 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 0.484 ± 0.005 | - | - | 0.853 ± 0.050 | 0.238 ± 0.012 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 0.412 ± 0.004 | - | - | 0.595 ± 0.022 | 0.331 ± 0.075 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 0.408 ± 0.003 | - | - | 0.560 ± 0.029 | 0.250 ± 0.003 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 0.786 ± 0.003 | - | - | 0.577 ± 0.005 | 0.606 ± 0.017 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 0.776 ± 0.010 | - | - | 0.539 ± 0.050 | 0.479 ± 0.003 |

## Sampling and unsupported routes

| Case | Backend | Ops/sample | Samples | Scope / reason |
| --- | --- | ---: | ---: | --- |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_4 | tenferro-cuda-trace | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_4 | tenferro-cuda-trace | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_4 | tenferro-cuda-trace | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_4 | tenferro-cuda-trace | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_4 | tenferro-cuda-trace | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_4 | tenferro-cuda-trace | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_4 | tenferro-cuda-trace | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_4 | tenferro-cuda-trace | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_4 | tenferro-cuda-trace | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_4 | tenferro-cuda-trace | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_4 | pytorch-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_4 | pytorch-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_4 | jax-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_4 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_4 | jax-cuda | 16 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_4 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_4 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_4 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_4 | jax-cuda | 16 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_4 | jax-cuda | 16 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_4 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_4 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

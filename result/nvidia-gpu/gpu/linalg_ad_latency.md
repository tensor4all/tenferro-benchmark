# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_ad_latency`
- Suite file: `/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/benchmarks/gpu/linalg_ad_latency.yaml`
- Timestamp: `2026-10-17T10:54:36.874152+00:00`
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
Small cases (n=2, 4, 8) use bounded batches of repeated declared operations with retained outputs. They are public-path latency diagnostics, not isolated GPU throughput or shared-session comparisons. Native synchronization ends each batch; numerical downloads are outside timing.
This suite exists to measure single-call latency and per-op overhead: the device kernels are a small fraction of each row, so the numbers are not GPU throughput. GPU-sized AD results live in `result/nvidia-gpu/gpu/linalg_jvp_vjp.md`.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 0.533 ± 0.002 | 0.445 ± 0.014 | 0.236 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.561 ± 0.042 | 0.443 ± 0.030 | 0.282 ± 0.004 |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 0.618 ± 0.062 | 0.419 ± 0.013 | 0.403 ± 0.028 |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 0.544 ± 0.035 | 0.470 ± 0.029 | 0.237 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.607 ± 0.060 | 0.478 ± 0.014 | 0.283 ± 0.003 |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 0.578 ± 0.006 | 0.474 ± 0.013 | 0.412 ± 0.032 |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 0.755 ± 0.024 | 0.683 ± 0.007 | 0.159 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 0.775 ± 0.007 | 0.624 ± 0.068 | 0.159 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 0.785 ± 0.042 | 0.580 ± 0.006 | 0.162 ± 0.006 |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 0.781 ± 0.071 | 0.816 ± 0.046 | 0.144 ± 0.015 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 0.811 ± 0.038 | 0.822 ± 0.007 | 0.155 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 0.799 ± 0.004 | 0.823 ± 0.013 | 0.155 ± 0.003 |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | 0.937 ± 0.008 | 0.479 ± 0.090 | 0.188 ± 0.003 |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | 1.094 ± 0.083 | 0.496 ± 0.074 | 0.189 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | `8x8` | 1.070 ± 0.010 | 0.504 ± 0.063 | 0.185 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | 1.116 ± 0.085 | 0.809 ± 0.038 | 0.174 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 1.194 ± 0.007 | 0.797 ± 0.034 | 0.175 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | 1.081 ± 0.046 | 0.748 ± 0.019 | 0.181 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 1.150 ± 0.013 | 0.662 ± 0.054 | 0.168 ± 0.002 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 1.171 ± 0.005 | 0.660 ± 0.101 | 0.161 ± 0.003 |
| small | `grad_sum_solve_jvp` | f64 | `8x8,rhs=1` | 1.185 ± 0.102 | 0.644 ± 0.095 | 0.163 ± 0.003 |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 1.149 ± 0.007 | 0.782 ± 0.010 | 0.166 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 1.315 ± 0.103 | 0.777 ± 0.052 | 0.164 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 1.180 ± 0.012 | 0.794 ± 0.065 | 0.162 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | `2x2` | 0.537 ± 0.004 | 0.513 ± 0.050 | 0.390 ± 0.005 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 0.698 ± 0.003 | 0.640 ± 0.010 | 0.392 ± 0.022 |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 0.831 ± 0.052 | 0.697 ± 0.043 | 0.477 ± 0.020 |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 0.565 ± 0.005 | 0.420 ± 0.016 | 0.255 ± 0.007 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 0.748 ± 0.033 | 0.531 ± 0.055 | 0.396 ± 0.014 |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 0.840 ± 0.027 | 0.558 ± 0.036 | 0.470 ± 0.007 |

## Sampling and unsupported routes

| Case | Backend | Ops/sample | Samples | Scope / reason |
| --- | --- | ---: | ---: | --- |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_2 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_2 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_qr_jvp_f64_2 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_qr_vjp_f64_2 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_eigh_jvp_f64_2 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_eigh_vjp_f64_2 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_lu_jvp_f64_2 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_lu_vjp_f64_2 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_solve_jvp_f64_2 | tenferro-cuda-trace | 2 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_solve_vjp_f64_2 | tenferro-cuda-trace | 2 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_4 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_4 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_qr_jvp_f64_4 | tenferro-cuda-trace | 2 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_qr_vjp_f64_4 | tenferro-cuda-trace | 2 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_eigh_jvp_f64_4 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_eigh_vjp_f64_4 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_lu_jvp_f64_4 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_lu_vjp_f64_4 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_solve_jvp_f64_4 | tenferro-cuda-trace | 2 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_solve_vjp_f64_4 | tenferro-cuda-trace | 2 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_8 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_8 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_qr_jvp_f64_8 | tenferro-cuda-trace | 2 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_qr_vjp_f64_8 | tenferro-cuda-trace | 2 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_eigh_jvp_f64_8 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_eigh_vjp_f64_8 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_lu_jvp_f64_8 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_lu_vjp_f64_8 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_solve_jvp_f64_8 | tenferro-cuda-trace | 2 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_solve_vjp_f64_8 | tenferro-cuda-trace | 2 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_2 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_2 | pytorch-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_2 | pytorch-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_2 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_2 | pytorch-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_2 | pytorch-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_2 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_2 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_2 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_2 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_4 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_4 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_4 | pytorch-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_4 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_4 | pytorch-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_4 | pytorch-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_4 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_4 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_4 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_4 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_8 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_8 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_8 | pytorch-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_8 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_8 | pytorch-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_8 | pytorch-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_8 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_8 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_8 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_8 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_2 | jax-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_2 | jax-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_2 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_2 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_2 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_2 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_2 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_2 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_2 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_2 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_4 | jax-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_4 | jax-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_4 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_4 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_4 | jax-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_4 | jax-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_4 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_4 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_4 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_4 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_8 | jax-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_8 | jax-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_8 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_8 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_8 | jax-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_8 | jax-cuda | 8 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_8 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_8 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_8 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_8 | jax-cuda | 16 | 7 | steady_state_host_api_plus_device_sync |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

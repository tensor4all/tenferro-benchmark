# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_ad_latency`
- Suite file: `/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/benchmarks/gpu/linalg_ad_latency.yaml`
- Timestamp: `2026-10-17T00:06:43.103915+00:00`
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
Small cases (n=2, 4, 8) use bounded batches of repeated declared operations with retained outputs. They are public-path latency diagnostics, not isolated GPU throughput or shared-session comparisons. Native synchronization ends each batch; numerical downloads are outside timing.
This suite exists to measure single-call latency and per-op overhead: the device kernels are a small fraction of each row, so the numbers are not GPU throughput. GPU-sized AD results live in `result/nvidia-gpu/gpu/linalg_jvp_vjp.md`.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 0.545 ± 0.005 | 0.383 ± 0.012 | 0.234 ± 0.004 |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.613 ± 0.080 | 0.391 ± 0.013 | 0.282 ± 0.003 |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 0.588 ± 0.060 | 0.425 ± 0.012 | 0.404 ± 0.004 |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 0.550 ± 0.033 | 0.359 ± 0.087 | 0.235 ± 0.004 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.556 ± 0.016 | 0.478 ± 0.035 | 0.285 ± 0.005 |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 0.590 ± 0.008 | 0.518 ± 0.046 | 0.404 ± 0.009 |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 0.765 ± 0.003 | 0.585 ± 0.008 | 0.160 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 0.787 ± 0.005 | 0.589 ± 0.039 | 0.153 ± 0.004 |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 0.792 ± 0.055 | 0.587 ± 0.022 | 0.158 ± 0.004 |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 0.787 ± 0.003 | 0.756 ± 0.016 | 0.155 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 0.800 ± 0.001 | 0.769 ± 0.007 | 0.150 ± 0.007 |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 0.871 ± 0.112 | 0.787 ± 0.047 | 0.151 ± 0.003 |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | 1.042 ± 0.012 | 0.433 ± 0.022 | 0.186 ± 0.003 |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | 1.073 ± 0.077 | 0.437 ± 0.057 | 0.179 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | `8x8` | 1.148 ± 0.142 | 0.436 ± 0.013 | 0.185 ± 0.003 |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | 1.067 ± 0.018 | 0.752 ± 0.025 | 0.176 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 1.096 ± 0.134 | 0.752 ± 0.025 | 0.178 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | 1.252 ± 0.013 | 0.748 ± 0.024 | 0.176 ± 0.003 |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 1.140 ± 0.004 | 0.568 ± 0.022 | 0.165 ± 0.003 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 1.301 ± 0.136 | 0.578 ± 0.039 | 0.160 ± 0.002 |
| small | `grad_sum_solve_jvp` | f64 | `8x8,rhs=1` | 1.084 ± 0.004 | 0.564 ± 0.017 | 0.163 ± 0.003 |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 1.159 ± 0.006 | 0.772 ± 0.151 | 0.163 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 1.167 ± 0.010 | 0.725 ± 0.009 | 0.160 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 1.186 ± 0.008 | 0.723 ± 0.021 | 0.163 ± 0.003 |
| small | `grad_sum_svd_s_jvp` | f64 | `2x2` | 0.541 ± 0.004 | 0.515 ± 0.010 | 0.381 ± 0.011 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 0.698 ± 0.005 | 0.650 ± 0.019 | 0.384 ± 0.004 |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 0.790 ± 0.005 | 0.613 ± 0.003 | 0.472 ± 0.005 |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 0.575 ± 0.006 | 0.374 ± 0.012 | 0.260 ± 0.004 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 0.731 ± 0.007 | 0.515 ± 0.007 | 0.393 ± 0.004 |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 0.824 ± 0.009 | 0.555 ± 0.005 | 0.467 ± 0.003 |

## Sampling and unsupported routes

| Case | Backend | Ops/sample | Samples | Scope / reason |
| --- | --- | ---: | ---: | --- |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_2 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_2 | tenferro-cuda-trace | 4 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_qr_jvp_f64_2 | tenferro-cuda-trace | 2 | 7 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_qr_vjp_f64_2 | tenferro-cuda-trace | 2 | 7 | public_path_latency_including_internal_session_entry |
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
| linalg_ad_small_grad_sum_eigh_vjp_f64_8 | pytorch-cuda | 4 | 7 | steady_state_host_api_plus_device_sync |
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

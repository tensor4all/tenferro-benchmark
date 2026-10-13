# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_ad_latency`
- Suite file: `/workspaces/tenferro-benchmark/benchmarks/gpu/linalg_ad_latency.yaml`
- Timestamp: `2026-10-13T14:28:47.837962+00:00`
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
Small cases (n=2, 4, 8) use bounded batches of repeated declared operations with retained outputs. They are public-path latency diagnostics, not isolated GPU throughput or shared-session comparisons. Native synchronization ends each batch; numerical downloads are outside timing.
This suite exists to measure single-call latency and per-op overhead: the device kernels are a small fraction of each row, so the numbers are not GPU throughput. GPU-sized AD results live in `result/nvidia-gpu/gpu/linalg_jvp_vjp.md`.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 1.115 ± 0.009 | 0.669 ± 0.032 | 0.388 ± 0.015 |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.941 ± 0.017 | 0.751 ± 0.102 | 0.579 ± 0.067 |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 1.150 ± 0.031 | 0.825 ± 0.083 | 0.854 ± 0.053 |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 1.085 ± 0.031 | 0.905 ± 0.112 | 0.409 ± 0.034 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.952 ± 0.029 | 0.889 ± 0.050 | 0.659 ± 0.063 |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 1.156 ± 0.027 | 0.997 ± 0.037 | 0.824 ± 0.010 |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 1.677 ± 0.155 | 1.084 ± 0.061 | 0.251 ± 0.007 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 1.275 ± 0.038 | 1.161 ± 0.111 | 0.283 ± 0.027 |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 1.358 ± 0.024 | 1.113 ± 0.077 | 0.287 ± 0.043 |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 1.435 ± 0.091 | 1.675 ± 0.150 | 0.253 ± 0.012 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 1.423 ± 0.023 | 1.716 ± 0.145 | 0.277 ± 0.037 |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 1.661 ± 0.097 | 1.682 ± 0.140 | 0.312 ± 0.049 |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | 2.709 ± 0.044 | 0.807 ± 0.027 | 0.307 ± 0.007 |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | 1.738 ± 0.034 | 0.833 ± 0.058 | 0.400 ± 0.046 |
| small | `grad_sum_qr_jvp` | f64 | `8x8` | 1.978 ± 0.058 | 0.805 ± 0.031 | 0.362 ± 0.061 |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | 2.882 ± 0.045 | 1.635 ± 0.099 | 0.306 ± 0.003 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 1.846 ± 0.112 | 1.577 ± 0.108 | 0.312 ± 0.007 |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | 2.120 ± 0.042 | 1.621 ± 0.157 | 0.350 ± 0.058 |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 2.522 ± 0.079 | 1.036 ± 0.119 | 0.282 ± 0.008 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 2.184 ± 0.140 | 1.078 ± 0.146 | 0.377 ± 0.084 |
| small | `grad_sum_solve_jvp` | f64 | `8x8,rhs=1` | 2.639 ± 0.070 | 1.076 ± 0.152 | 0.377 ± 0.051 |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 2.593 ± 0.043 | 1.626 ± 0.143 | 0.279 ± 0.007 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 2.499 ± 0.143 | 1.519 ± 0.119 | 0.308 ± 0.028 |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 3.042 ± 0.150 | 1.476 ± 0.112 | 0.329 ± 0.051 |
| small | `grad_sum_svd_s_jvp` | f64 | `2x2` | 0.988 ± 0.012 | 1.039 ± 0.056 | 0.679 ± 0.011 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 1.529 ± 0.080 | 1.335 ± 0.203 | 0.946 ± 0.045 |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 1.836 ± 0.053 | 1.437 ± 0.164 | 1.339 ± 0.043 |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 1.387 ± 0.023 | 0.880 ± 0.136 | 0.468 ± 0.017 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 1.424 ± 0.033 | 1.181 ± 0.116 | 1.040 ± 0.060 |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 1.962 ± 0.017 | 1.325 ± 0.035 | 1.408 ± 0.043 |

## Sampling and unsupported routes

| Case | Backend | Ops/sample | Samples | Scope / reason |
| --- | --- | ---: | ---: | --- |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_2 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_2 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_qr_jvp_f64_2 | tenferro-cuda-trace | 1 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_qr_vjp_f64_2 | tenferro-cuda-trace | 1 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_eigh_jvp_f64_2 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_eigh_vjp_f64_2 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_lu_jvp_f64_2 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_lu_vjp_f64_2 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_solve_jvp_f64_2 | tenferro-cuda-trace | 1 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_solve_vjp_f64_2 | tenferro-cuda-trace | 1 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_4 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_4 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_qr_jvp_f64_4 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_qr_vjp_f64_4 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_eigh_jvp_f64_4 | tenferro-cuda-trace | 4 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_eigh_vjp_f64_4 | tenferro-cuda-trace | 4 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_lu_jvp_f64_4 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_lu_vjp_f64_4 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_solve_jvp_f64_4 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_solve_vjp_f64_4 | tenferro-cuda-trace | 1 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_8 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_8 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_qr_jvp_f64_8 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_qr_vjp_f64_8 | tenferro-cuda-trace | 1 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_eigh_jvp_f64_8 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_eigh_vjp_f64_8 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_lu_jvp_f64_8 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_lu_vjp_f64_8 | tenferro-cuda-trace | 2 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_solve_jvp_f64_8 | tenferro-cuda-trace | 1 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_solve_vjp_f64_8 | tenferro-cuda-trace | 1 | 15 | public_path_latency_including_internal_session_entry |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_2 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_2 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_2 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_2 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_2 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_2 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_2 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_2 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_2 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_2 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_4 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_4 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_4 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_4 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_4 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_4 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_4 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_4 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_8 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_8 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_8 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_8 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_8 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_8 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_8 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_8 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_8 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_8 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_2 | jax-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_2 | jax-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_2 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_2 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_2 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_2 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_2 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_2 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_2 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_2 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_4 | jax-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_4 | jax-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_4 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_4 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_4 | jax-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_4 | jax-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_4 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_4 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_4 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_4 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_jvp_f64_8 | jax-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_svd_s_vjp_f64_8 | jax-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_jvp_f64_8 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_qr_vjp_f64_8 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_jvp_f64_8 | jax-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_eigh_vjp_f64_8 | jax-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_jvp_f64_8 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_lu_vjp_f64_8 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_jvp_f64_8 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_small_grad_sum_solve_vjp_f64_8 | jax-cuda | 8 | 15 | steady_state_host_api_plus_device_sync |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

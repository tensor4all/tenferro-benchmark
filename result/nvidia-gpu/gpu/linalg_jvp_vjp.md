# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261112_122130/gpu_linalg_jvp_vjp.yaml`
- Timestamp: `2026-11-12T12:21:30+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100-SXM4-80GB`
- UUID: `GPU-ac7cac48-cc6e-626e-35e3-ab12c40f953a`
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

- Model: `Intel(R) Xeon(R) Platinum 8470`
- Vendor: `GenuineIntel`
- Logical CPUs: `192`
- Sockets: `2`
- Cores per socket: `48`
- Threads per core: `2`
- NUMA nodes: `2`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-147-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-95; 96-191`
- Container: `devcontainer`

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext` on CUDA; PyTorch uses `torch.func.jvp` / `vjp` on CUDA.
JAX uses an explicitly lowered/compiled whole-operation JIT; compilation and lazy initialization occur before sampling. Primal loss and first-order derivatives with respect to A are both returned; the check that backs each row is stated below, and a row that carries none is marked as not checked. RHS and derivative seeds are prepared before timing.
The `tenferro-rs CUDA eager` column runs the same logical case through the eager API on one caller-held CUDA engine. Unlike the trace row, the eager API exposes no separate preparation step, so its own AD preparation is inside the sample together with the forward pass; the forward and derivative scope matches the trace column.
Standard and latency AD rows time one prepared operation plus device completion per call on every backend, repeated in bounded batches. tenferro uses Runtime::run on a reusable caller-held CUDA engine. Fixtures, transfers, graph construction, compilation, preparation and initialization are untimed; output conversion and verification follow the stopped clock. Small rows measure API latency, not kernel throughput. Legacy session-entry measurements require recollection (issue #309). Any unsupported row retains its actual reason in the raw records.
Inputs are uploaded to the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization without downloading AD outputs in the timed region.
Verification modes: `analytic` (compared element by element against a closed-form derivative built on the accelerator); `cpu_reference` (re-executed on the CPU in f64 and compared element by element); `finite_difference` (directional derivative compared with a central difference of the primal loss at a step derived from f64 precision). Every mode is held to the suite's own rtol/atol.

## Linalg JVP/VJP Benchmark Items

### Throughput regime

Compute-bound rows: every case here is over its family's floor in `docs/regime-floors.md`. They are never averaged with, or read as a ratio against, the other section.

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | tenferro-rs CUDA eager, per call, incl. AD prepare | tenferro-rs CUDA trace (deferred batch throughput) | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `4096x4096` | unsupported | unsupported | - | 230.485 ± 0.120 | 223.308 ± 0.175 |
| large | `grad_sum_eigh_vjp` | f64 | `4096x4096` | unsupported | unsupported | - | 215.924 ± 0.214 | 223.863 ± 0.135 |
| large | `grad_sum_lu_jvp` | f64 | `4096x4096` | 224.412 ± 0.084 | 265.666 ± 0.066 | - | 90.105 ± 0.051 | 48.246 ± 0.057 |
| large | `grad_sum_lu_vjp` | f64 | `4096x4096` | 224.425 ± 0.040 | 171.164 ± 0.044 | - | 55.606 ± 0.039 | 48.798 ± 0.062 |
| large | `grad_sum_qr_jvp` | f64 | `4096x4096` | 143.883 ± 0.067 | 191.889 ± 0.116 | - | 85.203 ± 0.076 | 84.583 ± 0.055 |
| large | `grad_sum_qr_vjp` | f64 | `4096x4096` | 143.209 ± 0.064 | 134.719 ± 1.345 | - | 84.107 ± 0.060 | 83.238 ± 0.055 |
| large | `grad_sum_solve_jvp` | f64 | `4096x4096,rhs=1` | 42.751 ± 0.037 | 43.667 ± 0.072 | - | 22.016 ± 0.096 | 21.602 ± 0.106 |
| large | `grad_sum_solve_vjp` | f64 | `4096x4096,rhs=1` | 43.008 ± 0.069 | 23.010 ± 0.048 | - | 41.444 ± 0.060 | 21.863 ± 0.024 |
| large | `grad_sum_svd_s_jvp` | f64 | `4096x4096` | unsupported | unsupported | - | 3188.068 ± 0.523 | 3533.077 ± 2.714 |
| large | `grad_sum_svd_s_vjp` | f64 | `4096x4096` | unsupported | unsupported | - | 3160.138 ± 0.821 | 3523.024 ± 1.639 |

### Latency regime

Per-call latency rows: the device kernels are a small fraction of each number, so these are not GPU throughput.

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | tenferro-rs CUDA eager, per call, incl. AD prepare | tenferro-rs CUDA trace (deferred batch throughput) | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 5.470 ± 0.018 | 6.057 ± 0.033 | - | 3.067 ± 0.038 | 3.555 ± 0.007 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 5.527 ± 0.019 | 3.107 ± 0.017 | - | 2.998 ± 0.040 | 3.521 ± 0.007 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 1.591 ± 0.009 | 2.844 ± 0.029 | - | 1.149 ± 0.012 | 1.091 ± 0.011 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 1.567 ± 0.004 | 1.690 ± 0.015 | - | 0.897 ± 0.004 | 1.119 ± 0.012 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 2.851 ± 0.016 | 4.239 ± 0.234 | - | 1.478 ± 0.012 | 1.473 ± 0.014 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 2.841 ± 0.025 | 2.145 ± 0.016 | - | 1.458 ± 0.009 | 1.508 ± 0.010 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 1.517 ± 0.004 | 1.876 ± 0.017 | - | 0.942 ± 0.028 | 1.000 ± 0.008 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 1.502 ± 0.006 | 1.154 ± 0.019 | - | 1.563 ± 0.018 | 1.015 ± 0.006 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 23.340 ± 0.022 | 23.874 ± 0.044 | - | 12.889 ± 0.094 | 14.570 ± 0.037 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 23.366 ± 0.022 | 12.006 ± 0.018 | - | 12.467 ± 0.048 | 11.775 ± 0.029 |

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
| linalg_ad_large_grad_sum_svd_s_jvp_f64_4096 | tenferro-cuda-trace | — | — | the CUDA runtime's pending-resource budget cannot admit the decomposition workspace at this size: initial run: svd_values: runtime state failure: insufficient pending-resource budget for 411699200 bytes |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_4096 | tenferro-cuda-trace | — | — | the CUDA runtime's pending-resource budget cannot admit the decomposition workspace at this size: initial run: svd_values: runtime state failure: insufficient pending-resource budget for 411699200 bytes |
| linalg_ad_large_grad_sum_qr_jvp_f64_4096 | tenferro-cuda-trace | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_4096 | tenferro-cuda-trace | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_4096 | tenferro-cuda-trace | — | — | the CUDA runtime's pending-resource budget cannot admit the decomposition workspace at this size: initial run: eigh_values: runtime state failure: insufficient pending-resource budget for 541296152 bytes |
| linalg_ad_large_grad_sum_eigh_vjp_f64_4096 | tenferro-cuda-trace | — | — | the CUDA runtime's pending-resource budget cannot admit the decomposition workspace at this size: initial run: eigh_values: runtime state failure: insufficient pending-resource budget for 541296152 bytes |
| linalg_ad_large_grad_sum_lu_jvp_f64_4096 | tenferro-cuda-trace | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_4096 | tenferro-cuda-trace | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_4096 | tenferro-cuda-trace | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_4096 | tenferro-cuda-trace | 1 | 15 | steady_state_host_api_plus_device_sync |
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
| linalg_ad_large_grad_sum_svd_s_jvp_f64_4096 | tenferro-cuda-eager | — | — | the CUDA runtime's pending-resource budget cannot admit the decomposition workspace at this size: eager forward: svd: runtime state failure: insufficient pending-resource budget for 411699200 bytes |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_4096 | tenferro-cuda-eager | — | — | the CUDA runtime's pending-resource budget cannot admit the decomposition workspace at this size: eager forward: svd: runtime state failure: insufficient pending-resource budget for 411699200 bytes |
| linalg_ad_large_grad_sum_qr_jvp_f64_4096 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_4096 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_4096 | tenferro-cuda-eager | — | — | the CUDA runtime's pending-resource budget cannot admit the decomposition workspace at this size: eager forward: eigh: runtime state failure: insufficient pending-resource budget for 541296152 bytes |
| linalg_ad_large_grad_sum_eigh_vjp_f64_4096 | tenferro-cuda-eager | — | — | the CUDA runtime's pending-resource budget cannot admit the decomposition workspace at this size: eager forward: eigh: runtime state failure: insufficient pending-resource budget for 541296152 bytes |
| linalg_ad_large_grad_sum_lu_jvp_f64_4096 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_4096 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_4096 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_4096 | tenferro-cuda-eager | 1 | 15 | steady_state_host_api_plus_device_sync |
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
| linalg_ad_large_grad_sum_svd_s_jvp_f64_4096 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_4096 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_4096 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_4096 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_4096 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_4096 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_4096 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_4096 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_4096 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_4096 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
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
| linalg_ad_large_grad_sum_svd_s_jvp_f64_4096 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_4096 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_4096 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_4096 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_4096 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_4096 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_4096 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_4096 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_4096 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_4096 | jax-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

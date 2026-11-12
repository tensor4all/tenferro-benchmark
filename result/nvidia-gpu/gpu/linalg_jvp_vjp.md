# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261112_085507/gpu_linalg_jvp_vjp.yaml`
- Timestamp: `2026-11-12T08:55:07+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100-SXM4-80GB`
- UUID: `GPU-7a9c1592-6482-bb81-7d03-d21878526fb7`
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

- Model: `AMD EPYC 7742 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `128`
- Sockets: `2`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `2`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-147-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3; 4-7; 8-11; 12-15; 16-19; 20-23; 24-27; 28-31; 32-35; 36-39; 40-43; 44-47; 48-51; 52-55; 56-59; 60-63; 64-67; 68-71; 72-75; 76-79; 80-83; 84-87; 88-91; 92-95; 96-99; 100-103; 104-107; 108-111; 112-115; 116-119; 120-123; 124-127`
- CPU boost: `enabled`
- Container: `devcontainer`

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext` on CUDA; PyTorch uses `torch.func.jvp` / `vjp` on CUDA.
JAX uses an explicitly lowered/compiled whole-operation JIT; compilation and lazy initialization occur before sampling. Primal loss and first-order derivatives with respect to A are both returned; the check that backs each row is stated below, and a row that carries none is marked as not checked. RHS and derivative seeds are prepared before timing.
Standard and latency AD rows time one prepared operation plus device completion per call on every backend, repeated in bounded batches. tenferro uses Runtime::run on a reusable caller-held CUDA engine. Fixtures, transfers, graph construction, compilation, preparation and initialization are untimed; output conversion and verification follow the stopped clock. Small rows measure API latency, not kernel throughput. Legacy session-entry measurements require recollection (issue #309). Any unsupported row retains its actual reason in the raw records.
Inputs are uploaded to the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization without downloading AD outputs in the timed region.
Verification modes: `analytic` (compared element by element against a closed-form derivative built on the accelerator); `cpu_reference` (re-executed on the CPU in f64 and compared element by element); `finite_difference` (directional derivative compared with a central difference of the primal loss at a step derived from f64 precision). Every mode is held to the suite's own rtol/atol.
Not measured: `linalg_ad_large_grad_sum_eigh_jvp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_eigh_jvp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_eigh_vjp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_eigh_vjp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_lu_jvp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_lu_jvp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_lu_vjp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_lu_vjp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_qr_jvp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_qr_jvp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_qr_vjp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_qr_vjp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_solve_jvp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_solve_jvp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_solve_vjp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_solve_vjp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_svd_s_jvp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_svd_s_jvp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_svd_s_vjp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_svd_s_vjp_f64_4096` on `tenferro-cuda-eager`. These cells carry no number; the run reports the failure.

## Linalg JVP/VJP Benchmark Items

### Throughput regime

Compute-bound rows: every case here is over its family's floor in `docs/regime-floors.md`. They are never averaged with, or read as a ratio against, the other section.

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | tenferro-rs CUDA eager, per call, incl. AD prepare | tenferro-rs CUDA trace (deferred batch throughput) | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `4096x4096` | unsupported | - | - | 230.519 ± 0.631 | 223.300 ± 0.362 |
| large | `grad_sum_eigh_vjp` | f64 | `4096x4096` | unsupported | - | - | 215.515 ± 0.326 | 223.402 ± 0.260 |
| large | `grad_sum_lu_jvp` | f64 | `4096x4096` | 225.375 ± 0.242 | - | - | 90.339 ± 0.184 | 48.277 ± 0.137 |
| large | `grad_sum_lu_vjp` | f64 | `4096x4096` | 225.232 ± 0.224 | - | - | 55.705 ± 0.143 | 48.824 ± 0.066 |
| large | `grad_sum_qr_jvp` | f64 | `4096x4096` | 144.395 ± 0.275 | - | - | 85.266 ± 0.167 | 84.665 ± 0.176 |
| large | `grad_sum_qr_vjp` | f64 | `4096x4096` | 143.435 ± 0.157 | - | - | 84.169 ± 0.115 | 83.290 ± 0.167 |
| large | `grad_sum_solve_jvp` | f64 | `4096x4096,rhs=1` | 42.891 ± 0.178 | - | - | 21.906 ± 0.061 | 21.531 ± 0.133 |
| large | `grad_sum_solve_vjp` | f64 | `4096x4096,rhs=1` | 43.117 ± 0.224 | - | - | 41.381 ± 0.153 | 21.816 ± 0.108 |
| large | `grad_sum_svd_s_jvp` | f64 | `4096x4096` | unsupported | - | - | 3196.078 ± 3.216 | 3532.662 ± 2.479 |
| large | `grad_sum_svd_s_vjp` | f64 | `4096x4096` | unsupported | - | - | 3167.182 ± 1.340 | 3523.856 ± 2.826 |

### Latency regime

Per-call latency rows: the device kernels are a small fraction of each number, so these are not GPU throughput.

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | tenferro-rs CUDA eager, per call, incl. AD prepare | tenferro-rs CUDA trace (deferred batch throughput) | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 5.568 ± 0.011 | - | - | 3.090 ± 0.044 | 3.474 ± 0.020 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 5.599 ± 0.011 | - | - | 3.140 ± 0.259 | 3.435 ± 0.027 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 1.629 ± 0.010 | - | - | 1.153 ± 0.003 | 1.064 ± 0.006 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 1.627 ± 0.010 | - | - | 0.913 ± 0.026 | 1.079 ± 0.008 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 2.888 ± 0.016 | - | - | 1.488 ± 0.013 | 1.477 ± 0.007 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 2.892 ± 0.017 | - | - | 1.484 ± 0.116 | 1.747 ± 0.019 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 1.550 ± 0.010 | - | - | 0.959 ± 0.037 | 0.955 ± 0.104 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 1.543 ± 0.003 | - | - | 1.675 ± 0.052 | 0.973 ± 0.008 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 23.564 ± 0.026 | - | - | 15.533 ± 2.567 | 14.291 ± 2.462 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 23.589 ± 0.024 | - | - | 12.901 ± 0.572 | 11.799 ± 0.019 |

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
| linalg_ad_large_grad_sum_solve_jvp_f64_256 | jax-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_vjp_f64_256 | jax-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
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

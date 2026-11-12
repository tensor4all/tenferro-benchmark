# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261112_075924/gpu_linalg_jvp_vjp.yaml`
- Timestamp: `2026-11-12T07:59:24+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-e55a1c63-28a0-c40b-b8fa-3d2351a78699`
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

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `252`
- Sockets: `2`
- Cores per socket: `126`
- Threads per core: `1`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-138-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-125; 126-251`
- Container: `devcontainer`

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext` on CUDA; PyTorch uses `torch.func.jvp` / `vjp` on CUDA.
JAX uses an explicitly lowered/compiled whole-operation JIT; compilation and lazy initialization occur before sampling. Primal loss and first-order derivatives with respect to A are both returned; the check that backs each row is stated below, and a row that carries none is marked as not checked. RHS and derivative seeds are prepared before timing.
Standard and latency AD rows time one prepared operation plus device completion per call on every backend, repeated in bounded batches. tenferro uses Runtime::run on a reusable caller-held CUDA engine. Fixtures, transfers, graph construction, compilation, preparation and initialization are untimed; output conversion and verification follow the stopped clock. Small rows measure API latency, not kernel throughput. Legacy session-entry measurements require recollection (issue #309). Any unsupported row retains its actual reason in the raw records.
Inputs are uploaded to the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization without downloading AD outputs in the timed region.
Verification modes: `analytic` (compared element by element against a closed-form derivative built on the accelerator); `cpu_reference` (re-executed on the CPU in f64 and compared element by element); `finite_difference` (directional derivative compared with a central difference of the primal loss at a step derived from f64 precision). Every mode is held to the suite's own rtol/atol.
Not measured: `linalg_ad_large_grad_sum_eigh_jvp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_eigh_jvp_f64_256` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_eigh_jvp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_eigh_jvp_f64_4096` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_eigh_vjp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_eigh_vjp_f64_256` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_eigh_vjp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_eigh_vjp_f64_4096` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_lu_jvp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_lu_jvp_f64_256` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_lu_jvp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_lu_jvp_f64_4096` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_lu_vjp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_lu_vjp_f64_256` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_lu_vjp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_lu_vjp_f64_4096` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_qr_jvp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_qr_jvp_f64_256` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_qr_jvp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_qr_jvp_f64_4096` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_qr_vjp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_qr_vjp_f64_256` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_qr_vjp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_qr_vjp_f64_4096` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_solve_jvp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_solve_jvp_f64_256` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_solve_jvp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_solve_jvp_f64_4096` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_solve_vjp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_solve_vjp_f64_256` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_solve_vjp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_solve_vjp_f64_4096` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_svd_s_jvp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_svd_s_jvp_f64_256` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_svd_s_jvp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_svd_s_jvp_f64_4096` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_svd_s_vjp_f64_256` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_svd_s_vjp_f64_256` on `tenferro-cuda-trace`; `linalg_ad_large_grad_sum_svd_s_vjp_f64_4096` on `tenferro-cuda-eager`; `linalg_ad_large_grad_sum_svd_s_vjp_f64_4096` on `tenferro-cuda-trace`. These cells carry no number; the run reports the failure.

## Linalg JVP/VJP Benchmark Items

### Throughput regime

Compute-bound rows: every case here is over its family's floor in `docs/regime-floors.md`. They are never averaged with, or read as a ratio against, the other section.

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | tenferro-rs CUDA eager, per call, incl. AD prepare | tenferro-rs CUDA trace (deferred batch throughput) | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `4096x4096` | - | - | - | 236.418 ± 0.793 | 229.742 ± 0.824 |
| large | `grad_sum_eigh_vjp` | f64 | `4096x4096` | - | - | - | 221.203 ± 0.971 | 228.541 ± 0.874 |
| large | `grad_sum_lu_jvp` | f64 | `4096x4096` | - | - | - | 92.685 ± 0.496 | 49.331 ± 0.297 |
| large | `grad_sum_lu_vjp` | f64 | `4096x4096` | - | - | - | 56.988 ± 0.736 | 48.840 ± 0.493 |
| large | `grad_sum_qr_jvp` | f64 | `4096x4096` | - | - | - | 89.249 ± 0.587 | 88.051 ± 0.565 |
| large | `grad_sum_qr_vjp` | f64 | `4096x4096` | - | - | - | 86.796 ± 1.024 | 84.728 ± 1.320 |
| large | `grad_sum_solve_jvp` | f64 | `4096x4096,rhs=1` | - | - | - | 22.022 ± 0.089 | 21.775 ± 0.211 |
| large | `grad_sum_solve_vjp` | f64 | `4096x4096,rhs=1` | - | - | - | 41.916 ± 0.592 | 21.974 ± 0.185 |
| large | `grad_sum_svd_s_jvp` | f64 | `4096x4096` | - | - | - | 3442.921 ± 10.791 | 3536.500 ± 12.555 |
| large | `grad_sum_svd_s_vjp` | f64 | `4096x4096` | - | - | - | 3420.950 ± 4.181 | 3520.632 ± 11.339 |

### Latency regime

Per-call latency rows: the device kernels are a small fraction of each number, so these are not GPU throughput.

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | tenferro-rs CUDA eager, per call, incl. AD prepare | tenferro-rs CUDA trace (deferred batch throughput) | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | - | - | - | 3.178 ± 0.065 | 3.009 ± 0.074 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | - | - | - | 3.251 ± 0.042 | 3.050 ± 0.512 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | - | - | - | 1.202 ± 0.021 | 0.967 ± 0.022 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | - | - | - | 0.958 ± 0.040 | 0.985 ± 0.008 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | - | - | - | 1.523 ± 0.023 | 1.543 ± 0.096 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | - | - | - | 1.508 ± 0.019 | 1.554 ± 0.118 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | - | - | - | 0.998 ± 0.017 | 0.913 ± 0.041 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | - | - | - | 1.858 ± 0.100 | 0.984 ± 0.102 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | - | - | - | 12.953 ± 0.306 | 11.936 ± 0.407 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | - | - | - | 12.729 ± 0.093 | 11.870 ± 0.160 |

## Sampling and unsupported routes

| Case | Backend | Ops/sample | Samples | Scope / reason |
| --- | --- | ---: | ---: | --- |
| linalg_ad_large_grad_sum_svd_s_jvp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_svd_s_vjp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_jvp_f64_256 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_qr_vjp_f64_256 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_jvp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_eigh_vjp_f64_256 | pytorch-cuda | 1 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_jvp_f64_256 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_256 | pytorch-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_solve_jvp_f64_256 | pytorch-cuda | 2 | 15 | steady_state_host_api_plus_device_sync |
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
| linalg_ad_large_grad_sum_lu_jvp_f64_256 | jax-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
| linalg_ad_large_grad_sum_lu_vjp_f64_256 | jax-cuda | 4 | 15 | steady_state_host_api_plus_device_sync |
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

# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/elementwise`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261117_091552/gpu_elementwise.yaml`
- Timestamp: `2026-11-17T09:15:52+00:00`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Effort: `standard`; manifest version: `2`
- Selection and case accounting: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261117_091552/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

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

Median time is reported in milliseconds for `ok` records.
Inputs are prepared on the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization. tenferro-rs CUDA uses the explicit tenferro-rs synchronize API without downloading result tensors in the timed region.
Dense and einsum inputs use the same deterministic benchmark generator in the Rust and Python runners; host-to-device layout conversion remains outside the timed region.
tenferro-rs uses native column-major GPU tensors; PyTorch and vendor-wrapper columns use their native row-major framework tensors unless noted.
The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd as a QR-based cuSOLVER comparison. tenferro-rs CUDA SVD uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.
Non-`ok` cells show the structured backend status.

## gpu/elementwise / elementwise_chain / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|---:|
| elementwise_chain_f64_1k | 0.175 | 0.904 | 0.138 | 0.069 |
| elementwise_chain_f64_1m | 0.358 | 1.186 | 0.349 | 0.140 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| elementwise_chain_f64_1k | tenferro-cuda-trace | 16 | 2.81343, 2.79306, 2.79283, 2.79596, 2.79181, 2.83531, 2.81908 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1m | tenferro-cuda-trace | 8 | 2.85902, 2.85486, 2.86849, 2.87033, 2.86761, 2.86958, 2.85854 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1k | tenferro-cuda-eager | 4 | 3.62626, 3.61486, 3.62206, 3.6138, 3.62645, 3.60298, 3.60731 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1m | tenferro-cuda-eager | 2 | 2.36123, 2.38079, 3.22899, 3.25555, 2.36856, 2.37167, 2.36401 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1k | pytorch-cuda | 16 | 2.22026, 2.21088, 2.21588, 2.20885, 2.2246, 2.21008, 2.20623 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| elementwise_chain_f64_1m | pytorch-cuda | 8 | 2.79415, 2.78755, 2.79141, 2.78832, 2.79271, 2.78927, 2.97654 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| elementwise_chain_f64_1k | jax-cuda | 16 | 1.78731, 1.1429, 1.07875, 1.08262, 1.07383, 1.10267, 1.1375 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| elementwise_chain_f64_1m | jax-cuda | 4 | 0.554629, 0.564069, 0.588659, 0.667859, 0.559419, 0.509369, 0.509389 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

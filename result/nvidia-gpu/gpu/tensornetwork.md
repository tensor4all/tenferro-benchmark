# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/tensornetwork`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261107_201728/gpu_tensornetwork.yaml`
- Timestamp: `2026-11-07T20:17:28+00:00`
- tenferro-rs commit: `ffd352bdb9f98537be747101bc278db8a50a1595`
- Effort: `standard`; manifest version: `2`
- Selection and case accounting: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261107_201728/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100-SXM4-80GB`
- UUID: `GPU-3f509591-db26-6474-253e-fcb29d2b374b`
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
- Logical CPUs: `256`
- Sockets: `2`
- Cores per socket: `64`
- Threads per core: `2`
- NUMA nodes: `8`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-138-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,128-131; 4-7,132-135; 8-11,136-139; 12-15,140-143; 16-19,144-147; 20-23,148-151; 24-27,152-155; 28-31,156-159; 32-35,160-163; 36-39,164-167; 40-43,168-171; 44-47,172-175; 48-51,176-179; 52-55,180-183; 56-59,184-187; 60-63,188-191; 64-67,192-195; 68-71,196-199; 72-75,200-203; 76-79,204-207; 80-83,208-211; 84-87,212-215; 88-91,216-219; 92-95,220-223; 96-99,224-227; 100-103,228-231; 104-107,232-235; 108-111,236-239; 112-115,240-243; 116-119,244-247; 120-123,248-251; 124-127,252-255`
- CPU boost: `enabled`
- Container: `devcontainer`

Median time is reported in milliseconds for `ok` records.
Inputs are prepared on the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization. tenferro-rs CUDA uses the explicit tenferro-rs synchronize API without downloading result tensors in the timed region.
Dense and einsum inputs use the same deterministic benchmark generator in the Rust and Python runners; host-to-device layout conversion remains outside the timed region.
tenferro-rs uses native column-major GPU tensors; PyTorch and vendor-wrapper columns use their native row-major framework tensors unless noted.
The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd as a QR-based cuSOLVER comparison. tenferro-rs CUDA SVD uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.
Non-`ok` cells show the structured backend status.

## gpu/tensornetwork / tensor_network_contract / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA tree JIT |
|---|---:|---:|---:|---:|
| tensornetwork_permutation_optimized_f32 | 78.021 | 103.628 | 107.810 | 131.363 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| tensornetwork_permutation_optimized_f32 | tenferro-cuda-trace | 1 | 77.9377, 78.0058, 77.8311, 103.577, 126.029, 77.9997, 77.9629, 78.0355, 78.7228, 121.708 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| tensornetwork_permutation_optimized_f32 | tenferro-cuda-eager | 1 | 116.731, 157.889, 103.27, 103.436, 103.821, 176.158, 102.252, 102.212, 102.026, 158.605 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| tensornetwork_permutation_optimized_f32 | pytorch-cuda | 1 | 107.861, 107.739, 111.013, 108.078, 107.659, 107.723, 107.76, 107.661, 107.973, 108.087 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| tensornetwork_permutation_optimized_f32 | jax-cuda-tree-jit | 1 | 121.478, 131.5, 130.172, 132.257, 130.563, 130.723, 131.915, 131.227, 136.961, 150.46 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/elementwise`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261126_062950/gpu_elementwise.yaml`
- Timestamp: `2026-11-26T06:29:50+00:00`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Effort: `standard`; manifest version: `2`
- Selection and case accounting: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261126_062950/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

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
| elementwise_chain_f64_1k | 0.172 | 0.151 | 0.140 | 0.087 |
| elementwise_chain_f64_1m | 0.350 | 0.331 | 0.302 | 0.085 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| elementwise_chain_f64_1k | tenferro-cuda-trace | 16 | 2.73315, 2.74791, 2.73804, 2.74996, 2.71782, 2.7449, 2.74962 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1m | tenferro-cuda-trace | 8 | 2.80606, 2.80144, 2.79557, 2.80258, 2.79916, 2.79855, 2.80465 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1k | tenferro-cuda-eager | 16 | 2.36309, 2.38253, 2.41717, 2.44867, 2.60181, 2.39781, 2.42053 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1m | tenferro-cuda-eager | 8 | 2.6512, 2.64643, 2.65458, 2.64031, 2.64714, 2.6538, 2.64425 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1k | pytorch-cuda | 16 | 2.16508, 2.25437, 2.25069, 2.19938, 2.25927, 2.22478, 2.23921 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| elementwise_chain_f64_1m | pytorch-cuda | 8 | 2.41591, 2.42023, 2.41669, 2.41909, 2.41663, 2.41695, 2.41331 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| elementwise_chain_f64_1k | jax-cuda | 32 | 2.78106, 2.78858, 2.77178, 2.72518, 2.73702, 2.81895, 2.75908 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| elementwise_chain_f64_1m | jax-cuda | 16 | 1.36245, 1.36758, 1.35542, 1.36707, 1.36245, 1.35583, 1.35658 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

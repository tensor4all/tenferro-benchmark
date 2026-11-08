# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/elementwise`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261108_090627/gpu_elementwise.yaml`
- Timestamp: `2026-11-08T09:06:27+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Effort: `standard`; manifest version: `2`
- Selection and case accounting: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261108_090627/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a`
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

- Model: `AMD EPYC 9554 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `252`
- Sockets: `2`
- Cores per socket: `126`
- Threads per core: `1`
- NUMA nodes: `2`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-147-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-125; 126-251`
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
| elementwise_chain_f64_1k | 0.190 | 0.719 | 0.097 | 0.036 |
| elementwise_chain_f64_1m | 0.504 | 1.240 | 0.313 | 0.082 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| elementwise_chain_f64_1k | tenferro-cuda-trace | 16 | 3.06125, 3.05226, 3.03351, 3.03789, 3.04317, 3.04497, 3.03289 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1m | tenferro-cuda-trace | 4 | 2.01785, 2.01799, 2.01474, 2.02206, 2.01139, 2.00902, 2.01366 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1k | tenferro-cuda-eager | 4 | 2.8589, 2.97248, 2.87782, 2.87364, 2.8677, 2.92695, 2.87636 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1m | tenferro-cuda-eager | 2 | 2.49131, 2.46795, 2.47963, 2.47885, 2.48505, 2.48321, 2.47674 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1k | pytorch-cuda | 32 | 3.14554, 3.0861, 3.06976, 3.12079, 3.10252, 3.11736, 3.16156 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| elementwise_chain_f64_1m | pytorch-cuda | 8 | 2.49629, 2.50487, 2.50249, 2.50046, 2.50383, 2.49899, 2.51747 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| elementwise_chain_f64_1k | jax-cuda | 64 | 2.32517, 2.23113, 2.31052, 2.29843, 2.19724, 2.25866, 2.37422 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| elementwise_chain_f64_1m | jax-cuda | 32 | 2.62576, 2.61571, 2.61243, 2.61516, 2.61306, 2.61737, 2.61903 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

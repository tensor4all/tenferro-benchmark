# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/einsum`
- Suite file: `/home/shinaoka/work/wallclock/data/results/nvidia-gpu/gpu/routine_selection/20261105_0820/gpu_einsum.yaml`
- Timestamp: `2026-11-05T08:02:00+00:00`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`
- Effort: `standard`; manifest version: `2`
- Selection and case accounting: `/home/shinaoka/work/wallclock/data/results/nvidia-gpu/gpu/routine_selection/20261105_0820/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.178.04`
- CUDA version: `13.0`
- CUDA runtime: `12.6`
- cuDNN version: `92700`
- Container: `none`
- CUDA toolkit: `12.6`
- CUDA toolkit root: `/usr/local/cuda-12.6`
- cuTENSOR version: `2.8.0`
- cuSOLVER version: `11.7.1`

## CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `unknown`
- Cores per socket: `64`
- Threads per core: `unknown`
- NUMA nodes: `unknown`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-147-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-7; 8-15; 16-23; 24-31; 32-39; 40-47; 48-55; 56-63`
- CPU boost: `enabled`
- Container: `none`

Median time is reported in milliseconds for `ok` records.
Inputs are prepared on the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization. tenferro-rs CUDA uses the explicit tenferro-rs synchronize API without downloading result tensors in the timed region.
Dense and einsum inputs use the same deterministic benchmark generator in the Rust and Python runners; host-to-device layout conversion remains outside the timed region.
tenferro-rs uses native column-major GPU tensors; PyTorch and vendor-wrapper columns use their native row-major framework tensors unless noted.
The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd as a QR-based cuSOLVER comparison. tenferro-rs CUDA SVD uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.
Non-`ok` cells show the structured backend status.

## gpu/einsum / einsum / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|---:|
| einsum_bin_matmul_3072_f64 | 3.171 | 3.161 | 3.169 | 3.239 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| einsum_bin_matmul_3072_f64 | tenferro-cuda-trace | 1 | 3.17078, 3.17199, 3.17164, 3.17149, 3.17113, 3.17154, 3.17111 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| einsum_bin_matmul_3072_f64 | tenferro-cuda-eager | 1 | 3.1623, 3.16189, 3.16129, 3.16143, 3.16175, 3.16137, 3.1614 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| einsum_bin_matmul_3072_f64 | pytorch-cuda | 1 | 3.17122, 3.1701, 3.1696, 3.16879, 3.16883, 3.169, 3.16867 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| einsum_bin_matmul_3072_f64 | jax-cuda | 1 | 3.25458, 3.24251, 3.23517, 3.23376, 3.23882, 3.23199, 3.24009 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

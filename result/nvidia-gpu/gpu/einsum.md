# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/einsum`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261108_110139/gpu_einsum.yaml`
- Timestamp: `2026-11-08T11:01:39+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Effort: `standard`; manifest version: `2`
- Selection and case accounting: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261108_110139/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

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

## gpu/einsum / einsum / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| einsum_bin_matmul_3072_f64 | 3.172 | 3.163 | 3.162 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| einsum_bin_matmul_3072_f64 | tenferro-cuda-trace | 1 | 3.17171, 3.17314, 3.16861, 3.17153, 3.17713, 3.17185, 3.17412 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| einsum_bin_matmul_3072_f64 | tenferro-cuda-eager | 1 | 3.16484, 3.17033, 3.16264, 3.16203, 3.16611, 3.16255, 3.1614 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| einsum_bin_matmul_3072_f64 | pytorch-cuda | 1 | 3.16558, 3.16162, 3.15892, 3.17231, 3.16166, 3.16816, 3.16039 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/tensornetwork`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261031_201808/gpu_tensornetwork.yaml`
- Timestamp: `2026-10-31T20:18:08.642468+00:00`
- tenferro-rs commit: `c15dfcf0ea5252effdef24cffbebf2f445916e20`
- Coverage: `full`; effort: `standard`; manifest version: `1`
- Selection and case accounting: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261031_201808/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100-SXM4-80GB`
- UUID: `GPU-8840a2cd-10a3-a70c-f3a2-598e6aa3610d`
- Memory: `80 GiB`
- Driver version: `595.91.07`
- CUDA version: `13.2`
- CUDA runtime: `12.8`
- cuDNN version: `91900`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `256`
- Sockets: `2`
- Cores per socket: `64`
- Threads per core: `2`
- NUMA nodes: `2`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-138-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-7,128-135; 8-15,136-143; 16-23,144-151; 24-31,152-159; 32-39,160-167; 40-47,168-175; 48-55,176-183; 56-63,184-191; 64-71,192-199; 72-79,200-207; 80-87,208-215; 88-95,216-223; 96-103,224-231; 104-111,232-239; 112-119,240-247; 120-127,248-255`
- CPU boost: `enabled`

Median time is reported in milliseconds for `ok` records.
Inputs are prepared on the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization. tenferro-rs CUDA uses the explicit tenferro-rs synchronize API without downloading result tensors in the timed region.
Dense and einsum inputs use the same deterministic benchmark generator in the Rust and Python runners; host-to-device layout conversion remains outside the timed region.
tenferro-rs uses native column-major GPU tensors; PyTorch and vendor-wrapper columns use their native row-major framework tensors unless noted.
The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd as a QR-based cuSOLVER comparison. tenferro-rs CUDA SVD uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.
Non-`ok` cells show the structured backend status.

## gpu/tensornetwork / tensor_network_contract / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| tensornetwork_permutation_optimized_f32 | runtime failed | 305.910 | 107.934 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| tensornetwork_permutation_optimized_f32 | tenferro-cuda-eager | 1 | 211.219, 310.596, 213.359, 316.039, 258.626, 301.225, 344.274, 289.937, 331.331, 322.634 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| tensornetwork_permutation_optimized_f32 | pytorch-cuda | 1 | 107.678, 108.011, 108.27, 108.85, 108.709, 107.385, 107.672, 108.127, 107.754, 107.857 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

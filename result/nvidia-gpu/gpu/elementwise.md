# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/elementwise`
- Suite file: `/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/data/results/nvidia-gpu/gpu/routine_selection/20261017_110108/gpu_elementwise.yaml`
- Timestamp: `2026-10-17T11:01:08.148886+00:00`
- tenferro-rs commit: `88daa885a5bb09e07f3031814bbb1a21f7a74e62`
- Coverage: `full`; effort: `standard`; manifest version: `1`
- Selection and case accounting: `/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/data/results/nvidia-gpu/gpu/routine_selection/20261017_110108/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.126.09`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92700`

## CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

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
| elementwise_chain_f64_1k | 0.090 | 0.228 | 0.140 | 0.100 |
| elementwise_chain_f64_1m | 0.158 | 0.353 | 0.320 | 0.095 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| elementwise_chain_f64_1k | tenferro-cuda-trace | 32 | 2.88653, 2.84984, 2.84654, 2.86854, 2.89071, 2.8676, 2.86248 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1m | tenferro-cuda-trace | 16 | 2.53153, 2.54199, 2.53638, 2.53535, 2.53941, 2.52357, 2.4874 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1k | tenferro-cuda-eager | 16 | 3.72836, 3.46761, 3.65576, 3.71982, 3.75159, 3.6431, 3.56965 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1m | tenferro-cuda-eager | 8 | 2.82231, 2.82725, 2.81454, 2.84346, 2.84136, 2.80587, 2.84843 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1k | pytorch-cuda | 16 | 2.26449, 2.25085, 2.24219, 2.2445, 2.24253, 2.24771, 2.24111 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| elementwise_chain_f64_1m | pytorch-cuda | 8 | 2.54799, 2.56094, 2.55465, 2.55608, 2.55909, 2.56074, 2.56011 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| elementwise_chain_f64_1k | jax-cuda | 32 | 3.2596, 3.16494, 3.14786, 3.20703, 3.20025, 3.19715, 3.20902 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| elementwise_chain_f64_1m | jax-cuda | 16 | 1.54525, 1.60307, 1.51673, 1.48296, 1.50644, 1.42633, 1.53291 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

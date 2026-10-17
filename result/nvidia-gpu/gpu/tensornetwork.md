# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/tensornetwork`
- Suite file: `/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/data/results/nvidia-gpu/gpu/routine_selection/20261017_104441/gpu_tensornetwork.yaml`
- Timestamp: `2026-10-17T10:44:41.382612+00:00`
- tenferro-rs commit: `88daa885a5bb09e07f3031814bbb1a21f7a74e62`
- Coverage: `full`; effort: `standard`; manifest version: `1`
- Selection and case accounting: `/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/data/results/nvidia-gpu/gpu/routine_selection/20261017_104441/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

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

## gpu/tensornetwork / tensor_network_contract / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | JAX CUDA tree JIT |
|---|---:|---:|---:|---:|---:|
| tensornetwork_permutation_optimized_f32 | 45.974 | 46.479 | 107.360 | 90.283 | 130.561 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| tensornetwork_permutation_optimized_f32 | tenferro-cuda-trace | 1 | 46.0023, 45.7714, 46.0334, 45.5368, 45.598, 46.0478, 45.9354, 46.5225, 46.0109, 45.9456 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| tensornetwork_permutation_optimized_f32 | tenferro-cuda-eager | 1 | 46.4307, 46.4193, 45.9239, 47.3795, 47.3863, 47.6763, 46.5279, 46.2756, 46.8842, 46.0559 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| tensornetwork_permutation_optimized_f32 | pytorch-cuda | 1 | 107.412, 107.306, 107.37, 107.301, 107.304, 107.711, 107.362, 107.279, 107.859, 107.357 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| tensornetwork_permutation_optimized_f32 | jax-cuda | 1 | 90.293, 90.2692, 90.3446, 90.2765, 90.282, 90.6894, 90.2533, 89.796, 90.2836, 90.3263 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| tensornetwork_permutation_optimized_f32 | jax-cuda-tree-jit | 1 | 130.395, 131.84, 131.586, 131.176, 130.134, 129.995, 130.727, 129.412, 130.268, 135.878 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

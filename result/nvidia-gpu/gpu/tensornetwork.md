# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/tensornetwork`
- Suite file: `/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/data/results/nvidia-gpu/gpu/routine_selection/20261016_235116/gpu_tensornetwork.yaml`
- Timestamp: `2026-10-16T23:51:16.689453+00:00`
- tenferro-rs commit: `2754c8c85dcc35c1b63cb7bbdd94ddfd7ea5746e`
- Coverage: `full`; effort: `standard`; manifest version: `1`
- Selection and case accounting: `/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/data/results/nvidia-gpu/gpu/routine_selection/20261016_235116/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

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

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | JAX CUDA tree JIT | JAX CUDA eager |
|---|---:|---:|---:|---:|---:|---:|
| tensornetwork_permutation_optimized_f32 | unsupported | 47.425 | 110.378 | 90.285 | 130.474 | 333.169 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| tensornetwork_permutation_optimized_f32 | tenferro-cuda-eager | 1 | 47.9736, 47.4246, 47.3955, 47.3379, 47.4262, 47.2526, 47.2872, 47.5505, 47.722, 47.5354 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| tensornetwork_permutation_optimized_f32 | pytorch-cuda | 1 | 108.222, 107.726, 110.573, 108.169, 110.306, 109.303, 110.578, 110.45, 110.533, 110.652 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| tensornetwork_permutation_optimized_f32 | jax-cuda | 1 | 90.2856, 90.2703, 90.2755, 90.2696, 90.3402, 90.3326, 90.2596, 90.3011, 90.3125, 90.2837 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| tensornetwork_permutation_optimized_f32 | jax-cuda-tree-jit | 1 | 129.439, 129.859, 130.427, 130.231, 130.539, 130.521, 130.601, 130.112, 131.37, 130.737 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| tensornetwork_permutation_optimized_f32 | jax-cuda-eager | 1 | 335.773, 337.736, 334.08, 333.021, 327.782, 323.447, 333.317, 330.626, 304.436, 357.727 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

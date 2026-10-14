# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/einsum`
- Suite file: `/workspaces/tenferro-benchmark/.worktrees/gpu-vendors/data/results/nvidia-gpu/gpu/routine_selection/20261014_005047/gpu_einsum.yaml`
- Timestamp: `2026-10-14T00:50:47.476431+00:00`
- tenferro-rs commit: `9655b4e0daced17e0aa35a30d35710b069e10644`
- Coverage: `full`; effort: `scan`; manifest version: `1`
- Selection and case accounting: `/workspaces/tenferro-benchmark/.worktrees/gpu-vendors/data/results/nvidia-gpu/gpu/routine_selection/20261014_005047/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA GeForce RTX 3060`
- UUID: `GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`
- Memory: `12 GiB`
- Driver version: `580.178.04`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92700`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-146-generic-x86_64-with-glibc2.39`

Median time is reported in milliseconds for `ok` records.
Inputs are prepared on the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization. tenferro-rs CUDA uses the explicit tenferro-rs synchronize API without downloading result tensors in the timed region.
Dense and einsum inputs use the same deterministic benchmark generator in the Rust and Python runners; host-to-device layout conversion remains outside the timed region.
tenferro-rs uses native column-major GPU tensors; PyTorch and vendor-wrapper columns use their native row-major framework tensors unless noted.
The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd as a QR-based cuSOLVER comparison. tenferro-rs CUDA SVD uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.
Non-`ok` cells show the structured backend status.

## gpu/einsum / einsum / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | PyTorch (cuBLASLt preferred) | CUTLASS |
|---|---:|---:|---:|---:|---:|---:|
| einsum_bin_matmul_3072_f64 | unsupported | 312.048 | 312.153 | 268.671 | 312.197 | 317.886 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| einsum_bin_matmul_3072_f64 | tenferro-cuda-eager | 1 | 312.055, 312.046, 312.048 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| einsum_bin_matmul_3072_f64 | pytorch-cuda | 1 | 312.161, 312.153, 312.102 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| einsum_bin_matmul_3072_f64 | jax-cuda | 1 | 268.66, 268.673, 268.671 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| einsum_bin_matmul_3072_f64 | cublaslt | 1 | 312.187, 312.197, 312.209 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| einsum_bin_matmul_3072_f64 | cutlass | 1 | 317.886, 317.879, 317.9 | inputs, layouts, native operators/workspaces, initialization and validation outside clocks; native float64; no dtype conversion in clocks |

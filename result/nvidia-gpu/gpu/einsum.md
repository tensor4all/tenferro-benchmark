# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/einsum`
- Suite file: `/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/data/results/nvidia-gpu/gpu/routine_selection/20261016_235116/gpu_einsum.yaml`
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

## gpu/einsum / einsum / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | PyTorch (cuBLASLt preferred) | CUTLASS |
|---|---:|---:|---:|---:|---:|---:|
| einsum_bin_matmul_3072_f64 | unsupported | 3.159 | 3.208 | 3.569 | 3.171 | 3.439 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| einsum_bin_matmul_3072_f64 | tenferro-cuda-eager | 1 | 3.1594, 3.15641, 3.16204, 3.15834, 3.16485, 3.15912, 3.15631 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| einsum_bin_matmul_3072_f64 | pytorch-cuda | 1 | 3.21249, 3.19694, 3.20113, 3.20063, 3.21234, 3.20778, 3.2115 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| einsum_bin_matmul_3072_f64 | jax-cuda | 1 | 3.89996, 4.39882, 3.56865, 3.39216, 3.83011, 3.42894, 3.4566 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| einsum_bin_matmul_3072_f64 | cublaslt | 1 | 3.17119, 3.17327, 3.17168, 3.16914, 3.17091, 3.16901, 3.16634 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| einsum_bin_matmul_3072_f64 | cutlass | 1 | 3.4447, 3.44557, 3.43826, 3.43882, 3.43676, 3.43907, 3.43856 | inputs, layouts, native operators/workspaces, initialization and validation outside clocks; native float64; no dtype conversion in clocks |

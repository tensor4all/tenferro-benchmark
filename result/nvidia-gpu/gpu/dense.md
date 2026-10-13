# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense`
- Suite file: `/workspaces/tenferro-benchmark/.worktrees/gpu-lane/data/results/nvidia-gpu/gpu/routine_selection/20261013_234113/gpu_dense.yaml`
- Timestamp: `2026-10-13T23:41:13.603481+00:00`
- tenferro-rs commit: `9655b4e0daced17e0aa35a30d35710b069e10644`
- Coverage: `quick`; effort: `standard`; manifest version: `1`
- Selection and case accounting: `/workspaces/tenferro-benchmark/.worktrees/gpu-lane/data/results/nvidia-gpu/gpu/routine_selection/20261013_234113/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

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

## gpu/dense / batched_matmul / allocating output

| Problem | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|
| dense_batched_matmul_f64_b16_32 | 0.039 | 0.022 | 0.082 |

## gpu/dense / eigh / allocating output

| Problem | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|
| dense_eigh_f64_1024 | 74.451 | 74.663 | 75.496 |

## gpu/dense / matmul / allocating output

| Problem | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|
| dense_matmul_f64_3072 | 313.714 | 313.684 | 271.708 |

## gpu/dense / qr / allocating output

| Problem | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|
| dense_qr_f64_1536 | 95.295 | 95.055 | 97.099 |

## gpu/dense / solve / allocating output

| Problem | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|
| dense_solve_f64_2048_rhs16 | 48.353 | 47.959 | 48.927 |
| dense_solve_f64_512_rhs16 | 3.979 | 3.001 | 3.471 |

## gpu/dense / svd / allocating output

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; PyTorch's default row may use a different SVD driver and row-major framework layout.

| Problem | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|
| dense_svd_f64_256 | 81.835 | 83.791 | 82.525 |

## Python sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| dense_matmul_f64_3072 | tenferro-cuda-eager | 1 | 313.711, 313.715, 313.713, 313.709, 313.724, 313.719, 313.714 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-eager | 64 | 2.51221, 2.45454, 2.43442, 2.44722, 2.48687, 2.90379, 2.63175 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_qr_f64_1536 | tenferro-cuda-eager | 1 | 95.2949, 95.2675, 95.289, 95.468, 95.3212 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_512_rhs16 | tenferro-cuda-eager | 1 | 3.9833, 3.97281, 3.97942, 3.98368, 3.97492 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-eager | 1 | 48.3592, 48.3942, 48.2762, 48.3531, 48.3158 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_svd_f64_256 | tenferro-cuda-eager | 1 | 81.905, 81.8244, 81.8354, 81.8072, 81.8414 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_eigh_f64_1024 | tenferro-cuda-eager | 1 | 74.4385, 74.6582, 74.4506, 74.4865, 74.3068 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_matmul_f64_3072 | pytorch-cuda | 1 | 313.677, 313.687, 313.692, 313.702, 313.683, 313.684, 313.678 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b16_32 | pytorch-cuda | 128 | 2.87371, 2.92741, 2.85617, 2.84772, 2.90994, 3.02101, 2.83475 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_qr_f64_1536 | pytorch-cuda | 1 | 95.0459, 95.0304, 95.0889, 95.0546, 95.0783 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_512_rhs16 | pytorch-cuda | 1 | 2.99044, 3.00553, 3.00083, 2.99588, 3.00189 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | pytorch-cuda | 1 | 47.9553, 47.9586, 47.9829, 47.8902, 47.9702 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | pytorch-cuda | 1 | 83.7789, 83.7641, 83.791, 83.7949, 83.8059 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | pytorch-cuda | 1 | 74.7179, 74.6705, 74.6627, 74.6371, 74.5808 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_matmul_f64_3072 | jax-cuda | 1 | 271.708, 271.73, 271.684, 271.717, 271.648, 271.737, 271.698 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b16_32 | jax-cuda | 32 | 2.65845, 2.6399, 2.62385, 2.64266, 2.65312, 2.61143, 2.5689 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_qr_f64_1536 | jax-cuda | 1 | 96.9615, 97.0992, 97.0423, 97.1572, 97.1311 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_512_rhs16 | jax-cuda | 1 | 3.46242, 3.47077, 3.48204, 3.44023, 3.47183 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | jax-cuda | 1 | 48.8774, 48.9179, 48.9321, 48.927, 48.9452 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | jax-cuda | 1 | 82.553, 82.5433, 82.5248, 82.4728, 82.452 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | jax-cuda | 1 | 75.4425, 75.6063, 75.4962, 75.4877, 75.6631 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

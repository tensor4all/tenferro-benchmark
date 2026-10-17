# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense`
- Suite file: `/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/data/results/nvidia-gpu/gpu/routine_selection/20261016_235116/gpu_dense.yaml`
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

## gpu/dense / batched_matmul / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | PyTorch (cuBLASLt preferred) | CUTLASS | PyTorch (cuSOLVER preferred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | unsupported | 2.103 | 2.086 | 2.292 | 2.086 | 2.102 | unsupported |
| dense_batched_matmul_f64_b16_32 | unsupported | 0.037 | 0.014 | 0.050 | 0.014 | 0.013 | unsupported |

## gpu/dense / eigh / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | PyTorch (cuBLASLt preferred) | CUTLASS | PyTorch (cuSOLVER preferred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_eigh_f64_1024 | unsupported | 14.427 | 14.320 | 14.105 | unsupported | unsupported | 14.410 |

## gpu/dense / matmul / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | PyTorch (cuBLASLt preferred) | CUTLASS | PyTorch (cuSOLVER preferred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_3072 | unsupported | 3.139 | 3.127 | 3.242 | 3.132 | 3.448 | unsupported |

## gpu/dense / qr / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | PyTorch (cuBLASLt preferred) | CUTLASS | PyTorch (cuSOLVER preferred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_qr_f64_1536 | unsupported | 11.936 | 11.747 | 12.031 | unsupported | unsupported | 11.851 |

## gpu/dense / solve / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | PyTorch (cuBLASLt preferred) | CUTLASS | PyTorch (cuSOLVER preferred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_solve_f64_1024_rhs16 | unsupported | 4.823 | 4.426 | 4.658 | unsupported | unsupported | 4.462 |
| dense_solve_f64_2048_rhs128 | unsupported | 9.736 | 9.144 | 9.465 | unsupported | unsupported | 9.157 |
| dense_solve_f64_2048_rhs16 | unsupported | 10.499 | 10.224 | 10.783 | unsupported | unsupported | 10.322 |
| dense_solve_f64_4096_rhs16 | unsupported | 27.910 | 27.531 | 28.175 | unsupported | unsupported | 27.701 |
| dense_solve_f64_512_rhs16 | unsupported | 2.527 | 2.066 | 2.231 | unsupported | unsupported | 2.073 |

## gpu/dense / svd / allocating output

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; PyTorch's default row may use a different SVD driver and row-major framework layout.

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | PyTorch (cuBLASLt preferred) | CUTLASS | PyTorch (cuSOLVER preferred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_svd_f64_256 | unsupported | 12.075 | 12.190 | 11.934 | unsupported | unsupported | 23.143 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| dense_matmul_f64_3072 | tenferro-cuda-eager | 1 | 3.14339, 3.14138, 3.13752, 3.13594, 3.13655, 3.14303, 3.13908 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | 1 | 2.10437, 2.10323, 2.1024, 2.10211, 2.10299, 2.1032, 2.10466 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_qr_f64_1536 | tenferro-cuda-eager | 1 | 11.9363, 11.9349, 11.9319, 11.9765, 11.9535 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_512_rhs16 | tenferro-cuda-eager | 1 | 2.52387, 2.52693, 2.5171, 2.54744, 2.55244 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_1024_rhs16 | tenferro-cuda-eager | 1 | 4.83715, 4.816, 4.81487, 4.82268, 4.82367 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-eager | 1 | 10.5161, 10.4674, 10.4722, 10.4986, 10.5016 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_4096_rhs16 | tenferro-cuda-eager | 1 | 27.8763, 27.9933, 27.8838, 27.9564, 27.9099 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs128 | tenferro-cuda-eager | 1 | 9.75648, 9.72352, 9.73571, 9.73006, 9.74428 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_svd_f64_256 | tenferro-cuda-eager | 1 | 12.0918, 12.1224, 12.0753, 12.0545, 12.0637 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_eigh_f64_1024 | tenferro-cuda-eager | 1 | 14.3881, 14.4599, 14.4483, 14.4269, 14.4095 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-eager | 64 | 2.35889, 2.29623, 2.41794, 2.3003, 2.38111, 2.2771, 2.36131 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_matmul_f64_3072 | pytorch-cuda | 1 | 3.12909, 3.13013, 3.12632, 3.12719, 3.12753, 3.12708, 3.1273 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | 1 | 2.08738, 2.0862, 2.08647, 2.08412, 2.08503, 2.08635, 2.09305 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_qr_f64_1536 | pytorch-cuda | 1 | 11.767, 11.7439, 11.7474, 11.8425, 11.7362 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_512_rhs16 | pytorch-cuda | 1 | 2.05412, 2.06578, 2.17212, 2.07118, 2.06046 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_1024_rhs16 | pytorch-cuda | 1 | 4.66464, 4.42614, 4.43731, 4.42247, 4.42399 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | pytorch-cuda | 1 | 10.2385, 10.213, 10.2096, 10.2241, 10.3391 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_4096_rhs16 | pytorch-cuda | 1 | 27.531, 27.5737, 27.5182, 27.5552, 27.503 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs128 | pytorch-cuda | 1 | 9.14144, 9.1516, 9.14376, 9.16298, 9.13548 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | pytorch-cuda | 1 | 12.2069, 12.1742, 12.1899, 12.183, 12.1918 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | pytorch-cuda | 1 | 14.3117, 14.3459, 14.2974, 14.3203, 14.3505 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b16_32 | pytorch-cuda | 128 | 1.70825, 1.76374, 1.77592, 1.81131, 1.76997, 1.78344, 1.76636 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_matmul_f64_3072 | jax-cuda | 1 | 3.23079, 3.24402, 3.23827, 3.24005, 3.26625, 3.24172, 3.2536 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b1024_256 | jax-cuda | 1 | 2.2915, 2.31997, 2.29084, 2.2921, 2.32542, 2.31744, 2.28285 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_qr_f64_1536 | jax-cuda | 1 | 12.0567, 12.0287, 12.0312, 12.0533, 12.0176 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_512_rhs16 | jax-cuda | 1 | 2.2401, 2.25735, 2.23124, 2.22191, 2.22884 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_1024_rhs16 | jax-cuda | 1 | 4.65765, 4.67249, 4.6488, 4.66291, 4.64712 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | jax-cuda | 1 | 10.7827, 10.811, 10.8369, 10.7611, 10.7569 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_4096_rhs16 | jax-cuda | 1 | 28.1755, 28.1263, 28.1578, 28.2273, 28.1972 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs128 | jax-cuda | 1 | 9.42903, 9.45631, 9.53516, 9.46499, 9.48351 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | jax-cuda | 1 | 11.9608, 11.9339, 11.9311, 11.9343, 11.9468 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | jax-cuda | 1 | 14.1244, 14.0925, 14.1011, 14.1046, 14.1569 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b16_32 | jax-cuda | 64 | 3.25731, 3.24902, 3.22076, 3.25995, 3.22971, 3.20465, 3.22677 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_matmul_f64_3072 | cublaslt | 1 | 3.13965, 3.13606, 3.12996, 3.1324, 3.13182, 3.13156, 3.13141 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b1024_256 | cublaslt | 1 | 2.0958, 2.08943, 2.08659, 2.08437, 2.08646, 2.08475, 2.08582 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b16_32 | cublaslt | 128 | 1.74073, 1.73096, 1.8108, 1.73806, 1.75035, 1.77648, 1.86539 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_matmul_f64_3072 | cutlass | 1 | 3.4493, 3.4467, 3.44821, 3.45912, 3.44746, 3.44354, 3.45005 | inputs, layouts, native operators/workspaces, initialization and validation outside clocks; native float64; no dtype conversion in clocks |
| dense_batched_matmul_f64_b1024_256 | cutlass | 1 | 2.10512, 2.10371, 2.10242, 2.10118, 2.10167, 2.09849, 2.10121 | inputs, layouts, native operators/workspaces, initialization and validation outside clocks; native float64; no dtype conversion in clocks |
| dense_batched_matmul_f64_b16_32 | cutlass | 256 | 3.4133, 3.41263, 3.41157, 3.41228, 3.41744, 3.41261, 3.41247 | inputs, layouts, native operators/workspaces, initialization and validation outside clocks; native float64; no dtype conversion in clocks |
| dense_qr_f64_1536 | cusolver | 1 | 11.8798, 11.8422, 11.8313, 11.871, 11.8514 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_512_rhs16 | cusolver | 1 | 2.08749, 2.07462, 2.06915, 2.06594, 2.0734 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_1024_rhs16 | cusolver | 1 | 4.45358, 4.47299, 4.45873, 4.48708, 4.46242 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | cusolver | 1 | 10.2929, 10.3147, 10.3297, 10.3342, 10.322 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_4096_rhs16 | cusolver | 1 | 27.7011, 27.6908, 27.7154, 27.719, 27.6646 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs128 | cusolver | 1 | 9.13728, 9.15738, 9.1565, 9.15777, 9.14361 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | cusolver | 1 | 23.3661, 23.1217, 23.1427, 23.1435, 23.1058 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | cusolver | 1 | 14.3712, 14.4096, 14.3967, 14.429, 14.4772 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

## Output retention bounds

These intrinsic single outputs exceed the configured batch budget. Samples contain one operation; the effective bound is raised explicitly to that one output, without shrinking the workload.

| Problem | Backend | Bytes/output | Configured budget | Effective bound | Ops/sample |
|---|---|---:|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | jax-cuda | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | cublaslt | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | cutlass | 536870912 | 268435456 | 536870912 | 1 |

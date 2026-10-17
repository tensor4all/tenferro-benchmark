# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense`
- Suite file: `/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/data/results/nvidia-gpu/gpu/routine_selection/20261017_104441/gpu_dense.yaml`
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

## gpu/dense / batched_matmul / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | PyTorch (cuBLASLt preferred) | CUTLASS | PyTorch (cuSOLVER preferred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | 2.132 | 2.104 | 2.081 | 2.260 | 2.082 | 2.099 | unsupported |
| dense_batched_matmul_f64_b16_32 | 0.049 | 0.026 | 0.010 | 0.051 | 0.010 | 0.014 | unsupported |

## gpu/dense / eigh / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | PyTorch (cuBLASLt preferred) | CUTLASS | PyTorch (cuSOLVER preferred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_eigh_f64_1024 | 14.043 | 14.362 | 14.270 | 14.491 | unsupported | unsupported | 13.975 |

## gpu/dense / matmul / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | PyTorch (cuBLASLt preferred) | CUTLASS | PyTorch (cuSOLVER preferred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_3072 | 3.178 | 3.143 | 3.128 | 3.378 | 3.129 | 3.435 | unsupported |

## gpu/dense / qr / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | PyTorch (cuBLASLt preferred) | CUTLASS | PyTorch (cuSOLVER preferred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_qr_f64_1536 | 11.960 | 11.884 | 11.766 | 12.008 | unsupported | unsupported | 11.791 |

## gpu/dense / solve / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | PyTorch (cuBLASLt preferred) | CUTLASS | PyTorch (cuSOLVER preferred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_solve_f64_1024_rhs16 | 4.785 | 4.698 | 4.417 | 4.632 | unsupported | unsupported | 4.424 |
| dense_solve_f64_2048_rhs128 | 9.731 | 9.763 | 9.124 | 9.509 | unsupported | unsupported | 9.149 |
| dense_solve_f64_2048_rhs16 | 10.447 | 10.550 | 10.220 | 10.719 | unsupported | unsupported | 10.269 |
| dense_solve_f64_4096_rhs16 | 27.828 | 27.888 | 27.528 | 28.133 | unsupported | unsupported | 28.299 |
| dense_solve_f64_512_rhs16 | 2.801 | 2.404 | 2.050 | 2.219 | unsupported | unsupported | 2.028 |

## gpu/dense / svd / allocating output

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; PyTorch's default row may use a different SVD driver and row-major framework layout.

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT | PyTorch (cuBLASLt preferred) | CUTLASS | PyTorch (cuSOLVER preferred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_svd_f64_256 | 12.053 | 12.069 | 12.139 | 11.997 | unsupported | unsupported | 23.000 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| dense_matmul_f64_3072 | tenferro-cuda-trace | 1 | 3.18619, 3.17913, 3.18058, 3.17494, 3.17769, 3.17207, 3.17721 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-trace | 1 | 2.13578, 2.13558, 2.13761, 2.13152, 2.12999, 2.13165, 2.1309 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_qr_f64_1536 | tenferro-cuda-trace | 1 | 11.9592, 11.9705, 11.9625, 11.9598, 11.9289 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_512_rhs16 | tenferro-cuda-trace | 1 | 2.82419, 2.81274, 2.79401, 2.78767, 2.80137 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_1024_rhs16 | tenferro-cuda-trace | 1 | 5.13766, 4.76084, 4.76861, 4.78526, 4.79905 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-trace | 1 | 10.4699, 10.4471, 10.4273, 10.4453, 10.4823 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_4096_rhs16 | tenferro-cuda-trace | 1 | 27.8279, 27.8043, 27.8169, 27.8529, 27.9315 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs128 | tenferro-cuda-trace | 1 | 9.73646, 9.73095, 9.7482, 9.71948, 9.7197 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_svd_f64_256 | tenferro-cuda-trace | 1 | 12.0544, 12.0534, 12.0482, 12.0529, 12.0675 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_eigh_f64_1024 | tenferro-cuda-trace | 1 | 14.0519, 14.0598, 14.0134, 14.0426, 13.984 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-trace | 64 | 3.1705, 3.24267, 3.14592, 3.14645, 3.1488, 3.14852, 3.16583 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_matmul_f64_3072 | tenferro-cuda-eager | 1 | 3.15145, 3.14288, 3.14468, 3.14422, 3.1427, 3.14227, 3.14306 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | 1 | 2.10629, 2.10476, 2.10198, 2.10387, 2.10355, 2.11588, 2.10378 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_qr_f64_1536 | tenferro-cuda-eager | 1 | 11.8898, 11.8796, 11.8836, 11.8757, 11.8936 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_512_rhs16 | tenferro-cuda-eager | 1 | 2.39632, 2.41057, 2.40397, 2.41344, 2.39564 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_1024_rhs16 | tenferro-cuda-eager | 1 | 4.69823, 4.6722, 4.67618, 4.93054, 4.81611 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-eager | 1 | 10.4952, 10.5503, 10.5379, 10.5924, 10.5827 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_4096_rhs16 | tenferro-cuda-eager | 1 | 27.8257, 27.9338, 27.8876, 27.9171, 27.8818 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs128 | tenferro-cuda-eager | 1 | 9.79609, 9.77513, 9.76347, 9.76111, 9.74927 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_svd_f64_256 | tenferro-cuda-eager | 1 | 12.0326, 12.042, 12.0946, 12.0874, 12.069 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_eigh_f64_1024 | tenferro-cuda-eager | 1 | 14.3746, 14.362, 14.4054, 14.3589, 14.3012 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-eager | 128 | 3.32543, 3.32115, 3.32378, 3.32788, 3.33544, 3.32079, 3.33405 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_matmul_f64_3072 | pytorch-cuda | 1 | 3.12823, 3.13057, 3.12661, 3.12722, 3.12835, 3.12807, 3.13253 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | 1 | 2.08131, 2.08064, 2.08058, 2.07975, 2.08493, 2.08064, 2.07927 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_qr_f64_1536 | pytorch-cuda | 1 | 11.7663, 11.7492, 11.7651, 11.768, 11.7832 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_512_rhs16 | pytorch-cuda | 1 | 2.0708, 2.03612, 2.0441, 2.04997, 2.05751 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_1024_rhs16 | pytorch-cuda | 1 | 4.41895, 4.41692, 4.40269, 4.40792, 4.4173 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | pytorch-cuda | 1 | 10.2284, 10.2005, 10.2281, 10.2074, 10.22 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_4096_rhs16 | pytorch-cuda | 1 | 27.5277, 27.548, 27.5666, 27.4717, 27.5028 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs128 | pytorch-cuda | 1 | 9.12373, 9.14657, 9.13099, 9.11958, 9.11893 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | pytorch-cuda | 1 | 12.1326, 12.1563, 12.1383, 12.1385, 12.1412 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | pytorch-cuda | 1 | 14.2895, 13.9444, 14.2698, 14.2946, 13.9234 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b16_32 | pytorch-cuda | 256 | 2.64621, 2.64535, 2.63139, 2.62398, 2.64129, 2.63331, 2.61186 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_matmul_f64_3072 | jax-cuda | 1 | 3.42533, 3.37088, 3.39172, 3.37763, 3.39566, 3.37031, 3.37748 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b1024_256 | jax-cuda | 1 | 2.26088, 2.25549, 2.2603, 2.26102, 2.25324, 2.25591, 2.2714 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_qr_f64_1536 | jax-cuda | 1 | 11.9287, 12.0084, 12.0152, 12.0204, 12.0078 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_512_rhs16 | jax-cuda | 1 | 2.24168, 2.24503, 2.20719, 2.21909, 2.19526 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_1024_rhs16 | jax-cuda | 1 | 4.64305, 4.66125, 4.63213, 4.60894, 4.62131 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | jax-cuda | 1 | 10.6808, 10.7289, 10.7426, 10.7195, 10.7057 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_4096_rhs16 | jax-cuda | 1 | 28.0416, 28.1334, 28.133, 28.1424, 28.1218 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs128 | jax-cuda | 1 | 9.49039, 9.50869, 9.499, 9.50893, 9.55637 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | jax-cuda | 1 | 11.9997, 11.9908, 12.0025, 11.9973, 11.9966 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | jax-cuda | 1 | 14.5229, 14.4933, 14.4363, 14.491, 14.4457 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b16_32 | jax-cuda | 64 | 3.30043, 3.25248, 3.26755, 3.25023, 3.25394, 3.16675, 3.24053 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_matmul_f64_3072 | cublaslt | 1 | 3.12952, 3.12978, 3.12714, 3.12816, 3.12917, 3.12742, 3.12854 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b1024_256 | cublaslt | 1 | 2.08466, 2.08214, 2.08426, 2.07985, 2.07999, 2.08202, 2.08169 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b16_32 | cublaslt | 256 | 2.60234, 2.58507, 2.59194, 2.59014, 2.58481, 2.59416, 2.58148 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_matmul_f64_3072 | cutlass | 1 | 3.43475, 3.43586, 3.43538, 3.43456, 3.43502, 3.43463, 3.43479 | inputs, layouts, native operators/workspaces, initialization and validation outside clocks; native float64; no dtype conversion in clocks |
| dense_batched_matmul_f64_b1024_256 | cutlass | 1 | 2.09813, 2.10112, 2.10075, 2.09595, 2.10109, 2.09889, 2.09675 | inputs, layouts, native operators/workspaces, initialization and validation outside clocks; native float64; no dtype conversion in clocks |
| dense_batched_matmul_f64_b16_32 | cutlass | 256 | 3.64715, 3.41366, 3.41533, 3.76399, 3.57621, 3.78759, 3.8312 | inputs, layouts, native operators/workspaces, initialization and validation outside clocks; native float64; no dtype conversion in clocks |
| dense_qr_f64_1536 | cusolver | 1 | 12.5235, 11.7829, 11.7906, 11.8028, 11.7745 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_512_rhs16 | cusolver | 1 | 2.02764, 2.03316, 2.03399, 2.02421, 2.0164 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_1024_rhs16 | cusolver | 1 | 4.40574, 4.42389, 4.41877, 4.42472, 4.42799 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | cusolver | 1 | 11.5643, 10.2692, 10.2512, 10.2697, 10.2656 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_4096_rhs16 | cusolver | 1 | 28.2991, 27.6291, 27.6419, 28.4933, 28.3409 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs128 | cusolver | 1 | 9.14933, 10.3092, 9.40395, 9.13626, 9.14615 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | cusolver | 1 | 22.9899, 22.9739, 24.2199, 23.1352, 22.9995 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | cusolver | 1 | 13.9603, 13.9896, 14.0813, 13.9751, 13.9609 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

## Output retention bounds

These intrinsic single outputs exceed the configured batch budget. Samples contain one operation; the effective bound is raised explicitly to that one output, without shrinking the workload.

| Problem | Backend | Bytes/output | Configured budget | Effective bound | Ops/sample |
|---|---|---:|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-trace | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | jax-cuda | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | cublaslt | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | cutlass | 536870912 | 268435456 | 536870912 | 1 |

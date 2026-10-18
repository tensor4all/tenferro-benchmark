# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261017_141327/gpu_dense.yaml`
- Timestamp: `2026-10-17T14:13:27.525120+00:00`
- tenferro-rs commit: `4de2e1226337e877066f129667b70c2a01308bfb`
- Coverage: `full`; effort: `standard`; manifest version: `1`
- Selection and case accounting: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261017_141327/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-836a43ba-6520-b027-a26d-7c22feb348e8`
- Memory: `80 GiB`
- Driver version: `580.159.04`
- CUDA version: `13.0`
- CUDA runtime: `12.8`
- cuDNN version: `92000`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `252`
- Sockets: `2`
- Cores per socket: `126`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-124-generic-x86_64-with-glibc2.39`

Median time is reported in milliseconds for `ok` records.
Inputs are prepared on the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization. tenferro-rs CUDA uses the explicit tenferro-rs synchronize API without downloading result tensors in the timed region.
Dense and einsum inputs use the same deterministic benchmark generator in the Rust and Python runners; host-to-device layout conversion remains outside the timed region.
tenferro-rs uses native column-major GPU tensors; PyTorch and vendor-wrapper columns use their native row-major framework tensors unless noted.
The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd as a QR-based cuSOLVER comparison. tenferro-rs CUDA SVD uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.
Non-`ok` cells show the structured backend status.

## gpu/dense / batched_matmul / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | 2.161 | 2.130 | 2.095 |
| dense_batched_matmul_f64_b16_32 | 0.034 | 0.051 | 0.011 |

## gpu/dense / eigh / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_eigh_f64_1024 | 14.677 | 14.190 | 14.318 |

## gpu/dense / matmul / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_matmul_f64_3072 | 3.243 | 3.182 | 3.145 |

## gpu/dense / qr / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_qr_f64_1536 | 12.147 | 11.976 | 11.810 |

## gpu/dense / solve / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_solve_f64_1024_rhs16 | 5.193 | 5.455 | 4.441 |
| dense_solve_f64_2048_rhs128 | 10.232 | 10.238 | 9.164 |
| dense_solve_f64_2048_rhs16 | 10.925 | 10.529 | 10.260 |
| dense_solve_f64_4096_rhs16 | 28.002 | 28.199 | 27.632 |
| dense_solve_f64_512_rhs16 | 2.876 | 3.087 | 2.054 |

## gpu/dense / svd / allocating output

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; PyTorch's default row may use a different SVD driver and row-major framework layout.

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_svd_f64_256 | 12.350 | 12.487 | 12.321 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| dense_matmul_f64_3072 | tenferro-cuda-trace | 1 | 3.2581, 3.24296, 3.27065, 3.24305, 3.24126, 3.22987, 3.22755 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-trace | 1 | 2.16656, 2.15471, 2.16363, 2.16737, 2.15216, 2.14297, 2.16136 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_qr_f64_1536 | tenferro-cuda-trace | 1 | 12.3381, 12.1787, 12.1471, 12.1431, 12.1227 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_512_rhs16 | tenferro-cuda-trace | 1 | 2.8569, 2.86248, 2.88913, 2.87573, 2.88599 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_1024_rhs16 | tenferro-cuda-trace | 1 | 5.19287, 5.24078, 5.1898, 5.18206, 5.20612 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-trace | 1 | 10.8732, 10.9245, 10.8863, 11.1917, 10.9949 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_4096_rhs16 | tenferro-cuda-trace | 1 | 28.1268, 28.002, 27.9982, 27.9888, 28.0624 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs128 | tenferro-cuda-trace | 1 | 10.2855, 10.0655, 10.174, 10.2315, 10.2388 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_svd_f64_256 | tenferro-cuda-trace | 1 | 12.2908, 12.573, 12.3706, 12.3501, 12.3463 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_eigh_f64_1024 | tenferro-cuda-trace | 1 | 14.4731, 14.5307, 14.684, 14.9852, 14.6773 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-trace | 64 | 2.11165, 2.13342, 2.19427, 2.48269, 2.09956, 2.18078, 2.16097 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| dense_matmul_f64_3072 | tenferro-cuda-eager | 1 | 3.20837, 3.18152, 3.19243, 3.18163, 3.18298, 3.18101, 3.17838 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | 1 | 2.12996, 2.12997, 2.13074, 2.13855, 2.12921, 2.12682, 2.12984 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_qr_f64_1536 | tenferro-cuda-eager | 1 | 11.9618, 11.9909, 11.9762, 12.0154, 11.9603 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_512_rhs16 | tenferro-cuda-eager | 1 | 3.16672, 3.08743, 3.1036, 3.04688, 3.05943 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_1024_rhs16 | tenferro-cuda-eager | 1 | 5.48145, 5.45515, 5.47299, 5.43281, 5.40815 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-eager | 1 | 10.4613, 10.5642, 10.526, 10.5292, 10.6433 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_4096_rhs16 | tenferro-cuda-eager | 1 | 28.199, 28.0258, 27.9909, 28.4186, 28.3047 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs128 | tenferro-cuda-eager | 1 | 10.1082, 10.0983, 10.2565, 10.2379, 10.408 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_svd_f64_256 | tenferro-cuda-eager | 1 | 12.485, 12.4831, 12.6432, 12.5941, 12.4872 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_eigh_f64_1024 | tenferro-cuda-eager | 1 | 14.1888, 14.1901, 14.2364, 14.1446, 14.2251 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-eager | 64 | 3.49943, 3.44895, 3.81761, 3.24561, 3.18497, 3.19294, 3.18406 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_matmul_f64_3072 | pytorch-cuda | 1 | 3.15084, 3.14765, 3.14571, 3.14517, 3.14549, 3.14355, 3.14365 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | 1 | 2.09545, 2.09557, 2.09429, 2.0916, 2.11231, 2.09402, 2.09528 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_qr_f64_1536 | pytorch-cuda | 1 | 11.8102, 11.8006, 11.8206, 11.8052, 12.4045 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_512_rhs16 | pytorch-cuda | 1 | 2.05174, 2.05209, 2.05862, 2.05437, 2.05908 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_1024_rhs16 | pytorch-cuda | 1 | 4.42721, 4.44317, 4.44075, 4.46432, 4.42465 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | pytorch-cuda | 1 | 10.2888, 10.2596, 10.2751, 10.258, 10.2586 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_4096_rhs16 | pytorch-cuda | 1 | 27.632, 27.5822, 27.9435, 27.5851, 27.6322 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs128 | pytorch-cuda | 1 | 9.16301, 9.15145, 9.16408, 9.17021, 9.16409 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | pytorch-cuda | 1 | 12.325, 12.3464, 12.2882, 12.3004, 12.3206 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | pytorch-cuda | 1 | 14.3178, 14.2456, 14.3783, 14.682, 14.3005 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b16_32 | pytorch-cuda | 128 | 1.47178, 1.42039, 1.4452, 1.4806, 1.40628, 1.42551, 1.45732 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

## Output retention bounds

These intrinsic single outputs exceed the configured batch budget. Samples contain one operation; the effective bound is raised explicitly to that one output, without shrinking the workload.

| Problem | Backend | Bytes/output | Configured budget | Effective bound | Ops/sample |
|---|---|---:|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-trace | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | 536870912 | 268435456 | 536870912 | 1 |

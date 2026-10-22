# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261022_065326/gpu_dense.yaml`
- Timestamp: `2026-10-22T06:53:26.276259+00:00`
- tenferro-rs commit: `b59cf27cbfea1f0d8cebcc1551f8dd919f9a3626`
- Coverage: `full`; effort: `standard`; manifest version: `1`
- Selection and case accounting: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261022_065326/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-bb034e0c-d802-9db1-0994-f06a3ddf287b`
- Memory: `80 GiB`
- Driver version: `595.91.07`
- CUDA version: `13.2`
- CUDA runtime: `12.8`
- cuDNN version: `91900`

## CPU Information

- Model: `AMD EPYC 9554 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `252`
- Sockets: `2`
- Cores per socket: `126`
- Threads per core: `1`
- NUMA nodes: `2`
- Python platform: `Linux-6.8.0-138-generic-x86_64-with-glibc2.39`

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
| dense_batched_matmul_f64_b1024_256 | 2.173 | 2.158 | 2.082 |
| dense_batched_matmul_f64_b16_32 | 0.057 | 0.022 | 0.007 |

## gpu/dense / eigh / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_eigh_f64_1024 | 14.495 | 14.394 | 14.362 |

## gpu/dense / matmul / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_matmul_f64_3072 | 3.222 | 3.191 | 3.130 |

## gpu/dense / qr / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_qr_f64_1536 | 12.001 | 11.902 | 11.802 |

## gpu/dense / solve / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_solve_f64_1024_rhs16 | 4.718 | 4.435 | 4.422 |
| dense_solve_f64_2048_rhs128 | 9.617 | 9.401 | 9.164 |
| dense_solve_f64_2048_rhs16 | 10.398 | 10.177 | 10.277 |
| dense_solve_f64_4096_rhs16 | 27.948 | 27.625 | 27.610 |
| dense_solve_f64_512_rhs16 | 2.366 | 2.118 | 2.039 |

## gpu/dense / svd / allocating output

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; PyTorch's default row may use a different SVD driver and row-major framework layout.

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_svd_f64_256 | 12.279 | 12.167 | 12.224 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| dense_matmul_f64_3072 | tenferro-cuda-trace | 1 | 3.21831, 3.22181, 3.20808, 3.20578, 3.22832, 3.22347, 3.22839 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-trace | 1 | 2.1725, 2.16291, 2.17216, 2.17365, 2.17406, 2.17258, 2.17348 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_qr_f64_1536 | tenferro-cuda-trace | 1 | 11.9865, 12.0059, 12.1123, 12.0014, 11.9913 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_512_rhs16 | tenferro-cuda-trace | 1 | 2.35978, 2.36236, 2.38311, 2.38075, 2.3658 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_1024_rhs16 | tenferro-cuda-trace | 1 | 4.70749, 4.71775, 4.74165, 4.71817, 4.6864 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-trace | 1 | 10.3976, 10.4134, 10.4259, 10.3891, 10.3903 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_4096_rhs16 | tenferro-cuda-trace | 1 | 27.9418, 27.9483, 27.9875, 28.025, 27.9051 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs128 | tenferro-cuda-trace | 1 | 9.59426, 9.63089, 9.61737, 9.62915, 9.59873 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_svd_f64_256 | tenferro-cuda-trace | 1 | 12.2792, 12.2865, 12.2887, 12.2525, 12.2543 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_eigh_f64_1024 | tenferro-cuda-trace | 1 | 14.4686, 14.4948, 14.5145, 14.4975, 14.4369 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-trace | 16 | 0.917665, 0.913349, 0.92111, 0.907481, 0.921892, 0.912147, 0.910224 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_matmul_f64_3072 | tenferro-cuda-eager | 1 | 3.18988, 3.19063, 3.19244, 3.18703, 3.19295, 3.18993, 3.19907 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | 1 | 2.1611, 2.15783, 2.16065, 2.15024, 2.16036, 2.15231, 2.15338 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_qr_f64_1536 | tenferro-cuda-eager | 1 | 11.9015, 11.9053, 11.8893, 11.9238, 11.8982 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_512_rhs16 | tenferro-cuda-eager | 1 | 2.096, 2.09531, 2.11779, 2.13779, 2.12232 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_1024_rhs16 | tenferro-cuda-eager | 1 | 4.44712, 4.4822, 4.43481, 4.4353, 4.43428 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-eager | 1 | 10.1789, 10.1903, 10.1571, 10.1748, 10.1769 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_4096_rhs16 | tenferro-cuda-eager | 1 | 27.5782, 27.6538, 27.6251, 27.6581, 27.6013 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs128 | tenferro-cuda-eager | 1 | 9.40573, 9.39296, 9.40262, 9.36929, 9.40098 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_svd_f64_256 | tenferro-cuda-eager | 1 | 12.1478, 12.1911, 12.1669, 12.1783, 12.1509 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_eigh_f64_1024 | tenferro-cuda-eager | 1 | 14.4345, 14.4173, 14.365, 14.3942, 14.3383 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-eager | 128 | 2.90636, 2.7973, 2.87614, 2.82285, 2.88208, 2.86718, 2.81108 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_matmul_f64_3072 | pytorch-cuda | 1 | 3.14071, 3.13159, 3.13019, 3.12796, 3.1324, 3.12416, 3.12518 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | 1 | 2.09348, 2.08212, 2.07896, 2.08235, 2.07965, 2.0884, 2.07764 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_qr_f64_1536 | pytorch-cuda | 1 | 11.7994, 11.8131, 11.8004, 11.8022, 11.8041 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_512_rhs16 | pytorch-cuda | 1 | 2.03929, 2.0637, 2.03234, 2.04589, 2.03758 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_1024_rhs16 | pytorch-cuda | 1 | 4.42201, 4.41233, 4.42245, 4.42507, 4.40679 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | pytorch-cuda | 1 | 10.2564, 10.282, 10.2662, 10.2771, 10.2805 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_4096_rhs16 | pytorch-cuda | 1 | 27.5762, 27.6574, 27.6105, 27.6471, 27.6063 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs128 | pytorch-cuda | 1 | 9.13944, 9.16533, 9.1635, 9.15346, 9.17254 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | pytorch-cuda | 1 | 12.2204, 12.2178, 12.2288, 12.2385, 12.2237 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | pytorch-cuda | 1 | 14.334, 14.3853, 14.3693, 14.3618, 14.3273 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b16_32 | pytorch-cuda | 256 | 1.85052, 1.91871, 1.82767, 1.81489, 1.81682, 1.81067, 1.80988 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

## Output retention bounds

These intrinsic single outputs exceed the configured batch budget. Samples contain one operation; the effective bound is raised explicitly to that one output, without shrinking the workload.

| Problem | Backend | Bytes/output | Configured budget | Effective bound | Ops/sample |
|---|---|---:|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-trace | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | 536870912 | 268435456 | 536870912 | 1 |

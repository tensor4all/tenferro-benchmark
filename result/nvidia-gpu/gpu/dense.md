# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261031_201808/gpu_dense.yaml`
- Timestamp: `2026-10-31T20:18:08.642468+00:00`
- tenferro-rs commit: `c15dfcf0ea5252effdef24cffbebf2f445916e20`
- Coverage: `full`; effort: `standard`; manifest version: `1`
- Selection and case accounting: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261031_201808/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100-SXM4-80GB`
- UUID: `GPU-8840a2cd-10a3-a70c-f3a2-598e6aa3610d`
- Memory: `80 GiB`
- Driver version: `595.91.07`
- CUDA version: `13.2`
- CUDA runtime: `12.8`
- cuDNN version: `91900`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `256`
- Sockets: `2`
- Cores per socket: `64`
- Threads per core: `2`
- NUMA nodes: `2`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-138-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-7,128-135; 8-15,136-143; 16-23,144-151; 24-31,152-159; 32-39,160-167; 40-47,168-175; 48-55,176-183; 56-63,184-191; 64-71,192-199; 72-79,200-207; 80-87,208-215; 88-95,216-223; 96-103,224-231; 104-111,232-239; 112-119,240-247; 120-127,248-255`
- CPU boost: `enabled`

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
| dense_batched_matmul_f64_b1024_256 | 43.821 | 43.816 | 2.545 |
| dense_batched_matmul_f64_b16_32 | 0.109 | 0.082 | 0.011 |

## gpu/dense / eigh / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_eigh_f64_1024 | 18.116 | 18.427 | 13.990 |

## gpu/dense / matmul / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_matmul_f64_3072 | 7.191 | 8.549 | 3.867 |

## gpu/dense / qr / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_qr_f64_1536 | 14.441 | 14.487 | 14.580 |

## gpu/dense / solve / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_solve_f64_1024_rhs16 | 7.247 | 6.944 | 5.114 |
| dense_solve_f64_2048_rhs128 | 12.951 | 12.309 | 9.152 |
| dense_solve_f64_2048_rhs16 | 13.197 | 30.643 | 11.563 |
| dense_solve_f64_4096_rhs16 | 45.926 | 37.113 | 29.319 |
| dense_solve_f64_512_rhs16 | 4.843 | 20.476 | 2.367 |

## gpu/dense / svd / allocating output

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; PyTorch's default row may use a different SVD driver and row-major framework layout.

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_svd_f64_256 | 15.005 | 14.541 | 12.263 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| dense_matmul_f64_3072 | tenferro-cuda-trace | 1 | 7.25228, 7.18361, 7.60964, 6.86501, 7.19082, 7.18617, 7.20146 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-trace | 1 | 51.0305, 95.9325, 22.6976, 44.5183, 43.7303, 43.8212, 43.7847 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_qr_f64_1536 | tenferro-cuda-trace | 1 | 14.4407, 14.4762, 14.4372, 14.4292, 14.4706 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_512_rhs16 | tenferro-cuda-trace | 1 | 4.84714, 4.83258, 4.84259, 4.84543, 4.83159 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_1024_rhs16 | tenferro-cuda-trace | 1 | 6.96915, 6.99486, 7.34659, 7.2555, 7.24681 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-trace | 1 | 13.1014, 13.2795, 13.197, 13.1059, 21.6683 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_4096_rhs16 | tenferro-cuda-trace | 1 | 47.1218, 68.3631, 45.9264, 35.1737, 35.48 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs128 | tenferro-cuda-trace | 1 | 13.1629, 12.9363, 12.9513, 12.9664, 12.9454 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_svd_f64_256 | tenferro-cuda-trace | 1 | 40.4791, 22.3175, 15.0053, 14.7909, 14.7003 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_eigh_f64_1024 | tenferro-cuda-trace | 1 | 18.1162, 18.0952, 19.6919, 18.1163, 18.1078 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-trace | 32 | 3.48163, 3.49344, 3.48587, 3.4687, 3.48871, 3.45361, 3.46884 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_matmul_f64_3072 | tenferro-cuda-eager | 1 | 8.55622, 8.54709, 8.54477, 8.55023, 8.5489, 8.5579, 8.53757 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | 1 | 43.7976, 43.816, 43.7813, 96.9667, 47.401, 43.7773, 44.3287 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_qr_f64_1536 | tenferro-cuda-eager | 1 | 14.6179, 14.4762, 14.4986, 14.4797, 14.4869 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_512_rhs16 | tenferro-cuda-eager | 1 | 17.6316, 20.3001, 30.4646, 83.6584, 20.4758 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_1024_rhs16 | tenferro-cuda-eager | 1 | 6.85671, 6.8847, 6.94426, 7.04814, 6.96137 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-eager | 1 | 21.2817, 37.5049, 35.4969, 30.6429, 19.7971 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_4096_rhs16 | tenferro-cuda-eager | 1 | 37.2273, 36.9492, 37.0719, 37.1131, 38.9313 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs128 | tenferro-cuda-eager | 1 | 12.2656, 12.312, 12.2594, 12.3086, 13.0087 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_svd_f64_256 | tenferro-cuda-eager | 1 | 14.5618, 15.3977, 14.5408, 14.4523, 14.458 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_eigh_f64_1024 | tenferro-cuda-eager | 1 | 18.469, 18.3908, 18.4272, 18.4243, 18.4308 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-eager | 32 | 2.61722, 2.61168, 2.61825, 2.64191, 2.66335, 2.62297, 2.60525 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_matmul_f64_3072 | pytorch-cuda | 1 | 3.86722, 3.89064, 3.86765, 3.87106, 3.86394, 3.86276, 3.86277 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | 1 | 2.55239, 2.54883, 2.55891, 2.54467, 2.53791, 2.53388, 2.53595 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_qr_f64_1536 | pytorch-cuda | 1 | 14.5813, 14.5653, 14.5602, 14.6363, 14.5798 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_512_rhs16 | pytorch-cuda | 1 | 2.36584, 2.36665, 2.36444, 2.36903, 2.37793 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_1024_rhs16 | pytorch-cuda | 1 | 5.10698, 5.1374, 5.14488, 5.11271, 5.11367 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | pytorch-cuda | 1 | 11.5582, 11.5634, 11.5663, 11.5575, 11.7144 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_4096_rhs16 | pytorch-cuda | 1 | 30.6674, 30.61, 29.3191, 27.3599, 27.3625 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs128 | pytorch-cuda | 1 | 9.15796, 9.14993, 9.1524, 9.34639, 9.13382 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | pytorch-cuda | 1 | 12.2608, 12.2762, 12.2731, 12.2629, 12.2516 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | pytorch-cuda | 1 | 13.9987, 14.0057, 13.9847, 13.99, 13.9826 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b16_32 | pytorch-cuda | 128 | 1.42578, 1.41289, 1.44592, 1.41186, 1.41258, 1.39044, 1.40834 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

## Output retention bounds

These intrinsic single outputs exceed the configured batch budget. Samples contain one operation; the effective bound is raised explicitly to that one output, without shrinking the workload.

| Problem | Backend | Bytes/output | Configured budget | Effective bound | Ops/sample |
|---|---|---:|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-trace | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | 536870912 | 268435456 | 536870912 | 1 |

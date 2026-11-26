# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261126_062950/gpu_dense.yaml`
- Timestamp: `2026-11-26T06:29:50+00:00`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Effort: `standard`; manifest version: `2`
- Selection and case accounting: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261126_062950/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd`
- Memory: `80 GiB`
- Driver version: `595.112.07`
- CUDA version: `13.2`
- CUDA runtime: `12.8`
- cuDNN version: `91900`
- Container: `devcontainer`
- CUDA toolkit: `unknown`
- CUDA toolkit root: `/opt/tenferro-benchmark-ci/cuda`
- cuTENSOR version: `2.2.0`
- cuSOLVER version: `11.7.3`

## CPU Information

- Model: `AMD EPYC 7543 32-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `128`
- Sockets: `2`
- Cores per socket: `32`
- Threads per core: `2`
- NUMA nodes: `2`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-138-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,64-67; 4-7,68-71; 8-11,72-75; 12-15,76-79; 16-19,80-83; 20-23,84-87; 24-27,88-91; 28-31,92-95; 32-35,96-99; 36-39,100-103; 40-43,104-107; 44-47,108-111; 48-51,112-115; 52-55,116-119; 56-59,120-123; 60-63,124-127`
- CPU boost: `enabled`
- Container: `devcontainer`

Median time is reported in milliseconds for `ok` records.
Inputs are prepared on the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization. tenferro-rs CUDA uses the explicit tenferro-rs synchronize API without downloading result tensors in the timed region.
Dense and einsum inputs use the same deterministic benchmark generator in the Rust and Python runners; host-to-device layout conversion remains outside the timed region.
tenferro-rs uses native column-major GPU tensors; PyTorch and vendor-wrapper columns use their native row-major framework tensors unless noted.
The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd as a QR-based cuSOLVER comparison. tenferro-rs CUDA SVD uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.
Non-`ok` cells show the structured backend status.

## gpu/dense / batched_matmul / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | 2.401 | 2.404 | 2.076 | 2.151 |
| dense_batched_matmul_f64_b16_32 | 0.027 | 0.015 | 0.011 | 0.073 |

## gpu/dense / eigh / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|---:|
| dense_eigh_f64_1024 | 14.234 | 13.975 | 14.252 | 14.242 |

## gpu/dense / matmul / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|---:|
| dense_matmul_f64_3072 | 3.168 | 3.192 | 3.125 | 3.403 |

## gpu/dense / qr / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|---:|
| dense_qr_f64_1536 | 11.839 | 11.820 | 11.777 | 11.996 |

## gpu/dense / solve / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|---:|
| dense_solve_f64_2048_rhs16 | 10.371 | 10.245 | 10.197 | 10.643 |

## gpu/dense / svd / allocating output

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; PyTorch's default row may use a different SVD driver and row-major framework layout.

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|---:|
| dense_svd_f64_256 | 11.907 | 11.947 | 12.169 | 12.082 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| dense_matmul_f64_3072 | tenferro-cuda-trace | 1 | 3.16737, 3.16851, 3.16445, 3.16558, 3.16781, 3.16954, 3.18174 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-trace | 1 | 2.40603, 2.40354, 2.39939, 2.3983, 2.40096, 2.39732, 2.40233 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_qr_f64_1536 | tenferro-cuda-trace | 1 | 11.8371, 11.844, 11.8443, 11.8213, 11.8391 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-trace | 1 | 10.3604, 10.5685, 10.3709, 10.3902, 10.3459 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_svd_f64_256 | tenferro-cuda-trace | 1 | 11.9256, 11.9353, 11.9066, 11.9046, 11.9074 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_eigh_f64_1024 | tenferro-cuda-trace | 1 | 14.2549, 14.1695, 14.2699, 14.2342, 14.1913 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-trace | 128 | 3.50548, 3.47272, 3.50171, 3.49794, 3.4965, 3.46921, 3.45606 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_matmul_f64_3072 | tenferro-cuda-eager | 1 | 3.19089, 3.1941, 3.19153, 3.19209, 3.19282, 3.19086, 3.18991 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | 1 | 2.40419, 2.41404, 2.40517, 2.40486, 2.40291, 2.40137, 2.40242 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_qr_f64_1536 | tenferro-cuda-eager | 1 | 11.9318, 11.8175, 11.8195, 11.8169, 11.8328 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-eager | 1 | 10.2245, 10.2454, 10.2649, 10.231, 10.2846 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_svd_f64_256 | tenferro-cuda-eager | 1 | 11.9535, 11.9481, 11.9406, 11.9453, 11.9475 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_eigh_f64_1024 | tenferro-cuda-eager | 1 | 14.0179, 13.9733, 13.9589, 13.9754, 13.9759 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-eager | 128 | 1.91771, 1.84045, 1.89111, 1.89722, 1.8801, 1.90519, 1.87147 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_matmul_f64_3072 | pytorch-cuda | 1 | 3.12837, 3.12455, 3.12357, 3.12335, 3.13103, 3.12521, 3.12667 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | 1 | 2.07797, 2.07418, 2.07675, 2.07624, 2.07483, 2.07741, 2.07509 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_qr_f64_1536 | pytorch-cuda | 1 | 11.7769, 11.7729, 11.7803, 11.7872, 11.7754 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | pytorch-cuda | 1 | 10.1943, 10.1968, 10.1929, 10.2003, 10.1967 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | pytorch-cuda | 1 | 12.1637, 12.1637, 12.1711, 12.1693, 12.1796 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | pytorch-cuda | 1 | 14.2519, 14.2447, 14.2407, 14.2751, 14.2793 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b16_32 | pytorch-cuda | 128 | 1.32828, 1.50342, 1.37312, 1.41849, 1.49356, 1.4257, 1.2261 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_matmul_f64_3072 | jax-cuda | 1 | 3.37987, 3.38319, 3.40363, 3.40284, 3.40508, 3.40148, 3.41505 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b1024_256 | jax-cuda | 1 | 2.17191, 2.15503, 2.1505, 2.16382, 2.14465, 2.13469, 2.14163 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_qr_f64_1536 | jax-cuda | 1 | 12.0829, 11.9706, 12.3733, 11.9959, 11.9954 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | jax-cuda | 1 | 10.7016, 10.6196, 10.5896, 10.643, 10.7137 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | jax-cuda | 1 | 12.0288, 12.2078, 13.2058, 12.0816, 12.0686 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | jax-cuda | 1 | 14.2067, 14.2111, 14.288, 14.2419, 14.2666 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b16_32 | jax-cuda | 32 | 2.34531, 2.39217, 2.32436, 2.36531, 2.26086, 2.33863, 2.3289 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

## Output retention bounds

These intrinsic single outputs exceed the configured batch budget. Samples contain one operation; the effective bound is raised explicitly to that one output, without shrinking the workload.

| Problem | Backend | Bytes/output | Configured budget | Effective bound | Ops/sample |
|---|---|---:|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-trace | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | jax-cuda | 536870912 | 268435456 | 536870912 | 1 |

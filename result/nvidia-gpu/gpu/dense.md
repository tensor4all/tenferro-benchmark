# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261108_110139/gpu_dense.yaml`
- Timestamp: `2026-11-08T11:01:39+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Effort: `standard`; manifest version: `2`
- Selection and case accounting: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261108_110139/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a`
- Memory: `80 GiB`
- Driver version: `595.91.07`
- CUDA version: `13.2`
- CUDA runtime: `12.8`
- cuDNN version: `91900`
- Container: `devcontainer`
- CUDA toolkit: `unknown`
- CUDA toolkit root: `/opt/tenferro-benchmark-ci/cuda`
- cuTENSOR version: `2.2.0`
- cuSOLVER version: `11.7.3`

## CPU Information

- Model: `AMD EPYC 9554 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `252`
- Sockets: `2`
- Cores per socket: `126`
- Threads per core: `1`
- NUMA nodes: `2`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-147-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-125; 126-251`
- Container: `devcontainer`

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
| dense_batched_matmul_f64_b1024_256 | 2.366 | 2.380 | 2.082 |
| dense_batched_matmul_f64_b16_32 | 0.021 | 0.019 | 0.007 |

## gpu/dense / eigh / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_eigh_f64_1024 | 14.329 | 13.962 | 14.318 |

## gpu/dense / matmul / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_matmul_f64_3072 | 3.174 | 3.176 | 3.128 |

## gpu/dense / qr / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_qr_f64_1536 | 11.853 | 11.873 | 11.822 |

## gpu/dense / solve / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_solve_f64_2048_rhs16 | 10.432 | 10.395 | 10.272 |

## gpu/dense / svd / allocating output

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; PyTorch's default row may use a different SVD driver and row-major framework layout.

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| dense_svd_f64_256 | 11.922 | 11.922 | 12.206 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| dense_matmul_f64_3072 | tenferro-cuda-trace | 1 | 3.17626, 3.18598, 3.17092, 3.16879, 3.16995, 3.17814, 3.17384 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-trace | 1 | 2.36188, 2.35852, 2.36077, 2.36584, 2.3668, 2.36741, 2.36714 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_qr_f64_1536 | tenferro-cuda-trace | 1 | 11.8531, 11.8418, 11.8665, 11.8544, 11.8499 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-trace | 1 | 10.4469, 10.4317, 10.4289, 10.432, 10.4477 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_svd_f64_256 | tenferro-cuda-trace | 1 | 11.9438, 11.9209, 11.9179, 11.9216, 15.7982 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_eigh_f64_1024 | tenferro-cuda-trace | 1 | 14.3295, 14.2736, 14.352, 14.2779, 14.3301 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-trace | 128 | 2.71161, 2.65054, 2.6398, 2.64048, 2.61949, 2.63436, 2.61393 | one caller-held CUDA engine for every sample; each Runtime::run call pays its own device-completion barrier inside the batch interval; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| dense_matmul_f64_3072 | tenferro-cuda-eager | 1 | 3.17543, 3.17551, 3.18172, 3.17778, 3.177, 3.17288, 3.17451 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | 1 | 2.37576, 2.37986, 2.37768, 2.37841, 2.38209, 2.38318, 2.38512 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_qr_f64_1536 | tenferro-cuda-eager | 1 | 11.8727, 11.8605, 11.8811, 11.8536, 11.88 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-eager | 1 | 10.4283, 10.4231, 10.3732, 10.3952, 10.3828 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_svd_f64_256 | tenferro-cuda-eager | 1 | 11.921, 11.9063, 11.9221, 11.9504, 11.929 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_eigh_f64_1024 | tenferro-cuda-eager | 1 | 13.9481, 13.9621, 14.0206, 14.0127, 13.9512 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-eager | 128 | 2.39574, 2.37508, 2.38529, 2.36169, 2.7662, 2.81419, 2.67007 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_matmul_f64_3072 | pytorch-cuda | 1 | 3.12835, 3.12851, 3.13382, 3.12411, 3.12766, 3.1268, 3.12452 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | 1 | 2.08221, 2.08244, 2.0787, 2.08424, 2.08387, 2.08228, 2.07584 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_qr_f64_1536 | pytorch-cuda | 1 | 11.8221, 11.8133, 11.8298, 11.8282, 11.8154 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | pytorch-cuda | 1 | 10.2535, 10.2719, 10.2829, 10.2891, 10.256 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | pytorch-cuda | 1 | 12.2011, 12.2063, 12.2717, 12.2197, 12.1784 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | pytorch-cuda | 1 | 14.3181, 14.3624, 14.2834, 14.3059, 14.3277 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b16_32 | pytorch-cuda | 256 | 1.73695, 1.73382, 1.73882, 1.76323, 1.71975, 1.74242, 1.72757 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

## Output retention bounds

These intrinsic single outputs exceed the configured batch budget. Samples contain one operation; the effective bound is raised explicitly to that one output, without shrinking the workload.

| Problem | Backend | Bytes/output | Configured budget | Effective bound | Ops/sample |
|---|---|---:|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-trace | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | 536870912 | 268435456 | 536870912 | 1 |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | 536870912 | 268435456 | 536870912 | 1 |

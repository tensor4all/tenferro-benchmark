# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense`
- Suite file: `benchmarks/gpu/dense.yaml`
- Timestamp: `2026-09-19T14:27:23.269205+00:00`
- tenferro-rs commit: `d7a8c60caedac9aa3e630eea793d606df02dfc09`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.126.09`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92000`

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

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [dense_batched_matmul_f64_b1024_256](../../../notes/nvidia-gpu/gpu/dense.md#dense_batched_matmul_f64_b1024_256) | 2.152 | 2.167 | 2.079 | 2.083 | not configured | unsupported | unsupported | unsupported |

## gpu/dense / eigh / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_eigh_f64_1024 | 14.362 | 14.334 | 14.315 | unsupported | unsupported | 14.295 | unsupported | unsupported |

## gpu/dense / matmul / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_3072 | 3.218 | 3.220 | 3.127 | 3.126 | not configured | unsupported | unsupported | unsupported |

## gpu/dense / qr / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_qr_f64_1536 | 11.948 | 11.976 | 11.797 | unsupported | unsupported | 11.788 | unsupported | unsupported |

## gpu/dense / solve / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_solve_f64_1024_rhs16 | 4.895 | 5.089 | 4.415 | unsupported | unsupported | 4.427 | unsupported | unsupported |
| dense_solve_f64_2048_rhs128 | 10.136 | 10.015 | 9.141 | unsupported | unsupported | 9.155 | unsupported | unsupported |
| dense_solve_f64_2048_rhs16 | 11.070 | 10.863 | 10.212 | unsupported | unsupported | 10.244 | unsupported | unsupported |
| dense_solve_f64_4096_rhs16 | 28.305 | 28.095 | 27.582 | unsupported | unsupported | 27.654 | unsupported | unsupported |
| dense_solve_f64_512_rhs16 | 2.576 | 2.509 | 2.049 | unsupported | unsupported | 2.040 | unsupported | unsupported |

## gpu/dense / svd / allocating output

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; PyTorch's default row may use a different SVD driver and row-major framework layout.

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_svd_f64_256 | 12.139 | 12.202 | 12.210 | unsupported | unsupported | 23.296 | unsupported | unsupported |

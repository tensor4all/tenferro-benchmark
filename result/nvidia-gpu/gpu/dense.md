# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense`
- Suite file: `benchmarks/gpu/dense.yaml`
- Timestamp: `2026-07-28T03:48:34.224093+00:00`
- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.126.09`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92400`

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

## gpu/dense / batched_matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | 2.155 | 2.298 | 2.078 | 2.078 | not configured | unsupported | unsupported | unsupported |

## gpu/dense / eigh

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_eigh_f64_1024 | 14.164 | 14.241 | 13.968 | unsupported | unsupported | 13.946 | unsupported | unsupported |

## gpu/dense / matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_3072 | 3.207 | 3.261 | 3.124 | 3.123 | not configured | unsupported | unsupported | unsupported |

## gpu/dense / qr

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_qr_f64_1536 | 12.003 | 11.994 | 11.780 | unsupported | unsupported | 11.777 | unsupported | unsupported |

## gpu/dense / solve

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_solve_f64_1024_rhs16 | 5.157 | 4.873 | 4.415 | unsupported | unsupported | 4.429 | unsupported | unsupported |
| dense_solve_f64_2048_rhs128 | 9.996 | 9.994 | 9.148 | unsupported | unsupported | 9.162 | unsupported | unsupported |
| dense_solve_f64_2048_rhs16 | 11.051 | 11.076 | 10.206 | unsupported | unsupported | 10.255 | unsupported | unsupported |
| dense_solve_f64_4096_rhs16 | 28.602 | 28.655 | 27.541 | unsupported | unsupported | 27.641 | unsupported | unsupported |
| dense_solve_f64_512_rhs16 | 2.766 | 2.536 | 2.048 | unsupported | unsupported | 2.037 | unsupported | unsupported |

## gpu/dense / svd

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; PyTorch's default row may use a different SVD driver and row-major framework layout.

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_svd_f64_256 | 12.202 | 12.197 | 12.198 | unsupported | unsupported | 23.297 | unsupported | unsupported |

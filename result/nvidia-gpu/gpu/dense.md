# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense`
- Suite file: `benchmarks/gpu/dense.yaml`
- Timestamp: `2026-06-09T03:56:28.462842+00:00`
- tenferro-rs commit: `d30e592289abf59a7d91f0ec56eaadbe864220a6`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.126.09`
- CUDA version: `13.0`
- CUDA runtime: `12.6`
- cuDNN version: `92300`

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
tenferro-rs uses native column-major GPU tensors; Torch, JAX, and vendor-wrapper columns use their native row-major framework tensors unless noted.
The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd as a QR-based cuSOLVER comparison. tenferro-rs CUDA SVD uses its backend default driver policy, currently JAX-compatible gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.
Non-`ok` cells show the structured backend status.

## gpu/dense / batched_matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | 2.283 | 2.214 | 2.100 | 2.285 | 2.098 | not configured | unsupported | unsupported | unsupported |

## gpu/dense / eigh

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_eigh_f64_1024 | 14.001 | 13.998 | 14.254 | 14.508 | unsupported | unsupported | 16.850 | unsupported | unsupported |

## gpu/dense / matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_3072 | 3.212 | 3.209 | 3.130 | 3.271 | 3.131 | not configured | unsupported | unsupported | unsupported |

## gpu/dense / qr

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_qr_f64_1536 | 11.995 | 11.998 | 11.768 | 11.885 | unsupported | unsupported | 11.761 | unsupported | unsupported |

## gpu/dense / solve

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_solve_f64_1024_rhs16 | 4.741 | 4.673 | 4.548 | 4.555 | unsupported | unsupported | 4.432 | unsupported | unsupported |
| dense_solve_f64_2048_rhs128 | 9.634 | 9.622 | 9.121 | 9.425 | unsupported | unsupported | 9.139 | unsupported | unsupported |
| dense_solve_f64_2048_rhs16 | 10.483 | 10.617 | 10.201 | 10.583 | unsupported | unsupported | 10.251 | unsupported | unsupported |
| dense_solve_f64_4096_rhs16 | 27.588 | 27.599 | 27.847 | 28.272 | unsupported | unsupported | 27.871 | unsupported | unsupported |
| dense_solve_f64_512_rhs16 | 2.382 | 2.392 | 2.048 | 2.152 | unsupported | unsupported | 2.048 | unsupported | unsupported |

## gpu/dense / svd

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently JAX-compatible gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; compare PyTorch/JAX default rows separately because they may use different SVD drivers and row-major framework layouts.

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_svd_f64_256 | 12.201 | 12.200 | 12.184 | 11.970 | unsupported | unsupported | 23.078 | unsupported | unsupported |

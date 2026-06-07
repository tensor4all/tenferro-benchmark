# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense`
- Suite file: `benchmarks/gpu/dense.yaml`
- Timestamp: `2026-06-07T02:37:44.351329+00:00`
- tenferro-rs commit: `d30e592289abf59a7d91f0ec56eaadbe864220a6`

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
| dense_batched_matmul_f64_b1024_256 | 2.212 | 2.304 | 2.094 | 2.311 | 2.091 | 61.241 | unsupported | unsupported | unsupported |

## gpu/dense / eigh

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_eigh_f64_1024 | 14.070 | 13.998 | 14.231 | 14.493 | unsupported | unsupported | 13.994 | unsupported | unsupported |

## gpu/dense / matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_3072 | 3.390 | 3.236 | 3.133 | 3.312 | 3.127 | 3.481 | unsupported | unsupported | unsupported |

## gpu/dense / qr

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_qr_f64_1536 | 12.151 | 12.145 | 11.794 | 11.869 | unsupported | unsupported | 11.873 | unsupported | unsupported |

## gpu/dense / solve

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_solve_f64_1024_rhs16 | 4.657 | 4.737 | 4.416 | 4.568 | unsupported | unsupported | 5.984 | unsupported | unsupported |
| dense_solve_f64_2048_rhs128 | 9.726 | 9.581 | 9.140 | 9.451 | unsupported | unsupported | 9.177 | unsupported | unsupported |
| dense_solve_f64_2048_rhs16 | 10.614 | 10.620 | 10.214 | 10.608 | unsupported | unsupported | 10.286 | unsupported | unsupported |
| dense_solve_f64_4096_rhs16 | 27.708 | 27.630 | 27.713 | 28.374 | unsupported | unsupported | 27.894 | unsupported | unsupported |
| dense_solve_f64_512_rhs16 | 2.464 | 2.295 | 2.074 | 2.161 | unsupported | unsupported | 2.042 | unsupported | unsupported |

## gpu/dense / svd

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently JAX-compatible gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; compare PyTorch/JAX default rows separately because they may use different SVD drivers and row-major framework layouts.

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_svd_f64_256 | 12.205 | 12.215 | 12.190 | 11.933 | unsupported | unsupported | 23.149 | unsupported | unsupported |

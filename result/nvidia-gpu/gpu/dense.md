# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense`
- Suite file: `benchmarks/gpu/dense.yaml`
- Timestamp: `2026-07-11T05:42:50.797056+00:00`
- tenferro-rs commit: `d5c768c7eb58f252e7855fea80bb6af5bb7ddb40`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA GeForce RTX 3060`
- UUID: `GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`
- Memory: `12 GiB`
- Driver version: `580.159.03`
- CUDA version: `13.0`
- CUDA runtime: `12.6`
- cuDNN version: `92000`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-134-generic-x86_64-with-glibc2.39`

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
| dense_batched_matmul_f64_b1024_256 | 187.771 | 187.789 | 185.451 | not configured | 185.415 | 1320.999 | unsupported | unsupported | unsupported |

## gpu/dense / eigh

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_eigh_f64_1024 | 74.624 | 74.604 | 74.920 | not configured | unsupported | unsupported | 74.799 | unsupported | unsupported |

## gpu/dense / matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_3072 | 314.397 | 314.413 | 312.095 | not configured | 312.061 | 317.962 | unsupported | unsupported | unsupported |

## gpu/dense / qr

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_qr_f64_1536 | 96.087 | 96.190 | 95.218 | not configured | unsupported | unsupported | 95.438 | unsupported | unsupported |

## gpu/dense / solve

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_solve_f64_1024_rhs16 | 11.840 | 11.588 | 10.591 | not configured | unsupported | unsupported | 10.604 | unsupported | unsupported |
| dense_solve_f64_2048_rhs128 | 55.168 | 54.826 | 53.560 | not configured | unsupported | unsupported | 53.610 | unsupported | unsupported |
| dense_solve_f64_2048_rhs16 | 48.586 | 48.886 | 48.109 | not configured | unsupported | unsupported | 48.072 | unsupported | unsupported |
| dense_solve_f64_4096_rhs16 | 288.860 | 288.893 | 288.060 | not configured | unsupported | unsupported | 288.111 | unsupported | unsupported |
| dense_solve_f64_512_rhs16 | 3.979 | 3.790 | 3.039 | not configured | unsupported | unsupported | 3.028 | unsupported | unsupported |

## gpu/dense / svd

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently JAX-compatible gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; compare PyTorch/JAX default rows separately because they may use different SVD drivers and row-major framework layouts.

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_svd_f64_256 | 82.595 | 82.595 | 84.124 | not configured | unsupported | unsupported | 27.538 | unsupported | unsupported |

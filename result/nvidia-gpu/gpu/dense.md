# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense`
- Suite file: `benchmarks/gpu/dense.yaml`
- Timestamp: `2026-07-16T09:04:58.729638+00:00`
- tenferro-rs commit: `8c267c636537d3d407925b61a2000f9bbe07e6a1`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA GeForce RTX 3060`
- UUID: `GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`
- Memory: `12 GiB`
- Driver version: `580.159.03`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92400`

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
| dense_batched_matmul_f64_b1024_256 | 186.999 | 186.998 | 186.093 | 160.777 | 186.834 | not configured | unsupported | unsupported | unsupported |

## gpu/dense / eigh

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_eigh_f64_1024 | 74.405 | 74.557 | 74.794 | 75.222 | unsupported | unsupported | 74.820 | unsupported | unsupported |

## gpu/dense / matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_3072 | 314.405 | 314.404 | 316.202 | 269.632 | 313.314 | not configured | unsupported | unsupported | unsupported |

## gpu/dense / qr

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_qr_f64_1536 | 95.794 | 95.385 | 95.467 | 96.527 | unsupported | unsupported | 95.452 | unsupported | unsupported |

## gpu/dense / solve

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_solve_f64_1024_rhs16 | 11.602 | 11.599 | 10.599 | 11.022 | unsupported | unsupported | 10.609 | unsupported | unsupported |
| dense_solve_f64_2048_rhs128 | 54.628 | 54.573 | 53.578 | 54.267 | unsupported | unsupported | 53.627 | unsupported | unsupported |
| dense_solve_f64_2048_rhs16 | 48.872 | 48.210 | 48.170 | 48.690 | unsupported | unsupported | 48.114 | unsupported | unsupported |
| dense_solve_f64_4096_rhs16 | 287.005 | 287.121 | 288.105 | 289.098 | unsupported | unsupported | 288.410 | unsupported | unsupported |
| dense_solve_f64_512_rhs16 | 3.802 | 3.821 | 3.049 | 3.371 | unsupported | unsupported | 3.019 | unsupported | unsupported |

## gpu/dense / svd

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently JAX-compatible gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; compare PyTorch/JAX default rows separately because they may use different SVD drivers and row-major framework layouts.

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_svd_f64_256 | 81.792 | 81.799 | 84.056 | 82.147 | unsupported | unsupported | 27.468 | unsupported | unsupported |

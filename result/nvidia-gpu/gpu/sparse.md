# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/sparse`
- Suite file: `benchmarks/gpu/sparse.yaml`
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

## gpu/sparse / spmm

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | unsupported | unsupported | 108.414 | unsupported | unsupported | unsupported | unsupported | 108.405 | 1182.073 |

## gpu/sparse / spmv

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_synthetic_4m_64m_spmv_f64 | unsupported | unsupported | 19.621 | unsupported | unsupported | unsupported | unsupported | 19.632 | 98.922 |

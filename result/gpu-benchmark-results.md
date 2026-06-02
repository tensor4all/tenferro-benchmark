# GPU Benchmark Results

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
The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd to match tenferro-rs raw gesvd more closely, but it is still not a raw cuSOLVER API benchmark.
Non-`ok` cells show the structured backend status.

## gpu_dense_contract_v1 / batched_matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | 2.208 | 2.320 | 2.102 | 2.101 | 2.356 | 2.101 | 61.361 | unsupported | unsupported | unsupported |

## gpu_dense_contract_v1 / eigh

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_eigh_f64_1024 | 14.074 | 13.998 | 14.325 | 14.223 | 14.465 | unsupported | unsupported | 13.959 | unsupported | unsupported |

## gpu_dense_contract_v1 / matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_3072 | 3.373 | 3.379 | 3.138 | 3.129 | 3.331 | 3.129 | 3.484 | unsupported | unsupported | unsupported |

## gpu_dense_contract_v1 / qr

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_qr_f64_1536 | 12.137 | 12.010 | 11.814 | 11.811 | 12.028 | unsupported | unsupported | 11.819 | unsupported | unsupported |

## gpu_dense_contract_v1 / solve

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_solve_f64_512_rhs16 | 2.679 | 2.517 | 2.060 | 2.085 | 2.147 | unsupported | unsupported | 2.058 | unsupported | unsupported |

## gpu_dense_contract_v1 / svd

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. The cuSOLVER column pins torch.linalg.svd driver=gesvd to match tenferro-rs raw gesvd more closely; compare PyTorch/LibTorch/JAX default rows separately because they may use different SVD drivers and row-major framework layouts.

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_svd_f64_256 | 26.629 | 27.018 | 12.402 | 12.234 | 11.982 | unsupported | unsupported | 23.865 | unsupported | unsupported |

## gpu_einsum_contract_v1 / einsum

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| einsum_bin_matmul_3072_f64 | 3.414 | 3.413 | 3.178 | 3.183 | 3.315 | 3.179 | 3.490 | unsupported | unsupported | unsupported |

## gpu_sparse_contract_v1 / spmm

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | unsupported | unsupported | 14.887 | 14.883 | unsupported | unsupported | unsupported | unsupported | 14.970 | 223.400 |

## gpu_sparse_contract_v1 / spmv

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_synthetic_4m_64m_spmv_f64 | unsupported | unsupported | 2.341 | 2.343 | unsupported | unsupported | unsupported | unsupported | 2.338 | 27.072 |

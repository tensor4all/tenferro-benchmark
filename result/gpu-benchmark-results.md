# GPU Benchmark Results

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-124-generic-x86_64-with-glibc2.39`

Median time is reported in milliseconds for `ok` records.
Inputs are prepared on the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization. tenferro-rs CUDA uses stream synchronization without downloading result tensors in the timed region.
Non-`ok` cells show the structured backend status.

## gpu_dense_contract_v1 / batched_matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | 186.204 | 186.203 | 186.381 | 186.368 | 160.525 | 186.370 | 1314.877 | unsupported | unsupported | unsupported |

## gpu_dense_contract_v1 / eigh

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_eigh_f64_1024 | 74.622 | 235.978 | 72.068 | 71.991 | 72.350 | unsupported | unsupported | 72.096 | unsupported | unsupported |

## gpu_dense_contract_v1 / matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_3072 | 313.230 | 313.421 | 316.857 | 316.844 | 273.015 | 313.723 | 319.684 | unsupported | unsupported | unsupported |

## gpu_dense_contract_v1 / qr

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_qr_f64_1536 | 96.202 | 234.781 | 95.247 | 95.055 | 96.316 | unsupported | unsupported | 95.061 | unsupported | unsupported |

## gpu_dense_contract_v1 / solve

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_solve_f64_512_rhs16 | 3.683 | 1930.995 | 3.032 | 3.016 | 3.298 | unsupported | unsupported | 3.022 | unsupported | unsupported |

## gpu_dense_contract_v1 / svd

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_svd_f64_256 | 28.582 | 186.998 | 336.165 | 336.180 | 327.900 | unsupported | unsupported | 336.192 | unsupported | unsupported |

## gpu_einsum_contract_v1 / einsum

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| einsum_bin_matmul_3072_f64 | 314.785 | 314.592 | 315.315 | 315.324 | 271.838 | 315.305 | 321.322 | unsupported | unsupported | unsupported |

## gpu_sparse_contract_v1 / spmm

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | unsupported | unsupported | 108.492 | 108.363 | unsupported | unsupported | unsupported | unsupported | 108.462 | 1154.358 |

## gpu_sparse_contract_v1 / spmv

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_synthetic_4m_64m_spmv_f64 | unsupported | unsupported | 19.809 | 19.805 | unsupported | unsupported | unsupported | unsupported | 19.813 | 102.793 |

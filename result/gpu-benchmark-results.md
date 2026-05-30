# GPU Benchmark Results

Median time is reported in milliseconds for `ok` records.
Non-`ok` cells show the structured backend status.

## gpu_dense_contract_v1 / batched_matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_batched_matmul_f64_b16_32 | 0.138 | 0.143 | 0.057 | 0.057 | 0.125 | 0.056 | 2.941 | unsupported | unsupported | unsupported |

## gpu_dense_contract_v1 / eigh

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_eigh_f64_64 | 2.220 | 160.378 | 1.849 | 1.847 | 2.191 | unsupported | unsupported | 1.914 | unsupported | unsupported |

## gpu_dense_contract_v1 / matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_256 | 0.574 | 0.544 | 0.296 | 0.297 | 0.116 | 0.293 | 1.408 | unsupported | unsupported | unsupported |

## gpu_dense_contract_v1 / qr

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_qr_f64_64 | 1.163 | 176.357 | 0.914 | 0.914 | 1.197 | unsupported | unsupported | 0.925 | unsupported | unsupported |

## gpu_dense_contract_v1 / solve

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_solve_f64_64_rhs4 | 0.689 | 2004.503 | 0.241 | 0.240 | 0.365 | unsupported | unsupported | 0.249 | unsupported | unsupported |

## gpu_dense_contract_v1 / svd

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_svd_f64_64 | 5.538 | 189.221 | 41.343 | 38.802 | 36.854 | unsupported | unsupported | 38.790 | unsupported | unsupported |

## gpu_einsum_contract_v1 / einsum

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| einsum_bin_matmul_256_f64 | 0.670 | 0.644 | 0.429 | 0.350 | 0.231 | 0.358 | 1.335 | unsupported | unsupported | unsupported |

## gpu_sparse_contract_v1 / spmm

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_bcsstk17_spmm_f64_rhs32 | unsupported | unsupported | 0.396 | 0.403 | unsupported | unsupported | unsupported | unsupported | 0.402 | 2.541 |

## gpu_sparse_contract_v1 / spmv

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_bcspwr10_spmv_f64 | unsupported | unsupported | 0.134 | 0.132 | unsupported | unsupported | unsupported | unsupported | 0.131 | 1.406 |

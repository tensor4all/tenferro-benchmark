# GPU Benchmark Results

Median time is reported in milliseconds for `ok` records.
Non-`ok` cells show the structured backend status.

## gpu_dense_contract_v1 / batched_matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_batched_matmul_f64_b16_32 | 0.142 | 0.134 | 0.057 | 0.060 | 0.120 | 0.054 | 2.942 | unsupported | unsupported | unsupported |

## gpu_dense_contract_v1 / eigh

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_eigh_f64_64 | 2.064 | 153.149 | 1.850 | 1.848 | 2.167 | unsupported | unsupported | 1.893 | unsupported | unsupported |

## gpu_dense_contract_v1 / matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_256 | 0.581 | 0.615 | 0.295 | 0.299 | 0.118 | 0.296 | 1.412 | unsupported | unsupported | unsupported |

## gpu_dense_contract_v1 / qr

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_qr_f64_64 | 1.182 | 154.654 | 0.930 | 0.929 | 1.193 | unsupported | unsupported | 0.921 | unsupported | unsupported |

## gpu_dense_contract_v1 / solve

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_solve_f64_64_rhs4 | 0.527 | 1968.921 | 0.240 | 0.263 | 0.437 | unsupported | unsupported | 0.251 | unsupported | unsupported |

## gpu_dense_contract_v1 / svd

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_svd_f64_64 | 5.201 | 176.360 | 41.350 | 38.810 | 36.949 | unsupported | unsupported | 38.793 | unsupported | unsupported |

## gpu_einsum_contract_v1 / einsum

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| einsum_bin_matmul_256_f64 | 0.653 | 0.652 | 0.411 | 0.408 | 0.125 | 0.399 | 1.319 | unsupported | unsupported | unsupported |

## gpu_sparse_contract_v1 / spmm

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_bcsstk17_spmm_f64_rhs32 | unsupported | unsupported | 0.392 | 0.393 | unsupported | unsupported | unsupported | unsupported | 0.397 | 2.508 |

## gpu_sparse_contract_v1 / spmv

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_bcspwr10_spmv_f64 | unsupported | unsupported | 0.096 | 0.096 | unsupported | unsupported | unsupported | unsupported | 0.098 | 1.375 |

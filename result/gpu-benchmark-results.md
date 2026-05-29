# GPU Benchmark Results

Median time is reported in milliseconds for `ok` records.
Non-`ok` cells show the structured backend status.

## gpu_dense_contract_v1 / batched_matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_batched_matmul_f64_b16_32 | 0.142 | 0.149 | 0.050 | not configured | 0.122 | not configured | not configured | unsupported | unsupported | unsupported |

## gpu_dense_contract_v1 / eigh

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_eigh_f64_64 | 2.089 | 157.785 | 1.853 | not configured | 2.170 | unsupported | unsupported | not configured | unsupported | unsupported |

## gpu_dense_contract_v1 / matmul

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_256 | 0.570 | 0.548 | 0.301 | not configured | 0.127 | not configured | not configured | unsupported | unsupported | unsupported |

## gpu_dense_contract_v1 / qr

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_qr_f64_64 | 1.255 | 165.763 | 0.924 | not configured | 1.220 | unsupported | unsupported | not configured | unsupported | unsupported |

## gpu_dense_contract_v1 / solve

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_solve_f64_64_rhs4 | 0.672 | 2021.470 | 0.308 | not configured | 0.418 | unsupported | unsupported | not configured | unsupported | unsupported |

## gpu_dense_contract_v1 / svd

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_svd_f64_64 | 5.606 | 191.521 | 41.322 | not configured | 36.874 | unsupported | unsupported | not configured | unsupported | unsupported |

## gpu_einsum_contract_v1 / einsum

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| einsum_bin_matmul_256_f64 | 0.563 | 0.560 | 0.328 | not configured | 0.120 | not configured | not configured | unsupported | unsupported | unsupported |

## gpu_sparse_contract_v1 / spmm

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_bcsstk17_spmm_f64_rhs32 | unsupported | unsupported | 50.839 | not configured | unsupported | unsupported | unsupported | unsupported | not configured | not configured |

## gpu_sparse_contract_v1 / spmv

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_bcspwr10_spmv_f64 | unsupported | unsupported | 1.450 | not configured | unsupported | unsupported | unsupported | unsupported | not configured | not configured |

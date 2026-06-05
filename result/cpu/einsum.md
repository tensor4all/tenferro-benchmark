# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/cpu/einsum/20260605_195601/run.yaml`
- Timestamp: `20260605_195601`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/cpu/einsum/20260605_195601`.

- tenferro-rs commit: `f28016c6dca2b1c69f95330e17025d59de714a2f`

## CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5-arm64-arm-64bit`

## Thread Environment

- OMP_NUM_THREADS: `1`
- OMP_THREAD_LIMIT: `1`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `1`
- OPENBLAS_NUM_THREADS: `1`
- GOTO_NUM_THREADS: `1`
- MKL_NUM_THREADS: `1`
- VECLIB_MAXIMUM_THREADS: `1`
- VECLIB_NUM_THREADS: `1`
- NUMEXPR_NUM_THREADS: `1`
- BLIS_NUM_THREADS: `1`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 1

- Source table: `data/results/cpu/einsum/20260605_195601/einsum_table_t1_20260605_195601.md`

Logs:

- `data/results/cpu/einsum/20260605_195601/tenferro_trace_t1_20260605_195601.log`
- `data/results/cpu/einsum/20260605_195601/tenferro_eager_t1_20260605_195601.log`
- `data/results/cpu/einsum/20260605_195601/pytorch_cpu_t1_20260605_195601.log`
- `data/results/cpu/einsum/20260605_195601/jax_cpu_t1_20260605_195601.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.396 ± 0.040 | 0.571 ± 0.100 | - | 0.398 ± 0.027 | **0.361 ± 0.039** |
| bin_batched_matmul_b32_m64_n64_k64 | **0.063 ± 0.002** | 0.083 ± 0.001 | - | 0.096 ± 0.002 | 0.161 ± 0.014 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.110 ± 0.017** | 0.138 ± 0.011 | - | 0.120 ± 0.006 | 0.126 ± 0.016 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.762 ± 0.069 | 0.128 ± 0.001 | - | **0.122 ± 0.003** | 0.124 ± 0.017 |
| bin_elementwise_mul_2048x2048 | 1.023 ± 0.081 | 1.096 ± 0.118 | - | **0.808 ± 0.031** | 0.825 ± 0.064 |
| bin_matmul_1024 | 4.810 ± 0.351 | 4.467 ± 0.202 | - | 4.825 ± 1.508 | **3.464 ± 0.119** |
| bin_matmul_256 | 0.088 ± 0.006 | **0.084 ± 0.005** | - | 0.095 ± 0.004 | 0.207 ± 0.016 |
| bin_outer_product_4096 | 1.229 ± 0.114 | 1.251 ± 0.040 | - | 1.482 ± 0.051 | **0.635 ± 0.030** |
| gm_queen5_5_3.wcsp | 1033.155 ± 38.589 | 1513.575 ± 11.067 | - | 861.097 ± 10.353 | **453.445 ± 3.427** |
| lm_batch_likelihood_brackets_4_4d | 7.595 ± 0.132 | 13.752 ± 0.328 | - | 9.467 ± 0.566 | **5.288 ± 0.144** |
| lm_batch_likelihood_sentence_3_12d | 9.087 ± 0.129 | 22.250 ± 0.132 | - | 18.179 ± 0.162 | **6.836 ± 0.103** |
| lm_batch_likelihood_sentence_4_4d | 6.702 ± 0.197 | 14.478 ± 0.349 | - | 9.536 ± 0.393 | **5.619 ± 0.141** |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.008 ± 0.000 | - | 0.027 ± 0.003 | 0.075 ± 0.004 |
| str_matrix_chain_multiplication_100 | **1.833 ± 0.043** | 2.272 ± 0.119 | - | 2.750 ± 0.042 | 6.336 ± 0.135 |
| str_mps_varying_inner_product_200 | **4.328 ± 0.043** | 7.629 ± 0.134 | - | 5.300 ± 0.814 | 13.856 ± 0.132 |
| str_nw_mera_closed_120 | 164.513 ± 3.073 | 182.162 ± 1.520 | - | 158.384 ± 0.950 | **87.182 ± 1.959** |
| str_nw_mera_open_26 | 181.677 ± 9.794 | 189.753 ± 1.263 | - | 119.728 ± 0.426 | **67.716 ± 3.765** |
| tensornetwork_permutation_focus_step409_316 | 120.688 ± 4.305 | 189.337 ± 10.598 | - | 157.849 ± 0.924 | **76.927 ± 6.249** |
| tensornetwork_permutation_light_415 | 120.925 ± 1.429 | 153.764 ± 2.089 | - | 137.990 ± 0.968 | **84.632 ± 5.765** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.396 ± 0.040 | 0.571 ± 0.100 | - | 0.398 ± 0.027 | **0.361 ± 0.039** |
| bin_batched_matmul_b32_m64_n64_k64 | **0.063 ± 0.002** | 0.083 ± 0.001 | - | 0.096 ± 0.002 | 0.161 ± 0.014 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.110 ± 0.017** | 0.138 ± 0.011 | - | 0.120 ± 0.006 | 0.126 ± 0.016 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.762 ± 0.069 | 0.128 ± 0.001 | - | **0.122 ± 0.003** | 0.124 ± 0.017 |
| bin_elementwise_mul_2048x2048 | 1.023 ± 0.081 | 1.096 ± 0.118 | - | **0.808 ± 0.031** | 0.825 ± 0.064 |
| bin_matmul_1024 | 4.810 ± 0.351 | 4.467 ± 0.202 | - | 4.825 ± 1.508 | **3.464 ± 0.119** |
| bin_matmul_256 | 0.088 ± 0.006 | **0.084 ± 0.005** | - | 0.095 ± 0.004 | 0.207 ± 0.016 |
| bin_outer_product_4096 | 1.229 ± 0.114 | 1.251 ± 0.040 | - | 1.482 ± 0.051 | **0.635 ± 0.030** |
| gm_queen5_5_3.wcsp | 338.972 ± 9.369 | 508.169 ± 2.360 | - | 319.908 ± 10.493 | **154.638 ± 7.278** |
| lm_batch_likelihood_brackets_4_4d | 7.972 ± 0.103 | 14.317 ± 0.394 | - | 9.605 ± 0.252 | **5.585 ± 0.282** |
| lm_batch_likelihood_sentence_3_12d | 11.022 ± 0.286 | 22.030 ± 0.427 | - | 17.754 ± 0.107 | **7.567 ± 0.270** |
| lm_batch_likelihood_sentence_4_4d | 7.803 ± 0.211 | 14.703 ± 0.120 | - | 9.404 ± 0.211 | **5.965 ± 0.529** |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.008 ± 0.000 | - | 0.027 ± 0.003 | 0.075 ± 0.004 |
| str_matrix_chain_multiplication_100 | **1.815 ± 0.039** | 2.375 ± 0.029 | - | 2.752 ± 0.066 | 6.946 ± 0.307 |
| str_mps_varying_inner_product_200 | **6.253 ± 0.246** | 8.103 ± 0.132 | - | 6.590 ± 0.710 | 15.916 ± 0.347 |
| str_nw_mera_closed_120 | 148.208 ± 0.934 | 155.576 ± 0.625 | - | 130.107 ± 0.424 | **82.426 ± 2.460** |
| str_nw_mera_open_26 | 180.165 ± 1.732 | 191.052 ± 1.794 | - | 124.755 ± 0.594 | **74.626 ± 1.556** |
| tensornetwork_permutation_focus_step409_316 | 120.688 ± 4.305 | 189.337 ± 10.598 | - | 157.849 ± 0.924 | **76.927 ± 6.249** |
| tensornetwork_permutation_light_415 | 120.925 ± 1.429 | 153.764 ± 2.089 | - | 137.990 ± 0.968 | **84.632 ± 5.765** |

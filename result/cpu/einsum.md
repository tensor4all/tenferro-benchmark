# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/cpu/einsum/20260605_225341/run.yaml`
- Timestamp: `20260605_225341`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/cpu/einsum/20260605_225341`.

- tenferro-rs commit: `b17186e352184fd4244be604d6482b82a1264aef`

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

- OMP_NUM_THREADS: `4`
- OMP_THREAD_LIMIT: `4`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `4`
- OPENBLAS_NUM_THREADS: `4`
- GOTO_NUM_THREADS: `4`
- MKL_NUM_THREADS: `4`
- VECLIB_MAXIMUM_THREADS: `4`
- VECLIB_NUM_THREADS: `4`
- NUMEXPR_NUM_THREADS: `4`
- BLIS_NUM_THREADS: `4`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4`

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 4

- Source table: `data/results/cpu/einsum/20260605_225341/einsum_table_t4_20260605_225341.md`

Logs:

- `data/results/cpu/einsum/20260605_225341/tenferro_trace_t4_20260605_225341.log`
- `data/results/cpu/einsum/20260605_225341/tenferro_eager_t4_20260605_225341.log`
- `data/results/cpu/einsum/20260605_225341/pytorch_cpu_t4_20260605_225341.log`
- `data/results/cpu/einsum/20260605_225341/jax_cpu_t4_20260605_225341.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.568 ± 0.077 | 0.573 ± 0.040 | - | 0.398 ± 0.006 | **0.346 ± 0.093** |
| bin_batched_matmul_b32_m64_n64_k64 | **0.081 ± 0.000** | 0.083 ± 0.001 | - | 0.096 ± 0.002 | 0.186 ± 0.020 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.111 ± 0.002 | 0.132 ± 0.003 | - | **0.065 ± 0.013** | 0.124 ± 0.023 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.817 ± 0.078 | 0.129 ± 0.003 | - | **0.059 ± 0.005** | 0.127 ± 0.018 |
| bin_elementwise_mul_2048x2048 | 0.960 ± 0.026 | 1.078 ± 0.077 | - | **0.572 ± 0.012** | 0.758 ± 0.050 |
| bin_matmul_1024 | 4.393 ± 0.181 | 4.428 ± 0.172 | - | 4.769 ± 0.154 | **3.323 ± 0.246** |
| bin_matmul_256 | **0.085 ± 0.000** | **0.085 ± 0.000** | - | 0.093 ± 0.001 | 0.220 ± 0.026 |
| bin_outer_product_4096 | 1.228 ± 0.032 | 1.231 ± 0.012 | - | **0.571 ± 0.016** | 0.624 ± 0.011 |
| gm_queen5_5_3.wcsp | 1006.463 ± 3.720 | 1495.411 ± 8.599 | - | **416.876 ± 10.742** | 450.149 ± 2.317 |
| lm_batch_likelihood_brackets_4_4d | 7.479 ± 0.108 | 13.078 ± 0.188 | - | 9.472 ± 1.107 | **5.799 ± 0.113** |
| lm_batch_likelihood_sentence_3_12d | 8.636 ± 0.075 | 21.001 ± 0.895 | - | 21.017 ± 6.913 | **6.693 ± 0.146** |
| lm_batch_likelihood_sentence_4_4d | 6.613 ± 0.091 | 14.273 ± 0.528 | - | 9.554 ± 0.606 | **5.745 ± 0.125** |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.008 ± 0.000 | - | 0.027 ± 0.003 | 0.074 ± 0.007 |
| str_matrix_chain_multiplication_100 | **1.733 ± 0.014** | 2.292 ± 0.024 | - | 2.580 ± 0.164 | 6.015 ± 0.157 |
| str_mps_varying_inner_product_200 | **4.325 ± 0.037** | 7.631 ± 0.161 | - | 5.677 ± 0.296 | 13.615 ± 0.217 |
| str_nw_mera_closed_120 | 160.421 ± 0.597 | 179.561 ± 3.269 | - | 148.012 ± 8.420 | **88.177 ± 1.752** |
| str_nw_mera_open_26 | 132.492 ± 0.343 | 186.658 ± 0.847 | - | 118.965 ± 13.178 | **68.388 ± 5.473** |
| tensornetwork_permutation_focus_step409_316 | 117.918 ± 0.816 | 188.934 ± 8.907 | - | **74.034 ± 12.260** | 74.663 ± 8.229 |
| tensornetwork_permutation_light_415 | 119.368 ± 3.012 | 153.401 ± 3.652 | - | **76.609 ± 0.451** | 83.243 ± 10.105 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.568 ± 0.077 | 0.573 ± 0.040 | - | 0.398 ± 0.006 | **0.346 ± 0.093** |
| bin_batched_matmul_b32_m64_n64_k64 | **0.081 ± 0.000** | 0.083 ± 0.001 | - | 0.096 ± 0.002 | 0.186 ± 0.020 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.111 ± 0.002 | 0.132 ± 0.003 | - | **0.065 ± 0.013** | 0.124 ± 0.023 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.817 ± 0.078 | 0.129 ± 0.003 | - | **0.059 ± 0.005** | 0.127 ± 0.018 |
| bin_elementwise_mul_2048x2048 | 0.960 ± 0.026 | 1.078 ± 0.077 | - | **0.572 ± 0.012** | 0.758 ± 0.050 |
| bin_matmul_1024 | 4.393 ± 0.181 | 4.428 ± 0.172 | - | 4.769 ± 0.154 | **3.323 ± 0.246** |
| bin_matmul_256 | **0.085 ± 0.000** | **0.085 ± 0.000** | - | 0.093 ± 0.001 | 0.220 ± 0.026 |
| bin_outer_product_4096 | 1.228 ± 0.032 | 1.231 ± 0.012 | - | **0.571 ± 0.016** | 0.624 ± 0.011 |
| gm_queen5_5_3.wcsp | 333.447 ± 0.921 | 502.606 ± 3.892 | - | 186.269 ± 2.672 | **150.855 ± 24.832** |
| lm_batch_likelihood_brackets_4_4d | 7.804 ± 0.082 | 13.668 ± 0.242 | - | 9.643 ± 0.203 | **6.962 ± 0.817** |
| lm_batch_likelihood_sentence_3_12d | 10.340 ± 0.071 | 21.127 ± 0.321 | - | 16.668 ± 0.212 | **7.436 ± 0.197** |
| lm_batch_likelihood_sentence_4_4d | 7.579 ± 0.086 | 14.328 ± 0.117 | - | 9.953 ± 0.381 | **6.607 ± 0.452** |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.008 ± 0.000 | - | 0.027 ± 0.003 | 0.074 ± 0.007 |
| str_matrix_chain_multiplication_100 | **1.771 ± 0.015** | 2.299 ± 0.049 | - | 2.718 ± 0.061 | 7.118 ± 0.438 |
| str_mps_varying_inner_product_200 | **5.844 ± 0.186** | 8.103 ± 0.180 | - | 9.885 ± 2.243 | 15.402 ± 0.313 |
| str_nw_mera_closed_120 | 146.248 ± 0.268 | 154.005 ± 0.468 | - | 120.827 ± 0.655 | **81.957 ± 1.318** |
| str_nw_mera_open_26 | 135.729 ± 0.453 | 187.860 ± 5.512 | - | 111.334 ± 7.536 | **78.184 ± 2.785** |
| tensornetwork_permutation_focus_step409_316 | 117.918 ± 0.816 | 188.934 ± 8.907 | - | **74.034 ± 12.260** | 74.663 ± 8.229 |
| tensornetwork_permutation_light_415 | 119.368 ± 3.012 | 153.401 ± 3.652 | - | **76.609 ± 0.451** | 83.243 ± 10.105 |

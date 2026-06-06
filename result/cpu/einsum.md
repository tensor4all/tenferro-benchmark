# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/cpu/einsum/20260606_234410/run.yaml`
- Timestamp: `20260606_234410`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/cpu/einsum/20260606_234410`.

- tenferro-rs commit: `3bd07ac07e5520e33fa6babd1ad537341fd71d50`

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

- Source table: `data/results/cpu/einsum/20260606_234410/einsum_table_t1_20260606_234410.md`

Logs:

- `data/results/cpu/einsum/20260606_234410/tenferro_trace_t1_20260606_234410.log`
- `data/results/cpu/einsum/20260606_234410/tenferro_eager_t1_20260606_234410.log`
- `data/results/cpu/einsum/20260606_234410/pytorch_cpu_t1_20260606_234410.log`
- `data/results/cpu/einsum/20260606_234410/jax_cpu_t1_20260606_234410.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.335 ± 0.005 | 0.566 ± 0.036 | - | 0.395 ± 0.001 | **0.333 ± 0.014** |
| bin_batched_matmul_b32_m64_n64_k64 | 0.092 ± 0.007 | **0.083 ± 0.001** | - | 0.095 ± 0.002 | 0.199 ± 0.018 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.180 ± 0.003 | 0.200 ± 0.007 | - | **0.117 ± 0.001** | 0.124 ± 0.022 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.180 ± 0.002 | 0.174 ± 0.003 | - | **0.120 ± 0.006** | 0.127 ± 0.018 |
| bin_elementwise_mul_2048x2048 | 1.029 ± 0.074 | 1.073 ± 0.075 | - | 0.804 ± 0.026 | **0.792 ± 0.022** |
| bin_matmul_1024 | 4.406 ± 0.178 | 4.426 ± 0.215 | - | 4.605 ± 0.929 | **3.332 ± 0.135** |
| bin_matmul_256 | **0.084 ± 0.001** | 0.085 ± 0.001 | - | 0.095 ± 0.006 | 0.218 ± 0.023 |
| bin_outer_product_4096 | 1.015 ± 0.037 | 1.023 ± 0.023 | - | 1.485 ± 0.027 | **0.630 ± 0.025** |
| gm_queen5_5_3.wcsp | 1075.949 ± 3.287 | 1388.807 ± 8.185 | - | 851.047 ± 3.637 | **448.871 ± 2.377** |
| lm_batch_likelihood_brackets_4_4d | 6.370 ± 0.071 | 11.669 ± 0.138 | - | 9.035 ± 0.188 | **5.143 ± 0.139** |
| lm_batch_likelihood_sentence_3_12d | 8.444 ± 0.125 | 22.737 ± 0.193 | - | 17.169 ± 0.192 | **6.708 ± 0.154** |
| lm_batch_likelihood_sentence_4_4d | 5.671 ± 0.090 | 13.208 ± 0.217 | - | 9.372 ± 0.195 | **5.442 ± 0.135** |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.008 ± 0.000 | - | 0.024 ± 0.001 | 0.075 ± 0.006 |
| str_matrix_chain_multiplication_100 | **1.848 ± 0.014** | 2.332 ± 0.044 | - | 2.706 ± 0.051 | 6.197 ± 0.191 |
| str_mps_varying_inner_product_200 | **4.236 ± 0.047** | 7.602 ± 0.100 | - | 5.207 ± 0.087 | 13.784 ± 0.077 |
| str_nw_mera_closed_120 | 160.320 ± 0.388 | 178.915 ± 0.430 | - | 155.088 ± 1.286 | **88.177 ± 1.447** |
| str_nw_mera_open_26 | 131.621 ± 0.496 | 176.458 ± 0.404 | - | 118.662 ± 0.238 | **67.594 ± 0.780** |
| tensornetwork_permutation_focus_step409_316 | 119.838 ± 3.568 | 177.650 ± 31.251 | - | 156.525 ± 0.725 | **81.342 ± 5.288** |
| tensornetwork_permutation_light_415 | 121.461 ± 0.760 | 153.079 ± 1.409 | - | 137.348 ± 1.235 | **83.906 ± 4.532** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.335 ± 0.005 | 0.566 ± 0.036 | - | 0.395 ± 0.001 | **0.333 ± 0.014** |
| bin_batched_matmul_b32_m64_n64_k64 | 0.092 ± 0.007 | **0.083 ± 0.001** | - | 0.095 ± 0.002 | 0.199 ± 0.018 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.180 ± 0.003 | 0.200 ± 0.007 | - | **0.117 ± 0.001** | 0.124 ± 0.022 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.180 ± 0.002 | 0.174 ± 0.003 | - | **0.120 ± 0.006** | 0.127 ± 0.018 |
| bin_elementwise_mul_2048x2048 | 1.029 ± 0.074 | 1.073 ± 0.075 | - | 0.804 ± 0.026 | **0.792 ± 0.022** |
| bin_matmul_1024 | 4.406 ± 0.178 | 4.426 ± 0.215 | - | 4.605 ± 0.929 | **3.332 ± 0.135** |
| bin_matmul_256 | **0.084 ± 0.001** | 0.085 ± 0.001 | - | 0.095 ± 0.006 | 0.218 ± 0.023 |
| bin_outer_product_4096 | 1.015 ± 0.037 | 1.023 ± 0.023 | - | 1.485 ± 0.027 | **0.630 ± 0.025** |
| gm_queen5_5_3.wcsp | 307.959 ± 0.571 | 499.512 ± 7.553 | - | 310.689 ± 2.171 | **161.814 ± 18.922** |
| lm_batch_likelihood_brackets_4_4d | 7.362 ± 0.117 | 12.729 ± 0.220 | - | 9.363 ± 0.119 | **5.727 ± 0.822** |
| lm_batch_likelihood_sentence_3_12d | 10.241 ± 0.060 | 23.640 ± 0.471 | - | 17.235 ± 0.141 | **7.358 ± 0.195** |
| lm_batch_likelihood_sentence_4_4d | 7.697 ± 0.092 | 14.099 ± 0.189 | - | 9.147 ± 0.136 | **6.560 ± 0.865** |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.008 ± 0.000 | - | 0.024 ± 0.001 | 0.075 ± 0.006 |
| str_matrix_chain_multiplication_100 | **1.791 ± 0.029** | 2.349 ± 0.044 | - | 2.727 ± 0.032 | 6.763 ± 0.909 |
| str_mps_varying_inner_product_200 | **5.823 ± 0.182** | 8.124 ± 0.132 | - | 6.266 ± 0.157 | 15.579 ± 0.381 |
| str_nw_mera_closed_120 | 145.864 ± 0.261 | 153.580 ± 0.888 | - | 128.902 ± 0.868 | **84.153 ± 2.669** |
| str_nw_mera_open_26 | 135.131 ± 0.912 | 173.850 ± 2.766 | - | 123.979 ± 1.148 | **75.153 ± 1.688** |
| tensornetwork_permutation_focus_step409_316 | 119.838 ± 3.568 | 177.650 ± 31.251 | - | 156.525 ± 0.725 | **81.342 ± 5.288** |
| tensornetwork_permutation_light_415 | 121.461 ± 0.760 | 153.079 ± 1.409 | - | 137.348 ± 1.235 | **83.906 ± 4.532** |

# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260630_071006/run.yaml`
- Timestamp: `20260630_071006`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260630_071006`.

- tenferro-rs commit: `3b4136ca3a3d53cbbdd5096954a470d9407ad25e`

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
- Python platform: `macOS-26.5.1-arm64-arm-64bit`

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

- Source table: `data/results/mac-cpu/cpu/einsum/20260630_071006/einsum_table_t4_20260630_071006.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260630_071006/tenferro_trace_t4_20260630_071006.log`
- `data/results/mac-cpu/cpu/einsum/20260630_071006/tenferro_eager_t4_20260630_071006.log`
- `data/results/mac-cpu/cpu/einsum/20260630_071006/pytorch_cpu_t4_20260630_071006.log`
- `data/results/mac-cpu/cpu/einsum/20260630_071006/jax_cpu_t4_20260630_071006.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.582 ± 0.179 | 0.536 ± 0.043 | 0.399 ± 0.013 | **0.372 ± 0.061** |
| bin_batched_matmul_b32_m64_n64_k64 | **0.066 ± 0.017** | 0.080 ± 0.011 | 0.095 ± 0.002 | 0.164 ± 0.040 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.067 ± 0.025 | 0.066 ± 0.011 | **0.062 ± 0.011** | 0.130 ± 0.016 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.076 ± 0.008 | 0.075 ± 0.016 | **0.066 ± 0.009** | 0.131 ± 0.021 |
| bin_elementwise_mul_2048x2048 | 0.584 ± 0.029 | 0.608 ± 0.044 | **0.581 ± 0.048** | 0.823 ± 0.049 |
| bin_matmul_1024 | 4.664 ± 0.280 | 4.680 ± 0.426 | 5.294 ± 1.156 | **3.346 ± 0.329** |
| bin_matmul_256 | **0.082 ± 0.001** | 0.097 ± 0.007 | 0.093 ± 0.004 | 0.208 ± 0.025 |
| bin_outer_product_4096 | 0.691 ± 0.253 | **0.581 ± 0.014** | 0.606 ± 0.009 | 0.634 ± 0.027 |
| gm_queen5_5_3.wcsp | 608.072 ± 75.024 | 519.882 ± 7.629 | **453.861 ± 7.260** | 455.137 ± 3.151 |
| lm_batch_likelihood_brackets_4_4d | 6.684 ± 0.523 | 10.409 ± 0.262 | 9.786 ± 0.151 | **6.005 ± 0.108** |
| lm_batch_likelihood_sentence_3_12d | 9.591 ± 1.712 | 15.481 ± 0.201 | 21.233 ± 0.898 | **6.774 ± 0.082** |
| lm_batch_likelihood_sentence_4_4d | **5.453 ± 0.252** | 11.436 ± 0.192 | 9.659 ± 0.346 | 5.883 ± 0.138 |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.024 ± 0.006 | 0.024 ± 0.001 | 0.074 ± 0.008 |
| str_matrix_chain_multiplication_100 | **1.772 ± 0.023** | 3.130 ± 0.085 | 2.777 ± 0.071 | 6.503 ± 0.175 |
| str_mps_varying_inner_product_200 | **4.897 ± 0.288** | 9.279 ± 0.497 | 7.761 ± 0.415 | 14.107 ± 0.180 |
| str_nw_mera_closed_120 | 147.859 ± 4.729 | 152.904 ± 2.040 | 144.407 ± 3.049 | **88.536 ± 2.158** |
| str_nw_mera_open_26 | 125.449 ± 3.652 | 143.173 ± 2.796 | 121.227 ± 1.327 | **68.624 ± 2.288** |
| tensornetwork_permutation_focus_step409_316 | 92.911 ± 7.523 | 92.115 ± 4.355 | 80.932 ± 6.935 | **73.640 ± 6.538** |
| tensornetwork_permutation_light_415 | 92.006 ± 1.340 | 85.281 ± 3.342 | 79.327 ± 1.565 | **78.077 ± 4.113** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.582 ± 0.179 | 0.536 ± 0.043 | 0.399 ± 0.013 | **0.372 ± 0.061** |
| bin_batched_matmul_b32_m64_n64_k64 | **0.066 ± 0.017** | 0.080 ± 0.011 | 0.095 ± 0.002 | 0.164 ± 0.040 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.067 ± 0.025 | 0.066 ± 0.011 | **0.062 ± 0.011** | 0.130 ± 0.016 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.076 ± 0.008 | 0.075 ± 0.016 | **0.066 ± 0.009** | 0.131 ± 0.021 |
| bin_elementwise_mul_2048x2048 | 0.584 ± 0.029 | 0.608 ± 0.044 | **0.581 ± 0.048** | 0.823 ± 0.049 |
| bin_matmul_1024 | 4.664 ± 0.280 | 4.680 ± 0.426 | 5.294 ± 1.156 | **3.346 ± 0.329** |
| bin_matmul_256 | **0.082 ± 0.001** | 0.097 ± 0.007 | 0.093 ± 0.004 | 0.208 ± 0.025 |
| bin_outer_product_4096 | 0.691 ± 0.253 | **0.581 ± 0.014** | 0.606 ± 0.009 | 0.634 ± 0.027 |
| gm_queen5_5_3.wcsp | 180.579 ± 10.016 | 185.605 ± 13.149 | 189.422 ± 7.573 | **158.206 ± 4.243** |
| lm_batch_likelihood_brackets_4_4d | 6.299 ± 0.210 | 11.048 ± 0.359 | 9.647 ± 0.271 | **5.953 ± 0.233** |
| lm_batch_likelihood_sentence_3_12d | 9.039 ± 0.260 | 15.904 ± 0.223 | 19.322 ± 0.645 | **7.301 ± 0.112** |
| lm_batch_likelihood_sentence_4_4d | 6.504 ± 0.047 | 11.431 ± 0.210 | 9.813 ± 0.257 | **5.724 ± 0.562** |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.024 ± 0.006 | 0.024 ± 0.001 | 0.074 ± 0.008 |
| str_matrix_chain_multiplication_100 | **1.836 ± 0.040** | 3.125 ± 0.113 | 2.759 ± 0.055 | 6.779 ± 0.267 |
| str_mps_varying_inner_product_200 | **6.779 ± 0.089** | 9.496 ± 0.170 | 7.465 ± 0.411 | 16.322 ± 0.394 |
| str_nw_mera_closed_120 | 128.245 ± 0.528 | 135.953 ± 1.751 | 131.339 ± 3.266 | **85.908 ± 2.091** |
| str_nw_mera_open_26 | 124.427 ± 1.337 | 144.057 ± 1.620 | 122.419 ± 2.985 | **75.817 ± 1.316** |
| tensornetwork_permutation_focus_step409_316 | 92.911 ± 7.523 | 92.115 ± 4.355 | 80.932 ± 6.935 | **73.640 ± 6.538** |
| tensornetwork_permutation_light_415 | 92.006 ± 1.340 | 85.281 ± 3.342 | 79.327 ± 1.565 | **78.077 ± 4.113** |

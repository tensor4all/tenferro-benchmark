# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/cpu/einsum/20260605_142400/run.yaml`
- Timestamp: `20260605_142400`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/cpu/einsum/20260605_142400`.

- tenferro-rs commit: `3c3cc0b1079ca84e6b700ce93c77606e2110f946`

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

- Source table: `data/results/cpu/einsum/20260605_142400/einsum_table_t4_20260605_142400.md`

Logs:

- `data/results/cpu/einsum/20260605_142400/tenferro_trace_t4_20260605_142400.log`
- `data/results/cpu/einsum/20260605_142400/tenferro_eager_t4_20260605_142400.log`
- `data/results/cpu/einsum/20260605_142400/pytorch_cpu_t4_20260605_142400.log`
- `data/results/cpu/einsum/20260605_142400/jax_cpu_t4_20260605_142400.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | **0.349 ± 0.059** | 0.366 ± 0.104 | - | 0.395 ± 0.005 | 0.393 ± 0.048 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.051 ± 0.003** | 0.056 ± 0.004 | - | 0.097 ± 0.004 | 0.173 ± 0.036 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.067 ± 0.001 | 0.079 ± 0.003 | - | **0.063 ± 0.009** | 0.127 ± 0.029 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.585 ± 0.057 | 0.077 ± 0.001 | - | **0.066 ± 0.005** | 0.130 ± 0.027 |
| bin_elementwise_mul_2048x2048 | 0.827 ± 0.012 | 0.917 ± 0.137 | - | **0.586 ± 0.058** | 0.871 ± 0.111 |
| bin_matmul_1024 | 5.147 ± 0.914 | 5.015 ± 0.510 | - | 4.678 ± 0.300 | **3.452 ± 0.197** |
| bin_matmul_256 | **0.082 ± 0.003** | 0.083 ± 0.005 | - | 0.094 ± 0.004 | 0.225 ± 0.025 |
| bin_outer_product_4096 | 1.216 ± 0.109 | 1.208 ± 0.040 | - | **0.594 ± 0.019** | 0.650 ± 0.045 |
| gm_queen5_5_3.wcsp | 1804.963 ± 21.216 | 3885.109 ± 149.558 | - | **421.623 ± 9.490** | 494.949 ± 60.685 |
| lm_batch_likelihood_brackets_4_4d | 7.981 ± 0.214 | 14.336 ± 0.413 | - | 8.929 ± 2.764 | **6.033 ± 0.157** |
| lm_batch_likelihood_sentence_3_12d | 9.658 ± 0.335 | 25.063 ± 0.443 | - | 18.489 ± 0.218 | **6.895 ± 0.252** |
| lm_batch_likelihood_sentence_4_4d | 6.901 ± 0.207 | 15.749 ± 0.844 | - | 8.049 ± 0.241 | **6.021 ± 0.136** |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.008 ± 0.000 | - | 0.024 ± 0.001 | 0.076 ± 0.008 |
| str_matrix_chain_multiplication_100 | **1.797 ± 0.028** | 2.375 ± 0.080 | - | 2.774 ± 0.090 | 6.602 ± 0.221 |
| str_mps_varying_inner_product_200 | **4.328 ± 0.091** | 7.617 ± 0.278 | - | 5.555 ± 0.174 | 14.689 ± 0.434 |
| str_nw_mera_closed_120 | 166.994 ± 2.424 | 188.582 ± 6.931 | - | 141.391 ± 1.745 | **90.816 ± 1.756** |
| str_nw_mera_open_26 | 180.389 ± 1.053 | 196.758 ± 5.432 | - | 112.684 ± 1.271 | **69.225 ± 1.304** |
| tensornetwork_permutation_focus_step409_316 | 124.118 ± 4.041 | 193.126 ± 6.883 | - | **73.215 ± 2.308** | 75.488 ± 3.874 |
| tensornetwork_permutation_light_415 | 124.429 ± 1.166 | 157.853 ± 2.605 | - | **68.549 ± 1.123** | 79.545 ± 3.472 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | **0.300 ± 0.003** | 0.318 ± 0.010 | - | 0.399 ± 0.011 | 0.357 ± 0.047 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.051 ± 0.001** | 0.052 ± 0.002 | - | 0.096 ± 0.001 | 0.165 ± 0.024 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.066 ± 0.000** | 0.079 ± 0.001 | - | 0.072 ± 0.016 | 0.123 ± 0.015 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.585 ± 0.017 | 0.079 ± 0.002 | - | **0.065 ± 0.005** | 0.138 ± 0.042 |
| bin_elementwise_mul_2048x2048 | 0.819 ± 0.025 | 0.862 ± 0.033 | - | **0.568 ± 0.019** | 0.819 ± 0.073 |
| bin_matmul_1024 | 6.281 ± 1.325 | 6.374 ± 4.226 | - | 4.930 ± 0.533 | **3.825 ± 0.195** |
| bin_matmul_256 | 0.084 ± 0.009 | **0.083 ± 0.007** | - | 0.095 ± 0.003 | 0.222 ± 0.026 |
| bin_outer_product_4096 | 1.200 ± 0.045 | 1.218 ± 0.095 | - | **0.581 ± 0.037** | 0.639 ± 0.032 |
| gm_queen5_5_3.wcsp | 484.808 ± 3.024 | 518.924 ± 6.708 | - | 187.876 ± 8.510 | **166.122 ± 8.189** |
| lm_batch_likelihood_brackets_4_4d | 8.132 ± 0.217 | 14.777 ± 0.408 | - | 11.040 ± 0.821 | **6.354 ± 0.744** |
| lm_batch_likelihood_sentence_3_12d | 11.470 ± 0.424 | 23.911 ± 0.472 | - | 19.961 ± 0.980 | **7.467 ± 0.148** |
| lm_batch_likelihood_sentence_4_4d | 7.868 ± 0.115 | 15.899 ± 1.194 | - | 11.060 ± 0.441 | **5.584 ± 0.296** |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.007 ± 0.000 | - | 0.024 ± 0.001 | 0.078 ± 0.019 |
| str_matrix_chain_multiplication_100 | **1.808 ± 0.032** | 2.326 ± 0.033 | - | 2.813 ± 0.045 | 7.413 ± 0.898 |
| str_mps_varying_inner_product_200 | **6.055 ± 0.176** | 7.852 ± 0.091 | - | 10.316 ± 0.615 | 15.381 ± 0.153 |
| str_nw_mera_closed_120 | 148.967 ± 0.840 | 159.137 ± 1.513 | - | 130.292 ± 3.696 | **89.473 ± 2.553** |
| str_nw_mera_open_26 | 181.962 ± 5.722 | 197.110 ± 4.471 | - | 115.502 ± 2.338 | **79.026 ± 1.950** |
| tensornetwork_permutation_focus_step409_316 | 124.248 ± 1.618 | 193.570 ± 11.692 | - | 77.318 ± 8.635 | **74.456 ± 4.822** |
| tensornetwork_permutation_light_415 | 125.306 ± 1.588 | 158.605 ± 3.775 | - | **70.232 ± 2.495** | 81.180 ± 7.893 |

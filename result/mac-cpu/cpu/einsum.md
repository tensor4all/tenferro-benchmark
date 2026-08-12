# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260812_171837/run.yaml`
- Timestamp: `20260812_171837`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260812_171837`.

- tenferro-rs commit: `3ce8fbed00c2aae40ba55ca38f04e32527c2116e`

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

- Source table: `data/results/mac-cpu/cpu/einsum/20260812_171837/einsum_table_t4_20260812_171837.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260812_171837/tenferro_trace_t4_20260812_171837.log`
- `data/results/mac-cpu/cpu/einsum/20260812_171837/tenferro_eager_t4_20260812_171837.log`
- `data/results/mac-cpu/cpu/einsum/20260812_171837/strided_faer_t4_20260812_171837.log`
- `data/results/mac-cpu/cpu/einsum/20260812_171837/pytorch_cpu_t4_20260812_171837.log`
- `data/results/mac-cpu/cpu/einsum/20260812_171837/jax_cpu_t4_20260812_171837.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | strided-rs faer (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.658 ± 0.011 | 0.674 ± 0.078 | - | 0.396 ± 0.005 | **0.357 ± 0.038** |
| bin_batched_matmul_b32_m64_n64_k64 | **0.092 ± 0.000** | 0.094 ± 0.001 | 0.959 ± 0.190 | 0.094 ± 0.001 | 0.173 ± 0.024 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.084 ± 0.016 | 0.084 ± 0.006 | - | **0.060 ± 0.005** | 0.134 ± 0.021 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.076 ± 0.019 | 0.083 ± 0.013 | - | **0.058 ± 0.001** | 0.164 ± 0.017 |
| bin_elementwise_mul_2048x2048 | 1.992 ± 0.403 | 0.632 ± 0.024 | 0.608 ± 0.034 | **0.581 ± 0.012** | 0.836 ± 0.052 |
| bin_matmul_1024 | 4.381 ± 0.034 | 4.609 ± 0.366 | - | 4.486 ± 0.247 | **3.429 ± 0.389** |
| bin_matmul_256 | **0.080 ± 0.001** | 0.083 ± 0.001 | 0.213 ± 0.007 | 0.092 ± 0.001 | 0.218 ± 0.021 |
| bin_outer_product_4096 | 0.561 ± 0.010 | **0.559 ± 0.007** | 1.084 ± 0.025 | 0.572 ± 0.005 | 0.630 ± 0.020 |
| gm_queen5_5_3.wcsp | 768.970 ± 48.016 | 521.360 ± 18.842 | 1642.166 ± 8.053 | 462.155 ± 5.249 | **452.676 ± 2.762** |
| lm_batch_likelihood_brackets_4_4d | **5.661 ± 0.105** | 10.061 ± 0.135 | 11.122 ± 0.254 | 8.560 ± 1.231 | 5.927 ± 0.355 |
| lm_batch_likelihood_sentence_3_12d | 7.606 ± 0.094 | 16.084 ± 1.165 | 22.860 ± 0.595 | 19.876 ± 4.772 | **6.935 ± 0.296** |
| lm_batch_likelihood_sentence_4_4d | **5.407 ± 0.046** | 11.867 ± 0.185 | 12.365 ± 0.574 | 9.056 ± 0.744 | 6.437 ± 0.962 |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.007 ± 0.000 | - | 0.026 ± 0.005 | 0.072 ± 0.004 |
| str_matrix_chain_multiplication_100 | **1.765 ± 0.034** | 2.279 ± 0.037 | 4.325 ± 0.127 | 2.623 ± 0.058 | 6.517 ± 1.067 |
| str_mps_varying_inner_product_200 | **4.758 ± 0.057** | 9.170 ± 0.104 | 8.817 ± 0.145 | 6.126 ± 0.476 | 13.701 ± 0.135 |
| str_nw_mera_closed_120 | 137.890 ± 1.191 | 154.489 ± 1.156 | 436.187 ± 12.731 | 142.255 ± 3.823 | **89.726 ± 1.618** |
| str_nw_mera_open_26 | 120.296 ± 0.717 | 146.655 ± 1.029 | 219.941 ± 1.917 | 123.814 ± 5.638 | **75.234 ± 1.343** |
| tensornetwork_permutation_focus_step409_316 | 87.355 ± 2.367 | 130.909 ± 0.919 | 79.788 ± 8.532 | **74.423 ± 8.265** | 79.769 ± 10.716 |
| tensornetwork_permutation_light_415 | **63.492 ± 0.782** | 119.446 ± 0.570 | 85.422 ± 9.931 | 78.381 ± 0.486 | 87.875 ± 12.197 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | strided-rs faer (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.658 ± 0.011 | 0.674 ± 0.078 | - | 0.396 ± 0.005 | **0.357 ± 0.038** |
| bin_batched_matmul_b32_m64_n64_k64 | **0.092 ± 0.000** | 0.094 ± 0.001 | 0.574 ± 0.009 | 0.094 ± 0.001 | 0.173 ± 0.024 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.084 ± 0.016 | 0.084 ± 0.006 | - | **0.060 ± 0.005** | 0.134 ± 0.021 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.076 ± 0.019 | 0.083 ± 0.013 | - | **0.058 ± 0.001** | 0.164 ± 0.017 |
| bin_elementwise_mul_2048x2048 | 1.992 ± 0.403 | 0.632 ± 0.024 | **0.544 ± 0.028** | 0.581 ± 0.012 | 0.836 ± 0.052 |
| bin_matmul_1024 | 4.381 ± 0.034 | 4.609 ± 0.366 | - | 4.486 ± 0.247 | **3.429 ± 0.389** |
| bin_matmul_256 | **0.080 ± 0.001** | 0.083 ± 0.001 | 0.172 ± 0.009 | 0.092 ± 0.001 | 0.218 ± 0.021 |
| bin_outer_product_4096 | 0.561 ± 0.010 | **0.559 ± 0.007** | 1.068 ± 0.010 | 0.572 ± 0.005 | 0.630 ± 0.020 |
| gm_queen5_5_3.wcsp | 203.123 ± 15.834 | 293.992 ± 5.070 | 436.580 ± 5.578 | 194.082 ± 4.676 | **176.884 ± 25.222** |
| lm_batch_likelihood_brackets_4_4d | **6.217 ± 0.027** | 11.000 ± 0.103 | 9.992 ± 0.120 | 10.613 ± 1.215 | 7.344 ± 0.910 |
| lm_batch_likelihood_sentence_3_12d | 8.522 ± 0.144 | 15.966 ± 0.286 | 23.041 ± 0.284 | 22.210 ± 3.433 | **7.367 ± 0.302** |
| lm_batch_likelihood_sentence_4_4d | **6.456 ± 0.038** | 11.539 ± 0.123 | 11.515 ± 0.137 | 11.041 ± 1.263 | 6.873 ± 0.429 |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.007 ± 0.000 | - | 0.026 ± 0.005 | 0.072 ± 0.004 |
| str_matrix_chain_multiplication_100 | **1.734 ± 0.016** | 2.263 ± 0.089 | 4.298 ± 0.075 | 2.690 ± 0.038 | 7.340 ± 0.522 |
| str_mps_varying_inner_product_200 | **6.386 ± 0.138** | 10.763 ± 0.198 | 8.768 ± 0.176 | 9.291 ± 0.529 | 15.608 ± 0.336 |
| str_nw_mera_closed_120 | 125.803 ± 0.459 | 136.042 ± 0.300 | 305.160 ± 13.779 | 139.533 ± 4.694 | **83.485 ± 2.847** |
| str_nw_mera_open_26 | 122.122 ± 0.881 | 146.243 ± 1.246 | 225.404 ± 2.081 | 127.989 ± 5.585 | **78.682 ± 2.229** |
| tensornetwork_permutation_focus_step409_316 | 87.355 ± 2.367 | 130.909 ± 0.919 | 79.289 ± 0.370 | **74.423 ± 8.265** | 79.769 ± 10.716 |
| tensornetwork_permutation_light_415 | **63.492 ± 0.782** | 119.446 ± 0.570 | 80.337 ± 0.669 | 78.381 ± 0.486 | 87.875 ± 12.197 |

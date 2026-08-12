# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260812_150722/run.yaml`
- Timestamp: `20260812_150722`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260812_150722`.

- tenferro-rs commit: `7f91eb7dadca2227a3043dc9e97997a2d287237c`

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

- Source table: `data/results/mac-cpu/cpu/einsum/20260812_150722/einsum_table_t4_20260812_150722.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260812_150722/tenferro_trace_t4_20260812_150722.log`
- `data/results/mac-cpu/cpu/einsum/20260812_150722/tenferro_eager_t4_20260812_150722.log`
- `data/results/mac-cpu/cpu/einsum/20260812_150722/strided_faer_t4_20260812_150722.log`
- `data/results/mac-cpu/cpu/einsum/20260812_150722/pytorch_cpu_t4_20260812_150722.log`
- `data/results/mac-cpu/cpu/einsum/20260812_150722/jax_cpu_t4_20260812_150722.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | strided-rs faer (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.522 ± 1.212 | 0.386 ± 0.028 | - | 0.401 ± 0.003 | **0.342 ± 0.027** |
| bin_batched_matmul_b32_m64_n64_k64 | 0.075 ± 0.002 | **0.062 ± 0.001** | 1.088 ± 0.177 | 0.094 ± 0.001 | 0.156 ± 0.035 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.081 ± 0.020 | 0.071 ± 0.026 | - | **0.062 ± 0.005** | 0.130 ± 0.021 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.077 ± 0.017 | 0.066 ± 0.017 | - | **0.059 ± 0.005** | 0.140 ± 0.031 |
| bin_elementwise_mul_2048x2048 | 1.684 ± 0.281 | 0.651 ± 0.053 | 0.618 ± 0.043 | **0.601 ± 0.020** | 0.812 ± 0.051 |
| bin_matmul_1024 | 5.778 ± 0.903 | 7.857 ± 2.416 | - | 4.703 ± 0.112 | **3.316 ± 0.235** |
| bin_matmul_256 | **0.083 ± 0.003** | 0.086 ± 0.007 | 0.222 ± 0.012 | 0.092 ± 0.001 | 0.219 ± 0.021 |
| bin_outer_product_4096 | 0.788 ± 0.251 | **0.588 ± 0.015** | 1.125 ± 0.049 | 0.589 ± 0.013 | 0.651 ± 0.031 |
| gm_queen5_5_3.wcsp | 738.154 ± 113.807 | 628.483 ± 51.794 | 1647.779 ± 26.742 | 472.112 ± 14.812 | **469.556 ± 70.316** |
| lm_batch_likelihood_brackets_4_4d | **5.497 ± 0.049** | 10.614 ± 0.745 | 12.325 ± 0.794 | 9.898 ± 1.725 | 5.927 ± 0.151 |
| lm_batch_likelihood_sentence_3_12d | 7.875 ± 0.219 | 15.617 ± 1.057 | 26.013 ± 1.496 | 20.300 ± 5.950 | **6.747 ± 0.161** |
| lm_batch_likelihood_sentence_4_4d | **5.255 ± 0.070** | 11.595 ± 0.226 | 13.623 ± 0.429 | 8.948 ± 1.324 | 5.821 ± 0.221 |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.008 ± 0.000 | - | 0.024 ± 0.002 | 0.073 ± 0.004 |
| str_matrix_chain_multiplication_100 | **1.824 ± 0.039** | 2.267 ± 0.085 | 5.061 ± 3.387 | 2.719 ± 0.058 | 6.301 ± 0.531 |
| str_mps_varying_inner_product_200 | **4.785 ± 0.064** | 9.039 ± 0.229 | 9.171 ± 0.972 | 5.719 ± 0.381 | 14.157 ± 0.249 |
| str_nw_mera_closed_120 | 143.044 ± 4.296 | 155.772 ± 0.926 | 438.286 ± 9.007 | 154.144 ± 8.490 | **90.208 ± 1.268** |
| str_nw_mera_open_26 | 124.748 ± 3.656 | 145.910 ± 0.597 | 222.324 ± 1.978 | 127.237 ± 12.419 | **74.620 ± 6.167** |
| tensornetwork_permutation_focus_step409_316 | 74.583 ± 24.681 | 92.420 ± 4.646 | 79.067 ± 0.969 | **73.960 ± 1.446** | 78.342 ± 8.516 |
| tensornetwork_permutation_light_415 | **64.961 ± 12.348** | 102.259 ± 36.567 | 80.274 ± 4.192 | 77.914 ± 1.055 | 89.004 ± 6.682 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | strided-rs faer (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.522 ± 1.212 | 0.386 ± 0.028 | - | 0.401 ± 0.003 | **0.342 ± 0.027** |
| bin_batched_matmul_b32_m64_n64_k64 | 0.075 ± 0.002 | **0.062 ± 0.001** | 0.569 ± 0.008 | 0.094 ± 0.001 | 0.156 ± 0.035 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.081 ± 0.020 | 0.071 ± 0.026 | - | **0.062 ± 0.005** | 0.130 ± 0.021 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.077 ± 0.017 | 0.066 ± 0.017 | - | **0.059 ± 0.005** | 0.140 ± 0.031 |
| bin_elementwise_mul_2048x2048 | 1.684 ± 0.281 | 0.651 ± 0.053 | **0.550 ± 0.019** | 0.601 ± 0.020 | 0.812 ± 0.051 |
| bin_matmul_1024 | 5.778 ± 0.903 | 7.857 ± 2.416 | - | 4.703 ± 0.112 | **3.316 ± 0.235** |
| bin_matmul_256 | **0.083 ± 0.003** | 0.086 ± 0.007 | 0.172 ± 0.006 | 0.092 ± 0.001 | 0.219 ± 0.021 |
| bin_outer_product_4096 | 0.788 ± 0.251 | **0.588 ± 0.015** | 1.069 ± 0.042 | 0.589 ± 0.013 | 0.651 ± 0.031 |
| gm_queen5_5_3.wcsp | 230.302 ± 20.048 | 184.720 ± 2.824 | 436.547 ± 3.983 | 189.308 ± 6.457 | **178.200 ± 11.942** |
| lm_batch_likelihood_brackets_4_4d | **6.185 ± 0.152** | 11.675 ± 0.235 | 10.148 ± 0.099 | 10.575 ± 0.985 | 6.531 ± 0.371 |
| lm_batch_likelihood_sentence_3_12d | 8.532 ± 0.046 | 15.789 ± 0.122 | 23.342 ± 0.252 | 27.006 ± 10.493 | **7.544 ± 0.351** |
| lm_batch_likelihood_sentence_4_4d | 6.296 ± 0.050 | 11.369 ± 0.149 | 11.562 ± 0.057 | 10.510 ± 1.866 | **5.682 ± 0.285** |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.008 ± 0.000 | - | 0.024 ± 0.002 | 0.073 ± 0.004 |
| str_matrix_chain_multiplication_100 | **1.804 ± 0.025** | 2.312 ± 0.049 | 4.333 ± 0.116 | 2.678 ± 0.059 | 6.769 ± 0.539 |
| str_mps_varying_inner_product_200 | **6.656 ± 0.299** | 10.556 ± 0.121 | 8.894 ± 0.183 | 11.804 ± 0.945 | 16.386 ± 0.603 |
| str_nw_mera_closed_120 | 130.217 ± 4.229 | 136.854 ± 3.479 | 310.407 ± 15.366 | 145.610 ± 10.946 | **91.529 ± 17.241** |
| str_nw_mera_open_26 | 127.074 ± 7.425 | 145.154 ± 1.968 | 228.016 ± 1.936 | 133.434 ± 6.206 | **85.835 ± 16.416** |
| tensornetwork_permutation_focus_step409_316 | 74.583 ± 24.681 | 92.420 ± 4.646 | 78.825 ± 0.519 | **73.960 ± 1.446** | 78.342 ± 8.516 |
| tensornetwork_permutation_light_415 | **64.961 ± 12.348** | 102.259 ± 36.567 | 79.432 ± 0.726 | 77.914 ± 1.055 | 89.004 ± 6.682 |

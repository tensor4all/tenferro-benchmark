# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/cpu/einsum/20260606_101358/run.yaml`
- Timestamp: `20260606_101358`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/cpu/einsum/20260606_101358`.

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

- Source table: `data/results/cpu/einsum/20260606_101358/einsum_table_t1_20260606_101358.md`

Logs:

- `data/results/cpu/einsum/20260606_101358/tenferro_trace_t1_20260606_101358.log`
- `data/results/cpu/einsum/20260606_101358/tenferro_eager_t1_20260606_101358.log`
- `data/results/cpu/einsum/20260606_101358/pytorch_cpu_t1_20260606_101358.log`
- `data/results/cpu/einsum/20260606_101358/jax_cpu_t1_20260606_101358.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.586 ± 0.092 | 0.596 ± 0.069 | - | 0.393 ± 0.002 | **0.360 ± 0.050** |
| bin_batched_matmul_b32_m64_n64_k64 | **0.081 ± 0.000** | 0.084 ± 0.002 | - | 0.098 ± 0.010 | 0.188 ± 0.044 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.110 ± 0.000** | 0.135 ± 0.017 | - | 0.119 ± 0.005 | 0.135 ± 0.024 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.846 ± 0.118 | **0.115 ± 0.011** | - | 0.125 ± 0.003 | 0.128 ± 0.019 |
| bin_elementwise_mul_2048x2048 | 0.995 ± 0.054 | 1.085 ± 0.086 | - | 0.864 ± 0.051 | **0.849 ± 0.056** |
| bin_matmul_1024 | 4.682 ± 0.333 | 4.671 ± 0.347 | - | 5.819 ± 0.691 | **3.539 ± 0.176** |
| bin_matmul_256 | **0.085 ± 0.006** | 0.087 ± 0.003 | - | 0.094 ± 0.002 | 0.228 ± 0.025 |
| bin_outer_product_4096 | 1.255 ± 0.062 | 1.268 ± 0.102 | - | 1.531 ± 0.051 | **0.649 ± 0.041** |
| gm_queen5_5_3.wcsp | 1050.763 ± 57.937 | 1535.995 ± 19.407 | - | 889.813 ± 7.428 | **473.682 ± 20.219** |
| lm_batch_likelihood_brackets_4_4d | 7.906 ± 0.360 | 13.534 ± 0.423 | - | 10.227 ± 0.230 | **5.419 ± 0.186** |
| lm_batch_likelihood_sentence_3_12d | 9.504 ± 0.180 | 22.217 ± 0.169 | - | 20.034 ± 0.415 | **6.741 ± 0.180** |
| lm_batch_likelihood_sentence_4_4d | 6.956 ± 0.220 | 14.413 ± 0.127 | - | 10.575 ± 0.659 | **5.988 ± 0.190** |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.008 ± 0.000 | - | 0.023 ± 0.001 | 0.075 ± 0.006 |
| str_matrix_chain_multiplication_100 | **1.846 ± 0.023** | 2.366 ± 0.068 | - | 2.724 ± 0.058 | 6.912 ± 0.133 |
| str_mps_varying_inner_product_200 | **4.413 ± 0.089** | 7.638 ± 0.059 | - | 5.316 ± 0.106 | 14.270 ± 0.150 |
| str_nw_mera_closed_120 | 165.153 ± 3.343 | 184.768 ± 2.793 | - | 159.068 ± 8.822 | **89.971 ± 1.862** |
| str_nw_mera_open_26 | 134.696 ± 6.556 | 191.484 ± 2.596 | - | 120.423 ± 0.896 | **72.661 ± 2.475** |
| tensornetwork_permutation_focus_step409_316 | 120.385 ± 2.945 | 258.311 ± 22.154 | - | 157.726 ± 1.331 | **78.734 ± 5.202** |
| tensornetwork_permutation_light_415 | 123.917 ± 4.656 | 157.701 ± 2.728 | - | 139.625 ± 1.583 | **84.829 ± 4.234** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.586 ± 0.092 | 0.596 ± 0.069 | - | 0.393 ± 0.002 | **0.360 ± 0.050** |
| bin_batched_matmul_b32_m64_n64_k64 | **0.081 ± 0.000** | 0.084 ± 0.002 | - | 0.098 ± 0.010 | 0.188 ± 0.044 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.110 ± 0.000** | 0.135 ± 0.017 | - | 0.119 ± 0.005 | 0.135 ± 0.024 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.846 ± 0.118 | **0.115 ± 0.011** | - | 0.125 ± 0.003 | 0.128 ± 0.019 |
| bin_elementwise_mul_2048x2048 | 0.995 ± 0.054 | 1.085 ± 0.086 | - | 0.864 ± 0.051 | **0.849 ± 0.056** |
| bin_matmul_1024 | 4.682 ± 0.333 | 4.671 ± 0.347 | - | 5.819 ± 0.691 | **3.539 ± 0.176** |
| bin_matmul_256 | **0.085 ± 0.006** | 0.087 ± 0.003 | - | 0.094 ± 0.002 | 0.228 ± 0.025 |
| bin_outer_product_4096 | 1.255 ± 0.062 | 1.268 ± 0.102 | - | 1.531 ± 0.051 | **0.649 ± 0.041** |
| gm_queen5_5_3.wcsp | 342.644 ± 7.653 | 515.513 ± 6.872 | - | 324.506 ± 10.395 | **158.219 ± 7.314** |
| lm_batch_likelihood_brackets_4_4d | 7.836 ± 0.287 | 15.266 ± 0.671 | - | 9.895 ± 0.225 | **5.640 ± 0.420** |
| lm_batch_likelihood_sentence_3_12d | 11.048 ± 0.281 | 27.728 ± 1.954 | - | 18.306 ± 0.212 | **7.633 ± 0.242** |
| lm_batch_likelihood_sentence_4_4d | 7.806 ± 0.156 | 16.318 ± 1.113 | - | 9.620 ± 0.126 | **5.500 ± 0.269** |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.008 ± 0.000 | - | 0.023 ± 0.001 | 0.075 ± 0.006 |
| str_matrix_chain_multiplication_100 | **1.831 ± 0.018** | 2.297 ± 0.093 | - | 2.778 ± 0.077 | 6.425 ± 0.540 |
| str_mps_varying_inner_product_200 | **6.253 ± 0.424** | 8.134 ± 0.140 | - | 6.503 ± 0.373 | 15.141 ± 0.213 |
| str_nw_mera_closed_120 | 150.222 ± 2.418 | 160.920 ± 4.860 | - | 131.632 ± 3.400 | **85.939 ± 0.845** |
| str_nw_mera_open_26 | 139.169 ± 4.876 | 198.355 ± 14.921 | - | 129.057 ± 2.370 | **76.647 ± 1.726** |
| tensornetwork_permutation_focus_step409_316 | 120.385 ± 2.945 | 258.311 ± 22.154 | - | 157.726 ± 1.331 | **78.734 ± 5.202** |
| tensornetwork_permutation_light_415 | 123.917 ± 4.656 | 157.701 ± 2.728 | - | 139.625 ± 1.583 | **84.829 ± 4.234** |

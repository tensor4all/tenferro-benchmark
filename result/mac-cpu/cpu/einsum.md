# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260614_142926/run.yaml`
- Timestamp: `20260614_142926`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260614_142926`.

- tenferro-rs commit: `db1990549801351308b631aef1bbca292d11a457`

## CPU Information

- Model: `Apple M4`
- Vendor: `Apple`
- Logical CPUs: `10`
- Physical CPUs: `10`
- Sockets: `1`
- Cores per socket: `10`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Performance: 4 physical / 4 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 4 CPUs/L2); Efficiency: 6 physical / 6 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

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

- Source table: `data/results/mac-cpu/cpu/einsum/20260614_142926/einsum_table_t4_20260614_142926.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260614_142926/tenferro_trace_t4_20260614_142926.log`
- `data/results/mac-cpu/cpu/einsum/20260614_142926/tenferro_eager_t4_20260614_142926.log`
- `data/results/mac-cpu/cpu/einsum/20260614_142926/pytorch_cpu_t4_20260614_142926.log`
- `data/results/mac-cpu/cpu/einsum/20260614_142926/jax_cpu_t4_20260614_142926.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.651 ± 0.020 | 0.674 ± 0.175 | **0.432 ± 0.049** | 0.659 ± 0.046 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.087 ± 0.000** | 0.099 ± 0.012 | 0.105 ± 0.003 | 0.205 ± 0.024 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.112 ± 0.014 | 0.144 ± 0.017 | **0.098 ± 0.007** | 0.140 ± 0.025 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.123 ± 0.032 | 0.133 ± 0.015 | **0.103 ± 0.005** | 0.152 ± 0.037 |
| bin_elementwise_mul_2048x2048 | **1.083 ± 0.033** | 1.122 ± 0.050 | 1.138 ± 0.122 | 1.542 ± 0.023 |
| bin_matmul_1024 | 4.605 ± 0.175 | 4.668 ± 0.134 | **4.414 ± 0.149** | 8.089 ± 0.334 |
| bin_matmul_256 | **0.082 ± 0.000** | 0.083 ± 0.002 | 0.091 ± 0.005 | 0.321 ± 0.067 |
| bin_outer_product_4096 | 1.839 ± 0.033 | 1.832 ± 0.024 | **1.780 ± 0.144** | 1.922 ± 0.081 |
| gm_queen5_5_3.wcsp | 616.957 ± 33.076 | 721.576 ± 30.561 | **606.275 ± 31.296** | 717.233 ± 36.130 |
| lm_batch_likelihood_brackets_4_4d | **6.743 ± 0.252** | 13.751 ± 0.545 | 9.264 ± 0.333 | 7.442 ± 0.208 |
| lm_batch_likelihood_sentence_3_12d | 12.060 ± 0.547 | 23.724 ± 0.378 | 26.334 ± 0.428 | **11.940 ± 0.436** |
| lm_batch_likelihood_sentence_4_4d | **6.670 ± 0.217** | 14.688 ± 0.472 | 9.284 ± 0.345 | 7.910 ± 0.507 |
| nary_matmul_chain_64 | **0.006 ± 0.000** | 0.020 ± 0.004 | 0.031 ± 0.001 | 0.096 ± 0.007 |
| str_matrix_chain_multiplication_100 | **2.142 ± 0.080** | 3.459 ± 0.154 | 3.239 ± 0.146 | 7.152 ± 0.179 |
| str_mps_varying_inner_product_200 | **5.370 ± 0.328** | 10.258 ± 0.366 | 6.708 ± 4.861 | 15.226 ± 0.076 |
| str_nw_mera_closed_120 | 143.522 ± 0.470 | 165.251 ± 0.499 | **143.484 ± 1.387** | 209.243 ± 4.507 |
| str_nw_mera_open_26 | **116.645 ± 0.630** | 158.205 ± 5.602 | 119.225 ± 2.861 | 162.753 ± 5.543 |
| tensornetwork_permutation_focus_step409_316 | 82.179 ± 3.523 | 101.308 ± 6.138 | **78.719 ± 0.766** | 134.287 ± 3.941 |
| tensornetwork_permutation_light_415 | 86.004 ± 1.680 | 97.253 ± 3.017 | **75.361 ± 2.227** | 139.150 ± 3.812 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.651 ± 0.020 | 0.674 ± 0.175 | **0.432 ± 0.049** | 0.659 ± 0.046 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.087 ± 0.000** | 0.099 ± 0.012 | 0.105 ± 0.003 | 0.205 ± 0.024 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.112 ± 0.014 | 0.144 ± 0.017 | **0.098 ± 0.007** | 0.140 ± 0.025 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.123 ± 0.032 | 0.133 ± 0.015 | **0.103 ± 0.005** | 0.152 ± 0.037 |
| bin_elementwise_mul_2048x2048 | **1.083 ± 0.033** | 1.122 ± 0.050 | 1.138 ± 0.122 | 1.542 ± 0.023 |
| bin_matmul_1024 | 4.605 ± 0.175 | 4.668 ± 0.134 | **4.414 ± 0.149** | 8.089 ± 0.334 |
| bin_matmul_256 | **0.082 ± 0.000** | 0.083 ± 0.002 | 0.091 ± 0.005 | 0.321 ± 0.067 |
| bin_outer_product_4096 | 1.839 ± 0.033 | 1.832 ± 0.024 | **1.780 ± 0.144** | 1.922 ± 0.081 |
| gm_queen5_5_3.wcsp | **193.175 ± 20.923** | 222.628 ± 8.590 | 229.440 ± 2.159 | 253.885 ± 2.686 |
| lm_batch_likelihood_brackets_4_4d | 7.708 ± 0.155 | 13.539 ± 0.436 | 10.655 ± 0.897 | **7.149 ± 0.081** |
| lm_batch_likelihood_sentence_3_12d | **12.406 ± 0.257** | 23.147 ± 0.240 | 25.226 ± 0.334 | 13.047 ± 0.281 |
| lm_batch_likelihood_sentence_4_4d | 7.978 ± 0.242 | 14.444 ± 0.400 | 9.727 ± 0.384 | **6.560 ± 0.278** |
| nary_matmul_chain_64 | **0.006 ± 0.000** | 0.020 ± 0.004 | 0.031 ± 0.001 | 0.096 ± 0.007 |
| str_matrix_chain_multiplication_100 | **2.137 ± 0.061** | 3.515 ± 0.181 | 3.334 ± 0.279 | 7.141 ± 0.249 |
| str_mps_varying_inner_product_200 | **6.917 ± 0.186** | 10.098 ± 0.457 | 7.918 ± 0.336 | 16.642 ± 0.103 |
| str_nw_mera_closed_120 | **125.416 ± 0.360** | 139.712 ± 1.880 | 126.507 ± 0.592 | 210.541 ± 6.803 |
| str_nw_mera_open_26 | 120.284 ± 0.944 | 158.384 ± 4.301 | **119.172 ± 3.719** | 174.560 ± 4.209 |
| tensornetwork_permutation_focus_step409_316 | 82.179 ± 3.523 | 101.308 ± 6.138 | **78.719 ± 0.766** | 134.287 ± 3.941 |
| tensornetwork_permutation_light_415 | 86.004 ± 1.680 | 97.253 ± 3.017 | **75.361 ± 2.227** | 139.150 ± 3.812 |

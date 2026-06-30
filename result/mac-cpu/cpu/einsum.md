# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260630_183200/run.yaml`
- Timestamp: `20260630_183200`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260630_183200`.

- tenferro-rs commit: `38b1c5f2a0f0229336dda751d1033cae3cfc106a`

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

- Source table: `data/results/mac-cpu/cpu/einsum/20260630_183200/einsum_table_t4_20260630_183200.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260630_183200/tenferro_trace_t4_20260630_183200.log`
- `data/results/mac-cpu/cpu/einsum/20260630_183200/tenferro_eager_t4_20260630_183200.log`
- `data/results/mac-cpu/cpu/einsum/20260630_183200/pytorch_cpu_t4_20260630_183200.log`
- `data/results/mac-cpu/cpu/einsum/20260630_183200/jax_cpu_t4_20260630_183200.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.583 ± 0.092 | 0.686 ± 0.140 | **0.411 ± 0.005** | 0.685 ± 0.054 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.081 ± 0.009** | 0.103 ± 0.008 | 0.104 ± 0.002 | 0.182 ± 0.008 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.109 ± 0.013 | 0.132 ± 0.052 | **0.103 ± 0.005** | 0.145 ± 0.024 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.111 ± 0.012** | 0.152 ± 0.027 | 0.137 ± 0.019 | 0.152 ± 0.020 |
| bin_elementwise_mul_2048x2048 | **1.115 ± 0.071** | 1.312 ± 0.530 | 1.134 ± 0.080 | 1.570 ± 0.074 |
| bin_matmul_1024 | 4.698 ± 0.232 | 4.660 ± 0.109 | **4.343 ± 0.083** | 8.210 ± 0.624 |
| bin_matmul_256 | **0.076 ± 0.001** | 0.083 ± 0.003 | 0.087 ± 0.003 | 0.333 ± 0.069 |
| bin_outer_product_4096 | **1.713 ± 0.155** | 1.835 ± 0.085 | 1.793 ± 0.045 | 1.924 ± 0.041 |
| gm_queen5_5_3.wcsp | **604.451 ± 48.758** | 745.004 ± 37.792 | 714.708 ± 14.017 | 899.141 ± 37.190 |
| lm_batch_likelihood_brackets_4_4d | 7.559 ± 0.302 | 13.997 ± 0.485 | 10.350 ± 1.104 | **7.360 ± 0.542** |
| lm_batch_likelihood_sentence_3_12d | 12.312 ± 1.616 | 23.926 ± 0.865 | 26.521 ± 1.112 | **12.067 ± 0.914** |
| lm_batch_likelihood_sentence_4_4d | **7.188 ± 0.199** | 14.659 ± 0.683 | 8.875 ± 0.857 | 7.783 ± 0.306 |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.038 ± 0.035 | 0.030 ± 0.001 | 0.089 ± 0.025 |
| str_matrix_chain_multiplication_100 | **2.174 ± 0.061** | 3.401 ± 0.200 | 3.208 ± 0.080 | 7.373 ± 2.725 |
| str_mps_varying_inner_product_200 | **5.339 ± 0.171** | 10.531 ± 0.731 | 6.478 ± 0.318 | 15.402 ± 0.348 |
| str_nw_mera_closed_120 | 146.388 ± 0.952 | 166.771 ± 2.085 | **144.394 ± 1.974** | 380.379 ± 112.389 |
| str_nw_mera_open_26 | 121.308 ± 0.922 | 160.934 ± 2.149 | **118.544 ± 3.015** | 283.055 ± 33.305 |
| tensornetwork_permutation_focus_step409_316 | **84.272 ± 6.211** | 105.227 ± 24.744 | 99.692 ± 11.489 | 135.503 ± 12.094 |
| tensornetwork_permutation_light_415 | **68.724 ± 20.159** | 106.259 ± 7.753 | 102.887 ± 3.466 | 156.614 ± 11.621 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.583 ± 0.092 | 0.686 ± 0.140 | **0.411 ± 0.005** | 0.685 ± 0.054 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.081 ± 0.009** | 0.103 ± 0.008 | 0.104 ± 0.002 | 0.182 ± 0.008 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.109 ± 0.013 | 0.132 ± 0.052 | **0.103 ± 0.005** | 0.145 ± 0.024 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.111 ± 0.012** | 0.152 ± 0.027 | 0.137 ± 0.019 | 0.152 ± 0.020 |
| bin_elementwise_mul_2048x2048 | **1.115 ± 0.071** | 1.312 ± 0.530 | 1.134 ± 0.080 | 1.570 ± 0.074 |
| bin_matmul_1024 | 4.698 ± 0.232 | 4.660 ± 0.109 | **4.343 ± 0.083** | 8.210 ± 0.624 |
| bin_matmul_256 | **0.076 ± 0.001** | 0.083 ± 0.003 | 0.087 ± 0.003 | 0.333 ± 0.069 |
| bin_outer_product_4096 | **1.713 ± 0.155** | 1.835 ± 0.085 | 1.793 ± 0.045 | 1.924 ± 0.041 |
| gm_queen5_5_3.wcsp | **176.633 ± 14.699** | 267.052 ± 29.350 | 264.919 ± 30.202 | 387.032 ± 34.388 |
| lm_batch_likelihood_brackets_4_4d | 8.020 ± 0.453 | 15.001 ± 0.831 | 10.112 ± 0.500 | **7.877 ± 0.376** |
| lm_batch_likelihood_sentence_3_12d | **12.675 ± 0.229** | 30.170 ± 2.678 | 25.403 ± 1.851 | 14.624 ± 0.924 |
| lm_batch_likelihood_sentence_4_4d | 8.369 ± 0.375 | 13.697 ± 0.519 | 8.943 ± 0.434 | **7.497 ± 0.329** |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.038 ± 0.035 | 0.030 ± 0.001 | 0.089 ± 0.025 |
| str_matrix_chain_multiplication_100 | **2.099 ± 0.152** | 3.206 ± 0.194 | 3.273 ± 0.201 | 8.764 ± 0.370 |
| str_mps_varying_inner_product_200 | **7.037 ± 0.307** | 8.960 ± 0.199 | 7.851 ± 0.217 | 16.997 ± 0.375 |
| str_nw_mera_closed_120 | **127.135 ± 1.272** | 141.464 ± 2.017 | 127.800 ± 2.436 | 404.189 ± 70.424 |
| str_nw_mera_open_26 | **124.560 ± 2.813** | 161.473 ± 5.125 | 129.493 ± 3.367 | 292.515 ± 16.471 |
| tensornetwork_permutation_focus_step409_316 | **84.272 ± 6.211** | 105.227 ± 24.744 | 99.692 ± 11.489 | 135.503 ± 12.094 |
| tensornetwork_permutation_light_415 | **68.724 ± 20.159** | 106.259 ± 7.753 | 102.887 ± 3.466 | 156.614 ± 11.621 |

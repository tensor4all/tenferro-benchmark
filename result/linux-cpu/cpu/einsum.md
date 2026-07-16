# Linux CPU Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `linux-cpu`
- Thread runs: `1 4`

Mirrored from `amd-cpu` runs (`BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh <THREADS>`);
regenerate with `./scripts/mirror_linux_cpu_cpu_reports.sh 1:20260716_094347 4:20260716_100934`.

This report is generated from sequential CPU runs. Do not compare it with
measurements collected while another CPU benchmark process was running.

## Run Inputs

- Threads 1: timestamp `20260716_094347`, raw run `data/results/linux-cpu/cpu/einsum/20260716_094347`
- Threads 4: timestamp `20260716_100934`, raw run `data/results/linux-cpu/cpu/einsum/20260716_100934`

## Threads: 1

### Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `linux-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/linux-cpu/cpu/einsum/20260716_094347/run.yaml`
- Timestamp: `20260716_094347`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/linux-cpu/cpu/einsum/20260716_094347`.

- tenferro-rs commit: `3abc9108e4f0500f4e75519711cdb4e21b9625df`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-134-generic-x86_64-with-glibc2.39`

#### Thread Environment

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

#### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-openblas`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `openblas`
- BLAS version: `OpenBLAS 0.3.26`
- BLAS root: `/opt/openblas`
- BLAS library: `/opt/openblas/lib/libopenblas.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 1

- Source table: `data/results/linux-cpu/cpu/einsum/20260716_094347/einsum_table_t1_20260716_094347.md`

Logs:

- `data/results/linux-cpu/cpu/einsum/20260716_094347/tenferro_trace_t1_20260716_094347.log`
- `data/results/linux-cpu/cpu/einsum/20260716_094347/tenferro_eager_t1_20260716_094347.log`
- `data/results/linux-cpu/cpu/einsum/20260716_094347/pytorch_cpu_t1_20260716_094347.log`
- `data/results/linux-cpu/cpu/einsum/20260716_094347/jax_cpu_t1_20260716_094347.log`

###### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.685 ± 0.054 | 3.721 ± 0.005 | 3.563 ± 0.049 | **2.294 ± 0.319** |
| bin_batched_matmul_b32_m64_n64_k64 | 0.569 ± 0.022 | 0.635 ± 0.020 | **0.527 ± 0.009** | 1.637 ± 0.035 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.679 ± 0.023 | 0.673 ± 0.013 | **0.547 ± 0.007** | 1.432 ± 0.107 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.713 ± 0.029 | 0.702 ± 0.034 | **0.547 ± 0.006** | 1.572 ± 0.059 |
| bin_elementwise_mul_2048x2048 | **8.709 ± 2.564** | 19.799 ± 0.113 | 19.484 ± 0.118 | 13.257 ± 6.846 |
| bin_matmul_1024 | 50.702 ± 2.533 | 48.704 ± 20.522 | 45.418 ± 0.124 | **12.696 ± 4.698** |
| bin_matmul_256 | **0.841 ± 0.035** | 0.849 ± 0.040 | 0.886 ± 0.003 | 1.517 ± 0.072 |
| bin_outer_product_4096 | 55.140 ± 1.275 | 55.637 ± 0.205 | 58.340 ± 0.429 | **34.848 ± 3.033** |
| gm_queen5_5_3.wcsp | 12211.420 ± 43.068 | 9687.375 ± 87.365 | **6325.834 ± 8.707** | - |
| lm_batch_likelihood_brackets_4_4d | **34.942 ± 0.242** | 65.502 ± 0.982 | 35.665 ± 0.241 | 36.730 ± 2.868 |
| lm_batch_likelihood_sentence_3_12d | 83.326 ± 0.292 | 287.630 ± 2.047 | 95.897 ± 0.383 | **49.343 ± 1.622** |
| lm_batch_likelihood_sentence_4_4d | **34.127 ± 0.313** | 71.450 ± 3.419 | 38.024 ± 0.169 | 38.137 ± 1.029 |
| nary_matmul_chain_64 | **0.046 ± 0.004** | 0.061 ± 0.015 | 0.170 ± 0.008 | 0.685 ± 0.029 |
| str_matrix_chain_multiplication_100 | **16.389 ± 0.069** | 17.634 ± 0.417 | 21.989 ± 0.180 | 43.776 ± 1.743 |
| str_mps_varying_inner_product_200 | **19.623 ± 0.065** | 34.218 ± 0.219 | 29.326 ± 0.176 | 93.493 ± 1.302 |
| str_nw_mera_closed_120 | 1907.146 ± 3.589 | 2274.814 ± 3.448 | 1876.564 ± 3.923 | **585.884 ± 24.506** |
| str_nw_mera_open_26 | 1274.322 ± 6.480 | 1991.531 ± 16.193 | 1252.706 ± 3.075 | **550.994 ± 56.156** |
| tensornetwork_permutation_focus_step409_316 | 630.571 ± 7.658 | 1184.040 ± 6.510 | 883.662 ± 1.552 | **473.911 ± 33.657** |
| tensornetwork_permutation_light_415 | 631.449 ± 3.264 | 1015.545 ± 6.529 | 857.464 ± 5.806 | **525.844 ± 59.384** |

###### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.685 ± 0.054 | 3.721 ± 0.005 | 3.563 ± 0.049 | **2.294 ± 0.319** |
| bin_batched_matmul_b32_m64_n64_k64 | 0.569 ± 0.022 | 0.635 ± 0.020 | **0.527 ± 0.009** | 1.637 ± 0.035 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.679 ± 0.023 | 0.673 ± 0.013 | **0.547 ± 0.007** | 1.432 ± 0.107 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.713 ± 0.029 | 0.702 ± 0.034 | **0.547 ± 0.006** | 1.572 ± 0.059 |
| bin_elementwise_mul_2048x2048 | **8.709 ± 2.564** | 19.799 ± 0.113 | 19.484 ± 0.118 | 13.257 ± 6.846 |
| bin_matmul_1024 | 50.702 ± 2.533 | 48.704 ± 20.522 | 45.418 ± 0.124 | **12.696 ± 4.698** |
| bin_matmul_256 | **0.841 ± 0.035** | 0.849 ± 0.040 | 0.886 ± 0.003 | 1.517 ± 0.072 |
| bin_outer_product_4096 | 55.140 ± 1.275 | 55.637 ± 0.205 | 58.340 ± 0.429 | **34.848 ± 3.033** |
| gm_queen5_5_3.wcsp | 3858.272 ± 156.177 | 4328.021 ± 104.831 | **2698.572 ± 8.088** | - |
| lm_batch_likelihood_brackets_4_4d | 36.438 ± 0.256 | 69.780 ± 3.998 | 39.321 ± 0.509 | **36.232 ± 1.124** |
| lm_batch_likelihood_sentence_3_12d | 108.326 ± 0.622 | 168.903 ± 3.256 | 94.102 ± 1.317 | **60.865 ± 9.552** |
| lm_batch_likelihood_sentence_4_4d | 40.272 ± 0.110 | 76.740 ± 1.442 | 39.885 ± 0.208 | **38.135 ± 1.518** |
| nary_matmul_chain_64 | **0.046 ± 0.004** | 0.061 ± 0.015 | 0.170 ± 0.008 | 0.685 ± 0.029 |
| str_matrix_chain_multiplication_100 | **17.104 ± 0.054** | 22.151 ± 0.962 | 21.131 ± 0.186 | 46.618 ± 3.370 |
| str_mps_varying_inner_product_200 | **25.561 ± 0.096** | 38.470 ± 1.804 | 34.045 ± 0.393 | 108.039 ± 0.570 |
| str_nw_mera_closed_120 | 1569.386 ± 2.293 | 1800.055 ± 5.737 | 1461.659 ± 3.492 | **456.428 ± 27.330** |
| str_nw_mera_open_26 | 1329.240 ± 2.027 | 2021.340 ± 5.914 | 1264.161 ± 1.711 | **563.296 ± 90.632** |
| tensornetwork_permutation_focus_step409_316 | 630.571 ± 7.658 | 1184.040 ± 6.510 | 883.662 ± 1.552 | **473.911 ± 33.657** |
| tensornetwork_permutation_light_415 | 631.449 ± 3.264 | 1015.545 ± 6.529 | 857.464 ± 5.806 | **525.844 ± 59.384** |

## Threads: 4

### Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `linux-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/linux-cpu/cpu/einsum/20260716_100934/run.yaml`
- Timestamp: `20260716_100934`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/linux-cpu/cpu/einsum/20260716_100934`.

- tenferro-rs commit: `3abc9108e4f0500f4e75519711cdb4e21b9625df`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-134-generic-x86_64-with-glibc2.39`

#### Thread Environment

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

#### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-openblas`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `openblas`
- BLAS version: `OpenBLAS 0.3.26`
- BLAS root: `/opt/openblas`
- BLAS library: `/opt/openblas/lib/libopenblas.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 4

- Source table: `data/results/linux-cpu/cpu/einsum/20260716_100934/einsum_table_t4_20260716_100934.md`

Logs:

- `data/results/linux-cpu/cpu/einsum/20260716_100934/tenferro_trace_t4_20260716_100934.log`
- `data/results/linux-cpu/cpu/einsum/20260716_100934/tenferro_eager_t4_20260716_100934.log`
- `data/results/linux-cpu/cpu/einsum/20260716_100934/pytorch_cpu_t4_20260716_100934.log`
- `data/results/linux-cpu/cpu/einsum/20260716_100934/jax_cpu_t4_20260716_100934.log`

###### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.043 ± 0.025 | 2.028 ± 0.059 | **1.367 ± 0.021** | 2.024 ± 0.063 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.711 ± 0.052 | 0.716 ± 0.051 | **0.267 ± 0.014** | 1.458 ± 0.128 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.211 ± 0.017 | **0.199 ± 0.094** | 0.245 ± 0.021 | 1.481 ± 0.065 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.210 ± 0.014 | **0.208 ± 1.206** | 0.247 ± 0.016 | 1.532 ± 0.048 |
| bin_elementwise_mul_2048x2048 | **2.320 ± 0.020** | 7.725 ± 0.267 | 8.396 ± 0.276 | 14.264 ± 3.667 |
| bin_matmul_1024 | 15.200 ± 0.114 | 18.921 ± 0.060 | 14.614 ± 0.165 | **12.515 ± 3.847** |
| bin_matmul_256 | **0.341 ± 0.034** | 0.379 ± 0.015 | 0.362 ± 0.024 | 1.488 ± 0.088 |
| bin_outer_product_4096 | **19.407 ± 1.201** | 29.466 ± 5.725 | 25.685 ± 0.558 | 46.452 ± 1.063 |
| gm_queen5_5_3.wcsp | 5992.980 ± 35.348 | 5001.396 ± 89.294 | **2242.928 ± 8.887** | - |
| lm_batch_likelihood_brackets_4_4d | 45.456 ± 0.364 | 67.241 ± 0.832 | **16.027 ± 0.123** | 44.026 ± 2.770 |
| lm_batch_likelihood_sentence_3_12d | 60.433 ± 0.713 | 219.457 ± 1.319 | **20.509 ± 0.290** | 43.346 ± 2.516 |
| lm_batch_likelihood_sentence_4_4d | 34.321 ± 0.627 | 74.848 ± 2.374 | **16.797 ± 0.387** | 42.631 ± 1.283 |
| nary_matmul_chain_64 | **0.051 ± 0.008** | 0.081 ± 0.001 | 0.169 ± 0.018 | 0.818 ± 0.019 |
| str_matrix_chain_multiplication_100 | **8.201 ± 0.090** | 11.740 ± 0.072 | 11.606 ± 0.096 | 45.791 ± 1.876 |
| str_mps_varying_inner_product_200 | **22.967 ± 0.167** | 47.551 ± 0.310 | 24.924 ± 0.883 | 95.156 ± 1.350 |
| str_nw_mera_closed_120 | 1012.553 ± 4.501 | 1404.778 ± 23.799 | 666.622 ± 4.033 | **578.510 ± 29.443** |
| str_nw_mera_open_26 | 568.954 ± 3.125 | 981.404 ± 4.919 | **441.084 ± 3.555** | 552.079 ± 68.527 |
| tensornetwork_permutation_focus_step409_316 | 382.143 ± 5.428 | 714.271 ± 5.196 | **287.472 ± 2.690** | 467.321 ± 25.979 |
| tensornetwork_permutation_light_415 | 389.337 ± 2.500 | 688.861 ± 4.019 | **288.944 ± 3.315** | 500.111 ± 12.909 |

###### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.043 ± 0.025 | 2.028 ± 0.059 | **1.367 ± 0.021** | 2.024 ± 0.063 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.711 ± 0.052 | 0.716 ± 0.051 | **0.267 ± 0.014** | 1.458 ± 0.128 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.211 ± 0.017 | **0.199 ± 0.094** | 0.245 ± 0.021 | 1.481 ± 0.065 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.210 ± 0.014 | **0.208 ± 1.206** | 0.247 ± 0.016 | 1.532 ± 0.048 |
| bin_elementwise_mul_2048x2048 | **2.320 ± 0.020** | 7.725 ± 0.267 | 8.396 ± 0.276 | 14.264 ± 3.667 |
| bin_matmul_1024 | 15.200 ± 0.114 | 18.921 ± 0.060 | 14.614 ± 0.165 | **12.515 ± 3.847** |
| bin_matmul_256 | **0.341 ± 0.034** | 0.379 ± 0.015 | 0.362 ± 0.024 | 1.488 ± 0.088 |
| bin_outer_product_4096 | **19.407 ± 1.201** | 29.466 ± 5.725 | 25.685 ± 0.558 | 46.452 ± 1.063 |
| gm_queen5_5_3.wcsp | 2012.211 ± 79.986 | 2371.789 ± 101.902 | **990.694 ± 6.465** | - |
| lm_batch_likelihood_brackets_4_4d | 36.762 ± 2.222 | 76.624 ± 2.504 | **16.435 ± 0.176** | 44.391 ± 1.356 |
| lm_batch_likelihood_sentence_3_12d | 76.388 ± 0.671 | 111.331 ± 0.461 | **42.556 ± 0.743** | 48.443 ± 10.575 |
| lm_batch_likelihood_sentence_4_4d | 39.786 ± 0.271 | 84.835 ± 0.688 | **16.875 ± 0.171** | 37.359 ± 0.888 |
| nary_matmul_chain_64 | **0.051 ± 0.008** | 0.081 ± 0.001 | 0.169 ± 0.018 | 0.818 ± 0.019 |
| str_matrix_chain_multiplication_100 | **8.325 ± 0.045** | 12.621 ± 0.077 | 11.332 ± 0.108 | 45.016 ± 2.593 |
| str_mps_varying_inner_product_200 | **32.858 ± 0.887** | 51.180 ± 0.305 | 33.390 ± 0.337 | 105.309 ± 1.675 |
| str_nw_mera_closed_120 | 745.229 ± 7.568 | 989.593 ± 6.893 | 518.851 ± 5.372 | **481.731 ± 22.469** |
| str_nw_mera_open_26 | 579.574 ± 2.909 | 970.828 ± 3.787 | **455.711 ± 5.332** | 546.115 ± 32.670 |
| tensornetwork_permutation_focus_step409_316 | 382.143 ± 5.428 | 714.271 ± 5.196 | **287.472 ± 2.690** | 467.321 ± 25.979 |
| tensornetwork_permutation_light_415 | 389.337 ± 2.500 | 688.861 ± 4.019 | **288.944 ± 3.315** | 500.111 ± 12.909 |

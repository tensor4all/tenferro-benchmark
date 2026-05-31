# Einsum Benchmark Results

Latest run: `./scripts/run_all.sh 4`.

This file collects the latest einsum benchmark table for each thread count under `data/results/`.

- tenferro-rs commit: `fa722375c8662b5532a6f875a6bef3494ace40b5`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-124-generic-x86_64-with-glibc2.39`

## Threads: 1

- Timestamp: `20260531_023422`
- Source table: `data/results/results_t1_20260531_023422.md`

Logs:

- `data/results/tenferro_trace_t1_20260531_023422.log`
- `data/results/tenferro_eager_t1_20260531_023422.log`
- `data/results/libtorch_cpu_t1_20260531_023422.log`
- `data/results/pytorch_cpu_t1_20260531_023422.log`
- `data/results/jax_cpu_t1_20260531_023422.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 3.322 ± 0.157 | 2.496 ± 0.172 | 0.745 ± 0.003 | **0.536 ± 0.009** | 1.352 ± 0.062 |
| bin_elementwise_mul_2048x2048 | 95.368 ± 2.000 | 73.649 ± 1.182 | 18.497 ± 0.432 | 20.851 ± 0.416 | **14.103 ± 1.010** |
| bin_matmul_256 | 1.720 ± 0.024 | 1.403 ± 0.059 | **0.884 ± 0.002** | 0.894 ± 0.003 | 1.531 ± 0.110 |
| bin_outer_product_4096 | 205.174 ± 10.087 | 281.176 ± 3.336 | 53.414 ± 2.153 | 59.128 ± 0.624 | **32.100 ± 4.672** |
| gm_queen5_5_3.wcsp | 10567.244 ± 121.034 | 16847.162 ± 109.759 | 8675.775 ± 34.471 | **6378.095 ± 22.295** | - |
| lm_batch_likelihood_brackets_4_4d | **34.460 ± 0.458** | 79.263 ± 2.157 | 55.642 ± 0.247 | 38.775 ± 0.998 | 43.935 ± 0.904 |
| lm_batch_likelihood_sentence_3_12d | 64.978 ± 1.203 | 233.234 ± 5.679 | 111.544 ± 1.245 | 92.040 ± 0.747 | **43.697 ± 1.021** |
| lm_batch_likelihood_sentence_4_4d | **30.300 ± 0.739** | 91.040 ± 26.692 | 54.897 ± 0.577 | 40.402 ± 0.459 | 42.745 ± 0.503 |
| str_matrix_chain_multiplication_100 | 41.300 ± 1.069 | 28.538 ± 0.506 | 29.880 ± 0.363 | **23.618 ± 0.246** | 46.533 ± 1.641 |
| str_mps_varying_inner_product_200 | 37.005 ± 0.599 | 39.839 ± 0.710 | 71.911 ± 0.607 | **32.957 ± 0.391** | 95.110 ± 1.807 |
| str_nw_mera_closed_120 | 2139.756 ± 7.231 | 2549.594 ± 14.967 | 1992.393 ± 38.943 | 1881.543 ± 5.805 | **613.892 ± 33.789** |
| str_nw_mera_open_26 | 1768.520 ± 2.542 | 2335.290 ± 3.458 | 1273.708 ± 24.938 | 1241.230 ± 5.991 | **536.221 ± 76.485** |
| tensornetwork_permutation_focus_step409_316 | 586.952 ± 2.768 | 1199.312 ± 10.228 | 761.717 ± 23.166 | 857.023 ± 3.465 | **467.854 ± 16.043** |
| tensornetwork_permutation_light_415 | 590.212 ± 3.345 | 902.107 ± 3.383 | 963.966 ± 21.460 | 829.332 ± 2.823 | **507.067 ± 21.812** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 2.113 ± 0.089 | 2.019 ± 0.036 | 0.739 ± 0.040 | **0.550 ± 0.011** | 1.489 ± 0.035 |
| bin_elementwise_mul_2048x2048 | 69.589 ± 0.441 | 73.707 ± 1.136 | 18.275 ± 0.298 | 19.997 ± 0.204 | **6.206 ± 0.615** |
| bin_matmul_256 | 1.152 ± 0.057 | 1.157 ± 0.070 | **0.881 ± 0.037** | 0.894 ± 0.066 | 1.436 ± 0.066 |
| bin_outer_product_4096 | 197.539 ± 1.254 | 277.771 ± 2.119 | 53.416 ± 0.729 | 60.489 ± 1.049 | **47.133 ± 2.062** |
| gm_queen5_5_3.wcsp | 3896.849 ± 16.235 | 5671.062 ± 32.988 | **2669.321 ± 54.804** | 2718.425 ± 7.370 | - |
| lm_batch_likelihood_brackets_4_4d | **29.652 ± 0.682** | 55.732 ± 0.979 | 51.809 ± 0.807 | 42.989 ± 0.613 | 44.621 ± 0.553 |
| lm_batch_likelihood_sentence_3_12d | 74.600 ± 1.108 | 213.429 ± 1.466 | 88.668 ± 1.660 | 59.117 ± 0.259 | **47.658 ± 4.295** |
| lm_batch_likelihood_sentence_4_4d | **31.387 ± 0.425** | 62.827 ± 1.619 | 51.388 ± 0.958 | 42.668 ± 0.620 | 37.429 ± 1.601 |
| str_matrix_chain_multiplication_100 | 41.778 ± 0.578 | 28.829 ± 0.633 | 30.593 ± 0.511 | **21.762 ± 0.637** | 42.823 ± 3.180 |
| str_mps_varying_inner_product_200 | 38.927 ± 0.492 | 41.337 ± 0.287 | 76.362 ± 1.244 | **37.345 ± 0.703** | 106.241 ± 3.096 |
| str_nw_mera_closed_120 | 1878.514 ± 30.337 | 2085.781 ± 5.055 | 2085.143 ± 5.164 | 1479.377 ± 4.474 | **443.481 ± 12.981** |
| str_nw_mera_open_26 | 1965.011 ± 53.997 | 2326.233 ± 20.247 | 1291.882 ± 3.418 | 1264.865 ± 2.949 | **537.081 ± 65.267** |
| tensornetwork_permutation_focus_step409_316 | 633.921 ± 5.501 | 1179.693 ± 2.942 | 736.570 ± 2.464 | 856.589 ± 3.541 | **493.820 ± 8.081** |
| tensornetwork_permutation_light_415 | 623.933 ± 15.973 | 885.740 ± 12.470 | 960.532 ± 2.867 | 832.840 ± 5.223 | **513.212 ± 20.782** |


## Threads: 4

- Timestamp: `20260531_031122`
- Source table: `data/results/results_t4_20260531_031122.md`

Logs:

- `data/results/tenferro_trace_t4_20260531_031122.log`
- `data/results/tenferro_eager_t4_20260531_031122.log`
- `data/results/libtorch_cpu_t4_20260531_031122.log`
- `data/results/pytorch_cpu_t4_20260531_031122.log`
- `data/results/jax_cpu_t4_20260531_031122.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 3.784 ± 0.296 | 3.434 ± 0.106 | 0.785 ± 0.060 | **0.287 ± 0.013** | 0.771 ± 0.045 |
| bin_elementwise_mul_2048x2048 | 96.026 ± 1.627 | 75.774 ± 0.576 | **7.100 ± 0.288** | 8.415 ± 0.252 | 13.973 ± 1.023 |
| bin_matmul_256 | 1.772 ± 0.287 | 1.357 ± 0.085 | 0.411 ± 0.015 | **0.363 ± 0.027** | 1.431 ± 0.066 |
| bin_outer_product_4096 | 196.979 ± 1.633 | 280.741 ± 2.224 | **17.161 ± 0.285** | 25.175 ± 0.382 | 45.562 ± 0.835 |
| gm_queen5_5_3.wcsp | 10448.705 ± 73.626 | 16525.941 ± 264.290 | 3052.326 ± 87.267 | **2326.261 ± 18.237** | - |
| lm_batch_likelihood_brackets_4_4d | 41.823 ± 0.461 | 88.262 ± 22.406 | 40.335 ± 0.308 | **17.370 ± 0.549** | 43.981 ± 1.896 |
| lm_batch_likelihood_sentence_3_12d | 100.123 ± 4.416 | 236.604 ± 1.947 | 84.937 ± 0.988 | **20.813 ± 0.182** | 53.978 ± 6.255 |
| lm_batch_likelihood_sentence_4_4d | 53.010 ± 3.415 | 105.400 ± 19.078 | 40.814 ± 0.450 | **17.876 ± 0.364** | 42.559 ± 1.144 |
| str_matrix_chain_multiplication_100 | 50.472 ± 14.048 | 67.119 ± 26.399 | 22.834 ± 0.440 | **11.847 ± 0.338** | 45.147 ± 2.643 |
| str_mps_varying_inner_product_200 | 91.099 ± 2.147 | 133.856 ± 4.249 | 83.923 ± 0.396 | **27.645 ± 0.685** | 93.032 ± 2.254 |
| str_nw_mera_closed_120 | 1387.162 ± 32.704 | 1879.354 ± 17.032 | 870.586 ± 12.382 | 689.154 ± 6.727 | **608.362 ± 23.201** |
| str_nw_mera_open_26 | 1471.695 ± 229.514 | 1980.055 ± 13.449 | 708.345 ± 17.240 | **469.918 ± 3.834** | 558.405 ± 69.263 |
| tensornetwork_permutation_focus_step409_316 | 677.278 ± 4.188 | 1255.673 ± 5.091 | 349.153 ± 27.618 | **297.463 ± 12.619** | 481.590 ± 15.898 |
| tensornetwork_permutation_light_415 | 681.599 ± 4.089 | 946.549 ± 5.636 | 457.013 ± 3.772 | **299.408 ± 3.200** | 466.286 ± 32.017 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 2.915 ± 0.088 | 2.895 ± 0.069 | 0.833 ± 0.016 | **0.277 ± 0.031** | 1.513 ± 0.033 |
| bin_elementwise_mul_2048x2048 | 96.321 ± 3.244 | 75.754 ± 0.961 | 7.237 ± 0.182 | 8.921 ± 0.315 | **6.874 ± 0.336** |
| bin_matmul_256 | 1.133 ± 0.068 | 1.079 ± 0.206 | **0.354 ± 0.033** | 0.367 ± 0.020 | 1.435 ± 0.050 |
| bin_outer_product_4096 | 201.960 ± 5.479 | 278.795 ± 0.904 | **18.419 ± 3.363** | 26.437 ± 0.913 | 43.450 ± 1.156 |
| gm_queen5_5_3.wcsp | 3674.739 ± 59.776 | 5463.520 ± 26.337 | 1156.217 ± 16.579 | **1002.195 ± 10.971** | - |
| lm_batch_likelihood_brackets_4_4d | 40.489 ± 3.137 | 65.505 ± 1.095 | 38.638 ± 0.398 | **17.822 ± 0.231** | 45.308 ± 1.419 |
| lm_batch_likelihood_sentence_3_12d | 128.009 ± 20.823 | 221.901 ± 8.557 | 66.006 ± 0.550 | **22.250 ± 0.697** | 47.301 ± 0.804 |
| lm_batch_likelihood_sentence_4_4d | 55.226 ± 3.952 | 73.076 ± 0.879 | 39.777 ± 0.501 | **18.251 ± 0.268** | 37.294 ± 1.263 |
| str_matrix_chain_multiplication_100 | 47.330 ± 12.869 | 61.707 ± 24.922 | 22.872 ± 0.282 | **11.746 ± 0.334** | 43.132 ± 2.754 |
| str_mps_varying_inner_product_200 | 86.565 ± 0.878 | 140.300 ± 1.999 | 87.835 ± 1.070 | **36.063 ± 0.447** | 105.317 ± 2.017 |
| str_nw_mera_closed_120 | 1138.864 ± 129.166 | 1397.215 ± 78.622 | 784.554 ± 4.424 | 543.410 ± 3.390 | **439.740 ± 26.239** |
| str_nw_mera_open_26 | 1575.655 ± 167.967 | 2067.310 ± 151.659 | 700.628 ± 4.619 | **466.307 ± 3.284** | 549.232 ± 50.435 |
| tensornetwork_permutation_focus_step409_316 | 674.423 ± 8.145 | 1253.254 ± 3.878 | 346.858 ± 1.543 | **294.137 ± 2.532** | 468.007 ± 16.024 |
| tensornetwork_permutation_light_415 | 678.127 ± 4.766 | 946.544 ± 4.755 | 457.647 ± 2.610 | **302.373 ± 5.526** | 515.374 ± 18.714 |


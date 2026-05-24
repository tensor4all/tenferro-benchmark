# Einsum Benchmark Results

Latest run: `./scripts/run_all.sh 4`.

This file collects the latest einsum benchmark table for each thread count under `data/results/`.

- tenferro-rs commit: `70dd331362dc02c87727ddee5d034da59e410ad5`

## Threads: 1

- Timestamp: `20260524_162107`
- Source table: `data/results/results_t1_20260524_162107.md`

Logs:

- `data/results/tenferro_trace_t1_20260524_162107.log`
- `data/results/tenferro_eager_t1_20260524_162107.log`
- `data/results/libtorch_cpu_t1_20260524_162107.log`
- `data/results/pytorch_cpu_t1_20260524_162107.log`
- `data/results/jax_cpu_t1_20260524_162107.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 1.692 ± 0.215 | 1.034 ± 0.028 | 0.798 ± 0.059 | **0.130 ± 0.004** | 0.217 ± 0.078 |
| bin_elementwise_mul_2048x2048 | 3.608 ± 0.264 | 2.625 ± 0.069 | **1.160 ± 0.055** | 1.177 ± 0.058 | 1.384 ± 0.486 |
| bin_matmul_256 | 0.824 ± 0.051 | 0.819 ± 0.027 | 0.791 ± 0.027 | **0.112 ± 0.003** | 0.307 ± 0.065 |
| bin_outer_product_4096 | 11.819 ± 0.209 | 14.577 ± 0.363 | 2.156 ± 0.073 | 2.566 ± 0.078 | **1.681 ± 0.485** |
| gm_queen5_5_3.wcsp | 1874.841 ± 51.751 | 2708.191 ± 20.293 | 1988.068 ± 113.760 | 1335.616 ± 13.733 | **843.043 ± 3.410** |
| lm_batch_likelihood_brackets_4_4d | 18.404 ± 0.389 | 27.096 ± 0.222 | 22.292 ± 0.343 | 13.633 ± 0.223 | **8.924 ± 0.285** |
| lm_batch_likelihood_sentence_3_12d | 42.150 ± 0.724 | 71.444 ± 0.548 | 48.658 ± 2.815 | 21.781 ± 0.191 | **11.249 ± 0.203** |
| lm_batch_likelihood_sentence_4_4d | 17.030 ± 0.455 | 28.940 ± 0.142 | 21.785 ± 0.287 | 14.061 ± 0.117 | **8.774 ± 0.331** |
| str_matrix_chain_multiplication_100 | 11.698 ± 0.160 | 10.160 ± 0.317 | 17.589 ± 0.269 | **4.379 ± 0.131** | 9.171 ± 0.239 |
| str_mps_varying_inner_product_200 | 13.621 ± 0.221 | 15.764 ± 0.301 | 40.624 ± 1.810 | **7.834 ± 0.149** | 19.901 ± 0.386 |
| str_nw_mera_closed_120 | 1014.287 ± 4.789 | 1060.896 ± 23.205 | 1016.282 ± 18.160 | **161.954 ± 1.767** | 166.152 ± 5.558 |
| str_nw_mera_open_26 | 738.622 ± 5.484 | 767.041 ± 11.388 | 698.612 ± 28.673 | **110.601 ± 5.131** | 129.190 ± 3.147 |
| tensornetwork_permutation_focus_step409_316 | 268.765 ± 5.918 | 375.652 ± 21.006 | 258.666 ± 5.056 | 230.649 ± 2.969 | **103.544 ± 6.028** |
| tensornetwork_permutation_light_415 | 267.278 ± 2.510 | 313.629 ± 9.934 | 352.603 ± 5.704 | 193.430 ± 2.097 | **113.342 ± 1.662** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.755 ± 0.041 | 0.696 ± 0.034 | 0.411 ± 0.003 | **0.129 ± 0.006** | 0.232 ± 0.052 |
| bin_elementwise_mul_2048x2048 | 3.617 ± 0.111 | 2.532 ± 0.077 | **1.141 ± 0.090** | 1.156 ± 0.061 | 1.762 ± 0.399 |
| bin_matmul_256 | 0.720 ± 0.047 | 0.741 ± 0.058 | 0.678 ± 0.015 | **0.118 ± 0.015** | 0.317 ± 0.032 |
| bin_outer_product_4096 | 11.801 ± 0.111 | 14.680 ± 0.240 | 2.163 ± 0.115 | 2.576 ± 0.050 | **1.766 ± 0.848** |
| gm_queen5_5_3.wcsp | 1029.941 ± 15.837 | 1261.372 ± 15.540 | 902.594 ± 6.560 | 559.981 ± 3.184 | **299.559 ± 4.443** |
| lm_batch_likelihood_brackets_4_4d | 18.608 ± 0.354 | 27.550 ± 0.472 | 19.692 ± 0.508 | 14.520 ± 0.288 | **8.916 ± 0.206** |
| lm_batch_likelihood_sentence_3_12d | 47.531 ± 1.956 | 70.464 ± 0.471 | 49.189 ± 1.014 | 21.826 ± 0.337 | **12.278 ± 0.390** |
| lm_batch_likelihood_sentence_4_4d | 19.200 ± 0.290 | 29.471 ± 0.211 | 20.898 ± 0.220 | 14.739 ± 0.136 | **7.683 ± 0.239** |
| str_matrix_chain_multiplication_100 | 12.248 ± 0.501 | 10.434 ± 0.255 | 17.375 ± 0.079 | **4.499 ± 0.215** | 9.565 ± 0.249 |
| str_mps_varying_inner_product_200 | 16.487 ± 0.413 | 15.898 ± 0.224 | 43.077 ± 2.688 | **9.137 ± 0.221** | 23.584 ± 0.582 |
| str_nw_mera_closed_120 | 1008.875 ± 14.248 | 1022.821 ± 18.389 | 1056.794 ± 8.086 | **121.802 ± 1.539** | 161.494 ± 11.927 |
| str_nw_mera_open_26 | 745.857 ± 6.209 | 764.324 ± 2.563 | 683.146 ± 3.200 | **118.559 ± 0.526** | 148.233 ± 27.533 |
| tensornetwork_permutation_focus_step409_316 | 266.028 ± 3.760 | 370.287 ± 11.134 | 255.883 ± 0.594 | 239.194 ± 3.249 | **106.623 ± 11.393** |
| tensornetwork_permutation_light_415 | 270.226 ± 5.720 | 311.545 ± 2.849 | 359.463 ± 3.500 | 193.905 ± 2.865 | **108.178 ± 8.119** |


## Threads: 4

- Timestamp: `20260524_163132`
- Source table: `data/results/results_t4_20260524_163132.md`

Logs:

- `data/results/tenferro_trace_t4_20260524_163132.log`
- `data/results/tenferro_eager_t4_20260524_163132.log`
- `data/results/libtorch_cpu_t4_20260524_163132.log`
- `data/results/pytorch_cpu_t4_20260524_163132.log`
- `data/results/jax_cpu_t4_20260524_163132.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 1.848 ± 0.402 | 1.608 ± 0.212 | 0.875 ± 0.274 | **0.125 ± 0.012** | 0.274 ± 0.063 |
| bin_elementwise_mul_2048x2048 | 3.788 ± 0.474 | 2.524 ± 0.545 | **0.896 ± 0.103** | 1.149 ± 0.153 | 1.502 ± 0.507 |
| bin_matmul_256 | 0.319 ± 0.024 | 0.274 ± 0.028 | 0.292 ± 0.027 | **0.113 ± 0.007** | 0.326 ± 0.053 |
| bin_outer_product_4096 | 12.164 ± 0.223 | 14.715 ± 0.522 | 1.459 ± 0.370 | 1.616 ± 0.298 | **1.263 ± 0.105** |
| gm_queen5_5_3.wcsp | 1757.097 ± 35.947 | 2616.964 ± 89.936 | 681.092 ± 27.489 | **521.011 ± 38.418** | 845.876 ± 11.418 |
| lm_batch_likelihood_brackets_4_4d | 16.185 ± 0.337 | 27.235 ± 0.685 | 16.620 ± 0.255 | 12.737 ± 0.593 | **9.165 ± 0.234** |
| lm_batch_likelihood_sentence_3_12d | 22.142 ± 0.618 | 51.680 ± 1.072 | 28.646 ± 0.304 | 20.768 ± 0.186 | **11.459 ± 0.250** |
| lm_batch_likelihood_sentence_4_4d | 14.429 ± 0.131 | 28.478 ± 0.314 | 16.190 ± 0.306 | 11.850 ± 0.457 | **9.098 ± 0.154** |
| str_matrix_chain_multiplication_100 | 8.823 ± 0.275 | 7.469 ± 0.257 | 12.857 ± 0.273 | **4.545 ± 0.202** | 9.602 ± 0.241 |
| str_mps_varying_inner_product_200 | 14.600 ± 0.233 | 17.559 ± 0.364 | 42.531 ± 0.308 | **8.138 ± 0.374** | 20.854 ± 0.228 |
| str_nw_mera_closed_120 | 335.806 ± 10.571 | 358.212 ± 1.867 | 301.570 ± 10.537 | **127.955 ± 2.745** | 165.042 ± 4.801 |
| str_nw_mera_open_26 | 287.083 ± 3.464 | 304.555 ± 1.739 | 188.146 ± 1.858 | **93.779 ± 20.994** | 129.097 ± 2.972 |
| tensornetwork_permutation_focus_step409_316 | 193.477 ± 1.476 | 291.831 ± 7.064 | 111.741 ± 6.953 | **101.720 ± 12.686** | 104.292 ± 6.992 |
| tensornetwork_permutation_light_415 | 197.062 ± 5.930 | 242.487 ± 1.850 | 175.491 ± 3.457 | **87.173 ± 15.398** | 109.187 ± 3.437 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.726 ± 0.029 | 0.654 ± 0.006 | 0.411 ± 0.005 | **0.129 ± 0.008** | 0.221 ± 0.051 |
| bin_elementwise_mul_2048x2048 | 3.628 ± 0.175 | 2.426 ± 0.072 | **0.618 ± 0.038** | 0.888 ± 0.111 | 1.464 ± 0.528 |
| bin_matmul_256 | 0.280 ± 0.016 | 0.249 ± 0.010 | 0.219 ± 0.013 | **0.111 ± 0.002** | 0.311 ± 0.042 |
| bin_outer_product_4096 | 12.047 ± 0.243 | 14.373 ± 0.119 | **1.136 ± 0.055** | 1.633 ± 0.173 | 1.193 ± 0.055 |
| gm_queen5_5_3.wcsp | 692.922 ± 4.415 | 922.899 ± 23.535 | 312.078 ± 16.667 | **243.575 ± 10.585** | 299.288 ± 7.666 |
| lm_batch_likelihood_brackets_4_4d | 16.780 ± 0.475 | 28.593 ± 1.050 | 15.438 ± 0.200 | 12.894 ± 0.489 | **9.124 ± 0.248** |
| lm_batch_likelihood_sentence_3_12d | 26.580 ± 0.378 | 51.026 ± 1.225 | 27.991 ± 0.427 | 20.209 ± 0.563 | **13.484 ± 1.232** |
| lm_batch_likelihood_sentence_4_4d | 16.325 ± 0.262 | 29.395 ± 0.611 | 15.762 ± 0.217 | 12.028 ± 0.483 | **8.498 ± 0.446** |
| str_matrix_chain_multiplication_100 | 8.776 ± 0.434 | 7.720 ± 0.365 | 12.828 ± 0.192 | **4.551 ± 0.281** | 9.937 ± 0.534 |
| str_mps_varying_inner_product_200 | 17.840 ± 0.414 | 18.160 ± 0.253 | 42.940 ± 0.570 | **10.230 ± 0.344** | 24.294 ± 0.745 |
| str_nw_mera_closed_120 | 311.978 ± 2.071 | 321.824 ± 2.900 | 310.811 ± 6.291 | **106.247 ± 1.093** | 153.836 ± 6.121 |
| str_nw_mera_open_26 | 286.476 ± 1.397 | 304.779 ± 2.620 | 186.837 ± 1.028 | **93.103 ± 1.221** | 133.975 ± 4.805 |
| tensornetwork_permutation_focus_step409_316 | 197.097 ± 2.752 | 293.549 ± 11.035 | 110.379 ± 2.461 | **102.908 ± 6.105** | 109.910 ± 7.328 |
| tensornetwork_permutation_light_415 | 199.119 ± 5.373 | 242.506 ± 1.965 | 175.924 ± 4.325 | **94.120 ± 6.628** | 110.615 ± 5.076 |


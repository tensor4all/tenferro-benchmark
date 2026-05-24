# Einsum Benchmark Results

Latest run: `./scripts/run_all.sh 4`.

This file collects the latest einsum benchmark table for each thread count under `data/results/`.

- tenferro-rs commit: `81ce9648ed5494d94756013a793273a9c69c649a`

## Threads: 1

- Timestamp: `20260524_154307`
- Source table: `data/results/results_t1_20260524_154307.md`

Logs:

- `data/results/tenferro_trace_t1_20260524_154307.log`
- `data/results/tenferro_eager_t1_20260524_154307.log`
- `data/results/libtorch_cpu_t1_20260524_154307.log`
- `data/results/pytorch_cpu_t1_20260524_154307.log`
- `data/results/jax_cpu_t1_20260524_154307.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 1.861 ± 0.452 | 1.805 ± 0.457 | 0.786 ± 0.028 | **0.124 ± 0.008** | 0.180 ± 0.022 |
| bin_elementwise_mul_2048x2048 | 1.473 ± 0.305 | 2.119 ± 0.039 | 1.243 ± 0.118 | **1.141 ± 0.101** | 1.421 ± 0.666 |
| bin_matmul_256 | 0.784 ± 0.037 | 0.799 ± 0.028 | 0.785 ± 0.039 | **0.111 ± 0.001** | 0.311 ± 0.033 |
| bin_outer_product_4096 | 12.020 ± 0.262 | 11.872 ± 0.278 | 2.123 ± 0.042 | 2.573 ± 0.101 | **1.220 ± 0.047** |
| gm_queen5_5_3.wcsp | 1895.620 ± 64.990 | 2734.293 ± 57.578 | 1980.145 ± 106.769 | 1344.222 ± 31.362 | **841.227 ± 7.579** |
| lm_batch_likelihood_brackets_4_4d | 17.728 ± 0.507 | 27.204 ± 2.473 | 22.352 ± 0.154 | 12.779 ± 0.983 | **9.048 ± 0.311** |
| lm_batch_likelihood_sentence_3_12d | 42.209 ± 0.354 | 71.790 ± 4.019 | 48.826 ± 0.950 | 19.954 ± 0.765 | **11.640 ± 0.479** |
| lm_batch_likelihood_sentence_4_4d | 16.653 ± 0.254 | 29.584 ± 0.422 | 21.727 ± 0.235 | 13.003 ± 0.226 | **8.955 ± 0.075** |
| str_matrix_chain_multiplication_100 | 9.109 ± 0.332 | 10.738 ± 0.775 | 17.486 ± 0.184 | **4.472 ± 0.429** | 9.487 ± 0.231 |
| str_mps_varying_inner_product_200 | 11.387 ± 0.306 | 16.518 ± 0.371 | 39.742 ± 0.153 | **8.007 ± 0.331** | 20.615 ± 0.220 |
| str_nw_mera_closed_120 | 1038.560 ± 15.219 | 1064.923 ± 13.540 | 1008.208 ± 21.305 | 166.212 ± 6.376 | **162.013 ± 6.432** |
| str_nw_mera_open_26 | 746.236 ± 14.153 | 757.163 ± 7.408 | 678.807 ± 9.423 | **116.414 ± 2.116** | 127.960 ± 4.732 |
| tensornetwork_permutation_focus_step409_316 | 269.848 ± 16.068 | 393.473 ± 4.677 | 255.850 ± 1.639 | 238.180 ± 10.141 | **110.861 ± 8.804** |
| tensornetwork_permutation_light_415 | 267.807 ± 3.828 | 314.061 ± 2.858 | 360.103 ± 5.197 | 193.843 ± 3.374 | **125.622 ± 15.015** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.595 ± 0.024 | 0.658 ± 0.015 | 0.414 ± 0.010 | **0.145 ± 0.038** | 0.323 ± 0.073 |
| bin_elementwise_mul_2048x2048 | 1.149 ± 0.019 | 1.729 ± 0.151 | **1.137 ± 0.026** | 1.226 ± 0.196 | 1.260 ± 0.262 |
| bin_matmul_256 | 0.663 ± 0.026 | 0.701 ± 0.092 | 0.676 ± 0.004 | **0.110 ± 0.001** | 0.333 ± 0.029 |
| bin_outer_product_4096 | 11.972 ± 0.100 | 11.944 ± 0.211 | 2.113 ± 0.060 | 2.553 ± 0.051 | **1.645 ± 0.312** |
| gm_queen5_5_3.wcsp | 1038.855 ± 27.987 | 1294.446 ± 34.416 | 904.686 ± 7.409 | 557.425 ± 5.303 | **292.244 ± 9.992** |
| lm_batch_likelihood_brackets_4_4d | 17.764 ± 0.286 | 29.217 ± 0.599 | 19.269 ± 0.104 | 13.624 ± 1.279 | **8.928 ± 0.424** |
| lm_batch_likelihood_sentence_3_12d | 45.731 ± 0.702 | 70.750 ± 2.349 | 49.368 ± 1.473 | 19.565 ± 0.775 | **12.317 ± 0.313** |
| lm_batch_likelihood_sentence_4_4d | 18.279 ± 0.220 | 29.068 ± 0.459 | 21.282 ± 0.528 | 13.578 ± 0.372 | **7.985 ± 0.229** |
| str_matrix_chain_multiplication_100 | 9.120 ± 0.164 | 10.276 ± 0.220 | 17.342 ± 0.216 | **4.787 ± 0.397** | 9.296 ± 0.166 |
| str_mps_varying_inner_product_200 | 12.307 ± 0.127 | 16.239 ± 0.195 | 40.689 ± 0.860 | **9.398 ± 0.210** | 22.931 ± 0.683 |
| str_nw_mera_closed_120 | 1007.154 ± 9.114 | 1024.310 ± 25.515 | 1066.063 ± 19.666 | **123.049 ± 4.409** | 151.554 ± 4.288 |
| str_nw_mera_open_26 | 746.216 ± 15.760 | 760.801 ± 10.397 | 687.813 ± 14.055 | **118.145 ± 0.452** | 132.245 ± 2.444 |
| tensornetwork_permutation_focus_step409_316 | 265.471 ± 5.679 | 390.285 ± 11.568 | 257.603 ± 4.984 | 235.251 ± 9.465 | **104.681 ± 3.616** |
| tensornetwork_permutation_light_415 | 265.115 ± 1.467 | 315.322 ± 6.243 | 362.748 ± 6.554 | 186.963 ± 1.628 | **115.618 ± 7.533** |


## Threads: 4

- Timestamp: `20260524_155444`
- Source table: `data/results/results_t4_20260524_155444.md`

Logs:

- `data/results/tenferro_trace_t4_20260524_155444.log`
- `data/results/tenferro_eager_t4_20260524_155444.log`
- `data/results/libtorch_cpu_t4_20260524_155444.log`
- `data/results/pytorch_cpu_t4_20260524_155444.log`
- `data/results/jax_cpu_t4_20260524_155444.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 1.865 ± 0.292 | 1.547 ± 0.315 | 0.784 ± 0.042 | **0.136 ± 0.008** | 0.239 ± 0.070 |
| bin_elementwise_mul_2048x2048 | 1.536 ± 0.231 | 1.696 ± 0.076 | **0.841 ± 0.169** | 1.158 ± 0.096 | 1.510 ± 0.472 |
| bin_matmul_256 | 0.248 ± 0.038 | 0.251 ± 0.010 | 0.278 ± 0.036 | **0.111 ± 0.004** | 0.312 ± 0.036 |
| bin_outer_product_4096 | 12.472 ± 0.375 | 11.733 ± 0.041 | 1.171 ± 0.050 | **1.119 ± 0.031** | 1.231 ± 0.084 |
| gm_queen5_5_3.wcsp | 1747.970 ± 49.514 | 2548.380 ± 62.282 | 698.778 ± 91.455 | **521.797 ± 7.611** | 846.911 ± 11.623 |
| lm_batch_likelihood_brackets_4_4d | 15.491 ± 0.496 | 26.930 ± 0.761 | 17.255 ± 0.266 | 14.315 ± 0.366 | **9.195 ± 0.125** |
| lm_batch_likelihood_sentence_3_12d | 22.874 ± 0.893 | 50.959 ± 0.478 | 28.837 ± 0.457 | 23.479 ± 0.208 | **11.426 ± 0.287** |
| lm_batch_likelihood_sentence_4_4d | 14.039 ± 0.432 | 27.957 ± 0.705 | 16.067 ± 0.156 | 13.740 ± 0.394 | **8.977 ± 0.200** |
| str_matrix_chain_multiplication_100 | 7.130 ± 0.910 | 7.592 ± 0.746 | 13.027 ± 0.540 | **4.332 ± 0.164** | 9.511 ± 0.182 |
| str_mps_varying_inner_product_200 | 12.466 ± 0.523 | 17.456 ± 0.289 | 43.339 ± 0.729 | **8.351 ± 0.168** | 20.875 ± 0.641 |
| str_nw_mera_closed_120 | 330.287 ± 9.548 | 354.332 ± 2.960 | 300.968 ± 18.000 | **126.932 ± 6.478** | 164.257 ± 2.034 |
| str_nw_mera_open_26 | 289.535 ± 2.735 | 308.521 ± 17.284 | 190.900 ± 8.640 | **91.269 ± 0.356** | 129.233 ± 3.127 |
| tensornetwork_permutation_focus_step409_316 | 195.567 ± 2.280 | 319.072 ± 1.157 | 111.182 ± 2.355 | **98.429 ± 10.575** | 103.932 ± 4.674 |
| tensornetwork_permutation_light_415 | 197.257 ± 2.614 | 245.874 ± 9.938 | 178.272 ± 5.719 | **97.827 ± 5.773** | 109.410 ± 3.833 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.612 ± 0.013 | 0.650 ± 0.041 | 0.415 ± 0.019 | **0.136 ± 0.006** | 0.254 ± 0.073 |
| bin_elementwise_mul_2048x2048 | 1.156 ± 0.031 | 1.731 ± 0.061 | **0.624 ± 0.036** | 0.905 ± 0.056 | 1.684 ± 0.587 |
| bin_matmul_256 | 0.214 ± 0.011 | 0.239 ± 0.023 | 0.233 ± 0.032 | **0.111 ± 0.001** | 0.312 ± 0.023 |
| bin_outer_product_4096 | 12.027 ± 0.256 | 12.171 ± 0.567 | **1.131 ± 0.030** | 1.625 ± 0.473 | 1.875 ± 0.841 |
| gm_queen5_5_3.wcsp | 690.548 ± 10.293 | 917.864 ± 6.171 | 326.494 ± 26.651 | **259.720 ± 16.380** | 290.624 ± 6.116 |
| lm_batch_likelihood_brackets_4_4d | 15.735 ± 0.249 | 27.743 ± 0.630 | 15.611 ± 0.322 | 14.478 ± 0.392 | **8.959 ± 0.218** |
| lm_batch_likelihood_sentence_3_12d | 25.947 ± 0.555 | 53.327 ± 0.986 | 27.970 ± 0.187 | 23.017 ± 0.552 | **12.328 ± 0.197** |
| lm_batch_likelihood_sentence_4_4d | 15.299 ± 0.457 | 29.393 ± 0.265 | 15.999 ± 0.294 | 14.356 ± 0.778 | **7.887 ± 0.302** |
| str_matrix_chain_multiplication_100 | 5.896 ± 0.354 | 7.571 ± 0.525 | 12.822 ± 0.231 | **4.604 ± 0.343** | 9.537 ± 0.211 |
| str_mps_varying_inner_product_200 | 13.085 ± 0.427 | 18.338 ± 0.280 | 44.483 ± 1.996 | **10.915 ± 0.305** | 22.892 ± 0.532 |
| str_nw_mera_closed_120 | 317.299 ± 10.994 | 318.528 ± 2.320 | 315.685 ± 17.721 | **105.282 ± 1.549** | 151.083 ± 2.267 |
| str_nw_mera_open_26 | 284.716 ± 2.418 | 314.206 ± 25.617 | 190.380 ± 2.254 | **95.007 ± 2.139** | 134.560 ± 4.473 |
| tensornetwork_permutation_focus_step409_316 | 198.755 ± 10.713 | 325.227 ± 10.953 | 110.959 ± 1.204 | **95.446 ± 13.965** | 102.812 ± 3.646 |
| tensornetwork_permutation_light_415 | 196.794 ± 5.758 | 245.164 ± 1.775 | 178.655 ± 2.458 | **90.314 ± 8.316** | 108.545 ± 2.824 |


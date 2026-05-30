# Einsum Benchmark Results

Latest run: `./scripts/run_all.sh 4`.

This file collects the latest einsum benchmark table for each thread count under `data/results/`.

- tenferro-rs commit: `fa722375c8662b5532a6f875a6bef3494ace40b5`

## Threads: 1

- Timestamp: `20260530_042522`
- Source table: `data/results/results_t1_20260530_042522.md`

Logs:

- `data/results/tenferro_trace_t1_20260530_042522.log`
- `data/results/tenferro_eager_t1_20260530_042522.log`
- `data/results/libtorch_cpu_t1_20260530_042522.log`
- `data/results/pytorch_cpu_t1_20260530_042522.log`
- `data/results/jax_cpu_t1_20260530_042522.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 4.084 ± 0.153 | 2.985 ± 0.013 | 0.899 ± 0.003 | **0.662 ± 0.022** | 1.040 ± 0.045 |
| bin_elementwise_mul_2048x2048 | 115.138 ± 2.426 | 88.158 ± 2.160 | 21.262 ± 0.053 | 23.273 ± 0.039 | **14.126 ± 0.187** |
| bin_matmul_256 | 2.157 ± 0.033 | 1.647 ± 0.009 | **1.074 ± 0.001** | 1.083 ± 0.008 | 1.198 ± 0.044 |
| bin_outer_product_4096 | 241.635 ± 4.902 | 328.541 ± 6.319 | 63.164 ± 0.074 | 71.339 ± 0.408 | **38.563 ± 2.584** |
| gm_queen5_5_3.wcsp | 12576.172 ± 20.940 | 19016.525 ± 127.123 | 9313.277 ± 76.146 | **7479.980 ± 26.226** | - |
| lm_batch_likelihood_brackets_4_4d | 43.436 ± 1.189 | 95.981 ± 1.742 | 66.917 ± 0.267 | 43.530 ± 0.182 | **42.842 ± 0.958** |
| lm_batch_likelihood_sentence_3_12d | 83.683 ± 3.056 | 275.219 ± 1.846 | 134.766 ± 0.849 | 101.233 ± 0.543 | **54.826 ± 5.437** |
| lm_batch_likelihood_sentence_4_4d | **38.747 ± 1.214** | 75.563 ± 1.191 | 65.044 ± 0.215 | 46.522 ± 0.464 | 42.044 ± 1.614 |
| str_matrix_chain_multiplication_100 | 48.223 ± 0.736 | 33.471 ± 0.627 | 35.284 ± 0.269 | **26.532 ± 0.086** | 43.609 ± 2.522 |
| str_mps_varying_inner_product_200 | 45.279 ± 1.612 | 46.821 ± 0.611 | 85.337 ± 0.321 | **36.378 ± 0.762** | 93.413 ± 1.039 |
| str_nw_mera_closed_120 | 2613.991 ± 7.665 | 2995.838 ± 10.763 | 2257.169 ± 3.495 | 2219.787 ± 13.428 | **622.084 ± 31.452** |
| str_nw_mera_open_26 | 2271.418 ± 9.741 | 2703.557 ± 3.433 | 1485.944 ± 8.937 | 1471.294 ± 2.329 | **592.279 ± 61.661** |
| tensornetwork_permutation_focus_step409_316 | 752.289 ± 6.395 | 1368.221 ± 4.687 | 811.400 ± 2.173 | 1015.309 ± 3.196 | **529.530 ± 15.527** |
| tensornetwork_permutation_light_415 | 757.575 ± 5.890 | 1042.957 ± 2.339 | 1063.055 ± 1.217 | 986.796 ± 4.425 | **558.860 ± 44.889** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 2.668 ± 0.083 | 2.369 ± 0.051 | 0.875 ± 0.003 | **0.654 ± 0.009** | 1.148 ± 0.176 |
| bin_elementwise_mul_2048x2048 | 102.468 ± 1.004 | 85.321 ± 0.291 | 21.014 ± 0.078 | 23.713 ± 0.092 | **12.773 ± 3.497** |
| bin_matmul_256 | 1.441 ± 0.023 | 1.347 ± 0.042 | 1.054 ± 0.003 | 1.086 ± 0.004 | **0.959 ± 0.154** |
| bin_outer_product_4096 | 237.468 ± 1.250 | 316.359 ± 2.850 | 64.159 ± 0.185 | 72.295 ± 0.251 | **47.916 ± 1.083** |
| gm_queen5_5_3.wcsp | 4691.327 ± 37.045 | 6371.828 ± 10.441 | **2999.934 ± 3.421** | 3225.882 ± 13.212 | - |
| lm_batch_likelihood_brackets_4_4d | **37.303 ± 2.277** | 65.930 ± 0.482 | 60.501 ± 0.375 | 48.167 ± 0.509 | 43.964 ± 1.587 |
| lm_batch_likelihood_sentence_3_12d | 94.796 ± 4.462 | 253.104 ± 1.403 | 104.793 ± 1.266 | 70.933 ± 0.326 | **59.686 ± 9.463** |
| lm_batch_likelihood_sentence_4_4d | 39.398 ± 1.176 | 74.988 ± 1.210 | 60.368 ± 0.297 | 49.322 ± 0.180 | **37.475 ± 1.260** |
| str_matrix_chain_multiplication_100 | 48.701 ± 1.333 | 33.321 ± 0.442 | 35.721 ± 0.364 | **28.230 ± 0.071** | 43.295 ± 2.931 |
| str_mps_varying_inner_product_200 | 47.653 ± 0.900 | 49.384 ± 0.254 | 90.746 ± 0.336 | **46.319 ± 0.621** | 103.797 ± 1.442 |
| str_nw_mera_closed_120 | 2322.322 ± 17.072 | 2446.370 ± 3.368 | 2336.123 ± 2.403 | 1773.204 ± 5.446 | **472.079 ± 30.042** |
| str_nw_mera_open_26 | 2282.913 ± 8.065 | 2665.165 ± 5.462 | 1498.553 ± 6.648 | 1496.002 ± 2.676 | **604.755 ± 91.539** |
| tensornetwork_permutation_focus_step409_316 | 718.018 ± 6.753 | 1331.612 ± 4.540 | 808.736 ± 2.530 | 1017.415 ± 5.358 | **507.377 ± 13.257** |
| tensornetwork_permutation_light_415 | 719.958 ± 7.572 | 1007.558 ± 6.164 | 1062.120 ± 1.774 | 987.742 ± 2.173 | **542.761 ± 16.839** |


## Threads: 4

- Timestamp: `20260530_050719`
- Source table: `data/results/results_t4_20260530_050719.md`

Logs:

- `data/results/tenferro_trace_t4_20260530_050719.log`
- `data/results/tenferro_eager_t4_20260530_050719.log`
- `data/results/libtorch_cpu_t4_20260530_050719.log`
- `data/results/pytorch_cpu_t4_20260530_050719.log`
- `data/results/jax_cpu_t4_20260530_050719.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 6.291 ± 0.282 | 3.335 ± 0.077 | 0.890 ± 0.021 | **0.319 ± 0.021** | 1.238 ± 0.230 |
| bin_elementwise_mul_2048x2048 | 111.792 ± 2.260 | 85.310 ± 0.446 | **7.689 ± 0.123** | 8.630 ± 0.758 | 13.966 ± 0.937 |
| bin_matmul_256 | 1.908 ± 0.142 | 1.391 ± 0.058 | **0.392 ± 0.052** | 0.422 ± 0.022 | 1.103 ± 0.059 |
| bin_outer_product_4096 | 231.349 ± 2.320 | 314.549 ± 1.947 | **19.008 ± 0.360** | 28.008 ± 0.615 | 45.580 ± 1.044 |
| gm_queen5_5_3.wcsp | 11886.099 ± 82.524 | 18609.818 ± 87.356 | 3460.992 ± 22.763 | **2558.145 ± 35.061** | - |
| lm_batch_likelihood_brackets_4_4d | 45.150 ± 0.299 | 101.083 ± 4.045 | 41.908 ± 0.252 | **19.227 ± 0.131** | 43.260 ± 1.880 |
| lm_batch_likelihood_sentence_3_12d | 101.051 ± 1.065 | 232.287 ± 6.139 | 90.421 ± 1.144 | **23.660 ± 0.091** | 57.062 ± 2.788 |
| lm_batch_likelihood_sentence_4_4d | 54.637 ± 5.290 | 81.905 ± 17.751 | 42.965 ± 0.192 | **20.033 ± 0.191** | 41.653 ± 0.976 |
| str_matrix_chain_multiplication_100 | 54.154 ± 3.086 | 61.322 ± 25.283 | 24.830 ± 0.150 | **14.353 ± 0.244** | 46.342 ± 2.239 |
| str_mps_varying_inner_product_200 | 87.245 ± 1.652 | 131.894 ± 0.988 | 88.409 ± 0.386 | **31.332 ± 0.243** | 92.467 ± 1.566 |
| str_nw_mera_closed_120 | 1522.157 ± 4.250 | 2048.930 ± 21.986 | 937.599 ± 14.441 | 749.905 ± 2.487 | **631.926 ± 43.107** |
| str_nw_mera_open_26 | 1742.110 ± 231.092 | 2300.714 ± 19.254 | 784.209 ± 5.577 | **506.221 ± 5.851** | 580.462 ± 37.953 |
| tensornetwork_permutation_focus_step409_316 | 679.812 ± 4.714 | 1343.566 ± 5.854 | 373.097 ± 13.693 | **341.849 ± 4.101** | 486.924 ± 9.678 |
| tensornetwork_permutation_light_415 | 681.329 ± 6.613 | 1013.995 ± 6.139 | 498.139 ± 2.492 | **348.518 ± 2.983** | 549.142 ± 21.982 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 3.082 ± 0.042 | 2.929 ± 0.086 | 0.911 ± 0.021 | **0.277 ± 0.022** | 1.285 ± 0.128 |
| bin_elementwise_mul_2048x2048 | 63.565 ± 0.189 | 66.633 ± 0.752 | 7.864 ± 0.273 | 9.697 ± 0.144 | **6.222 ± 0.296** |
| bin_matmul_256 | 1.191 ± 0.087 | 1.143 ± 0.154 | **0.381 ± 0.027** | 0.413 ± 0.020 | 1.117 ± 0.090 |
| bin_outer_product_4096 | 232.597 ± 1.893 | 313.403 ± 0.515 | **22.479 ± 4.170** | 30.362 ± 0.301 | 46.872 ± 0.896 |
| gm_queen5_5_3.wcsp | 4032.358 ± 19.923 | 5854.341 ± 120.146 | 1268.799 ± 16.483 | **1117.805 ± 11.020** | - |
| lm_batch_likelihood_brackets_4_4d | 41.881 ± 0.681 | 68.604 ± 0.925 | 39.958 ± 0.263 | **19.820 ± 0.220** | 43.823 ± 0.996 |
| lm_batch_likelihood_sentence_3_12d | 112.238 ± 3.095 | 230.600 ± 2.327 | 69.971 ± 0.811 | **26.059 ± 0.358** | 51.945 ± 1.235 |
| lm_batch_likelihood_sentence_4_4d | 56.270 ± 0.589 | 77.740 ± 1.778 | 41.446 ± 0.216 | **20.328 ± 0.314** | 36.999 ± 1.003 |
| str_matrix_chain_multiplication_100 | 53.642 ± 1.794 | 62.980 ± 25.556 | 24.883 ± 0.169 | **14.292 ± 0.188** | 44.894 ± 1.983 |
| str_mps_varying_inner_product_200 | 88.297 ± 2.793 | 134.434 ± 4.285 | 92.585 ± 0.327 | **41.034 ± 0.429** | 104.871 ± 3.668 |
| str_nw_mera_closed_120 | 1274.954 ± 108.672 | 1524.732 ± 27.475 | 831.207 ± 18.109 | 607.524 ± 9.604 | **475.567 ± 41.585** |
| str_nw_mera_open_26 | 1584.912 ± 236.993 | 2291.624 ± 89.753 | 769.097 ± 1.885 | **520.975 ± 4.229** | 602.099 ± 54.014 |
| tensornetwork_permutation_focus_step409_316 | 674.401 ± 7.839 | 1346.249 ± 8.493 | 372.357 ± 2.692 | **340.962 ± 2.725** | 505.161 ± 11.491 |
| tensornetwork_permutation_light_415 | 675.857 ± 6.714 | 1013.258 ± 3.784 | 516.625 ± 3.793 | **344.049 ± 4.339** | 543.412 ± 20.452 |


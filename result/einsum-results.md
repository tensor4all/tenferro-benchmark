# Einsum Benchmark Results

Latest run: `./scripts/run_all.sh 4`.

This file collects the latest einsum benchmark table for each thread count under `data/results/`.

- tenferro-rs commit: `81ce9648ed5494d94756013a793273a9c69c649a`

## Threads: 1

- Timestamp: `20260524_144633`
- Source table: `data/results/results_t1_20260524_144633.md`

Logs:

- `data/results/tenferro_trace_t1_20260524_144633.log`
- `data/results/tenferro_eager_t1_20260524_144633.log`
- `data/results/libtorch_cpu_t1_20260524_144633.log`
- `data/results/pytorch_cpu_t1_20260524_144633.log`
- `data/results/jax_cpu_t1_20260524_144633.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.948 ± 0.070 | 1.042 ± 0.054 | 0.740 ± 0.018 | **0.130 ± 0.005** | 0.249 ± 0.050 |
| bin_elementwise_mul_2048x2048 | **1.135 ± 0.005** | 1.758 ± 0.067 | 1.173 ± 0.113 | 1.214 ± 0.115 | 1.523 ± 0.383 |
| bin_matmul_256 | 0.785 ± 0.009 | 0.782 ± 0.047 | 0.769 ± 0.027 | **0.111 ± 0.002** | 0.305 ± 0.044 |
| bin_outer_product_4096 | 11.746 ± 1.081 | 12.289 ± 0.302 | 2.095 ± 0.076 | 2.535 ± 0.061 | **1.248 ± 0.048** |
| gm_queen5_5_3.wcsp | 1918.588 ± 288.828 | 2785.996 ± 466.242 | 2029.179 ± 108.279 | 1324.260 ± 10.650 | **866.485 ± 6.702** |
| lm_batch_likelihood_brackets_4_4d | 17.682 ± 0.345 | 27.024 ± 0.304 | 22.855 ± 2.362 | 13.592 ± 0.236 | **9.132 ± 0.511** |
| lm_batch_likelihood_sentence_3_12d | 42.604 ± 0.694 | 72.734 ± 0.603 | 48.786 ± 0.735 | 22.306 ± 0.496 | **11.998 ± 0.293** |
| lm_batch_likelihood_sentence_4_4d | 16.515 ± 0.258 | 29.758 ± 0.669 | 21.583 ± 0.250 | 13.870 ± 0.194 | **8.729 ± 0.159** |
| str_matrix_chain_multiplication_100 | 9.039 ± 0.373 | 10.911 ± 0.357 | 17.709 ± 0.565 | **4.436 ± 0.201** | 9.395 ± 0.496 |
| str_mps_varying_inner_product_200 | 11.163 ± 0.137 | 16.291 ± 0.325 | 39.469 ± 0.390 | **7.778 ± 0.188** | 20.598 ± 0.134 |
| str_nw_mera_closed_120 | 1022.746 ± 8.123 | 1060.510 ± 6.630 | 1016.134 ± 13.207 | **162.466 ± 1.411** | 177.174 ± 5.330 |
| str_nw_mera_open_26 | 863.909 ± 508.554 | 756.135 ± 4.632 | 684.554 ± 9.962 | **114.749 ± 2.015** | 138.185 ± 5.083 |
| tensornetwork_permutation_focus_step409_316 | 268.015 ± 7.455 | 385.950 ± 18.121 | 256.179 ± 8.993 | 239.883 ± 3.222 | **113.312 ± 9.207** |
| tensornetwork_permutation_light_415 | 268.308 ± 3.881 | 319.305 ± 6.811 | 373.605 ± 21.446 | 193.277 ± 3.957 | **182.582 ± 29.605** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.596 ± 0.006 | 0.652 ± 0.022 | 0.415 ± 0.012 | **0.131 ± 0.008** | 1.345 ± 0.969 |
| bin_elementwise_mul_2048x2048 | 1.177 ± 0.060 | 1.924 ± 0.102 | **1.128 ± 0.043** | 1.151 ± 0.032 | 2.760 ± 1.056 |
| bin_matmul_256 | 0.666 ± 0.012 | 0.683 ± 0.027 | 0.681 ± 0.026 | **0.110 ± 0.001** | 1.376 ± 0.460 |
| bin_outer_product_4096 | 11.886 ± 0.162 | 12.060 ± 0.389 | **2.126 ± 0.040** | 2.529 ± 0.085 | 2.735 ± 0.434 |
| gm_queen5_5_3.wcsp | 1018.301 ± 21.246 | 1278.641 ± 48.408 | 919.309 ± 54.530 | 558.183 ± 11.633 | **455.942 ± 54.917** |
| lm_batch_likelihood_brackets_4_4d | 18.317 ± 0.735 | 27.784 ± 0.308 | 19.155 ± 0.169 | 14.351 ± 0.213 | **8.938 ± 0.109** |
| lm_batch_likelihood_sentence_3_12d | 45.587 ± 1.330 | 70.098 ± 0.415 | 48.415 ± 1.245 | 22.044 ± 0.557 | **14.277 ± 1.503** |
| lm_batch_likelihood_sentence_4_4d | 18.145 ± 0.354 | 29.289 ± 0.330 | 21.122 ± 0.288 | 14.694 ± 0.180 | **8.436 ± 0.490** |
| str_matrix_chain_multiplication_100 | 9.344 ± 0.369 | 10.648 ± 0.195 | 17.533 ± 0.236 | **4.515 ± 0.464** | 9.935 ± 0.403 |
| str_mps_varying_inner_product_200 | 12.558 ± 0.401 | 16.490 ± 0.446 | 40.209 ± 0.215 | **9.200 ± 0.288** | 23.525 ± 1.328 |
| str_nw_mera_closed_120 | 1003.625 ± 8.150 | 1029.529 ± 14.819 | 1060.297 ± 20.003 | **121.209 ± 1.826** | 166.792 ± 16.018 |
| str_nw_mera_open_26 | 746.007 ± 5.078 | 762.133 ± 13.492 | 680.595 ± 3.709 | **119.201 ± 0.946** | 151.668 ± 8.836 |
| tensornetwork_permutation_focus_step409_316 | 265.710 ± 5.979 | 398.467 ± 4.008 | 262.935 ± 2.565 | 233.118 ± 3.227 | **111.115 ± 10.697** |
| tensornetwork_permutation_light_415 | 268.399 ± 2.892 | 314.900 ± 1.174 | 365.512 ± 2.126 | 191.442 ± 2.225 | **114.554 ± 7.225** |


## Threads: 4

- Timestamp: `20260524_145716`
- Source table: `data/results/results_t4_20260524_145716.md`

Logs:

- `data/results/tenferro_trace_t4_20260524_145716.log`
- `data/results/tenferro_eager_t4_20260524_145716.log`
- `data/results/libtorch_cpu_t4_20260524_145716.log`
- `data/results/pytorch_cpu_t4_20260524_145716.log`
- `data/results/jax_cpu_t4_20260524_145716.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 1.615 ± 0.430 | 1.738 ± 0.351 | 0.804 ± 0.024 | **0.137 ± 0.015** | 0.277 ± 0.094 |
| bin_elementwise_mul_2048x2048 | 1.250 ± 0.068 | 1.686 ± 0.053 | **0.871 ± 0.141** | 1.147 ± 0.143 | 1.339 ± 0.274 |
| bin_matmul_256 | 0.237 ± 0.018 | 0.259 ± 0.011 | 0.249 ± 0.004 | **0.111 ± 0.001** | 0.346 ± 0.037 |
| bin_outer_product_4096 | 11.886 ± 0.707 | 11.889 ± 0.205 | 1.603 ± 0.212 | **1.183 ± 0.506** | 1.496 ± 0.379 |
| gm_queen5_5_3.wcsp | 1739.245 ± 68.005 | 2552.257 ± 27.758 | 756.395 ± 32.237 | **530.150 ± 16.674** | 868.096 ± 12.182 |
| lm_batch_likelihood_brackets_4_4d | 15.609 ± 0.270 | 27.096 ± 0.438 | 16.826 ± 0.533 | 13.798 ± 1.233 | **9.032 ± 0.396** |
| lm_batch_likelihood_sentence_3_12d | 21.825 ± 0.866 | 51.906 ± 1.283 | 78.952 ± 53.353 | 22.726 ± 0.513 | **11.903 ± 0.567** |
| lm_batch_likelihood_sentence_4_4d | 14.026 ± 0.283 | 28.499 ± 0.813 | 16.297 ± 0.199 | 13.742 ± 0.297 | **9.116 ± 0.273** |
| str_matrix_chain_multiplication_100 | 6.299 ± 0.568 | 7.629 ± 0.577 | 12.937 ± 0.226 | **4.843 ± 0.202** | 9.449 ± 0.197 |
| str_mps_varying_inner_product_200 | 11.974 ± 0.388 | 17.331 ± 0.440 | 42.030 ± 0.608 | **8.270 ± 0.112** | 21.087 ± 0.489 |
| str_nw_mera_closed_120 | 326.890 ± 3.284 | 356.067 ± 2.622 | 314.023 ± 4.817 | **128.466 ± 3.304** | 172.898 ± 24.656 |
| str_nw_mera_open_26 | 294.947 ± 19.648 | 296.193 ± 2.480 | 188.616 ± 3.356 | **92.247 ± 1.357** | 129.132 ± 4.370 |
| tensornetwork_permutation_focus_step409_316 | 196.513 ± 2.896 | 322.154 ± 3.839 | 110.445 ± 1.621 | **91.134 ± 2.824** | 106.207 ± 5.109 |
| tensornetwork_permutation_light_415 | 196.136 ± 3.280 | 241.834 ± 1.780 | 175.852 ± 7.401 | **94.414 ± 7.722** | 113.332 ± 1.827 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.605 ± 0.011 | 0.638 ± 0.011 | 0.422 ± 0.024 | **0.129 ± 0.004** | 0.256 ± 0.062 |
| bin_elementwise_mul_2048x2048 | 1.162 ± 0.095 | 1.655 ± 0.021 | **0.610 ± 0.023** | 0.886 ± 0.036 | 1.416 ± 0.185 |
| bin_matmul_256 | 0.221 ± 0.007 | 0.226 ± 0.035 | 0.229 ± 0.014 | **0.112 ± 0.016** | 0.338 ± 0.028 |
| bin_outer_product_4096 | 11.773 ± 0.114 | 11.775 ± 0.165 | **1.135 ± 0.074** | 1.140 ± 0.062 | 1.241 ± 0.042 |
| gm_queen5_5_3.wcsp | 685.228 ± 9.587 | 913.132 ± 11.140 | 313.685 ± 9.261 | **259.601 ± 13.744** | 301.208 ± 11.992 |
| lm_batch_likelihood_brackets_4_4d | 15.732 ± 0.311 | 26.054 ± 0.749 | 15.678 ± 0.184 | 14.212 ± 0.475 | **9.025 ± 0.310** |
| lm_batch_likelihood_sentence_3_12d | 25.808 ± 0.512 | 51.026 ± 1.721 | 28.014 ± 0.378 | 21.683 ± 0.316 | **12.650 ± 0.589** |
| lm_batch_likelihood_sentence_4_4d | 15.385 ± 0.223 | 27.500 ± 0.642 | 15.953 ± 0.336 | 13.743 ± 0.996 | **7.998 ± 0.301** |
| str_matrix_chain_multiplication_100 | 6.101 ± 0.251 | 7.335 ± 0.290 | 12.800 ± 0.410 | **4.592 ± 0.166** | 9.590 ± 0.218 |
| str_mps_varying_inner_product_200 | 13.986 ± 0.516 | 16.158 ± 0.537 | 43.170 ± 0.406 | **11.447 ± 0.266** | 23.403 ± 0.681 |
| str_nw_mera_closed_120 | 309.690 ± 2.161 | 328.760 ± 5.703 | 313.139 ± 5.460 | **107.455 ± 1.743** | 158.225 ± 7.874 |
| str_nw_mera_open_26 | 288.009 ± 10.110 | 310.692 ± 10.938 | 189.616 ± 1.483 | **94.196 ± 1.020** | 137.622 ± 7.030 |
| tensornetwork_permutation_focus_step409_316 | 196.704 ± 4.402 | 332.799 ± 9.672 | 111.647 ± 1.968 | **97.489 ± 14.931** | 104.780 ± 5.059 |
| tensornetwork_permutation_light_415 | 198.644 ± 4.630 | 253.902 ± 7.426 | 175.479 ± 12.676 | **93.792 ± 8.265** | 111.544 ± 5.523 |

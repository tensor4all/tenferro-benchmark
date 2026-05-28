# Einsum Benchmark Results

Latest run: `./scripts/run_all.sh 4`.

This file collects the latest einsum benchmark table for each thread count under `data/results/`.

- tenferro-rs commit: `948c5791c4a102a14b2005b5d1657b85dbba797c`

## Threads: 1

- Timestamp: `20260528_072400`
- Source table: `data/results/results_t1_20260528_072400.md`

Logs:

- `data/results/tenferro_trace_t1_20260528_072400.log`
- `data/results/tenferro_eager_t1_20260528_072400.log`
- `data/results/libtorch_cpu_t1_20260528_072400.log`
- `data/results/pytorch_cpu_t1_20260528_072400.log`
- `data/results/jax_cpu_t1_20260528_072400.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 1.259 ± 0.079 | 1.069 ± 0.775 | 0.455 ± 0.029 | **0.371 ± 0.033** | 0.650 ± 0.052 |
| bin_elementwise_mul_2048x2048 | 22.814 ± 1.612 | 17.368 ± 0.621 | 5.573 ± 0.186 | 5.673 ± 0.308 | **5.337 ± 0.196** |
| bin_matmul_256 | 0.725 ± 0.432 | 0.816 ± 0.028 | 0.619 ± 0.033 | **0.618 ± 0.031** | 0.992 ± 0.104 |
| bin_outer_product_4096 | 43.251 ± 2.213 | 57.315 ± 1.000 | 9.316 ± 0.217 | 10.213 ± 0.192 | **7.092 ± 0.271** |
| gm_queen5_5_3.wcsp | 3389.405 ± 73.102 | 5438.523 ± 929.597 | 4268.344 ± 20.662 | **2955.215 ± 20.225** | - |
| lm_batch_likelihood_brackets_4_4d | 30.541 ± 0.537 | 61.642 ± 0.715 | 42.526 ± 0.696 | 28.096 ± 0.797 | **21.804 ± 1.109** |
| lm_batch_likelihood_sentence_3_12d | 53.919 ± 1.018 | 113.388 ± 6.489 | 69.067 ± 0.685 | 49.379 ± 0.622 | **31.795 ± 3.267** |
| lm_batch_likelihood_sentence_4_4d | 28.008 ± 1.449 | 69.227 ± 15.151 | 42.228 ± 0.696 | 30.824 ± 0.472 | **23.268 ± 0.800** |
| str_matrix_chain_multiplication_100 | 23.677 ± 0.472 | 21.197 ± 1.018 | 20.294 ± 0.129 | **14.175 ± 0.159** | 17.985 ± 0.441 |
| str_mps_varying_inner_product_200 | 26.730 ± 0.691 | 30.048 ± 0.584 | 54.105 ± 0.064 | **21.899 ± 0.387** | 41.307 ± 1.092 |
| str_nw_mera_closed_120 | 974.097 ± 10.082 | 1057.343 ± 6.198 | 1031.249 ± 23.149 | 931.808 ± 6.804 | **594.442 ± 142.628** |
| str_nw_mera_open_26 | 750.238 ± 7.951 | 873.155 ± 7.174 | 670.364 ± 4.112 | **610.635 ± 7.296** | 662.144 ± 35.993 |
| tensornetwork_permutation_focus_step409_316 | 329.546 ± 2.284 | 528.219 ± 2.353 | 419.249 ± 3.975 | 562.013 ± 1.098 | **209.796 ± 3.587** |
| tensornetwork_permutation_light_415 | 331.773 ± 1.130 | 446.482 ± 1.951 | 609.858 ± 6.387 | 547.748 ± 2.874 | **242.898 ± 7.394** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.887 ± 0.053 | 0.876 ± 0.011 | 0.497 ± 0.060 | **0.362 ± 0.026** | 0.751 ± 0.082 |
| bin_elementwise_mul_2048x2048 | 23.076 ± 0.349 | 21.848 ± 0.257 | 5.681 ± 0.341 | 6.227 ± 0.230 | **5.353 ± 0.224** |
| bin_matmul_256 | 0.688 ± 0.041 | 0.669 ± 0.023 | **0.603 ± 0.044** | 0.618 ± 0.046 | 1.013 ± 0.081 |
| bin_outer_product_4096 | 40.434 ± 0.969 | 57.726 ± 0.868 | 9.512 ± 0.443 | 10.134 ± 0.456 | **7.850 ± 0.501** |
| gm_queen5_5_3.wcsp | **1318.626 ± 3.507** | 1919.560 ± 11.066 | 1440.305 ± 17.262 | 1595.829 ± 1015.581 | - |
| lm_batch_likelihood_brackets_4_4d | 29.280 ± 1.037 | 52.864 ± 0.960 | 41.045 ± 1.166 | 33.499 ± 0.562 | **23.151 ± 0.697** |
| lm_batch_likelihood_sentence_3_12d | 59.861 ± 1.837 | 121.672 ± 1.897 | 80.793 ± 0.868 | 44.102 ± 1.189 | **33.859 ± 1.694** |
| lm_batch_likelihood_sentence_4_4d | **30.325 ± 1.318** | 58.156 ± 1.458 | 41.359 ± 1.331 | 32.802 ± 1.207 | 38.846 ± 1.705 |
| str_matrix_chain_multiplication_100 | 23.534 ± 0.806 | 21.784 ± 0.630 | 20.747 ± 0.217 | **13.876 ± 0.106** | 30.083 ± 4.317 |
| str_mps_varying_inner_product_200 | 30.063 ± 1.235 | 31.275 ± 0.545 | 58.850 ± 0.369 | **23.517 ± 0.891** | 38.296 ± 1.738 |
| str_nw_mera_closed_120 | 871.487 ± 3.065 | 931.485 ± 4.940 | 1366.763 ± 625.308 | 833.364 ± 20.938 | **368.157 ± 24.698** |
| str_nw_mera_open_26 | 773.264 ± 4.420 | 888.611 ± 5.506 | 687.575 ± 209.745 | 619.020 ± 2.931 | **505.816 ± 21.195** |
| tensornetwork_permutation_focus_step409_316 | 334.980 ± 7.834 | 533.134 ± 3.221 | 428.971 ± 20.953 | 568.713 ± 15.456 | **208.310 ± 4.542** |
| tensornetwork_permutation_light_415 | 334.384 ± 2.818 | 450.786 ± 1.995 | 612.263 ± 33.114 | 543.689 ± 1.666 | **230.908 ± 3.958** |


## Threads: 4

- Timestamp: `20260528_075309`
- Source table: `data/results/results_t4_20260528_075309.md`

Logs:

- `data/results/tenferro_trace_t4_20260528_075309.log`
- `data/results/tenferro_eager_t4_20260528_075309.log`
- `data/results/libtorch_cpu_t4_20260528_075309.log`
- `data/results/pytorch_cpu_t4_20260528_075309.log`
- `data/results/jax_cpu_t4_20260528_075309.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 1.298 ± 0.136 | 1.404 ± 0.623 | 0.455 ± 0.031 | **0.236 ± 0.075** | 0.685 ± 0.102 |
| bin_elementwise_mul_2048x2048 | 22.704 ± 1.278 | 17.305 ± 1.265 | **4.520 ± 0.405** | 4.992 ± 0.264 | 5.784 ± 0.442 |
| bin_matmul_256 | 0.788 ± 0.075 | 0.437 ± 0.039 | **0.191 ± 0.018** | 0.239 ± 0.040 | 1.033 ± 0.188 |
| bin_outer_product_4096 | 43.098 ± 1.440 | 57.568 ± 1.366 | **5.982 ± 0.802** | 7.158 ± 0.458 | 7.308 ± 0.607 |
| gm_queen5_5_3.wcsp | 3280.560 ± 140.868 | 5214.225 ± 40.966 | 1842.109 ± 116.764 | **1196.184 ± 48.925** | - |
| lm_batch_likelihood_brackets_4_4d | 29.675 ± 0.732 | 60.682 ± 1.447 | 33.032 ± 4.605 | **15.188 ± 0.368** | 22.620 ± 0.835 |
| lm_batch_likelihood_sentence_3_12d | 40.617 ± 0.517 | 102.019 ± 5.247 | 60.796 ± 1.325 | **23.961 ± 1.249** | 32.145 ± 1.654 |
| lm_batch_likelihood_sentence_4_4d | 26.918 ± 0.774 | 67.298 ± 1.314 | 31.158 ± 0.693 | **16.118 ± 0.896** | 23.752 ± 1.070 |
| str_matrix_chain_multiplication_100 | 18.748 ± 0.646 | 16.798 ± 0.518 | 16.594 ± 0.178 | **9.445 ± 0.225** | 18.255 ± 0.519 |
| str_mps_varying_inner_product_200 | 24.600 ± 0.564 | 28.684 ± 0.559 | 59.601 ± 1.700 | **20.481 ± 0.821** | 43.448 ± 0.829 |
| str_nw_mera_closed_120 | 518.215 ± 4.405 | 612.327 ± 7.613 | 559.140 ± 14.142 | **350.786 ± 5.992** | 453.707 ± 14.060 |
| str_nw_mera_open_26 | 412.470 ± 7.066 | 535.784 ± 5.238 | 342.134 ± 4.045 | **246.525 ± 1.415** | 504.310 ± 23.841 |
| tensornetwork_permutation_focus_step409_316 | 301.573 ± 2.457 | 505.509 ± 21.863 | 246.277 ± 3.311 | **174.049 ± 2.618** | 214.386 ± 5.569 |
| tensornetwork_permutation_light_415 | 304.403 ± 5.248 | 405.455 ± 5.018 | 343.696 ± 17.093 | **194.120 ± 14.593** | 412.419 ± 31.135 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.916 ± 0.035 | 0.919 ± 0.025 | 0.611 ± 0.172 | **0.222 ± 0.085** | 1.326 ± 0.133 |
| bin_elementwise_mul_2048x2048 | 19.451 ± 0.280 | 20.195 ± 0.752 | **4.791 ± 0.425** | 5.096 ± 0.290 | 7.102 ± 0.541 |
| bin_matmul_256 | 0.323 ± 0.070 | 0.279 ± 0.048 | 0.288 ± 0.135 | **0.271 ± 0.072** | 1.851 ± 0.205 |
| bin_outer_product_4096 | 40.783 ± 0.852 | 57.141 ± 1.062 | **6.910 ± 0.329** | 7.786 ± 0.736 | 11.817 ± 1.320 |
| gm_queen5_5_3.wcsp | 1109.588 ± 11.746 | 1687.960 ± 10.160 | 687.377 ± 674.551 | **543.353 ± 22.857** | - |
| lm_batch_likelihood_brackets_4_4d | 29.067 ± 1.410 | 49.943 ± 0.941 | 107.102 ± 30.223 | **14.914 ± 0.400** | 46.105 ± 1.846 |
| lm_batch_likelihood_sentence_3_12d | 47.513 ± 0.902 | 101.182 ± 2.083 | 98.584 ± 6.907 | **23.363 ± 0.622** | 56.843 ± 6.465 |
| lm_batch_likelihood_sentence_4_4d | 30.681 ± 1.112 | 55.707 ± 1.708 | 48.267 ± 9.347 | **14.278 ± 0.367** | 37.379 ± 3.204 |
| str_matrix_chain_multiplication_100 | 18.922 ± 0.766 | 17.446 ± 0.328 | 18.871 ± 0.630 | **9.067 ± 0.189** | 30.843 ± 1.569 |
| str_mps_varying_inner_product_200 | 27.404 ± 0.841 | 31.529 ± 0.351 | 82.675 ± 2.544 | **21.525 ± 0.400** | 39.147 ± 0.639 |
| str_nw_mera_closed_120 | 390.883 ± 7.199 | 459.979 ± 13.068 | 1203.993 ± 513.429 | **303.743 ± 6.688** | 440.098 ± 136.468 |
| str_nw_mera_open_26 | 442.182 ± 19.951 | 549.110 ± 9.070 | 346.020 ± 38.718 | **258.961 ± 13.826** | 511.886 ± 17.857 |
| tensornetwork_permutation_focus_step409_316 | 308.382 ± 13.691 | 491.011 ± 4.428 | 245.976 ± 3.984 | **192.828 ± 6.538** | 214.409 ± 3.785 |
| tensornetwork_permutation_light_415 | 313.890 ± 19.047 | 407.895 ± 3.285 | 347.778 ± 107.937 | **187.988 ± 1.825** | 245.821 ± 37.979 |


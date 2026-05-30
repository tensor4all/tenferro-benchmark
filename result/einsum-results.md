# Einsum Benchmark Results

Latest run: `./scripts/run_all.sh 1`.

This file collects the latest einsum benchmark table for each thread count under `data/results/`.

- tenferro-rs commit: `fa722375c8662b5532a6f875a6bef3494ace40b5`

## Threads: 1

- Timestamp: `20260530_040811`
- Source table: `data/results/results_t1_20260530_040811.md`

Logs:

- `data/results/tenferro_trace_t1_20260530_040811.log`
- `data/results/tenferro_eager_t1_20260530_040811.log`
- `data/results/libtorch_cpu_t1_20260530_040811.log`
- `data/results/pytorch_cpu_t1_20260530_040811.log`
- `data/results/jax_cpu_t1_20260530_040811.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_matmul_256 | 2.046 ± 0.014 | 1.602 ± 0.008 | **1.088 ± 0.056** | 1.090 ± 0.038 | 1.149 ± 0.074 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_matmul_256 | 2.007 ± 0.068 | 1.581 ± 0.008 | **1.075 ± 0.012** | 1.111 ± 0.085 | 1.180 ± 0.043 |


## Threads: 4

- Timestamp: `20260528_102430`
- Source table: `data/results/results_t4_20260528_102430.md`

Logs:

- `data/results/tenferro_trace_t4_20260528_102430.log`
- `data/results/tenferro_eager_t4_20260528_102430.log`
- `data/results/libtorch_cpu_t4_20260528_102430.log`
- `data/results/pytorch_cpu_t4_20260528_102430.log`
- `data/results/jax_cpu_t4_20260528_102430.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 2.249 ± 0.098 | 1.976 ± 1.253 | 0.922 ± 0.003 | **0.314 ± 0.011** | 1.391 ± 0.297 |
| bin_elementwise_mul_2048x2048 | 107.827 ± 0.544 | 85.593 ± 0.355 | **7.785 ± 0.128** | 10.363 ± 0.818 | 14.613 ± 4.143 |
| bin_matmul_256 | 1.376 ± 0.041 | 0.847 ± 0.041 | 0.630 ± 0.008 | **0.423 ± 0.028** | 1.154 ± 0.129 |
| bin_outer_product_4096 | 227.145 ± 1.099 | 316.731 ± 2.131 | **19.379 ± 2.368** | 28.871 ± 0.674 | 47.927 ± 1.031 |
| gm_queen5_5_3.wcsp | 11205.954 ± 18.554 | 18316.192 ± 36.045 | 3576.086 ± 24.213 | **2667.848 ± 11.741** | - |
| lm_batch_likelihood_brackets_4_4d | 49.324 ± 0.299 | 122.417 ± 1.079 | 43.185 ± 0.215 | **19.358 ± 0.312** | 42.636 ± 1.200 |
| lm_batch_likelihood_sentence_3_12d | 57.265 ± 0.773 | 225.149 ± 24.648 | 95.060 ± 0.823 | **24.717 ± 0.296** | 53.053 ± 4.741 |
| lm_batch_likelihood_sentence_4_4d | 44.139 ± 0.968 | 136.254 ± 1.484 | 43.944 ± 0.228 | **19.825 ± 0.242** | 42.030 ± 0.726 |
| str_matrix_chain_multiplication_100 | 31.831 ± 0.184 | 28.487 ± 0.718 | 24.766 ± 0.263 | **13.554 ± 0.143** | 48.065 ± 5.821 |
| str_mps_varying_inner_product_200 | 42.719 ± 0.690 | 48.803 ± 0.760 | 87.301 ± 0.257 | **30.902 ± 0.824** | 92.938 ± 3.009 |
| str_nw_mera_closed_120 | 1344.259 ± 10.950 | 1780.370 ± 8.551 | 946.327 ± 4.973 | 766.021 ± 19.838 | **652.895 ± 29.138** |
| str_nw_mera_open_26 | 1087.035 ± 9.368 | 1735.344 ± 12.578 | 790.794 ± 1.598 | **505.936 ± 3.936** | 604.660 ± 52.765 |
| tensornetwork_permutation_focus_step409_316 | 599.762 ± 3.090 | 1294.158 ± 4.850 | 367.662 ± 6.066 | **327.690 ± 3.801** | 506.536 ± 16.531 |
| tensornetwork_permutation_light_415 | 635.119 ± 4.034 | 965.523 ± 11.451 | 501.219 ± 8.526 | **335.029 ± 4.122** | 547.786 ± 35.175 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 1.689 ± 0.045 | 1.687 ± 0.048 | 0.918 ± 0.037 | **0.273 ± 0.021** | 1.361 ± 0.051 |
| bin_elementwise_mul_2048x2048 | 101.911 ± 0.321 | 85.644 ± 0.538 | 8.158 ± 0.233 | 9.797 ± 0.388 | **6.162 ± 0.824** |
| bin_matmul_256 | 0.624 ± 0.024 | 0.612 ± 0.031 | **0.377 ± 0.033** | 0.396 ± 0.029 | 1.027 ± 0.122 |
| bin_outer_product_4096 | 226.805 ± 1.373 | 317.357 ± 1.598 | **21.826 ± 4.372** | 30.130 ± 0.438 | 45.882 ± 0.941 |
| gm_queen5_5_3.wcsp | 3794.419 ± 17.579 | 5965.157 ± 13.470 | 1300.511 ± 4.079 | **1147.083 ± 9.674** | - |
| lm_batch_likelihood_brackets_4_4d | 45.661 ± 0.979 | 93.597 ± 2.631 | 41.111 ± 0.354 | **19.783 ± 0.191** | 44.551 ± 1.440 |
| lm_batch_likelihood_sentence_3_12d | 69.072 ± 8.094 | 285.890 ± 32.981 | 73.849 ± 0.820 | **26.086 ± 0.620** | 50.902 ± 1.581 |
| lm_batch_likelihood_sentence_4_4d | 48.072 ± 0.770 | 105.371 ± 0.953 | 42.457 ± 0.193 | **19.968 ± 0.148** | 37.225 ± 1.242 |
| str_matrix_chain_multiplication_100 | 32.182 ± 0.228 | 28.717 ± 0.398 | 25.126 ± 0.177 | **13.752 ± 0.326** | 44.237 ± 1.737 |
| str_mps_varying_inner_product_200 | 46.815 ± 0.762 | 53.372 ± 0.312 | 92.474 ± 0.114 | **39.986 ± 0.414** | 106.706 ± 3.048 |
| str_nw_mera_closed_120 | 968.951 ± 5.493 | 1262.961 ± 7.184 | 840.243 ± 14.143 | 586.460 ± 4.858 | **453.950 ± 49.656** |
| str_nw_mera_open_26 | 1098.699 ± 4.459 | 1729.207 ± 10.486 | 770.527 ± 9.203 | **526.143 ± 3.371** | 618.384 ± 66.372 |
| tensornetwork_permutation_focus_step409_316 | 639.209 ± 4.548 | 1293.663 ± 4.469 | 377.187 ± 7.306 | **343.075 ± 1.831** | 512.016 ± 18.357 |
| tensornetwork_permutation_light_415 | 642.505 ± 3.899 | 966.919 ± 5.023 | 515.301 ± 5.635 | **348.444 ± 7.439** | 533.186 ± 57.764 |


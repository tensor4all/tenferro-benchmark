# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/cpu/einsum/20260603_184022/run.yaml`
- Timestamp: `20260603_184022`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/cpu/einsum/20260603_184022`.

- tenferro-rs commit: `cc46577429a8cc5a91cb0ed86910b47b2055b420`

## CPU Information

- Model: `arm`
- Vendor: `unknown`
- Logical CPUs: `18`
- Sockets: `unknown`
- Cores per socket: `unknown`
- Threads per core: `unknown`
- NUMA nodes: `unknown`
- Python platform: `macOS-26.5-arm64-arm-64bit`

## Threads: 1

- Source table: `data/results/cpu/einsum/20260603_184022/einsum_table_t1_20260603_184022.md`

Logs:

- `data/results/cpu/einsum/20260603_184022/tenferro_trace_t1_20260603_184022.log`
- `data/results/cpu/einsum/20260603_184022/tenferro_eager_t1_20260603_184022.log`
- `data/results/cpu/einsum/20260603_184022/pytorch_cpu_t1_20260603_184022.log`
- `data/results/cpu/einsum/20260603_184022/jax_cpu_t1_20260603_184022.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 1.329 ± 0.118 | 1.241 ± 0.235 | - | **0.097 ± 0.005** | 0.187 ± 0.037 |
| bin_elementwise_mul_2048x2048 | 2.875 ± 0.183 | 2.735 ± 0.120 | - | 0.843 ± 0.045 | **0.766 ± 0.117** |
| bin_matmul_256 | 0.638 ± 0.047 | 0.636 ± 0.036 | - | **0.094 ± 0.003** | 0.225 ± 0.045 |
| bin_outer_product_4096 | 8.347 ± 0.104 | 9.980 ± 0.239 | - | 1.512 ± 0.027 | **0.652 ± 0.027** |
| gm_queen5_5_3.wcsp | 1086.051 ± 12.488 | 1645.214 ± 26.998 | - | 904.821 ± 14.723 | **454.399 ± 19.742** |
| lm_batch_likelihood_brackets_4_4d | 11.455 ± 0.246 | 17.886 ± 0.502 | - | 9.568 ± 0.416 | **5.935 ± 0.144** |
| lm_batch_likelihood_sentence_3_12d | 32.996 ± 0.080 | 51.328 ± 0.981 | - | 18.004 ± 0.353 | **7.090 ± 0.256** |
| lm_batch_likelihood_sentence_4_4d | 11.132 ± 0.225 | 19.530 ± 0.405 | - | 10.009 ± 0.521 | **5.997 ± 0.294** |
| str_matrix_chain_multiplication_100 | 10.395 ± 0.163 | 9.547 ± 0.198 | - | **2.701 ± 0.088** | 6.259 ± 0.092 |
| str_mps_varying_inner_product_200 | 9.980 ± 0.193 | 13.033 ± 0.630 | - | **5.348 ± 0.124** | 14.198 ± 1.264 |
| str_nw_mera_closed_120 | 862.784 ± 9.427 | 879.085 ± 11.247 | - | 167.049 ± 8.806 | **85.862 ± 1.317** |
| str_nw_mera_open_26 | 610.861 ± 3.115 | 629.306 ± 3.222 | - | 127.836 ± 0.808 | **67.241 ± 1.251** |
| tensornetwork_permutation_focus_step409_316 | 168.764 ± 1.711 | 239.769 ± 9.645 | - | 165.482 ± 10.685 | **73.713 ± 7.192** |
| tensornetwork_permutation_light_415 | 169.067 ± 1.376 | 204.185 ± 1.230 | - | 149.086 ± 11.299 | **84.940 ± 9.589** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.928 ± 0.054 | 0.753 ± 0.055 | - | **0.110 ± 0.023** | 0.192 ± 0.038 |
| bin_elementwise_mul_2048x2048 | 2.782 ± 0.047 | 2.534 ± 0.100 | - | 0.856 ± 0.027 | **0.681 ± 0.090** |
| bin_matmul_256 | 0.616 ± 0.016 | 0.621 ± 0.015 | - | **0.093 ± 0.003** | 0.255 ± 0.054 |
| bin_outer_product_4096 | 8.256 ± 0.209 | 10.040 ± 0.093 | - | 1.517 ± 0.027 | **0.676 ± 0.163** |
| gm_queen5_5_3.wcsp | 698.667 ± 1.874 | 874.146 ± 13.625 | - | 348.507 ± 31.264 | **170.603 ± 17.502** |
| lm_batch_likelihood_brackets_4_4d | 10.874 ± 0.110 | 18.465 ± 5.688 | - | 10.152 ± 0.443 | **5.907 ± 0.202** |
| lm_batch_likelihood_sentence_3_12d | 34.505 ± 0.347 | 55.776 ± 4.676 | - | 18.736 ± 0.194 | **7.357 ± 0.229** |
| lm_batch_likelihood_sentence_4_4d | 12.302 ± 0.224 | 19.631 ± 0.294 | - | 9.859 ± 0.294 | **5.760 ± 0.660** |
| str_matrix_chain_multiplication_100 | 10.488 ± 0.487 | 9.749 ± 0.345 | - | **2.718 ± 0.099** | 6.685 ± 0.575 |
| str_mps_varying_inner_product_200 | 11.659 ± 0.209 | 13.518 ± 0.178 | - | **6.859 ± 0.290** | 16.223 ± 0.361 |
| str_nw_mera_closed_120 | 850.407 ± 5.271 | 864.522 ± 7.039 | - | 135.207 ± 4.731 | **83.876 ± 1.366** |
| str_nw_mera_open_26 | 611.057 ± 1.034 | 636.247 ± 7.423 | - | 141.189 ± 11.742 | **75.206 ± 1.167** |
| tensornetwork_permutation_focus_step409_316 | 167.717 ± 0.971 | 245.567 ± 10.057 | - | 157.221 ± 2.020 | **78.105 ± 7.299** |
| tensornetwork_permutation_light_415 | 168.993 ± 1.327 | 210.376 ± 3.573 | - | 140.190 ± 4.796 | **83.704 ± 6.489** |


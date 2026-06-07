# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Timestamp: `20260607_080940`
- Run metadata: `data/results/cpu/einsum/20260607_080940/run.yaml`
- tenferro-rs commit: `801d4345213800eceb25d0000e4b562344e83682`
- tenferro-rs dirty: not recorded in this pre-target-profile raw run
- tenferro-rs features: `system-accelerate`
- tenferro BLAS: `accelerate`
- PyTorch: BLAS provider `accelerate`, version `2.12.0`
- JAX: dot backend `xla_cpu`, version `0.10.1`

Latest run: `./scripts/run_all.sh 1`.

This latest report was regenerated from the existing macOS CPU raw run after the target-profile layout cleanup. New runs write raw data under `data/results/mac-cpu/cpu/einsum/<timestamp>/`.

## Threads: 1

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.569 ± 0.071 | 0.578 ± 0.035 | 0.397 ± 0.006 | **0.385 ± 0.074** |
| bin_batched_matmul_b32_m64_n64_k64 | **0.082 ± 0.001** | 0.084 ± 0.001 | 0.095 ± 0.002 | 0.187 ± 0.042 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.113 ± 0.002 | 0.136 ± 0.011 | **0.112 ± 0.002** | 0.130 ± 0.021 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.113 ± 0.003 | 0.117 ± 0.007 | **0.110 ± 0.001** | 0.136 ± 0.030 |
| bin_elementwise_mul_2048x2048 | 1.080 ± 0.079 | 1.121 ± 0.065 | 0.830 ± 0.050 | **0.819 ± 0.082** |
| bin_matmul_1024 | 4.411 ± 0.360 | 4.475 ± 1.580 | 4.514 ± 0.422 | **3.464 ± 0.396** |
| bin_matmul_256 | **0.083 ± 0.004** | 0.085 ± 0.001 | 0.094 ± 0.003 | 0.211 ± 0.041 |
| bin_outer_product_4096 | 0.950 ± 0.025 | 1.007 ± 0.024 | 1.491 ± 0.013 | **0.655 ± 0.045** |
| gm_queen5_5_3.wcsp | 813.303 ± 4.916 | 866.032 ± 5.702 | 860.965 ± 11.478 | **464.472 ± 44.530** |
| lm_batch_likelihood_brackets_4_4d | 6.637 ± 0.219 | 11.640 ± 0.248 | 9.606 ± 0.476 | **5.072 ± 0.133** |
| lm_batch_likelihood_sentence_3_12d | 8.745 ± 0.462 | 22.662 ± 0.165 | 18.227 ± 0.697 | **6.854 ± 0.076** |
| lm_batch_likelihood_sentence_4_4d | 6.003 ± 0.896 | 13.243 ± 0.285 | 9.944 ± 0.212 | **5.619 ± 0.145** |
| nary_matmul_chain_64 | 0.019 ± 0.002 | **0.008 ± 0.000** | 0.024 ± 0.001 | 0.078 ± 0.009 |
| str_matrix_chain_multiplication_100 | **1.823 ± 0.024** | 2.341 ± 0.064 | 2.694 ± 0.120 | 6.084 ± 0.178 |
| str_mps_varying_inner_product_200 | **4.225 ± 0.122** | 7.644 ± 0.132 | 5.271 ± 0.231 | 13.703 ± 0.242 |
| str_nw_mera_closed_120 | 160.491 ± 0.473 | 182.863 ± 10.854 | 157.146 ± 1.082 | **88.611 ± 1.695** |
| str_nw_mera_open_26 | 132.804 ± 4.630 | 178.401 ± 1.742 | 119.907 ± 1.258 | **67.181 ± 0.966** |
| tensornetwork_permutation_focus_step409_316 | 123.528 ± 6.797 | 179.214 ± 7.353 | 157.481 ± 1.223 | **78.850 ± 5.545** |
| tensornetwork_permutation_light_415 | 122.631 ± 6.896 | 151.278 ± 1.184 | 138.474 ± 1.334 | **82.179 ± 6.509** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.569 ± 0.071 | 0.578 ± 0.035 | 0.397 ± 0.006 | **0.385 ± 0.074** |
| bin_batched_matmul_b32_m64_n64_k64 | **0.082 ± 0.001** | 0.084 ± 0.001 | 0.095 ± 0.002 | 0.187 ± 0.042 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.113 ± 0.002 | 0.136 ± 0.011 | **0.112 ± 0.002** | 0.130 ± 0.021 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.113 ± 0.003 | 0.117 ± 0.007 | **0.110 ± 0.001** | 0.136 ± 0.030 |
| bin_elementwise_mul_2048x2048 | 1.080 ± 0.079 | 1.121 ± 0.065 | 0.830 ± 0.050 | **0.819 ± 0.082** |
| bin_matmul_1024 | 4.411 ± 0.360 | 4.475 ± 1.580 | 4.514 ± 0.422 | **3.464 ± 0.396** |
| bin_matmul_256 | **0.083 ± 0.004** | 0.085 ± 0.001 | 0.094 ± 0.003 | 0.211 ± 0.041 |
| bin_outer_product_4096 | 0.950 ± 0.025 | 1.007 ± 0.024 | 1.491 ± 0.013 | **0.655 ± 0.045** |
| gm_queen5_5_3.wcsp | 239.991 ± 4.383 | 356.737 ± 2.468 | 318.698 ± 4.448 | **151.931 ± 2.682** |
| lm_batch_likelihood_brackets_4_4d | 7.363 ± 0.178 | 12.970 ± 0.335 | 10.139 ± 0.451 | **6.023 ± 0.650** |
| lm_batch_likelihood_sentence_3_12d | 10.637 ± 0.374 | 24.471 ± 0.628 | 17.868 ± 0.329 | **7.468 ± 0.317** |
| lm_batch_likelihood_sentence_4_4d | 7.731 ± 0.088 | 14.350 ± 0.735 | 9.738 ± 0.267 | **6.418 ± 0.777** |
| nary_matmul_chain_64 | 0.019 ± 0.002 | **0.008 ± 0.000** | 0.024 ± 0.001 | 0.078 ± 0.009 |
| str_matrix_chain_multiplication_100 | **1.764 ± 0.031** | 2.352 ± 0.060 | 2.677 ± 0.076 | 8.132 ± 1.664 |
| str_mps_varying_inner_product_200 | **5.803 ± 0.159** | 8.116 ± 0.158 | 6.252 ± 0.207 | 15.507 ± 0.920 |
| str_nw_mera_closed_120 | 147.053 ± 4.077 | 154.052 ± 1.048 | 129.540 ± 1.286 | **84.401 ± 3.063** |
| str_nw_mera_open_26 | 135.214 ± 1.156 | 176.496 ± 5.505 | 124.774 ± 1.128 | **80.503 ± 2.806** |
| tensornetwork_permutation_focus_step409_316 | 123.528 ± 6.797 | 179.214 ± 7.353 | 157.481 ± 1.223 | **78.850 ± 5.545** |
| tensornetwork_permutation_light_415 | 122.631 ± 6.896 | 151.278 ± 1.184 | 138.474 ± 1.334 | **82.179 ± 6.509** |

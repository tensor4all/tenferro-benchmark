# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/cpu/einsum/20260604_121558/run.yaml`
- Timestamp: `20260604_121558`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/cpu/einsum/20260604_121558`.

- tenferro-rs commit: `ba6523f81e3fb35b00ba3b3f96b551ed95cd6d1b`

## CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5-arm64-arm-64bit`

## Thread Environment

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

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 1

- Source table: `data/results/cpu/einsum/20260604_121558/einsum_table_t1_20260604_121558.md`

Logs:

- `data/results/cpu/einsum/20260604_121558/tenferro_trace_t1_20260604_121558.log`
- `data/results/cpu/einsum/20260604_121558/tenferro_eager_t1_20260604_121558.log`
- `data/results/cpu/einsum/20260604_121558/pytorch_cpu_t1_20260604_121558.log`
- `data/results/cpu/einsum/20260604_121558/jax_cpu_t1_20260604_121558.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.609 ± 0.075 | 0.443 ± 0.080 | - | **0.140 ± 0.003** | 0.243 ± 0.028 |
| bin_elementwise_mul_2048x2048 | 3.893 ± 0.064 | 3.193 ± 0.031 | - | 1.247 ± 0.009 | **0.924 ± 0.048** |
| bin_matmul_256 | 0.234 ± 0.018 | 0.220 ± 0.025 | - | **0.137 ± 0.003** | 0.290 ± 0.031 |
| bin_outer_product_4096 | 54.504 ± 0.646 | 56.503 ± 0.396 | - | 2.232 ± 0.016 | **0.831 ± 0.037** |
| gm_queen5_5_3.wcsp | 1639.189 ± 36.857 | 2500.230 ± 60.797 | - | 1320.753 ± 18.208 | **613.058 ± 48.309** |
| lm_batch_likelihood_brackets_4_4d | 11.968 ± 0.147 | 23.226 ± 0.410 | - | 13.824 ± 0.284 | **7.375 ± 0.154** |
| lm_batch_likelihood_sentence_3_12d | 13.116 ± 0.377 | 41.841 ± 0.382 | - | 24.773 ± 0.189 | **8.356 ± 0.254** |
| lm_batch_likelihood_sentence_4_4d | 10.290 ± 0.106 | 25.395 ± 0.352 | - | 13.861 ± 0.110 | **8.060 ± 0.243** |
| nary_matmul_chain_64 | 0.028 ± 0.002 | **0.025 ± 0.005** | - | 0.034 ± 0.001 | 0.109 ± 0.008 |
| str_matrix_chain_multiplication_100 | 4.627 ± 0.099 | 4.151 ± 0.321 | - | **3.586 ± 0.036** | 8.823 ± 0.171 |
| str_mps_varying_inner_product_200 | 8.443 ± 0.085 | 14.757 ± 0.168 | - | **7.645 ± 0.070** | 20.032 ± 0.187 |
| str_nw_mera_closed_120 | 237.618 ± 8.371 | 252.793 ± 7.963 | - | **231.730 ± 1.292** | 232.060 ± 4.577 |
| str_nw_mera_open_26 | 247.207 ± 2.914 | 279.987 ± 3.552 | - | 181.865 ± 1.744 | **165.378 ± 5.001** |
| tensornetwork_permutation_focus_step409_316 | 179.851 ± 10.408 | 293.342 ± 19.864 | - | 233.414 ± 1.128 | **105.448 ± 4.069** |
| tensornetwork_permutation_light_415 | 178.274 ± 3.767 | 234.684 ± 3.533 | - | 209.342 ± 0.906 | **113.859 ± 4.712** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.271 ± 0.007 | 0.246 ± 0.030 | - | **0.141 ± 0.001** | 0.204 ± 0.021 |
| bin_elementwise_mul_2048x2048 | 3.855 ± 0.025 | 3.237 ± 0.071 | - | 1.277 ± 0.043 | **0.863 ± 0.029** |
| bin_matmul_256 | 0.233 ± 0.002 | 0.218 ± 0.001 | - | **0.136 ± 0.001** | 0.291 ± 0.015 |
| bin_outer_product_4096 | 54.139 ± 0.608 | 56.633 ± 0.256 | - | 2.257 ± 0.040 | **0.850 ± 0.010** |
| gm_queen5_5_3.wcsp | 515.509 ± 22.045 | 770.490 ± 2.149 | - | 438.864 ± 9.328 | **271.721 ± 10.404** |
| lm_batch_likelihood_brackets_4_4d | 12.400 ± 0.798 | 24.679 ± 0.783 | - | 14.278 ± 0.301 | **7.240 ± 0.280** |
| lm_batch_likelihood_sentence_3_12d | 15.678 ± 0.060 | 43.121 ± 0.285 | - | 25.748 ± 0.553 | **8.940 ± 0.292** |
| lm_batch_likelihood_sentence_4_4d | 11.832 ± 0.107 | 26.362 ± 0.702 | - | 13.790 ± 0.083 | **7.895 ± 0.322** |
| nary_matmul_chain_64 | 0.026 ± 0.002 | **0.024 ± 0.004** | - | 0.035 ± 0.002 | 0.104 ± 0.008 |
| str_matrix_chain_multiplication_100 | 4.666 ± 0.039 | 4.190 ± 0.205 | - | **3.603 ± 0.031** | 8.783 ± 0.148 |
| str_mps_varying_inner_product_200 | 10.890 ± 0.210 | 16.105 ± 0.298 | - | **8.770 ± 0.102** | 22.540 ± 1.726 |
| str_nw_mera_closed_120 | 211.468 ± 0.691 | 225.707 ± 1.758 | - | **188.992 ± 4.635** | 218.782 ± 4.219 |
| str_nw_mera_open_26 | 248.961 ± 1.599 | 284.892 ± 5.283 | - | 190.292 ± 1.077 | **187.731 ± 32.272** |
| tensornetwork_permutation_focus_step409_316 | 180.346 ± 14.115 | 286.036 ± 15.863 | - | 233.754 ± 5.880 | **114.254 ± 12.892** |
| tensornetwork_permutation_light_415 | 177.208 ± 11.068 | 234.221 ± 4.149 | - | 210.220 ± 2.532 | **118.140 ± 6.676** |

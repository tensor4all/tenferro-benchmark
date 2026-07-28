# Linux CPU Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `linux-cpu`
- Thread runs: `1 4`

Mirrored from `amd-cpu` runs (`BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh <THREADS>`);
regenerate with `./scripts/mirror_linux_cpu_cpu_reports.sh 1:20260728_013419 4:20260728_015909 permutation:20260728_021651`.

This report is generated from sequential CPU runs. Do not compare it with
measurements collected while another CPU benchmark process was running.

## Run Inputs

- Threads 1: timestamp `20260728_013419`, raw run `data/results/linux-cpu/cpu/einsum/20260728_013419`
- Threads 4: timestamp `20260728_015909`, raw run `data/results/linux-cpu/cpu/einsum/20260728_015909`

## Threads: 1

### Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `linux-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/linux-cpu/cpu/einsum/20260728_013419/run.yaml`
- Timestamp: `20260728_013419`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/linux-cpu/cpu/einsum/20260728_013419`.

- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

#### CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

#### Thread Environment

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

#### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 1

- Source table: `data/results/linux-cpu/cpu/einsum/20260728_013419/einsum_table_t1_20260728_013419.md`

Logs:

- `data/results/linux-cpu/cpu/einsum/20260728_013419/tenferro_trace_t1_20260728_013419.log`
- `data/results/linux-cpu/cpu/einsum/20260728_013419/tenferro_eager_t1_20260728_013419.log`
- `data/results/linux-cpu/cpu/einsum/20260728_013419/pytorch_cpu_t1_20260728_013419.log`
- `data/results/linux-cpu/cpu/einsum/20260728_013419/jax_cpu_t1_20260728_013419.log`

###### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 8.208 ± 1.219 | 4.728 ± 0.215 | 3.561 ± 0.007 | **1.088 ± 0.102** |
| bin_batched_matmul_b32_m64_n64_k64 | 0.987 ± 0.116 | 0.852 ± 0.013 | **0.594 ± 0.072** | 0.732 ± 0.124 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.227 ± 0.003 | 0.254 ± 0.009 | **0.218 ± 0.014** | 0.721 ± 0.041 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.404 ± 0.038 | 0.249 ± 0.008 | **0.203 ± 0.004** | 0.733 ± 0.090 |
| bin_elementwise_mul_2048x2048 | 19.064 ± 0.879 | 20.830 ± 0.869 | 19.412 ± 0.589 | **12.788 ± 0.706** |
| bin_matmul_1024 | 52.860 ± 0.746 | 45.160 ± 2.809 | 51.944 ± 0.342 | **7.243 ± 0.493** |
| bin_matmul_256 | 1.193 ± 0.013 | 1.013 ± 0.013 | 1.199 ± 0.006 | **0.470 ± 0.046** |
| bin_outer_product_4096 | 66.255 ± 0.848 | 72.007 ± 4.660 | 69.718 ± 1.546 | **33.831 ± 8.448** |
| gm_queen5_5_3.wcsp | 7996.424 ± 135.868 | 8675.601 ± 53.612 | **6349.304 ± 31.371** | - |
| lm_batch_likelihood_brackets_4_4d | 47.261 ± 2.551 | 54.344 ± 3.241 | 31.753 ± 1.021 | **21.677 ± 2.118** |
| lm_batch_likelihood_sentence_3_12d | 86.147 ± 3.207 | 152.733 ± 4.604 | 103.513 ± 1.024 | **27.728 ± 3.053** |
| lm_batch_likelihood_sentence_4_4d | 49.765 ± 2.118 | 46.915 ± 1.563 | 33.433 ± 1.298 | **24.930 ± 0.632** |
| nary_matmul_chain_64 | 0.135 ± 0.003 | **0.092 ± 0.002** | 0.136 ± 0.005 | 0.277 ± 0.013 |
| str_matrix_chain_multiplication_100 | 16.199 ± 1.441 | **15.806 ± 1.656** | 16.465 ± 1.049 | 20.537 ± 4.551 |
| str_mps_varying_inner_product_200 | **24.061 ± 0.754** | 33.275 ± 3.890 | 26.074 ± 0.357 | 42.008 ± 0.688 |
| str_nw_mera_closed_120 | 1953.649 ± 21.156 | 2022.645 ± 42.129 | 1898.425 ± 13.346 | **369.072 ± 20.715** |
| str_nw_mera_open_26 | 1753.463 ± 25.372 | 1874.329 ± 25.897 | 1196.909 ± 11.408 | **354.072 ± 14.348** |
| tensornetwork_permutation_focus_step409_316 | 813.135 ± 20.511 | 1165.719 ± 33.858 | 859.065 ± 12.581 | **208.476 ± 12.433** |
| tensornetwork_permutation_light_415 | 802.140 ± 21.175 | 1110.512 ± 21.345 | 831.107 ± 12.438 | **243.097 ± 25.744** |

###### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 8.208 ± 1.219 | 4.728 ± 0.215 | 3.561 ± 0.007 | **1.088 ± 0.102** |
| bin_batched_matmul_b32_m64_n64_k64 | 0.987 ± 0.116 | 0.852 ± 0.013 | **0.594 ± 0.072** | 0.732 ± 0.124 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.227 ± 0.003 | 0.254 ± 0.009 | **0.218 ± 0.014** | 0.721 ± 0.041 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.404 ± 0.038 | 0.249 ± 0.008 | **0.203 ± 0.004** | 0.733 ± 0.090 |
| bin_elementwise_mul_2048x2048 | 19.064 ± 0.879 | 20.830 ± 0.869 | 19.412 ± 0.589 | **12.788 ± 0.706** |
| bin_matmul_1024 | 52.860 ± 0.746 | 45.160 ± 2.809 | 51.944 ± 0.342 | **7.243 ± 0.493** |
| bin_matmul_256 | 1.193 ± 0.013 | 1.013 ± 0.013 | 1.199 ± 0.006 | **0.470 ± 0.046** |
| bin_outer_product_4096 | 66.255 ± 0.848 | 72.007 ± 4.660 | 69.718 ± 1.546 | **33.831 ± 8.448** |
| gm_queen5_5_3.wcsp | 2651.264 ± 51.608 | 3644.513 ± 36.550 | **2335.566 ± 22.307** | - |
| lm_batch_likelihood_brackets_4_4d | 31.806 ± 2.707 | 56.167 ± 1.285 | 32.191 ± 0.574 | **22.952 ± 0.926** |
| lm_batch_likelihood_sentence_3_12d | 78.463 ± 6.849 | 141.176 ± 4.009 | 109.105 ± 2.812 | **28.191 ± 1.089** |
| lm_batch_likelihood_sentence_4_4d | 34.889 ± 4.183 | 61.933 ± 1.382 | 34.196 ± 0.362 | **23.211 ± 1.985** |
| nary_matmul_chain_64 | 0.135 ± 0.003 | **0.092 ± 0.002** | 0.136 ± 0.005 | 0.277 ± 0.013 |
| str_matrix_chain_multiplication_100 | 16.254 ± 0.211 | 22.817 ± 1.195 | **15.875 ± 0.263** | 20.649 ± 2.928 |
| str_mps_varying_inner_product_200 | **24.848 ± 0.326** | 57.743 ± 1.280 | 26.103 ± 0.310 | 36.745 ± 1.535 |
| str_nw_mera_closed_120 | 1532.969 ± 17.854 | 1538.391 ± 12.556 | 1431.806 ± 13.077 | **206.523 ± 17.049** |
| str_nw_mera_open_26 | 1726.535 ± 15.675 | 1761.130 ± 6.979 | 1231.796 ± 20.262 | **342.244 ± 15.194** |
| tensornetwork_permutation_focus_step409_316 | 813.135 ± 20.511 | 1165.719 ± 33.858 | 859.065 ± 12.581 | **208.476 ± 12.433** |
| tensornetwork_permutation_light_415 | 802.140 ± 21.175 | 1110.512 ± 21.345 | 831.107 ± 12.438 | **243.097 ± 25.744** |

## Threads: 4

### Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `linux-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/linux-cpu/cpu/einsum/20260728_015909/run.yaml`
- Timestamp: `20260728_015909`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/linux-cpu/cpu/einsum/20260728_015909`.

- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

#### CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

#### Thread Environment

- OMP_NUM_THREADS: `4`
- OMP_THREAD_LIMIT: `4`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `4`
- OPENBLAS_NUM_THREADS: `4`
- GOTO_NUM_THREADS: `4`
- MKL_NUM_THREADS: `4`
- VECLIB_MAXIMUM_THREADS: `4`
- VECLIB_NUM_THREADS: `4`
- NUMEXPR_NUM_THREADS: `4`
- BLIS_NUM_THREADS: `4`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4`

#### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 4

- Source table: `data/results/linux-cpu/cpu/einsum/20260728_015909/einsum_table_t4_20260728_015909.md`

Logs:

- `data/results/linux-cpu/cpu/einsum/20260728_015909/tenferro_trace_t4_20260728_015909.log`
- `data/results/linux-cpu/cpu/einsum/20260728_015909/tenferro_eager_t4_20260728_015909.log`
- `data/results/linux-cpu/cpu/einsum/20260728_015909/pytorch_cpu_t4_20260728_015909.log`
- `data/results/linux-cpu/cpu/einsum/20260728_015909/jax_cpu_t4_20260728_015909.log`

###### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 18.171 ± 0.109 | 6.363 ± 0.631 | 1.212 ± 0.014 | **1.016 ± 0.139** |
| bin_batched_matmul_b32_m64_n64_k64 | 4.539 ± 0.251 | 3.274 ± 0.102 | **0.240 ± 0.007** | 0.713 ± 0.107 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.288 ± 1.312 | 0.256 ± 2.258 | **0.156 ± 0.011** | 0.727 ± 0.081 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.320 ± 1.253 | 0.638 ± 0.086 | **0.150 ± 0.006** | 0.713 ± 0.119 |
| bin_elementwise_mul_2048x2048 | **5.881 ± 0.817** | 10.628 ± 0.419 | 8.337 ± 0.824 | 12.208 ± 0.279 |
| bin_matmul_1024 | 77.708 ± 0.422 | 47.225 ± 2.571 | 24.180 ± 2.396 | **7.625 ± 0.646** |
| bin_matmul_256 | 3.886 ± 0.726 | 1.890 ± 0.013 | 0.824 ± 0.076 | **0.486 ± 0.038** |
| bin_outer_product_4096 | **19.382 ± 0.448** | 20.847 ± 1.355 | 29.856 ± 2.495 | 25.658 ± 5.686 |
| gm_queen5_5_3.wcsp | 5523.191 ± 1170.990 | 4552.404 ± 49.954 | **2111.001 ± 10.920** | - |
| lm_batch_likelihood_brackets_4_4d | 200.484 ± 12.068 | 134.517 ± 6.116 | **15.540 ± 0.465** | 26.034 ± 2.735 |
| lm_batch_likelihood_sentence_3_12d | 180.591 ± 84.638 | 475.117 ± 128.201 | 42.337 ± 1.202 | **33.829 ± 2.428** |
| lm_batch_likelihood_sentence_4_4d | 160.133 ± 26.114 | 258.083 ± 128.896 | **16.670 ± 1.445** | 27.522 ± 0.808 |
| nary_matmul_chain_64 | 0.677 ± 0.009 | 0.748 ± 0.041 | **0.141 ± 0.009** | 0.274 ± 0.014 |
| str_matrix_chain_multiplication_100 | 20.409 ± 0.556 | 44.592 ± 1.176 | **8.405 ± 0.614** | 20.879 ± 0.891 |
| str_mps_varying_inner_product_200 | 118.706 ± 22.563 | 148.116 ± 5.744 | **25.376 ± 1.847** | 48.075 ± 1.129 |
| str_nw_mera_closed_120 | 2384.341 ± 52.316 | 2223.581 ± 291.331 | 604.207 ± 8.664 | **331.531 ± 20.247** |
| str_nw_mera_open_26 | 1520.238 ± 36.555 | 2301.592 ± 132.318 | 409.863 ± 10.333 | **322.839 ± 10.893** |
| tensornetwork_permutation_focus_step409_316 | 1134.838 ± 52.751 | 1098.381 ± 33.636 | 273.121 ± 10.274 | **209.715 ± 6.959** |
| tensornetwork_permutation_light_415 | 1097.645 ± 44.505 | 1417.837 ± 47.368 | 272.294 ± 7.386 | **221.635 ± 16.963** |

###### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 18.171 ± 0.109 | 6.363 ± 0.631 | 1.212 ± 0.014 | **1.016 ± 0.139** |
| bin_batched_matmul_b32_m64_n64_k64 | 4.539 ± 0.251 | 3.274 ± 0.102 | **0.240 ± 0.007** | 0.713 ± 0.107 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.288 ± 1.312 | 0.256 ± 2.258 | **0.156 ± 0.011** | 0.727 ± 0.081 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.320 ± 1.253 | 0.638 ± 0.086 | **0.150 ± 0.006** | 0.713 ± 0.119 |
| bin_elementwise_mul_2048x2048 | **5.881 ± 0.817** | 10.628 ± 0.419 | 8.337 ± 0.824 | 12.208 ± 0.279 |
| bin_matmul_1024 | 77.708 ± 0.422 | 47.225 ± 2.571 | 24.180 ± 2.396 | **7.625 ± 0.646** |
| bin_matmul_256 | 3.886 ± 0.726 | 1.890 ± 0.013 | 0.824 ± 0.076 | **0.486 ± 0.038** |
| bin_outer_product_4096 | **19.382 ± 0.448** | 20.847 ± 1.355 | 29.856 ± 2.495 | 25.658 ± 5.686 |
| gm_queen5_5_3.wcsp | 1982.258 ± 60.117 | 2321.538 ± 64.498 | **764.319 ± 14.252** | - |
| lm_batch_likelihood_brackets_4_4d | 162.528 ± 14.993 | 191.976 ± 93.575 | **14.738 ± 0.729** | 21.824 ± 1.868 |
| lm_batch_likelihood_sentence_3_12d | 314.887 ± 21.607 | 308.679 ± 7.067 | **20.806 ± 2.072** | 28.212 ± 1.344 |
| lm_batch_likelihood_sentence_4_4d | 152.729 ± 15.329 | 214.640 ± 135.336 | **15.524 ± 0.259** | 21.758 ± 2.860 |
| nary_matmul_chain_64 | 0.677 ± 0.009 | 0.748 ± 0.041 | **0.141 ± 0.009** | 0.274 ± 0.014 |
| str_matrix_chain_multiplication_100 | 60.176 ± 17.034 | 27.187 ± 4.312 | **8.665 ± 1.003** | 20.146 ± 0.712 |
| str_mps_varying_inner_product_200 | 72.007 ± 3.508 | 106.093 ± 61.673 | **27.688 ± 1.337** | 45.127 ± 0.849 |
| str_nw_mera_closed_120 | 1864.544 ± 72.333 | 1725.116 ± 78.543 | 448.656 ± 6.918 | **241.863 ± 17.520** |
| str_nw_mera_open_26 | 1500.272 ± 34.570 | 2121.960 ± 107.965 | 412.498 ± 13.603 | **314.505 ± 9.665** |
| tensornetwork_permutation_focus_step409_316 | 1134.838 ± 52.751 | 1098.381 ± 33.636 | 273.121 ± 10.274 | **209.715 ± 6.959** |
| tensornetwork_permutation_light_415 | 1097.645 ± 44.505 | 1417.837 ± 47.368 | 272.294 ± 7.386 | **221.635 ± 16.963** |

# Linux CPU Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `linux-cpu`
- Thread runs: `1 4`

Mirrored from `amd-cpu` runs (`BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh <THREADS>`);
regenerate with `./scripts/mirror_linux_cpu_cpu_reports.sh 1:20260723_035222 4:20260723_042457 permutation:20260723_051703`.

This report is generated from sequential CPU runs. Do not compare it with
measurements collected while another CPU benchmark process was running.

## Run Inputs

- Threads 1: timestamp `20260723_035222`, raw run `data/results/linux-cpu/cpu/einsum/20260723_035222`
- Threads 4: timestamp `20260723_042457`, raw run `data/results/linux-cpu/cpu/einsum/20260723_042457`

## Threads: 1

### Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `linux-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/linux-cpu/cpu/einsum/20260723_035222/run.yaml`
- Timestamp: `20260723_035222`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/linux-cpu/cpu/einsum/20260723_035222`.

- tenferro-rs commit: `68855c2b65b5adc42dccca9bac04fd136a8f14c8`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`

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
- BLAS version: `2026.0.0`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 1

- Source table: `data/results/linux-cpu/cpu/einsum/20260723_035222/einsum_table_t1_20260723_035222.md`

Logs:

- `data/results/linux-cpu/cpu/einsum/20260723_035222/tenferro_trace_t1_20260723_035222.log`
- `data/results/linux-cpu/cpu/einsum/20260723_035222/tenferro_eager_t1_20260723_035222.log`
- `data/results/linux-cpu/cpu/einsum/20260723_035222/pytorch_cpu_t1_20260723_035222.log`
- `data/results/linux-cpu/cpu/einsum/20260723_035222/jax_cpu_t1_20260723_035222.log`

###### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.624 ± 0.202 | 3.430 ± 0.195 | 8.776 ± 0.092 | **1.994 ± 0.119** |
| bin_batched_matmul_b32_m64_n64_k64 | 0.481 ± 0.082 | **0.474 ± 0.019** | 1.311 ± 0.032 | 1.494 ± 0.020 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.646 ± 0.026** | 0.659 ± 0.030 | 0.959 ± 0.110 | 1.423 ± 0.049 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.655 ± 0.024** | 0.686 ± 0.018 | 0.845 ± 0.341 | 1.487 ± 0.065 |
| bin_elementwise_mul_2048x2048 | **6.614 ± 0.304** | 19.953 ± 0.662 | 35.880 ± 0.922 | 14.158 ± 4.129 |
| bin_matmul_1024 | 51.173 ± 17.108 | 50.665 ± 16.656 | 104.393 ± 1.440 | **11.414 ± 2.905** |
| bin_matmul_256 | 0.913 ± 0.053 | **0.882 ± 0.038** | 2.118 ± 0.165 | 1.221 ± 0.079 |
| bin_outer_product_4096 | 60.125 ± 2.365 | 60.928 ± 4.241 | 113.812 ± 0.872 | **44.797 ± 3.697** |
| gm_queen5_5_3.wcsp | 12978.589 ± 162.088 | **10306.465 ± 214.476** | 15286.641 ± 1001.995 | - |
| lm_batch_likelihood_brackets_4_4d | **30.472 ± 0.388** | 98.331 ± 4.405 | 89.193 ± 2.663 | 34.302 ± 0.861 |
| lm_batch_likelihood_sentence_3_12d | 72.510 ± 0.556 | 323.402 ± 3.251 | 96.673 ± 48.797 | **47.112 ± 5.539** |
| lm_batch_likelihood_sentence_4_4d | **27.709 ± 0.436** | 106.052 ± 1.102 | 57.743 ± 1.671 | 38.238 ± 1.167 |
| nary_matmul_chain_64 | **0.053 ± 0.023** | 0.075 ± 0.002 | 0.205 ± 0.011 | 0.661 ± 0.019 |
| str_matrix_chain_multiplication_100 | **12.215 ± 0.475** | 20.872 ± 0.247 | 29.285 ± 0.523 | 42.852 ± 1.793 |
| str_mps_varying_inner_product_200 | **15.591 ± 0.360** | 46.705 ± 7.391 | 37.918 ± 0.340 | 85.489 ± 1.473 |
| str_nw_mera_closed_120 | 1982.580 ± 24.757 | 2756.998 ± 272.527 | 2345.325 ± 53.218 | **605.297 ± 34.281** |
| str_nw_mera_open_26 | 1307.156 ± 4.470 | 2497.934 ± 241.434 | 1492.519 ± 10.721 | **574.203 ± 45.470** |
| tensornetwork_permutation_focus_step409_316 | 622.172 ± 5.287 | 2259.141 ± 1007.459 | 948.809 ± 6.525 | **454.800 ± 9.792** |
| tensornetwork_permutation_light_415 | 629.851 ± 11.201 | 1965.813 ± 90.214 | 920.540 ± 4.941 | **490.452 ± 11.804** |

###### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.624 ± 0.202 | 3.430 ± 0.195 | 8.776 ± 0.092 | **1.994 ± 0.119** |
| bin_batched_matmul_b32_m64_n64_k64 | 0.481 ± 0.082 | **0.474 ± 0.019** | 1.311 ± 0.032 | 1.494 ± 0.020 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.646 ± 0.026** | 0.659 ± 0.030 | 0.959 ± 0.110 | 1.423 ± 0.049 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.655 ± 0.024** | 0.686 ± 0.018 | 0.845 ± 0.341 | 1.487 ± 0.065 |
| bin_elementwise_mul_2048x2048 | **6.614 ± 0.304** | 19.953 ± 0.662 | 35.880 ± 0.922 | 14.158 ± 4.129 |
| bin_matmul_1024 | 51.173 ± 17.108 | 50.665 ± 16.656 | 104.393 ± 1.440 | **11.414 ± 2.905** |
| bin_matmul_256 | 0.913 ± 0.053 | **0.882 ± 0.038** | 2.118 ± 0.165 | 1.221 ± 0.079 |
| bin_outer_product_4096 | 60.125 ± 2.365 | 60.928 ± 4.241 | 113.812 ± 0.872 | **44.797 ± 3.697** |
| gm_queen5_5_3.wcsp | 3772.948 ± 48.421 | 8168.234 ± 3008.751 | **3258.033 ± 156.705** | - |
| lm_batch_likelihood_brackets_4_4d | **31.926 ± 0.697** | 89.062 ± 4.699 | 44.535 ± 1.864 | 35.918 ± 1.952 |
| lm_batch_likelihood_sentence_3_12d | 74.655 ± 0.881 | 456.414 ± 28.216 | 66.248 ± 3.008 | **45.710 ± 11.454** |
| lm_batch_likelihood_sentence_4_4d | **34.823 ± 0.529** | 168.541 ± 12.644 | 45.704 ± 1.182 | 38.157 ± 2.016 |
| nary_matmul_chain_64 | **0.053 ± 0.023** | 0.075 ± 0.002 | 0.205 ± 0.011 | 0.661 ± 0.019 |
| str_matrix_chain_multiplication_100 | **18.310 ± 0.300** | 39.661 ± 2.079 | 25.898 ± 0.239 | 43.096 ± 1.662 |
| str_mps_varying_inner_product_200 | **21.913 ± 0.649** | 65.429 ± 1.606 | 41.790 ± 0.352 | 95.674 ± 1.583 |
| str_nw_mera_closed_120 | 1644.739 ± 4.880 | 3474.437 ± 70.271 | 1664.635 ± 17.381 | **453.223 ± 38.219** |
| str_nw_mera_open_26 | 1339.760 ± 6.435 | 4049.808 ± 253.573 | 1337.081 ± 5.152 | **573.960 ± 67.894** |
| tensornetwork_permutation_focus_step409_316 | 622.172 ± 5.287 | 2259.141 ± 1007.459 | 948.809 ± 6.525 | **454.800 ± 9.792** |
| tensornetwork_permutation_light_415 | 629.851 ± 11.201 | 1965.813 ± 90.214 | 920.540 ± 4.941 | **490.452 ± 11.804** |

## Threads: 4

### Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `linux-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/linux-cpu/cpu/einsum/20260723_042457/run.yaml`
- Timestamp: `20260723_042457`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/linux-cpu/cpu/einsum/20260723_042457`.

- tenferro-rs commit: `68855c2b65b5adc42dccca9bac04fd136a8f14c8`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`

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
- BLAS version: `2026.0.0`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 4

- Source table: `data/results/linux-cpu/cpu/einsum/20260723_042457/einsum_table_t4_20260723_042457.md`

Logs:

- `data/results/linux-cpu/cpu/einsum/20260723_042457/tenferro_trace_t4_20260723_042457.log`
- `data/results/linux-cpu/cpu/einsum/20260723_042457/tenferro_eager_t4_20260723_042457.log`
- `data/results/linux-cpu/cpu/einsum/20260723_042457/pytorch_cpu_t4_20260723_042457.log`
- `data/results/linux-cpu/cpu/einsum/20260723_042457/jax_cpu_t4_20260723_042457.log`

###### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | **1.149 ± 0.023** | 1.178 ± 0.023 | 1.373 ± 0.014 | 1.862 ± 0.096 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.429 ± 0.015 | **0.274 ± 0.029** | 0.275 ± 0.006 | 1.415 ± 0.041 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.218 ± 0.005 | **0.207 ± 0.056** | 0.260 ± 0.011 | 1.464 ± 0.060 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.221 ± 1.248 | **0.210 ± 1.153** | 0.262 ± 0.011 | 1.500 ± 0.032 |
| bin_elementwise_mul_2048x2048 | **3.172 ± 0.370** | 7.581 ± 0.300 | 10.211 ± 0.455 | 14.248 ± 1.661 |
| bin_matmul_1024 | 15.715 ± 0.148 | 15.306 ± 0.213 | 16.150 ± 1.603 | **11.294 ± 3.433** |
| bin_matmul_256 | **0.281 ± 0.006** | 0.286 ± 0.034 | 0.387 ± 0.011 | 1.067 ± 0.140 |
| bin_outer_product_4096 | 22.891 ± 1.561 | **22.800 ± 2.087** | 30.562 ± 0.339 | 40.935 ± 2.260 |
| gm_queen5_5_3.wcsp | 6013.174 ± 578.278 | 4984.180 ± 141.605 | **2781.355 ± 667.556** | - |
| lm_batch_likelihood_brackets_4_4d | 28.619 ± 0.340 | 83.900 ± 1.554 | **18.354 ± 0.532** | 43.779 ± 2.101 |
| lm_batch_likelihood_sentence_3_12d | 48.603 ± 1.474 | 189.784 ± 1.340 | **24.302 ± 6.323** | 49.028 ± 8.362 |
| lm_batch_likelihood_sentence_4_4d | 26.851 ± 0.407 | 94.551 ± 8.733 | **17.640 ± 0.622** | 42.707 ± 1.057 |
| nary_matmul_chain_64 | **0.030 ± 0.003** | 0.079 ± 0.006 | 0.177 ± 0.014 | 0.770 ± 0.020 |
| str_matrix_chain_multiplication_100 | **5.112 ± 0.413** | 7.518 ± 0.151 | 12.788 ± 0.719 | 48.580 ± 4.011 |
| str_mps_varying_inner_product_200 | **21.036 ± 0.324** | 55.395 ± 0.520 | 27.397 ± 1.540 | 97.504 ± 0.505 |
| str_nw_mera_closed_120 | 820.816 ± 13.576 | 1429.528 ± 73.669 | 763.287 ± 198.515 | **602.125 ± 56.347** |
| str_nw_mera_open_26 | 536.790 ± 7.050 | 1231.131 ± 51.138 | **478.273 ± 19.019** | 537.561 ± 55.541 |
| tensornetwork_permutation_focus_step409_316 | 307.457 ± 3.231 | 781.039 ± 50.222 | **294.754 ± 2.904** | 493.748 ± 19.610 |
| tensornetwork_permutation_light_415 | 324.471 ± 4.733 | 778.383 ± 7.365 | **303.218 ± 7.861** | 524.296 ± 20.059 |

###### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | **1.149 ± 0.023** | 1.178 ± 0.023 | 1.373 ± 0.014 | 1.862 ± 0.096 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.429 ± 0.015 | **0.274 ± 0.029** | 0.275 ± 0.006 | 1.415 ± 0.041 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.218 ± 0.005 | **0.207 ± 0.056** | 0.260 ± 0.011 | 1.464 ± 0.060 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.221 ± 1.248 | **0.210 ± 1.153** | 0.262 ± 0.011 | 1.500 ± 0.032 |
| bin_elementwise_mul_2048x2048 | **3.172 ± 0.370** | 7.581 ± 0.300 | 10.211 ± 0.455 | 14.248 ± 1.661 |
| bin_matmul_1024 | 15.715 ± 0.148 | 15.306 ± 0.213 | 16.150 ± 1.603 | **11.294 ± 3.433** |
| bin_matmul_256 | **0.281 ± 0.006** | 0.286 ± 0.034 | 0.387 ± 0.011 | 1.067 ± 0.140 |
| bin_outer_product_4096 | 22.891 ± 1.561 | **22.800 ± 2.087** | 30.562 ± 0.339 | 40.935 ± 2.260 |
| gm_queen5_5_3.wcsp | 1701.905 ± 80.549 | 2371.906 ± 126.623 | **1155.221 ± 28.672** | - |
| lm_batch_likelihood_brackets_4_4d | 31.148 ± 0.856 | 103.277 ± 8.412 | **18.985 ± 0.882** | 45.334 ± 1.084 |
| lm_batch_likelihood_sentence_3_12d | 37.775 ± 0.364 | 232.795 ± 0.658 | **24.089 ± 0.331** | 47.545 ± 0.705 |
| lm_batch_likelihood_sentence_4_4d | 33.581 ± 0.651 | 102.031 ± 0.609 | **19.034 ± 0.988** | 37.908 ± 1.354 |
| nary_matmul_chain_64 | **0.030 ± 0.003** | 0.079 ± 0.006 | 0.177 ± 0.014 | 0.770 ± 0.020 |
| str_matrix_chain_multiplication_100 | **6.266 ± 0.175** | 9.284 ± 0.127 | 13.210 ± 0.274 | 48.613 ± 1.929 |
| str_mps_varying_inner_product_200 | **30.475 ± 1.833** | 84.611 ± 1.498 | 39.219 ± 0.653 | 115.303 ± 3.246 |
| str_nw_mera_closed_120 | 642.931 ± 2.904 | 1037.770 ± 81.444 | 590.036 ± 17.750 | **421.069 ± 12.045** |
| str_nw_mera_open_26 | 552.250 ± 5.234 | 1237.961 ± 38.663 | **513.953 ± 11.675** | 562.037 ± 54.459 |
| tensornetwork_permutation_focus_step409_316 | 307.457 ± 3.231 | 781.039 ± 50.222 | **294.754 ± 2.904** | 493.748 ± 19.610 |
| tensornetwork_permutation_light_415 | 324.471 ± 4.733 | 778.383 ± 7.365 | **303.218 ± 7.861** | 524.296 ± 20.059 |

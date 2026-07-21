# Linux CPU Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `linux-cpu`
- Thread runs: `1 4`

Mirrored from `amd-cpu` runs (`BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh <THREADS>`);
regenerate with `./scripts/mirror_linux_cpu_cpu_reports.sh 1:20260721_064813 4:20260721_071519 permutation:20260721_073112`.

This report is generated from sequential CPU runs. Do not compare it with
measurements collected while another CPU benchmark process was running.

## Run Inputs

- Threads 1: timestamp `20260721_064813`, raw run `data/results/linux-cpu/cpu/einsum/20260721_064813`
- Threads 4: timestamp `20260721_071519`, raw run `data/results/linux-cpu/cpu/einsum/20260721_071519`

## Threads: 1

### Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `linux-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/linux-cpu/cpu/einsum/20260721_064813/run.yaml`
- Timestamp: `20260721_064813`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/linux-cpu/cpu/einsum/20260721_064813`.

- tenferro-rs commit: `85855e272b1495611deb601a9ee06f3546772c3c`

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

- Source table: `data/results/linux-cpu/cpu/einsum/20260721_064813/einsum_table_t1_20260721_064813.md`

Logs:

- `data/results/linux-cpu/cpu/einsum/20260721_064813/tenferro_trace_t1_20260721_064813.log`
- `data/results/linux-cpu/cpu/einsum/20260721_064813/tenferro_eager_t1_20260721_064813.log`
- `data/results/linux-cpu/cpu/einsum/20260721_064813/pytorch_cpu_t1_20260721_064813.log`
- `data/results/linux-cpu/cpu/einsum/20260721_064813/jax_cpu_t1_20260721_064813.log`

###### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.385 ± 0.083 | 3.213 ± 0.015 | 3.564 ± 0.015 | **1.817 ± 0.340** |
| bin_batched_matmul_b32_m64_n64_k64 | **0.449 ± 0.038** | 0.475 ± 0.031 | 0.577 ± 0.039 | 1.549 ± 0.059 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.667 ± 0.013 | 0.657 ± 0.029 | **0.546 ± 0.009** | 1.456 ± 0.072 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.670 ± 0.012 | 0.673 ± 0.013 | **0.547 ± 0.029** | 1.545 ± 0.080 |
| bin_elementwise_mul_2048x2048 | **5.653 ± 0.146** | 20.491 ± 0.343 | 20.345 ± 0.275 | 15.612 ± 4.273 |
| bin_matmul_1024 | 44.670 ± 0.628 | 48.397 ± 19.553 | 45.804 ± 0.160 | **11.760 ± 4.432** |
| bin_matmul_256 | 0.810 ± 0.053 | **0.797 ± 0.028** | 0.886 ± 0.004 | 1.471 ± 0.071 |
| bin_outer_product_4096 | 56.838 ± 0.456 | 57.094 ± 0.670 | 60.311 ± 1.543 | **34.378 ± 4.324** |
| gm_queen5_5_3.wcsp | 12739.611 ± 63.791 | 10042.722 ± 124.236 | **6539.445 ± 15.169** | - |
| lm_batch_likelihood_brackets_4_4d | **30.168 ± 0.287** | 116.924 ± 5.578 | 35.927 ± 0.369 | 35.741 ± 1.058 |
| lm_batch_likelihood_sentence_3_12d | 69.537 ± 0.726 | 335.327 ± 4.674 | 93.320 ± 0.978 | **46.893 ± 4.444** |
| lm_batch_likelihood_sentence_4_4d | **27.187 ± 0.595** | 126.425 ± 1.755 | 38.394 ± 0.268 | 38.634 ± 0.756 |
| nary_matmul_chain_64 | **0.045 ± 0.015** | 0.096 ± 0.002 | 0.169 ± 0.007 | 0.753 ± 0.056 |
| str_matrix_chain_multiplication_100 | **11.849 ± 0.486** | 18.504 ± 0.358 | 22.265 ± 0.250 | 45.518 ± 2.726 |
| str_mps_varying_inner_product_200 | **14.973 ± 0.402** | 59.023 ± 0.682 | 29.702 ± 0.224 | 92.572 ± 1.417 |
| str_nw_mera_closed_120 | 1903.551 ± 10.159 | 2497.232 ± 8.588 | 1914.300 ± 4.901 | **609.222 ± 39.523** |
| str_nw_mera_open_26 | 1252.849 ± 5.638 | 2167.426 ± 16.613 | 1277.277 ± 23.402 | **555.020 ± 68.873** |
| tensornetwork_permutation_focus_step409_316 | 598.625 ± 1.695 | 1216.178 ± 9.001 | 904.850 ± 1.801 | **494.969 ± 25.709** |
| tensornetwork_permutation_light_415 | 603.747 ± 2.333 | 1080.295 ± 5.053 | 880.003 ± 10.010 | **474.452 ± 43.523** |

###### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.385 ± 0.083 | 3.213 ± 0.015 | 3.564 ± 0.015 | **1.817 ± 0.340** |
| bin_batched_matmul_b32_m64_n64_k64 | **0.449 ± 0.038** | 0.475 ± 0.031 | 0.577 ± 0.039 | 1.549 ± 0.059 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.667 ± 0.013 | 0.657 ± 0.029 | **0.546 ± 0.009** | 1.456 ± 0.072 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.670 ± 0.012 | 0.673 ± 0.013 | **0.547 ± 0.029** | 1.545 ± 0.080 |
| bin_elementwise_mul_2048x2048 | **5.653 ± 0.146** | 20.491 ± 0.343 | 20.345 ± 0.275 | 15.612 ± 4.273 |
| bin_matmul_1024 | 44.670 ± 0.628 | 48.397 ± 19.553 | 45.804 ± 0.160 | **11.760 ± 4.432** |
| bin_matmul_256 | 0.810 ± 0.053 | **0.797 ± 0.028** | 0.886 ± 0.004 | 1.471 ± 0.071 |
| bin_outer_product_4096 | 56.838 ± 0.456 | 57.094 ± 0.670 | 60.311 ± 1.543 | **34.378 ± 4.324** |
| gm_queen5_5_3.wcsp | 3764.960 ± 88.885 | 4576.571 ± 15.723 | **2834.428 ± 16.650** | - |
| lm_batch_likelihood_brackets_4_4d | **31.398 ± 0.288** | 91.812 ± 16.389 | 39.969 ± 0.286 | 36.474 ± 2.346 |
| lm_batch_likelihood_sentence_3_12d | 71.204 ± 0.724 | 372.862 ± 4.509 | 97.141 ± 0.546 | **54.688 ± 7.994** |
| lm_batch_likelihood_sentence_4_4d | **34.270 ± 0.553** | 126.347 ± 3.657 | 40.763 ± 0.435 | 38.193 ± 0.983 |
| nary_matmul_chain_64 | **0.045 ± 0.015** | 0.096 ± 0.002 | 0.169 ± 0.007 | 0.753 ± 0.056 |
| str_matrix_chain_multiplication_100 | **17.161 ± 0.512** | 24.276 ± 0.236 | 23.874 ± 0.370 | 47.356 ± 1.264 |
| str_mps_varying_inner_product_200 | **20.701 ± 0.454** | 76.224 ± 1.283 | 36.717 ± 1.027 | 110.645 ± 0.972 |
| str_nw_mera_closed_120 | 1560.460 ± 5.922 | 1964.061 ± 9.590 | 1514.520 ± 4.915 | **443.081 ± 35.338** |
| str_nw_mera_open_26 | 1272.714 ± 3.028 | 2119.559 ± 5.873 | 1315.795 ± 3.367 | **546.564 ± 59.147** |
| tensornetwork_permutation_focus_step409_316 | 598.625 ± 1.695 | 1216.178 ± 9.001 | 904.850 ± 1.801 | **494.969 ± 25.709** |
| tensornetwork_permutation_light_415 | 603.747 ± 2.333 | 1080.295 ± 5.053 | 880.003 ± 10.010 | **474.452 ± 43.523** |

## Threads: 4

### Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `linux-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/linux-cpu/cpu/einsum/20260721_071519/run.yaml`
- Timestamp: `20260721_071519`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/linux-cpu/cpu/einsum/20260721_071519`.

- tenferro-rs commit: `85855e272b1495611deb601a9ee06f3546772c3c`

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

- Source table: `data/results/linux-cpu/cpu/einsum/20260721_071519/einsum_table_t4_20260721_071519.md`

Logs:

- `data/results/linux-cpu/cpu/einsum/20260721_071519/tenferro_trace_t4_20260721_071519.log`
- `data/results/linux-cpu/cpu/einsum/20260721_071519/tenferro_eager_t4_20260721_071519.log`
- `data/results/linux-cpu/cpu/einsum/20260721_071519/pytorch_cpu_t4_20260721_071519.log`
- `data/results/linux-cpu/cpu/einsum/20260721_071519/jax_cpu_t4_20260721_071519.log`

###### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | **1.156 ± 0.039** | 1.200 ± 0.021 | 1.348 ± 0.020 | 2.083 ± 0.143 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.254 ± 0.026** | 0.331 ± 0.033 | 0.264 ± 0.014 | 1.558 ± 0.026 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.201 ± 0.021** | 0.218 ± 0.136 | 0.249 ± 0.021 | 1.488 ± 0.097 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.210 ± 1.207** | 0.214 ± 0.100 | 0.247 ± 0.024 | 1.558 ± 0.061 |
| bin_elementwise_mul_2048x2048 | **2.275 ± 0.028** | 7.500 ± 0.261 | 8.633 ± 0.471 | 15.502 ± 0.287 |
| bin_matmul_1024 | 17.996 ± 0.128 | 13.491 ± 0.266 | 14.427 ± 0.425 | **11.260 ± 4.863** |
| bin_matmul_256 | 0.316 ± 0.007 | **0.275 ± 0.035** | 0.352 ± 0.028 | 1.468 ± 0.114 |
| bin_outer_product_4096 | 23.857 ± 4.473 | **21.916 ± 1.366** | 26.428 ± 0.421 | 48.747 ± 1.828 |
| gm_queen5_5_3.wcsp | 5879.462 ± 128.559 | 5148.654 ± 112.480 | **2390.457 ± 46.732** | - |
| lm_batch_likelihood_brackets_4_4d | 28.721 ± 0.807 | 107.734 ± 1.221 | **16.436 ± 0.283** | 44.272 ± 2.539 |
| lm_batch_likelihood_sentence_3_12d | 46.078 ± 0.776 | 233.413 ± 4.760 | **20.201 ± 0.543** | 56.502 ± 3.394 |
| lm_batch_likelihood_sentence_4_4d | 26.295 ± 0.394 | 116.143 ± 0.581 | **17.269 ± 0.387** | 43.042 ± 1.612 |
| nary_matmul_chain_64 | **0.028 ± 0.003** | 0.078 ± 0.002 | 0.168 ± 0.021 | 0.737 ± 0.021 |
| str_matrix_chain_multiplication_100 | **4.219 ± 0.163** | 6.972 ± 0.137 | 11.845 ± 0.282 | 46.618 ± 1.567 |
| str_mps_varying_inner_product_200 | **19.599 ± 0.478** | 71.353 ± 0.790 | 25.611 ± 0.251 | 93.689 ± 1.082 |
| str_nw_mera_closed_120 | 808.200 ± 9.722 | 1589.785 ± 42.105 | 695.179 ± 8.218 | **611.569 ± 25.545** |
| str_nw_mera_open_26 | 524.267 ± 2.996 | 1264.420 ± 8.399 | **452.991 ± 4.513** | 536.458 ± 54.561 |
| tensornetwork_permutation_focus_step409_316 | 313.984 ± 3.744 | 878.825 ± 7.626 | **295.601 ± 2.151** | 498.718 ± 23.899 |
| tensornetwork_permutation_light_415 | 321.662 ± 3.306 | 891.039 ± 11.073 | **299.575 ± 9.735** | 523.041 ± 49.676 |

###### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | **1.156 ± 0.039** | 1.200 ± 0.021 | 1.348 ± 0.020 | 2.083 ± 0.143 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.254 ± 0.026** | 0.331 ± 0.033 | 0.264 ± 0.014 | 1.558 ± 0.026 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.201 ± 0.021** | 0.218 ± 0.136 | 0.249 ± 0.021 | 1.488 ± 0.097 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.210 ± 1.207** | 0.214 ± 0.100 | 0.247 ± 0.024 | 1.558 ± 0.061 |
| bin_elementwise_mul_2048x2048 | **2.275 ± 0.028** | 7.500 ± 0.261 | 8.633 ± 0.471 | 15.502 ± 0.287 |
| bin_matmul_1024 | 17.996 ± 0.128 | 13.491 ± 0.266 | 14.427 ± 0.425 | **11.260 ± 4.863** |
| bin_matmul_256 | 0.316 ± 0.007 | **0.275 ± 0.035** | 0.352 ± 0.028 | 1.468 ± 0.114 |
| bin_outer_product_4096 | 23.857 ± 4.473 | **21.916 ± 1.366** | 26.428 ± 0.421 | 48.747 ± 1.828 |
| gm_queen5_5_3.wcsp | 1792.235 ± 32.550 | 2442.400 ± 28.840 | **1034.708 ± 6.580** | - |
| lm_batch_likelihood_brackets_4_4d | 34.239 ± 0.144 | 116.992 ± 4.495 | **16.637 ± 0.456** | 46.297 ± 1.679 |
| lm_batch_likelihood_sentence_3_12d | 36.847 ± 0.562 | 247.186 ± 0.570 | **22.075 ± 0.149** | 55.776 ± 7.004 |
| lm_batch_likelihood_sentence_4_4d | 32.678 ± 0.275 | 123.538 ± 0.376 | **17.244 ± 1.276** | 37.910 ± 0.891 |
| nary_matmul_chain_64 | **0.028 ± 0.003** | 0.078 ± 0.002 | 0.168 ± 0.021 | 0.737 ± 0.021 |
| str_matrix_chain_multiplication_100 | **6.057 ± 0.253** | 9.053 ± 0.128 | 11.678 ± 0.330 | 45.918 ± 1.708 |
| str_mps_varying_inner_product_200 | **31.647 ± 1.501** | 87.214 ± 0.756 | 34.163 ± 0.401 | 108.311 ± 3.061 |
| str_nw_mera_closed_120 | 619.916 ± 9.662 | 1162.008 ± 116.783 | 537.960 ± 2.547 | **454.953 ± 35.655** |
| str_nw_mera_open_26 | 530.615 ± 5.202 | 1254.397 ± 11.520 | **457.482 ± 3.323** | 594.543 ± 58.199 |
| tensornetwork_permutation_focus_step409_316 | 313.984 ± 3.744 | 878.825 ± 7.626 | **295.601 ± 2.151** | 498.718 ± 23.899 |
| tensornetwork_permutation_light_415 | 321.662 ± 3.306 | 891.039 ± 11.073 | **299.575 ± 9.735** | 523.041 ± 49.676 |

# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20260721_071519/run.yaml`
- Timestamp: `20260721_071519`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20260721_071519`.

- tenferro-rs commit: `85855e272b1495611deb601a9ee06f3546772c3c`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`

## Thread Environment

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

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.0`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- Source table: `data/results/amd-cpu/cpu/einsum/20260721_071519/einsum_table_t4_20260721_071519.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20260721_071519/tenferro_trace_t4_20260721_071519.log`
- `data/results/amd-cpu/cpu/einsum/20260721_071519/tenferro_eager_t4_20260721_071519.log`
- `data/results/amd-cpu/cpu/einsum/20260721_071519/pytorch_cpu_t4_20260721_071519.log`
- `data/results/amd-cpu/cpu/einsum/20260721_071519/jax_cpu_t4_20260721_071519.log`

#### Strategy: opt_flops

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

#### Strategy: opt_size

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

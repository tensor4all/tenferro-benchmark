# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20260716_081717/run.yaml`
- Timestamp: `20260716_081717`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20260716_081717`.

- tenferro-rs commit: `8c267c636537d3d407925b61a2000f9bbe07e6a1`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-134-generic-x86_64-with-glibc2.39`

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

- tenferro-rs features: `system-openblas`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `openblas`
- BLAS version: `OpenBLAS 0.3.26`
- BLAS root: `/opt/openblas`
- BLAS library: `/opt/openblas/lib/libopenblas.so`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- Source table: `data/results/amd-cpu/cpu/einsum/20260716_081717/einsum_table_t4_20260716_081717.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20260716_081717/tenferro_trace_t4_20260716_081717.log`
- `data/results/amd-cpu/cpu/einsum/20260716_081717/tenferro_eager_t4_20260716_081717.log`
- `data/results/amd-cpu/cpu/einsum/20260716_081717/pytorch_cpu_t4_20260716_081717.log`
- `data/results/amd-cpu/cpu/einsum/20260716_081717/jax_cpu_t4_20260716_081717.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.968 ± 0.054 | 2.118 ± 0.032 | **1.385 ± 0.030** | 2.116 ± 0.055 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.669 ± 0.039 | 0.930 ± 0.025 | **0.291 ± 0.032** | 1.525 ± 0.084 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.189 ± 0.013** | 0.223 ± 0.196 | 0.269 ± 0.014 | 1.498 ± 0.098 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.203 ± 0.069** | **0.203 ± 0.022** | 0.268 ± 0.022 | 1.514 ± 0.067 |
| bin_elementwise_mul_2048x2048 | **1.193 ± 0.024** | 6.672 ± 0.915 | 8.977 ± 0.472 | 13.434 ± 6.634 |
| bin_matmul_1024 | 14.834 ± 0.030 | 19.111 ± 2.860 | 15.533 ± 0.619 | **11.978 ± 3.648** |
| bin_matmul_256 | **0.320 ± 0.036** | 0.429 ± 0.048 | 0.392 ± 0.013 | 1.495 ± 0.078 |
| bin_outer_product_4096 | **20.612 ± 2.257** | 25.999 ± 2.945 | 24.844 ± 0.201 | 48.199 ± 2.068 |
| gm_queen5_5_3.wcsp | 5809.651 ± 163.364 | 4795.538 ± 60.149 | **2281.762 ± 22.997** | - |
| lm_batch_likelihood_brackets_4_4d | 45.673 ± 0.435 | 114.479 ± 3.106 | **16.154 ± 0.144** | 43.854 ± 2.578 |
| lm_batch_likelihood_sentence_3_12d | 60.976 ± 0.569 | 287.080 ± 37.848 | **20.448 ± 0.211** | 50.841 ± 4.510 |
| lm_batch_likelihood_sentence_4_4d | 33.845 ± 0.393 | 83.847 ± 1.531 | **16.990 ± 0.189** | 42.062 ± 1.051 |
| nary_matmul_chain_64 | **0.054 ± 0.007** | 0.158 ± 0.058 | 0.175 ± 0.015 | 0.731 ± 0.017 |
| str_matrix_chain_multiplication_100 | **8.169 ± 0.072** | 19.766 ± 0.725 | 11.358 ± 0.159 | 43.927 ± 3.011 |
| str_mps_varying_inner_product_200 | **24.112 ± 0.361** | 63.232 ± 1.868 | 25.103 ± 0.446 | 96.807 ± 1.897 |
| str_nw_mera_closed_120 | 995.644 ± 8.920 | 1417.557 ± 20.372 | 678.834 ± 8.407 | **582.046 ± 18.437** |
| str_nw_mera_open_26 | 568.463 ± 5.358 | 1305.853 ± 7.813 | **441.868 ± 3.150** | 540.744 ± 63.341 |
| tensornetwork_permutation_focus_step409_316 | 372.804 ± 4.809 | 708.049 ± 53.434 | **275.543 ± 2.427** | 480.240 ± 13.725 |
| tensornetwork_permutation_light_415 | 379.231 ± 3.660 | 779.813 ± 18.791 | **283.386 ± 1.907** | 468.994 ± 32.229 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.968 ± 0.054 | 2.118 ± 0.032 | **1.385 ± 0.030** | 2.116 ± 0.055 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.669 ± 0.039 | 0.930 ± 0.025 | **0.291 ± 0.032** | 1.525 ± 0.084 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.189 ± 0.013** | 0.223 ± 0.196 | 0.269 ± 0.014 | 1.498 ± 0.098 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.203 ± 0.069** | **0.203 ± 0.022** | 0.268 ± 0.022 | 1.514 ± 0.067 |
| bin_elementwise_mul_2048x2048 | **1.193 ± 0.024** | 6.672 ± 0.915 | 8.977 ± 0.472 | 13.434 ± 6.634 |
| bin_matmul_1024 | 14.834 ± 0.030 | 19.111 ± 2.860 | 15.533 ± 0.619 | **11.978 ± 3.648** |
| bin_matmul_256 | **0.320 ± 0.036** | 0.429 ± 0.048 | 0.392 ± 0.013 | 1.495 ± 0.078 |
| bin_outer_product_4096 | **20.612 ± 2.257** | 25.999 ± 2.945 | 24.844 ± 0.201 | 48.199 ± 2.068 |
| gm_queen5_5_3.wcsp | 2042.097 ± 58.516 | 2321.246 ± 23.389 | **1001.506 ± 10.204** | - |
| lm_batch_likelihood_brackets_4_4d | 36.795 ± 0.401 | 78.324 ± 4.100 | **16.832 ± 0.292** | 44.563 ± 0.901 |
| lm_batch_likelihood_sentence_3_12d | 77.559 ± 0.541 | 209.585 ± 10.954 | **22.525 ± 0.158** | 48.756 ± 0.710 |
| lm_batch_likelihood_sentence_4_4d | 40.136 ± 0.389 | 86.742 ± 2.416 | **17.533 ± 0.272** | 37.907 ± 0.844 |
| nary_matmul_chain_64 | **0.054 ± 0.007** | 0.158 ± 0.058 | 0.175 ± 0.015 | 0.731 ± 0.017 |
| str_matrix_chain_multiplication_100 | **8.234 ± 0.053** | 20.476 ± 1.595 | 11.623 ± 0.124 | 44.470 ± 2.598 |
| str_mps_varying_inner_product_200 | **30.992 ± 0.170** | 58.860 ± 1.192 | 33.969 ± 0.354 | 103.364 ± 5.293 |
| str_nw_mera_closed_120 | 749.072 ± 5.120 | 1056.055 ± 39.731 | 535.247 ± 18.672 | **479.136 ± 52.937** |
| str_nw_mera_open_26 | 601.253 ± 6.434 | 1238.445 ± 6.028 | **452.229 ± 2.160** | 543.972 ± 66.232 |
| tensornetwork_permutation_focus_step409_316 | 372.804 ± 4.809 | 708.049 ± 53.434 | **275.543 ± 2.427** | 480.240 ± 13.725 |
| tensornetwork_permutation_light_415 | 379.231 ± 3.660 | 779.813 ± 18.791 | **283.386 ± 1.907** | 468.994 ± 32.229 |

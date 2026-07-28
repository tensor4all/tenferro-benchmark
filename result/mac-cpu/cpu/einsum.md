# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260728_100627/run.yaml`
- Timestamp: `20260728_100627`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260728_100627`.

- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

## CPU Information

- Model: `Apple M4`
- Vendor: `Apple`
- Logical CPUs: `10`
- Physical CPUs: `10`
- Sockets: `1`
- Cores per socket: `10`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Performance: 4 physical / 4 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 4 CPUs/L2); Efficiency: 6 physical / 6 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.2-arm64-arm-64bit`

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

- tenferro-rs features: `system-accelerate`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 4

- Source table: `data/results/mac-cpu/cpu/einsum/20260728_100627/einsum_table_t4_20260728_100627.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260728_100627/tenferro_trace_t4_20260728_100627.log`
- `data/results/mac-cpu/cpu/einsum/20260728_100627/tenferro_eager_t4_20260728_100627.log`
- `data/results/mac-cpu/cpu/einsum/20260728_100627/pytorch_cpu_t4_20260728_100627.log`
- `data/results/mac-cpu/cpu/einsum/20260728_100627/jax_cpu_t4_20260728_100627.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.865 ± 0.178 | 0.561 ± 0.078 | **0.405 ± 0.007** | 0.629 ± 0.050 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.155 ± 0.034 | **0.094 ± 0.005** | 0.099 ± 0.008 | 0.194 ± 0.017 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.110 ± 0.011 | 0.130 ± 0.033 | **0.096 ± 0.003** | 0.135 ± 0.030 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.163 ± 0.011 | 0.133 ± 0.017 | **0.107 ± 0.005** | 0.150 ± 0.019 |
| bin_elementwise_mul_2048x2048 | 1.233 ± 0.092 | 1.210 ± 0.084 | **1.119 ± 0.062** | 1.532 ± 0.037 |
| bin_matmul_1024 | 5.273 ± 0.096 | 4.739 ± 0.162 | **4.379 ± 0.075** | 7.979 ± 0.616 |
| bin_matmul_256 | 0.130 ± 0.008 | 0.091 ± 0.001 | **0.086 ± 0.001** | 0.266 ± 0.025 |
| bin_outer_product_4096 | **1.777 ± 0.028** | 1.797 ± 0.078 | 1.803 ± 0.060 | 1.905 ± 0.141 |
| gm_queen5_5_3.wcsp | 691.927 ± 35.689 | 763.248 ± 36.270 | **654.675 ± 90.630** | 812.936 ± 108.915 |
| lm_batch_likelihood_brackets_4_4d | 8.997 ± 0.317 | 14.382 ± 1.956 | 10.847 ± 3.185 | **7.275 ± 0.229** |
| lm_batch_likelihood_sentence_3_12d | 14.363 ± 0.363 | 25.089 ± 2.373 | 24.621 ± 1.611 | **12.090 ± 0.341** |
| lm_batch_likelihood_sentence_4_4d | 9.228 ± 0.695 | 16.343 ± 0.549 | 9.610 ± 0.930 | **7.318 ± 0.369** |
| nary_matmul_chain_64 | 0.039 ± 0.019 | 0.032 ± 0.001 | **0.030 ± 0.002** | 0.085 ± 0.004 |
| str_matrix_chain_multiplication_100 | **2.903 ± 0.132** | 4.086 ± 0.160 | 3.185 ± 0.097 | 7.176 ± 0.244 |
| str_mps_varying_inner_product_200 | 7.206 ± 0.169 | 13.151 ± 0.244 | **6.468 ± 0.305** | 15.121 ± 0.268 |
| str_nw_mera_closed_120 | 166.124 ± 0.566 | 174.089 ± 16.452 | **148.611 ± 6.641** | 369.009 ± 119.774 |
| str_nw_mera_open_26 | 159.343 ± 2.464 | 169.556 ± 6.434 | **124.079 ± 7.923** | 283.661 ± 16.811 |
| tensornetwork_permutation_focus_step409_316 | **84.095 ± 15.248** | 98.837 ± 14.739 | 98.140 ± 5.585 | 128.573 ± 10.311 |
| tensornetwork_permutation_light_415 | **72.746 ± 14.908** | 121.196 ± 5.045 | 98.842 ± 3.139 | 157.064 ± 3.665 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.865 ± 0.178 | 0.561 ± 0.078 | **0.405 ± 0.007** | 0.629 ± 0.050 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.155 ± 0.034 | **0.094 ± 0.005** | 0.099 ± 0.008 | 0.194 ± 0.017 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.110 ± 0.011 | 0.130 ± 0.033 | **0.096 ± 0.003** | 0.135 ± 0.030 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.163 ± 0.011 | 0.133 ± 0.017 | **0.107 ± 0.005** | 0.150 ± 0.019 |
| bin_elementwise_mul_2048x2048 | 1.233 ± 0.092 | 1.210 ± 0.084 | **1.119 ± 0.062** | 1.532 ± 0.037 |
| bin_matmul_1024 | 5.273 ± 0.096 | 4.739 ± 0.162 | **4.379 ± 0.075** | 7.979 ± 0.616 |
| bin_matmul_256 | 0.130 ± 0.008 | 0.091 ± 0.001 | **0.086 ± 0.001** | 0.266 ± 0.025 |
| bin_outer_product_4096 | **1.777 ± 0.028** | 1.797 ± 0.078 | 1.803 ± 0.060 | 1.905 ± 0.141 |
| gm_queen5_5_3.wcsp | **184.349 ± 21.848** | 281.121 ± 13.975 | 254.245 ± 5.347 | 374.315 ± 15.906 |
| lm_batch_likelihood_brackets_4_4d | 9.939 ± 0.429 | 14.743 ± 0.762 | 9.312 ± 0.381 | **7.690 ± 0.406** |
| lm_batch_likelihood_sentence_3_12d | 15.676 ± 0.946 | 23.112 ± 0.794 | 23.448 ± 1.141 | **13.839 ± 0.860** |
| lm_batch_likelihood_sentence_4_4d | 10.279 ± 0.500 | 15.015 ± 0.304 | 9.290 ± 0.744 | **7.080 ± 0.310** |
| nary_matmul_chain_64 | 0.039 ± 0.019 | 0.032 ± 0.001 | **0.030 ± 0.002** | 0.085 ± 0.004 |
| str_matrix_chain_multiplication_100 | **2.868 ± 0.173** | 3.993 ± 0.129 | 3.173 ± 0.085 | 8.474 ± 0.579 |
| str_mps_varying_inner_product_200 | **7.486 ± 0.249** | 13.399 ± 0.254 | 7.665 ± 0.315 | 16.533 ± 0.194 |
| str_nw_mera_closed_120 | 139.720 ± 1.226 | 141.201 ± 0.980 | **126.118 ± 0.363** | 401.041 ± 33.258 |
| str_nw_mera_open_26 | 166.008 ± 4.107 | 164.788 ± 1.397 | **119.180 ± 1.590** | 290.024 ± 9.123 |
| tensornetwork_permutation_focus_step409_316 | **84.095 ± 15.248** | 98.837 ± 14.739 | 98.140 ± 5.585 | 128.573 ± 10.311 |
| tensornetwork_permutation_light_415 | **72.746 ± 14.908** | 121.196 ± 5.045 | 98.842 ± 3.139 | 157.064 ± 3.665 |

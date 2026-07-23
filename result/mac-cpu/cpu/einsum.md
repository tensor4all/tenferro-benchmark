# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260723_121605/run.yaml`
- Timestamp: `20260723_121605`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260723_121605`.

- tenferro-rs commit: `68855c2b65b5adc42dccca9bac04fd136a8f14c8`

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

- Source table: `data/results/mac-cpu/cpu/einsum/20260723_121605/einsum_table_t4_20260723_121605.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260723_121605/tenferro_trace_t4_20260723_121605.log`
- `data/results/mac-cpu/cpu/einsum/20260723_121605/tenferro_eager_t4_20260723_121605.log`
- `data/results/mac-cpu/cpu/einsum/20260723_121605/pytorch_cpu_t4_20260723_121605.log`
- `data/results/mac-cpu/cpu/einsum/20260723_121605/jax_cpu_t4_20260723_121605.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.619 ± 0.086 | 0.664 ± 0.472 | **0.409 ± 0.008** | 0.626 ± 0.052 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.080 ± 0.006** | 0.090 ± 0.014 | 0.100 ± 0.003 | 0.192 ± 0.036 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.116 ± 0.013 | 0.115 ± 0.021 | **0.097 ± 0.006** | 0.157 ± 0.049 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.115 ± 0.009 | 0.125 ± 0.007 | **0.107 ± 0.015** | 0.153 ± 0.030 |
| bin_elementwise_mul_2048x2048 | 1.902 ± 0.241 | 1.145 ± 0.090 | **1.133 ± 0.098** | 1.535 ± 0.067 |
| bin_matmul_1024 | 4.632 ± 0.122 | 4.588 ± 0.107 | **4.408 ± 0.051** | 7.795 ± 0.679 |
| bin_matmul_256 | **0.077 ± 0.001** | **0.077 ± 0.003** | 0.086 ± 0.001 | 0.300 ± 0.078 |
| bin_outer_product_4096 | 1.818 ± 0.094 | **1.784 ± 0.110** | 1.814 ± 0.125 | 1.891 ± 0.024 |
| gm_queen5_5_3.wcsp | 1357.167 ± 107.515 | 982.503 ± 43.472 | **827.239 ± 37.313** | 951.820 ± 68.662 |
| lm_batch_likelihood_brackets_4_4d | 9.845 ± 0.831 | 18.882 ± 1.562 | 12.159 ± 0.604 | **7.886 ± 0.346** |
| lm_batch_likelihood_sentence_3_12d | 18.618 ± 0.568 | 37.382 ± 1.194 | 29.401 ± 1.672 | **13.047 ± 1.431** |
| lm_batch_likelihood_sentence_4_4d | 8.560 ± 0.510 | 18.488 ± 1.883 | 9.496 ± 0.263 | **8.021 ± 0.395** |
| nary_matmul_chain_64 | **0.006 ± 0.000** | 0.009 ± 0.000 | 0.031 ± 0.001 | 0.102 ± 0.006 |
| str_matrix_chain_multiplication_100 | **2.808 ± 0.084** | 2.958 ± 0.150 | 3.257 ± 0.066 | 7.251 ± 0.133 |
| str_mps_varying_inner_product_200 | **5.996 ± 0.317** | 10.743 ± 0.266 | 6.822 ± 0.189 | 15.273 ± 0.292 |
| str_nw_mera_closed_120 | 182.829 ± 11.934 | 233.022 ± 12.515 | **180.958 ± 16.530** | 442.205 ± 52.668 |
| str_nw_mera_open_26 | 147.139 ± 10.022 | 208.081 ± 17.767 | **146.464 ± 7.385** | 325.239 ± 21.434 |
| tensornetwork_permutation_focus_step409_316 | 139.573 ± 3.769 | 170.062 ± 17.478 | **120.442 ± 12.182** | 145.669 ± 28.835 |
| tensornetwork_permutation_light_415 | **97.957 ± 2.627** | 173.019 ± 11.113 | 120.296 ± 5.303 | 173.975 ± 10.213 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.619 ± 0.086 | 0.664 ± 0.472 | **0.409 ± 0.008** | 0.626 ± 0.052 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.080 ± 0.006** | 0.090 ± 0.014 | 0.100 ± 0.003 | 0.192 ± 0.036 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.116 ± 0.013 | 0.115 ± 0.021 | **0.097 ± 0.006** | 0.157 ± 0.049 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.115 ± 0.009 | 0.125 ± 0.007 | **0.107 ± 0.015** | 0.153 ± 0.030 |
| bin_elementwise_mul_2048x2048 | 1.902 ± 0.241 | 1.145 ± 0.090 | **1.133 ± 0.098** | 1.535 ± 0.067 |
| bin_matmul_1024 | 4.632 ± 0.122 | 4.588 ± 0.107 | **4.408 ± 0.051** | 7.795 ± 0.679 |
| bin_matmul_256 | **0.077 ± 0.001** | **0.077 ± 0.003** | 0.086 ± 0.001 | 0.300 ± 0.078 |
| bin_outer_product_4096 | 1.818 ± 0.094 | **1.784 ± 0.110** | 1.814 ± 0.125 | 1.891 ± 0.024 |
| gm_queen5_5_3.wcsp | 311.137 ± 19.188 | 445.476 ± 29.250 | **296.187 ± 10.415** | 437.713 ± 53.133 |
| lm_batch_likelihood_brackets_4_4d | 12.337 ± 1.077 | 19.080 ± 2.014 | **11.179 ± 0.653** | 12.692 ± 3.236 |
| lm_batch_likelihood_sentence_3_12d | 19.967 ± 0.911 | 38.218 ± 1.375 | 27.907 ± 1.722 | **16.486 ± 4.730** |
| lm_batch_likelihood_sentence_4_4d | 11.293 ± 1.021 | 18.692 ± 1.810 | **9.267 ± 0.431** | 18.046 ± 6.388 |
| nary_matmul_chain_64 | **0.006 ± 0.000** | 0.009 ± 0.000 | 0.031 ± 0.001 | 0.102 ± 0.006 |
| str_matrix_chain_multiplication_100 | **2.918 ± 0.100** | 3.068 ± 0.136 | 3.258 ± 0.090 | 15.023 ± 0.922 |
| str_mps_varying_inner_product_200 | **7.943 ± 0.590** | 13.317 ± 0.320 | 8.227 ± 0.256 | 21.647 ± 2.654 |
| str_nw_mera_closed_120 | 159.217 ± 8.580 | 184.110 ± 3.855 | **146.380 ± 7.747** | 453.026 ± 51.191 |
| str_nw_mera_open_26 | 153.220 ± 29.871 | 222.100 ± 6.705 | **146.658 ± 2.328** | 357.608 ± 55.079 |
| tensornetwork_permutation_focus_step409_316 | 139.573 ± 3.769 | 170.062 ± 17.478 | **120.442 ± 12.182** | 145.669 ± 28.835 |
| tensornetwork_permutation_light_415 | **97.957 ± 2.627** | 173.019 ± 11.113 | 120.296 ± 5.303 | 173.975 ± 10.213 |

# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260716_171807/run.yaml`
- Timestamp: `20260716_171807`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260716_171807`.

- tenferro-rs commit: `3abc9108e4f0500f4e75519711cdb4e21b9625df`

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

- Source table: `data/results/mac-cpu/cpu/einsum/20260716_171807/einsum_table_t4_20260716_171807.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260716_171807/tenferro_trace_t4_20260716_171807.log`
- `data/results/mac-cpu/cpu/einsum/20260716_171807/tenferro_eager_t4_20260716_171807.log`
- `data/results/mac-cpu/cpu/einsum/20260716_171807/pytorch_cpu_t4_20260716_171807.log`
- `data/results/mac-cpu/cpu/einsum/20260716_171807/jax_cpu_t4_20260716_171807.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.596 ± 0.095 | 0.655 ± 0.141 | **0.450 ± 0.087** | 0.682 ± 0.093 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.080 ± 0.008** | 0.090 ± 0.001 | 0.096 ± 0.001 | 0.202 ± 0.039 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.157 ± 0.055 | 0.146 ± 0.046 | **0.104 ± 0.003** | 0.162 ± 0.031 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.143 ± 0.013 | 0.150 ± 0.022 | **0.111 ± 0.009** | 0.144 ± 0.020 |
| bin_elementwise_mul_2048x2048 | 1.857 ± 0.169 | 1.200 ± 0.027 | **1.161 ± 0.074** | 1.575 ± 0.092 |
| bin_matmul_1024 | 4.699 ± 0.193 | 4.698 ± 0.287 | **4.427 ± 0.225** | 7.979 ± 0.596 |
| bin_matmul_256 | **0.076 ± 0.000** | 0.078 ± 0.000 | 0.089 ± 0.007 | 0.336 ± 0.028 |
| bin_outer_product_4096 | 1.808 ± 0.123 | 1.805 ± 0.182 | 1.817 ± 0.273 | **1.742 ± 0.101** |
| gm_queen5_5_3.wcsp | 1359.353 ± 78.727 | 988.595 ± 124.045 | **817.447 ± 87.295** | 1035.891 ± 159.713 |
| lm_batch_likelihood_brackets_4_4d | 11.437 ± 0.658 | 19.345 ± 1.189 | 13.582 ± 0.930 | **9.156 ± 0.397** |
| lm_batch_likelihood_sentence_3_12d | 18.930 ± 0.940 | 36.669 ± 1.891 | 29.064 ± 1.268 | **13.544 ± 1.524** |
| lm_batch_likelihood_sentence_4_4d | 8.851 ± 1.107 | 18.709 ± 2.847 | 10.787 ± 1.201 | **7.752 ± 0.343** |
| nary_matmul_chain_64 | **0.007 ± 0.000** | 0.011 ± 0.001 | 0.035 ± 0.002 | 0.094 ± 0.003 |
| str_matrix_chain_multiplication_100 | **3.143 ± 0.422** | 3.212 ± 0.217 | 3.547 ± 0.141 | 7.940 ± 0.689 |
| str_mps_varying_inner_product_200 | **6.523 ± 0.436** | 11.711 ± 1.090 | 7.295 ± 0.321 | 15.192 ± 0.905 |
| str_nw_mera_closed_120 | 216.853 ± 22.333 | 233.594 ± 10.379 | **174.639 ± 2.282** | 409.409 ± 68.432 |
| str_nw_mera_open_26 | **143.483 ± 8.478** | 194.640 ± 10.518 | 151.097 ± 16.974 | 302.443 ± 16.261 |
| tensornetwork_permutation_focus_step409_316 | 136.059 ± 21.146 | 205.495 ± 5.529 | **117.617 ± 15.310** | 133.797 ± 21.746 |
| tensornetwork_permutation_light_415 | 138.417 ± 11.876 | 148.717 ± 16.626 | **126.067 ± 13.410** | 162.138 ± 4.600 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.596 ± 0.095 | 0.655 ± 0.141 | **0.450 ± 0.087** | 0.682 ± 0.093 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.080 ± 0.008** | 0.090 ± 0.001 | 0.096 ± 0.001 | 0.202 ± 0.039 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.157 ± 0.055 | 0.146 ± 0.046 | **0.104 ± 0.003** | 0.162 ± 0.031 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.143 ± 0.013 | 0.150 ± 0.022 | **0.111 ± 0.009** | 0.144 ± 0.020 |
| bin_elementwise_mul_2048x2048 | 1.857 ± 0.169 | 1.200 ± 0.027 | **1.161 ± 0.074** | 1.575 ± 0.092 |
| bin_matmul_1024 | 4.699 ± 0.193 | 4.698 ± 0.287 | **4.427 ± 0.225** | 7.979 ± 0.596 |
| bin_matmul_256 | **0.076 ± 0.000** | 0.078 ± 0.000 | 0.089 ± 0.007 | 0.336 ± 0.028 |
| bin_outer_product_4096 | 1.808 ± 0.123 | 1.805 ± 0.182 | 1.817 ± 0.273 | **1.742 ± 0.101** |
| gm_queen5_5_3.wcsp | 373.217 ± 32.312 | 346.852 ± 32.307 | **305.543 ± 32.944** | 409.244 ± 46.437 |
| lm_batch_likelihood_brackets_4_4d | 12.838 ± 0.391 | 16.655 ± 2.312 | 11.308 ± 0.645 | **8.346 ± 0.251** |
| lm_batch_likelihood_sentence_3_12d | 20.700 ± 1.080 | 35.840 ± 1.667 | 27.895 ± 1.660 | **14.781 ± 1.305** |
| lm_batch_likelihood_sentence_4_4d | 11.880 ± 0.438 | 16.094 ± 1.886 | 10.108 ± 2.046 | **7.227 ± 0.428** |
| nary_matmul_chain_64 | **0.007 ± 0.000** | 0.011 ± 0.001 | 0.035 ± 0.002 | 0.094 ± 0.003 |
| str_matrix_chain_multiplication_100 | 3.053 ± 0.268 | **2.959 ± 0.263** | 3.256 ± 0.115 | 8.635 ± 0.401 |
| str_mps_varying_inner_product_200 | 8.927 ± 0.626 | 10.685 ± 3.790 | **8.021 ± 0.593** | 20.802 ± 1.248 |
| str_nw_mera_closed_120 | 175.118 ± 13.849 | 177.225 ± 19.614 | **138.877 ± 25.644** | 465.365 ± 55.937 |
| str_nw_mera_open_26 | 162.545 ± 20.601 | 204.567 ± 21.988 | **154.526 ± 5.625** | 368.858 ± 25.873 |
| tensornetwork_permutation_focus_step409_316 | 136.059 ± 21.146 | 205.495 ± 5.529 | **117.617 ± 15.310** | 133.797 ± 21.746 |
| tensornetwork_permutation_light_415 | 138.417 ± 11.876 | 148.717 ± 16.626 | **126.067 ± 13.410** | 162.138 ± 4.600 |

# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260614_135639/run.yaml`
- Timestamp: `20260614_135639`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260614_135639`.

- tenferro-rs commit: `db1990549801351308b631aef1bbca292d11a457`

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
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

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

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 4

- Source table: `data/results/mac-cpu/cpu/einsum/20260614_135639/einsum_table_t4_20260614_135639.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260614_135639/tenferro_trace_t4_20260614_135639.log`
- `data/results/mac-cpu/cpu/einsum/20260614_135639/tenferro_eager_t4_20260614_135639.log`
- `data/results/mac-cpu/cpu/einsum/20260614_135639/pytorch_cpu_t4_20260614_135639.log`
- `data/results/mac-cpu/cpu/einsum/20260614_135639/jax_cpu_t4_20260614_135639.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.650 ± 0.102 | 0.678 ± 0.103 | **0.414 ± 0.020** | 0.711 ± 0.104 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.087 ± 0.002** | 0.099 ± 0.005 | 0.104 ± 0.009 | 0.211 ± 0.059 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.113 ± 0.039 | 0.141 ± 0.011 | **0.105 ± 0.018** | 0.130 ± 0.013 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.110 ± 0.018** | 0.145 ± 0.020 | 0.114 ± 0.002 | 0.153 ± 0.047 |
| bin_elementwise_mul_2048x2048 | **1.114 ± 0.119** | 1.180 ± 0.137 | 1.140 ± 0.116 | 1.620 ± 0.361 |
| bin_matmul_1024 | 4.600 ± 0.108 | 4.746 ± 0.193 | **4.494 ± 0.223** | 8.643 ± 0.283 |
| bin_matmul_256 | **0.076 ± 0.000** | 0.084 ± 0.002 | 0.086 ± 0.005 | 0.327 ± 0.096 |
| bin_outer_product_4096 | **1.790 ± 0.016** | 1.829 ± 0.027 | 1.846 ± 0.090 | 1.926 ± 0.119 |
| gm_queen5_5_3.wcsp | 703.064 ± 106.305 | 778.506 ± 53.910 | **674.161 ± 93.816** | 749.338 ± 372.194 |
| lm_batch_likelihood_brackets_4_4d | **7.833 ± 3.075** | 13.642 ± 0.839 | 9.546 ± 0.438 | 8.243 ± 0.569 |
| lm_batch_likelihood_sentence_3_12d | **12.310 ± 0.310** | 23.418 ± 0.462 | 26.295 ± 0.420 | 15.036 ± 1.192 |
| lm_batch_likelihood_sentence_4_4d | **7.093 ± 0.152** | 14.526 ± 0.486 | 9.455 ± 0.520 | 7.548 ± 0.312 |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.021 ± 0.003 | 0.032 ± 0.006 | 0.088 ± 0.002 |
| str_matrix_chain_multiplication_100 | **2.128 ± 0.026** | 3.439 ± 0.179 | 3.217 ± 0.098 | 7.317 ± 0.350 |
| str_mps_varying_inner_product_200 | **5.476 ± 0.306** | 10.115 ± 0.243 | 6.433 ± 0.308 | 15.282 ± 0.175 |
| str_nw_mera_closed_120 | **142.093 ± 0.397** | 165.986 ± 2.384 | 143.972 ± 0.976 | 279.845 ± 28.205 |
| str_nw_mera_open_26 | **117.047 ± 2.620** | 162.986 ± 9.761 | 118.469 ± 4.331 | 207.819 ± 9.762 |
| tensornetwork_permutation_focus_step409_316 | 85.617 ± 4.533 | 128.814 ± 15.092 | **81.456 ± 1.644** | 120.984 ± 7.239 |
| tensornetwork_permutation_light_415 | 89.095 ± 5.510 | 121.264 ± 5.718 | **86.916 ± 9.353** | 130.955 ± 5.696 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.650 ± 0.102 | 0.678 ± 0.103 | **0.414 ± 0.020** | 0.711 ± 0.104 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.087 ± 0.002** | 0.099 ± 0.005 | 0.104 ± 0.009 | 0.211 ± 0.059 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.113 ± 0.039 | 0.141 ± 0.011 | **0.105 ± 0.018** | 0.130 ± 0.013 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.110 ± 0.018** | 0.145 ± 0.020 | 0.114 ± 0.002 | 0.153 ± 0.047 |
| bin_elementwise_mul_2048x2048 | **1.114 ± 0.119** | 1.180 ± 0.137 | 1.140 ± 0.116 | 1.620 ± 0.361 |
| bin_matmul_1024 | 4.600 ± 0.108 | 4.746 ± 0.193 | **4.494 ± 0.223** | 8.643 ± 0.283 |
| bin_matmul_256 | **0.076 ± 0.000** | 0.084 ± 0.002 | 0.086 ± 0.005 | 0.327 ± 0.096 |
| bin_outer_product_4096 | **1.790 ± 0.016** | 1.829 ± 0.027 | 1.846 ± 0.090 | 1.926 ± 0.119 |
| gm_queen5_5_3.wcsp | **194.964 ± 10.665** | 260.728 ± 12.456 | 235.363 ± 7.430 | 268.437 ± 7.515 |
| lm_batch_likelihood_brackets_4_4d | 7.607 ± 0.714 | 16.301 ± 3.989 | 9.539 ± 0.174 | **7.353 ± 0.407** |
| lm_batch_likelihood_sentence_3_12d | **12.166 ± 0.387** | 25.630 ± 0.838 | 25.157 ± 0.314 | 13.540 ± 0.358 |
| lm_batch_likelihood_sentence_4_4d | 7.940 ± 0.164 | 15.266 ± 1.592 | 9.507 ± 0.666 | **6.664 ± 0.518** |
| nary_matmul_chain_64 | **0.005 ± 0.000** | 0.021 ± 0.003 | 0.032 ± 0.006 | 0.088 ± 0.002 |
| str_matrix_chain_multiplication_100 | **2.177 ± 0.044** | 3.335 ± 0.123 | 3.242 ± 0.099 | 7.257 ± 0.453 |
| str_mps_varying_inner_product_200 | **6.804 ± 0.649** | 10.035 ± 0.259 | 7.761 ± 0.232 | 16.670 ± 0.173 |
| str_nw_mera_closed_120 | 131.403 ± 17.464 | 140.983 ± 2.177 | **127.057 ± 0.735** | 250.004 ± 15.209 |
| str_nw_mera_open_26 | 123.722 ± 3.322 | 171.277 ± 14.813 | **120.279 ± 1.521** | 258.303 ± 11.740 |
| tensornetwork_permutation_focus_step409_316 | 85.617 ± 4.533 | 128.814 ± 15.092 | **81.456 ± 1.644** | 120.984 ± 7.239 |
| tensornetwork_permutation_light_415 | 89.095 ± 5.510 | 121.264 ± 5.718 | **86.916 ± 9.353** | 130.955 ± 5.696 |

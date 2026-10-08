# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261008_044146/run.yaml`
- Timestamp: `20261008_044146`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261008_044146`.

- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`

## CPU Information

- Model: `AMD Ryzen 9 9955HX 16-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `32`
- Sockets: `1`
- Cores per socket: `16`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=1`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Julia / OMEinsum.jl Backend

- Julia: version `1.13.1`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `1`

- `omeinsum-jl` (mode `omeinsum_path` in the log/report) always
  executes the instance's precomputed `opt_flops`/`opt_size` path via
  `OMEinsum.DynamicEinCode` pairwise contractions; OMEinsum's own
  contraction-order optimizer is never invoked, so the comparison
  against tenferro/PyTorch/JAX (which also use the precomputed path)
  stays fair.
- `JULIA_NUM_THREADS` also pins `LinearAlgebra.BLAS.set_num_threads`,
  matching the BLAS thread pinning used by the other CPU backends.
- Julia is column-major like tenferro-rs, so the einsum runner uses
  `format_string_colmajor` / `shapes_colmajor` directly with no
  PyTorch/JAX-style layout reconstruction.
- OMEinsum dispatches its pairwise contractions to Julia's own BLAS
  (libblastrampoline, OpenBLAS by default; the provider that ran is
  reported above). When that differs from the provider tenferro-rs and
  PyTorch link against, matmul-shaped rows partly compare BLAS
  implementations rather than einsum-runtime overhead; read them
  together with the recorded providers. `docs/einsum-suite.md` records
  the measured OMEinsum-over-its-own-BLAS overhead.

## Threads: 1

- Source table: `data/results/amd-cpu/cpu/einsum/20261008_044146/einsum_table_t1_20261008_044146.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20261008_044146/tenferro_trace_t1_20261008_044146.log`
- `data/results/amd-cpu/cpu/einsum/20261008_044146/tenferro_eager_t1_20261008_044146.log`
- `data/results/amd-cpu/cpu/einsum/20261008_044146/pytorch_cpu_t1_20261008_044146.log`
- `data/results/amd-cpu/cpu/einsum/20261008_044146/jax_cpu_t1_20261008_044146.log`
- `data/results/amd-cpu/cpu/einsum/20261008_044146/julia_omeinsum_t1_20261008_044146.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 8.041 ± 1.233 | 4.497 ± 0.023 | 4.499 ± 0.016 | **4.424 ± 0.034** | 7.573 ± 6.740 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.928 ± 0.013 | **0.643 ± 0.004** | 0.665 ± 0.005 | 0.696 ± 0.017 | 1.355 ± 0.858 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.228 ± 0.011 | 0.201 ± 0.010 | **0.145 ± 0.003** | 0.390 ± 0.013 | 1.866 ± 1.033 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.447 ± 0.020 | 0.203 ± 0.002 | **0.144 ± 0.002** | 0.389 ± 0.029 | 4.313 ± 8.472 |
| bin_elementwise_mul_2048x2048 | 8.014 ± 0.720 | **7.203 ± 0.057** | 8.546 ± 0.071 | 11.574 ± 0.109 | 9.267 ± 0.526 |
| bin_elementwise_mul_512x512 | 0.120 ± 0.028 | 0.109 ± 0.001 | **0.102 ± 0.004** | 0.215 ± 0.014 | **0.102 ± 0.013** |
| bin_matmul_1024 | 73.319 ± 0.251 | 60.173 ± 0.124 | 68.856 ± 0.120 | 60.932 ± 0.326 | **58.290 ± 21.616** |
| bin_matmul_256 | 1.407 ± 0.022 | 1.023 ± 0.004 | 1.418 ± 0.011 | 1.136 ± 0.154 | **0.931 ± 0.005** |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.037 ± 0.008 | 0.025 ± 0.001 | 0.024 ± 0.001 | 0.132 ± 0.015 | **0.012 ± 0.001** |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.086 ± 0.027 | **0.055 ± 0.002** | 0.112 ± 0.002 | 0.175 ± 0.018 | 0.061 ± 0.014 |
| bin_omeinsum_matmul_10x10 | 0.040 ± 0.001 | **0.012 ± 0.000** | 0.023 ± 0.000 | 0.081 ± 0.014 | 0.021 ± 0.001 |
| bin_outer_product_4096 | 32.203 ± 0.792 | 31.695 ± 0.212 | **29.972 ± 0.274** | 32.704 ± 0.515 | 248.725 ± 3.003 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 64.389 ± 0.071 | 61.233 ± 0.169 | 64.450 ± 0.152 | 63.156 ± 0.180 | **54.196 ± 21.077** |
| gm_queen5_5_3.wcsp | **3730.311 ± 29.708** | 5226.689 ± 24.361 | 5016.644 ± 24.283 | 3797.926 ± 29.786 | 6100.841 ± 38.125 |
| lm_batch_likelihood_brackets_4_4d | 29.651 ± 0.163 | 41.360 ± 0.186 | 28.332 ± 0.187 | **14.072 ± 0.147** | 21.846 ± 10.555 |
| lm_batch_likelihood_sentence_3_12d | **72.438 ± 0.290** | 94.953 ± 0.921 | 86.351 ± 0.463 | 75.642 ± 1.047 | 90.402 ± 39.068 |
| lm_batch_likelihood_sentence_4_4d | 30.104 ± 0.327 | 36.808 ± 0.061 | 31.382 ± 0.086 | **16.019 ± 0.139** | 31.566 ± 23.883 |
| nary_matmul_chain_64 | 0.081 ± 0.024 | **0.047 ± 0.000** | 0.063 ± 0.002 | 0.108 ± 0.014 | 0.071 ± 0.010 |
| str_matrix_chain_multiplication_100 | **12.198 ± 0.049** | 13.691 ± 0.112 | 14.764 ± 0.123 | 13.299 ± 0.129 | 13.238 ± 0.637 |
| str_mps_varying_inner_product_200 | **17.989 ± 0.197** | 21.493 ± 0.633 | 21.501 ± 0.115 | 20.941 ± 0.174 | 21.596 ± 8.797 |
| str_nw_mera_closed_120 | 1881.070 ± 15.787 | 1845.474 ± 22.745 | 1789.876 ± 4.668 | 1953.739 ± 3.886 | **1357.225 ± 13.432** |
| str_nw_mera_open_26 | 1589.292 ± 14.882 | 1350.582 ± 2.822 | **1144.561 ± 1.022** | 1521.554 ± 1.993 | 1206.042 ± 48.242 |
| tensornetwork_permutation_focus_step409_316 | **360.671 ± 1.948** | 507.990 ± 0.825 | 776.886 ± 1.904 | 580.989 ± 13.488 | 462.041 ± 43.418 |
| tensornetwork_permutation_light_415 | **351.515 ± 2.001** | 420.994 ± 2.438 | 736.212 ± 2.696 | 600.092 ± 17.738 | 444.987 ± 77.202 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 8.041 ± 1.233 | 4.497 ± 0.023 | 4.499 ± 0.016 | **4.424 ± 0.034** | 7.573 ± 6.740 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.928 ± 0.013 | **0.643 ± 0.004** | 0.665 ± 0.005 | 0.696 ± 0.017 | 1.355 ± 0.858 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.228 ± 0.011 | 0.201 ± 0.010 | **0.145 ± 0.003** | 0.390 ± 0.013 | 1.866 ± 1.033 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.447 ± 0.020 | 0.203 ± 0.002 | **0.144 ± 0.002** | 0.389 ± 0.029 | 4.313 ± 8.472 |
| bin_elementwise_mul_2048x2048 | 8.014 ± 0.720 | **7.203 ± 0.057** | 8.546 ± 0.071 | 11.574 ± 0.109 | 9.267 ± 0.526 |
| bin_elementwise_mul_512x512 | 0.120 ± 0.028 | 0.109 ± 0.001 | **0.102 ± 0.004** | 0.215 ± 0.014 | **0.102 ± 0.013** |
| bin_matmul_1024 | 73.319 ± 0.251 | 60.173 ± 0.124 | 68.856 ± 0.120 | 60.932 ± 0.326 | **58.290 ± 21.616** |
| bin_matmul_256 | 1.407 ± 0.022 | 1.023 ± 0.004 | 1.418 ± 0.011 | 1.136 ± 0.154 | **0.931 ± 0.005** |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.037 ± 0.008 | 0.025 ± 0.001 | 0.024 ± 0.001 | 0.132 ± 0.015 | **0.012 ± 0.001** |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.086 ± 0.027 | **0.055 ± 0.002** | 0.112 ± 0.002 | 0.175 ± 0.018 | 0.061 ± 0.014 |
| bin_omeinsum_matmul_10x10 | 0.040 ± 0.001 | **0.012 ± 0.000** | 0.023 ± 0.000 | 0.081 ± 0.014 | 0.021 ± 0.001 |
| bin_outer_product_4096 | 32.203 ± 0.792 | 31.695 ± 0.212 | **29.972 ± 0.274** | 32.704 ± 0.515 | 248.725 ± 3.003 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 64.389 ± 0.071 | 61.233 ± 0.169 | 64.450 ± 0.152 | 63.156 ± 0.180 | **54.196 ± 21.077** |
| gm_queen5_5_3.wcsp | **1517.324 ± 6.332** | 2044.036 ± 8.997 | 2145.622 ± 6.188 | 1608.049 ± 5.848 | 2317.757 ± 40.953 |
| lm_batch_likelihood_brackets_4_4d | 26.444 ± 0.094 | 31.902 ± 0.177 | 30.527 ± 0.120 | **15.014 ± 0.351** | 43.782 ± 24.807 |
| lm_batch_likelihood_sentence_3_12d | **74.271 ± 2.239** | 109.947 ± 0.317 | 87.391 ± 0.419 | 81.782 ± 0.761 | 91.309 ± 33.481 |
| lm_batch_likelihood_sentence_4_4d | 31.381 ± 0.299 | 37.568 ± 0.250 | 32.541 ± 0.188 | **15.049 ± 0.095** | 35.930 ± 34.565 |
| nary_matmul_chain_64 | 0.081 ± 0.024 | **0.047 ± 0.000** | 0.063 ± 0.002 | 0.108 ± 0.014 | 0.071 ± 0.010 |
| str_matrix_chain_multiplication_100 | 13.720 ± 0.499 | 13.884 ± 0.089 | 14.793 ± 0.118 | **12.985 ± 0.150** | 13.388 ± 0.673 |
| str_mps_varying_inner_product_200 | **16.634 ± 0.203** | 18.282 ± 0.091 | 20.223 ± 0.177 | 19.867 ± 0.331 | 18.182 ± 1.384 |
| str_nw_mera_closed_120 | 1609.278 ± 5.937 | 1618.480 ± 5.772 | 1521.999 ± 2.704 | 1693.001 ± 2.855 | **1201.490 ± 49.617** |
| str_nw_mera_open_26 | 1603.150 ± 4.701 | 1361.131 ± 3.967 | **1151.588 ± 1.545** | 1567.663 ± 2.097 | 1183.234 ± 45.784 |
| tensornetwork_permutation_focus_step409_316 | **360.671 ± 1.948** | 507.990 ± 0.825 | 776.886 ± 1.904 | 580.989 ± 13.488 | 462.041 ± 43.418 |
| tensornetwork_permutation_light_415 | **351.515 ± 2.001** | 420.994 ± 2.438 | 736.212 ± 2.696 | 600.092 ± 17.738 | 444.987 ± 77.202 |

## Collection recovery

Completed einsum measurements retained from the initial collection; CPU ops were recollected after the input-restoration fix. Component commits and original metadata are recorded in `run.yaml` and `einsum_run.yaml`. Exact commands: `data/results/amd-cpu/cpu/refresh/20261008_044136/commands_remaining.sh`.

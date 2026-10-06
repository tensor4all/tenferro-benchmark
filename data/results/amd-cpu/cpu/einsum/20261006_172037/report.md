# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261006_172037/run.yaml`
- Timestamp: `20261006_172037`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261006_172037`.

- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`

## CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/t4a-2010-bench-migrate/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
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

- Source table: `data/results/amd-cpu/cpu/einsum/20261006_172037/einsum_table_t1_20261006_172037.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20261006_172037/tenferro_trace_t1_20261006_172037.log`
- `data/results/amd-cpu/cpu/einsum/20261006_172037/tenferro_eager_t1_20261006_172037.log`
- `data/results/amd-cpu/cpu/einsum/20261006_172037/pytorch_cpu_t1_20261006_172037.log`
- `data/results/amd-cpu/cpu/einsum/20261006_172037/jax_cpu_t1_20261006_172037.log`
- `data/results/amd-cpu/cpu/einsum/20261006_172037/julia_omeinsum_t1_20261006_172037.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.620 ± 0.014 | 3.407 ± 0.007 | 3.538 ± 0.004 | **3.313 ± 0.300** | 11.178 ± 5.419 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.911 ± 0.007 | **0.512 ± 0.004** | 0.567 ± 0.005 | 1.272 ± 0.080 | 2.180 ± 1.639 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.268 ± 0.009 | 0.179 ± 0.013 | **0.175 ± 0.017** | 1.167 ± 0.036 | 3.778 ± 2.563 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.470 ± 0.007 | 0.217 ± 0.021 | **0.179 ± 0.030** | 1.160 ± 0.049 | 4.842 ± 4.117 |
| bin_elementwise_mul_2048x2048 | 18.337 ± 1.645 | **14.749 ± 0.526** | 18.141 ± 1.372 | 20.356 ± 0.655 | 18.922 ± 3.735 |
| bin_elementwise_mul_512x512 | 0.120 ± 0.004 | **0.108 ± 0.001** | 0.139 ± 0.017 | 0.621 ± 0.098 | 0.156 ± 0.061 |
| bin_matmul_1024 | 51.633 ± 0.695 | **42.010 ± 0.884** | 51.919 ± 1.043 | 54.600 ± 3.865 | 68.644 ± 14.849 |
| bin_matmul_256 | 1.132 ± 0.029 | **0.755 ± 0.037** | 1.166 ± 0.006 | 2.414 ± 0.101 | 1.460 ± 1.023 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.044 ± 0.004 | 0.027 ± 0.013 | 0.037 ± 0.004 | 0.420 ± 0.024 | **0.012 ± 0.001** |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.094 ± 0.078 | 0.062 ± 0.006 | 0.126 ± 0.005 | 0.527 ± 0.080 | **0.044 ± 0.004** |
| bin_omeinsum_matmul_10x10 | 0.062 ± 0.003 | 0.026 ± 0.001 | 0.036 ± 0.009 | 0.305 ± 0.024 | **0.023 ± 0.001** |
| bin_outer_product_4096 | 67.465 ± 5.941 | **60.096 ± 0.575** | 61.352 ± 1.274 | 67.682 ± 0.848 | 305.778 ± 5.302 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 49.776 ± 0.835 | **43.667 ± 1.420** | 54.958 ± 1.842 | 46.818 ± 1.508 | 67.913 ± 27.851 |
| gm_queen5_5_3.wcsp | **5608.750 ± 49.986** | 7669.158 ± 98.557 | 6395.060 ± 18.993 | - | 9539.658 ± 311.766 |
| lm_batch_likelihood_brackets_4_4d | 28.197 ± 0.434 | 26.561 ± 0.469 | 29.159 ± 0.115 | **12.865 ± 0.456** | 55.826 ± 27.613 |
| lm_batch_likelihood_sentence_3_12d | 62.609 ± 1.976 | 124.125 ± 3.225 | **58.426 ± 1.027** | 95.064 ± 2.157 | 101.833 ± 32.426 |
| lm_batch_likelihood_sentence_4_4d | 28.228 ± 1.627 | 30.137 ± 0.448 | 31.463 ± 0.250 | **13.657 ± 0.446** | 58.900 ± 24.983 |
| nary_matmul_chain_64 | 0.093 ± 0.003 | **0.041 ± 0.002** | 0.083 ± 0.009 | 0.169 ± 0.014 | 0.128 ± 0.030 |
| str_matrix_chain_multiplication_100 | **9.851 ± 0.095** | 13.531 ± 0.560 | 14.448 ± 0.677 | 13.457 ± 0.451 | 15.074 ± 0.573 |
| str_mps_varying_inner_product_200 | 16.866 ± 0.909 | **16.613 ± 0.680** | 21.813 ± 0.243 | 19.297 ± 0.701 | 23.710 ± 6.476 |
| str_nw_mera_closed_120 | 2016.123 ± 18.100 | 1982.778 ± 13.811 | 1903.403 ± 7.265 | 1967.592 ± 18.043 | **1733.861 ± 18.461** |
| str_nw_mera_open_26 | 1834.512 ± 26.168 | 1555.348 ± 10.919 | **1162.996 ± 4.665** | 1691.471 ± 16.689 | 1611.468 ± 95.427 |
| tensornetwork_permutation_focus_step409_316 | **404.080 ± 6.097** | 578.689 ± 5.471 | 816.437 ± 7.001 | 647.702 ± 27.804 | 600.297 ± 37.963 |
| tensornetwork_permutation_light_415 | **385.115 ± 6.983** | 548.462 ± 7.978 | 781.676 ± 4.375 | 691.410 ± 8.943 | 602.745 ± 30.415 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.620 ± 0.014 | 3.407 ± 0.007 | 3.538 ± 0.004 | **3.313 ± 0.300** | 11.178 ± 5.419 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.911 ± 0.007 | **0.512 ± 0.004** | 0.567 ± 0.005 | 1.272 ± 0.080 | 2.180 ± 1.639 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.268 ± 0.009 | 0.179 ± 0.013 | **0.175 ± 0.017** | 1.167 ± 0.036 | 3.778 ± 2.563 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.470 ± 0.007 | 0.217 ± 0.021 | **0.179 ± 0.030** | 1.160 ± 0.049 | 4.842 ± 4.117 |
| bin_elementwise_mul_2048x2048 | 18.337 ± 1.645 | **14.749 ± 0.526** | 18.141 ± 1.372 | 20.356 ± 0.655 | 18.922 ± 3.735 |
| bin_elementwise_mul_512x512 | 0.120 ± 0.004 | **0.108 ± 0.001** | 0.139 ± 0.017 | 0.621 ± 0.098 | 0.156 ± 0.061 |
| bin_matmul_1024 | 51.633 ± 0.695 | **42.010 ± 0.884** | 51.919 ± 1.043 | 54.600 ± 3.865 | 68.644 ± 14.849 |
| bin_matmul_256 | 1.132 ± 0.029 | **0.755 ± 0.037** | 1.166 ± 0.006 | 2.414 ± 0.101 | 1.460 ± 1.023 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.044 ± 0.004 | 0.027 ± 0.013 | 0.037 ± 0.004 | 0.420 ± 0.024 | **0.012 ± 0.001** |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.094 ± 0.078 | 0.062 ± 0.006 | 0.126 ± 0.005 | 0.527 ± 0.080 | **0.044 ± 0.004** |
| bin_omeinsum_matmul_10x10 | 0.062 ± 0.003 | 0.026 ± 0.001 | 0.036 ± 0.009 | 0.305 ± 0.024 | **0.023 ± 0.001** |
| bin_outer_product_4096 | 67.465 ± 5.941 | **60.096 ± 0.575** | 61.352 ± 1.274 | 67.682 ± 0.848 | 305.778 ± 5.302 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 49.776 ± 0.835 | **43.667 ± 1.420** | 54.958 ± 1.842 | 46.818 ± 1.508 | 67.913 ± 27.851 |
| gm_queen5_5_3.wcsp | **1893.987 ± 25.253** | 2634.418 ± 16.764 | 2333.014 ± 19.675 | - | 3315.315 ± 43.219 |
| lm_batch_likelihood_brackets_4_4d | 25.699 ± 0.630 | 26.398 ± 0.364 | 30.980 ± 0.310 | **12.717 ± 0.854** | 62.372 ± 25.967 |
| lm_batch_likelihood_sentence_3_12d | 65.768 ± 1.604 | 130.444 ± 2.205 | 64.602 ± 0.971 | **51.357 ± 1.550** | 104.678 ± 13.157 |
| lm_batch_likelihood_sentence_4_4d | 29.667 ± 1.638 | 31.227 ± 0.748 | 32.935 ± 0.182 | **13.018 ± 0.507** | 40.793 ± 33.157 |
| nary_matmul_chain_64 | 0.093 ± 0.003 | **0.041 ± 0.002** | 0.083 ± 0.009 | 0.169 ± 0.014 | 0.128 ± 0.030 |
| str_matrix_chain_multiplication_100 | 14.207 ± 0.093 | **13.417 ± 0.633** | 14.307 ± 0.376 | 15.697 ± 0.816 | 14.946 ± 0.777 |
| str_mps_varying_inner_product_200 | 17.341 ± 0.115 | **17.334 ± 0.681** | 22.059 ± 0.106 | 19.334 ± 0.283 | 22.281 ± 1.349 |
| str_nw_mera_closed_120 | 1514.243 ± 12.216 | 1519.665 ± 14.836 | 1478.847 ± 13.375 | 1594.256 ± 4.927 | **1423.466 ± 48.417** |
| str_nw_mera_open_26 | 1845.449 ± 10.063 | 1552.086 ± 16.298 | **1200.401 ± 14.504** | 1755.921 ± 11.999 | 1602.027 ± 44.038 |
| tensornetwork_permutation_focus_step409_316 | **404.080 ± 6.097** | 578.689 ± 5.471 | 816.437 ± 7.001 | 647.702 ± 27.804 | 600.297 ± 37.963 |
| tensornetwork_permutation_light_415 | **385.115 ± 6.983** | 548.462 ± 7.978 | 781.676 ± 4.375 | 691.410 ± 8.943 | 602.745 ± 30.415 |

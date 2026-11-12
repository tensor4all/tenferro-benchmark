# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_020537/run.yaml`
- Timestamp: `2026-11-12T17:05:37.788001+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261113_020537`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `disabled`
- Container: `none`

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

- tenferro-rs features: `native`
- BLAS implementation: `none`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
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

Logs:

- `data/results/zen5-cpu/cpu/einsum/20261113_020537/jax_cpu_t1_20261113_020537.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_020537/julia_omeinsum_t1_20261113_020537.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_020537/pytorch_cpu_t1_20261113_020537.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_020537/tenferro_eager_t1_20261113_020537.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_020537/tenferro_trace_t1_20261113_020537.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.229 ± 0.013 | **4.351 ± 0.009** | 5.982 ± 0.030 | 5.338 ± 0.035 | 13.307 ± 6.928 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.770 ± 0.004 | **0.553 ± 0.008** | 0.877 ± 0.002 | 0.895 ± 0.021 | 0.967 ± 1.216 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.990 ± 0.014 | 0.979 ± 0.008 | **0.185 ± 0.002** | 0.516 ± 0.037 | 2.572 ± 1.121 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.654 ± 0.010 | 0.981 ± 0.006 | **0.182 ± 0.004** | 0.522 ± 0.025 | 6.964 ± 7.608 |
| bin_elementwise_mul_2048x2048 | 1.926 ± 0.036 | **1.924 ± 0.029** | 11.422 ± 0.416 | 15.592 ± 0.144 | 12.903 ± 0.939 |
| bin_elementwise_mul_512x512 | 0.135 ± 0.030 | 0.124 ± 0.006 | 0.151 ± 0.001 | 0.339 ± 0.022 | **0.120 ± 0.561** |
| bin_matmul_1024 | 75.076 ± 0.283 | **68.761 ± 0.111** | 89.362 ± 0.155 | 77.286 ± 0.182 | 110.540 ± 24.786 |
| bin_matmul_256 | 1.216 ± 0.035 | **1.073 ± 0.003** | 1.838 ± 0.051 | 1.389 ± 0.015 | 1.771 ± 0.014 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.004 | **0.007 ± 0.000** | 0.029 ± 0.001 | 0.172 ± 0.013 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.036 ± 0.001 | **0.033 ± 0.001** | 0.145 ± 0.004 | 0.228 ± 0.025 | 0.051 ± 0.001 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.001 | **0.011 ± 0.000** | 0.029 ± 0.001 | 0.117 ± 0.016 | 0.027 ± 0.001 |
| bin_outer_product_4096 | 49.518 ± 0.813 | 49.535 ± 1.057 | **39.480 ± 0.578** | 40.985 ± 7.934 | 315.983 ± 2.170 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 73.671 ± 0.236 | **71.482 ± 0.322** | 85.061 ± 0.102 | 81.833 ± 0.301 | 101.781 ± 14.438 |
| gm_queen5_5_3.wcsp | 5454.580 ± 87.034 | 5378.205 ± 29.854 | 6498.289 ± 19.755 | **4906.584 ± 10.641** | 7871.360 ± 63.234 |
| lm_batch_likelihood_brackets_4_4d | **30.344 ± 0.104** | 35.429 ± 0.187 | 37.232 ± 0.294 | 30.817 ± 0.177 | 37.084 ± 15.839 |
| lm_batch_likelihood_sentence_3_12d | **72.000 ± 0.230** | 115.791 ± 1.578 | 87.096 ± 0.235 | 107.335 ± 0.550 | 143.858 ± 38.769 |
| lm_batch_likelihood_sentence_4_4d | **32.253 ± 0.235** | 37.312 ± 0.279 | 41.586 ± 0.231 | 32.580 ± 0.137 | 74.895 ± 31.939 |
| nary_matmul_chain_64 | 0.053 ± 0.000 | **0.043 ± 0.000** | 0.080 ± 0.000 | 0.192 ± 0.012 | 0.107 ± 0.035 |
| str_matrix_chain_multiplication_100 | **13.065 ± 0.054** | 13.145 ± 0.092 | 19.549 ± 0.024 | 23.229 ± 0.294 | 21.353 ± 0.790 |
| str_mps_varying_inner_product_200 | **17.811 ± 0.061** | 18.158 ± 0.227 | 27.434 ± 0.083 | 39.949 ± 0.194 | 27.641 ± 8.526 |
| str_nw_mera_closed_120 | **1793.395 ± 4.505** | 1842.398 ± 4.307 | 2314.945 ± 3.806 | 2541.785 ± 6.283 | 2579.359 ± 17.562 |
| str_nw_mera_open_26 | **1313.768 ± 2.120** | 1577.358 ± 1.758 | 1466.816 ± 2.351 | 1975.463 ± 2.525 | 2107.425 ± 41.968 |
| tensornetwork_permutation_focus_step409_316 | **313.199 ± 1.922** | 444.561 ± 1.585 | 1024.017 ± 1.522 | 756.701 ± 1.974 | 678.009 ± 31.741 |
| tensornetwork_permutation_light_415 | **312.373 ± 0.854** | 343.792 ± 1.652 | 980.970 ± 0.745 | 770.725 ± 3.352 | 684.016 ± 16.280 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.229 ± 0.013 | **4.351 ± 0.009** | 5.982 ± 0.030 | 5.338 ± 0.035 | 13.307 ± 6.928 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.770 ± 0.004 | **0.553 ± 0.008** | 0.877 ± 0.002 | 0.895 ± 0.021 | 0.967 ± 1.216 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.990 ± 0.014 | 0.979 ± 0.008 | **0.185 ± 0.002** | 0.516 ± 0.037 | 2.572 ± 1.121 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.654 ± 0.010 | 0.981 ± 0.006 | **0.182 ± 0.004** | 0.522 ± 0.025 | 6.964 ± 7.608 |
| bin_elementwise_mul_2048x2048 | 1.926 ± 0.036 | **1.924 ± 0.029** | 11.422 ± 0.416 | 15.592 ± 0.144 | 12.903 ± 0.939 |
| bin_elementwise_mul_512x512 | 0.135 ± 0.030 | 0.124 ± 0.006 | 0.151 ± 0.001 | 0.339 ± 0.022 | **0.120 ± 0.561** |
| bin_matmul_1024 | 75.076 ± 0.283 | **68.761 ± 0.111** | 89.362 ± 0.155 | 77.286 ± 0.182 | 110.540 ± 24.786 |
| bin_matmul_256 | 1.216 ± 0.035 | **1.073 ± 0.003** | 1.838 ± 0.051 | 1.389 ± 0.015 | 1.771 ± 0.014 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.004 | **0.007 ± 0.000** | 0.029 ± 0.001 | 0.172 ± 0.013 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.036 ± 0.001 | **0.033 ± 0.001** | 0.145 ± 0.004 | 0.228 ± 0.025 | 0.051 ± 0.001 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.001 | **0.011 ± 0.000** | 0.029 ± 0.001 | 0.117 ± 0.016 | 0.027 ± 0.001 |
| bin_outer_product_4096 | 49.518 ± 0.813 | 49.535 ± 1.057 | **39.480 ± 0.578** | 40.985 ± 7.934 | 315.983 ± 2.170 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 73.671 ± 0.236 | **71.482 ± 0.322** | 85.061 ± 0.102 | 81.833 ± 0.301 | 101.781 ± 14.438 |
| gm_queen5_5_3.wcsp | **1869.616 ± 10.094** | 2314.439 ± 4.806 | 2805.429 ± 12.682 | 2110.599 ± 4.758 | 3352.583 ± 45.833 |
| lm_batch_likelihood_brackets_4_4d | **30.304 ± 0.103** | 33.180 ± 0.392 | 40.804 ± 0.461 | 30.654 ± 0.281 | 46.216 ± 33.822 |
| lm_batch_likelihood_sentence_3_12d | **81.446 ± 0.381** | 117.039 ± 1.468 | 86.463 ± 0.388 | 131.192 ± 0.738 | 155.634 ± 39.550 |
| lm_batch_likelihood_sentence_4_4d | **29.334 ± 0.066** | 35.831 ± 0.282 | 43.263 ± 0.139 | 30.971 ± 0.147 | 47.703 ± 33.460 |
| nary_matmul_chain_64 | 0.053 ± 0.000 | **0.043 ± 0.000** | 0.080 ± 0.000 | 0.192 ± 0.012 | 0.107 ± 0.035 |
| str_matrix_chain_multiplication_100 | 13.199 ± 0.067 | **13.145 ± 0.051** | 19.456 ± 0.123 | 22.979 ± 0.271 | 21.176 ± 0.355 |
| str_mps_varying_inner_product_200 | 21.890 ± 0.110 | **19.079 ± 0.168** | 25.729 ± 0.108 | 38.447 ± 0.336 | 24.139 ± 12.407 |
| str_nw_mera_closed_120 | 1756.935 ± 3.204 | **1718.794 ± 3.389** | 1962.542 ± 1.554 | 2193.382 ± 1.885 | 2330.125 ± 76.026 |
| str_nw_mera_open_26 | **1329.213 ± 2.072** | 1589.296 ± 2.025 | 1482.108 ± 2.209 | 2014.623 ± 4.061 | 2110.799 ± 30.527 |
| tensornetwork_permutation_focus_step409_316 | **313.199 ± 1.922** | 444.561 ± 1.585 | 1024.017 ± 1.522 | 756.701 ± 1.974 | 678.009 ± 31.741 |
| tensornetwork_permutation_light_415 | **312.373 ± 0.854** | 343.792 ± 1.652 | 980.970 ± 0.745 | 770.725 ± 3.352 | 684.016 ± 16.280 |

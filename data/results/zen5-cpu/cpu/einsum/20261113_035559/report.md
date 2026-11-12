# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_035559/run.yaml`
- Timestamp: `2026-11-12T18:55:59.554485+00:00`

Latest run: `./scripts/run_all.sh 8`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261113_035559`.

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

- OMP_NUM_THREADS: `8`
- OMP_THREAD_LIMIT: `8`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `8`
- OPENBLAS_NUM_THREADS: `8`
- GOTO_NUM_THREADS: `8`
- MKL_NUM_THREADS: `8`
- VECLIB_MAXIMUM_THREADS: `8`
- VECLIB_NUM_THREADS: `8`
- NUMEXPR_NUM_THREADS: `8`
- BLIS_NUM_THREADS: `8`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=8`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `native`
- BLAS implementation: `none`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Julia / OMEinsum.jl Backend

- Julia: version `1.13.1`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `8`

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

## Threads: 8

Logs:

- `data/results/zen5-cpu/cpu/einsum/20261113_035559/jax_cpu_t8_20261113_035559.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_035559/julia_omeinsum_t8_20261113_035559.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_035559/pytorch_cpu_t8_20261113_035559.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_035559/tenferro_eager_t8_20261113_035559.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_035559/tenferro_trace_t8_20261113_035559.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=8, OMP_NUM_THREADS=8, RAYON_NUM_THREADS=8.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.503 ± 0.027 | **1.065 ± 0.010** | 5.997 ± 0.806 | 1.470 ± 0.034 | 11.321 ± 9.333 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.469 ± 0.009 | **0.318 ± 0.004** | 5.998 ± 0.012 | 1.053 ± 0.007 | 1.110 ± 1.150 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.131 ± 0.001 | **0.129 ± 0.025** | 5.999 ± 0.017 | 0.562 ± 0.245 | 1.120 ± 0.351 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.110 ± 0.001** | 0.157 ± 0.001 | 5.998 ± 0.003 | 0.573 ± 0.184 | 5.738 ± 5.004 |
| bin_elementwise_mul_2048x2048 | 1.139 ± 0.161 | **1.000 ± 0.241** | 5.112 ± 0.869 | 5.984 ± 0.416 | 13.803 ± 0.355 |
| bin_elementwise_mul_512x512 | 0.027 ± 0.001 | **0.020 ± 0.001** | 5.999 ± 0.002 | 0.405 ± 0.126 | 0.138 ± 0.591 |
| bin_matmul_1024 | 10.611 ± 0.042 | **8.932 ± 0.031** | 19.068 ± 0.231 | 13.780 ± 1.665 | 41.607 ± 9.865 |
| bin_matmul_256 | 0.236 ± 0.004 | **0.169 ± 0.001** | 0.459 ± 0.004 | 0.453 ± 0.116 | 0.671 ± 0.012 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.008 ± 0.000** | 0.030 ± 0.001 | 0.169 ± 0.009 | 0.015 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.001 | **0.028 ± 0.001** | 0.145 ± 0.001 | 0.232 ± 0.055 | 0.053 ± 0.004 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.000 | **0.013 ± 0.000** | 0.028 ± 0.001 | 0.099 ± 0.010 | 0.028 ± 0.003 |
| bin_outer_product_4096 | 8.945 ± 0.573 | 8.403 ± 0.303 | **7.533 ± 0.261** | 16.672 ± 0.208 | 296.975 ± 2.278 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 11.647 ± 0.074 | **10.479 ± 0.016** | 14.781 ± 0.179 | 13.560 ± 0.381 | 34.008 ± 10.037 |
| gm_queen5_5_3.wcsp | 2304.747 ± 8.647 | 1722.025 ± 6.888 | **1498.003 ± 10.183** | 2011.325 ± 34.284 | 7195.208 ± 118.102 |
| lm_batch_likelihood_brackets_4_4d | 11.145 ± 0.169 | 12.173 ± 0.215 | **8.754 ± 0.220** | 19.615 ± 0.364 | 48.479 ± 37.945 |
| lm_batch_likelihood_sentence_3_12d | **15.731 ± 0.180** | 29.639 ± 0.505 | 32.477 ± 0.525 | 32.773 ± 5.615 | 87.204 ± 30.087 |
| lm_batch_likelihood_sentence_4_4d | 11.776 ± 0.150 | 11.999 ± 0.288 | **9.201 ± 0.260** | 19.453 ± 0.180 | 45.997 ± 33.653 |
| nary_matmul_chain_64 | 0.041 ± 0.000 | **0.031 ± 0.000** | 0.051 ± 0.001 | 0.140 ± 0.021 | 0.110 ± 0.037 |
| str_matrix_chain_multiplication_100 | 5.755 ± 0.026 | 5.593 ± 0.206 | **5.517 ± 0.044** | 7.531 ± 0.248 | 11.107 ± 4.215 |
| str_mps_varying_inner_product_200 | 10.539 ± 0.083 | **9.791 ± 0.118** | 15.227 ± 0.128 | 22.173 ± 0.235 | 29.346 ± 19.074 |
| str_nw_mera_closed_120 | **332.194 ± 1.815** | 380.764 ± 1.195 | 451.545 ± 2.463 | 451.160 ± 19.682 | 789.496 ± 27.406 |
| str_nw_mera_open_26 | 339.213 ± 1.421 | 366.468 ± 1.554 | **281.542 ± 1.445** | 364.311 ± 9.437 | 900.171 ± 51.206 |
| tensornetwork_permutation_focus_step409_316 | **95.871 ± 0.844** | 190.690 ± 1.074 | 177.489 ± 0.578 | 352.515 ± 2.566 | 445.029 ± 44.251 |
| tensornetwork_permutation_light_415 | **99.303 ± 0.484** | 137.426 ± 1.116 | 175.056 ± 0.778 | 365.836 ± 5.811 | 424.774 ± 41.120 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=8, OMP_NUM_THREADS=8, RAYON_NUM_THREADS=8.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.503 ± 0.027 | **1.065 ± 0.010** | 5.997 ± 0.806 | 1.470 ± 0.034 | 11.321 ± 9.333 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.469 ± 0.009 | **0.318 ± 0.004** | 5.998 ± 0.012 | 1.053 ± 0.007 | 1.110 ± 1.150 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.131 ± 0.001 | **0.129 ± 0.025** | 5.999 ± 0.017 | 0.562 ± 0.245 | 1.120 ± 0.351 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.110 ± 0.001** | 0.157 ± 0.001 | 5.998 ± 0.003 | 0.573 ± 0.184 | 5.738 ± 5.004 |
| bin_elementwise_mul_2048x2048 | 1.139 ± 0.161 | **1.000 ± 0.241** | 5.112 ± 0.869 | 5.984 ± 0.416 | 13.803 ± 0.355 |
| bin_elementwise_mul_512x512 | 0.027 ± 0.001 | **0.020 ± 0.001** | 5.999 ± 0.002 | 0.405 ± 0.126 | 0.138 ± 0.591 |
| bin_matmul_1024 | 10.611 ± 0.042 | **8.932 ± 0.031** | 19.068 ± 0.231 | 13.780 ± 1.665 | 41.607 ± 9.865 |
| bin_matmul_256 | 0.236 ± 0.004 | **0.169 ± 0.001** | 0.459 ± 0.004 | 0.453 ± 0.116 | 0.671 ± 0.012 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.008 ± 0.000** | 0.030 ± 0.001 | 0.169 ± 0.009 | 0.015 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.001 | **0.028 ± 0.001** | 0.145 ± 0.001 | 0.232 ± 0.055 | 0.053 ± 0.004 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.000 | **0.013 ± 0.000** | 0.028 ± 0.001 | 0.099 ± 0.010 | 0.028 ± 0.003 |
| bin_outer_product_4096 | 8.945 ± 0.573 | 8.403 ± 0.303 | **7.533 ± 0.261** | 16.672 ± 0.208 | 296.975 ± 2.278 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 11.647 ± 0.074 | **10.479 ± 0.016** | 14.781 ± 0.179 | 13.560 ± 0.381 | 34.008 ± 10.037 |
| gm_queen5_5_3.wcsp | 566.941 ± 2.808 | **505.853 ± 2.288** | 662.734 ± 6.290 | 766.268 ± 24.962 | 2535.796 ± 18.163 |
| lm_batch_likelihood_brackets_4_4d | 11.489 ± 0.086 | 11.764 ± 0.095 | **8.689 ± 0.346** | 19.242 ± 0.434 | 59.152 ± 27.429 |
| lm_batch_likelihood_sentence_3_12d | **16.151 ± 0.232** | 31.376 ± 3.102 | 18.539 ± 0.443 | 37.775 ± 6.246 | 92.395 ± 21.588 |
| lm_batch_likelihood_sentence_4_4d | 10.565 ± 0.118 | 12.348 ± 0.293 | **8.996 ± 0.174** | 19.142 ± 0.334 | 40.433 ± 36.840 |
| nary_matmul_chain_64 | 0.041 ± 0.000 | **0.031 ± 0.000** | 0.051 ± 0.001 | 0.140 ± 0.021 | 0.110 ± 0.037 |
| str_matrix_chain_multiplication_100 | 5.943 ± 0.097 | 5.654 ± 0.104 | **5.569 ± 0.119** | 7.588 ± 0.228 | 10.975 ± 0.704 |
| str_mps_varying_inner_product_200 | 13.486 ± 0.134 | **10.753 ± 0.123** | 15.174 ± 0.078 | 21.444 ± 0.266 | 25.335 ± 18.593 |
| str_nw_mera_closed_120 | 419.920 ± 1.951 | 374.475 ± 3.876 | 345.865 ± 1.424 | **345.495 ± 16.665** | 644.041 ± 53.031 |
| str_nw_mera_open_26 | 344.670 ± 1.329 | 365.728 ± 1.108 | **282.947 ± 1.101** | 388.286 ± 6.008 | 912.801 ± 48.881 |
| tensornetwork_permutation_focus_step409_316 | **95.871 ± 0.844** | 190.690 ± 1.074 | 177.489 ± 0.578 | 352.515 ± 2.566 | 445.029 ± 44.251 |
| tensornetwork_permutation_light_415 | **99.303 ± 0.484** | 137.426 ± 1.116 | 175.056 ± 0.778 | 365.836 ± 5.811 | 424.774 ± 41.120 |

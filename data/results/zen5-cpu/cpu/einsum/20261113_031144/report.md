# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_031144/run.yaml`
- Timestamp: `2026-11-12T18:11:44.458524+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261113_031144`.

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=4`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `native`
- BLAS implementation: `none`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Julia / OMEinsum.jl Backend

- Julia: version `1.13.1`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `4`

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

## Threads: 4

Logs:

- `data/results/zen5-cpu/cpu/einsum/20261113_031144/jax_cpu_t4_20261113_031144.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_031144/julia_omeinsum_t4_20261113_031144.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_031144/pytorch_cpu_t4_20261113_031144.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_031144/tenferro_eager_t4_20261113_031144.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_031144/tenferro_trace_t4_20261113_031144.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.783 ± 0.005 | **1.469 ± 0.008** | 1.705 ± 4.297 | 2.378 ± 0.496 | 10.056 ± 6.054 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.478 ± 0.009 | 0.363 ± 0.007 | **0.273 ± 0.006** | 1.038 ± 0.032 | 1.097 ± 1.141 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.245 ± 0.004 | 0.243 ± 0.003 | **0.069 ± 0.001** | 0.396 ± 0.186 | 0.924 ± 1.997 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.203 ± 0.004 | 0.240 ± 0.004 | **0.070 ± 0.002** | 0.485 ± 0.151 | 6.738 ± 2.837 |
| bin_elementwise_mul_2048x2048 | 0.806 ± 0.075 | **0.785 ± 0.011** | 3.902 ± 0.283 | 8.020 ± 0.079 | 13.216 ± 0.324 |
| bin_elementwise_mul_512x512 | 0.042 ± 0.001 | **0.038 ± 0.001** | 0.052 ± 0.002 | 0.412 ± 0.177 | 0.119 ± 0.462 |
| bin_matmul_1024 | 19.375 ± 0.066 | **17.324 ± 0.030** | 35.340 ± 0.443 | 20.230 ± 0.159 | 51.507 ± 17.816 |
| bin_matmul_256 | 0.372 ± 0.007 | **0.296 ± 0.003** | 0.921 ± 0.004 | 0.645 ± 0.172 | 0.833 ± 0.017 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.007 ± 0.000** | 0.030 ± 0.002 | 0.167 ± 0.015 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.031 ± 0.001 | **0.028 ± 0.000** | 0.143 ± 0.002 | 0.225 ± 0.049 | 0.049 ± 0.001 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.001 | **0.012 ± 0.001** | 0.029 ± 0.002 | 0.099 ± 0.011 | 0.027 ± 0.001 |
| bin_outer_product_4096 | 13.648 ± 0.141 | 13.885 ± 0.358 | **11.497 ± 0.231** | 21.768 ± 0.764 | 296.433 ± 2.438 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 19.910 ± 0.056 | **18.501 ± 0.035** | 26.404 ± 0.283 | 23.847 ± 0.590 | 44.905 ± 16.396 |
| gm_queen5_5_3.wcsp | 2747.871 ± 6.929 | 2710.089 ± 10.592 | **2194.640 ± 14.282** | 2984.680 ± 55.678 | 7147.330 ± 96.589 |
| lm_batch_likelihood_brackets_4_4d | 14.050 ± 0.136 | 14.422 ± 0.111 | **11.000 ± 0.074** | 21.071 ± 0.511 | 27.837 ± 8.390 |
| lm_batch_likelihood_sentence_3_12d | **22.677 ± 0.024** | 38.656 ± 0.552 | 42.116 ± 0.862 | 36.661 ± 6.745 | 101.528 ± 42.595 |
| lm_batch_likelihood_sentence_4_4d | 14.193 ± 0.051 | 14.375 ± 0.200 | **11.824 ± 0.130** | 21.564 ± 0.727 | 43.170 ± 31.092 |
| nary_matmul_chain_64 | 0.042 ± 0.001 | **0.032 ± 0.001** | 0.062 ± 0.001 | 0.159 ± 0.029 | 0.109 ± 0.019 |
| str_matrix_chain_multiplication_100 | **6.114 ± 0.067** | 6.332 ± 0.190 | 7.525 ± 0.027 | 10.905 ± 0.433 | 12.647 ± 0.640 |
| str_mps_varying_inner_product_200 | 10.931 ± 0.088 | **10.137 ± 0.290** | 18.135 ± 0.107 | 22.309 ± 0.293 | 29.156 ± 19.849 |
| str_nw_mera_closed_120 | **541.084 ± 1.434** | 552.667 ± 2.014 | 694.159 ± 3.565 | 737.012 ± 5.057 | 1016.287 ± 14.951 |
| str_nw_mera_open_26 | 465.598 ± 1.393 | 508.464 ± 2.490 | **435.378 ± 1.260** | 599.459 ± 5.829 | 1024.447 ± 49.401 |
| tensornetwork_permutation_focus_step409_316 | **147.729 ± 0.958** | 160.473 ± 3.353 | 297.606 ± 0.711 | 369.550 ± 5.136 | 480.513 ± 17.920 |
| tensornetwork_permutation_light_415 | 153.372 ± 1.396 | **114.628 ± 0.647** | 289.275 ± 4.637 | 381.054 ± 6.776 | 481.028 ± 71.652 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.783 ± 0.005 | **1.469 ± 0.008** | 1.705 ± 4.297 | 2.378 ± 0.496 | 10.056 ± 6.054 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.478 ± 0.009 | 0.363 ± 0.007 | **0.273 ± 0.006** | 1.038 ± 0.032 | 1.097 ± 1.141 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.245 ± 0.004 | 0.243 ± 0.003 | **0.069 ± 0.001** | 0.396 ± 0.186 | 0.924 ± 1.997 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.203 ± 0.004 | 0.240 ± 0.004 | **0.070 ± 0.002** | 0.485 ± 0.151 | 6.738 ± 2.837 |
| bin_elementwise_mul_2048x2048 | 0.806 ± 0.075 | **0.785 ± 0.011** | 3.902 ± 0.283 | 8.020 ± 0.079 | 13.216 ± 0.324 |
| bin_elementwise_mul_512x512 | 0.042 ± 0.001 | **0.038 ± 0.001** | 0.052 ± 0.002 | 0.412 ± 0.177 | 0.119 ± 0.462 |
| bin_matmul_1024 | 19.375 ± 0.066 | **17.324 ± 0.030** | 35.340 ± 0.443 | 20.230 ± 0.159 | 51.507 ± 17.816 |
| bin_matmul_256 | 0.372 ± 0.007 | **0.296 ± 0.003** | 0.921 ± 0.004 | 0.645 ± 0.172 | 0.833 ± 0.017 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.007 ± 0.000** | 0.030 ± 0.002 | 0.167 ± 0.015 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.031 ± 0.001 | **0.028 ± 0.000** | 0.143 ± 0.002 | 0.225 ± 0.049 | 0.049 ± 0.001 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.001 | **0.012 ± 0.001** | 0.029 ± 0.002 | 0.099 ± 0.011 | 0.027 ± 0.001 |
| bin_outer_product_4096 | 13.648 ± 0.141 | 13.885 ± 0.358 | **11.497 ± 0.231** | 21.768 ± 0.764 | 296.433 ± 2.438 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 19.910 ± 0.056 | **18.501 ± 0.035** | 26.404 ± 0.283 | 23.847 ± 0.590 | 44.905 ± 16.396 |
| gm_queen5_5_3.wcsp | **712.250 ± 4.538** | 765.541 ± 0.960 | 965.135 ± 10.877 | 1161.872 ± 12.288 | 2620.191 ± 29.083 |
| lm_batch_likelihood_brackets_4_4d | 14.034 ± 0.073 | 13.546 ± 0.126 | **11.586 ± 0.183** | 21.145 ± 0.352 | 45.943 ± 34.425 |
| lm_batch_likelihood_sentence_3_12d | **25.244 ± 0.098** | 36.825 ± 0.401 | 43.463 ± 0.540 | 41.320 ± 0.764 | 98.894 ± 35.310 |
| lm_batch_likelihood_sentence_4_4d | **12.739 ± 0.038** | 14.490 ± 0.089 | 12.786 ± 0.122 | 21.023 ± 0.510 | 42.587 ± 28.604 |
| nary_matmul_chain_64 | 0.042 ± 0.001 | **0.032 ± 0.001** | 0.062 ± 0.001 | 0.159 ± 0.029 | 0.109 ± 0.019 |
| str_matrix_chain_multiplication_100 | 6.492 ± 0.050 | **6.309 ± 0.049** | 7.490 ± 0.069 | 10.890 ± 0.340 | 12.624 ± 1.144 |
| str_mps_varying_inner_product_200 | 13.415 ± 0.059 | **10.934 ± 0.115** | 17.849 ± 0.048 | 21.680 ± 0.265 | 24.750 ± 5.982 |
| str_nw_mera_closed_120 | 565.553 ± 1.769 | **527.749 ± 4.927** | 565.849 ± 1.549 | 593.195 ± 10.107 | 877.502 ± 35.219 |
| str_nw_mera_open_26 | 469.735 ± 0.770 | 508.566 ± 1.624 | **442.513 ± 1.748** | 619.900 ± 10.620 | 1044.841 ± 49.863 |
| tensornetwork_permutation_focus_step409_316 | **147.729 ± 0.958** | 160.473 ± 3.353 | 297.606 ± 0.711 | 369.550 ± 5.136 | 480.513 ± 17.920 |
| tensornetwork_permutation_light_415 | 153.372 ± 1.396 | **114.628 ± 0.647** | 289.275 ± 4.637 | 381.054 ± 6.776 | 481.028 ± 71.652 |

# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261031_104351/run.yaml`
- Timestamp: `2026-10-31T01:43:51.562747+00:00`

Latest run: `./scripts/run_all.sh 8`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261031_104351`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

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

- tenferro-rs features: `blas-openblas`
- BLAS implementation: `openblas`
- BLAS version: `unknown`
- BLAS root: `/usr`
- BLAS library: `/usr`

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

- `data/results/zen5-cpu/cpu/einsum/20261031_104351/jax_cpu_t8_20261031_104351.log`
- `data/results/zen5-cpu/cpu/einsum/20261031_104351/julia_omeinsum_t8_20261031_104351.log`
- `data/results/zen5-cpu/cpu/einsum/20261031_104351/pytorch_cpu_t8_20261031_104351.log`
- `data/results/zen5-cpu/cpu/einsum/20261031_104351/tenferro_eager_t8_20261031_104351.log`
- `data/results/zen5-cpu/cpu/einsum/20261031_104351/tenferro_trace_t8_20261031_104351.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=8, OMP_NUM_THREADS=8, RAYON_NUM_THREADS=8.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 15.002 ± 3.988 | 1.922 ± 0.012 | **0.999 ± 3.002** | 1.493 ± 0.079 | 9.723 ± 10.556 |
| bin_batched_matmul_b32_m64_n64_k64 | 5.999 ± 2.999 | 0.548 ± 0.006 | **0.155 ± 0.003** | 1.046 ± 0.010 | 2.201 ± 1.240 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.135 ± 5.869 | 2.999 ± 0.004 | **0.047 ± 0.001** | 0.561 ± 0.091 | 1.237 ± 1.038 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 3.000 ± 0.002 | 2.999 ± 0.094 | **0.044 ± 0.001** | 0.430 ± 0.239 | 5.298 ± 4.992 |
| bin_elementwise_mul_2048x2048 | 2.992 ± 1.850 | **1.200 ± 0.224** | 3.001 ± 0.328 | 6.356 ± 0.534 | 14.421 ± 0.154 |
| bin_elementwise_mul_512x512 | 0.026 ± 0.002 | **0.023 ± 2.973** | 0.034 ± 0.001 | 0.329 ± 0.225 | 0.137 ± 0.599 |
| bin_matmul_1024 | 25.033 ± 4.989 | **10.662 ± 0.049** | 18.879 ± 0.184 | 12.688 ± 0.965 | 41.707 ± 22.511 |
| bin_matmul_256 | 11.999 ± 3.002 | **0.187 ± 0.012** | 0.462 ± 0.006 | 0.455 ± 0.057 | 1.226 ± 0.530 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.013 ± 0.001 | **0.007 ± 0.000** | 0.030 ± 0.001 | 0.168 ± 0.012 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.038 ± 0.001 | **0.035 ± 0.001** | 0.144 ± 0.002 | 0.237 ± 0.050 | 0.081 ± 0.022 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.000 | **0.011 ± 0.000** | 0.028 ± 0.001 | 0.100 ± 0.013 | 0.029 ± 0.002 |
| bin_outer_product_4096 | 8.609 ± 0.248 | 8.785 ± 0.194 | **7.826 ± 0.490** | 16.524 ± 0.297 | 300.835 ± 2.868 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 31.998 ± 9.955 | **13.234 ± 0.491** | 14.863 ± 0.146 | 13.591 ± 0.450 | 34.807 ± 19.423 |
| gm_queen5_5_3.wcsp | 4708.471 ± 26.127 | 3041.928 ± 61.785 | **1524.995 ± 6.747** | 2051.948 ± 15.641 | 7325.962 ± 107.670 |
| lm_batch_likelihood_brackets_4_4d | 165.007 ± 81.006 | 198.400 ± 13.032 | **8.712 ± 0.067** | 19.563 ± 0.371 | 31.659 ± 11.169 |
| lm_batch_likelihood_sentence_3_12d | 92.997 ± 15.978 | 205.275 ± 14.737 | 33.530 ± 0.305 | **32.372 ± 7.422** | 93.010 ± 29.289 |
| lm_batch_likelihood_sentence_4_4d | 212.057 ± 96.011 | 246.832 ± 101.013 | **9.153 ± 0.076** | 19.564 ± 0.461 | 58.654 ± 38.060 |
| nary_matmul_chain_64 | 0.056 ± 0.001 | **0.043 ± 0.000** | 0.051 ± 0.001 | 0.139 ± 0.007 | 0.105 ± 0.039 |
| str_matrix_chain_multiplication_100 | 6.093 ± 0.019 | 5.840 ± 0.385 | **5.562 ± 0.034** | 7.666 ± 0.111 | 11.007 ± 0.670 |
| str_mps_varying_inner_product_200 | 17.504 ± 0.110 | 18.025 ± 0.147 | **15.467 ± 0.113** | 22.126 ± 0.168 | 29.594 ± 3.602 |
| str_nw_mera_closed_120 | 655.967 ± 2.505 | 833.046 ± 7.971 | **469.939 ± 3.980** | 470.571 ± 29.326 | 806.312 ± 78.966 |
| str_nw_mera_open_26 | 618.988 ± 10.604 | 542.280 ± 45.116 | **287.764 ± 1.516** | 374.591 ± 5.632 | 929.563 ± 28.929 |
| tensornetwork_permutation_focus_step409_316 | 383.328 ± 2.827 | 474.720 ± 31.515 | **180.032 ± 0.837** | 358.166 ± 2.537 | 467.950 ± 59.989 |
| tensornetwork_permutation_light_415 | 383.888 ± 4.663 | 431.838 ± 16.663 | **177.829 ± 0.925** | 372.396 ± 2.425 | 437.121 ± 32.936 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=8, OMP_NUM_THREADS=8, RAYON_NUM_THREADS=8.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 15.002 ± 3.988 | 1.922 ± 0.012 | **0.999 ± 3.002** | 1.493 ± 0.079 | 9.723 ± 10.556 |
| bin_batched_matmul_b32_m64_n64_k64 | 5.999 ± 2.999 | 0.548 ± 0.006 | **0.155 ± 0.003** | 1.046 ± 0.010 | 2.201 ± 1.240 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.135 ± 5.869 | 2.999 ± 0.004 | **0.047 ± 0.001** | 0.561 ± 0.091 | 1.237 ± 1.038 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 3.000 ± 0.002 | 2.999 ± 0.094 | **0.044 ± 0.001** | 0.430 ± 0.239 | 5.298 ± 4.992 |
| bin_elementwise_mul_2048x2048 | 2.992 ± 1.850 | **1.200 ± 0.224** | 3.001 ± 0.328 | 6.356 ± 0.534 | 14.421 ± 0.154 |
| bin_elementwise_mul_512x512 | 0.026 ± 0.002 | **0.023 ± 2.973** | 0.034 ± 0.001 | 0.329 ± 0.225 | 0.137 ± 0.599 |
| bin_matmul_1024 | 25.033 ± 4.989 | **10.662 ± 0.049** | 18.879 ± 0.184 | 12.688 ± 0.965 | 41.707 ± 22.511 |
| bin_matmul_256 | 11.999 ± 3.002 | **0.187 ± 0.012** | 0.462 ± 0.006 | 0.455 ± 0.057 | 1.226 ± 0.530 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.013 ± 0.001 | **0.007 ± 0.000** | 0.030 ± 0.001 | 0.168 ± 0.012 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.038 ± 0.001 | **0.035 ± 0.001** | 0.144 ± 0.002 | 0.237 ± 0.050 | 0.081 ± 0.022 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.000 | **0.011 ± 0.000** | 0.028 ± 0.001 | 0.100 ± 0.013 | 0.029 ± 0.002 |
| bin_outer_product_4096 | 8.609 ± 0.248 | 8.785 ± 0.194 | **7.826 ± 0.490** | 16.524 ± 0.297 | 300.835 ± 2.868 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 31.998 ± 9.955 | **13.234 ± 0.491** | 14.863 ± 0.146 | 13.591 ± 0.450 | 34.807 ± 19.423 |
| gm_queen5_5_3.wcsp | 1563.900 ± 5.916 | 1186.599 ± 8.067 | **650.471 ± 2.162** | 777.289 ± 15.561 | 2586.607 ± 45.876 |
| lm_batch_likelihood_brackets_4_4d | 191.976 ± 45.981 | 213.872 ± 45.999 | **8.512 ± 0.053** | 19.341 ± 0.518 | 44.110 ± 32.656 |
| lm_batch_likelihood_sentence_3_12d | 100.044 ± 35.066 | 214.055 ± 33.846 | **18.607 ± 0.114** | 35.775 ± 2.468 | 95.446 ± 34.616 |
| lm_batch_likelihood_sentence_4_4d | 251.858 ± 117.030 | 233.845 ± 33.017 | **9.033 ± 0.207** | 19.111 ± 0.231 | 41.264 ± 27.102 |
| nary_matmul_chain_64 | 0.056 ± 0.001 | **0.043 ± 0.000** | 0.051 ± 0.001 | 0.139 ± 0.007 | 0.105 ± 0.039 |
| str_matrix_chain_multiplication_100 | 6.260 ± 0.195 | 5.939 ± 0.374 | **5.526 ± 0.077** | 7.613 ± 0.193 | 11.196 ± 1.277 |
| str_mps_varying_inner_product_200 | 19.303 ± 0.048 | 18.751 ± 0.134 | **15.073 ± 0.042** | 21.513 ± 0.146 | 25.647 ± 2.301 |
| str_nw_mera_closed_120 | 667.668 ± 5.418 | 634.365 ± 5.421 | 349.632 ± 2.793 | **345.280 ± 13.032** | 650.673 ± 40.836 |
| str_nw_mera_open_26 | 622.630 ± 5.249 | 534.470 ± 46.560 | **289.301 ± 1.781** | 390.816 ± 3.620 | 925.918 ± 32.765 |
| tensornetwork_permutation_focus_step409_316 | 383.328 ± 2.827 | 474.720 ± 31.515 | **180.032 ± 0.837** | 358.166 ± 2.537 | 467.950 ± 59.989 |
| tensornetwork_permutation_light_415 | 383.888 ± 4.663 | 431.838 ± 16.663 | **177.829 ± 0.925** | 372.396 ± 2.425 | 437.121 ± 32.936 |

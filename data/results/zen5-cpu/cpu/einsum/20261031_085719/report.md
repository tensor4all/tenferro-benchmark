# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261031_085719/run.yaml`
- Timestamp: `2026-10-30T23:57:19.436239+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261031_085719`.

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

- `data/results/zen5-cpu/cpu/einsum/20261031_085719/jax_cpu_t1_20261031_085719.log`
- `data/results/zen5-cpu/cpu/einsum/20261031_085719/julia_omeinsum_t1_20261031_085719.log`
- `data/results/zen5-cpu/cpu/einsum/20261031_085719/pytorch_cpu_t1_20261031_085719.log`
- `data/results/zen5-cpu/cpu/einsum/20261031_085719/tenferro_eager_t1_20261031_085719.log`
- `data/results/zen5-cpu/cpu/einsum/20261031_085719/tenferro_trace_t1_20261031_085719.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.380 ± 0.028 | 5.610 ± 0.016 | 5.979 ± 0.023 | **5.338 ± 0.041** | 13.187 ± 7.680 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.676 ± 0.004 | **0.545 ± 0.009** | 0.875 ± 0.004 | 0.887 ± 0.027 | 0.999 ± 1.338 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.962 ± 0.006 | 0.969 ± 0.010 | **0.186 ± 0.018** | 0.513 ± 0.027 | 2.417 ± 2.342 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.453 ± 0.012 | 0.961 ± 0.004 | **0.200 ± 0.015** | 0.515 ± 0.020 | 5.363 ± 1.834 |
| bin_elementwise_mul_2048x2048 | **1.920 ± 0.026** | 1.965 ± 0.014 | 11.431 ± 0.390 | 15.541 ± 0.148 | 13.019 ± 3.245 |
| bin_elementwise_mul_512x512 | **0.116 ± 0.002** | 0.135 ± 0.001 | 0.129 ± 0.002 | 0.336 ± 0.013 | 0.123 ± 0.531 |
| bin_matmul_1024 | 83.086 ± 0.184 | 80.013 ± 0.425 | 89.310 ± 0.241 | **78.339 ± 0.106** | 110.203 ± 23.506 |
| bin_matmul_256 | 1.392 ± 0.008 | **1.314 ± 0.008** | 1.795 ± 0.061 | 1.435 ± 0.022 | 1.979 ± 0.651 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.007 ± 0.000** | 0.029 ± 0.001 | 0.171 ± 0.016 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.036 ± 0.000 | **0.033 ± 0.000** | 0.141 ± 0.001 | 0.229 ± 0.024 | 0.085 ± 0.011 |
| bin_omeinsum_matmul_10x10 | 0.021 ± 0.001 | **0.011 ± 0.000** | 0.029 ± 0.001 | 0.118 ± 0.023 | 0.028 ± 0.001 |
| bin_outer_product_4096 | 49.806 ± 0.353 | 48.991 ± 0.435 | **39.968 ± 0.374** | 47.982 ± 7.830 | 317.587 ± 2.787 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 91.064 ± 0.196 | 82.575 ± 0.268 | 84.798 ± 0.386 | **80.631 ± 0.213** | 97.046 ± 11.431 |
| gm_queen5_5_3.wcsp | 6891.365 ± 26.328 | 6464.305 ± 14.090 | 6652.288 ± 27.605 | **4987.326 ± 18.295** | 7986.464 ± 45.828 |
| lm_batch_likelihood_brackets_4_4d | **29.403 ± 0.139** | 29.519 ± 1.241 | 37.323 ± 0.355 | 31.039 ± 0.216 | 36.101 ± 13.819 |
| lm_batch_likelihood_sentence_3_12d | **82.507 ± 0.375** | 146.372 ± 0.403 | 117.146 ± 0.472 | 107.438 ± 0.172 | 139.088 ± 23.402 |
| lm_batch_likelihood_sentence_4_4d | **31.407 ± 0.094** | 35.548 ± 0.824 | 41.384 ± 0.194 | 33.179 ± 0.848 | 74.402 ± 31.851 |
| nary_matmul_chain_64 | 0.055 ± 0.001 | **0.041 ± 0.000** | 0.080 ± 0.001 | 0.191 ± 0.010 | 0.113 ± 0.017 |
| str_matrix_chain_multiplication_100 | **15.529 ± 0.033** | 15.876 ± 0.047 | 18.639 ± 0.047 | 23.285 ± 0.332 | 21.412 ± 0.907 |
| str_mps_varying_inner_product_200 | **18.389 ± 0.045** | 18.565 ± 0.147 | 27.496 ± 0.098 | 40.350 ± 0.531 | 28.397 ± 40.113 |
| str_nw_mera_closed_120 | **2220.628 ± 2.424** | 2359.155 ± 26.789 | 2338.038 ± 1.960 | 2556.952 ± 11.655 | 2587.945 ± 34.378 |
| str_nw_mera_open_26 | 1684.299 ± 9.418 | 1819.493 ± 2.278 | **1474.810 ± 4.928** | 1985.507 ± 2.997 | 2131.980 ± 20.717 |
| tensornetwork_permutation_focus_step409_316 | **521.901 ± 1.413** | 697.080 ± 1.967 | 1027.888 ± 1.314 | 758.610 ± 1.767 | 680.933 ± 22.586 |
| tensornetwork_permutation_light_415 | **520.550 ± 2.152** | 559.408 ± 1.920 | 983.773 ± 0.999 | 774.204 ± 2.540 | 687.998 ± 14.146 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.380 ± 0.028 | 5.610 ± 0.016 | 5.979 ± 0.023 | **5.338 ± 0.041** | 13.187 ± 7.680 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.676 ± 0.004 | **0.545 ± 0.009** | 0.875 ± 0.004 | 0.887 ± 0.027 | 0.999 ± 1.338 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.962 ± 0.006 | 0.969 ± 0.010 | **0.186 ± 0.018** | 0.513 ± 0.027 | 2.417 ± 2.342 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.453 ± 0.012 | 0.961 ± 0.004 | **0.200 ± 0.015** | 0.515 ± 0.020 | 5.363 ± 1.834 |
| bin_elementwise_mul_2048x2048 | **1.920 ± 0.026** | 1.965 ± 0.014 | 11.431 ± 0.390 | 15.541 ± 0.148 | 13.019 ± 3.245 |
| bin_elementwise_mul_512x512 | **0.116 ± 0.002** | 0.135 ± 0.001 | 0.129 ± 0.002 | 0.336 ± 0.013 | 0.123 ± 0.531 |
| bin_matmul_1024 | 83.086 ± 0.184 | 80.013 ± 0.425 | 89.310 ± 0.241 | **78.339 ± 0.106** | 110.203 ± 23.506 |
| bin_matmul_256 | 1.392 ± 0.008 | **1.314 ± 0.008** | 1.795 ± 0.061 | 1.435 ± 0.022 | 1.979 ± 0.651 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.007 ± 0.000** | 0.029 ± 0.001 | 0.171 ± 0.016 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.036 ± 0.000 | **0.033 ± 0.000** | 0.141 ± 0.001 | 0.229 ± 0.024 | 0.085 ± 0.011 |
| bin_omeinsum_matmul_10x10 | 0.021 ± 0.001 | **0.011 ± 0.000** | 0.029 ± 0.001 | 0.118 ± 0.023 | 0.028 ± 0.001 |
| bin_outer_product_4096 | 49.806 ± 0.353 | 48.991 ± 0.435 | **39.968 ± 0.374** | 47.982 ± 7.830 | 317.587 ± 2.787 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 91.064 ± 0.196 | 82.575 ± 0.268 | 84.798 ± 0.386 | **80.631 ± 0.213** | 97.046 ± 11.431 |
| gm_queen5_5_3.wcsp | 2547.905 ± 10.351 | 2816.870 ± 13.314 | 2833.311 ± 7.913 | **2134.510 ± 2.507** | 3397.218 ± 41.547 |
| lm_batch_likelihood_brackets_4_4d | **28.991 ± 0.106** | 31.036 ± 0.921 | 40.911 ± 0.114 | 31.927 ± 0.218 | 46.843 ± 34.492 |
| lm_batch_likelihood_sentence_3_12d | 92.235 ± 0.605 | 170.900 ± 1.564 | **86.359 ± 0.201** | 131.721 ± 0.284 | 154.401 ± 36.584 |
| lm_batch_likelihood_sentence_4_4d | **30.339 ± 0.137** | 38.821 ± 1.483 | 43.621 ± 0.258 | 31.814 ± 0.188 | 47.739 ± 39.482 |
| nary_matmul_chain_64 | 0.055 ± 0.001 | **0.041 ± 0.000** | 0.080 ± 0.001 | 0.191 ± 0.010 | 0.113 ± 0.017 |
| str_matrix_chain_multiplication_100 | 15.964 ± 0.038 | **15.849 ± 0.080** | 18.642 ± 0.049 | 23.021 ± 0.238 | 21.461 ± 0.698 |
| str_mps_varying_inner_product_200 | 19.188 ± 0.075 | **18.009 ± 0.102** | 25.426 ± 0.059 | 38.891 ± 0.448 | 24.356 ± 12.446 |
| str_nw_mera_closed_120 | 2157.570 ± 2.029 | 2154.214 ± 11.903 | **1976.520 ± 3.141** | 2200.904 ± 2.157 | 2341.329 ± 54.645 |
| str_nw_mera_open_26 | 1684.418 ± 6.236 | 1816.297 ± 4.096 | **1497.591 ± 2.745** | 2014.187 ± 3.089 | 2130.536 ± 19.441 |
| tensornetwork_permutation_focus_step409_316 | **521.901 ± 1.413** | 697.080 ± 1.967 | 1027.888 ± 1.314 | 758.610 ± 1.767 | 680.933 ± 22.586 |
| tensornetwork_permutation_light_415 | **520.550 ± 2.152** | 559.408 ± 1.920 | 983.773 ± 0.999 | 774.204 ± 2.540 | 687.998 ± 14.146 |

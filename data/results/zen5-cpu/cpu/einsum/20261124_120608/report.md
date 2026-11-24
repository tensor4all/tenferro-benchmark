# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_120608/run.yaml`
- Timestamp: `2026-11-24T03:06:08.486161+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261124_120608`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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
- Python platform: `Linux-7.0.0-43-generic-x86_64-with-glibc2.39`
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

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
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

- `data/results/zen5-cpu/cpu/einsum/20261124_120608/jax_cpu_t1_20261124_120608.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_120608/julia_omeinsum_t1_20261124_120608.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_120608/pytorch_cpu_t1_20261124_120608.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_120608/tenferro_eager_t1_20261124_120608.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_120608/tenferro_trace_t1_20261124_120608.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.343 ± 0.078 | **4.299 ± 0.017** | 5.966 ± 0.071 | 5.359 ± 0.091 | 13.394 ± 7.280 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.790 ± 0.031 | **0.552 ± 0.007** | 0.876 ± 0.007 | 0.888 ± 0.016 | 0.961 ± 1.170 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.977 ± 0.004 | 0.967 ± 0.011 | **0.180 ± 0.004** | 0.508 ± 0.023 | 2.342 ± 1.938 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.650 ± 0.007 | 0.968 ± 0.029 | **0.182 ± 0.002** | 0.524 ± 0.037 | 5.196 ± 4.305 |
| bin_elementwise_mul_2048x2048 | **1.938 ± 0.031** | 1.968 ± 0.118 | 11.305 ± 0.129 | 15.726 ± 0.225 | 12.967 ± 3.661 |
| bin_elementwise_mul_512x512 | 0.142 ± 0.003 | 0.132 ± 0.002 | 0.151 ± 0.005 | 0.334 ± 0.026 | **0.118 ± 0.015** |
| bin_matmul_1024 | 75.231 ± 0.546 | **69.121 ± 0.413** | 89.583 ± 0.240 | 77.370 ± 0.290 | 112.986 ± 8.212 |
| bin_matmul_256 | 1.229 ± 0.045 | **1.074 ± 0.005** | 1.784 ± 0.053 | 1.387 ± 0.010 | 1.765 ± 0.010 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.008 ± 0.000** | 0.029 ± 0.002 | 0.172 ± 0.012 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.036 ± 0.008 | **0.034 ± 0.001** | 0.137 ± 0.002 | 0.231 ± 0.023 | 0.081 ± 0.031 |
| bin_omeinsum_matmul_10x10 | 0.022 ± 0.000 | **0.011 ± 0.001** | 0.028 ± 0.000 | 0.115 ± 0.022 | 0.028 ± 0.001 |
| bin_outer_product_4096 | 50.609 ± 0.767 | 50.314 ± 0.499 | **40.131 ± 0.380** | 40.406 ± 8.059 | 315.212 ± 3.935 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 74.125 ± 0.301 | **71.423 ± 0.367** | 84.615 ± 0.324 | 81.464 ± 0.633 | 106.331 ± 27.018 |
| gm_queen5_5_3.wcsp | 5571.311 ± 31.221 | 5494.751 ± 33.719 | 6512.463 ± 24.561 | **4925.471 ± 20.400** | 8235.024 ± 109.691 |
| lm_batch_likelihood_brackets_4_4d | **29.682 ± 0.490** | 35.914 ± 0.428 | 37.672 ± 0.455 | 30.884 ± 0.627 | 41.141 ± 12.223 |
| lm_batch_likelihood_sentence_3_12d | **71.798 ± 0.637** | 96.736 ± 22.391 | 86.909 ± 0.487 | 107.119 ± 0.496 | 144.019 ± 46.797 |
| lm_batch_likelihood_sentence_4_4d | **31.538 ± 0.461** | 37.741 ± 0.617 | 41.842 ± 0.285 | 32.617 ± 0.272 | 75.199 ± 32.199 |
| nary_matmul_chain_64 | 0.053 ± 0.000 | **0.043 ± 0.000** | 0.079 ± 0.001 | 0.195 ± 0.015 | 0.109 ± 0.012 |
| str_matrix_chain_multiplication_100 | **13.069 ± 0.131** | 13.171 ± 0.064 | 18.738 ± 0.213 | 23.293 ± 0.349 | 21.454 ± 1.185 |
| str_mps_varying_inner_product_200 | **17.920 ± 0.129** | 18.073 ± 0.408 | 27.195 ± 0.290 | 40.142 ± 0.351 | 29.197 ± 22.497 |
| str_nw_mera_closed_120 | **1799.255 ± 4.198** | 1843.077 ± 5.009 | 2320.298 ± 5.484 | 2535.904 ± 5.764 | 2610.816 ± 56.029 |
| str_nw_mera_open_26 | **1313.664 ± 4.851** | 1590.001 ± 6.241 | 1473.362 ± 5.348 | 1975.236 ± 4.737 | 2134.187 ± 41.691 |
| tensornetwork_permutation_focus_step409_316 | **312.989 ± 2.063** | 473.170 ± 3.501 | 1026.884 ± 2.692 | 763.557 ± 6.007 | 686.465 ± 52.932 |
| tensornetwork_permutation_light_415 | **313.456 ± 2.027** | 360.297 ± 5.957 | 985.912 ± 2.796 | 775.658 ± 3.473 | 663.938 ± 36.421 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.343 ± 0.078 | **4.299 ± 0.017** | 5.966 ± 0.071 | 5.359 ± 0.091 | 13.394 ± 7.280 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.790 ± 0.031 | **0.552 ± 0.007** | 0.876 ± 0.007 | 0.888 ± 0.016 | 0.961 ± 1.170 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.977 ± 0.004 | 0.967 ± 0.011 | **0.180 ± 0.004** | 0.508 ± 0.023 | 2.342 ± 1.938 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.650 ± 0.007 | 0.968 ± 0.029 | **0.182 ± 0.002** | 0.524 ± 0.037 | 5.196 ± 4.305 |
| bin_elementwise_mul_2048x2048 | **1.938 ± 0.031** | 1.968 ± 0.118 | 11.305 ± 0.129 | 15.726 ± 0.225 | 12.967 ± 3.661 |
| bin_elementwise_mul_512x512 | 0.142 ± 0.003 | 0.132 ± 0.002 | 0.151 ± 0.005 | 0.334 ± 0.026 | **0.118 ± 0.015** |
| bin_matmul_1024 | 75.231 ± 0.546 | **69.121 ± 0.413** | 89.583 ± 0.240 | 77.370 ± 0.290 | 112.986 ± 8.212 |
| bin_matmul_256 | 1.229 ± 0.045 | **1.074 ± 0.005** | 1.784 ± 0.053 | 1.387 ± 0.010 | 1.765 ± 0.010 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.008 ± 0.000** | 0.029 ± 0.002 | 0.172 ± 0.012 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.036 ± 0.008 | **0.034 ± 0.001** | 0.137 ± 0.002 | 0.231 ± 0.023 | 0.081 ± 0.031 |
| bin_omeinsum_matmul_10x10 | 0.022 ± 0.000 | **0.011 ± 0.001** | 0.028 ± 0.000 | 0.115 ± 0.022 | 0.028 ± 0.001 |
| bin_outer_product_4096 | 50.609 ± 0.767 | 50.314 ± 0.499 | **40.131 ± 0.380** | 40.406 ± 8.059 | 315.212 ± 3.935 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 74.125 ± 0.301 | **71.423 ± 0.367** | 84.615 ± 0.324 | 81.464 ± 0.633 | 106.331 ± 27.018 |
| gm_queen5_5_3.wcsp | **1899.831 ± 9.433** | 2331.011 ± 10.622 | 2844.881 ± 18.830 | 2123.924 ± 4.694 | 3460.737 ± 51.533 |
| lm_batch_likelihood_brackets_4_4d | **29.628 ± 0.203** | 33.452 ± 0.349 | 40.976 ± 1.107 | 30.900 ± 0.375 | 48.319 ± 35.046 |
| lm_batch_likelihood_sentence_3_12d | **80.801 ± 0.996** | 118.504 ± 1.283 | 86.648 ± 0.594 | 132.050 ± 1.649 | 154.121 ± 25.588 |
| lm_batch_likelihood_sentence_4_4d | **28.521 ± 0.360** | 36.913 ± 0.599 | 43.034 ± 0.661 | 31.008 ± 0.367 | 48.027 ± 34.212 |
| nary_matmul_chain_64 | 0.053 ± 0.000 | **0.043 ± 0.000** | 0.079 ± 0.001 | 0.195 ± 0.015 | 0.109 ± 0.012 |
| str_matrix_chain_multiplication_100 | 13.234 ± 0.130 | **13.122 ± 0.077** | 18.693 ± 0.150 | 23.105 ± 0.751 | 21.309 ± 0.661 |
| str_mps_varying_inner_product_200 | 21.043 ± 0.269 | **19.663 ± 0.452** | 25.274 ± 0.292 | 39.569 ± 0.890 | 24.361 ± 1.936 |
| str_nw_mera_closed_120 | 1756.266 ± 4.612 | **1724.712 ± 3.481** | 1966.043 ± 3.203 | 2195.425 ± 6.963 | 2346.740 ± 49.159 |
| str_nw_mera_open_26 | **1327.621 ± 5.638** | 1595.062 ± 4.076 | 1490.201 ± 4.644 | 2016.867 ± 4.023 | 2145.233 ± 28.917 |
| tensornetwork_permutation_focus_step409_316 | **312.989 ± 2.063** | 473.170 ± 3.501 | 1026.884 ± 2.692 | 763.557 ± 6.007 | 686.465 ± 52.932 |
| tensornetwork_permutation_light_415 | **313.456 ± 2.027** | 360.297 ± 5.957 | 985.912 ± 2.796 | 775.658 ± 3.473 | 663.938 ± 36.421 |

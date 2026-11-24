# CPU Einsum Benchmark Results

- Target profile: `zen5-cpu`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Benchmark commit: `bb9869533cc2ad586b71cdb49a3e1f85439cd2b2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/zen5-cpu/cpu/einsum/20261124_120608/report.md`.


- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_120608/run.yaml`
- Timestamp: `2026-11-24T03:06:08.486161+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261124_120608`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

### CPU Information

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

### Thread Environment

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

### Tenferro CPU BLAS Backend

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

### Julia / OMEinsum.jl Backend

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


Logs:

- `data/results/zen5-cpu/cpu/einsum/20261124_120608/jax_cpu_t1_20261124_120608.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_120608/julia_omeinsum_t1_20261124_120608.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_120608/pytorch_cpu_t1_20261124_120608.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_120608/tenferro_eager_t1_20261124_120608.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_120608/tenferro_trace_t1_20261124_120608.log`

##### Strategy: opt_flops

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

##### Strategy: opt_size

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

## Threads: 4

Source report: `data/results/zen5-cpu/cpu/einsum/20261124_131021/report.md`.


- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_131021/run.yaml`
- Timestamp: `2026-11-24T04:10:21.867323+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261124_131021`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

### CPU Information

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

### Thread Environment

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

### Tenferro CPU BLAS Backend

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

### Julia / OMEinsum.jl Backend

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


Logs:

- `data/results/zen5-cpu/cpu/einsum/20261124_131021/jax_cpu_t4_20261124_131021.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_131021/julia_omeinsum_t4_20261124_131021.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_131021/pytorch_cpu_t4_20261124_131021.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_131021/tenferro_eager_t4_20261124_131021.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_131021/tenferro_trace_t4_20261124_131021.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.772 ± 0.011 | **1.469 ± 0.004** | 1.664 ± 1.854 | 2.129 ± 0.259 | 11.198 ± 22.393 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.478 ± 0.005 | 0.365 ± 0.004 | **0.270 ± 0.005** | 0.926 ± 0.112 | 0.980 ± 1.257 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.243 ± 0.003 | 0.241 ± 0.004 | **0.070 ± 0.002** | 0.421 ± 0.195 | 2.517 ± 1.372 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.201 ± 0.006 | 0.243 ± 0.005 | **0.071 ± 0.002** | 0.521 ± 0.055 | 5.326 ± 3.707 |
| bin_elementwise_mul_2048x2048 | 0.807 ± 0.111 | **0.805 ± 0.152** | 4.049 ± 0.419 | 7.836 ± 0.464 | 12.971 ± 1.029 |
| bin_elementwise_mul_512x512 | 0.041 ± 0.002 | **0.038 ± 0.001** | 0.052 ± 0.002 | 0.394 ± 0.136 | 0.119 ± 0.530 |
| bin_matmul_1024 | 19.481 ± 0.140 | **17.328 ± 0.050** | 34.446 ± 0.185 | 20.393 ± 0.221 | 51.157 ± 11.884 |
| bin_matmul_256 | 0.369 ± 0.005 | **0.298 ± 0.004** | 0.897 ± 0.005 | 0.620 ± 0.126 | 0.837 ± 0.498 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.013 ± 0.000 | **0.008 ± 0.000** | 0.029 ± 0.001 | 0.168 ± 0.014 | 0.023 ± 0.008 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.001 | **0.029 ± 0.001** | 0.143 ± 0.002 | 0.234 ± 0.056 | 0.077 ± 0.016 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.001 | **0.012 ± 0.000** | 0.029 ± 0.001 | 0.100 ± 0.011 | 0.028 ± 0.003 |
| bin_outer_product_4096 | 14.264 ± 0.567 | 14.449 ± 0.426 | **12.228 ± 0.266** | 23.147 ± 0.347 | 299.938 ± 4.227 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 19.917 ± 0.094 | **18.438 ± 0.062** | 26.034 ± 0.906 | 24.005 ± 0.979 | 44.625 ± 12.821 |
| gm_queen5_5_3.wcsp | 2821.935 ± 25.228 | 2785.921 ± 69.682 | **2283.418 ± 13.351** | 3050.912 ± 38.183 | 7427.256 ± 129.092 |
| lm_batch_likelihood_brackets_4_4d | 12.752 ± 0.177 | 14.812 ± 0.331 | **11.122 ± 0.239** | 21.721 ± 1.005 | 29.208 ± 18.938 |
| lm_batch_likelihood_sentence_3_12d | **21.986 ± 0.306** | 39.014 ± 0.593 | 43.634 ± 0.724 | 36.252 ± 1.686 | 85.459 ± 23.056 |
| lm_batch_likelihood_sentence_4_4d | 13.133 ± 0.306 | 14.789 ± 0.361 | **11.785 ± 0.170** | 21.400 ± 0.484 | 42.687 ± 24.934 |
| nary_matmul_chain_64 | 0.042 ± 0.001 | **0.033 ± 0.001** | 0.062 ± 0.001 | 0.134 ± 0.021 | 0.099 ± 0.036 |
| str_matrix_chain_multiplication_100 | **6.305 ± 0.199** | 6.397 ± 0.171 | 7.567 ± 0.013 | 10.970 ± 0.873 | 12.514 ± 6.547 |
| str_mps_varying_inner_product_200 | 10.908 ± 0.714 | **10.764 ± 0.157** | 18.033 ± 0.116 | 22.906 ± 0.416 | 28.434 ± 25.086 |
| str_nw_mera_closed_120 | **543.669 ± 1.647** | 557.848 ± 3.823 | 723.022 ± 6.224 | 741.762 ± 12.666 | 1049.017 ± 34.205 |
| str_nw_mera_open_26 | 472.548 ± 5.518 | 516.236 ± 7.827 | **447.467 ± 11.046** | 613.193 ± 12.826 | 1070.026 ± 38.566 |
| tensornetwork_permutation_focus_step409_316 | **147.932 ± 2.348** | 175.751 ± 1.507 | 307.124 ± 14.548 | 371.532 ± 11.717 | 481.585 ± 36.299 |
| tensornetwork_permutation_light_415 | 153.585 ± 4.040 | **125.953 ± 1.631** | 295.661 ± 4.940 | 378.906 ± 11.692 | 490.835 ± 32.409 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.772 ± 0.011 | **1.469 ± 0.004** | 1.664 ± 1.854 | 2.129 ± 0.259 | 11.198 ± 22.393 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.478 ± 0.005 | 0.365 ± 0.004 | **0.270 ± 0.005** | 0.926 ± 0.112 | 0.980 ± 1.257 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.243 ± 0.003 | 0.241 ± 0.004 | **0.070 ± 0.002** | 0.421 ± 0.195 | 2.517 ± 1.372 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.201 ± 0.006 | 0.243 ± 0.005 | **0.071 ± 0.002** | 0.521 ± 0.055 | 5.326 ± 3.707 |
| bin_elementwise_mul_2048x2048 | 0.807 ± 0.111 | **0.805 ± 0.152** | 4.049 ± 0.419 | 7.836 ± 0.464 | 12.971 ± 1.029 |
| bin_elementwise_mul_512x512 | 0.041 ± 0.002 | **0.038 ± 0.001** | 0.052 ± 0.002 | 0.394 ± 0.136 | 0.119 ± 0.530 |
| bin_matmul_1024 | 19.481 ± 0.140 | **17.328 ± 0.050** | 34.446 ± 0.185 | 20.393 ± 0.221 | 51.157 ± 11.884 |
| bin_matmul_256 | 0.369 ± 0.005 | **0.298 ± 0.004** | 0.897 ± 0.005 | 0.620 ± 0.126 | 0.837 ± 0.498 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.013 ± 0.000 | **0.008 ± 0.000** | 0.029 ± 0.001 | 0.168 ± 0.014 | 0.023 ± 0.008 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.001 | **0.029 ± 0.001** | 0.143 ± 0.002 | 0.234 ± 0.056 | 0.077 ± 0.016 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.001 | **0.012 ± 0.000** | 0.029 ± 0.001 | 0.100 ± 0.011 | 0.028 ± 0.003 |
| bin_outer_product_4096 | 14.264 ± 0.567 | 14.449 ± 0.426 | **12.228 ± 0.266** | 23.147 ± 0.347 | 299.938 ± 4.227 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 19.917 ± 0.094 | **18.438 ± 0.062** | 26.034 ± 0.906 | 24.005 ± 0.979 | 44.625 ± 12.821 |
| gm_queen5_5_3.wcsp | **727.270 ± 6.620** | 781.110 ± 8.790 | 992.582 ± 4.394 | 1177.189 ± 18.986 | 2705.795 ± 33.229 |
| lm_batch_likelihood_brackets_4_4d | 12.939 ± 0.485 | 14.138 ± 0.323 | **12.022 ± 0.311** | 21.439 ± 0.527 | 52.688 ± 25.679 |
| lm_batch_likelihood_sentence_3_12d | **24.807 ± 0.373** | 39.826 ± 0.322 | 25.863 ± 0.520 | 43.057 ± 1.984 | 101.623 ± 31.290 |
| lm_batch_likelihood_sentence_4_4d | **11.796 ± 0.098** | 14.775 ± 0.152 | 12.874 ± 0.286 | 21.306 ± 0.260 | 45.188 ± 35.212 |
| nary_matmul_chain_64 | 0.042 ± 0.001 | **0.033 ± 0.001** | 0.062 ± 0.001 | 0.134 ± 0.021 | 0.099 ± 0.036 |
| str_matrix_chain_multiplication_100 | 6.528 ± 0.060 | **6.365 ± 0.126** | 7.498 ± 0.065 | 11.229 ± 0.441 | 12.611 ± 1.600 |
| str_mps_varying_inner_product_200 | 13.201 ± 0.110 | **11.179 ± 0.496** | 17.854 ± 0.256 | 22.005 ± 0.295 | 24.537 ± 6.290 |
| str_nw_mera_closed_120 | 568.491 ± 2.306 | **529.873 ± 5.530** | 592.986 ± 4.374 | 592.807 ± 7.552 | 877.491 ± 47.914 |
| str_nw_mera_open_26 | 475.198 ± 3.648 | 515.173 ± 2.620 | **453.967 ± 8.603** | 628.130 ± 5.710 | 1068.171 ± 20.878 |
| tensornetwork_permutation_focus_step409_316 | **147.932 ± 2.348** | 175.751 ± 1.507 | 307.124 ± 14.548 | 371.532 ± 11.717 | 481.585 ± 36.299 |
| tensornetwork_permutation_light_415 | 153.585 ± 4.040 | **125.953 ± 1.631** | 295.661 ± 4.940 | 378.906 ± 11.692 | 490.835 ± 32.409 |

## Threads: 8

Source report: `data/results/zen5-cpu/cpu/einsum/20261124_135202/report.md`.


- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_135202/run.yaml`
- Timestamp: `2026-11-24T04:52:02.812988+00:00`

Latest run: `./scripts/run_all.sh 8`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261124_135202`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

### CPU Information

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

### Thread Environment

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

### Tenferro CPU BLAS Backend

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

### Julia / OMEinsum.jl Backend

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


Logs:

- `data/results/zen5-cpu/cpu/einsum/20261124_135202/jax_cpu_t8_20261124_135202.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_135202/julia_omeinsum_t8_20261124_135202.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_135202/pytorch_cpu_t8_20261124_135202.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_135202/tenferro_eager_t8_20261124_135202.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_135202/tenferro_trace_t8_20261124_135202.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=8, OMP_NUM_THREADS=8, RAYON_NUM_THREADS=8.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.498 ± 0.016 | **1.065 ± 0.014** | 5.997 ± 0.009 | 1.468 ± 0.047 | 11.750 ± 5.936 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.473 ± 0.004 | **0.312 ± 0.008** | 5.999 ± 0.016 | 1.040 ± 0.027 | 1.065 ± 0.070 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.131 ± 0.028 | 0.129 ± 0.013 | **0.049 ± 0.001** | 0.550 ± 0.218 | 0.947 ± 0.377 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.109 ± 0.000 | 0.128 ± 0.001 | **0.049 ± 0.001** | 0.575 ± 0.199 | 3.460 ± 1.273 |
| bin_elementwise_mul_2048x2048 | **1.114 ± 0.269** | 1.215 ± 0.204 | 3.087 ± 0.292 | 6.158 ± 0.374 | 9.863 ± 6.715 |
| bin_elementwise_mul_512x512 | 0.026 ± 0.001 | **0.021 ± 0.001** | 0.037 ± 0.003 | 0.320 ± 0.192 | 0.143 ± 0.021 |
| bin_matmul_1024 | 10.619 ± 0.081 | **8.906 ± 0.017** | 18.822 ± 0.145 | 13.157 ± 1.436 | 43.805 ± 20.307 |
| bin_matmul_256 | 0.243 ± 0.006 | **0.169 ± 0.002** | 0.460 ± 0.005 | 0.429 ± 0.144 | 0.680 ± 0.006 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.001 | **0.008 ± 0.000** | 0.029 ± 0.001 | 0.167 ± 0.010 | 0.015 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.030 ± 0.001** | **0.030 ± 0.001** | 0.143 ± 0.002 | 0.228 ± 0.052 | 0.053 ± 0.002 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.001 | **0.012 ± 0.000** | 0.028 ± 0.001 | 0.100 ± 0.015 | 0.028 ± 0.002 |
| bin_outer_product_4096 | 9.065 ± 0.308 | 9.056 ± 0.333 | **8.491 ± 0.340** | 16.531 ± 0.271 | 298.982 ± 3.450 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 11.698 ± 0.068 | **10.361 ± 0.007** | 14.862 ± 0.189 | 13.548 ± 0.633 | 33.551 ± 16.843 |
| gm_queen5_5_3.wcsp | 2399.922 ± 129.566 | 1749.726 ± 22.292 | **1584.361 ± 8.948** | 2074.625 ± 20.812 | 7512.527 ± 83.772 |
| lm_batch_likelihood_brackets_4_4d | 10.262 ± 0.079 | 12.291 ± 0.136 | **8.805 ± 0.206** | 19.681 ± 0.737 | 31.868 ± 14.230 |
| lm_batch_likelihood_sentence_3_12d | **15.284 ± 0.401** | 20.813 ± 0.300 | 33.369 ± 0.455 | 31.697 ± 6.889 | 92.496 ± 19.234 |
| lm_batch_likelihood_sentence_4_4d | 11.113 ± 0.110 | 12.247 ± 0.114 | **9.252 ± 1.014** | 19.313 ± 0.171 | 42.297 ± 33.111 |
| nary_matmul_chain_64 | 0.040 ± 0.001 | **0.030 ± 0.000** | 0.051 ± 0.002 | 0.135 ± 0.008 | 0.110 ± 0.038 |
| str_matrix_chain_multiplication_100 | 5.777 ± 0.038 | 5.747 ± 0.022 | **5.564 ± 0.076** | 7.659 ± 0.313 | 11.510 ± 5.521 |
| str_mps_varying_inner_product_200 | 10.372 ± 0.144 | **10.302 ± 0.355** | 15.560 ± 0.163 | 22.207 ± 0.174 | 29.292 ± 17.779 |
| str_nw_mera_closed_120 | **337.421 ± 2.897** | 386.004 ± 2.369 | 488.238 ± 9.677 | 453.802 ± 13.712 | 811.968 ± 46.228 |
| str_nw_mera_open_26 | 343.211 ± 1.595 | 374.407 ± 6.475 | **301.721 ± 8.519** | 362.850 ± 7.202 | 890.828 ± 27.734 |
| tensornetwork_permutation_focus_step409_316 | **97.679 ± 0.965** | 189.167 ± 5.304 | 183.995 ± 2.217 | 360.322 ± 9.275 | 462.313 ± 49.129 |
| tensornetwork_permutation_light_415 | **100.743 ± 1.113** | 137.980 ± 1.399 | 181.555 ± 1.428 | 443.780 ± 164.762 | 427.616 ± 51.157 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=8, OMP_NUM_THREADS=8, RAYON_NUM_THREADS=8.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.498 ± 0.016 | **1.065 ± 0.014** | 5.997 ± 0.009 | 1.468 ± 0.047 | 11.750 ± 5.936 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.473 ± 0.004 | **0.312 ± 0.008** | 5.999 ± 0.016 | 1.040 ± 0.027 | 1.065 ± 0.070 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.131 ± 0.028 | 0.129 ± 0.013 | **0.049 ± 0.001** | 0.550 ± 0.218 | 0.947 ± 0.377 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.109 ± 0.000 | 0.128 ± 0.001 | **0.049 ± 0.001** | 0.575 ± 0.199 | 3.460 ± 1.273 |
| bin_elementwise_mul_2048x2048 | **1.114 ± 0.269** | 1.215 ± 0.204 | 3.087 ± 0.292 | 6.158 ± 0.374 | 9.863 ± 6.715 |
| bin_elementwise_mul_512x512 | 0.026 ± 0.001 | **0.021 ± 0.001** | 0.037 ± 0.003 | 0.320 ± 0.192 | 0.143 ± 0.021 |
| bin_matmul_1024 | 10.619 ± 0.081 | **8.906 ± 0.017** | 18.822 ± 0.145 | 13.157 ± 1.436 | 43.805 ± 20.307 |
| bin_matmul_256 | 0.243 ± 0.006 | **0.169 ± 0.002** | 0.460 ± 0.005 | 0.429 ± 0.144 | 0.680 ± 0.006 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.001 | **0.008 ± 0.000** | 0.029 ± 0.001 | 0.167 ± 0.010 | 0.015 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.030 ± 0.001** | **0.030 ± 0.001** | 0.143 ± 0.002 | 0.228 ± 0.052 | 0.053 ± 0.002 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.001 | **0.012 ± 0.000** | 0.028 ± 0.001 | 0.100 ± 0.015 | 0.028 ± 0.002 |
| bin_outer_product_4096 | 9.065 ± 0.308 | 9.056 ± 0.333 | **8.491 ± 0.340** | 16.531 ± 0.271 | 298.982 ± 3.450 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 11.698 ± 0.068 | **10.361 ± 0.007** | 14.862 ± 0.189 | 13.548 ± 0.633 | 33.551 ± 16.843 |
| gm_queen5_5_3.wcsp | 577.333 ± 5.209 | **514.001 ± 9.459** | 669.535 ± 4.912 | 805.819 ± 16.083 | 2635.690 ± 56.795 |
| lm_batch_likelihood_brackets_4_4d | 10.565 ± 0.148 | 11.628 ± 0.094 | **8.367 ± 0.434** | 19.261 ± 0.431 | 70.851 ± 42.284 |
| lm_batch_likelihood_sentence_3_12d | **15.575 ± 0.082** | 30.165 ± 0.226 | 18.596 ± 0.081 | 37.776 ± 2.739 | 95.772 ± 40.057 |
| lm_batch_likelihood_sentence_4_4d | 9.751 ± 0.083 | 12.656 ± 1.004 | **9.108 ± 0.647** | 18.974 ± 0.295 | 46.292 ± 34.753 |
| nary_matmul_chain_64 | 0.040 ± 0.001 | **0.030 ± 0.000** | 0.051 ± 0.002 | 0.135 ± 0.008 | 0.110 ± 0.038 |
| str_matrix_chain_multiplication_100 | **5.549 ± 0.401** | 5.692 ± 0.150 | 5.581 ± 0.067 | 7.716 ± 0.151 | 11.603 ± 0.403 |
| str_mps_varying_inner_product_200 | 13.458 ± 0.082 | **11.433 ± 0.199** | 15.039 ± 0.058 | 21.478 ± 0.317 | 25.839 ± 5.707 |
| str_nw_mera_closed_120 | 428.966 ± 2.775 | 380.753 ± 3.957 | 369.825 ± 3.714 | **348.774 ± 7.352** | 644.892 ± 36.744 |
| str_nw_mera_open_26 | 349.525 ± 2.889 | 380.406 ± 3.398 | **296.618 ± 2.823** | 403.211 ± 5.584 | 916.589 ± 50.586 |
| tensornetwork_permutation_focus_step409_316 | **97.679 ± 0.965** | 189.167 ± 5.304 | 183.995 ± 2.217 | 360.322 ± 9.275 | 462.313 ± 49.129 |
| tensornetwork_permutation_light_415 | **100.743 ± 1.113** | 137.980 ± 1.399 | 181.555 ± 1.428 | 443.780 ± 164.762 | 427.616 ± 51.157 |

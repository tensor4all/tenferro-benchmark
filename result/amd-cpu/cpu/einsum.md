# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `89c207f1ca97cde3ae0f31586747a039ca4e5cc6`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261114_180711/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261114_180711/run.yaml`
- Timestamp: `2026-11-14T18:07:11.056991+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261114_180711`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

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

- tenferro-rs features: `blas-mkl`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.2`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.4`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `1`

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

- `data/results/amd-cpu/cpu/einsum/20261114_180711/jax_cpu_t1_20261114_180711.log`
- `data/results/amd-cpu/cpu/einsum/20261114_180711/julia_omeinsum_t1_20261114_180711.log`
- `data/results/amd-cpu/cpu/einsum/20261114_180711/pytorch_cpu_t1_20261114_180711.log`
- `data/results/amd-cpu/cpu/einsum/20261114_180711/tenferro_eager_t1_20261114_180711.log`
- `data/results/amd-cpu/cpu/einsum/20261114_180711/tenferro_trace_t1_20261114_180711.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.183 ± 0.211 | **3.877 ± 0.039** | 4.357 ± 0.152 | 4.273 ± 0.096 | 10.057 ± 3.429 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.764 ± 0.006 | **0.596 ± 0.018** | 0.666 ± 0.023 | 0.831 ± 0.016 | 0.947 ± 1.389 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.740 ± 0.009 | 0.733 ± 0.013 | **0.235 ± 0.033** | 0.607 ± 0.025 | 4.058 ± 1.910 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.240 ± 0.018 | 0.731 ± 0.006 | **0.224 ± 0.015** | 0.610 ± 0.036 | 6.086 ± 4.620 |
| bin_elementwise_mul_2048x2048 | 2.098 ± 0.062 | **1.933 ± 0.144** | 17.569 ± 0.171 | 18.534 ± 0.577 | 18.984 ± 3.623 |
| bin_elementwise_mul_512x512 | **0.125 ± 0.017** | 0.136 ± 0.032 | 0.163 ± 0.019 | 0.431 ± 0.028 | 0.171 ± 0.728 |
| bin_matmul_1024 | 57.366 ± 0.635 | **50.133 ± 0.890** | 61.204 ± 0.747 | 64.358 ± 1.185 | 75.100 ± 5.577 |
| bin_matmul_256 | 1.042 ± 0.015 | **0.848 ± 0.020** | 1.358 ± 0.027 | 1.158 ± 0.019 | 1.262 ± 0.241 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | **0.010 ± 0.000** | **0.010 ± 0.001** | 0.036 ± 0.005 | 0.201 ± 0.019 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.044 ± 0.000 | **0.040 ± 0.019** | 0.145 ± 0.009 | 0.247 ± 0.029 | 0.053 ± 0.019 |
| bin_omeinsum_matmul_10x10 | 0.022 ± 0.001 | **0.011 ± 0.007** | 0.035 ± 0.004 | 0.155 ± 0.030 | 0.026 ± 0.003 |
| bin_outer_product_4096 | 66.593 ± 0.806 | 66.537 ± 0.695 | **56.252 ± 1.018** | 57.349 ± 1.662 | 321.504 ± 81.965 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 58.566 ± 0.700 | **52.416 ± 1.979** | 62.860 ± 0.957 | 68.094 ± 1.325 | 65.027 ± 9.627 |
| gm_queen5_5_3.wcsp | 7364.090 ± 156.853 | 7049.672 ± 46.178 | **6338.743 ± 59.996** | 8328.996 ± 127.399 | 9433.235 ± 142.970 |
| lm_batch_likelihood_brackets_4_4d | 29.117 ± 1.546 | 36.296 ± 1.321 | 35.801 ± 2.763 | **16.750 ± 1.336** | 33.297 ± 13.932 |
| lm_batch_likelihood_sentence_3_12d | **61.476 ± 2.027** | 103.749 ± 26.132 | 69.346 ± 1.277 | 101.311 ± 5.184 | 79.682 ± 9.855 |
| lm_batch_likelihood_sentence_4_4d | 30.105 ± 0.338 | 40.570 ± 1.924 | 38.102 ± 0.553 | **17.150 ± 0.473** | 35.851 ± 0.498 |
| nary_matmul_chain_64 | **0.057 ± 0.001** | 0.079 ± 0.001 | 0.079 ± 0.004 | 0.198 ± 0.015 | 0.078 ± 0.006 |
| str_matrix_chain_multiplication_100 | **14.842 ± 0.651** | 16.138 ± 0.236 | 16.464 ± 0.288 | 15.429 ± 1.155 | 16.951 ± 0.333 |
| str_mps_varying_inner_product_200 | **20.144 ± 0.267** | 22.888 ± 0.277 | 26.036 ± 0.570 | 24.472 ± 0.680 | 26.076 ± 1.307 |
| str_nw_mera_closed_120 | **1685.250 ± 23.828** | 1888.309 ± 41.995 | 2038.806 ± 8.799 | 2523.644 ± 25.906 | 1861.333 ± 42.436 |
| str_nw_mera_open_26 | 1342.904 ± 5.858 | 1532.209 ± 5.652 | **1248.062 ± 14.116** | 1980.600 ± 14.651 | 1656.050 ± 35.833 |
| tensornetwork_permutation_focus_step409_316 | **486.495 ± 7.717** | 567.959 ± 6.481 | 924.738 ± 4.467 | 1192.991 ± 374.190 | 599.536 ± 37.405 |
| tensornetwork_permutation_light_415 | **475.234 ± 5.550** | 515.117 ± 14.503 | 886.318 ± 4.132 | 1138.166 ± 202.963 | 574.035 ± 26.920 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.183 ± 0.211 | **3.877 ± 0.039** | 4.357 ± 0.152 | 4.273 ± 0.096 | 10.057 ± 3.429 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.764 ± 0.006 | **0.596 ± 0.018** | 0.666 ± 0.023 | 0.831 ± 0.016 | 0.947 ± 1.389 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.740 ± 0.009 | 0.733 ± 0.013 | **0.235 ± 0.033** | 0.607 ± 0.025 | 4.058 ± 1.910 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.240 ± 0.018 | 0.731 ± 0.006 | **0.224 ± 0.015** | 0.610 ± 0.036 | 6.086 ± 4.620 |
| bin_elementwise_mul_2048x2048 | 2.098 ± 0.062 | **1.933 ± 0.144** | 17.569 ± 0.171 | 18.534 ± 0.577 | 18.984 ± 3.623 |
| bin_elementwise_mul_512x512 | **0.125 ± 0.017** | 0.136 ± 0.032 | 0.163 ± 0.019 | 0.431 ± 0.028 | 0.171 ± 0.728 |
| bin_matmul_1024 | 57.366 ± 0.635 | **50.133 ± 0.890** | 61.204 ± 0.747 | 64.358 ± 1.185 | 75.100 ± 5.577 |
| bin_matmul_256 | 1.042 ± 0.015 | **0.848 ± 0.020** | 1.358 ± 0.027 | 1.158 ± 0.019 | 1.262 ± 0.241 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | **0.010 ± 0.000** | **0.010 ± 0.001** | 0.036 ± 0.005 | 0.201 ± 0.019 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.044 ± 0.000 | **0.040 ± 0.019** | 0.145 ± 0.009 | 0.247 ± 0.029 | 0.053 ± 0.019 |
| bin_omeinsum_matmul_10x10 | 0.022 ± 0.001 | **0.011 ± 0.007** | 0.035 ± 0.004 | 0.155 ± 0.030 | 0.026 ± 0.003 |
| bin_outer_product_4096 | 66.593 ± 0.806 | 66.537 ± 0.695 | **56.252 ± 1.018** | 57.349 ± 1.662 | 321.504 ± 81.965 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 58.566 ± 0.700 | **52.416 ± 1.979** | 62.860 ± 0.957 | 68.094 ± 1.325 | 65.027 ± 9.627 |
| gm_queen5_5_3.wcsp | **2490.376 ± 36.211** | 2693.159 ± 30.803 | 2613.279 ± 56.446 | 2844.656 ± 32.822 | 3396.351 ± 84.535 |
| lm_batch_likelihood_brackets_4_4d | 28.610 ± 1.024 | 36.524 ± 2.624 | 36.949 ± 0.565 | **15.571 ± 0.459** | 36.750 ± 15.611 |
| lm_batch_likelihood_sentence_3_12d | **60.410 ± 0.915** | 179.674 ± 12.005 | 73.123 ± 2.473 | 71.224 ± 3.352 | 95.401 ± 32.837 |
| lm_batch_likelihood_sentence_4_4d | 30.012 ± 0.842 | 42.235 ± 6.553 | 39.186 ± 0.689 | **15.990 ± 0.301** | 37.728 ± 35.412 |
| nary_matmul_chain_64 | **0.057 ± 0.001** | 0.079 ± 0.001 | 0.079 ± 0.004 | 0.198 ± 0.015 | 0.078 ± 0.006 |
| str_matrix_chain_multiplication_100 | 15.608 ± 0.214 | 15.852 ± 0.244 | 16.584 ± 0.632 | **15.115 ± 0.307** | 17.094 ± 0.321 |
| str_mps_varying_inner_product_200 | **24.100 ± 0.308** | 24.462 ± 0.336 | 25.793 ± 0.306 | 24.325 ± 0.811 | 25.783 ± 9.602 |
| str_nw_mera_closed_120 | 1531.343 ± 13.634 | 1524.015 ± 14.192 | 1585.823 ± 14.526 | 2110.012 ± 33.208 | **1522.202 ± 14.186** |
| str_nw_mera_open_26 | 1340.769 ± 10.280 | 1484.568 ± 14.872 | **1274.529 ± 8.641** | 2032.300 ± 20.555 | 1652.035 ± 72.539 |
| tensornetwork_permutation_focus_step409_316 | **486.495 ± 7.717** | 567.959 ± 6.481 | 924.738 ± 4.467 | 1192.991 ± 374.190 | 599.536 ± 37.405 |
| tensornetwork_permutation_light_415 | **475.234 ± 5.550** | 515.117 ± 14.503 | 886.318 ± 4.132 | 1138.166 ± 202.963 | 574.035 ± 26.920 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261114_184911/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261114_184911/run.yaml`
- Timestamp: `2026-11-14T18:49:11.517432+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261114_184911`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

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

- tenferro-rs features: `blas-mkl`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.2`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.4`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `4`

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

- `data/results/amd-cpu/cpu/einsum/20261114_184911/jax_cpu_t4_20261114_184911.log`
- `data/results/amd-cpu/cpu/einsum/20261114_184911/julia_omeinsum_t4_20261114_184911.log`
- `data/results/amd-cpu/cpu/einsum/20261114_184911/pytorch_cpu_t4_20261114_184911.log`
- `data/results/amd-cpu/cpu/einsum/20261114_184911/tenferro_eager_t4_20261114_184911.log`
- `data/results/amd-cpu/cpu/einsum/20261114_184911/tenferro_trace_t4_20261114_184911.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 9.804 ± 0.175 | 5.264 ± 0.094 | **3.223 ± 3.662** | 3.242 ± 0.606 | 12.593 ± 5.261 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.953 ± 0.012 | 1.374 ± 0.025 | **0.389 ± 0.012** | 0.848 ± 0.015 | 0.929 ± 1.244 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.307 ± 0.021 | 0.316 ± 0.023 | **0.209 ± 0.014** | 0.478 ± 0.056 | 2.798 ± 3.354 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.289 ± 0.014 | 0.311 ± 0.015 | **0.214 ± 0.036** | 0.481 ± 0.017 | 5.600 ± 4.267 |
| bin_elementwise_mul_2048x2048 | 1.023 ± 0.021 | **0.904 ± 0.056** | 5.904 ± 0.112 | 9.715 ± 0.261 | 19.149 ± 1.093 |
| bin_elementwise_mul_512x512 | **0.137 ± 0.014** | 0.139 ± 0.020 | 0.167 ± 0.029 | 0.401 ± 0.035 | 0.178 ± 0.823 |
| bin_matmul_1024 | 62.750 ± 1.636 | 52.789 ± 1.205 | 36.580 ± 0.831 | **36.121 ± 0.655** | 51.505 ± 10.429 |
| bin_matmul_256 | 1.266 ± 0.009 | 0.938 ± 0.172 | 0.856 ± 0.008 | **0.850 ± 0.069** | 0.925 ± 0.017 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.001 | **0.010 ± 0.000** | 0.052 ± 0.005 | 0.211 ± 0.025 | 0.014 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.071 ± 0.004 | **0.066 ± 0.004** | 0.223 ± 0.018 | 0.263 ± 0.033 | 0.079 ± 0.026 |
| bin_omeinsum_matmul_10x10 | 0.070 ± 0.003 | 0.047 ± 0.002 | 0.049 ± 0.007 | 0.152 ± 0.040 | **0.026 ± 0.002** |
| bin_outer_product_4096 | 22.344 ± 0.491 | 22.440 ± 0.578 | **18.578 ± 0.522** | 29.141 ± 1.533 | 306.692 ± 36.557 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 60.939 ± 1.216 | 55.657 ± 1.618 | **31.130 ± 0.976** | 39.362 ± 2.148 | 46.241 ± 13.148 |
| gm_queen5_5_3.wcsp | 6663.345 ± 80.595 | 6061.519 ± 81.797 | **2705.358 ± 57.181** | 5534.722 ± 92.916 | 8779.609 ± 111.212 |
| lm_batch_likelihood_brackets_4_4d | 30.664 ± 0.747 | 53.322 ± 2.360 | **17.057 ± 0.566** | 24.578 ± 1.423 | 49.331 ± 15.829 |
| lm_batch_likelihood_sentence_3_12d | 64.207 ± 1.849 | 170.266 ± 25.145 | **33.353 ± 2.723** | 54.030 ± 8.918 | 67.005 ± 11.932 |
| lm_batch_likelihood_sentence_4_4d | 33.613 ± 1.412 | 59.206 ± 3.114 | **17.908 ± 0.567** | 25.305 ± 2.243 | 41.788 ± 2.339 |
| nary_matmul_chain_64 | 0.148 ± 0.011 | 0.130 ± 0.001 | **0.096 ± 0.035** | 0.219 ± 0.015 | 0.154 ± 0.040 |
| str_matrix_chain_multiplication_100 | 17.863 ± 0.350 | 19.195 ± 0.381 | **11.108 ± 0.228** | 15.733 ± 0.709 | 14.782 ± 0.680 |
| str_mps_varying_inner_product_200 | 27.047 ± 0.563 | 29.061 ± 0.392 | **26.596 ± 0.596** | 31.501 ± 0.688 | 34.238 ± 2.063 |
| str_nw_mera_closed_120 | 1699.776 ± 9.540 | 1904.487 ± 21.743 | **966.393 ± 9.715** | 1187.862 ± 41.213 | 1247.161 ± 50.441 |
| str_nw_mera_open_26 | 1352.193 ± 13.534 | 1368.928 ± 30.260 | **592.683 ± 16.775** | 950.608 ± 19.364 | 1291.221 ± 35.376 |
| tensornetwork_permutation_focus_step409_316 | 476.713 ± 6.822 | 485.166 ± 6.970 | **345.892 ± 4.746** | 420.174 ± 13.526 | 551.413 ± 82.642 |
| tensornetwork_permutation_light_415 | 472.269 ± 4.801 | 465.742 ± 5.731 | **348.819 ± 22.746** | 432.042 ± 15.375 | 552.473 ± 37.124 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 9.804 ± 0.175 | 5.264 ± 0.094 | **3.223 ± 3.662** | 3.242 ± 0.606 | 12.593 ± 5.261 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.953 ± 0.012 | 1.374 ± 0.025 | **0.389 ± 0.012** | 0.848 ± 0.015 | 0.929 ± 1.244 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.307 ± 0.021 | 0.316 ± 0.023 | **0.209 ± 0.014** | 0.478 ± 0.056 | 2.798 ± 3.354 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.289 ± 0.014 | 0.311 ± 0.015 | **0.214 ± 0.036** | 0.481 ± 0.017 | 5.600 ± 4.267 |
| bin_elementwise_mul_2048x2048 | 1.023 ± 0.021 | **0.904 ± 0.056** | 5.904 ± 0.112 | 9.715 ± 0.261 | 19.149 ± 1.093 |
| bin_elementwise_mul_512x512 | **0.137 ± 0.014** | 0.139 ± 0.020 | 0.167 ± 0.029 | 0.401 ± 0.035 | 0.178 ± 0.823 |
| bin_matmul_1024 | 62.750 ± 1.636 | 52.789 ± 1.205 | 36.580 ± 0.831 | **36.121 ± 0.655** | 51.505 ± 10.429 |
| bin_matmul_256 | 1.266 ± 0.009 | 0.938 ± 0.172 | 0.856 ± 0.008 | **0.850 ± 0.069** | 0.925 ± 0.017 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.001 | **0.010 ± 0.000** | 0.052 ± 0.005 | 0.211 ± 0.025 | 0.014 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.071 ± 0.004 | **0.066 ± 0.004** | 0.223 ± 0.018 | 0.263 ± 0.033 | 0.079 ± 0.026 |
| bin_omeinsum_matmul_10x10 | 0.070 ± 0.003 | 0.047 ± 0.002 | 0.049 ± 0.007 | 0.152 ± 0.040 | **0.026 ± 0.002** |
| bin_outer_product_4096 | 22.344 ± 0.491 | 22.440 ± 0.578 | **18.578 ± 0.522** | 29.141 ± 1.533 | 306.692 ± 36.557 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 60.939 ± 1.216 | 55.657 ± 1.618 | **31.130 ± 0.976** | 39.362 ± 2.148 | 46.241 ± 13.148 |
| gm_queen5_5_3.wcsp | 2397.898 ± 40.050 | 2130.635 ± 31.462 | **1126.129 ± 22.487** | 1877.479 ± 23.575 | 3178.279 ± 61.163 |
| lm_batch_likelihood_brackets_4_4d | 31.243 ± 0.759 | 54.695 ± 2.340 | **18.367 ± 0.327** | 24.012 ± 1.936 | 46.718 ± 5.003 |
| lm_batch_likelihood_sentence_3_12d | 58.106 ± 1.636 | 211.059 ± 6.621 | **33.969 ± 2.759** | 50.219 ± 7.143 | 114.239 ± 46.047 |
| lm_batch_likelihood_sentence_4_4d | 33.067 ± 0.735 | 62.790 ± 1.470 | **18.760 ± 0.574** | 23.799 ± 1.399 | 51.277 ± 38.077 |
| nary_matmul_chain_64 | 0.148 ± 0.011 | 0.130 ± 0.001 | **0.096 ± 0.035** | 0.219 ± 0.015 | 0.154 ± 0.040 |
| str_matrix_chain_multiplication_100 | 16.694 ± 0.292 | 19.078 ± 0.394 | **11.185 ± 0.210** | 15.927 ± 0.628 | 14.744 ± 1.557 |
| str_mps_varying_inner_product_200 | 29.721 ± 1.666 | 31.473 ± 0.877 | 29.692 ± 0.879 | 31.811 ± 1.293 | **26.533 ± 10.385** |
| str_nw_mera_closed_120 | 1555.482 ± 8.438 | 1573.006 ± 18.532 | **772.427 ± 10.193** | 1001.401 ± 20.971 | 964.985 ± 55.105 |
| str_nw_mera_open_26 | 1346.032 ± 11.276 | 1369.749 ± 15.379 | **593.990 ± 15.079** | 979.331 ± 34.562 | 1258.974 ± 22.201 |
| tensornetwork_permutation_focus_step409_316 | 476.713 ± 6.822 | 485.166 ± 6.970 | **345.892 ± 4.746** | 420.174 ± 13.526 | 551.413 ± 82.642 |
| tensornetwork_permutation_light_415 | 472.269 ± 4.801 | 465.742 ± 5.731 | **348.819 ± 22.746** | 432.042 ± 15.375 | 552.473 ± 37.124 |

# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `4edf89377c13d5c8a287dd562b529de6bb85f4dc`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261112_180742/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261112_180742/run.yaml`
- Timestamp: `2026-11-12T18:07:42.528102+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261112_180742`.

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

- Julia: version `1.13.3`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `1`

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

- `data/results/amd-cpu/cpu/einsum/20261112_180742/jax_cpu_t1_20261112_180742.log`
- `data/results/amd-cpu/cpu/einsum/20261112_180742/julia_omeinsum_t1_20261112_180742.log`
- `data/results/amd-cpu/cpu/einsum/20261112_180742/pytorch_cpu_t1_20261112_180742.log`
- `data/results/amd-cpu/cpu/einsum/20261112_180742/tenferro_eager_t1_20261112_180742.log`
- `data/results/amd-cpu/cpu/einsum/20261112_180742/tenferro_trace_t1_20261112_180742.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.926 ± 0.045 | **3.867 ± 0.049** | 4.273 ± 0.112 | 4.262 ± 0.124 | 11.529 ± 5.471 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.788 ± 0.016 | **0.608 ± 0.035** | 0.665 ± 0.017 | 0.789 ± 0.016 | 0.935 ± 0.027 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.733 ± 0.006 | 0.732 ± 0.003 | **0.241 ± 0.025** | 0.565 ± 0.019 | 0.899 ± 3.212 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.210 ± 0.015 | 0.736 ± 0.005 | **0.236 ± 0.019** | 0.567 ± 0.025 | 5.384 ± 9.548 |
| bin_elementwise_mul_2048x2048 | 1.989 ± 0.093 | **1.803 ± 0.164** | 16.329 ± 0.320 | 17.599 ± 0.514 | 18.471 ± 2.921 |
| bin_elementwise_mul_512x512 | 0.139 ± 0.011 | **0.123 ± 0.006** | 0.149 ± 0.008 | 0.372 ± 0.030 | 0.177 ± 0.018 |
| bin_matmul_1024 | 57.068 ± 0.147 | **49.223 ± 0.703** | 60.288 ± 0.445 | 63.445 ± 1.179 | 75.625 ± 25.577 |
| bin_matmul_256 | 1.042 ± 0.021 | **0.836 ± 0.016** | 1.348 ± 0.023 | 1.230 ± 0.018 | 2.096 ± 0.847 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.010 ± 0.000 | **0.006 ± 0.005** | 0.037 ± 0.006 | 0.205 ± 0.025 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.042 ± 0.001** | 0.045 ± 0.008 | 0.146 ± 0.005 | 0.282 ± 0.041 | 0.090 ± 0.029 |
| bin_omeinsum_matmul_10x10 | 0.036 ± 0.015 | **0.010 ± 0.000** | 0.035 ± 0.004 | 0.178 ± 0.026 | 0.027 ± 0.002 |
| bin_outer_product_4096 | 63.773 ± 0.430 | 63.719 ± 0.338 | **53.953 ± 0.649** | 56.546 ± 0.657 | 311.938 ± 1.873 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 59.313 ± 0.602 | **51.280 ± 0.612** | 61.183 ± 0.239 | 66.975 ± 2.308 | 67.335 ± 14.463 |
| gm_queen5_5_3.wcsp | 7009.584 ± 100.917 | 6679.561 ± 74.482 | **6249.053 ± 35.796** | 8057.694 ± 33.189 | 9697.886 ± 191.834 |
| lm_batch_likelihood_brackets_4_4d | 27.546 ± 0.485 | 34.375 ± 1.756 | 34.884 ± 0.587 | **15.270 ± 0.698** | 65.921 ± 28.543 |
| lm_batch_likelihood_sentence_3_12d | **59.524 ± 1.006** | 120.154 ± 18.763 | 106.738 ± 0.875 | 95.186 ± 2.375 | 139.888 ± 21.774 |
| lm_batch_likelihood_sentence_4_4d | 29.013 ± 0.754 | 38.257 ± 1.274 | 36.729 ± 0.868 | **16.977 ± 0.366** | 69.221 ± 34.605 |
| nary_matmul_chain_64 | **0.057 ± 0.001** | **0.057 ± 0.022** | 0.078 ± 0.003 | 0.199 ± 0.010 | 0.136 ± 0.038 |
| str_matrix_chain_multiplication_100 | **14.188 ± 0.186** | 15.282 ± 0.365 | 15.970 ± 0.153 | 15.050 ± 1.987 | 17.355 ± 0.929 |
| str_mps_varying_inner_product_200 | **19.896 ± 0.170** | 21.598 ± 0.116 | 26.781 ± 0.530 | 24.066 ± 0.337 | 26.465 ± 7.701 |
| str_nw_mera_closed_120 | **1643.894 ± 8.636** | 1828.350 ± 11.966 | 1965.612 ± 10.499 | 2504.718 ± 9.400 | 1909.657 ± 31.503 |
| str_nw_mera_open_26 | 1299.883 ± 7.969 | 1510.919 ± 35.350 | **1210.734 ± 8.432** | 1949.743 ± 15.762 | 1732.135 ± 20.576 |
| tensornetwork_permutation_focus_step409_316 | **472.583 ± 4.241** | 585.156 ± 7.499 | 879.114 ± 3.080 | 923.549 ± 350.002 | 635.339 ± 23.867 |
| tensornetwork_permutation_light_415 | **464.386 ± 2.803** | 527.780 ± 5.030 | 839.974 ± 4.334 | 1032.052 ± 316.895 | 658.186 ± 15.602 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.926 ± 0.045 | **3.867 ± 0.049** | 4.273 ± 0.112 | 4.262 ± 0.124 | 11.529 ± 5.471 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.788 ± 0.016 | **0.608 ± 0.035** | 0.665 ± 0.017 | 0.789 ± 0.016 | 0.935 ± 0.027 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.733 ± 0.006 | 0.732 ± 0.003 | **0.241 ± 0.025** | 0.565 ± 0.019 | 0.899 ± 3.212 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.210 ± 0.015 | 0.736 ± 0.005 | **0.236 ± 0.019** | 0.567 ± 0.025 | 5.384 ± 9.548 |
| bin_elementwise_mul_2048x2048 | 1.989 ± 0.093 | **1.803 ± 0.164** | 16.329 ± 0.320 | 17.599 ± 0.514 | 18.471 ± 2.921 |
| bin_elementwise_mul_512x512 | 0.139 ± 0.011 | **0.123 ± 0.006** | 0.149 ± 0.008 | 0.372 ± 0.030 | 0.177 ± 0.018 |
| bin_matmul_1024 | 57.068 ± 0.147 | **49.223 ± 0.703** | 60.288 ± 0.445 | 63.445 ± 1.179 | 75.625 ± 25.577 |
| bin_matmul_256 | 1.042 ± 0.021 | **0.836 ± 0.016** | 1.348 ± 0.023 | 1.230 ± 0.018 | 2.096 ± 0.847 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.010 ± 0.000 | **0.006 ± 0.005** | 0.037 ± 0.006 | 0.205 ± 0.025 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.042 ± 0.001** | 0.045 ± 0.008 | 0.146 ± 0.005 | 0.282 ± 0.041 | 0.090 ± 0.029 |
| bin_omeinsum_matmul_10x10 | 0.036 ± 0.015 | **0.010 ± 0.000** | 0.035 ± 0.004 | 0.178 ± 0.026 | 0.027 ± 0.002 |
| bin_outer_product_4096 | 63.773 ± 0.430 | 63.719 ± 0.338 | **53.953 ± 0.649** | 56.546 ± 0.657 | 311.938 ± 1.873 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 59.313 ± 0.602 | **51.280 ± 0.612** | 61.183 ± 0.239 | 66.975 ± 2.308 | 67.335 ± 14.463 |
| gm_queen5_5_3.wcsp | **2388.411 ± 43.757** | 2629.370 ± 26.348 | 2491.366 ± 69.605 | 2773.583 ± 25.858 | 3372.235 ± 59.436 |
| lm_batch_likelihood_brackets_4_4d | 27.152 ± 0.198 | 34.534 ± 1.254 | 35.926 ± 0.565 | **15.163 ± 0.496** | 70.800 ± 28.980 |
| lm_batch_likelihood_sentence_3_12d | **57.363 ± 1.679** | 177.340 ± 5.662 | 111.501 ± 0.992 | 71.931 ± 1.439 | 127.153 ± 25.781 |
| lm_batch_likelihood_sentence_4_4d | 28.141 ± 0.198 | 40.726 ± 1.707 | 38.917 ± 0.908 | **16.524 ± 0.154** | 53.280 ± 44.701 |
| nary_matmul_chain_64 | **0.057 ± 0.001** | **0.057 ± 0.022** | 0.078 ± 0.003 | 0.199 ± 0.010 | 0.136 ± 0.038 |
| str_matrix_chain_multiplication_100 | 14.995 ± 0.213 | 15.646 ± 0.150 | 16.014 ± 0.169 | **14.986 ± 0.274** | 16.924 ± 0.527 |
| str_mps_varying_inner_product_200 | **23.833 ± 0.346** | 24.196 ± 0.206 | 26.212 ± 0.256 | 24.325 ± 6.533 | 25.393 ± 9.226 |
| str_nw_mera_closed_120 | 1511.427 ± 12.805 | **1510.371 ± 17.064** | 1528.733 ± 22.957 | 2050.601 ± 39.689 | 1552.164 ± 51.058 |
| str_nw_mera_open_26 | 1314.014 ± 6.800 | 1474.874 ± 11.440 | **1253.928 ± 15.249** | 2009.513 ± 29.434 | 1740.186 ± 57.682 |
| tensornetwork_permutation_focus_step409_316 | **472.583 ± 4.241** | 585.156 ± 7.499 | 879.114 ± 3.080 | 923.549 ± 350.002 | 635.339 ± 23.867 |
| tensornetwork_permutation_light_415 | **464.386 ± 2.803** | 527.780 ± 5.030 | 839.974 ± 4.334 | 1032.052 ± 316.895 | 658.186 ± 15.602 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261112_184910/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261112_184910/run.yaml`
- Timestamp: `2026-11-12T18:49:10.227921+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261112_184910`.

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

- Julia: version `1.13.3`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `4`

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

- `data/results/amd-cpu/cpu/einsum/20261112_184910/jax_cpu_t4_20261112_184910.log`
- `data/results/amd-cpu/cpu/einsum/20261112_184910/julia_omeinsum_t4_20261112_184910.log`
- `data/results/amd-cpu/cpu/einsum/20261112_184910/pytorch_cpu_t4_20261112_184910.log`
- `data/results/amd-cpu/cpu/einsum/20261112_184910/tenferro_eager_t4_20261112_184910.log`
- `data/results/amd-cpu/cpu/einsum/20261112_184910/tenferro_trace_t4_20261112_184910.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 9.750 ± 0.101 | 5.204 ± 0.040 | **2.751 ± 3.324** | 2.962 ± 0.645 | 11.155 ± 6.380 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.975 ± 0.058 | 1.367 ± 0.014 | **0.388 ± 0.019** | 0.859 ± 0.035 | 0.993 ± 0.188 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.322 ± 0.022 | 0.311 ± 0.022 | **0.230 ± 0.029** | 0.474 ± 0.031 | 0.982 ± 0.241 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.292 ± 0.030 | 0.318 ± 0.023 | **0.193 ± 0.027** | 0.468 ± 0.033 | 5.928 ± 5.633 |
| bin_elementwise_mul_2048x2048 | 1.012 ± 0.063 | **0.878 ± 0.033** | 6.205 ± 0.263 | 9.520 ± 0.251 | 18.722 ± 0.277 |
| bin_elementwise_mul_512x512 | 0.139 ± 0.013 | **0.130 ± 0.001** | 0.159 ± 0.058 | 0.411 ± 0.027 | 0.183 ± 0.022 |
| bin_matmul_1024 | 62.387 ± 0.548 | 50.915 ± 0.624 | 37.991 ± 0.456 | **36.286 ± 0.531** | 55.472 ± 6.501 |
| bin_matmul_256 | 1.285 ± 0.009 | 0.929 ± 0.163 | **0.863 ± 0.007** | 0.865 ± 0.129 | 0.930 ± 0.012 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.016 ± 0.000 | **0.010 ± 0.000** | 0.053 ± 0.005 | 0.208 ± 0.033 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.065 ± 0.003 | 0.062 ± 0.003 | 0.222 ± 0.015 | 0.245 ± 0.031 | **0.051 ± 0.003** |
| bin_omeinsum_matmul_10x10 | 0.071 ± 0.003 | 0.047 ± 0.003 | 0.049 ± 0.005 | 0.154 ± 0.019 | **0.025 ± 0.001** |
| bin_outer_product_4096 | 24.236 ± 0.431 | 23.894 ± 0.291 | **19.241 ± 0.266** | 29.113 ± 0.528 | 301.184 ± 2.986 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 61.159 ± 0.526 | 53.357 ± 0.756 | **31.605 ± 0.497** | 39.471 ± 4.310 | 52.145 ± 25.805 |
| gm_queen5_5_3.wcsp | 6516.194 ± 73.996 | 6060.916 ± 85.392 | **2804.089 ± 34.943** | 5335.142 ± 105.802 | 8711.704 ± 232.554 |
| lm_batch_likelihood_brackets_4_4d | 31.216 ± 0.607 | 49.812 ± 1.932 | **17.791 ± 0.420** | 23.161 ± 1.071 | 38.912 ± 9.360 |
| lm_batch_likelihood_sentence_3_12d | 64.394 ± 0.440 | 189.856 ± 9.314 | 60.588 ± 0.895 | **44.579 ± 8.472** | 102.958 ± 16.547 |
| lm_batch_likelihood_sentence_4_4d | 33.461 ± 0.268 | 55.780 ± 3.084 | **18.769 ± 0.416** | 23.883 ± 0.939 | 71.103 ± 23.548 |
| nary_matmul_chain_64 | 0.148 ± 0.008 | 0.133 ± 0.001 | **0.099 ± 0.017** | 0.195 ± 0.009 | 0.153 ± 0.045 |
| str_matrix_chain_multiplication_100 | 17.386 ± 0.182 | 19.092 ± 0.306 | **11.148 ± 0.124** | 15.611 ± 0.613 | 14.857 ± 0.733 |
| str_mps_varying_inner_product_200 | **26.726 ± 0.267** | 28.844 ± 0.272 | 113.147 ± 24.344 | 31.148 ± 0.825 | 31.063 ± 4.331 |
| str_nw_mera_closed_120 | 1702.619 ± 6.911 | 1910.088 ± 32.911 | **976.574 ± 14.640** | 1173.355 ± 27.611 | 1221.983 ± 67.161 |
| str_nw_mera_open_26 | 1332.902 ± 9.898 | 1354.762 ± 18.427 | **596.354 ± 13.264** | 949.303 ± 20.786 | 1250.100 ± 137.585 |
| tensornetwork_permutation_focus_step409_316 | 481.969 ± 5.817 | 508.114 ± 8.272 | **346.452 ± 16.380** | 407.226 ± 13.650 | 535.881 ± 50.897 |
| tensornetwork_permutation_light_415 | 474.467 ± 2.696 | 475.625 ± 4.173 | **343.377 ± 1.782** | 421.047 ± 16.343 | 547.829 ± 37.724 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 9.750 ± 0.101 | 5.204 ± 0.040 | **2.751 ± 3.324** | 2.962 ± 0.645 | 11.155 ± 6.380 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.975 ± 0.058 | 1.367 ± 0.014 | **0.388 ± 0.019** | 0.859 ± 0.035 | 0.993 ± 0.188 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.322 ± 0.022 | 0.311 ± 0.022 | **0.230 ± 0.029** | 0.474 ± 0.031 | 0.982 ± 0.241 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.292 ± 0.030 | 0.318 ± 0.023 | **0.193 ± 0.027** | 0.468 ± 0.033 | 5.928 ± 5.633 |
| bin_elementwise_mul_2048x2048 | 1.012 ± 0.063 | **0.878 ± 0.033** | 6.205 ± 0.263 | 9.520 ± 0.251 | 18.722 ± 0.277 |
| bin_elementwise_mul_512x512 | 0.139 ± 0.013 | **0.130 ± 0.001** | 0.159 ± 0.058 | 0.411 ± 0.027 | 0.183 ± 0.022 |
| bin_matmul_1024 | 62.387 ± 0.548 | 50.915 ± 0.624 | 37.991 ± 0.456 | **36.286 ± 0.531** | 55.472 ± 6.501 |
| bin_matmul_256 | 1.285 ± 0.009 | 0.929 ± 0.163 | **0.863 ± 0.007** | 0.865 ± 0.129 | 0.930 ± 0.012 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.016 ± 0.000 | **0.010 ± 0.000** | 0.053 ± 0.005 | 0.208 ± 0.033 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.065 ± 0.003 | 0.062 ± 0.003 | 0.222 ± 0.015 | 0.245 ± 0.031 | **0.051 ± 0.003** |
| bin_omeinsum_matmul_10x10 | 0.071 ± 0.003 | 0.047 ± 0.003 | 0.049 ± 0.005 | 0.154 ± 0.019 | **0.025 ± 0.001** |
| bin_outer_product_4096 | 24.236 ± 0.431 | 23.894 ± 0.291 | **19.241 ± 0.266** | 29.113 ± 0.528 | 301.184 ± 2.986 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 61.159 ± 0.526 | 53.357 ± 0.756 | **31.605 ± 0.497** | 39.471 ± 4.310 | 52.145 ± 25.805 |
| gm_queen5_5_3.wcsp | 2372.947 ± 38.415 | 2104.145 ± 21.797 | **1117.239 ± 30.720** | 1882.038 ± 34.825 | 3067.677 ± 81.430 |
| lm_batch_likelihood_brackets_4_4d | 30.036 ± 0.267 | 51.644 ± 2.146 | **18.404 ± 0.296** | 23.891 ± 1.404 | 82.806 ± 48.880 |
| lm_batch_likelihood_sentence_3_12d | 57.774 ± 0.721 | 228.298 ± 2.872 | **34.386 ± 1.068** | 56.790 ± 9.012 | 102.033 ± 33.238 |
| lm_batch_likelihood_sentence_4_4d | 31.467 ± 0.252 | 57.988 ± 2.880 | **18.755 ± 0.173** | 24.315 ± 0.917 | 56.994 ± 43.277 |
| nary_matmul_chain_64 | 0.148 ± 0.008 | 0.133 ± 0.001 | **0.099 ± 0.017** | 0.195 ± 0.009 | 0.153 ± 0.045 |
| str_matrix_chain_multiplication_100 | 18.326 ± 0.136 | 19.071 ± 0.471 | **11.139 ± 0.111** | 15.734 ± 0.869 | 14.336 ± 0.691 |
| str_mps_varying_inner_product_200 | 29.972 ± 0.264 | 32.479 ± 0.257 | 29.552 ± 0.554 | 31.981 ± 0.784 | **24.512 ± 7.283** |
| str_nw_mera_closed_120 | 1546.470 ± 12.758 | 1563.736 ± 26.004 | **774.234 ± 14.643** | 993.946 ± 39.286 | 954.138 ± 36.774 |
| str_nw_mera_open_26 | 1339.584 ± 12.293 | 1353.634 ± 11.840 | **594.000 ± 12.145** | 974.245 ± 29.242 | 1199.690 ± 88.541 |
| tensornetwork_permutation_focus_step409_316 | 481.969 ± 5.817 | 508.114 ± 8.272 | **346.452 ± 16.380** | 407.226 ± 13.650 | 535.881 ± 50.897 |
| tensornetwork_permutation_light_415 | 474.467 ± 2.696 | 475.625 ± 4.173 | **343.377 ± 1.782** | 421.047 ± 16.343 | 547.829 ± 37.724 |

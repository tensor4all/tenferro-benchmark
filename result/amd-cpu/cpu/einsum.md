# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `f1d4d8d961ff6b937768111888ad2a493314892a`
- Benchmark commit: `aecee8b998098c968f4a0f59d6228ac81d3e7b09`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261021_180521/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261021_180521/run.yaml`
- Timestamp: `20261021_180521`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261021_180521`.

- tenferro-rs commit: `f1d4d8d961ff6b937768111888ad2a493314892a`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

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

- Julia: version `1.13.2`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `1`

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


- Source table: `data/results/amd-cpu/cpu/einsum/20261021_180521/einsum_table_t1_20261021_180521.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20261021_180521/tenferro_trace_t1_20261021_180521.log`
- `data/results/amd-cpu/cpu/einsum/20261021_180521/tenferro_eager_t1_20261021_180521.log`
- `data/results/amd-cpu/cpu/einsum/20261021_180521/pytorch_cpu_t1_20261021_180521.log`
- `data/results/amd-cpu/cpu/einsum/20261021_180521/jax_cpu_t1_20261021_180521.log`
- `data/results/amd-cpu/cpu/einsum/20261021_180521/julia_omeinsum_t1_20261021_180521.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.840 ± 0.170 | **3.837 ± 0.037** | 4.118 ± 0.057 | 4.251 ± 0.053 | 9.921 ± 6.059 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.787 ± 0.016 | **0.634 ± 0.017** | 0.673 ± 0.030 | 0.847 ± 0.011 | 0.879 ± 1.557 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.200 ± 0.024 | **0.195 ± 0.023** | 0.227 ± 0.058 | 0.597 ± 0.028 | 1.492 ± 0.821 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.419 ± 0.025 | 0.205 ± 0.032 | **0.195 ± 0.018** | 0.619 ± 0.024 | 2.333 ± 2.245 |
| bin_elementwise_mul_2048x2048 | 1.717 ± 0.107 | **1.619 ± 0.064** | 4.400 ± 0.250 | 5.466 ± 0.151 | 5.052 ± 0.366 |
| bin_elementwise_mul_512x512 | **0.102 ± 0.032** | 0.156 ± 0.049 | 0.132 ± 0.010 | 0.401 ± 0.022 | 0.159 ± 0.749 |
| bin_matmul_1024 | 65.292 ± 0.414 | **49.172 ± 0.691** | 74.503 ± 0.131 | 65.378 ± 3.957 | 73.755 ± 15.297 |
| bin_matmul_256 | 1.094 ± 0.026 | **0.847 ± 0.023** | 1.356 ± 0.024 | 1.186 ± 0.010 | 1.265 ± 0.036 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.011 ± 0.000 | **0.006 ± 0.005** | 0.035 ± 0.007 | 0.200 ± 0.027 | 0.013 ± 0.006 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.059 ± 0.001 | **0.043 ± 0.002** | 0.146 ± 0.012 | 0.244 ± 0.025 | 0.085 ± 0.028 |
| bin_omeinsum_matmul_10x10 | 0.026 ± 0.001 | **0.017 ± 0.007** | 0.034 ± 0.006 | 0.148 ± 0.019 | 0.024 ± 0.001 |
| bin_outer_product_4096 | **9.876 ± 0.152** | 9.877 ± 0.198 | 9.970 ± 0.384 | 10.147 ± 0.269 | 952.551 ± 2.252 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 65.437 ± 0.379 | **51.419 ± 0.244** | 60.991 ± 0.391 | 67.353 ± 1.480 | 70.779 ± 5.527 |
| gm_queen5_5_3.wcsp | 5661.253 ± 254.692 | 4507.611 ± 36.746 | **3939.195 ± 915.378** | 5290.317 ± 191.982 | 4957.270 ± 773.447 |
| lm_batch_likelihood_brackets_4_4d | 23.133 ± 0.603 | 34.114 ± 1.499 | 35.047 ± 0.585 | **14.847 ± 0.377** | 61.079 ± 30.434 |
| lm_batch_likelihood_sentence_3_12d | **55.630 ± 1.333** | 91.285 ± 2.363 | 73.689 ± 2.331 | 68.113 ± 3.115 | 107.529 ± 13.711 |
| lm_batch_likelihood_sentence_4_4d | 24.743 ± 0.298 | 38.062 ± 1.895 | 36.114 ± 0.703 | **16.268 ± 0.641** | 65.730 ± 25.212 |
| nary_matmul_chain_64 | 0.059 ± 0.005 | **0.046 ± 0.000** | 0.079 ± 0.008 | 0.298 ± 0.021 | 0.119 ± 0.044 |
| str_matrix_chain_multiplication_100 | **13.983 ± 0.198** | 15.384 ± 0.149 | 15.711 ± 0.204 | 15.268 ± 0.354 | 16.169 ± 0.565 |
| str_mps_varying_inner_product_200 | **20.792 ± 0.171** | 21.534 ± 0.379 | 25.605 ± 0.383 | 23.962 ± 0.384 | 25.622 ± 30.978 |
| str_nw_mera_closed_120 | **1334.461 ± 5.498** | 1359.582 ± 13.440 | 1391.770 ± 6.379 | 1891.008 ± 5.874 | 1378.425 ± 29.030 |
| str_nw_mera_open_26 | 899.928 ± 2.364 | 1026.035 ± 9.279 | **874.643 ± 4.452** | 1418.752 ± 6.530 | 1155.310 ± 34.588 |
| tensornetwork_permutation_focus_step409_316 | **300.887 ± 1.339** | 342.490 ± 3.891 | 738.282 ± 2.327 | 1539.693 ± 14.351 | 415.511 ± 45.520 |
| tensornetwork_permutation_light_415 | **292.267 ± 1.808** | 302.128 ± 2.207 | 703.303 ± 3.348 | 1555.094 ± 11.078 | 418.451 ± 87.798 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.840 ± 0.170 | **3.837 ± 0.037** | 4.118 ± 0.057 | 4.251 ± 0.053 | 9.921 ± 6.059 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.787 ± 0.016 | **0.634 ± 0.017** | 0.673 ± 0.030 | 0.847 ± 0.011 | 0.879 ± 1.557 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.200 ± 0.024 | **0.195 ± 0.023** | 0.227 ± 0.058 | 0.597 ± 0.028 | 1.492 ± 0.821 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.419 ± 0.025 | 0.205 ± 0.032 | **0.195 ± 0.018** | 0.619 ± 0.024 | 2.333 ± 2.245 |
| bin_elementwise_mul_2048x2048 | 1.717 ± 0.107 | **1.619 ± 0.064** | 4.400 ± 0.250 | 5.466 ± 0.151 | 5.052 ± 0.366 |
| bin_elementwise_mul_512x512 | **0.102 ± 0.032** | 0.156 ± 0.049 | 0.132 ± 0.010 | 0.401 ± 0.022 | 0.159 ± 0.749 |
| bin_matmul_1024 | 65.292 ± 0.414 | **49.172 ± 0.691** | 74.503 ± 0.131 | 65.378 ± 3.957 | 73.755 ± 15.297 |
| bin_matmul_256 | 1.094 ± 0.026 | **0.847 ± 0.023** | 1.356 ± 0.024 | 1.186 ± 0.010 | 1.265 ± 0.036 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.011 ± 0.000 | **0.006 ± 0.005** | 0.035 ± 0.007 | 0.200 ± 0.027 | 0.013 ± 0.006 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.059 ± 0.001 | **0.043 ± 0.002** | 0.146 ± 0.012 | 0.244 ± 0.025 | 0.085 ± 0.028 |
| bin_omeinsum_matmul_10x10 | 0.026 ± 0.001 | **0.017 ± 0.007** | 0.034 ± 0.006 | 0.148 ± 0.019 | 0.024 ± 0.001 |
| bin_outer_product_4096 | **9.876 ± 0.152** | 9.877 ± 0.198 | 9.970 ± 0.384 | 10.147 ± 0.269 | 952.551 ± 2.252 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 65.437 ± 0.379 | **51.419 ± 0.244** | 60.991 ± 0.391 | 67.353 ± 1.480 | 70.779 ± 5.527 |
| gm_queen5_5_3.wcsp | 1731.788 ± 9.226 | 1591.269 ± 7.626 | **1583.318 ± 12.262** | 1850.201 ± 33.206 | 1778.136 ± 18.135 |
| lm_batch_likelihood_brackets_4_4d | 23.074 ± 0.241 | 32.411 ± 0.870 | 36.424 ± 0.401 | **14.612 ± 0.608** | 44.714 ± 27.149 |
| lm_batch_likelihood_sentence_3_12d | **53.658 ± 0.892** | 99.164 ± 2.444 | 66.015 ± 1.094 | 74.562 ± 1.988 | 114.458 ± 29.141 |
| lm_batch_likelihood_sentence_4_4d | 24.243 ± 0.368 | 39.909 ± 1.521 | 39.461 ± 0.661 | **16.009 ± 0.225** | 46.086 ± 35.876 |
| nary_matmul_chain_64 | 0.059 ± 0.005 | **0.046 ± 0.000** | 0.079 ± 0.008 | 0.298 ± 0.021 | 0.119 ± 0.044 |
| str_matrix_chain_multiplication_100 | 14.779 ± 0.210 | 15.204 ± 0.303 | 15.655 ± 0.191 | **14.756 ± 0.376** | 16.175 ± 0.411 |
| str_mps_varying_inner_product_200 | 23.437 ± 0.139 | **23.142 ± 0.289** | 25.999 ± 0.219 | 24.032 ± 0.328 | 25.107 ± 9.629 |
| str_nw_mera_closed_120 | **1209.917 ± 2.549** | 1211.671 ± 3.767 | 1244.010 ± 6.271 | 1732.717 ± 6.670 | 1272.046 ± 78.167 |
| str_nw_mera_open_26 | 906.361 ± 5.978 | 1019.259 ± 6.400 | **896.612 ± 1.357** | 1493.656 ± 10.623 | 1151.419 ± 87.748 |
| tensornetwork_permutation_focus_step409_316 | **300.887 ± 1.339** | 342.490 ± 3.891 | 738.282 ± 2.327 | 1539.693 ± 14.351 | 415.511 ± 45.520 |
| tensornetwork_permutation_light_415 | **292.267 ± 1.808** | 302.128 ± 2.207 | 703.303 ± 3.348 | 1555.094 ± 11.078 | 418.451 ± 87.798 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261021_183748/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261021_183748/run.yaml`
- Timestamp: `20261021_183748`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261021_183748`.

- tenferro-rs commit: `f1d4d8d961ff6b937768111888ad2a493314892a`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

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

- Julia: version `1.13.2`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `4`

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


- Source table: `data/results/amd-cpu/cpu/einsum/20261021_183748/einsum_table_t4_20261021_183748.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20261021_183748/tenferro_trace_t4_20261021_183748.log`
- `data/results/amd-cpu/cpu/einsum/20261021_183748/tenferro_eager_t4_20261021_183748.log`
- `data/results/amd-cpu/cpu/einsum/20261021_183748/pytorch_cpu_t4_20261021_183748.log`
- `data/results/amd-cpu/cpu/einsum/20261021_183748/jax_cpu_t4_20261021_183748.log`
- `data/results/amd-cpu/cpu/einsum/20261021_183748/julia_omeinsum_t4_20261021_183748.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.359 ± 0.091 | 2.819 ± 0.051 | **2.346 ± 1.649** | 2.970 ± 0.434 | 10.226 ± 6.261 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.888 ± 0.037 | 0.667 ± 0.031 | **0.388 ± 0.017** | 0.810 ± 0.015 | 0.873 ± 1.179 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.279 ± 0.030 | 0.430 ± 0.030 | **0.218 ± 0.036** | 0.473 ± 0.025 | 1.070 ± 1.003 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.372 ± 0.024 | 0.254 ± 0.022 | **0.178 ± 0.028** | 0.477 ± 0.031 | 2.468 ± 3.057 |
| bin_elementwise_mul_2048x2048 | 1.820 ± 0.053 | **1.642 ± 0.385** | 3.185 ± 0.109 | 4.213 ± 0.234 | 5.048 ± 0.342 |
| bin_elementwise_mul_512x512 | 0.167 ± 0.026 | **0.114 ± 0.010** | **0.114 ± 0.009** | 0.408 ± 0.020 | 0.138 ± 0.187 |
| bin_matmul_1024 | 32.262 ± 0.381 | **24.356 ± 0.100** | 55.945 ± 2.548 | 36.563 ± 0.427 | 57.474 ± 4.113 |
| bin_matmul_256 | 0.713 ± 0.046 | **0.455 ± 0.017** | 0.879 ± 0.016 | 0.869 ± 0.060 | 0.938 ± 0.033 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.018 ± 0.000 | **0.009 ± 0.000** | 0.065 ± 0.012 | 0.204 ± 0.021 | 0.012 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.068 ± 0.005 | 0.063 ± 0.000 | 0.223 ± 0.019 | 0.329 ± 0.027 | **0.049 ± 0.004** |
| bin_omeinsum_matmul_10x10 | 0.043 ± 0.001 | **0.018 ± 0.000** | 0.048 ± 0.010 | 0.250 ± 0.061 | 0.025 ± 0.001 |
| bin_outer_product_4096 | 8.711 ± 2.421 | 7.750 ± 0.332 | **5.993 ± 0.147** | 7.011 ± 0.353 | 927.887 ± 1.664 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 32.934 ± 0.833 | **26.121 ± 0.320** | 32.931 ± 0.492 | 38.558 ± 1.248 | 47.011 ± 7.528 |
| gm_queen5_5_3.wcsp | 4149.610 ± 85.956 | 3557.309 ± 52.163 | **1586.223 ± 11.796** | 3678.632 ± 68.065 | 4483.199 ± 169.108 |
| lm_batch_likelihood_brackets_4_4d | 23.898 ± 2.779 | 41.202 ± 1.713 | **16.859 ± 0.612** | 22.119 ± 1.272 | 38.349 ± 8.086 |
| lm_batch_likelihood_sentence_3_12d | **41.710 ± 6.283** | 86.837 ± 13.118 | 42.904 ± 1.899 | 46.757 ± 1.821 | 101.850 ± 36.093 |
| lm_batch_likelihood_sentence_4_4d | 25.050 ± 0.905 | 44.107 ± 2.226 | **17.428 ± 0.141** | 22.715 ± 1.966 | 54.969 ± 28.082 |
| nary_matmul_chain_64 | 0.076 ± 0.004 | **0.049 ± 0.001** | 0.087 ± 0.003 | 0.296 ± 0.023 | 0.152 ± 0.052 |
| str_matrix_chain_multiplication_100 | **8.953 ± 0.401** | 9.365 ± 0.122 | 10.881 ± 0.115 | 15.682 ± 0.802 | 14.780 ± 0.850 |
| str_mps_varying_inner_product_200 | 24.565 ± 0.316 | **24.073 ± 0.515** | 26.143 ± 0.384 | 32.509 ± 0.716 | 33.674 ± 8.849 |
| str_nw_mera_closed_120 | 691.454 ± 19.460 | 693.085 ± 48.225 | **691.050 ± 5.749** | 912.211 ± 5.947 | 837.879 ± 59.552 |
| str_nw_mera_open_26 | 535.939 ± 22.273 | 584.979 ± 30.168 | **446.795 ± 1.893** | 684.631 ± 7.228 | 849.399 ± 25.939 |
| tensornetwork_permutation_focus_step409_316 | 234.234 ± 23.044 | **221.570 ± 18.667** | 265.921 ± 2.838 | 797.379 ± 11.283 | 425.633 ± 28.250 |
| tensornetwork_permutation_light_415 | **199.504 ± 28.124** | 209.595 ± 16.449 | 263.553 ± 2.338 | 810.080 ± 10.804 | 424.463 ± 55.168 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.359 ± 0.091 | 2.819 ± 0.051 | **2.346 ± 1.649** | 2.970 ± 0.434 | 10.226 ± 6.261 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.888 ± 0.037 | 0.667 ± 0.031 | **0.388 ± 0.017** | 0.810 ± 0.015 | 0.873 ± 1.179 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.279 ± 0.030 | 0.430 ± 0.030 | **0.218 ± 0.036** | 0.473 ± 0.025 | 1.070 ± 1.003 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.372 ± 0.024 | 0.254 ± 0.022 | **0.178 ± 0.028** | 0.477 ± 0.031 | 2.468 ± 3.057 |
| bin_elementwise_mul_2048x2048 | 1.820 ± 0.053 | **1.642 ± 0.385** | 3.185 ± 0.109 | 4.213 ± 0.234 | 5.048 ± 0.342 |
| bin_elementwise_mul_512x512 | 0.167 ± 0.026 | **0.114 ± 0.010** | **0.114 ± 0.009** | 0.408 ± 0.020 | 0.138 ± 0.187 |
| bin_matmul_1024 | 32.262 ± 0.381 | **24.356 ± 0.100** | 55.945 ± 2.548 | 36.563 ± 0.427 | 57.474 ± 4.113 |
| bin_matmul_256 | 0.713 ± 0.046 | **0.455 ± 0.017** | 0.879 ± 0.016 | 0.869 ± 0.060 | 0.938 ± 0.033 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.018 ± 0.000 | **0.009 ± 0.000** | 0.065 ± 0.012 | 0.204 ± 0.021 | 0.012 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.068 ± 0.005 | 0.063 ± 0.000 | 0.223 ± 0.019 | 0.329 ± 0.027 | **0.049 ± 0.004** |
| bin_omeinsum_matmul_10x10 | 0.043 ± 0.001 | **0.018 ± 0.000** | 0.048 ± 0.010 | 0.250 ± 0.061 | 0.025 ± 0.001 |
| bin_outer_product_4096 | 8.711 ± 2.421 | 7.750 ± 0.332 | **5.993 ± 0.147** | 7.011 ± 0.353 | 927.887 ± 1.664 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 32.934 ± 0.833 | **26.121 ± 0.320** | 32.931 ± 0.492 | 38.558 ± 1.248 | 47.011 ± 7.528 |
| gm_queen5_5_3.wcsp | 1159.148 ± 82.532 | 1088.647 ± 66.216 | **742.015 ± 5.025** | 1164.434 ± 14.408 | 1537.196 ± 20.163 |
| lm_batch_likelihood_brackets_4_4d | 22.212 ± 4.037 | 40.588 ± 0.758 | **17.784 ± 0.496** | 22.561 ± 1.922 | 60.479 ± 41.541 |
| lm_batch_likelihood_sentence_3_12d | 34.009 ± 4.482 | 85.081 ± 8.038 | **32.777 ± 0.469** | 50.055 ± 1.713 | 98.153 ± 35.149 |
| lm_batch_likelihood_sentence_4_4d | 23.563 ± 3.111 | 44.114 ± 1.890 | **18.330 ± 0.190** | 22.425 ± 0.578 | 61.520 ± 39.133 |
| nary_matmul_chain_64 | 0.076 ± 0.004 | **0.049 ± 0.001** | 0.087 ± 0.003 | 0.296 ± 0.023 | 0.152 ± 0.052 |
| str_matrix_chain_multiplication_100 | **9.094 ± 0.201** | 9.518 ± 0.398 | 10.996 ± 0.101 | 15.639 ± 1.022 | 14.226 ± 0.957 |
| str_mps_varying_inner_product_200 | 28.284 ± 0.814 | 27.243 ± 0.408 | 28.589 ± 0.451 | 32.544 ± 0.978 | **23.989 ± 5.960** |
| str_nw_mera_closed_120 | **620.739 ± 21.685** | 625.031 ± 28.469 | 621.096 ± 6.810 | 840.335 ± 8.610 | 754.893 ± 40.748 |
| str_nw_mera_open_26 | 537.397 ± 20.677 | 584.850 ± 15.844 | **461.987 ± 1.577** | 703.393 ± 6.319 | 840.793 ± 59.506 |
| tensornetwork_permutation_focus_step409_316 | 234.234 ± 23.044 | **221.570 ± 18.667** | 265.921 ± 2.838 | 797.379 ± 11.283 | 425.633 ± 28.250 |
| tensornetwork_permutation_light_415 | **199.504 ± 28.124** | 209.595 ± 16.449 | 263.553 ± 2.338 | 810.080 ± 10.804 | 424.463 ± 55.168 |

# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `9b03dd718025b3350d93eaab13a8b93dcc66cab2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261126_175240/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261126_175240/run.yaml`
- Timestamp: `2026-11-26T17:52:40.572826+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261126_175240`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
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

- tenferro-rs features: `native`

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

- `data/results/amd-cpu/cpu/einsum/20261126_175240/jax_cpu_t1_20261126_175240.log`
- `data/results/amd-cpu/cpu/einsum/20261126_175240/julia_omeinsum_t1_20261126_175240.log`
- `data/results/amd-cpu/cpu/einsum/20261126_175240/pytorch_cpu_t1_20261126_175240.log`
- `data/results/amd-cpu/cpu/einsum/20261126_175240/tenferro_eager_t1_20261126_175240.log`
- `data/results/amd-cpu/cpu/einsum/20261126_175240/tenferro_trace_t1_20261126_175240.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.135 ± 0.444 | **2.914 ± 0.018** | 4.268 ± 0.326 | 4.282 ± 0.098 | 9.577 ± 3.219 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.540 ± 0.028 | **0.406 ± 0.015** | 0.660 ± 0.016 | 0.830 ± 0.015 | 1.493 ± 1.129 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.748 ± 0.015 | 0.743 ± 0.016 | **0.208 ± 0.024** | 0.608 ± 0.034 | 3.935 ± 2.538 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.192 ± 0.008 | 0.760 ± 0.002 | **0.197 ± 0.014** | 0.602 ± 0.026 | 4.954 ± 8.757 |
| bin_elementwise_mul_2048x2048 | 2.070 ± 0.098 | **1.886 ± 0.152** | 16.696 ± 0.552 | 18.315 ± 0.344 | 18.517 ± 4.355 |
| bin_elementwise_mul_512x512 | **0.125 ± 0.015** | 0.134 ± 0.020 | 0.142 ± 0.019 | 0.353 ± 0.016 | 0.169 ± 0.799 |
| bin_matmul_1024 | 128.418 ± 0.535 | **48.513 ± 0.171** | 60.610 ± 0.517 | 63.117 ± 0.737 | 69.369 ± 8.691 |
| bin_matmul_256 | 1.202 ± 0.052 | **0.749 ± 0.006** | 1.358 ± 0.011 | 1.224 ± 0.017 | 1.257 ± 0.438 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.010 ± 0.000 | **0.006 ± 0.000** | 0.037 ± 0.004 | 0.208 ± 0.017 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.033 ± 0.001 | **0.032 ± 0.001** | 0.146 ± 0.011 | 0.280 ± 0.029 | 0.080 ± 0.038 |
| bin_omeinsum_matmul_10x10 | 0.017 ± 0.001 | **0.009 ± 0.009** | 0.034 ± 0.003 | 0.181 ± 0.029 | 0.025 ± 0.001 |
| bin_outer_product_4096 | 65.837 ± 0.794 | 64.493 ± 0.883 | **55.079 ± 0.489** | 58.642 ± 1.762 | 316.002 ± 41.747 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 50.468 ± 0.282 | **49.482 ± 0.366** | 60.825 ± 0.617 | 65.513 ± 1.408 | 68.090 ± 5.583 |
| gm_queen5_5_3.wcsp | **5437.865 ± 36.639** | 5657.860 ± 44.543 | 6321.570 ± 39.681 | 8215.347 ± 42.584 | 9208.367 ± 125.361 |
| lm_batch_likelihood_brackets_4_4d | 23.187 ± 0.691 | 30.623 ± 0.324 | 35.131 ± 1.336 | **14.991 ± 0.336** | 36.309 ± 23.168 |
| lm_batch_likelihood_sentence_3_12d | **53.955 ± 0.662** | 90.814 ± 0.976 | 106.617 ± 1.650 | 98.645 ± 3.425 | 77.446 ± 5.083 |
| lm_batch_likelihood_sentence_4_4d | 24.867 ± 0.228 | 31.988 ± 0.459 | 36.342 ± 0.444 | **17.107 ± 0.268** | 42.260 ± 19.143 |
| nary_matmul_chain_64 | 0.053 ± 0.000 | **0.034 ± 0.015** | 0.078 ± 0.004 | 0.198 ± 0.012 | 0.072 ± 0.002 |
| str_matrix_chain_multiplication_100 | **12.168 ± 0.088** | 13.280 ± 0.095 | 15.730 ± 0.174 | 14.983 ± 0.289 | 16.585 ± 0.225 |
| str_mps_varying_inner_product_200 | 15.193 ± 0.311 | **15.042 ± 0.308** | 25.358 ± 0.608 | 23.928 ± 2.742 | 25.040 ± 1.417 |
| str_nw_mera_closed_120 | **1353.870 ± 26.725** | 1406.706 ± 4.788 | 1973.902 ± 9.261 | 2513.066 ± 9.949 | 1889.344 ± 41.610 |
| str_nw_mera_open_26 | **1085.434 ± 6.610** | 1483.555 ± 7.467 | 1244.877 ± 2.917 | 1959.270 ± 13.963 | 1653.994 ± 13.570 |
| tensornetwork_permutation_focus_step409_316 | **299.424 ± 3.240** | 411.125 ± 5.944 | 913.310 ± 3.028 | 711.858 ± 35.407 | 566.180 ± 32.368 |
| tensornetwork_permutation_light_415 | **294.578 ± 2.337** | 339.713 ± 1.142 | 875.977 ± 2.306 | 734.676 ± 48.237 | 567.588 ± 18.850 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.135 ± 0.444 | **2.914 ± 0.018** | 4.268 ± 0.326 | 4.282 ± 0.098 | 9.577 ± 3.219 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.540 ± 0.028 | **0.406 ± 0.015** | 0.660 ± 0.016 | 0.830 ± 0.015 | 1.493 ± 1.129 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.748 ± 0.015 | 0.743 ± 0.016 | **0.208 ± 0.024** | 0.608 ± 0.034 | 3.935 ± 2.538 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.192 ± 0.008 | 0.760 ± 0.002 | **0.197 ± 0.014** | 0.602 ± 0.026 | 4.954 ± 8.757 |
| bin_elementwise_mul_2048x2048 | 2.070 ± 0.098 | **1.886 ± 0.152** | 16.696 ± 0.552 | 18.315 ± 0.344 | 18.517 ± 4.355 |
| bin_elementwise_mul_512x512 | **0.125 ± 0.015** | 0.134 ± 0.020 | 0.142 ± 0.019 | 0.353 ± 0.016 | 0.169 ± 0.799 |
| bin_matmul_1024 | 128.418 ± 0.535 | **48.513 ± 0.171** | 60.610 ± 0.517 | 63.117 ± 0.737 | 69.369 ± 8.691 |
| bin_matmul_256 | 1.202 ± 0.052 | **0.749 ± 0.006** | 1.358 ± 0.011 | 1.224 ± 0.017 | 1.257 ± 0.438 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.010 ± 0.000 | **0.006 ± 0.000** | 0.037 ± 0.004 | 0.208 ± 0.017 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.033 ± 0.001 | **0.032 ± 0.001** | 0.146 ± 0.011 | 0.280 ± 0.029 | 0.080 ± 0.038 |
| bin_omeinsum_matmul_10x10 | 0.017 ± 0.001 | **0.009 ± 0.009** | 0.034 ± 0.003 | 0.181 ± 0.029 | 0.025 ± 0.001 |
| bin_outer_product_4096 | 65.837 ± 0.794 | 64.493 ± 0.883 | **55.079 ± 0.489** | 58.642 ± 1.762 | 316.002 ± 41.747 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 50.468 ± 0.282 | **49.482 ± 0.366** | 60.825 ± 0.617 | 65.513 ± 1.408 | 68.090 ± 5.583 |
| gm_queen5_5_3.wcsp | **1849.645 ± 22.497** | 2223.492 ± 10.793 | 2460.671 ± 28.143 | 2827.069 ± 22.494 | 3324.321 ± 64.962 |
| lm_batch_likelihood_brackets_4_4d | 24.184 ± 1.193 | 28.837 ± 0.451 | 36.431 ± 1.447 | **15.503 ± 5.011** | 34.821 ± 38.906 |
| lm_batch_likelihood_sentence_3_12d | **55.544 ± 0.941** | 117.341 ± 1.343 | 66.585 ± 0.749 | 73.370 ± 2.569 | 107.456 ± 47.501 |
| lm_batch_likelihood_sentence_4_4d | 22.683 ± 0.326 | 31.796 ± 0.381 | 38.638 ± 0.679 | **16.242 ± 3.130** | 36.145 ± 35.466 |
| nary_matmul_chain_64 | 0.053 ± 0.000 | **0.034 ± 0.015** | 0.078 ± 0.004 | 0.198 ± 0.012 | 0.072 ± 0.002 |
| str_matrix_chain_multiplication_100 | **13.417 ± 0.258** | 13.791 ± 0.346 | 15.917 ± 0.189 | 14.870 ± 0.396 | 16.695 ± 0.342 |
| str_mps_varying_inner_product_200 | 17.021 ± 0.186 | **15.094 ± 0.099** | 25.488 ± 0.246 | 24.304 ± 2.481 | 24.910 ± 9.185 |
| str_nw_mera_closed_120 | 1333.686 ± 8.831 | **1288.506 ± 17.965** | 1532.454 ± 15.798 | 2114.508 ± 26.304 | 1534.107 ± 31.449 |
| str_nw_mera_open_26 | **1088.302 ± 7.900** | 1479.717 ± 9.989 | 1253.716 ± 13.626 | 2011.458 ± 19.385 | 1652.507 ± 14.596 |
| tensornetwork_permutation_focus_step409_316 | **299.424 ± 3.240** | 411.125 ± 5.944 | 913.310 ± 3.028 | 711.858 ± 35.407 | 566.180 ± 32.368 |
| tensornetwork_permutation_light_415 | **294.578 ± 2.337** | 339.713 ± 1.142 | 875.977 ± 2.306 | 734.676 ± 48.237 | 567.588 ± 18.850 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261126_183225/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261126_183225/run.yaml`
- Timestamp: `2026-11-26T18:32:25.628209+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261126_183225`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
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

- tenferro-rs features: `native`

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

- `data/results/amd-cpu/cpu/einsum/20261126_183225/jax_cpu_t4_20261126_183225.log`
- `data/results/amd-cpu/cpu/einsum/20261126_183225/julia_omeinsum_t4_20261126_183225.log`
- `data/results/amd-cpu/cpu/einsum/20261126_183225/pytorch_cpu_t4_20261126_183225.log`
- `data/results/amd-cpu/cpu/einsum/20261126_183225/tenferro_eager_t4_20261126_183225.log`
- `data/results/amd-cpu/cpu/einsum/20261126_183225/tenferro_trace_t4_20261126_183225.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.423 ± 0.046 | **1.731 ± 0.011** | 2.306 ± 0.105 | 2.895 ± 0.605 | 12.887 ± 6.332 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.578 ± 0.015 | 0.458 ± 0.015 | **0.380 ± 0.028** | 0.842 ± 0.028 | 1.702 ± 1.419 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.426 ± 0.014 | 0.433 ± 0.018 | **0.170 ± 0.062** | 0.468 ± 0.036 | 3.869 ± 2.028 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.300 ± 0.013 | 0.425 ± 0.015 | **0.133 ± 0.022** | 0.447 ± 0.032 | 4.910 ± 2.576 |
| bin_elementwise_mul_2048x2048 | 0.978 ± 0.034 | **0.844 ± 0.024** | 5.781 ± 0.282 | 9.566 ± 0.308 | 18.962 ± 0.464 |
| bin_elementwise_mul_512x512 | 0.147 ± 0.030 | **0.141 ± 0.007** | 0.153 ± 0.031 | 0.380 ± 0.028 | 0.164 ± 0.777 |
| bin_matmul_1024 | 44.216 ± 0.994 | **22.886 ± 0.129** | 35.830 ± 0.587 | 36.182 ± 0.412 | 50.801 ± 8.052 |
| bin_matmul_256 | 0.586 ± 0.013 | **0.458 ± 0.012** | 0.885 ± 0.024 | 0.849 ± 0.067 | 0.991 ± 0.377 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.016 ± 0.001 | **0.010 ± 0.000** | 0.050 ± 0.006 | 0.198 ± 0.025 | 0.013 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.049 ± 0.003 | **0.047 ± 0.003** | 0.222 ± 0.017 | 0.254 ± 0.023 | 0.064 ± 0.028 |
| bin_omeinsum_matmul_10x10 | 0.027 ± 0.002 | **0.014 ± 0.001** | 0.047 ± 0.004 | 0.149 ± 0.019 | 0.026 ± 0.001 |
| bin_outer_product_4096 | 22.260 ± 0.366 | 22.461 ± 0.842 | **19.447 ± 0.616** | 29.725 ± 0.463 | 307.011 ± 33.508 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 24.430 ± 0.392 | **23.837 ± 0.150** | 30.682 ± 0.329 | 38.354 ± 1.247 | 42.495 ± 9.347 |
| gm_queen5_5_3.wcsp | 3324.033 ± 42.639 | 3939.264 ± 37.858 | **2698.264 ± 28.509** | 5324.860 ± 86.638 | 8688.795 ± 32.776 |
| lm_batch_likelihood_brackets_4_4d | 17.784 ± 0.389 | 22.568 ± 0.183 | **17.087 ± 0.524** | 23.261 ± 0.576 | 39.548 ± 17.997 |
| lm_batch_likelihood_sentence_3_12d | **30.069 ± 0.301** | 68.563 ± 1.395 | 57.483 ± 1.936 | 50.173 ± 5.669 | 74.148 ± 21.622 |
| lm_batch_likelihood_sentence_4_4d | 18.752 ± 0.392 | 22.706 ± 0.352 | **17.455 ± 0.434** | 23.676 ± 1.151 | 39.734 ± 5.407 |
| nary_matmul_chain_64 | 0.067 ± 0.001 | **0.065 ± 0.001** | 0.095 ± 0.012 | 0.279 ± 0.024 | 0.095 ± 0.074 |
| str_matrix_chain_multiplication_100 | **7.758 ± 0.132** | 9.436 ± 0.275 | 10.832 ± 0.237 | 15.295 ± 0.502 | 14.584 ± 8.272 |
| str_mps_varying_inner_product_200 | 26.787 ± 0.434 | 28.923 ± 0.254 | 25.811 ± 0.497 | 30.814 ± 0.556 | **24.544 ± 2.765** |
| str_nw_mera_closed_120 | **693.575 ± 12.103** | 743.389 ± 8.750 | 947.768 ± 16.011 | 1174.566 ± 29.236 | 1220.727 ± 46.272 |
| str_nw_mera_open_26 | 629.269 ± 14.480 | 725.732 ± 11.671 | **587.003 ± 5.092** | 945.684 ± 23.151 | 1241.965 ± 86.118 |
| tensornetwork_permutation_focus_step409_316 | **188.733 ± 1.657** | 192.164 ± 1.987 | 343.238 ± 8.377 | 401.513 ± 14.892 | 538.425 ± 37.301 |
| tensornetwork_permutation_light_415 | 188.906 ± 1.606 | **154.176 ± 0.827** | 343.757 ± 16.934 | 424.960 ± 38.338 | 548.307 ± 31.595 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.423 ± 0.046 | **1.731 ± 0.011** | 2.306 ± 0.105 | 2.895 ± 0.605 | 12.887 ± 6.332 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.578 ± 0.015 | 0.458 ± 0.015 | **0.380 ± 0.028** | 0.842 ± 0.028 | 1.702 ± 1.419 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.426 ± 0.014 | 0.433 ± 0.018 | **0.170 ± 0.062** | 0.468 ± 0.036 | 3.869 ± 2.028 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.300 ± 0.013 | 0.425 ± 0.015 | **0.133 ± 0.022** | 0.447 ± 0.032 | 4.910 ± 2.576 |
| bin_elementwise_mul_2048x2048 | 0.978 ± 0.034 | **0.844 ± 0.024** | 5.781 ± 0.282 | 9.566 ± 0.308 | 18.962 ± 0.464 |
| bin_elementwise_mul_512x512 | 0.147 ± 0.030 | **0.141 ± 0.007** | 0.153 ± 0.031 | 0.380 ± 0.028 | 0.164 ± 0.777 |
| bin_matmul_1024 | 44.216 ± 0.994 | **22.886 ± 0.129** | 35.830 ± 0.587 | 36.182 ± 0.412 | 50.801 ± 8.052 |
| bin_matmul_256 | 0.586 ± 0.013 | **0.458 ± 0.012** | 0.885 ± 0.024 | 0.849 ± 0.067 | 0.991 ± 0.377 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.016 ± 0.001 | **0.010 ± 0.000** | 0.050 ± 0.006 | 0.198 ± 0.025 | 0.013 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.049 ± 0.003 | **0.047 ± 0.003** | 0.222 ± 0.017 | 0.254 ± 0.023 | 0.064 ± 0.028 |
| bin_omeinsum_matmul_10x10 | 0.027 ± 0.002 | **0.014 ± 0.001** | 0.047 ± 0.004 | 0.149 ± 0.019 | 0.026 ± 0.001 |
| bin_outer_product_4096 | 22.260 ± 0.366 | 22.461 ± 0.842 | **19.447 ± 0.616** | 29.725 ± 0.463 | 307.011 ± 33.508 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 24.430 ± 0.392 | **23.837 ± 0.150** | 30.682 ± 0.329 | 38.354 ± 1.247 | 42.495 ± 9.347 |
| gm_queen5_5_3.wcsp | **916.243 ± 16.220** | 1099.103 ± 12.468 | 1130.408 ± 19.560 | 1855.957 ± 37.462 | 3085.164 ± 59.265 |
| lm_batch_likelihood_brackets_4_4d | 17.777 ± 0.246 | 21.563 ± 0.173 | **17.548 ± 0.407** | 22.787 ± 2.425 | 44.674 ± 1.667 |
| lm_batch_likelihood_sentence_3_12d | **30.819 ± 0.353** | 68.016 ± 0.677 | 32.960 ± 0.403 | 55.546 ± 17.637 | 96.066 ± 68.994 |
| lm_batch_likelihood_sentence_4_4d | **16.340 ± 0.323** | 21.982 ± 0.307 | 18.106 ± 0.220 | 23.255 ± 2.178 | 46.122 ± 17.438 |
| nary_matmul_chain_64 | 0.067 ± 0.001 | **0.065 ± 0.001** | 0.095 ± 0.012 | 0.279 ± 0.024 | 0.095 ± 0.074 |
| str_matrix_chain_multiplication_100 | **8.267 ± 0.130** | 9.808 ± 0.202 | 11.039 ± 0.129 | 15.042 ± 1.041 | 14.348 ± 0.453 |
| str_mps_varying_inner_product_200 | 31.088 ± 0.672 | 30.402 ± 0.683 | **28.942 ± 0.446** | 31.134 ± 0.842 | 29.175 ± 9.573 |
| str_nw_mera_closed_120 | 728.204 ± 8.173 | **671.329 ± 14.874** | 773.423 ± 12.024 | 992.732 ± 21.605 | 937.380 ± 66.133 |
| str_nw_mera_open_26 | 638.588 ± 12.124 | 742.156 ± 13.348 | **596.477 ± 13.081** | 969.798 ± 43.900 | 1238.061 ± 71.012 |
| tensornetwork_permutation_focus_step409_316 | **188.733 ± 1.657** | 192.164 ± 1.987 | 343.238 ± 8.377 | 401.513 ± 14.892 | 538.425 ± 37.301 |
| tensornetwork_permutation_light_415 | 188.906 ± 1.606 | **154.176 ± 0.827** | 343.757 ± 16.934 | 424.960 ± 38.338 | 548.307 ± 31.595 |

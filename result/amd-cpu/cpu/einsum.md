# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `ec30d812749ef2bd6dc72245a56150aa77cb2f9c`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261108_180750/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261108_180750/run.yaml`
- Timestamp: `2026-11-08T18:07:50.799046+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261108_180750`.

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

- `data/results/amd-cpu/cpu/einsum/20261108_180750/jax_cpu_t1_20261108_180750.log`
- `data/results/amd-cpu/cpu/einsum/20261108_180750/julia_omeinsum_t1_20261108_180750.log`
- `data/results/amd-cpu/cpu/einsum/20261108_180750/pytorch_cpu_t1_20261108_180750.log`
- `data/results/amd-cpu/cpu/einsum/20261108_180750/tenferro_eager_t1_20261108_180750.log`
- `data/results/amd-cpu/cpu/einsum/20261108_180750/tenferro_trace_t1_20261108_180750.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.182 ± 0.230 | **3.850 ± 0.025** | 4.248 ± 0.069 | 4.411 ± 0.117 | 11.531 ± 5.941 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.786 ± 0.026 | **0.598 ± 0.012** | 0.682 ± 0.032 | 0.846 ± 0.027 | 2.410 ± 1.479 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.734 ± 0.008 | 0.733 ± 0.011 | **0.272 ± 0.073** | 0.613 ± 0.032 | 3.796 ± 2.035 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.227 ± 0.014 | 0.731 ± 0.007 | **0.247 ± 0.022** | 0.616 ± 0.026 | 4.629 ± 3.640 |
| bin_elementwise_mul_2048x2048 | 2.237 ± 0.107 | **1.911 ± 0.088** | 17.112 ± 0.271 | 17.995 ± 0.438 | 20.144 ± 5.001 |
| bin_elementwise_mul_512x512 | 0.167 ± 0.023 | **0.125 ± 0.008** | 0.157 ± 0.018 | 0.398 ± 0.036 | 0.165 ± 0.184 |
| bin_matmul_1024 | 57.687 ± 0.838 | **48.625 ± 0.758** | 61.396 ± 0.882 | 63.877 ± 1.063 | 74.583 ± 19.601 |
| bin_matmul_256 | 1.059 ± 0.010 | **0.810 ± 0.018** | 1.359 ± 0.020 | 1.156 ± 0.029 | 1.252 ± 0.016 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.007 | **0.009 ± 0.005** | 0.055 ± 0.024 | 0.208 ± 0.013 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.052 ± 0.001 | **0.039 ± 0.002** | 0.141 ± 0.007 | 0.248 ± 0.037 | 0.098 ± 0.023 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.000 | **0.017 ± 0.000** | 0.034 ± 0.002 | 0.148 ± 0.012 | 0.026 ± 0.002 |
| bin_outer_product_4096 | 66.082 ± 1.208 | 64.067 ± 1.405 | **54.802 ± 0.601** | 56.315 ± 0.472 | 316.498 ± 5.988 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 57.120 ± 0.377 | **51.093 ± 0.538** | 62.852 ± 0.812 | 66.676 ± 1.451 | 71.767 ± 15.055 |
| gm_queen5_5_3.wcsp | 7487.387 ± 156.229 | 6871.623 ± 78.590 | **6315.223 ± 51.295** | 8326.906 ± 71.735 | 9395.995 ± 240.291 |
| lm_batch_likelihood_brackets_4_4d | 28.202 ± 0.752 | 36.171 ± 2.066 | 36.150 ± 0.450 | **16.339 ± 0.479** | 32.310 ± 6.960 |
| lm_batch_likelihood_sentence_3_12d | **64.496 ± 2.477** | 123.742 ± 22.541 | 108.938 ± 2.253 | 102.706 ± 2.790 | 119.095 ± 22.376 |
| lm_batch_likelihood_sentence_4_4d | 30.618 ± 1.169 | 38.948 ± 1.877 | 37.550 ± 0.479 | **18.113 ± 0.364** | 66.693 ± 25.345 |
| nary_matmul_chain_64 | **0.059 ± 0.001** | 0.115 ± 0.038 | 0.078 ± 0.006 | 0.210 ± 0.015 | 0.121 ± 0.035 |
| str_matrix_chain_multiplication_100 | **14.790 ± 0.204** | 16.823 ± 0.989 | 16.061 ± 0.192 | 15.738 ± 0.356 | 16.679 ± 0.519 |
| str_mps_varying_inner_product_200 | **20.881 ± 0.165** | 21.725 ± 0.198 | 26.189 ± 0.546 | 24.511 ± 0.542 | 27.026 ± 11.700 |
| str_nw_mera_closed_120 | **1695.265 ± 26.016** | 1870.966 ± 26.573 | 2010.908 ± 10.439 | 2508.904 ± 12.916 | 1905.623 ± 31.491 |
| str_nw_mera_open_26 | 1345.521 ± 19.279 | 1442.692 ± 23.479 | **1235.566 ± 8.900** | 1990.422 ± 14.851 | 1707.051 ± 29.052 |
| tensornetwork_permutation_focus_step409_316 | **484.373 ± 13.242** | 587.800 ± 5.500 | 897.861 ± 11.868 | 990.145 ± 585.480 | 635.016 ± 13.842 |
| tensornetwork_permutation_light_415 | **480.732 ± 6.209** | 535.242 ± 11.646 | 861.025 ± 9.154 | 889.755 ± 121.731 | 641.779 ± 24.193 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.182 ± 0.230 | **3.850 ± 0.025** | 4.248 ± 0.069 | 4.411 ± 0.117 | 11.531 ± 5.941 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.786 ± 0.026 | **0.598 ± 0.012** | 0.682 ± 0.032 | 0.846 ± 0.027 | 2.410 ± 1.479 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.734 ± 0.008 | 0.733 ± 0.011 | **0.272 ± 0.073** | 0.613 ± 0.032 | 3.796 ± 2.035 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.227 ± 0.014 | 0.731 ± 0.007 | **0.247 ± 0.022** | 0.616 ± 0.026 | 4.629 ± 3.640 |
| bin_elementwise_mul_2048x2048 | 2.237 ± 0.107 | **1.911 ± 0.088** | 17.112 ± 0.271 | 17.995 ± 0.438 | 20.144 ± 5.001 |
| bin_elementwise_mul_512x512 | 0.167 ± 0.023 | **0.125 ± 0.008** | 0.157 ± 0.018 | 0.398 ± 0.036 | 0.165 ± 0.184 |
| bin_matmul_1024 | 57.687 ± 0.838 | **48.625 ± 0.758** | 61.396 ± 0.882 | 63.877 ± 1.063 | 74.583 ± 19.601 |
| bin_matmul_256 | 1.059 ± 0.010 | **0.810 ± 0.018** | 1.359 ± 0.020 | 1.156 ± 0.029 | 1.252 ± 0.016 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.007 | **0.009 ± 0.005** | 0.055 ± 0.024 | 0.208 ± 0.013 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.052 ± 0.001 | **0.039 ± 0.002** | 0.141 ± 0.007 | 0.248 ± 0.037 | 0.098 ± 0.023 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.000 | **0.017 ± 0.000** | 0.034 ± 0.002 | 0.148 ± 0.012 | 0.026 ± 0.002 |
| bin_outer_product_4096 | 66.082 ± 1.208 | 64.067 ± 1.405 | **54.802 ± 0.601** | 56.315 ± 0.472 | 316.498 ± 5.988 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 57.120 ± 0.377 | **51.093 ± 0.538** | 62.852 ± 0.812 | 66.676 ± 1.451 | 71.767 ± 15.055 |
| gm_queen5_5_3.wcsp | **2486.860 ± 76.216** | 2705.946 ± 50.111 | 2600.568 ± 89.608 | 2858.763 ± 24.943 | 3405.182 ± 40.585 |
| lm_batch_likelihood_brackets_4_4d | 28.261 ± 0.608 | 34.317 ± 1.876 | 37.120 ± 0.786 | **16.120 ± 0.631** | 70.330 ± 29.473 |
| lm_batch_likelihood_sentence_3_12d | **60.456 ± 2.168** | 169.044 ± 3.994 | 70.431 ± 2.021 | 128.427 ± 3.747 | 121.463 ± 28.065 |
| lm_batch_likelihood_sentence_4_4d | 29.774 ± 0.544 | 40.759 ± 1.861 | 39.333 ± 0.423 | **16.472 ± 0.461** | 47.933 ± 37.540 |
| nary_matmul_chain_64 | **0.059 ± 0.001** | 0.115 ± 0.038 | 0.078 ± 0.006 | 0.210 ± 0.015 | 0.121 ± 0.035 |
| str_matrix_chain_multiplication_100 | 15.570 ± 0.268 | 16.668 ± 1.464 | 16.247 ± 0.529 | **15.168 ± 0.172** | 16.976 ± 0.884 |
| str_mps_varying_inner_product_200 | 24.622 ± 0.500 | **23.526 ± 0.471** | 26.131 ± 1.045 | 24.530 ± 0.447 | 28.381 ± 14.021 |
| str_nw_mera_closed_120 | 1545.680 ± 12.611 | **1527.428 ± 20.008** | 1587.163 ± 14.777 | 2091.665 ± 28.165 | 1543.953 ± 52.665 |
| str_nw_mera_open_26 | 1341.681 ± 16.354 | 1506.441 ± 25.458 | **1266.650 ± 12.798** | 2037.137 ± 23.742 | 1686.767 ± 57.050 |
| tensornetwork_permutation_focus_step409_316 | **484.373 ± 13.242** | 587.800 ± 5.500 | 897.861 ± 11.868 | 990.145 ± 585.480 | 635.016 ± 13.842 |
| tensornetwork_permutation_light_415 | **480.732 ± 6.209** | 535.242 ± 11.646 | 861.025 ± 9.154 | 889.755 ± 121.731 | 641.779 ± 24.193 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261108_184953/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261108_184953/run.yaml`
- Timestamp: `2026-11-08T18:49:53.361000+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261108_184953`.

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

- `data/results/amd-cpu/cpu/einsum/20261108_184953/jax_cpu_t4_20261108_184953.log`
- `data/results/amd-cpu/cpu/einsum/20261108_184953/julia_omeinsum_t4_20261108_184953.log`
- `data/results/amd-cpu/cpu/einsum/20261108_184953/pytorch_cpu_t4_20261108_184953.log`
- `data/results/amd-cpu/cpu/einsum/20261108_184953/tenferro_eager_t4_20261108_184953.log`
- `data/results/amd-cpu/cpu/einsum/20261108_184953/tenferro_trace_t4_20261108_184953.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 9.771 ± 0.140 | 5.282 ± 0.138 | **2.293 ± 2.789** | 2.802 ± 0.654 | 12.594 ± 13.340 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.964 ± 0.017 | 1.370 ± 0.024 | **0.399 ± 0.017** | 0.831 ± 0.012 | 0.978 ± 0.383 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.309 ± 0.015 | 0.305 ± 0.025 | **0.205 ± 0.014** | 0.459 ± 0.051 | 0.849 ± 3.180 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.287 ± 0.014 | 0.322 ± 0.031 | **0.210 ± 0.028** | 0.448 ± 0.019 | 8.178 ± 3.352 |
| bin_elementwise_mul_2048x2048 | 1.008 ± 0.019 | **0.880 ± 0.028** | 5.769 ± 0.143 | 9.429 ± 0.307 | 18.925 ± 0.355 |
| bin_elementwise_mul_512x512 | **0.136 ± 0.002** | 0.139 ± 0.021 | 0.167 ± 0.012 | 0.386 ± 0.029 | 0.216 ± 0.025 |
| bin_matmul_1024 | 60.843 ± 0.175 | 51.647 ± 0.608 | **35.226 ± 0.130** | 35.895 ± 0.362 | 55.639 ± 22.517 |
| bin_matmul_256 | 1.286 ± 0.012 | 0.922 ± 0.165 | 0.852 ± 0.013 | **0.842 ± 0.049** | 0.971 ± 0.018 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.000 | **0.009 ± 0.000** | 0.050 ± 0.004 | 0.210 ± 0.016 | 0.013 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.066 ± 0.004 | **0.061 ± 0.003** | 0.218 ± 0.018 | 0.261 ± 0.018 | 0.063 ± 0.034 |
| bin_omeinsum_matmul_10x10 | 0.069 ± 0.002 | 0.046 ± 0.003 | 0.048 ± 0.017 | 0.147 ± 0.023 | **0.026 ± 0.001** |
| bin_outer_product_4096 | 23.463 ± 0.358 | 23.410 ± 0.577 | **18.744 ± 0.518** | 28.940 ± 0.278 | 299.098 ± 5.271 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 60.953 ± 0.647 | 54.715 ± 1.277 | **30.505 ± 0.415** | 38.083 ± 1.183 | 47.326 ± 8.279 |
| gm_queen5_5_3.wcsp | 6623.771 ± 88.219 | 5875.650 ± 39.126 | **2625.763 ± 31.866** | 5287.935 ± 95.855 | 8657.719 ± 197.360 |
| lm_batch_likelihood_brackets_4_4d | 29.865 ± 0.511 | 48.644 ± 1.964 | **16.757 ± 0.429** | 23.325 ± 1.789 | 75.195 ± 44.891 |
| lm_batch_likelihood_sentence_3_12d | 62.280 ± 0.573 | 199.991 ± 3.367 | 58.434 ± 1.940 | **51.842 ± 10.286** | 115.418 ± 26.357 |
| lm_batch_likelihood_sentence_4_4d | 32.367 ± 0.500 | 54.883 ± 3.100 | **17.520 ± 0.687** | 23.869 ± 1.956 | 55.022 ± 32.194 |
| nary_matmul_chain_64 | 0.147 ± 0.003 | 0.136 ± 0.009 | **0.085 ± 0.016** | 0.276 ± 0.017 | 0.139 ± 0.036 |
| str_matrix_chain_multiplication_100 | 17.278 ± 0.068 | 17.558 ± 0.194 | **10.894 ± 0.473** | 16.194 ± 0.696 | 14.989 ± 0.732 |
| str_mps_varying_inner_product_200 | 26.858 ± 0.278 | 28.600 ± 0.573 | **25.998 ± 0.243** | 31.032 ± 0.981 | 31.141 ± 4.561 |
| str_nw_mera_closed_120 | 1691.010 ± 22.642 | 1868.775 ± 46.881 | **957.313 ± 6.153** | 1165.657 ± 21.264 | 1258.227 ± 41.958 |
| str_nw_mera_open_26 | 1347.725 ± 23.030 | 1295.599 ± 9.848 | **585.017 ± 3.595** | 946.936 ± 17.736 | 1284.672 ± 46.221 |
| tensornetwork_permutation_focus_step409_316 | 485.541 ± 6.457 | 421.168 ± 4.571 | **347.109 ± 3.030** | 415.960 ± 22.104 | 605.655 ± 23.027 |
| tensornetwork_permutation_light_415 | 476.004 ± 4.067 | 471.716 ± 11.544 | **344.724 ± 4.214** | 440.920 ± 42.400 | 587.357 ± 44.251 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 9.771 ± 0.140 | 5.282 ± 0.138 | **2.293 ± 2.789** | 2.802 ± 0.654 | 12.594 ± 13.340 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.964 ± 0.017 | 1.370 ± 0.024 | **0.399 ± 0.017** | 0.831 ± 0.012 | 0.978 ± 0.383 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.309 ± 0.015 | 0.305 ± 0.025 | **0.205 ± 0.014** | 0.459 ± 0.051 | 0.849 ± 3.180 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.287 ± 0.014 | 0.322 ± 0.031 | **0.210 ± 0.028** | 0.448 ± 0.019 | 8.178 ± 3.352 |
| bin_elementwise_mul_2048x2048 | 1.008 ± 0.019 | **0.880 ± 0.028** | 5.769 ± 0.143 | 9.429 ± 0.307 | 18.925 ± 0.355 |
| bin_elementwise_mul_512x512 | **0.136 ± 0.002** | 0.139 ± 0.021 | 0.167 ± 0.012 | 0.386 ± 0.029 | 0.216 ± 0.025 |
| bin_matmul_1024 | 60.843 ± 0.175 | 51.647 ± 0.608 | **35.226 ± 0.130** | 35.895 ± 0.362 | 55.639 ± 22.517 |
| bin_matmul_256 | 1.286 ± 0.012 | 0.922 ± 0.165 | 0.852 ± 0.013 | **0.842 ± 0.049** | 0.971 ± 0.018 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.000 | **0.009 ± 0.000** | 0.050 ± 0.004 | 0.210 ± 0.016 | 0.013 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.066 ± 0.004 | **0.061 ± 0.003** | 0.218 ± 0.018 | 0.261 ± 0.018 | 0.063 ± 0.034 |
| bin_omeinsum_matmul_10x10 | 0.069 ± 0.002 | 0.046 ± 0.003 | 0.048 ± 0.017 | 0.147 ± 0.023 | **0.026 ± 0.001** |
| bin_outer_product_4096 | 23.463 ± 0.358 | 23.410 ± 0.577 | **18.744 ± 0.518** | 28.940 ± 0.278 | 299.098 ± 5.271 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 60.953 ± 0.647 | 54.715 ± 1.277 | **30.505 ± 0.415** | 38.083 ± 1.183 | 47.326 ± 8.279 |
| gm_queen5_5_3.wcsp | 2406.731 ± 35.989 | 2100.308 ± 23.155 | **1108.717 ± 28.172** | 1905.092 ± 33.999 | 3105.670 ± 43.248 |
| lm_batch_likelihood_brackets_4_4d | 30.047 ± 0.956 | 50.347 ± 3.473 | **17.729 ± 0.389** | 23.569 ± 1.776 | 76.788 ± 35.320 |
| lm_batch_likelihood_sentence_3_12d | 58.761 ± 0.892 | 217.665 ± 3.221 | **33.240 ± 1.190** | 61.925 ± 6.837 | 112.185 ± 35.155 |
| lm_batch_likelihood_sentence_4_4d | 31.419 ± 0.644 | 58.388 ± 2.644 | **18.244 ± 0.528** | 23.527 ± 1.145 | 60.203 ± 37.752 |
| nary_matmul_chain_64 | 0.147 ± 0.003 | 0.136 ± 0.009 | **0.085 ± 0.016** | 0.276 ± 0.017 | 0.139 ± 0.036 |
| str_matrix_chain_multiplication_100 | 17.197 ± 0.122 | 19.133 ± 0.690 | **11.261 ± 0.565** | 15.861 ± 0.796 | 15.693 ± 4.649 |
| str_mps_varying_inner_product_200 | 29.006 ± 0.463 | 31.298 ± 1.348 | 28.883 ± 1.143 | 31.527 ± 0.881 | **24.938 ± 8.157** |
| str_nw_mera_closed_120 | 1542.198 ± 7.657 | 1549.454 ± 27.217 | **771.461 ± 11.826** | 984.993 ± 33.457 | 978.803 ± 83.355 |
| str_nw_mera_open_26 | 1357.401 ± 10.617 | 1346.894 ± 9.882 | **597.698 ± 4.278** | 961.270 ± 18.881 | 1271.863 ± 47.490 |
| tensornetwork_permutation_focus_step409_316 | 485.541 ± 6.457 | 421.168 ± 4.571 | **347.109 ± 3.030** | 415.960 ± 22.104 | 605.655 ± 23.027 |
| tensornetwork_permutation_light_415 | 476.004 ± 4.067 | 471.716 ± 11.544 | **344.724 ± 4.214** | 440.920 ± 42.400 | 587.357 ± 44.251 |

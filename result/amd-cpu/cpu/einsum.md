# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`
- Benchmark commit: `8eb4f087ca7cfae8e8f724cdb0b344ad5123e372`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261106_180658/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261106_180658/run.yaml`
- Timestamp: `2026-11-06T18:06:58.553919+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261106_180658`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

### CPU Information

- Model: `INTEL(R) XEON(R) PLATINUM 8573C`
- Vendor: `GenuineIntel`
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

- `data/results/amd-cpu/cpu/einsum/20261106_180658/jax_cpu_t1_20261106_180658.log`
- `data/results/amd-cpu/cpu/einsum/20261106_180658/julia_omeinsum_t1_20261106_180658.log`
- `data/results/amd-cpu/cpu/einsum/20261106_180658/pytorch_cpu_t1_20261106_180658.log`
- `data/results/amd-cpu/cpu/einsum/20261106_180658/tenferro_eager_t1_20261106_180658.log`
- `data/results/amd-cpu/cpu/einsum/20261106_180658/tenferro_trace_t1_20261106_180658.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.463 ± 0.201 | **1.508 ± 0.054** | 1.664 ± 0.021 | 4.774 ± 0.107 | 9.732 ± 9.764 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.449 ± 0.024 | **0.187 ± 0.005** | 0.257 ± 0.028 | 0.948 ± 0.040 | 0.765 ± 1.427 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.520 ± 0.010 | 0.517 ± 0.010 | **0.359 ± 0.015** | 0.601 ± 0.060 | 3.163 ± 1.797 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.105 ± 0.011 | 0.517 ± 0.013 | **0.359 ± 0.013** | 0.604 ± 0.020 | 7.233 ± 8.571 |
| bin_elementwise_mul_2048x2048 | 1.509 ± 0.086 | **1.391 ± 0.052** | 12.909 ± 0.216 | 18.848 ± 0.683 | 14.462 ± 0.586 |
| bin_elementwise_mul_512x512 | 0.231 ± 0.020 | **0.225 ± 0.016** | 0.244 ± 0.016 | 0.653 ± 0.055 | 0.335 ± 0.027 |
| bin_matmul_1024 | 28.015 ± 0.663 | 24.057 ± 0.547 | **24.044 ± 0.500** | 67.722 ± 0.617 | 52.042 ± 26.278 |
| bin_matmul_256 | 0.538 ± 0.018 | **0.413 ± 0.017** | 0.458 ± 0.015 | 1.220 ± 0.029 | 0.869 ± 0.025 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.000 | **0.008 ± 0.003** | 0.024 ± 0.002 | 0.145 ± 0.018 | 0.015 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.031 ± 0.002 | **0.028 ± 0.001** | 0.100 ± 0.005 | 0.209 ± 0.041 | 0.072 ± 0.026 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.001 | **0.012 ± 0.000** | 0.020 ± 0.001 | 0.097 ± 0.013 | 0.024 ± 0.001 |
| bin_outer_product_4096 | 56.859 ± 0.410 | 56.018 ± 0.364 | **47.318 ± 0.197** | 50.105 ± 0.242 | 322.645 ± 41.029 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 36.290 ± 0.805 | **25.358 ± 0.623** | 35.467 ± 0.416 | 72.925 ± 1.757 | 41.841 ± 11.743 |
| gm_queen5_5_3.wcsp | 7924.420 ± 30.073 | 5928.743 ± 26.966 | **4849.079 ± 55.907** | 5223.083 ± 20.687 | 7568.025 ± 143.681 |
| lm_batch_likelihood_brackets_4_4d | 18.269 ± 0.490 | 33.101 ± 1.523 | 24.724 ± 0.590 | **17.812 ± 0.708** | 30.770 ± 14.542 |
| lm_batch_likelihood_sentence_3_12d | 34.256 ± 0.527 | 129.837 ± 2.490 | **29.890 ± 7.818** | 93.727 ± 1.293 | 105.612 ± 22.391 |
| lm_batch_likelihood_sentence_4_4d | **19.934 ± 0.200** | 39.919 ± 1.271 | 26.675 ± 0.397 | 20.109 ± 0.312 | 67.098 ± 26.467 |
| nary_matmul_chain_64 | 0.032 ± 0.016 | **0.020 ± 0.017** | 0.044 ± 0.008 | 0.161 ± 0.009 | 0.095 ± 0.041 |
| str_matrix_chain_multiplication_100 | 9.473 ± 0.400 | **7.014 ± 0.461** | 7.295 ± 0.074 | 14.916 ± 0.662 | 13.111 ± 1.538 |
| str_mps_varying_inner_product_200 | 10.937 ± 0.830 | 11.975 ± 0.184 | **9.413 ± 0.245** | 24.140 ± 0.519 | 21.240 ± 35.700 |
| str_nw_mera_closed_120 | **1010.876 ± 10.198** | 1172.827 ± 11.428 | 1191.915 ± 9.312 | 2386.320 ± 10.616 | 1452.139 ± 40.460 |
| str_nw_mera_open_26 | 979.135 ± 4.526 | 1091.468 ± 3.866 | **788.784 ± 5.263** | 1910.039 ± 12.383 | 1275.846 ± 53.647 |
| tensornetwork_permutation_focus_step409_316 | **360.291 ± 3.695** | 488.179 ± 4.232 | 553.122 ± 2.007 | 704.916 ± 11.819 | 531.904 ± 13.886 |
| tensornetwork_permutation_light_415 | **364.771 ± 2.030** | 387.834 ± 2.996 | 522.054 ± 2.070 | 725.190 ± 10.726 | 526.554 ± 40.893 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.463 ± 0.201 | **1.508 ± 0.054** | 1.664 ± 0.021 | 4.774 ± 0.107 | 9.732 ± 9.764 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.449 ± 0.024 | **0.187 ± 0.005** | 0.257 ± 0.028 | 0.948 ± 0.040 | 0.765 ± 1.427 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.520 ± 0.010 | 0.517 ± 0.010 | **0.359 ± 0.015** | 0.601 ± 0.060 | 3.163 ± 1.797 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.105 ± 0.011 | 0.517 ± 0.013 | **0.359 ± 0.013** | 0.604 ± 0.020 | 7.233 ± 8.571 |
| bin_elementwise_mul_2048x2048 | 1.509 ± 0.086 | **1.391 ± 0.052** | 12.909 ± 0.216 | 18.848 ± 0.683 | 14.462 ± 0.586 |
| bin_elementwise_mul_512x512 | 0.231 ± 0.020 | **0.225 ± 0.016** | 0.244 ± 0.016 | 0.653 ± 0.055 | 0.335 ± 0.027 |
| bin_matmul_1024 | 28.015 ± 0.663 | 24.057 ± 0.547 | **24.044 ± 0.500** | 67.722 ± 0.617 | 52.042 ± 26.278 |
| bin_matmul_256 | 0.538 ± 0.018 | **0.413 ± 0.017** | 0.458 ± 0.015 | 1.220 ± 0.029 | 0.869 ± 0.025 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.000 | **0.008 ± 0.003** | 0.024 ± 0.002 | 0.145 ± 0.018 | 0.015 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.031 ± 0.002 | **0.028 ± 0.001** | 0.100 ± 0.005 | 0.209 ± 0.041 | 0.072 ± 0.026 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.001 | **0.012 ± 0.000** | 0.020 ± 0.001 | 0.097 ± 0.013 | 0.024 ± 0.001 |
| bin_outer_product_4096 | 56.859 ± 0.410 | 56.018 ± 0.364 | **47.318 ± 0.197** | 50.105 ± 0.242 | 322.645 ± 41.029 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 36.290 ± 0.805 | **25.358 ± 0.623** | 35.467 ± 0.416 | 72.925 ± 1.757 | 41.841 ± 11.743 |
| gm_queen5_5_3.wcsp | 2261.214 ± 8.394 | 2217.697 ± 9.422 | **2123.676 ± 10.416** | 2190.260 ± 4.984 | 2735.812 ± 13.259 |
| lm_batch_likelihood_brackets_4_4d | 22.122 ± 0.332 | 35.768 ± 1.361 | 23.120 ± 0.502 | **16.943 ± 0.550** | 69.261 ± 27.330 |
| lm_batch_likelihood_sentence_3_12d | 40.860 ± 1.066 | 142.114 ± 1.068 | **35.548 ± 0.440** | 75.633 ± 1.360 | 108.224 ± 32.137 |
| lm_batch_likelihood_sentence_4_4d | 19.772 ± 0.237 | 41.858 ± 1.025 | 23.939 ± 0.170 | **18.618 ± 0.403** | 44.235 ± 40.571 |
| nary_matmul_chain_64 | 0.032 ± 0.016 | **0.020 ± 0.017** | 0.044 ± 0.008 | 0.161 ± 0.009 | 0.095 ± 0.041 |
| str_matrix_chain_multiplication_100 | **5.928 ± 0.175** | 7.451 ± 0.225 | 7.248 ± 0.425 | 14.185 ± 0.256 | 12.718 ± 1.053 |
| str_mps_varying_inner_product_200 | 11.568 ± 0.376 | 12.271 ± 0.230 | **10.137 ± 0.348** | 21.827 ± 0.677 | 18.387 ± 12.470 |
| str_nw_mera_closed_120 | 943.245 ± 8.059 | 956.280 ± 16.081 | **916.751 ± 10.276** | 2045.815 ± 9.369 | 1152.022 ± 33.509 |
| str_nw_mera_open_26 | 970.778 ± 9.945 | 1122.436 ± 8.337 | **798.035 ± 7.079** | 1953.362 ± 10.398 | 1249.398 ± 29.791 |
| tensornetwork_permutation_focus_step409_316 | **360.291 ± 3.695** | 488.179 ± 4.232 | 553.122 ± 2.007 | 704.916 ± 11.819 | 531.904 ± 13.886 |
| tensornetwork_permutation_light_415 | **364.771 ± 2.030** | 387.834 ± 2.996 | 522.054 ± 2.070 | 725.190 ± 10.726 | 526.554 ± 40.893 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261106_184117/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261106_184117/run.yaml`
- Timestamp: `2026-11-06T18:41:17.633321+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261106_184117`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

### CPU Information

- Model: `INTEL(R) XEON(R) PLATINUM 8573C`
- Vendor: `GenuineIntel`
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

- `data/results/amd-cpu/cpu/einsum/20261106_184117/jax_cpu_t4_20261106_184117.log`
- `data/results/amd-cpu/cpu/einsum/20261106_184117/julia_omeinsum_t4_20261106_184117.log`
- `data/results/amd-cpu/cpu/einsum/20261106_184117/pytorch_cpu_t4_20261106_184117.log`
- `data/results/amd-cpu/cpu/einsum/20261106_184117/tenferro_eager_t4_20261106_184117.log`
- `data/results/amd-cpu/cpu/einsum/20261106_184117/tenferro_trace_t4_20261106_184117.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 8.999 ± 4.973 | 1.033 ± 0.054 | **0.857 ± 0.031** | 3.118 ± 0.620 | 11.522 ± 13.682 |
| bin_batched_matmul_b32_m64_n64_k64 | 8.997 ± 0.012 | 0.267 ± 0.013 | **0.132 ± 0.010** | 0.821 ± 0.019 | 0.738 ± 1.110 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 3.002 ± 0.132 | 5.997 ± 0.158 | **0.208 ± 0.014** | 0.516 ± 0.033 | 3.342 ± 1.717 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.999 ± 0.003 | 3.091 ± 3.196 | **0.200 ± 0.014** | 0.502 ± 0.033 | 7.103 ± 5.472 |
| bin_elementwise_mul_2048x2048 | 3.002 ± 5.303 | **0.662 ± 0.013** | 5.310 ± 0.241 | 10.330 ± 0.290 | 15.731 ± 0.405 |
| bin_elementwise_mul_512x512 | **0.122 ± 0.006** | **0.122 ± 0.010** | 0.144 ± 0.013 | 0.473 ± 0.024 | 0.344 ± 0.039 |
| bin_matmul_1024 | 21.999 ± 1.002 | 12.373 ± 0.243 | **12.121 ± 0.240** | 33.863 ± 0.991 | 39.224 ± 9.217 |
| bin_matmul_256 | 9.000 ± 0.993 | **0.225 ± 0.033** | 0.243 ± 0.020 | 0.824 ± 0.161 | 0.724 ± 0.068 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.010 ± 0.000** | 0.030 ± 0.001 | 0.146 ± 0.022 | 0.015 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.042 ± 0.001 | **0.036 ± 0.001** | 0.136 ± 0.007 | 0.288 ± 0.063 | 0.056 ± 0.030 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.001 | **0.010 ± 0.000** | 0.025 ± 0.002 | 0.101 ± 0.027 | 0.027 ± 0.009 |
| bin_outer_product_4096 | 19.683 ± 0.296 | 20.119 ± 0.220 | **17.410 ± 0.246** | 27.479 ± 0.512 | 307.558 ± 1.372 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 21.268 ± 0.953 | **13.898 ± 0.235** | 16.018 ± 0.540 | 38.653 ± 2.225 | 31.390 ± 17.259 |
| gm_queen5_5_3.wcsp | 5632.811 ± 58.335 | 4468.397 ± 37.431 | **2068.750 ± 16.354** | 3449.701 ± 27.143 | 7023.613 ± 91.617 |
| lm_batch_likelihood_brackets_4_4d | 83.982 ± 10.966 | 104.878 ± 48.722 | **11.558 ± 0.188** | 24.685 ± 1.346 | 60.519 ± 24.801 |
| lm_batch_likelihood_sentence_3_12d | 72.683 ± 26.681 | 136.597 ± 14.799 | **15.362 ± 0.497** | 57.682 ± 11.926 | 104.968 ± 26.841 |
| lm_batch_likelihood_sentence_4_4d | 84.006 ± 43.052 | 128.576 ± 169.358 | **12.281 ± 0.127** | 25.630 ± 1.910 | 45.817 ± 43.873 |
| nary_matmul_chain_64 | 0.046 ± 0.003 | **0.032 ± 0.006** | 0.047 ± 0.002 | 0.154 ± 0.003 | 0.117 ± 0.032 |
| str_matrix_chain_multiplication_100 | **3.946 ± 0.104** | 3.978 ± 0.156 | 4.869 ± 0.176 | 14.988 ± 1.168 | 13.607 ± 1.077 |
| str_mps_varying_inner_product_200 | 16.259 ± 0.343 | 15.310 ± 0.600 | **10.584 ± 0.200** | 28.658 ± 0.805 | 24.541 ± 22.257 |
| str_nw_mera_closed_120 | 758.129 ± 5.260 | 857.445 ± 18.681 | **619.383 ± 4.953** | 1098.024 ± 14.403 | 991.009 ± 60.491 |
| str_nw_mera_open_26 | 813.006 ± 7.819 | 731.019 ± 60.436 | **383.739 ± 5.028** | 886.714 ± 14.437 | 1036.797 ± 15.711 |
| tensornetwork_permutation_focus_step409_316 | 396.294 ± 1.894 | 462.079 ± 16.817 | **255.797 ± 2.357** | 383.433 ± 10.167 | 510.906 ± 61.837 |
| tensornetwork_permutation_light_415 | 402.644 ± 3.805 | 368.180 ± 3.989 | **249.107 ± 1.043** | 396.271 ± 10.704 | 515.895 ± 33.436 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 8.999 ± 4.973 | 1.033 ± 0.054 | **0.857 ± 0.031** | 3.118 ± 0.620 | 11.522 ± 13.682 |
| bin_batched_matmul_b32_m64_n64_k64 | 8.997 ± 0.012 | 0.267 ± 0.013 | **0.132 ± 0.010** | 0.821 ± 0.019 | 0.738 ± 1.110 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 3.002 ± 0.132 | 5.997 ± 0.158 | **0.208 ± 0.014** | 0.516 ± 0.033 | 3.342 ± 1.717 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.999 ± 0.003 | 3.091 ± 3.196 | **0.200 ± 0.014** | 0.502 ± 0.033 | 7.103 ± 5.472 |
| bin_elementwise_mul_2048x2048 | 3.002 ± 5.303 | **0.662 ± 0.013** | 5.310 ± 0.241 | 10.330 ± 0.290 | 15.731 ± 0.405 |
| bin_elementwise_mul_512x512 | **0.122 ± 0.006** | **0.122 ± 0.010** | 0.144 ± 0.013 | 0.473 ± 0.024 | 0.344 ± 0.039 |
| bin_matmul_1024 | 21.999 ± 1.002 | 12.373 ± 0.243 | **12.121 ± 0.240** | 33.863 ± 0.991 | 39.224 ± 9.217 |
| bin_matmul_256 | 9.000 ± 0.993 | **0.225 ± 0.033** | 0.243 ± 0.020 | 0.824 ± 0.161 | 0.724 ± 0.068 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.010 ± 0.000** | 0.030 ± 0.001 | 0.146 ± 0.022 | 0.015 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.042 ± 0.001 | **0.036 ± 0.001** | 0.136 ± 0.007 | 0.288 ± 0.063 | 0.056 ± 0.030 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.001 | **0.010 ± 0.000** | 0.025 ± 0.002 | 0.101 ± 0.027 | 0.027 ± 0.009 |
| bin_outer_product_4096 | 19.683 ± 0.296 | 20.119 ± 0.220 | **17.410 ± 0.246** | 27.479 ± 0.512 | 307.558 ± 1.372 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 21.268 ± 0.953 | **13.898 ± 0.235** | 16.018 ± 0.540 | 38.653 ± 2.225 | 31.390 ± 17.259 |
| gm_queen5_5_3.wcsp | 1922.832 ± 5.609 | 1579.017 ± 39.682 | **879.228 ± 9.477** | 1429.119 ± 20.608 | 2593.443 ± 16.965 |
| lm_batch_likelihood_brackets_4_4d | 88.023 ± 39.997 | 118.345 ± 55.749 | **11.947 ± 0.122** | 25.350 ± 2.645 | 47.938 ± 58.580 |
| lm_batch_likelihood_sentence_3_12d | 75.627 ± 24.289 | 148.033 ± 2.134 | **16.472 ± 0.367** | 63.248 ± 6.118 | 105.216 ± 32.021 |
| lm_batch_likelihood_sentence_4_4d | 91.136 ± 16.980 | 117.896 ± 27.924 | **11.907 ± 0.283** | 24.545 ± 1.653 | 47.460 ± 40.304 |
| nary_matmul_chain_64 | 0.046 ± 0.003 | **0.032 ± 0.006** | 0.047 ± 0.002 | 0.154 ± 0.003 | 0.117 ± 0.032 |
| str_matrix_chain_multiplication_100 | **4.082 ± 0.239** | 4.115 ± 0.066 | 5.366 ± 0.426 | 14.661 ± 0.782 | 15.826 ± 0.921 |
| str_mps_varying_inner_product_200 | 16.978 ± 0.441 | 14.950 ± 0.634 | **12.030 ± 0.378** | 29.714 ± 1.373 | 19.508 ± 17.358 |
| str_nw_mera_closed_120 | 717.884 ± 5.112 | 692.351 ± 13.090 | **502.437 ± 4.434** | 940.058 ± 17.469 | 775.800 ± 55.975 |
| str_nw_mera_open_26 | 808.297 ± 4.579 | 707.172 ± 36.889 | **387.075 ± 6.539** | 900.238 ± 4.645 | 962.027 ± 54.043 |
| tensornetwork_permutation_focus_step409_316 | 396.294 ± 1.894 | 462.079 ± 16.817 | **255.797 ± 2.357** | 383.433 ± 10.167 | 510.906 ± 61.837 |
| tensornetwork_permutation_light_415 | 402.644 ± 3.805 | 368.180 ± 3.989 | **249.107 ± 1.043** | 396.271 ± 10.704 | 515.895 ± 33.436 |

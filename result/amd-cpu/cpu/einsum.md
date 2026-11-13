# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `f3a05ec9a2a7262482348fd5bddccbe90389866b`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261113_180725/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261113_180725/run.yaml`
- Timestamp: `2026-11-13T18:07:25.705019+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261113_180725`.

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

- `data/results/amd-cpu/cpu/einsum/20261113_180725/jax_cpu_t1_20261113_180725.log`
- `data/results/amd-cpu/cpu/einsum/20261113_180725/julia_omeinsum_t1_20261113_180725.log`
- `data/results/amd-cpu/cpu/einsum/20261113_180725/pytorch_cpu_t1_20261113_180725.log`
- `data/results/amd-cpu/cpu/einsum/20261113_180725/tenferro_eager_t1_20261113_180725.log`
- `data/results/amd-cpu/cpu/einsum/20261113_180725/tenferro_trace_t1_20261113_180725.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.996 ± 0.125 | **3.930 ± 0.107** | 4.072 ± 0.047 | 4.301 ± 0.155 | 12.672 ± 3.313 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.769 ± 0.015 | **0.596 ± 0.008** | 0.653 ± 0.019 | 0.782 ± 0.012 | 0.947 ± 0.115 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.742 ± 0.012 | 0.731 ± 0.012 | **0.201 ± 0.015** | 0.557 ± 0.009 | 1.976 ± 3.170 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.208 ± 0.007 | 0.740 ± 0.032 | **0.200 ± 0.008** | 0.561 ± 0.021 | 8.042 ± 5.993 |
| bin_elementwise_mul_2048x2048 | 1.899 ± 0.033 | **1.764 ± 0.090** | 15.822 ± 0.253 | 18.032 ± 0.359 | 17.880 ± 0.401 |
| bin_elementwise_mul_512x512 | 0.125 ± 0.003 | **0.122 ± 0.005** | 0.145 ± 0.007 | 0.349 ± 0.019 | 0.156 ± 0.721 |
| bin_matmul_1024 | 55.949 ± 0.284 | **49.039 ± 0.746** | 59.467 ± 0.340 | 63.991 ± 3.154 | 74.476 ± 23.088 |
| bin_matmul_256 | 1.018 ± 0.006 | **0.836 ± 0.016** | 1.330 ± 0.010 | 1.168 ± 0.022 | 2.072 ± 0.845 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.017 ± 0.009 | **0.010 ± 0.000** | 0.036 ± 0.002 | 0.201 ± 0.024 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.043 ± 0.005 | **0.040 ± 0.001** | 0.144 ± 0.007 | 0.240 ± 0.030 | 0.078 ± 0.049 |
| bin_omeinsum_matmul_10x10 | 0.022 ± 0.001 | **0.011 ± 0.001** | 0.035 ± 0.001 | 0.149 ± 0.024 | 0.025 ± 0.002 |
| bin_outer_product_4096 | 63.432 ± 0.454 | 62.566 ± 1.594 | **54.058 ± 0.771** | 56.188 ± 0.571 | 311.628 ± 2.001 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 57.052 ± 0.130 | **51.140 ± 0.658** | 60.483 ± 0.467 | 65.956 ± 4.180 | 66.685 ± 13.171 |
| gm_queen5_5_3.wcsp | 6894.398 ± 113.094 | 6650.598 ± 60.809 | **6195.396 ± 35.351** | 8048.622 ± 32.424 | 9255.863 ± 257.105 |
| lm_batch_likelihood_brackets_4_4d | 27.733 ± 0.434 | 34.305 ± 1.851 | 34.932 ± 0.737 | **15.290 ± 0.674** | 61.975 ± 31.047 |
| lm_batch_likelihood_sentence_3_12d | **59.038 ± 1.917** | 121.370 ± 9.698 | 66.686 ± 0.974 | 95.919 ± 1.369 | 121.451 ± 30.797 |
| lm_batch_likelihood_sentence_4_4d | 29.612 ± 0.499 | 39.835 ± 0.706 | 37.633 ± 0.483 | **16.729 ± 0.255** | 78.253 ± 37.920 |
| nary_matmul_chain_64 | 0.057 ± 0.000 | **0.047 ± 0.023** | 0.078 ± 0.011 | 0.213 ± 0.018 | 0.116 ± 0.036 |
| str_matrix_chain_multiplication_100 | **14.189 ± 0.234** | 15.380 ± 5.514 | 15.761 ± 0.349 | 14.758 ± 0.324 | 16.773 ± 0.662 |
| str_mps_varying_inner_product_200 | **20.089 ± 0.485** | 22.605 ± 0.200 | 25.269 ± 0.305 | 23.492 ± 2.666 | 26.488 ± 10.510 |
| str_nw_mera_closed_120 | **1639.134 ± 7.337** | 1832.979 ± 14.537 | 1974.141 ± 8.235 | 2485.345 ± 16.212 | 1847.774 ± 38.372 |
| str_nw_mera_open_26 | 1298.262 ± 7.723 | 1469.237 ± 14.923 | **1228.419 ± 4.982** | 1934.035 ± 4.199 | 1692.395 ± 46.421 |
| tensornetwork_permutation_focus_step409_316 | **467.685 ± 4.672** | 547.718 ± 2.700 | 901.636 ± 2.830 | 681.617 ± 26.473 | 675.486 ± 61.394 |
| tensornetwork_permutation_light_415 | **452.868 ± 3.741** | 513.634 ± 3.667 | 865.693 ± 4.627 | 702.297 ± 26.097 | 682.333 ± 7.998 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.996 ± 0.125 | **3.930 ± 0.107** | 4.072 ± 0.047 | 4.301 ± 0.155 | 12.672 ± 3.313 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.769 ± 0.015 | **0.596 ± 0.008** | 0.653 ± 0.019 | 0.782 ± 0.012 | 0.947 ± 0.115 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.742 ± 0.012 | 0.731 ± 0.012 | **0.201 ± 0.015** | 0.557 ± 0.009 | 1.976 ± 3.170 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.208 ± 0.007 | 0.740 ± 0.032 | **0.200 ± 0.008** | 0.561 ± 0.021 | 8.042 ± 5.993 |
| bin_elementwise_mul_2048x2048 | 1.899 ± 0.033 | **1.764 ± 0.090** | 15.822 ± 0.253 | 18.032 ± 0.359 | 17.880 ± 0.401 |
| bin_elementwise_mul_512x512 | 0.125 ± 0.003 | **0.122 ± 0.005** | 0.145 ± 0.007 | 0.349 ± 0.019 | 0.156 ± 0.721 |
| bin_matmul_1024 | 55.949 ± 0.284 | **49.039 ± 0.746** | 59.467 ± 0.340 | 63.991 ± 3.154 | 74.476 ± 23.088 |
| bin_matmul_256 | 1.018 ± 0.006 | **0.836 ± 0.016** | 1.330 ± 0.010 | 1.168 ± 0.022 | 2.072 ± 0.845 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.017 ± 0.009 | **0.010 ± 0.000** | 0.036 ± 0.002 | 0.201 ± 0.024 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.043 ± 0.005 | **0.040 ± 0.001** | 0.144 ± 0.007 | 0.240 ± 0.030 | 0.078 ± 0.049 |
| bin_omeinsum_matmul_10x10 | 0.022 ± 0.001 | **0.011 ± 0.001** | 0.035 ± 0.001 | 0.149 ± 0.024 | 0.025 ± 0.002 |
| bin_outer_product_4096 | 63.432 ± 0.454 | 62.566 ± 1.594 | **54.058 ± 0.771** | 56.188 ± 0.571 | 311.628 ± 2.001 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 57.052 ± 0.130 | **51.140 ± 0.658** | 60.483 ± 0.467 | 65.956 ± 4.180 | 66.685 ± 13.171 |
| gm_queen5_5_3.wcsp | **2346.158 ± 21.299** | 2588.025 ± 18.854 | 2477.129 ± 16.089 | 2731.227 ± 40.062 | 3321.933 ± 25.917 |
| lm_batch_likelihood_brackets_4_4d | 28.602 ± 1.070 | 34.220 ± 1.135 | 36.389 ± 0.644 | **15.380 ± 0.554** | 69.355 ± 32.708 |
| lm_batch_likelihood_sentence_3_12d | **56.917 ± 0.543** | 161.863 ± 3.557 | 66.047 ± 0.787 | 69.094 ± 1.811 | 122.910 ± 30.928 |
| lm_batch_likelihood_sentence_4_4d | 28.861 ± 0.546 | 40.690 ± 1.235 | 38.567 ± 0.467 | **15.734 ± 0.427** | 69.927 ± 48.882 |
| nary_matmul_chain_64 | 0.057 ± 0.000 | **0.047 ± 0.023** | 0.078 ± 0.011 | 0.213 ± 0.018 | 0.116 ± 0.036 |
| str_matrix_chain_multiplication_100 | 15.001 ± 0.168 | 15.217 ± 0.209 | 15.563 ± 0.219 | **14.726 ± 0.440** | 16.614 ± 0.278 |
| str_mps_varying_inner_product_200 | **23.281 ± 0.507** | 23.335 ± 0.336 | 25.090 ± 0.329 | 23.829 ± 0.430 | 24.984 ± 9.047 |
| str_nw_mera_closed_120 | **1486.165 ± 11.292** | 1489.958 ± 16.639 | 1512.819 ± 15.676 | 2032.466 ± 32.885 | 1491.315 ± 55.187 |
| str_nw_mera_open_26 | 1299.456 ± 9.491 | 1450.548 ± 10.187 | **1231.705 ± 4.073** | 1985.372 ± 6.016 | 1641.244 ± 27.236 |
| tensornetwork_permutation_focus_step409_316 | **467.685 ± 4.672** | 547.718 ± 2.700 | 901.636 ± 2.830 | 681.617 ± 26.473 | 675.486 ± 61.394 |
| tensornetwork_permutation_light_415 | **452.868 ± 3.741** | 513.634 ± 3.667 | 865.693 ± 4.627 | 702.297 ± 26.097 | 682.333 ± 7.998 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261113_184804/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261113_184804/run.yaml`
- Timestamp: `2026-11-13T18:48:04.785108+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261113_184804`.

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

- `data/results/amd-cpu/cpu/einsum/20261113_184804/jax_cpu_t4_20261113_184804.log`
- `data/results/amd-cpu/cpu/einsum/20261113_184804/julia_omeinsum_t4_20261113_184804.log`
- `data/results/amd-cpu/cpu/einsum/20261113_184804/pytorch_cpu_t4_20261113_184804.log`
- `data/results/amd-cpu/cpu/einsum/20261113_184804/tenferro_eager_t4_20261113_184804.log`
- `data/results/amd-cpu/cpu/einsum/20261113_184804/tenferro_trace_t4_20261113_184804.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 9.722 ± 0.071 | 5.186 ± 0.058 | **2.329 ± 2.365** | 3.107 ± 0.596 | 12.463 ± 3.789 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.947 ± 0.013 | 1.361 ± 0.028 | **0.377 ± 0.010** | 0.845 ± 0.093 | 0.955 ± 0.344 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.308 ± 0.014 | 0.315 ± 0.020 | **0.201 ± 0.052** | 0.584 ± 0.020 | 0.802 ± 1.394 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.289 ± 0.014 | 0.315 ± 0.023 | **0.188 ± 0.026** | 0.586 ± 0.016 | 4.866 ± 3.995 |
| bin_elementwise_mul_2048x2048 | **0.838 ± 0.047** | 0.880 ± 0.050 | 5.893 ± 0.305 | 9.356 ± 0.219 | 17.895 ± 0.389 |
| bin_elementwise_mul_512x512 | **0.123 ± 0.014** | 0.130 ± 0.005 | 0.165 ± 0.019 | 0.410 ± 0.033 | 0.135 ± 0.051 |
| bin_matmul_1024 | 60.824 ± 0.900 | 51.579 ± 0.612 | **35.581 ± 0.249** | 36.080 ± 0.442 | 53.944 ± 10.264 |
| bin_matmul_256 | 1.260 ± 0.017 | 0.925 ± 0.163 | 0.855 ± 0.005 | **0.843 ± 0.048** | 0.975 ± 0.023 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.001 | **0.009 ± 0.000** | 0.052 ± 0.006 | 0.202 ± 0.019 | 0.012 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.069 ± 0.003 | 0.064 ± 0.003 | 0.218 ± 0.016 | 0.251 ± 0.038 | **0.056 ± 0.043** |
| bin_omeinsum_matmul_10x10 | 0.068 ± 0.002 | 0.046 ± 0.003 | 0.049 ± 0.004 | 0.152 ± 0.034 | **0.026 ± 0.001** |
| bin_outer_product_4096 | 22.247 ± 0.232 | 22.462 ± 0.282 | **18.591 ± 0.585** | 28.858 ± 0.580 | 308.239 ± 3.687 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 60.156 ± 0.689 | 54.400 ± 0.597 | **30.401 ± 0.391** | 38.810 ± 4.615 | 43.321 ± 10.173 |
| gm_queen5_5_3.wcsp | 6326.875 ± 107.386 | 5984.790 ± 84.183 | **2705.730 ± 39.950** | 5289.282 ± 82.339 | 8577.861 ± 202.490 |
| lm_batch_likelihood_brackets_4_4d | 29.790 ± 0.301 | 51.016 ± 2.713 | **16.828 ± 0.757** | 23.522 ± 0.920 | 47.772 ± 34.654 |
| lm_batch_likelihood_sentence_3_12d | 62.269 ± 0.612 | 185.469 ± 4.830 | 59.093 ± 1.207 | **44.903 ± 8.457** | 107.149 ± 31.254 |
| lm_batch_likelihood_sentence_4_4d | 32.572 ± 0.711 | 57.548 ± 2.244 | **17.483 ± 0.258** | 23.984 ± 1.958 | 75.261 ± 26.696 |
| nary_matmul_chain_64 | 0.146 ± 0.003 | 0.131 ± 0.002 | **0.094 ± 0.008** | 0.278 ± 0.023 | 0.150 ± 0.054 |
| str_matrix_chain_multiplication_100 | 17.482 ± 0.383 | 18.335 ± 0.223 | **10.660 ± 0.100** | 15.705 ± 0.564 | 17.823 ± 4.266 |
| str_mps_varying_inner_product_200 | 26.208 ± 0.321 | 28.598 ± 0.308 | **25.820 ± 0.695** | 31.109 ± 0.397 | 31.384 ± 3.907 |
| str_nw_mera_closed_120 | 1670.393 ± 5.655 | 1885.536 ± 34.734 | **951.344 ± 19.919** | 1163.285 ± 30.957 | 1259.284 ± 44.885 |
| str_nw_mera_open_26 | 1331.536 ± 11.298 | 1292.126 ± 9.301 | **590.526 ± 13.162** | 935.924 ± 16.382 | 1289.095 ± 55.882 |
| tensornetwork_permutation_focus_step409_316 | 472.125 ± 3.682 | 506.488 ± 4.552 | **347.000 ± 14.586** | 391.130 ± 6.919 | 594.356 ± 24.600 |
| tensornetwork_permutation_light_415 | 466.117 ± 2.629 | 466.649 ± 5.314 | **343.394 ± 8.396** | 408.769 ± 11.583 | 590.019 ± 17.348 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 9.722 ± 0.071 | 5.186 ± 0.058 | **2.329 ± 2.365** | 3.107 ± 0.596 | 12.463 ± 3.789 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.947 ± 0.013 | 1.361 ± 0.028 | **0.377 ± 0.010** | 0.845 ± 0.093 | 0.955 ± 0.344 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.308 ± 0.014 | 0.315 ± 0.020 | **0.201 ± 0.052** | 0.584 ± 0.020 | 0.802 ± 1.394 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.289 ± 0.014 | 0.315 ± 0.023 | **0.188 ± 0.026** | 0.586 ± 0.016 | 4.866 ± 3.995 |
| bin_elementwise_mul_2048x2048 | **0.838 ± 0.047** | 0.880 ± 0.050 | 5.893 ± 0.305 | 9.356 ± 0.219 | 17.895 ± 0.389 |
| bin_elementwise_mul_512x512 | **0.123 ± 0.014** | 0.130 ± 0.005 | 0.165 ± 0.019 | 0.410 ± 0.033 | 0.135 ± 0.051 |
| bin_matmul_1024 | 60.824 ± 0.900 | 51.579 ± 0.612 | **35.581 ± 0.249** | 36.080 ± 0.442 | 53.944 ± 10.264 |
| bin_matmul_256 | 1.260 ± 0.017 | 0.925 ± 0.163 | 0.855 ± 0.005 | **0.843 ± 0.048** | 0.975 ± 0.023 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.001 | **0.009 ± 0.000** | 0.052 ± 0.006 | 0.202 ± 0.019 | 0.012 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.069 ± 0.003 | 0.064 ± 0.003 | 0.218 ± 0.016 | 0.251 ± 0.038 | **0.056 ± 0.043** |
| bin_omeinsum_matmul_10x10 | 0.068 ± 0.002 | 0.046 ± 0.003 | 0.049 ± 0.004 | 0.152 ± 0.034 | **0.026 ± 0.001** |
| bin_outer_product_4096 | 22.247 ± 0.232 | 22.462 ± 0.282 | **18.591 ± 0.585** | 28.858 ± 0.580 | 308.239 ± 3.687 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 60.156 ± 0.689 | 54.400 ± 0.597 | **30.401 ± 0.391** | 38.810 ± 4.615 | 43.321 ± 10.173 |
| gm_queen5_5_3.wcsp | 2314.679 ± 15.612 | 2059.384 ± 18.162 | **1141.158 ± 14.932** | 1884.702 ± 21.075 | 3034.148 ± 84.453 |
| lm_batch_likelihood_brackets_4_4d | 30.503 ± 0.520 | 53.716 ± 1.957 | **17.782 ± 0.240** | 23.417 ± 1.983 | 56.765 ± 29.553 |
| lm_batch_likelihood_sentence_3_12d | **57.048 ± 0.970** | 217.874 ± 2.361 | 61.038 ± 1.588 | 61.798 ± 10.743 | 112.433 ± 17.387 |
| lm_batch_likelihood_sentence_4_4d | 32.291 ± 0.912 | 60.691 ± 3.201 | **18.093 ± 0.311** | 22.646 ± 2.208 | 56.168 ± 36.996 |
| nary_matmul_chain_64 | 0.146 ± 0.003 | 0.131 ± 0.002 | **0.094 ± 0.008** | 0.278 ± 0.023 | 0.150 ± 0.054 |
| str_matrix_chain_multiplication_100 | 18.396 ± 0.193 | 18.753 ± 0.192 | **10.666 ± 0.104** | 15.309 ± 0.985 | 14.596 ± 0.558 |
| str_mps_varying_inner_product_200 | 30.042 ± 0.909 | 30.747 ± 0.282 | 28.141 ± 0.667 | 31.173 ± 0.507 | **24.257 ± 7.956** |
| str_nw_mera_closed_120 | 1563.358 ± 30.356 | 1537.728 ± 18.874 | **779.400 ± 15.221** | 999.812 ± 35.605 | 977.394 ± 61.441 |
| str_nw_mera_open_26 | 1336.917 ± 11.310 | 1334.354 ± 6.136 | **605.880 ± 7.844** | 959.181 ± 24.113 | 1263.500 ± 71.065 |
| tensornetwork_permutation_focus_step409_316 | 472.125 ± 3.682 | 506.488 ± 4.552 | **347.000 ± 14.586** | 391.130 ± 6.919 | 594.356 ± 24.600 |
| tensornetwork_permutation_light_415 | 466.117 ± 2.629 | 466.649 ± 5.314 | **343.394 ± 8.396** | 408.769 ± 11.583 | 590.019 ± 17.348 |

# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `35979dcd7110a29d0728b2e128c72914f4c3f67b`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261111_180426/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261111_180426/run.yaml`
- Timestamp: `2026-11-11T18:04:26.325001+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261111_180426`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
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

- `data/results/amd-cpu/cpu/einsum/20261111_180426/jax_cpu_t1_20261111_180426.log`
- `data/results/amd-cpu/cpu/einsum/20261111_180426/julia_omeinsum_t1_20261111_180426.log`
- `data/results/amd-cpu/cpu/einsum/20261111_180426/pytorch_cpu_t1_20261111_180426.log`
- `data/results/amd-cpu/cpu/einsum/20261111_180426/tenferro_eager_t1_20261111_180426.log`
- `data/results/amd-cpu/cpu/einsum/20261111_180426/tenferro_trace_t1_20261111_180426.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.633 ± 0.025 | **3.319 ± 0.013** | 3.629 ± 0.027 | 3.534 ± 0.124 | 9.362 ± 5.317 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.625 ± 0.014 | **0.502 ± 0.009** | 0.553 ± 0.023 | 0.629 ± 0.022 | 0.809 ± 1.412 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.625 ± 0.011 | 0.623 ± 0.011 | **0.168 ± 0.047** | 0.486 ± 0.017 | 0.613 ± 0.106 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.942 ± 0.004 | 0.643 ± 0.010 | **0.130 ± 0.009** | 0.485 ± 0.021 | 1.737 ± 1.189 |
| bin_elementwise_mul_2048x2048 | **1.619 ± 0.091** | 1.836 ± 0.103 | 13.740 ± 0.433 | 17.591 ± 0.989 | 13.791 ± 10.970 |
| bin_elementwise_mul_512x512 | **0.073 ± 0.007** | 0.094 ± 0.020 | 0.081 ± 0.006 | 0.256 ± 0.024 | 0.120 ± 0.038 |
| bin_matmul_1024 | 50.234 ± 0.144 | **42.125 ± 0.171** | 51.952 ± 0.332 | 46.880 ± 0.425 | 59.150 ± 22.927 |
| bin_matmul_256 | 0.928 ± 0.007 | **0.704 ± 0.025** | 1.240 ± 0.009 | 0.798 ± 0.018 | 1.085 ± 0.723 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.011 ± 0.003 | **0.005 ± 0.002** | 0.021 ± 0.001 | 0.133 ± 0.034 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.032 ± 0.001 | **0.029 ± 0.001** | 0.096 ± 0.003 | 0.164 ± 0.015 | 0.063 ± 0.033 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.000 | **0.009 ± 0.000** | 0.019 ± 0.001 | 0.075 ± 0.012 | 0.018 ± 0.001 |
| bin_outer_product_4096 | 51.397 ± 0.126 | 51.613 ± 0.291 | **44.171 ± 0.966** | 48.914 ± 0.623 | 269.511 ± 4.045 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 48.303 ± 0.168 | **43.699 ± 0.304** | 54.138 ± 0.635 | 49.395 ± 0.451 | 57.976 ± 28.547 |
| gm_queen5_5_3.wcsp | 5938.270 ± 82.268 | 5941.061 ± 75.099 | **5562.516 ± 86.730** | 5776.396 ± 64.020 | 8080.501 ± 183.589 |
| lm_batch_likelihood_brackets_4_4d | 22.526 ± 0.122 | 25.625 ± 1.181 | 24.971 ± 0.335 | **11.737 ± 0.194** | 52.631 ± 41.094 |
| lm_batch_likelihood_sentence_3_12d | **50.067 ± 0.279** | 103.140 ± 3.764 | 58.339 ± 0.882 | 78.632 ± 1.924 | 102.643 ± 21.245 |
| lm_batch_likelihood_sentence_4_4d | 23.421 ± 0.139 | 29.065 ± 1.348 | 27.808 ± 0.295 | **13.503 ± 0.106** | 37.050 ± 25.209 |
| nary_matmul_chain_64 | 0.045 ± 0.001 | **0.037 ± 0.012** | 0.050 ± 0.001 | 0.118 ± 0.011 | 0.090 ± 0.041 |
| str_matrix_chain_multiplication_100 | 12.671 ± 0.216 | 14.943 ± 0.151 | 15.376 ± 0.248 | **12.666 ± 0.115** | 16.469 ± 10.450 |
| str_mps_varying_inner_product_200 | **16.975 ± 0.243** | 19.519 ± 0.130 | 20.339 ± 0.105 | 20.066 ± 0.638 | 25.534 ± 7.504 |
| str_nw_mera_closed_120 | **1437.926 ± 14.169** | 1587.328 ± 22.228 | 1715.709 ± 20.673 | 1878.699 ± 22.831 | 1741.212 ± 26.900 |
| str_nw_mera_open_26 | 1113.991 ± 6.878 | 1317.791 ± 26.121 | **1042.647 ± 12.279** | 1516.400 ± 25.105 | 1582.413 ± 10.654 |
| tensornetwork_permutation_focus_step409_316 | **403.147 ± 5.093** | 500.840 ± 4.933 | 708.883 ± 8.324 | 657.131 ± 26.298 | 513.176 ± 74.768 |
| tensornetwork_permutation_light_415 | **395.514 ± 1.700** | 438.776 ± 3.089 | 683.249 ± 19.432 | 685.333 ± 21.229 | 538.248 ± 54.880 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.633 ± 0.025 | **3.319 ± 0.013** | 3.629 ± 0.027 | 3.534 ± 0.124 | 9.362 ± 5.317 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.625 ± 0.014 | **0.502 ± 0.009** | 0.553 ± 0.023 | 0.629 ± 0.022 | 0.809 ± 1.412 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.625 ± 0.011 | 0.623 ± 0.011 | **0.168 ± 0.047** | 0.486 ± 0.017 | 0.613 ± 0.106 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.942 ± 0.004 | 0.643 ± 0.010 | **0.130 ± 0.009** | 0.485 ± 0.021 | 1.737 ± 1.189 |
| bin_elementwise_mul_2048x2048 | **1.619 ± 0.091** | 1.836 ± 0.103 | 13.740 ± 0.433 | 17.591 ± 0.989 | 13.791 ± 10.970 |
| bin_elementwise_mul_512x512 | **0.073 ± 0.007** | 0.094 ± 0.020 | 0.081 ± 0.006 | 0.256 ± 0.024 | 0.120 ± 0.038 |
| bin_matmul_1024 | 50.234 ± 0.144 | **42.125 ± 0.171** | 51.952 ± 0.332 | 46.880 ± 0.425 | 59.150 ± 22.927 |
| bin_matmul_256 | 0.928 ± 0.007 | **0.704 ± 0.025** | 1.240 ± 0.009 | 0.798 ± 0.018 | 1.085 ± 0.723 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.011 ± 0.003 | **0.005 ± 0.002** | 0.021 ± 0.001 | 0.133 ± 0.034 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.032 ± 0.001 | **0.029 ± 0.001** | 0.096 ± 0.003 | 0.164 ± 0.015 | 0.063 ± 0.033 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.000 | **0.009 ± 0.000** | 0.019 ± 0.001 | 0.075 ± 0.012 | 0.018 ± 0.001 |
| bin_outer_product_4096 | 51.397 ± 0.126 | 51.613 ± 0.291 | **44.171 ± 0.966** | 48.914 ± 0.623 | 269.511 ± 4.045 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 48.303 ± 0.168 | **43.699 ± 0.304** | 54.138 ± 0.635 | 49.395 ± 0.451 | 57.976 ± 28.547 |
| gm_queen5_5_3.wcsp | **2004.834 ± 21.404** | 2325.725 ± 22.338 | 2353.542 ± 48.885 | 2193.261 ± 27.969 | 2982.901 ± 38.029 |
| lm_batch_likelihood_brackets_4_4d | 23.145 ± 0.082 | 26.472 ± 0.724 | 27.868 ± 0.837 | **12.691 ± 0.165** | 58.998 ± 27.788 |
| lm_batch_likelihood_sentence_3_12d | **50.199 ± 0.258** | 132.593 ± 2.648 | 66.486 ± 1.126 | 77.668 ± 1.698 | 105.931 ± 52.290 |
| lm_batch_likelihood_sentence_4_4d | 23.872 ± 0.256 | 31.572 ± 1.045 | 30.483 ± 0.712 | **13.238 ± 1.226** | 32.900 ± 37.465 |
| nary_matmul_chain_64 | 0.045 ± 0.001 | **0.037 ± 0.012** | 0.050 ± 0.001 | 0.118 ± 0.011 | 0.090 ± 0.041 |
| str_matrix_chain_multiplication_100 | 13.915 ± 0.061 | 14.507 ± 0.056 | 16.820 ± 0.659 | **12.730 ± 0.177** | 14.979 ± 0.737 |
| str_mps_varying_inner_product_200 | **20.011 ± 0.088** | 20.131 ± 0.108 | 20.060 ± 0.142 | 23.558 ± 0.922 | 21.508 ± 1.616 |
| str_nw_mera_closed_120 | 1312.517 ± 7.282 | **1293.043 ± 13.512** | 1400.428 ± 26.241 | 1498.081 ± 23.921 | 1452.744 ± 49.919 |
| str_nw_mera_open_26 | 1120.757 ± 17.618 | 1281.649 ± 26.518 | **1099.760 ± 11.210** | 1609.765 ± 21.871 | 1516.144 ± 84.310 |
| tensornetwork_permutation_focus_step409_316 | **403.147 ± 5.093** | 500.840 ± 4.933 | 708.883 ± 8.324 | 657.131 ± 26.298 | 513.176 ± 74.768 |
| tensornetwork_permutation_light_415 | **395.514 ± 1.700** | 438.776 ± 3.089 | 683.249 ± 19.432 | 685.333 ± 21.229 | 538.248 ± 54.880 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261111_183853/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261111_183853/run.yaml`
- Timestamp: `2026-11-11T18:38:53.952407+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261111_183853`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
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

- `data/results/amd-cpu/cpu/einsum/20261111_183853/jax_cpu_t4_20261111_183853.log`
- `data/results/amd-cpu/cpu/einsum/20261111_183853/julia_omeinsum_t4_20261111_183853.log`
- `data/results/amd-cpu/cpu/einsum/20261111_183853/pytorch_cpu_t4_20261111_183853.log`
- `data/results/amd-cpu/cpu/einsum/20261111_183853/tenferro_eager_t4_20261111_183853.log`
- `data/results/amd-cpu/cpu/einsum/20261111_183853/tenferro_trace_t4_20261111_183853.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 8.510 ± 0.026 | 4.451 ± 0.014 | **2.257 ± 3.015** | 2.349 ± 0.625 | 9.822 ± 8.305 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.618 ± 0.006 | 1.157 ± 0.012 | **0.288 ± 0.013** | 0.616 ± 0.023 | 0.733 ± 1.293 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.241 ± 0.002 | 0.238 ± 0.009 | **0.082 ± 0.008** | 0.349 ± 0.021 | 1.617 ± 2.579 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.243 ± 0.011 | 0.237 ± 0.011 | **0.077 ± 0.004** | 0.348 ± 0.011 | 6.522 ± 4.652 |
| bin_elementwise_mul_2048x2048 | 0.846 ± 0.049 | **0.787 ± 0.042** | 5.341 ± 0.112 | 8.681 ± 0.431 | 15.736 ± 0.748 |
| bin_elementwise_mul_512x512 | 0.049 ± 0.001 | **0.045 ± 0.004** | 0.062 ± 0.012 | 0.230 ± 0.025 | 0.098 ± 0.215 |
| bin_matmul_1024 | 54.943 ± 0.148 | 44.651 ± 0.615 | 30.611 ± 0.353 | **26.256 ± 0.721** | 41.507 ± 15.252 |
| bin_matmul_256 | 1.175 ± 0.013 | 0.795 ± 0.151 | 0.828 ± 0.006 | **0.604 ± 0.045** | 0.696 ± 0.044 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.007 ± 0.000** | 0.030 ± 0.010 | 0.130 ± 0.018 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.044 ± 0.003 | 0.043 ± 0.002 | 0.135 ± 0.009 | 0.175 ± 0.020 | **0.035 ± 0.002** |
| bin_omeinsum_matmul_10x10 | 0.047 ± 0.002 | 0.035 ± 0.002 | 0.029 ± 0.007 | 0.077 ± 0.020 | **0.019 ± 0.005** |
| bin_outer_product_4096 | 17.998 ± 0.475 | 18.086 ± 0.390 | **15.012 ± 0.425** | 24.888 ± 2.331 | 248.230 ± 2.268 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 51.673 ± 0.284 | 46.800 ± 1.431 | **26.022 ± 0.281** | 29.063 ± 1.634 | 35.161 ± 12.179 |
| gm_queen5_5_3.wcsp | 5277.901 ± 28.005 | 4999.667 ± 60.881 | **2379.297 ± 30.141** | 3807.356 ± 48.193 | 7435.468 ± 167.620 |
| lm_batch_likelihood_brackets_4_4d | 23.874 ± 0.085 | 38.518 ± 1.659 | **12.630 ± 0.133** | 17.676 ± 0.931 | 35.551 ± 16.095 |
| lm_batch_likelihood_sentence_3_12d | 54.044 ± 0.200 | 134.968 ± 9.845 | **28.432 ± 0.391** | 47.061 ± 3.480 | 89.863 ± 14.641 |
| lm_batch_likelihood_sentence_4_4d | 26.273 ± 0.099 | 43.258 ± 1.447 | **13.481 ± 0.088** | 18.623 ± 0.767 | 47.871 ± 24.761 |
| nary_matmul_chain_64 | 0.111 ± 0.002 | 0.101 ± 0.001 | **0.059 ± 0.002** | 0.163 ± 0.024 | 0.101 ± 0.039 |
| str_matrix_chain_multiplication_100 | 13.876 ± 0.042 | 16.315 ± 0.157 | **9.395 ± 0.077** | 12.770 ± 0.845 | 11.967 ± 1.454 |
| str_mps_varying_inner_product_200 | 21.964 ± 0.121 | 24.917 ± 0.127 | **21.064 ± 0.342** | 25.523 ± 0.540 | 26.453 ± 17.111 |
| str_nw_mera_closed_120 | 1490.519 ± 13.008 | 1658.098 ± 33.809 | **837.608 ± 12.339** | 922.107 ± 19.791 | 1133.072 ± 48.726 |
| str_nw_mera_open_26 | 1126.631 ± 15.644 | 1140.789 ± 16.273 | **516.834 ± 7.952** | 767.408 ± 14.198 | 1135.514 ± 67.936 |
| tensornetwork_permutation_focus_step409_316 | 406.244 ± 3.711 | 437.587 ± 3.356 | **297.170 ± 6.609** | 376.159 ± 9.898 | 507.015 ± 21.074 |
| tensornetwork_permutation_light_415 | 405.223 ± 1.887 | 402.135 ± 4.957 | **292.268 ± 6.394** | 389.501 ± 10.332 | 494.147 ± 41.424 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 8.510 ± 0.026 | 4.451 ± 0.014 | **2.257 ± 3.015** | 2.349 ± 0.625 | 9.822 ± 8.305 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.618 ± 0.006 | 1.157 ± 0.012 | **0.288 ± 0.013** | 0.616 ± 0.023 | 0.733 ± 1.293 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.241 ± 0.002 | 0.238 ± 0.009 | **0.082 ± 0.008** | 0.349 ± 0.021 | 1.617 ± 2.579 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.243 ± 0.011 | 0.237 ± 0.011 | **0.077 ± 0.004** | 0.348 ± 0.011 | 6.522 ± 4.652 |
| bin_elementwise_mul_2048x2048 | 0.846 ± 0.049 | **0.787 ± 0.042** | 5.341 ± 0.112 | 8.681 ± 0.431 | 15.736 ± 0.748 |
| bin_elementwise_mul_512x512 | 0.049 ± 0.001 | **0.045 ± 0.004** | 0.062 ± 0.012 | 0.230 ± 0.025 | 0.098 ± 0.215 |
| bin_matmul_1024 | 54.943 ± 0.148 | 44.651 ± 0.615 | 30.611 ± 0.353 | **26.256 ± 0.721** | 41.507 ± 15.252 |
| bin_matmul_256 | 1.175 ± 0.013 | 0.795 ± 0.151 | 0.828 ± 0.006 | **0.604 ± 0.045** | 0.696 ± 0.044 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.007 ± 0.000** | 0.030 ± 0.010 | 0.130 ± 0.018 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.044 ± 0.003 | 0.043 ± 0.002 | 0.135 ± 0.009 | 0.175 ± 0.020 | **0.035 ± 0.002** |
| bin_omeinsum_matmul_10x10 | 0.047 ± 0.002 | 0.035 ± 0.002 | 0.029 ± 0.007 | 0.077 ± 0.020 | **0.019 ± 0.005** |
| bin_outer_product_4096 | 17.998 ± 0.475 | 18.086 ± 0.390 | **15.012 ± 0.425** | 24.888 ± 2.331 | 248.230 ± 2.268 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 51.673 ± 0.284 | 46.800 ± 1.431 | **26.022 ± 0.281** | 29.063 ± 1.634 | 35.161 ± 12.179 |
| gm_queen5_5_3.wcsp | 1965.432 ± 16.294 | 1813.351 ± 32.256 | **1048.327 ± 20.681** | 1483.910 ± 28.781 | 2731.770 ± 53.424 |
| lm_batch_likelihood_brackets_4_4d | 24.498 ± 0.069 | 40.543 ± 2.157 | **13.208 ± 0.082** | 18.712 ± 2.624 | 48.185 ± 28.834 |
| lm_batch_likelihood_sentence_3_12d | 50.966 ± 0.218 | 184.662 ± 1.196 | 51.471 ± 1.831 | **40.702 ± 2.417** | 99.929 ± 40.455 |
| lm_batch_likelihood_sentence_4_4d | 26.497 ± 0.108 | 46.933 ± 2.163 | **14.368 ± 0.401** | 18.630 ± 1.363 | 45.162 ± 35.437 |
| nary_matmul_chain_64 | 0.111 ± 0.002 | 0.101 ± 0.001 | **0.059 ± 0.002** | 0.163 ± 0.024 | 0.101 ± 0.039 |
| str_matrix_chain_multiplication_100 | 15.943 ± 0.203 | 17.105 ± 0.320 | **9.617 ± 0.060** | 13.017 ± 1.163 | 12.835 ± 8.648 |
| str_mps_varying_inner_product_200 | 25.220 ± 0.596 | 25.401 ± 0.342 | 24.732 ± 0.514 | 27.112 ± 0.636 | **22.286 ± 8.993** |
| str_nw_mera_closed_120 | 1339.772 ± 9.421 | 1343.951 ± 25.860 | **674.214 ± 19.942** | 769.979 ± 31.747 | 905.842 ± 45.255 |
| str_nw_mera_open_26 | 1123.790 ± 5.272 | 1189.340 ± 21.487 | **533.297 ± 12.514** | 777.708 ± 10.127 | 1131.858 ± 52.961 |
| tensornetwork_permutation_focus_step409_316 | 406.244 ± 3.711 | 437.587 ± 3.356 | **297.170 ± 6.609** | 376.159 ± 9.898 | 507.015 ± 21.074 |
| tensornetwork_permutation_light_415 | 405.223 ± 1.887 | 402.135 ± 4.957 | **292.268 ± 6.394** | 389.501 ± 10.332 | 494.147 ± 41.424 |

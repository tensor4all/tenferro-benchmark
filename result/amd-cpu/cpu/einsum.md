# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `9e383937df4e365bd68bf5b16dbfcaaa269bcc82`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261115_180744/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261115_180744/run.yaml`
- Timestamp: `2026-11-15T18:07:44.075207+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261115_180744`.

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

- `data/results/amd-cpu/cpu/einsum/20261115_180744/jax_cpu_t1_20261115_180744.log`
- `data/results/amd-cpu/cpu/einsum/20261115_180744/julia_omeinsum_t1_20261115_180744.log`
- `data/results/amd-cpu/cpu/einsum/20261115_180744/pytorch_cpu_t1_20261115_180744.log`
- `data/results/amd-cpu/cpu/einsum/20261115_180744/tenferro_eager_t1_20261115_180744.log`
- `data/results/amd-cpu/cpu/einsum/20261115_180744/tenferro_trace_t1_20261115_180744.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.517 ± 0.105 | **3.977 ± 0.105** | 4.316 ± 0.080 | 4.498 ± 0.236 | 10.020 ± 5.040 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.803 ± 0.042 | **0.603 ± 0.011** | 0.676 ± 0.022 | 0.832 ± 0.020 | 0.979 ± 0.271 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.737 ± 0.017 | 0.731 ± 0.015 | **0.253 ± 0.098** | 0.557 ± 0.019 | 0.904 ± 3.199 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.243 ± 0.010 | 0.742 ± 0.025 | **0.224 ± 0.025** | 0.563 ± 0.021 | 5.151 ± 6.472 |
| bin_elementwise_mul_2048x2048 | 2.080 ± 0.192 | **2.009 ± 0.191** | 16.250 ± 0.249 | 17.954 ± 0.358 | 18.330 ± 4.696 |
| bin_elementwise_mul_512x512 | **0.141 ± 0.012** | 0.144 ± 0.015 | 0.147 ± 0.010 | 0.364 ± 0.044 | 0.191 ± 0.207 |
| bin_matmul_1024 | 58.016 ± 0.465 | **48.900 ± 0.378** | 60.533 ± 0.697 | 62.865 ± 1.360 | 76.557 ± 7.373 |
| bin_matmul_256 | 1.027 ± 0.007 | **0.807 ± 0.012** | 1.335 ± 0.015 | 1.162 ± 0.034 | 1.258 ± 0.023 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | **0.010 ± 0.000** | 0.011 ± 0.002 | 0.035 ± 0.005 | 0.210 ± 0.018 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.046 ± 0.005 | **0.040 ± 0.001** | 0.144 ± 0.016 | 0.240 ± 0.030 | 0.062 ± 0.037 |
| bin_omeinsum_matmul_10x10 | 0.022 ± 0.001 | **0.019 ± 0.002** | 0.034 ± 0.007 | 0.146 ± 0.016 | 0.026 ± 0.002 |
| bin_outer_product_4096 | 66.754 ± 0.445 | 64.394 ± 0.473 | **54.748 ± 0.502** | 56.852 ± 0.763 | 322.711 ± 195.331 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 58.953 ± 0.925 | **51.560 ± 0.377** | 60.754 ± 0.603 | 67.003 ± 3.473 | 69.239 ± 7.705 |
| gm_queen5_5_3.wcsp | 7390.962 ± 87.996 | 6830.543 ± 54.748 | **6248.406 ± 52.402** | 8589.099 ± 204.123 | 9327.715 ± 159.614 |
| lm_batch_likelihood_brackets_4_4d | 29.256 ± 0.417 | 36.567 ± 1.377 | 36.355 ± 1.564 | **16.041 ± 0.894** | 41.243 ± 21.248 |
| lm_batch_likelihood_sentence_3_12d | **63.606 ± 0.479** | 143.673 ± 42.674 | 67.806 ± 0.602 | 103.347 ± 2.036 | 81.085 ± 7.183 |
| lm_batch_likelihood_sentence_4_4d | 31.096 ± 0.639 | 40.679 ± 1.743 | 36.864 ± 0.969 | **18.852 ± 0.898** | 35.848 ± 1.048 |
| nary_matmul_chain_64 | 0.057 ± 0.005 | **0.046 ± 0.001** | 0.078 ± 0.004 | 0.198 ± 0.003 | 0.071 ± 0.005 |
| str_matrix_chain_multiplication_100 | **14.517 ± 0.159** | 15.776 ± 0.138 | 16.354 ± 0.337 | 16.370 ± 1.485 | 17.095 ± 0.329 |
| str_mps_varying_inner_product_200 | **20.619 ± 0.229** | 22.504 ± 0.387 | 25.963 ± 0.315 | 24.444 ± 0.289 | 26.170 ± 1.385 |
| str_nw_mera_closed_120 | **1686.444 ± 13.238** | 1887.601 ± 35.029 | 1998.264 ± 7.063 | 2527.725 ± 21.357 | 1894.228 ± 27.103 |
| str_nw_mera_open_26 | 1332.095 ± 15.228 | 1538.962 ± 22.923 | **1233.281 ± 10.104** | 1975.629 ± 13.460 | 1675.994 ± 73.137 |
| tensornetwork_permutation_focus_step409_316 | **482.700 ± 4.948** | 587.791 ± 5.660 | 894.491 ± 8.565 | 794.022 ± 252.830 | 653.490 ± 99.993 |
| tensornetwork_permutation_light_415 | **476.242 ± 4.212** | 518.697 ± 4.149 | 859.931 ± 7.077 | 891.334 ± 138.610 | 589.005 ± 20.840 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.517 ± 0.105 | **3.977 ± 0.105** | 4.316 ± 0.080 | 4.498 ± 0.236 | 10.020 ± 5.040 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.803 ± 0.042 | **0.603 ± 0.011** | 0.676 ± 0.022 | 0.832 ± 0.020 | 0.979 ± 0.271 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.737 ± 0.017 | 0.731 ± 0.015 | **0.253 ± 0.098** | 0.557 ± 0.019 | 0.904 ± 3.199 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.243 ± 0.010 | 0.742 ± 0.025 | **0.224 ± 0.025** | 0.563 ± 0.021 | 5.151 ± 6.472 |
| bin_elementwise_mul_2048x2048 | 2.080 ± 0.192 | **2.009 ± 0.191** | 16.250 ± 0.249 | 17.954 ± 0.358 | 18.330 ± 4.696 |
| bin_elementwise_mul_512x512 | **0.141 ± 0.012** | 0.144 ± 0.015 | 0.147 ± 0.010 | 0.364 ± 0.044 | 0.191 ± 0.207 |
| bin_matmul_1024 | 58.016 ± 0.465 | **48.900 ± 0.378** | 60.533 ± 0.697 | 62.865 ± 1.360 | 76.557 ± 7.373 |
| bin_matmul_256 | 1.027 ± 0.007 | **0.807 ± 0.012** | 1.335 ± 0.015 | 1.162 ± 0.034 | 1.258 ± 0.023 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | **0.010 ± 0.000** | 0.011 ± 0.002 | 0.035 ± 0.005 | 0.210 ± 0.018 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.046 ± 0.005 | **0.040 ± 0.001** | 0.144 ± 0.016 | 0.240 ± 0.030 | 0.062 ± 0.037 |
| bin_omeinsum_matmul_10x10 | 0.022 ± 0.001 | **0.019 ± 0.002** | 0.034 ± 0.007 | 0.146 ± 0.016 | 0.026 ± 0.002 |
| bin_outer_product_4096 | 66.754 ± 0.445 | 64.394 ± 0.473 | **54.748 ± 0.502** | 56.852 ± 0.763 | 322.711 ± 195.331 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 58.953 ± 0.925 | **51.560 ± 0.377** | 60.754 ± 0.603 | 67.003 ± 3.473 | 69.239 ± 7.705 |
| gm_queen5_5_3.wcsp | **2461.912 ± 25.273** | 2671.185 ± 22.621 | 2560.646 ± 95.514 | 2881.086 ± 27.607 | 3433.189 ± 96.828 |
| lm_batch_likelihood_brackets_4_4d | 29.092 ± 0.166 | 36.568 ± 0.734 | 37.126 ± 0.885 | **16.872 ± 0.377** | 36.504 ± 36.432 |
| lm_batch_likelihood_sentence_3_12d | **61.018 ± 1.227** | 183.724 ± 2.678 | 73.673 ± 0.762 | 74.750 ± 2.196 | 114.800 ± 49.009 |
| lm_batch_likelihood_sentence_4_4d | 30.090 ± 0.387 | 42.959 ± 3.287 | 38.831 ± 0.730 | **17.266 ± 0.258** | 37.807 ± 11.728 |
| nary_matmul_chain_64 | 0.057 ± 0.005 | **0.046 ± 0.001** | 0.078 ± 0.004 | 0.198 ± 0.003 | 0.071 ± 0.005 |
| str_matrix_chain_multiplication_100 | **15.600 ± 0.176** | 15.800 ± 0.375 | 16.207 ± 0.195 | 15.692 ± 0.277 | 17.295 ± 0.415 |
| str_mps_varying_inner_product_200 | **24.542 ± 0.129** | 24.727 ± 0.389 | 26.267 ± 0.407 | 24.942 ± 0.304 | 26.072 ± 1.340 |
| str_nw_mera_closed_120 | 1531.123 ± 14.811 | 1536.191 ± 23.770 | 1598.620 ± 18.107 | 2102.877 ± 48.351 | **1526.571 ± 14.648** |
| str_nw_mera_open_26 | 1333.615 ± 17.038 | 1524.734 ± 17.477 | **1289.150 ± 9.960** | 2030.684 ± 12.887 | 1616.997 ± 56.046 |
| tensornetwork_permutation_focus_step409_316 | **482.700 ± 4.948** | 587.791 ± 5.660 | 894.491 ± 8.565 | 794.022 ± 252.830 | 653.490 ± 99.993 |
| tensornetwork_permutation_light_415 | **476.242 ± 4.212** | 518.697 ± 4.149 | 859.931 ± 7.077 | 891.334 ± 138.610 | 589.005 ± 20.840 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261115_184936/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261115_184936/run.yaml`
- Timestamp: `2026-11-15T18:49:36.321391+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261115_184936`.

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

- `data/results/amd-cpu/cpu/einsum/20261115_184936/jax_cpu_t4_20261115_184936.log`
- `data/results/amd-cpu/cpu/einsum/20261115_184936/julia_omeinsum_t4_20261115_184936.log`
- `data/results/amd-cpu/cpu/einsum/20261115_184936/pytorch_cpu_t4_20261115_184936.log`
- `data/results/amd-cpu/cpu/einsum/20261115_184936/tenferro_eager_t4_20261115_184936.log`
- `data/results/amd-cpu/cpu/einsum/20261115_184936/tenferro_trace_t4_20261115_184936.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 9.978 ± 0.191 | 5.300 ± 0.067 | **2.320 ± 3.186** | 3.055 ± 0.615 | 13.044 ± 4.091 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.980 ± 0.038 | 1.376 ± 0.034 | **0.377 ± 0.031** | 0.855 ± 0.015 | 0.903 ± 0.112 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.320 ± 0.014 | 0.313 ± 0.032 | **0.207 ± 0.015** | 0.464 ± 0.021 | 3.801 ± 3.352 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.289 ± 0.013 | 0.306 ± 0.023 | **0.205 ± 0.012** | 0.469 ± 0.033 | 6.205 ± 3.242 |
| bin_elementwise_mul_2048x2048 | 1.048 ± 0.021 | **0.900 ± 0.056** | 5.767 ± 0.202 | 9.622 ± 0.182 | 20.289 ± 0.849 |
| bin_elementwise_mul_512x512 | 0.145 ± 0.012 | **0.140 ± 0.010** | 0.161 ± 0.021 | 0.410 ± 0.029 | 0.184 ± 0.756 |
| bin_matmul_1024 | 62.855 ± 0.461 | 52.956 ± 0.434 | **35.625 ± 0.168** | 36.040 ± 0.553 | 51.503 ± 8.823 |
| bin_matmul_256 | 1.284 ± 0.024 | 0.954 ± 0.165 | 0.855 ± 0.024 | **0.850 ± 0.081** | 0.922 ± 0.019 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.001 | **0.010 ± 0.000** | 0.057 ± 0.011 | 0.204 ± 0.026 | 0.014 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.069 ± 0.004 | 0.065 ± 0.004 | 0.225 ± 0.026 | 0.245 ± 0.031 | **0.052 ± 0.003** |
| bin_omeinsum_matmul_10x10 | 0.074 ± 0.002 | 0.048 ± 0.003 | 0.047 ± 0.003 | 0.146 ± 0.021 | **0.037 ± 0.004** |
| bin_outer_product_4096 | 22.611 ± 0.606 | 22.194 ± 0.343 | **18.773 ± 0.473** | 28.181 ± 0.802 | 304.696 ± 36.619 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 60.829 ± 1.423 | 54.532 ± 0.449 | **30.368 ± 0.676** | 39.840 ± 4.767 | 45.624 ± 12.929 |
| gm_queen5_5_3.wcsp | 6616.688 ± 142.425 | 6060.282 ± 46.776 | **2702.861 ± 35.641** | 5401.554 ± 93.673 | 8907.885 ± 172.000 |
| lm_batch_likelihood_brackets_4_4d | 30.721 ± 0.697 | 52.847 ± 3.428 | **17.084 ± 0.874** | 23.870 ± 1.766 | 45.033 ± 15.256 |
| lm_batch_likelihood_sentence_3_12d | 64.086 ± 1.054 | 166.470 ± 15.638 | 57.686 ± 1.405 | **57.543 ± 6.848** | 75.639 ± 22.343 |
| lm_batch_likelihood_sentence_4_4d | 33.108 ± 0.534 | 58.901 ± 2.632 | 74.324 ± 33.669 | **23.852 ± 1.670** | 51.296 ± 19.432 |
| nary_matmul_chain_64 | 0.149 ± 0.005 | 0.137 ± 0.007 | **0.096 ± 0.023** | 0.195 ± 0.010 | 0.147 ± 0.057 |
| str_matrix_chain_multiplication_100 | 18.185 ± 0.147 | 18.091 ± 0.117 | **11.214 ± 1.375** | 15.402 ± 0.969 | 15.516 ± 11.398 |
| str_mps_varying_inner_product_200 | 27.602 ± 0.161 | 28.777 ± 0.203 | **26.017 ± 0.239** | 30.976 ± 1.412 | 31.659 ± 4.198 |
| str_nw_mera_closed_120 | 1702.311 ± 21.346 | 1910.402 ± 30.486 | **951.391 ± 8.999** | 1180.110 ± 21.508 | 1271.079 ± 81.815 |
| str_nw_mera_open_26 | 1347.389 ± 8.596 | 1362.307 ± 10.724 | **594.494 ± 12.307** | 952.580 ± 14.215 | 1297.891 ± 22.979 |
| tensornetwork_permutation_focus_step409_316 | 477.314 ± 4.336 | 511.319 ± 15.269 | **346.494 ± 14.093** | 416.462 ± 11.519 | 584.655 ± 50.444 |
| tensornetwork_permutation_light_415 | 472.486 ± 4.716 | 467.359 ± 10.518 | **343.052 ± 3.523** | 437.318 ± 13.183 | 575.118 ± 39.132 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 9.978 ± 0.191 | 5.300 ± 0.067 | **2.320 ± 3.186** | 3.055 ± 0.615 | 13.044 ± 4.091 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.980 ± 0.038 | 1.376 ± 0.034 | **0.377 ± 0.031** | 0.855 ± 0.015 | 0.903 ± 0.112 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.320 ± 0.014 | 0.313 ± 0.032 | **0.207 ± 0.015** | 0.464 ± 0.021 | 3.801 ± 3.352 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.289 ± 0.013 | 0.306 ± 0.023 | **0.205 ± 0.012** | 0.469 ± 0.033 | 6.205 ± 3.242 |
| bin_elementwise_mul_2048x2048 | 1.048 ± 0.021 | **0.900 ± 0.056** | 5.767 ± 0.202 | 9.622 ± 0.182 | 20.289 ± 0.849 |
| bin_elementwise_mul_512x512 | 0.145 ± 0.012 | **0.140 ± 0.010** | 0.161 ± 0.021 | 0.410 ± 0.029 | 0.184 ± 0.756 |
| bin_matmul_1024 | 62.855 ± 0.461 | 52.956 ± 0.434 | **35.625 ± 0.168** | 36.040 ± 0.553 | 51.503 ± 8.823 |
| bin_matmul_256 | 1.284 ± 0.024 | 0.954 ± 0.165 | 0.855 ± 0.024 | **0.850 ± 0.081** | 0.922 ± 0.019 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.001 | **0.010 ± 0.000** | 0.057 ± 0.011 | 0.204 ± 0.026 | 0.014 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.069 ± 0.004 | 0.065 ± 0.004 | 0.225 ± 0.026 | 0.245 ± 0.031 | **0.052 ± 0.003** |
| bin_omeinsum_matmul_10x10 | 0.074 ± 0.002 | 0.048 ± 0.003 | 0.047 ± 0.003 | 0.146 ± 0.021 | **0.037 ± 0.004** |
| bin_outer_product_4096 | 22.611 ± 0.606 | 22.194 ± 0.343 | **18.773 ± 0.473** | 28.181 ± 0.802 | 304.696 ± 36.619 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 60.829 ± 1.423 | 54.532 ± 0.449 | **30.368 ± 0.676** | 39.840 ± 4.767 | 45.624 ± 12.929 |
| gm_queen5_5_3.wcsp | 2398.306 ± 39.919 | 2114.098 ± 36.614 | **1138.754 ± 15.728** | 1889.880 ± 38.340 | 3152.891 ± 46.822 |
| lm_batch_likelihood_brackets_4_4d | 31.515 ± 0.263 | 55.436 ± 0.805 | **17.996 ± 0.332** | 24.341 ± 1.355 | 53.334 ± 35.850 |
| lm_batch_likelihood_sentence_3_12d | 58.998 ± 1.097 | 231.521 ± 15.349 | **33.874 ± 1.064** | 56.631 ± 9.266 | 115.507 ± 65.968 |
| lm_batch_likelihood_sentence_4_4d | 32.622 ± 0.543 | 63.053 ± 3.243 | **18.795 ± 0.736** | 24.549 ± 1.679 | 48.912 ± 29.842 |
| nary_matmul_chain_64 | 0.149 ± 0.005 | 0.137 ± 0.007 | **0.096 ± 0.023** | 0.195 ± 0.010 | 0.147 ± 0.057 |
| str_matrix_chain_multiplication_100 | 18.801 ± 0.331 | 19.239 ± 0.304 | **10.902 ± 0.160** | 15.481 ± 0.998 | 15.157 ± 0.470 |
| str_mps_varying_inner_product_200 | **29.269 ± 0.467** | 32.138 ± 0.397 | 29.553 ± 1.428 | 32.253 ± 1.014 | 30.673 ± 8.225 |
| str_nw_mera_closed_120 | 1558.928 ± 8.432 | 1567.427 ± 23.001 | **762.763 ± 9.459** | 997.720 ± 49.826 | 972.049 ± 61.389 |
| str_nw_mera_open_26 | 1357.055 ± 11.889 | 1375.158 ± 15.680 | **605.315 ± 13.242** | 983.098 ± 28.390 | 1263.360 ± 47.150 |
| tensornetwork_permutation_focus_step409_316 | 477.314 ± 4.336 | 511.319 ± 15.269 | **346.494 ± 14.093** | 416.462 ± 11.519 | 584.655 ± 50.444 |
| tensornetwork_permutation_light_415 | 472.486 ± 4.716 | 467.359 ± 10.518 | **343.052 ± 3.523** | 437.318 ± 13.183 | 575.118 ± 39.132 |

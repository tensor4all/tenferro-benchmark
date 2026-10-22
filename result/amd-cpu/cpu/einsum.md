# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`
- Benchmark commit: `12e0b50f983ca3253ff46841eb15912448ae3e20`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261022_180452/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261022_180452/run.yaml`
- Timestamp: `20261022_180452`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261022_180452`.

- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`

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


- Source table: `data/results/amd-cpu/cpu/einsum/20261022_180452/einsum_table_t1_20261022_180452.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20261022_180452/tenferro_trace_t1_20261022_180452.log`
- `data/results/amd-cpu/cpu/einsum/20261022_180452/tenferro_eager_t1_20261022_180452.log`
- `data/results/amd-cpu/cpu/einsum/20261022_180452/pytorch_cpu_t1_20261022_180452.log`
- `data/results/amd-cpu/cpu/einsum/20261022_180452/jax_cpu_t1_20261022_180452.log`
- `data/results/amd-cpu/cpu/einsum/20261022_180452/julia_omeinsum_t1_20261022_180452.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.527 ± 0.092 | **4.207 ± 0.049** | 4.618 ± 0.266 | 4.732 ± 0.080 | 7.132 ± 5.304 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.848 ± 0.079 | **0.637 ± 0.023** | 0.706 ± 0.027 | 0.845 ± 0.038 | 0.871 ± 0.073 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.215 ± 0.019** | 0.217 ± 0.037 | 0.258 ± 0.034 | 0.583 ± 0.028 | 0.709 ± 0.403 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.417 ± 0.028 | **0.206 ± 0.044** | 0.219 ± 0.021 | 0.597 ± 0.032 | 2.419 ± 5.233 |
| bin_elementwise_mul_2048x2048 | 1.881 ± 0.065 | **1.798 ± 0.031** | 15.602 ± 0.340 | 5.485 ± 0.161 | 5.334 ± 0.354 |
| bin_elementwise_mul_512x512 | **0.123 ± 0.017** | **0.123 ± 0.010** | 0.156 ± 0.024 | 0.406 ± 0.043 | 0.136 ± 0.018 |
| bin_matmul_1024 | 64.446 ± 0.316 | **49.693 ± 0.336** | 81.030 ± 0.378 | 65.136 ± 1.364 | 75.381 ± 9.128 |
| bin_matmul_256 | 1.097 ± 0.027 | **0.835 ± 0.014** | 1.404 ± 0.055 | 1.183 ± 0.039 | 2.161 ± 0.076 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.011 ± 0.000 | **0.006 ± 0.000** | 0.037 ± 0.003 | 0.205 ± 0.028 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.045 ± 0.001 | **0.040 ± 0.007** | 0.148 ± 0.007 | 0.239 ± 0.043 | 0.090 ± 0.028 |
| bin_omeinsum_matmul_10x10 | 0.026 ± 0.001 | **0.018 ± 0.001** | 0.037 ± 0.002 | 0.198 ± 0.051 | 0.025 ± 0.001 |
| bin_outer_product_4096 | 10.591 ± 0.303 | **9.849 ± 0.453** | 51.991 ± 15.491 | 10.485 ± 0.191 | 942.753 ± 3.595 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 63.148 ± 0.513 | **52.577 ± 0.563** | 61.640 ± 0.958 | 68.702 ± 2.898 | 72.040 ± 26.581 |
| gm_queen5_5_3.wcsp | 5947.689 ± 77.133 | 4890.597 ± 25.461 | **4080.374 ± 930.495** | 5925.622 ± 208.427 | 5575.400 ± 87.381 |
| lm_batch_likelihood_brackets_4_4d | 23.691 ± 0.426 | 37.061 ± 2.150 | 36.167 ± 0.736 | **16.512 ± 0.418** | 34.267 ± 15.164 |
| lm_batch_likelihood_sentence_3_12d | **57.669 ± 0.590** | 104.983 ± 2.062 | 68.876 ± 1.742 | 72.221 ± 1.447 | 107.090 ± 6.938 |
| lm_batch_likelihood_sentence_4_4d | 25.553 ± 0.523 | 40.454 ± 2.089 | 37.748 ± 0.489 | **17.462 ± 0.376** | 65.942 ± 26.333 |
| nary_matmul_chain_64 | 0.060 ± 0.037 | **0.046 ± 0.016** | 0.081 ± 0.004 | 0.213 ± 0.013 | 0.113 ± 0.027 |
| str_matrix_chain_multiplication_100 | **14.347 ± 0.121** | 15.274 ± 0.197 | 15.895 ± 0.109 | 15.367 ± 0.207 | 16.662 ± 0.364 |
| str_mps_varying_inner_product_200 | **20.515 ± 0.091** | 22.256 ± 0.165 | 25.645 ± 0.273 | 24.382 ± 0.264 | 26.351 ± 6.246 |
| str_nw_mera_closed_120 | **1344.896 ± 9.408** | 1386.752 ± 14.652 | 1420.058 ± 6.308 | 1914.548 ± 9.457 | 1360.937 ± 25.340 |
| str_nw_mera_open_26 | 924.435 ± 6.070 | 1052.878 ± 6.427 | **872.245 ± 5.087** | 1431.723 ± 6.895 | 1168.643 ± 57.024 |
| tensornetwork_permutation_focus_step409_316 | **311.999 ± 2.521** | 356.686 ± 1.855 | 749.258 ± 1.853 | 1573.229 ± 6.945 | 422.250 ± 4.162 |
| tensornetwork_permutation_light_415 | **301.137 ± 2.060** | 317.702 ± 1.864 | 712.478 ± 2.110 | 1594.894 ± 8.347 | 417.249 ± 84.214 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.527 ± 0.092 | **4.207 ± 0.049** | 4.618 ± 0.266 | 4.732 ± 0.080 | 7.132 ± 5.304 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.848 ± 0.079 | **0.637 ± 0.023** | 0.706 ± 0.027 | 0.845 ± 0.038 | 0.871 ± 0.073 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.215 ± 0.019** | 0.217 ± 0.037 | 0.258 ± 0.034 | 0.583 ± 0.028 | 0.709 ± 0.403 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.417 ± 0.028 | **0.206 ± 0.044** | 0.219 ± 0.021 | 0.597 ± 0.032 | 2.419 ± 5.233 |
| bin_elementwise_mul_2048x2048 | 1.881 ± 0.065 | **1.798 ± 0.031** | 15.602 ± 0.340 | 5.485 ± 0.161 | 5.334 ± 0.354 |
| bin_elementwise_mul_512x512 | **0.123 ± 0.017** | **0.123 ± 0.010** | 0.156 ± 0.024 | 0.406 ± 0.043 | 0.136 ± 0.018 |
| bin_matmul_1024 | 64.446 ± 0.316 | **49.693 ± 0.336** | 81.030 ± 0.378 | 65.136 ± 1.364 | 75.381 ± 9.128 |
| bin_matmul_256 | 1.097 ± 0.027 | **0.835 ± 0.014** | 1.404 ± 0.055 | 1.183 ± 0.039 | 2.161 ± 0.076 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.011 ± 0.000 | **0.006 ± 0.000** | 0.037 ± 0.003 | 0.205 ± 0.028 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.045 ± 0.001 | **0.040 ± 0.007** | 0.148 ± 0.007 | 0.239 ± 0.043 | 0.090 ± 0.028 |
| bin_omeinsum_matmul_10x10 | 0.026 ± 0.001 | **0.018 ± 0.001** | 0.037 ± 0.002 | 0.198 ± 0.051 | 0.025 ± 0.001 |
| bin_outer_product_4096 | 10.591 ± 0.303 | **9.849 ± 0.453** | 51.991 ± 15.491 | 10.485 ± 0.191 | 942.753 ± 3.595 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 63.148 ± 0.513 | **52.577 ± 0.563** | 61.640 ± 0.958 | 68.702 ± 2.898 | 72.040 ± 26.581 |
| gm_queen5_5_3.wcsp | 1810.610 ± 12.061 | **1649.245 ± 17.433** | 1683.578 ± 38.884 | 1927.618 ± 12.769 | 1805.568 ± 8.312 |
| lm_batch_likelihood_brackets_4_4d | 23.860 ± 0.501 | 35.695 ± 1.132 | 37.600 ± 0.602 | **16.288 ± 0.443** | 46.471 ± 33.482 |
| lm_batch_likelihood_sentence_3_12d | **56.798 ± 0.657** | 106.058 ± 1.531 | 69.251 ± 1.730 | 78.097 ± 2.815 | 111.592 ± 36.161 |
| lm_batch_likelihood_sentence_4_4d | 25.295 ± 0.698 | 42.191 ± 0.917 | 39.083 ± 0.882 | **16.833 ± 0.202** | 48.380 ± 38.794 |
| nary_matmul_chain_64 | 0.060 ± 0.037 | **0.046 ± 0.016** | 0.081 ± 0.004 | 0.213 ± 0.013 | 0.113 ± 0.027 |
| str_matrix_chain_multiplication_100 | **15.147 ± 0.163** | 15.634 ± 0.108 | 15.821 ± 0.202 | 15.225 ± 0.250 | 16.472 ± 0.392 |
| str_mps_varying_inner_product_200 | 24.558 ± 0.357 | **24.189 ± 0.187** | 26.410 ± 0.353 | 25.290 ± 0.449 | 25.090 ± 9.226 |
| str_nw_mera_closed_120 | **1220.164 ± 4.925** | 1237.659 ± 7.426 | 1265.069 ± 4.858 | 1737.336 ± 6.098 | 1267.451 ± 82.955 |
| str_nw_mera_open_26 | 929.195 ± 5.726 | 1054.031 ± 4.513 | **910.851 ± 3.527** | 1469.142 ± 5.861 | 1138.363 ± 81.827 |
| tensornetwork_permutation_focus_step409_316 | **311.999 ± 2.521** | 356.686 ± 1.855 | 749.258 ± 1.853 | 1573.229 ± 6.945 | 422.250 ± 4.162 |
| tensornetwork_permutation_light_415 | **301.137 ± 2.060** | 317.702 ± 1.864 | 712.478 ± 2.110 | 1594.894 ± 8.347 | 417.249 ± 84.214 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261022_183841/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261022_183841/run.yaml`
- Timestamp: `20261022_183841`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261022_183841`.

- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`

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


- Source table: `data/results/amd-cpu/cpu/einsum/20261022_183841/einsum_table_t4_20261022_183841.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20261022_183841/tenferro_trace_t4_20261022_183841.log`
- `data/results/amd-cpu/cpu/einsum/20261022_183841/tenferro_eager_t4_20261022_183841.log`
- `data/results/amd-cpu/cpu/einsum/20261022_183841/pytorch_cpu_t4_20261022_183841.log`
- `data/results/amd-cpu/cpu/einsum/20261022_183841/jax_cpu_t4_20261022_183841.log`
- `data/results/amd-cpu/cpu/einsum/20261022_183841/julia_omeinsum_t4_20261022_183841.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.358 ± 0.040 | 2.808 ± 0.035 | **2.295 ± 1.058** | 2.848 ± 0.343 | 10.635 ± 6.346 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.907 ± 0.032 | 0.668 ± 0.031 | **0.383 ± 0.010** | 0.797 ± 0.024 | 0.887 ± 0.126 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.276 ± 0.030 | 0.254 ± 0.019 | **0.204 ± 0.008** | 0.589 ± 0.021 | 0.800 ± 0.201 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.246 ± 0.018 | 0.253 ± 0.018 | **0.201 ± 0.019** | 0.596 ± 0.020 | 3.083 ± 3.877 |
| bin_elementwise_mul_2048x2048 | **1.477 ± 0.018** | 1.761 ± 0.037 | 4.243 ± 0.136 | 4.291 ± 0.420 | 5.071 ± 0.312 |
| bin_elementwise_mul_512x512 | 0.151 ± 0.017 | **0.115 ± 0.009** | 0.125 ± 0.019 | 0.403 ± 0.032 | 0.159 ± 0.046 |
| bin_matmul_1024 | 32.161 ± 0.224 | **24.243 ± 0.190** | 51.735 ± 0.611 | 36.566 ± 0.218 | 58.638 ± 18.586 |
| bin_matmul_256 | 0.647 ± 0.034 | **0.456 ± 0.018** | 0.931 ± 0.009 | 0.872 ± 0.067 | 0.951 ± 0.021 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.016 ± 0.001 | **0.009 ± 0.001** | 0.052 ± 0.004 | 0.207 ± 0.022 | 0.013 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.069 ± 0.003 | 0.064 ± 0.001 | 0.223 ± 0.021 | 0.262 ± 0.018 | **0.051 ± 0.024** |
| bin_omeinsum_matmul_10x10 | 0.043 ± 0.002 | **0.018 ± 0.000** | 0.048 ± 0.001 | 0.151 ± 0.024 | 0.024 ± 0.001 |
| bin_outer_product_4096 | 7.955 ± 5.913 | 7.269 ± 0.146 | **5.522 ± 0.159** | 8.125 ± 0.361 | 934.689 ± 4.060 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 30.581 ± 1.254 | **25.644 ± 0.572** | 32.034 ± 5.086 | 38.314 ± 0.944 | 48.107 ± 12.109 |
| gm_queen5_5_3.wcsp | 4524.184 ± 128.969 | 3600.396 ± 66.626 | **1636.932 ± 10.820** | 3575.136 ± 30.678 | 4591.759 ± 157.358 |
| lm_batch_likelihood_brackets_4_4d | 22.795 ± 17.159 | 42.353 ± 2.262 | **16.812 ± 0.466** | 22.515 ± 1.125 | 36.242 ± 4.890 |
| lm_batch_likelihood_sentence_3_12d | **36.323 ± 5.637** | 91.962 ± 10.727 | 39.654 ± 0.371 | 44.168 ± 1.575 | 100.137 ± 32.583 |
| lm_batch_likelihood_sentence_4_4d | 25.061 ± 0.722 | 45.472 ± 3.422 | **17.873 ± 0.344** | 23.063 ± 0.759 | 50.974 ± 35.731 |
| nary_matmul_chain_64 | 0.070 ± 0.003 | **0.056 ± 0.001** | 0.094 ± 0.003 | 0.216 ± 0.026 | 0.144 ± 0.040 |
| str_matrix_chain_multiplication_100 | **9.004 ± 0.190** | 9.490 ± 0.091 | 10.996 ± 0.084 | 15.302 ± 0.877 | 14.444 ± 0.653 |
| str_mps_varying_inner_product_200 | **23.489 ± 0.205** | 25.942 ± 1.527 | 26.217 ± 0.607 | 31.908 ± 0.858 | 30.902 ± 7.611 |
| str_nw_mera_closed_120 | **674.903 ± 34.090** | 691.690 ± 28.527 | 684.782 ± 4.315 | 913.590 ± 15.393 | 821.497 ± 12.199 |
| str_nw_mera_open_26 | 524.552 ± 19.093 | 580.207 ± 21.519 | **449.941 ± 2.213** | 683.654 ± 7.917 | 851.007 ± 16.942 |
| tensornetwork_permutation_focus_step409_316 | 217.698 ± 38.025 | **215.404 ± 18.161** | 270.011 ± 1.780 | 789.156 ± 16.838 | 426.698 ± 19.137 |
| tensornetwork_permutation_light_415 | **210.313 ± 29.825** | 219.770 ± 39.495 | 266.842 ± 1.548 | 805.404 ± 9.053 | 422.284 ± 72.324 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.358 ± 0.040 | 2.808 ± 0.035 | **2.295 ± 1.058** | 2.848 ± 0.343 | 10.635 ± 6.346 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.907 ± 0.032 | 0.668 ± 0.031 | **0.383 ± 0.010** | 0.797 ± 0.024 | 0.887 ± 0.126 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.276 ± 0.030 | 0.254 ± 0.019 | **0.204 ± 0.008** | 0.589 ± 0.021 | 0.800 ± 0.201 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.246 ± 0.018 | 0.253 ± 0.018 | **0.201 ± 0.019** | 0.596 ± 0.020 | 3.083 ± 3.877 |
| bin_elementwise_mul_2048x2048 | **1.477 ± 0.018** | 1.761 ± 0.037 | 4.243 ± 0.136 | 4.291 ± 0.420 | 5.071 ± 0.312 |
| bin_elementwise_mul_512x512 | 0.151 ± 0.017 | **0.115 ± 0.009** | 0.125 ± 0.019 | 0.403 ± 0.032 | 0.159 ± 0.046 |
| bin_matmul_1024 | 32.161 ± 0.224 | **24.243 ± 0.190** | 51.735 ± 0.611 | 36.566 ± 0.218 | 58.638 ± 18.586 |
| bin_matmul_256 | 0.647 ± 0.034 | **0.456 ± 0.018** | 0.931 ± 0.009 | 0.872 ± 0.067 | 0.951 ± 0.021 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.016 ± 0.001 | **0.009 ± 0.001** | 0.052 ± 0.004 | 0.207 ± 0.022 | 0.013 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.069 ± 0.003 | 0.064 ± 0.001 | 0.223 ± 0.021 | 0.262 ± 0.018 | **0.051 ± 0.024** |
| bin_omeinsum_matmul_10x10 | 0.043 ± 0.002 | **0.018 ± 0.000** | 0.048 ± 0.001 | 0.151 ± 0.024 | 0.024 ± 0.001 |
| bin_outer_product_4096 | 7.955 ± 5.913 | 7.269 ± 0.146 | **5.522 ± 0.159** | 8.125 ± 0.361 | 934.689 ± 4.060 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 30.581 ± 1.254 | **25.644 ± 0.572** | 32.034 ± 5.086 | 38.314 ± 0.944 | 48.107 ± 12.109 |
| gm_queen5_5_3.wcsp | 1176.695 ± 65.261 | 1082.143 ± 66.271 | **738.084 ± 5.100** | 1179.666 ± 29.978 | 1506.764 ± 24.681 |
| lm_batch_likelihood_brackets_4_4d | 22.889 ± 1.487 | 41.170 ± 1.147 | **17.917 ± 0.403** | 22.718 ± 1.865 | 50.162 ± 33.194 |
| lm_batch_likelihood_sentence_3_12d | 43.936 ± 6.988 | 95.380 ± 26.638 | **33.112 ± 0.464** | 48.626 ± 3.354 | 96.471 ± 27.440 |
| lm_batch_likelihood_sentence_4_4d | 23.558 ± 2.880 | 46.457 ± 2.864 | **18.393 ± 0.501** | 23.322 ± 1.165 | 62.101 ± 24.935 |
| nary_matmul_chain_64 | 0.070 ± 0.003 | **0.056 ± 0.001** | 0.094 ± 0.003 | 0.216 ± 0.026 | 0.144 ± 0.040 |
| str_matrix_chain_multiplication_100 | **9.252 ± 0.102** | 9.448 ± 0.126 | 11.038 ± 0.109 | 15.032 ± 1.512 | 14.200 ± 1.323 |
| str_mps_varying_inner_product_200 | 27.728 ± 0.196 | 28.631 ± 0.886 | 28.288 ± 0.306 | 32.257 ± 1.043 | **24.045 ± 14.371** |
| str_nw_mera_closed_120 | **617.565 ± 22.655** | 617.599 ± 29.219 | 618.833 ± 4.335 | 842.786 ± 6.040 | 733.819 ± 40.863 |
| str_nw_mera_open_26 | 521.225 ± 13.698 | 578.331 ± 24.050 | **462.403 ± 2.063** | 722.821 ± 24.257 | 845.395 ± 26.718 |
| tensornetwork_permutation_focus_step409_316 | 217.698 ± 38.025 | **215.404 ± 18.161** | 270.011 ± 1.780 | 789.156 ± 16.838 | 426.698 ± 19.137 |
| tensornetwork_permutation_light_415 | **210.313 ± 29.825** | 219.770 ± 39.495 | 266.842 ± 1.548 | 805.404 ± 9.053 | 422.284 ± 72.324 |

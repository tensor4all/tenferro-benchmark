# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `14bc1ab933e3c7a9c966e08355670331ab9d993c`
- Benchmark commit: `ce89bccec28bc7c863b30c80764d3570747f6df6`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261019_175954/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261019_175954/run.yaml`
- Timestamp: `20261019_175954`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261019_175954`.

- tenferro-rs commit: `14bc1ab933e3c7a9c966e08355670331ab9d993c`

### CPU Information

- Model: `AMD EPYC 9V45 96-Core Processor`
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


- Source table: `data/results/amd-cpu/cpu/einsum/20261019_175954/einsum_table_t1_20261019_175954.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20261019_175954/tenferro_trace_t1_20261019_175954.log`
- `data/results/amd-cpu/cpu/einsum/20261019_175954/tenferro_eager_t1_20261019_175954.log`
- `data/results/amd-cpu/cpu/einsum/20261019_175954/pytorch_cpu_t1_20261019_175954.log`
- `data/results/amd-cpu/cpu/einsum/20261019_175954/jax_cpu_t1_20261019_175954.log`
- `data/results/amd-cpu/cpu/einsum/20261019_175954/julia_omeinsum_t1_20261019_175954.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.334 ± 0.168 | 2.768 ± 0.171 | 3.179 ± 0.106 | **2.478 ± 0.570** | 4.818 ± 5.456 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.500 ± 0.087 | **0.399 ± 0.021** | 0.422 ± 0.061 | 0.427 ± 0.017 | **0.399 ± 1.182** |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.131 ± 0.016 | 0.127 ± 0.016 | **0.088 ± 0.009** | 0.262 ± 0.019 | 0.894 ± 0.604 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.232 ± 0.020 | 0.123 ± 0.006 | **0.088 ± 0.003** | 0.505 ± 0.118 | 2.442 ± 1.127 |
| bin_elementwise_mul_2048x2048 | 1.209 ± 0.027 | **1.177 ± 0.019** | 3.362 ± 0.105 | 4.664 ± 0.043 | 3.772 ± 0.156 |
| bin_elementwise_mul_512x512 | 0.069 ± 0.002 | **0.061 ± 0.003** | 0.068 ± 0.012 | 0.175 ± 0.035 | 0.073 ± 0.544 |
| bin_matmul_1024 | 54.367 ± 2.643 | 36.806 ± 1.606 | 62.176 ± 0.699 | **35.965 ± 0.933** | 38.210 ± 13.072 |
| bin_matmul_256 | 0.732 ± 0.021 | 0.564 ± 0.024 | 0.840 ± 0.009 | 0.621 ± 0.022 | **0.535 ± 0.007** |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.003 ± 0.000** | 0.021 ± 0.007 | 0.078 ± 0.008 | 0.007 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.023 ± 0.000 | **0.021 ± 0.000** | 0.067 ± 0.001 | 0.114 ± 0.019 | 0.056 ± 0.027 |
| bin_omeinsum_matmul_10x10 | 0.013 ± 0.001 | **0.006 ± 0.000** | 0.013 ± 0.000 | 0.048 ± 0.008 | 0.013 ± 0.002 |
| bin_outer_product_4096 | 7.562 ± 0.348 | **7.066 ± 6.073** | 7.635 ± 0.156 | 7.441 ± 0.142 | 381.460 ± 13.922 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 52.563 ± 1.903 | 37.495 ± 1.243 | 47.682 ± 1.492 | 37.190 ± 1.669 | **33.872 ± 15.816** |
| gm_queen5_5_3.wcsp | 4759.231 ± 147.162 | 3660.659 ± 109.729 | 3299.698 ± 390.491 | **2113.115 ± 21.479** | 4039.770 ± 262.407 |
| lm_batch_likelihood_brackets_4_4d | 13.871 ± 0.598 | 19.599 ± 0.965 | 17.121 ± 0.323 | **8.808 ± 0.475** | 14.885 ± 3.425 |
| lm_batch_likelihood_sentence_3_12d | 39.875 ± 1.105 | 68.049 ± 3.205 | 51.584 ± 1.136 | **39.814 ± 0.676** | 69.745 ± 30.238 |
| lm_batch_likelihood_sentence_4_4d | 15.231 ± 0.594 | 22.219 ± 1.456 | 19.598 ± 0.605 | **10.272 ± 0.631** | 20.249 ± 37.241 |
| nary_matmul_chain_64 | 0.034 ± 0.004 | **0.026 ± 0.000** | 0.047 ± 0.009 | 0.084 ± 0.007 | 0.037 ± 0.036 |
| str_matrix_chain_multiplication_100 | **9.164 ± 0.107** | 9.550 ± 0.235 | 10.385 ± 0.193 | 9.737 ± 0.486 | 10.056 ± 0.621 |
| str_mps_varying_inner_product_200 | **12.874 ± 0.130** | 14.332 ± 0.494 | 15.286 ± 0.377 | 14.993 ± 0.758 | 17.140 ± 32.919 |
| str_nw_mera_closed_120 | 965.749 ± 28.254 | 980.203 ± 30.599 | 968.998 ± 40.921 | 1023.767 ± 14.603 | **720.239 ± 18.691** |
| str_nw_mera_open_26 | 694.981 ± 22.497 | 766.153 ± 6.612 | **625.623 ± 48.503** | 833.179 ± 10.440 | 738.429 ± 22.913 |
| tensornetwork_permutation_focus_step409_316 | **217.792 ± 1.476** | 288.224 ± 7.645 | 473.368 ± 10.059 | 1336.468 ± 72.316 | 272.811 ± 38.822 |
| tensornetwork_permutation_light_415 | **212.069 ± 3.937** | 221.212 ± 3.392 | 461.399 ± 6.665 | 1358.708 ± 24.231 | 248.168 ± 37.191 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.334 ± 0.168 | 2.768 ± 0.171 | 3.179 ± 0.106 | **2.478 ± 0.570** | 4.818 ± 5.456 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.500 ± 0.087 | **0.399 ± 0.021** | 0.422 ± 0.061 | 0.427 ± 0.017 | **0.399 ± 1.182** |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.131 ± 0.016 | 0.127 ± 0.016 | **0.088 ± 0.009** | 0.262 ± 0.019 | 0.894 ± 0.604 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.232 ± 0.020 | 0.123 ± 0.006 | **0.088 ± 0.003** | 0.505 ± 0.118 | 2.442 ± 1.127 |
| bin_elementwise_mul_2048x2048 | 1.209 ± 0.027 | **1.177 ± 0.019** | 3.362 ± 0.105 | 4.664 ± 0.043 | 3.772 ± 0.156 |
| bin_elementwise_mul_512x512 | 0.069 ± 0.002 | **0.061 ± 0.003** | 0.068 ± 0.012 | 0.175 ± 0.035 | 0.073 ± 0.544 |
| bin_matmul_1024 | 54.367 ± 2.643 | 36.806 ± 1.606 | 62.176 ± 0.699 | **35.965 ± 0.933** | 38.210 ± 13.072 |
| bin_matmul_256 | 0.732 ± 0.021 | 0.564 ± 0.024 | 0.840 ± 0.009 | 0.621 ± 0.022 | **0.535 ± 0.007** |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.003 ± 0.000** | 0.021 ± 0.007 | 0.078 ± 0.008 | 0.007 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.023 ± 0.000 | **0.021 ± 0.000** | 0.067 ± 0.001 | 0.114 ± 0.019 | 0.056 ± 0.027 |
| bin_omeinsum_matmul_10x10 | 0.013 ± 0.001 | **0.006 ± 0.000** | 0.013 ± 0.000 | 0.048 ± 0.008 | 0.013 ± 0.002 |
| bin_outer_product_4096 | 7.562 ± 0.348 | **7.066 ± 6.073** | 7.635 ± 0.156 | 7.441 ± 0.142 | 381.460 ± 13.922 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 52.563 ± 1.903 | 37.495 ± 1.243 | 47.682 ± 1.492 | 37.190 ± 1.669 | **33.872 ± 15.816** |
| gm_queen5_5_3.wcsp | 1346.971 ± 46.654 | 1337.454 ± 32.435 | 1433.791 ± 72.140 | **882.717 ± 24.066** | 1361.002 ± 29.544 |
| lm_batch_likelihood_brackets_4_4d | 13.002 ± 0.571 | 20.543 ± 0.637 | 18.274 ± 0.344 | **9.546 ± 0.245** | 25.355 ± 31.507 |
| lm_batch_likelihood_sentence_3_12d | 42.017 ± 1.526 | 77.833 ± 0.818 | **41.398 ± 1.161** | 48.606 ± 1.292 | 79.031 ± 32.828 |
| lm_batch_likelihood_sentence_4_4d | 14.511 ± 1.548 | 24.716 ± 2.741 | 19.528 ± 0.448 | **9.464 ± 0.546** | 23.404 ± 32.955 |
| nary_matmul_chain_64 | 0.034 ± 0.004 | **0.026 ± 0.000** | 0.047 ± 0.009 | 0.084 ± 0.007 | 0.037 ± 0.036 |
| str_matrix_chain_multiplication_100 | **9.351 ± 0.171** | 9.904 ± 0.169 | 10.288 ± 0.303 | 9.384 ± 0.132 | 10.054 ± 0.510 |
| str_mps_varying_inner_product_200 | 14.208 ± 0.194 | 14.197 ± 0.423 | **13.410 ± 0.425** | 13.779 ± 0.162 | 15.145 ± 8.969 |
| str_nw_mera_closed_120 | 911.027 ± 28.834 | 914.354 ± 14.380 | 875.760 ± 15.376 | 938.037 ± 6.457 | **716.999 ± 43.651** |
| str_nw_mera_open_26 | 716.799 ± 19.601 | 761.628 ± 2.498 | **650.295 ± 13.973** | 846.506 ± 24.468 | 733.350 ± 41.793 |
| tensornetwork_permutation_focus_step409_316 | **217.792 ± 1.476** | 288.224 ± 7.645 | 473.368 ± 10.059 | 1336.468 ± 72.316 | 272.811 ± 38.822 |
| tensornetwork_permutation_light_415 | **212.069 ± 3.937** | 221.212 ± 3.392 | 461.399 ± 6.665 | 1358.708 ± 24.231 | 248.168 ± 37.191 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261019_182327/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261019_182327/run.yaml`
- Timestamp: `20261019_182327`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261019_182327`.

- tenferro-rs commit: `14bc1ab933e3c7a9c966e08355670331ab9d993c`

### CPU Information

- Model: `AMD EPYC 9V45 96-Core Processor`
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


- Source table: `data/results/amd-cpu/cpu/einsum/20261019_182327/einsum_table_t4_20261019_182327.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20261019_182327/tenferro_trace_t4_20261019_182327.log`
- `data/results/amd-cpu/cpu/einsum/20261019_182327/tenferro_eager_t4_20261019_182327.log`
- `data/results/amd-cpu/cpu/einsum/20261019_182327/pytorch_cpu_t4_20261019_182327.log`
- `data/results/amd-cpu/cpu/einsum/20261019_182327/jax_cpu_t4_20261019_182327.log`
- `data/results/amd-cpu/cpu/einsum/20261019_182327/julia_omeinsum_t4_20261019_182327.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.634 ± 0.075 | **1.620 ± 0.035** | 1.885 ± 2.621 | 1.957 ± 0.260 | 6.833 ± 5.950 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.500 ± 0.019 | 0.356 ± 0.019 | **0.224 ± 0.002** | 0.429 ± 0.036 | 0.451 ± 1.237 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.141 ± 0.014 | 0.138 ± 0.007 | **0.069 ± 0.003** | 0.243 ± 0.010 | 0.334 ± 0.253 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.208 ± 0.016 | 0.137 ± 0.008 | **0.066 ± 0.002** | 0.256 ± 0.015 | 3.015 ± 2.602 |
| bin_elementwise_mul_2048x2048 | **0.849 ± 0.027** | 1.076 ± 0.021 | 2.569 ± 0.167 | 3.604 ± 0.095 | 4.075 ± 0.129 |
| bin_elementwise_mul_512x512 | 0.072 ± 0.008 | 0.081 ± 0.018 | **0.061 ± 0.020** | 0.185 ± 0.012 | 0.084 ± 0.590 |
| bin_matmul_1024 | 27.265 ± 0.660 | **19.202 ± 0.339** | 48.832 ± 7.863 | 19.296 ± 0.277 | 28.937 ± 22.459 |
| bin_matmul_256 | 0.459 ± 0.036 | **0.330 ± 0.016** | 0.598 ± 0.010 | 0.578 ± 0.145 | 1.065 ± 0.529 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.000 | **0.005 ± 0.000** | 0.021 ± 0.002 | 0.079 ± 0.006 | 0.007 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.049 ± 0.007** | **0.049 ± 0.005** | 0.100 ± 0.003 | 0.127 ± 0.014 | 0.050 ± 0.029 |
| bin_omeinsum_matmul_10x10 | 0.036 ± 0.004 | 0.085 ± 0.002 | 0.020 ± 0.000 | 0.047 ± 0.010 | **0.013 ± 0.001** |
| bin_outer_product_4096 | 5.905 ± 0.695 | 5.609 ± 0.590 | **4.390 ± 0.117** | 5.302 ± 0.578 | 332.464 ± 11.095 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 29.237 ± 1.937 | **20.074 ± 0.776** | 25.069 ± 0.381 | 20.992 ± 0.898 | 22.944 ± 6.935 |
| gm_queen5_5_3.wcsp | 3186.980 ± 135.709 | 2587.115 ± 54.210 | **1328.788 ± 30.947** | 1511.359 ± 52.577 | 3710.418 ± 164.653 |
| lm_batch_likelihood_brackets_4_4d | 13.887 ± 2.441 | 27.357 ± 1.489 | **8.754 ± 0.085** | 14.177 ± 0.982 | 26.302 ± 12.909 |
| lm_batch_likelihood_sentence_3_12d | 24.484 ± 4.329 | 63.606 ± 9.643 | **23.979 ± 1.332** | 25.502 ± 1.053 | 42.753 ± 30.103 |
| lm_batch_likelihood_sentence_4_4d | 14.372 ± 2.147 | 26.650 ± 2.747 | **9.582 ± 0.097** | 13.649 ± 0.578 | 27.683 ± 31.253 |
| nary_matmul_chain_64 | 0.038 ± 0.000 | **0.030 ± 0.000** | 0.043 ± 0.001 | 0.096 ± 0.006 | 0.059 ± 0.040 |
| str_matrix_chain_multiplication_100 | **5.368 ± 0.129** | 6.009 ± 0.183 | 6.300 ± 0.090 | 9.168 ± 0.469 | 8.706 ± 2.489 |
| str_mps_varying_inner_product_200 | **13.596 ± 0.306** | 15.148 ± 1.293 | 13.760 ± 2.115 | 18.317 ± 0.470 | 17.328 ± 17.901 |
| str_nw_mera_closed_120 | **494.804 ± 13.272** | 546.099 ± 24.198 | 499.738 ± 4.844 | 497.113 ± 11.148 | 508.203 ± 13.776 |
| str_nw_mera_open_26 | 409.253 ± 25.697 | 487.143 ± 33.604 | **331.052 ± 2.674** | 403.098 ± 9.670 | 644.607 ± 22.566 |
| tensornetwork_permutation_focus_step409_316 | **153.907 ± 21.787** | 205.057 ± 22.542 | 192.003 ± 2.418 | 681.272 ± 23.117 | 273.171 ± 38.998 |
| tensornetwork_permutation_light_415 | **138.049 ± 28.105** | 174.689 ± 24.762 | 186.415 ± 2.131 | 676.205 ± 27.580 | 254.371 ± 18.881 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.634 ± 0.075 | **1.620 ± 0.035** | 1.885 ± 2.621 | 1.957 ± 0.260 | 6.833 ± 5.950 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.500 ± 0.019 | 0.356 ± 0.019 | **0.224 ± 0.002** | 0.429 ± 0.036 | 0.451 ± 1.237 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.141 ± 0.014 | 0.138 ± 0.007 | **0.069 ± 0.003** | 0.243 ± 0.010 | 0.334 ± 0.253 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.208 ± 0.016 | 0.137 ± 0.008 | **0.066 ± 0.002** | 0.256 ± 0.015 | 3.015 ± 2.602 |
| bin_elementwise_mul_2048x2048 | **0.849 ± 0.027** | 1.076 ± 0.021 | 2.569 ± 0.167 | 3.604 ± 0.095 | 4.075 ± 0.129 |
| bin_elementwise_mul_512x512 | 0.072 ± 0.008 | 0.081 ± 0.018 | **0.061 ± 0.020** | 0.185 ± 0.012 | 0.084 ± 0.590 |
| bin_matmul_1024 | 27.265 ± 0.660 | **19.202 ± 0.339** | 48.832 ± 7.863 | 19.296 ± 0.277 | 28.937 ± 22.459 |
| bin_matmul_256 | 0.459 ± 0.036 | **0.330 ± 0.016** | 0.598 ± 0.010 | 0.578 ± 0.145 | 1.065 ± 0.529 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.000 | **0.005 ± 0.000** | 0.021 ± 0.002 | 0.079 ± 0.006 | 0.007 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.049 ± 0.007** | **0.049 ± 0.005** | 0.100 ± 0.003 | 0.127 ± 0.014 | 0.050 ± 0.029 |
| bin_omeinsum_matmul_10x10 | 0.036 ± 0.004 | 0.085 ± 0.002 | 0.020 ± 0.000 | 0.047 ± 0.010 | **0.013 ± 0.001** |
| bin_outer_product_4096 | 5.905 ± 0.695 | 5.609 ± 0.590 | **4.390 ± 0.117** | 5.302 ± 0.578 | 332.464 ± 11.095 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 29.237 ± 1.937 | **20.074 ± 0.776** | 25.069 ± 0.381 | 20.992 ± 0.898 | 22.944 ± 6.935 |
| gm_queen5_5_3.wcsp | 837.111 ± 42.648 | 846.318 ± 108.684 | 617.908 ± 7.760 | **595.594 ± 16.761** | 1271.750 ± 23.893 |
| lm_batch_likelihood_brackets_4_4d | 11.905 ± 2.432 | 25.958 ± 2.622 | **9.463 ± 0.123** | 14.323 ± 0.497 | 25.567 ± 37.219 |
| lm_batch_likelihood_sentence_3_12d | 28.854 ± 12.618 | 64.877 ± 12.276 | **23.687 ± 0.258** | 29.751 ± 1.713 | 81.235 ± 37.083 |
| lm_batch_likelihood_sentence_4_4d | 13.275 ± 0.462 | 34.142 ± 6.496 | **10.480 ± 0.189** | 14.277 ± 0.776 | 34.207 ± 45.256 |
| nary_matmul_chain_64 | 0.038 ± 0.000 | **0.030 ± 0.000** | 0.043 ± 0.001 | 0.096 ± 0.006 | 0.059 ± 0.040 |
| str_matrix_chain_multiplication_100 | **5.743 ± 0.274** | 6.085 ± 0.145 | 6.291 ± 0.143 | 9.119 ± 0.334 | 8.837 ± 1.137 |
| str_mps_varying_inner_product_200 | 16.373 ± 0.385 | 16.680 ± 0.374 | **14.443 ± 0.228** | 17.989 ± 0.317 | 16.432 ± 6.792 |
| str_nw_mera_closed_120 | 469.576 ± 17.357 | 493.932 ± 14.756 | **450.791 ± 8.755** | 470.946 ± 6.461 | 482.761 ± 36.313 |
| str_nw_mera_open_26 | 410.279 ± 31.711 | 467.964 ± 30.441 | **336.125 ± 5.314** | 427.504 ± 9.221 | 635.328 ± 32.595 |
| tensornetwork_permutation_focus_step409_316 | **153.907 ± 21.787** | 205.057 ± 22.542 | 192.003 ± 2.418 | 681.272 ± 23.117 | 273.171 ± 38.998 |
| tensornetwork_permutation_light_415 | **138.049 ± 28.105** | 174.689 ± 24.762 | 186.415 ± 2.131 | 676.205 ± 27.580 | 254.371 ± 18.881 |

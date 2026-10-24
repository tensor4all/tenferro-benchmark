# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`
- Benchmark commit: `50a7dc9b3ea08f5dba14db07df1e09314cac56f0`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261024_180212/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261024_180212/run.yaml`
- Timestamp: `20261024_180212`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261024_180212`.

- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
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


- Source table: `data/results/amd-cpu/cpu/einsum/20261024_180212/einsum_table_t1_20261024_180212.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20261024_180212/tenferro_trace_t1_20261024_180212.log`
- `data/results/amd-cpu/cpu/einsum/20261024_180212/tenferro_eager_t1_20261024_180212.log`
- `data/results/amd-cpu/cpu/einsum/20261024_180212/pytorch_cpu_t1_20261024_180212.log`
- `data/results/amd-cpu/cpu/einsum/20261024_180212/jax_cpu_t1_20261024_180212.log`
- `data/results/amd-cpu/cpu/einsum/20261024_180212/julia_omeinsum_t1_20261024_180212.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.874 ± 0.065 | **3.337 ± 0.070** | 4.145 ± 0.442 | 3.338 ± 0.232 | 7.733 ± 5.298 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.619 ± 0.012 | **0.487 ± 0.011** | 0.556 ± 0.025 | 0.620 ± 0.025 | 1.402 ± 1.491 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.160 ± 0.005 | 0.156 ± 0.002 | **0.128 ± 0.005** | 0.474 ± 0.024 | 1.035 ± 0.347 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.348 ± 0.016 | 0.155 ± 0.013 | **0.127 ± 0.010** | 0.487 ± 0.012 | 2.716 ± 1.039 |
| bin_elementwise_mul_2048x2048 | **1.602 ± 0.031** | 1.659 ± 0.034 | 3.766 ± 0.072 | 4.687 ± 0.084 | 4.320 ± 0.298 |
| bin_elementwise_mul_512x512 | 0.077 ± 0.005 | **0.068 ± 0.001** | 0.086 ± 0.002 | 0.255 ± 0.015 | 0.113 ± 0.884 |
| bin_matmul_1024 | 54.420 ± 0.191 | **42.575 ± 0.423** | 55.184 ± 0.171 | 47.595 ± 0.538 | 60.413 ± 11.840 |
| bin_matmul_256 | 0.931 ± 0.012 | **0.695 ± 0.015** | 1.217 ± 0.010 | 0.802 ± 0.006 | 1.855 ± 0.038 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.000 | **0.007 ± 0.002** | 0.021 ± 0.002 | 0.129 ± 0.019 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.031 ± 0.001 | **0.029 ± 0.001** | 0.095 ± 0.010 | 0.176 ± 0.018 | 0.056 ± 0.040 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.000 | **0.009 ± 0.000** | 0.019 ± 0.001 | 0.076 ± 0.021 | 0.018 ± 0.001 |
| bin_outer_product_4096 | 8.166 ± 0.120 | 8.073 ± 0.133 | 7.963 ± 0.070 | **7.943 ± 0.092** | 277.862 ± 5.063 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 53.983 ± 0.558 | **44.152 ± 0.332** | 51.414 ± 0.854 | 49.864 ± 0.493 | 58.094 ± 6.494 |
| gm_queen5_5_3.wcsp | 4984.761 ± 293.917 | 4303.773 ± 105.038 | 3630.372 ± 342.092 | **3325.798 ± 54.600** | 4275.927 ± 298.892 |
| lm_batch_likelihood_brackets_4_4d | 18.983 ± 0.957 | 25.692 ± 1.063 | 26.127 ± 0.421 | **12.062 ± 0.343** | 64.381 ± 47.452 |
| lm_batch_likelihood_sentence_3_12d | **46.582 ± 1.199** | 81.337 ± 2.919 | 66.167 ± 1.060 | 54.400 ± 1.299 | 97.530 ± 39.221 |
| lm_batch_likelihood_sentence_4_4d | 19.579 ± 0.428 | 29.563 ± 1.622 | 27.921 ± 0.451 | **13.396 ± 0.221** | 42.658 ± 28.331 |
| nary_matmul_chain_64 | **0.046 ± 0.001** | 0.049 ± 0.010 | 0.050 ± 0.020 | 0.124 ± 0.011 | 0.050 ± 0.023 |
| str_matrix_chain_multiplication_100 | **12.491 ± 0.249** | 14.171 ± 0.088 | 15.092 ± 0.212 | 12.861 ± 0.183 | 15.230 ± 1.133 |
| str_mps_varying_inner_product_200 | **17.005 ± 0.136** | 18.944 ± 0.331 | 20.994 ± 0.302 | 20.204 ± 0.361 | 28.055 ± 13.408 |
| str_nw_mera_closed_120 | **1149.707 ± 12.283** | 1174.226 ± 16.227 | 1222.704 ± 11.317 | 1339.993 ± 15.528 | 1204.270 ± 31.433 |
| str_nw_mera_open_26 | 773.110 ± 13.401 | 929.360 ± 10.124 | **751.331 ± 2.838** | 1076.576 ± 9.792 | 1061.170 ± 63.948 |
| tensornetwork_permutation_focus_step409_316 | **269.209 ± 2.680** | 298.415 ± 5.086 | 592.049 ± 1.998 | 1312.244 ± 12.034 | 353.512 ± 23.756 |
| tensornetwork_permutation_light_415 | **258.525 ± 3.496** | 263.870 ± 1.875 | 557.299 ± 3.771 | 1318.035 ± 15.488 | 349.675 ± 80.394 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.874 ± 0.065 | **3.337 ± 0.070** | 4.145 ± 0.442 | 3.338 ± 0.232 | 7.733 ± 5.298 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.619 ± 0.012 | **0.487 ± 0.011** | 0.556 ± 0.025 | 0.620 ± 0.025 | 1.402 ± 1.491 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.160 ± 0.005 | 0.156 ± 0.002 | **0.128 ± 0.005** | 0.474 ± 0.024 | 1.035 ± 0.347 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.348 ± 0.016 | 0.155 ± 0.013 | **0.127 ± 0.010** | 0.487 ± 0.012 | 2.716 ± 1.039 |
| bin_elementwise_mul_2048x2048 | **1.602 ± 0.031** | 1.659 ± 0.034 | 3.766 ± 0.072 | 4.687 ± 0.084 | 4.320 ± 0.298 |
| bin_elementwise_mul_512x512 | 0.077 ± 0.005 | **0.068 ± 0.001** | 0.086 ± 0.002 | 0.255 ± 0.015 | 0.113 ± 0.884 |
| bin_matmul_1024 | 54.420 ± 0.191 | **42.575 ± 0.423** | 55.184 ± 0.171 | 47.595 ± 0.538 | 60.413 ± 11.840 |
| bin_matmul_256 | 0.931 ± 0.012 | **0.695 ± 0.015** | 1.217 ± 0.010 | 0.802 ± 0.006 | 1.855 ± 0.038 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.000 | **0.007 ± 0.002** | 0.021 ± 0.002 | 0.129 ± 0.019 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.031 ± 0.001 | **0.029 ± 0.001** | 0.095 ± 0.010 | 0.176 ± 0.018 | 0.056 ± 0.040 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.000 | **0.009 ± 0.000** | 0.019 ± 0.001 | 0.076 ± 0.021 | 0.018 ± 0.001 |
| bin_outer_product_4096 | 8.166 ± 0.120 | 8.073 ± 0.133 | 7.963 ± 0.070 | **7.943 ± 0.092** | 277.862 ± 5.063 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 53.983 ± 0.558 | **44.152 ± 0.332** | 51.414 ± 0.854 | 49.864 ± 0.493 | 58.094 ± 6.494 |
| gm_queen5_5_3.wcsp | 1495.408 ± 14.372 | 1476.086 ± 65.014 | 1789.748 ± 132.512 | **1386.625 ± 29.243** | 1608.936 ± 14.280 |
| lm_batch_likelihood_brackets_4_4d | 19.194 ± 0.227 | 25.671 ± 1.346 | 28.736 ± 1.099 | **12.507 ± 0.188** | 35.299 ± 33.453 |
| lm_batch_likelihood_sentence_3_12d | **49.128 ± 0.785** | 85.181 ± 4.368 | 69.042 ± 2.143 | 55.709 ± 0.304 | 96.204 ± 32.316 |
| lm_batch_likelihood_sentence_4_4d | 19.735 ± 0.358 | 29.925 ± 1.503 | 29.079 ± 0.680 | **12.421 ± 0.092** | 31.050 ± 37.515 |
| nary_matmul_chain_64 | **0.046 ± 0.001** | 0.049 ± 0.010 | 0.050 ± 0.020 | 0.124 ± 0.011 | 0.050 ± 0.023 |
| str_matrix_chain_multiplication_100 | 13.279 ± 0.083 | 14.088 ± 0.082 | 15.051 ± 0.061 | **12.704 ± 0.195** | 14.579 ± 0.892 |
| str_mps_varying_inner_product_200 | 20.630 ± 0.209 | **19.787 ± 0.394** | 20.133 ± 0.229 | 24.196 ± 0.365 | 20.732 ± 10.000 |
| str_nw_mera_closed_120 | **1031.938 ± 12.184** | 1057.581 ± 8.387 | 1075.735 ± 17.181 | 1220.077 ± 7.848 | 1155.914 ± 78.325 |
| str_nw_mera_open_26 | 773.239 ± 10.196 | 918.753 ± 17.040 | **768.027 ± 3.353** | 1121.244 ± 5.590 | 1044.791 ± 37.550 |
| tensornetwork_permutation_focus_step409_316 | **269.209 ± 2.680** | 298.415 ± 5.086 | 592.049 ± 1.998 | 1312.244 ± 12.034 | 353.512 ± 23.756 |
| tensornetwork_permutation_light_415 | **258.525 ± 3.496** | 263.870 ± 1.875 | 557.299 ± 3.771 | 1318.035 ± 15.488 | 349.675 ± 80.394 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261024_183020/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261024_183020/run.yaml`
- Timestamp: `20261024_183020`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261024_183020`.

- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
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


- Source table: `data/results/amd-cpu/cpu/einsum/20261024_183020/einsum_table_t4_20261024_183020.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20261024_183020/tenferro_trace_t4_20261024_183020.log`
- `data/results/amd-cpu/cpu/einsum/20261024_183020/tenferro_eager_t4_20261024_183020.log`
- `data/results/amd-cpu/cpu/einsum/20261024_183020/pytorch_cpu_t4_20261024_183020.log`
- `data/results/amd-cpu/cpu/einsum/20261024_183020/jax_cpu_t4_20261024_183020.log`
- `data/results/amd-cpu/cpu/einsum/20261024_183020/julia_omeinsum_t4_20261024_183020.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.461 ± 0.131 | 2.334 ± 0.029 | **1.808 ± 1.922** | 2.227 ± 0.293 | 8.723 ± 7.451 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.721 ± 0.023 | 0.544 ± 0.010 | **0.293 ± 0.007** | 0.569 ± 0.020 | 0.578 ± 0.029 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.184 ± 0.010 | 0.117 ± 0.016 | **0.079 ± 0.002** | 0.352 ± 0.017 | 0.620 ± 0.699 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.297 ± 0.018 | 0.179 ± 0.012 | **0.080 ± 0.004** | 0.345 ± 0.010 | 1.858 ± 3.369 |
| bin_elementwise_mul_2048x2048 | **1.029 ± 0.038** | **1.029 ± 0.030** | 2.926 ± 0.091 | 3.874 ± 0.034 | 4.318 ± 0.054 |
| bin_elementwise_mul_512x512 | 0.075 ± 0.012 | 0.102 ± 0.018 | **0.060 ± 0.001** | 0.260 ± 0.017 | 0.099 ± 0.186 |
| bin_matmul_1024 | 29.223 ± 0.174 | **21.249 ± 0.262** | 41.619 ± 0.827 | 26.219 ± 0.447 | 39.188 ± 16.804 |
| bin_matmul_256 | 0.551 ± 0.012 | **0.393 ± 0.012** | 0.815 ± 0.011 | 0.609 ± 0.049 | 0.688 ± 0.087 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.008 ± 0.000** | 0.027 ± 0.002 | 0.125 ± 0.015 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.046 ± 0.000 | 0.042 ± 0.001 | 0.135 ± 0.004 | 0.184 ± 0.011 | **0.034 ± 0.002** |
| bin_omeinsum_matmul_10x10 | 0.028 ± 0.005 | **0.013 ± 0.000** | 0.029 ± 0.001 | 0.077 ± 0.021 | 0.017 ± 0.001 |
| bin_outer_product_4096 | 6.414 ± 3.797 | 6.943 ± 0.398 | 5.180 ± 0.595 | **5.041 ± 0.253** | 260.183 ± 3.409 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 31.167 ± 5.066 | **23.114 ± 0.935** | 27.117 ± 0.419 | 27.683 ± 0.966 | 36.861 ± 5.385 |
| gm_queen5_5_3.wcsp | 3501.798 ± 167.040 | 2875.257 ± 85.678 | **1414.828 ± 38.821** | 2204.383 ± 56.737 | 3792.392 ± 134.355 |
| lm_batch_likelihood_brackets_4_4d | 18.594 ± 4.114 | 31.579 ± 1.556 | **12.361 ± 0.155** | 17.641 ± 0.943 | 27.869 ± 17.610 |
| lm_batch_likelihood_sentence_3_12d | 34.639 ± 6.853 | 67.484 ± 6.684 | **33.448 ± 0.425** | 34.114 ± 0.888 | 81.528 ± 29.069 |
| lm_batch_likelihood_sentence_4_4d | 19.922 ± 0.544 | 33.689 ± 4.534 | **13.172 ± 0.078** | 18.159 ± 0.835 | 55.350 ± 35.651 |
| nary_matmul_chain_64 | 0.052 ± 0.001 | **0.040 ± 0.001** | 0.065 ± 0.001 | 0.109 ± 0.032 | 0.101 ± 0.045 |
| str_matrix_chain_multiplication_100 | 7.839 ± 0.122 | **7.778 ± 0.086** | 9.233 ± 0.125 | 12.696 ± 0.565 | 11.929 ± 1.457 |
| str_mps_varying_inner_product_200 | 21.189 ± 0.344 | **20.021 ± 0.222** | 20.817 ± 0.371 | 25.847 ± 0.584 | 26.885 ± 16.827 |
| str_nw_mera_closed_120 | **598.912 ± 24.705** | 605.379 ± 33.401 | 605.602 ± 10.686 | 680.225 ± 15.596 | 740.420 ± 45.006 |
| str_nw_mera_open_26 | 455.045 ± 11.958 | 546.141 ± 15.124 | **390.585 ± 2.684** | 550.405 ± 9.192 | 802.575 ± 25.351 |
| tensornetwork_permutation_focus_step409_316 | **195.175 ± 19.234** | 201.391 ± 23.988 | 225.570 ± 1.841 | 620.067 ± 20.056 | 339.952 ± 55.880 |
| tensornetwork_permutation_light_415 | **175.594 ± 9.791** | 181.852 ± 19.655 | 219.752 ± 1.990 | 627.023 ± 21.013 | 331.658 ± 33.945 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.461 ± 0.131 | 2.334 ± 0.029 | **1.808 ± 1.922** | 2.227 ± 0.293 | 8.723 ± 7.451 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.721 ± 0.023 | 0.544 ± 0.010 | **0.293 ± 0.007** | 0.569 ± 0.020 | 0.578 ± 0.029 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.184 ± 0.010 | 0.117 ± 0.016 | **0.079 ± 0.002** | 0.352 ± 0.017 | 0.620 ± 0.699 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.297 ± 0.018 | 0.179 ± 0.012 | **0.080 ± 0.004** | 0.345 ± 0.010 | 1.858 ± 3.369 |
| bin_elementwise_mul_2048x2048 | **1.029 ± 0.038** | **1.029 ± 0.030** | 2.926 ± 0.091 | 3.874 ± 0.034 | 4.318 ± 0.054 |
| bin_elementwise_mul_512x512 | 0.075 ± 0.012 | 0.102 ± 0.018 | **0.060 ± 0.001** | 0.260 ± 0.017 | 0.099 ± 0.186 |
| bin_matmul_1024 | 29.223 ± 0.174 | **21.249 ± 0.262** | 41.619 ± 0.827 | 26.219 ± 0.447 | 39.188 ± 16.804 |
| bin_matmul_256 | 0.551 ± 0.012 | **0.393 ± 0.012** | 0.815 ± 0.011 | 0.609 ± 0.049 | 0.688 ± 0.087 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.008 ± 0.000** | 0.027 ± 0.002 | 0.125 ± 0.015 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.046 ± 0.000 | 0.042 ± 0.001 | 0.135 ± 0.004 | 0.184 ± 0.011 | **0.034 ± 0.002** |
| bin_omeinsum_matmul_10x10 | 0.028 ± 0.005 | **0.013 ± 0.000** | 0.029 ± 0.001 | 0.077 ± 0.021 | 0.017 ± 0.001 |
| bin_outer_product_4096 | 6.414 ± 3.797 | 6.943 ± 0.398 | 5.180 ± 0.595 | **5.041 ± 0.253** | 260.183 ± 3.409 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 31.167 ± 5.066 | **23.114 ± 0.935** | 27.117 ± 0.419 | 27.683 ± 0.966 | 36.861 ± 5.385 |
| gm_queen5_5_3.wcsp | 1011.076 ± 43.109 | 928.357 ± 66.918 | **749.735 ± 7.325** | 903.806 ± 6.544 | 1410.095 ± 8.317 |
| lm_batch_likelihood_brackets_4_4d | 20.568 ± 2.670 | 31.846 ± 0.982 | **13.211 ± 0.238** | 17.422 ± 0.889 | 33.091 ± 29.800 |
| lm_batch_likelihood_sentence_3_12d | 32.198 ± 3.808 | 77.981 ± 12.971 | **28.458 ± 0.326** | 39.776 ± 1.747 | 89.367 ± 28.244 |
| lm_batch_likelihood_sentence_4_4d | 18.577 ± 1.418 | 35.379 ± 1.998 | **14.044 ± 0.167** | 17.763 ± 0.694 | 35.429 ± 33.055 |
| nary_matmul_chain_64 | 0.052 ± 0.001 | **0.040 ± 0.001** | 0.065 ± 0.001 | 0.109 ± 0.032 | 0.101 ± 0.045 |
| str_matrix_chain_multiplication_100 | **8.261 ± 0.145** | 8.314 ± 0.091 | 9.283 ± 0.261 | 12.603 ± 0.606 | 12.014 ± 0.579 |
| str_mps_varying_inner_product_200 | 25.756 ± 0.424 | 24.622 ± 0.597 | 25.310 ± 0.312 | 27.506 ± 0.491 | **22.227 ± 6.062** |
| str_nw_mera_closed_120 | 546.465 ± 24.929 | **545.220 ± 19.519** | 548.318 ± 4.633 | 622.611 ± 7.010 | 707.851 ± 50.237 |
| str_nw_mera_open_26 | 459.595 ± 17.069 | 542.335 ± 25.114 | **397.505 ± 3.113** | 563.604 ± 4.736 | 792.862 ± 38.377 |
| tensornetwork_permutation_focus_step409_316 | **195.175 ± 19.234** | 201.391 ± 23.988 | 225.570 ± 1.841 | 620.067 ± 20.056 | 339.952 ± 55.880 |
| tensornetwork_permutation_light_415 | **175.594 ± 9.791** | 181.852 ± 19.655 | 219.752 ± 1.990 | 627.023 ± 21.013 | 331.658 ± 33.945 |

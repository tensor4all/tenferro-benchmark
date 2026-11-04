# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`
- Benchmark commit: `9192ee1056de2203697691c63cc9ea04b0c5298e`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261104_205416/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261104_205416/run.yaml`
- Timestamp: `2026-11-04T20:54:16.992391+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261104_205416`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
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

- `data/results/amd-cpu/cpu/einsum/20261104_205416/jax_cpu_t1_20261104_205416.log`
- `data/results/amd-cpu/cpu/einsum/20261104_205416/julia_omeinsum_t1_20261104_205416.log`
- `data/results/amd-cpu/cpu/einsum/20261104_205416/pytorch_cpu_t1_20261104_205416.log`
- `data/results/amd-cpu/cpu/einsum/20261104_205416/tenferro_eager_t1_20261104_205416.log`
- `data/results/amd-cpu/cpu/einsum/20261104_205416/tenferro_trace_t1_20261104_205416.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.801 ± 0.314 | **4.079 ± 0.045** | 5.162 ± 0.240 | 5.030 ± 0.102 | 12.118 ± 10.662 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.785 ± 0.012 | **0.619 ± 0.023** | 0.749 ± 0.147 | 0.865 ± 0.052 | 1.350 ± 1.727 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.744 ± 0.018 | 0.738 ± 0.011 | **0.211 ± 0.043** | 0.615 ± 0.030 | 4.245 ± 2.378 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.210 ± 0.016 | 0.743 ± 0.009 | **0.206 ± 0.018** | 0.622 ± 0.033 | 5.633 ± 4.613 |
| bin_elementwise_mul_2048x2048 | 2.470 ± 0.137 | **2.465 ± 0.161** | 19.081 ± 0.870 | 20.235 ± 1.036 | 21.248 ± 2.486 |
| bin_elementwise_mul_512x512 | **0.134 ± 0.013** | 0.137 ± 0.015 | 0.168 ± 0.026 | 0.408 ± 0.038 | 0.199 ± 0.830 |
| bin_matmul_1024 | 58.218 ± 0.593 | **50.921 ± 0.963** | 61.956 ± 0.515 | 63.953 ± 0.994 | 83.082 ± 20.638 |
| bin_matmul_256 | 1.069 ± 0.014 | **0.827 ± 0.012** | 1.371 ± 0.014 | 1.252 ± 0.021 | 1.277 ± 0.032 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.011 ± 0.000 | **0.006 ± 0.000** | 0.037 ± 0.025 | 0.205 ± 0.032 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.044 ± 0.001 | **0.039 ± 0.001** | 0.147 ± 0.007 | 0.291 ± 0.056 | 0.053 ± 0.008 |
| bin_omeinsum_matmul_10x10 | 0.026 ± 0.000 | **0.012 ± 0.000** | 0.034 ± 0.002 | 0.194 ± 0.037 | 0.025 ± 0.002 |
| bin_outer_product_4096 | 72.127 ± 0.526 | 70.638 ± 1.755 | **61.336 ± 1.010** | 63.931 ± 1.192 | 331.928 ± 64.117 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 58.257 ± 0.471 | **53.958 ± 0.693** | 63.974 ± 0.842 | 67.053 ± 2.245 | 71.463 ± 12.468 |
| gm_queen5_5_3.wcsp | 7845.902 ± 85.550 | 7286.118 ± 66.568 | **6826.974 ± 101.480** | 8996.028 ± 205.585 | 10146.802 ± 182.696 |
| lm_batch_likelihood_brackets_4_4d | 26.839 ± 0.888 | 37.627 ± 1.136 | 38.311 ± 0.915 | **17.428 ± 0.913** | 33.200 ± 10.092 |
| lm_batch_likelihood_sentence_3_12d | **63.971 ± 1.495** | 141.494 ± 7.180 | 124.021 ± 4.618 | 110.370 ± 4.586 | 124.943 ± 11.974 |
| lm_batch_likelihood_sentence_4_4d | 28.119 ± 1.034 | 41.378 ± 1.843 | 40.519 ± 1.683 | **18.968 ± 0.309** | 73.200 ± 35.225 |
| nary_matmul_chain_64 | 0.059 ± 0.011 | **0.046 ± 0.016** | 0.080 ± 0.010 | 0.209 ± 0.020 | 0.138 ± 0.020 |
| str_matrix_chain_multiplication_100 | **15.288 ± 0.482** | 16.102 ± 0.213 | 17.065 ± 0.485 | 15.801 ± 0.463 | 17.911 ± 0.743 |
| str_mps_varying_inner_product_200 | **22.129 ± 0.303** | 22.275 ± 0.372 | 26.808 ± 0.257 | 24.694 ± 0.406 | 28.742 ± 36.345 |
| str_nw_mera_closed_120 | **1746.531 ± 8.847** | 1912.782 ± 25.529 | 2124.374 ± 19.922 | 2569.762 ± 21.858 | 1974.936 ± 46.649 |
| str_nw_mera_open_26 | 1394.127 ± 13.590 | 1585.738 ± 19.290 | **1306.375 ± 18.367** | 2030.514 ± 15.612 | 1811.222 ± 46.058 |
| tensornetwork_permutation_focus_step409_316 | **500.959 ± 7.932** | 610.216 ± 5.794 | 916.579 ± 4.111 | 1017.226 ± 169.438 | 685.747 ± 36.734 |
| tensornetwork_permutation_light_415 | **496.148 ± 8.711** | 542.169 ± 8.172 | 871.771 ± 9.217 | 1187.811 ± 253.471 | 692.227 ± 17.231 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.801 ± 0.314 | **4.079 ± 0.045** | 5.162 ± 0.240 | 5.030 ± 0.102 | 12.118 ± 10.662 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.785 ± 0.012 | **0.619 ± 0.023** | 0.749 ± 0.147 | 0.865 ± 0.052 | 1.350 ± 1.727 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.744 ± 0.018 | 0.738 ± 0.011 | **0.211 ± 0.043** | 0.615 ± 0.030 | 4.245 ± 2.378 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.210 ± 0.016 | 0.743 ± 0.009 | **0.206 ± 0.018** | 0.622 ± 0.033 | 5.633 ± 4.613 |
| bin_elementwise_mul_2048x2048 | 2.470 ± 0.137 | **2.465 ± 0.161** | 19.081 ± 0.870 | 20.235 ± 1.036 | 21.248 ± 2.486 |
| bin_elementwise_mul_512x512 | **0.134 ± 0.013** | 0.137 ± 0.015 | 0.168 ± 0.026 | 0.408 ± 0.038 | 0.199 ± 0.830 |
| bin_matmul_1024 | 58.218 ± 0.593 | **50.921 ± 0.963** | 61.956 ± 0.515 | 63.953 ± 0.994 | 83.082 ± 20.638 |
| bin_matmul_256 | 1.069 ± 0.014 | **0.827 ± 0.012** | 1.371 ± 0.014 | 1.252 ± 0.021 | 1.277 ± 0.032 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.011 ± 0.000 | **0.006 ± 0.000** | 0.037 ± 0.025 | 0.205 ± 0.032 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.044 ± 0.001 | **0.039 ± 0.001** | 0.147 ± 0.007 | 0.291 ± 0.056 | 0.053 ± 0.008 |
| bin_omeinsum_matmul_10x10 | 0.026 ± 0.000 | **0.012 ± 0.000** | 0.034 ± 0.002 | 0.194 ± 0.037 | 0.025 ± 0.002 |
| bin_outer_product_4096 | 72.127 ± 0.526 | 70.638 ± 1.755 | **61.336 ± 1.010** | 63.931 ± 1.192 | 331.928 ± 64.117 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 58.257 ± 0.471 | **53.958 ± 0.693** | 63.974 ± 0.842 | 67.053 ± 2.245 | 71.463 ± 12.468 |
| gm_queen5_5_3.wcsp | **2566.614 ± 30.203** | 2837.894 ± 51.092 | 2674.447 ± 73.379 | 2967.127 ± 28.165 | 3633.508 ± 28.740 |
| lm_batch_likelihood_brackets_4_4d | 25.936 ± 0.394 | 37.006 ± 1.395 | 38.586 ± 0.818 | **17.185 ± 0.781** | 75.795 ± 30.901 |
| lm_batch_likelihood_sentence_3_12d | **62.525 ± 2.581** | 182.240 ± 3.605 | 124.768 ± 2.098 | 107.054 ± 2.823 | 147.630 ± 20.602 |
| lm_batch_likelihood_sentence_4_4d | 28.907 ± 1.101 | 47.073 ± 3.449 | 40.174 ± 0.527 | **18.295 ± 0.740** | 49.477 ± 40.924 |
| nary_matmul_chain_64 | 0.059 ± 0.011 | **0.046 ± 0.016** | 0.080 ± 0.010 | 0.209 ± 0.020 | 0.138 ± 0.020 |
| str_matrix_chain_multiplication_100 | 15.906 ± 0.274 | 16.757 ± 0.334 | 16.540 ± 0.239 | **15.717 ± 0.503** | 17.658 ± 0.588 |
| str_mps_varying_inner_product_200 | 25.296 ± 0.396 | **25.268 ± 0.924** | 26.481 ± 0.422 | 25.861 ± 0.409 | 26.572 ± 11.364 |
| str_nw_mera_closed_120 | 1583.839 ± 19.140 | **1555.748 ± 22.921** | 1635.975 ± 12.961 | 2103.966 ± 15.753 | 1616.408 ± 77.983 |
| str_nw_mera_open_26 | 1388.596 ± 11.008 | 1579.546 ± 14.207 | **1360.563 ± 17.784** | 2107.560 ± 10.064 | 1809.753 ± 36.949 |
| tensornetwork_permutation_focus_step409_316 | **500.959 ± 7.932** | 610.216 ± 5.794 | 916.579 ± 4.111 | 1017.226 ± 169.438 | 685.747 ± 36.734 |
| tensornetwork_permutation_light_415 | **496.148 ± 8.711** | 542.169 ± 8.172 | 871.771 ± 9.217 | 1187.811 ± 253.471 | 692.227 ± 17.231 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261104_213752/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261104_213752/run.yaml`
- Timestamp: `2026-11-04T21:37:52.180072+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261104_213752`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
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

- `data/results/amd-cpu/cpu/einsum/20261104_213752/jax_cpu_t4_20261104_213752.log`
- `data/results/amd-cpu/cpu/einsum/20261104_213752/julia_omeinsum_t4_20261104_213752.log`
- `data/results/amd-cpu/cpu/einsum/20261104_213752/pytorch_cpu_t4_20261104_213752.log`
- `data/results/amd-cpu/cpu/einsum/20261104_213752/tenferro_eager_t4_20261104_213752.log`
- `data/results/amd-cpu/cpu/einsum/20261104_213752/tenferro_trace_t4_20261104_213752.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 17.991 ± 0.124 | 2.867 ± 0.037 | **2.405 ± 3.564** | 3.172 ± 0.724 | 12.526 ± 13.701 |
| bin_batched_matmul_b32_m64_n64_k64 | 12.004 ± 3.002 | 0.686 ± 0.017 | **0.399 ± 0.021** | 0.858 ± 0.019 | 0.988 ± 0.073 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 2.999 ± 0.003 | 2.989 ± 0.611 | **0.225 ± 0.013** | 0.487 ± 0.025 | 2.895 ± 3.387 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.999 ± 0.002 | 2.998 ± 0.010 | **0.234 ± 0.035** | 0.492 ± 0.042 | 6.262 ± 6.079 |
| bin_elementwise_mul_2048x2048 | 2.976 ± 0.206 | **0.987 ± 0.115** | 6.373 ± 0.365 | 10.540 ± 0.108 | 20.389 ± 0.560 |
| bin_elementwise_mul_512x512 | 0.136 ± 0.017 | **0.119 ± 0.012** | 0.179 ± 0.020 | 0.409 ± 0.042 | 0.198 ± 0.814 |
| bin_matmul_1024 | 40.996 ± 1.014 | **25.342 ± 0.344** | 36.801 ± 0.535 | 36.541 ± 0.460 | 58.634 ± 29.300 |
| bin_matmul_256 | 8.567 ± 6.010 | **0.464 ± 0.020** | 0.859 ± 0.011 | 0.911 ± 0.081 | 0.960 ± 0.027 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.019 ± 0.000 | **0.011 ± 0.000** | 0.053 ± 0.004 | 0.204 ± 0.024 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.071 ± 0.002 | 0.064 ± 0.004 | 0.227 ± 0.038 | 0.309 ± 0.046 | **0.052 ± 0.002** |
| bin_omeinsum_matmul_10x10 | 0.044 ± 0.011 | **0.020 ± 0.001** | 0.051 ± 0.014 | 0.189 ± 0.038 | 0.025 ± 0.001 |
| bin_outer_product_4096 | 24.380 ± 0.748 | 23.787 ± 0.665 | **20.345 ± 0.272** | 32.163 ± 0.735 | 312.085 ± 3.871 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 42.211 ± 7.422 | **28.386 ± 0.320** | 31.529 ± 0.714 | 39.980 ± 1.970 | 44.290 ± 22.025 |
| gm_queen5_5_3.wcsp | 6716.659 ± 156.590 | 5504.015 ± 63.992 | **2797.303 ± 24.336** | 5836.364 ± 104.597 | 9265.370 ± 193.971 |
| lm_batch_likelihood_brackets_4_4d | 128.004 ± 27.014 | 125.700 ± 39.895 | **17.825 ± 0.483** | 24.756 ± 1.507 | 44.553 ± 14.454 |
| lm_batch_likelihood_sentence_3_12d | 88.274 ± 26.484 | 173.482 ± 28.186 | 62.153 ± 1.311 | **55.139 ± 4.081** | 111.434 ± 36.757 |
| lm_batch_likelihood_sentence_4_4d | 140.810 ± 33.117 | 155.660 ± 13.584 | **18.487 ± 0.367** | 25.028 ± 0.998 | 63.399 ± 31.780 |
| nary_matmul_chain_64 | 0.080 ± 0.006 | **0.068 ± 0.008** | 0.103 ± 0.028 | 0.288 ± 0.024 | 0.150 ± 0.061 |
| str_matrix_chain_multiplication_100 | **9.170 ± 0.198** | 9.966 ± 0.175 | 11.495 ± 0.235 | 15.838 ± 0.865 | 15.209 ± 4.311 |
| str_mps_varying_inner_product_200 | 25.468 ± 0.260 | **24.211 ± 0.180** | 27.276 ± 0.324 | 32.011 ± 1.642 | 26.558 ± 6.472 |
| str_nw_mera_closed_120 | 1103.523 ± 8.657 | 1235.072 ± 22.063 | **988.817 ± 12.395** | 1187.356 ± 24.771 | 1289.173 ± 45.177 |
| str_nw_mera_open_26 | 1131.716 ± 16.069 | 949.827 ± 53.923 | **615.273 ± 6.524** | 970.621 ± 13.323 | 1323.008 ± 46.296 |
| tensornetwork_permutation_focus_step409_316 | 493.944 ± 3.472 | 482.870 ± 8.808 | **357.260 ± 4.702** | 422.327 ± 9.349 | 609.460 ± 27.162 |
| tensornetwork_permutation_light_415 | 478.707 ± 5.359 | 459.446 ± 61.982 | **354.047 ± 7.064** | 449.354 ± 43.825 | 612.514 ± 27.089 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 17.991 ± 0.124 | 2.867 ± 0.037 | **2.405 ± 3.564** | 3.172 ± 0.724 | 12.526 ± 13.701 |
| bin_batched_matmul_b32_m64_n64_k64 | 12.004 ± 3.002 | 0.686 ± 0.017 | **0.399 ± 0.021** | 0.858 ± 0.019 | 0.988 ± 0.073 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 2.999 ± 0.003 | 2.989 ± 0.611 | **0.225 ± 0.013** | 0.487 ± 0.025 | 2.895 ± 3.387 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.999 ± 0.002 | 2.998 ± 0.010 | **0.234 ± 0.035** | 0.492 ± 0.042 | 6.262 ± 6.079 |
| bin_elementwise_mul_2048x2048 | 2.976 ± 0.206 | **0.987 ± 0.115** | 6.373 ± 0.365 | 10.540 ± 0.108 | 20.389 ± 0.560 |
| bin_elementwise_mul_512x512 | 0.136 ± 0.017 | **0.119 ± 0.012** | 0.179 ± 0.020 | 0.409 ± 0.042 | 0.198 ± 0.814 |
| bin_matmul_1024 | 40.996 ± 1.014 | **25.342 ± 0.344** | 36.801 ± 0.535 | 36.541 ± 0.460 | 58.634 ± 29.300 |
| bin_matmul_256 | 8.567 ± 6.010 | **0.464 ± 0.020** | 0.859 ± 0.011 | 0.911 ± 0.081 | 0.960 ± 0.027 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.019 ± 0.000 | **0.011 ± 0.000** | 0.053 ± 0.004 | 0.204 ± 0.024 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.071 ± 0.002 | 0.064 ± 0.004 | 0.227 ± 0.038 | 0.309 ± 0.046 | **0.052 ± 0.002** |
| bin_omeinsum_matmul_10x10 | 0.044 ± 0.011 | **0.020 ± 0.001** | 0.051 ± 0.014 | 0.189 ± 0.038 | 0.025 ± 0.001 |
| bin_outer_product_4096 | 24.380 ± 0.748 | 23.787 ± 0.665 | **20.345 ± 0.272** | 32.163 ± 0.735 | 312.085 ± 3.871 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 42.211 ± 7.422 | **28.386 ± 0.320** | 31.529 ± 0.714 | 39.980 ± 1.970 | 44.290 ± 22.025 |
| gm_queen5_5_3.wcsp | 2228.337 ± 16.412 | 1803.965 ± 47.646 | **1178.745 ± 21.519** | 1952.174 ± 32.121 | 3301.890 ± 45.871 |
| lm_batch_likelihood_brackets_4_4d | 114.998 ± 24.476 | 143.626 ± 15.802 | **19.149 ± 0.535** | 24.300 ± 1.197 | 68.101 ± 35.579 |
| lm_batch_likelihood_sentence_3_12d | 95.428 ± 19.131 | 209.562 ± 33.236 | **34.737 ± 1.045** | 65.871 ± 9.442 | 121.417 ± 25.358 |
| lm_batch_likelihood_sentence_4_4d | 134.218 ± 65.348 | 144.958 ± 41.182 | **18.688 ± 0.340** | 24.453 ± 1.252 | 58.473 ± 46.723 |
| nary_matmul_chain_64 | 0.080 ± 0.006 | **0.068 ± 0.008** | 0.103 ± 0.028 | 0.288 ± 0.024 | 0.150 ± 0.061 |
| str_matrix_chain_multiplication_100 | **9.314 ± 0.223** | 9.920 ± 0.222 | 11.153 ± 0.195 | 15.679 ± 0.765 | 15.025 ± 1.464 |
| str_mps_varying_inner_product_200 | 30.618 ± 0.396 | 29.029 ± 0.925 | 29.120 ± 0.345 | 32.623 ± 0.865 | **25.117 ± 8.123** |
| str_nw_mera_closed_120 | 998.125 ± 6.679 | 931.457 ± 5.470 | **791.230 ± 9.300** | 996.075 ± 20.683 | 1009.308 ± 54.033 |
| str_nw_mera_open_26 | 1137.082 ± 13.902 | 896.112 ± 84.384 | **624.527 ± 11.079** | 991.794 ± 16.848 | 1321.462 ± 43.125 |
| tensornetwork_permutation_focus_step409_316 | 493.944 ± 3.472 | 482.870 ± 8.808 | **357.260 ± 4.702** | 422.327 ± 9.349 | 609.460 ± 27.162 |
| tensornetwork_permutation_light_415 | 478.707 ± 5.359 | 459.446 ± 61.982 | **354.047 ± 7.064** | 449.354 ± 43.825 | 612.514 ± 27.089 |

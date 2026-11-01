# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`
- Benchmark commit: `4cc17b1c7061855cfd9ddc2d549a09f50199de3c`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261101_012518/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261101_012518/run.yaml`
- Timestamp: `2026-11-01T01:25:18.859596+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261101_012518`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

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

- `data/results/amd-cpu/cpu/einsum/20261101_012518/jax_cpu_t1_20261101_012518.log`
- `data/results/amd-cpu/cpu/einsum/20261101_012518/julia_omeinsum_t1_20261101_012518.log`
- `data/results/amd-cpu/cpu/einsum/20261101_012518/pytorch_cpu_t1_20261101_012518.log`
- `data/results/amd-cpu/cpu/einsum/20261101_012518/tenferro_eager_t1_20261101_012518.log`
- `data/results/amd-cpu/cpu/einsum/20261101_012518/tenferro_trace_t1_20261101_012518.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.945 ± 0.100 | **3.896 ± 0.034** | 4.103 ± 0.053 | 4.286 ± 0.112 | 13.431 ± 3.591 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.779 ± 0.048 | **0.600 ± 0.014** | 0.654 ± 0.010 | 0.842 ± 0.014 | 0.916 ± 0.116 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.762 ± 0.019 | 0.750 ± 0.008 | **0.200 ± 0.018** | 0.612 ± 0.037 | 0.796 ± 3.347 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.237 ± 0.016 | 0.745 ± 0.024 | **0.196 ± 0.016** | 0.615 ± 0.035 | 5.305 ± 6.826 |
| bin_elementwise_mul_2048x2048 | 1.898 ± 0.137 | **1.820 ± 0.095** | 16.754 ± 0.496 | 17.964 ± 0.189 | 19.345 ± 0.348 |
| bin_elementwise_mul_512x512 | 0.126 ± 0.003 | **0.122 ± 0.007** | 0.146 ± 0.007 | 0.394 ± 0.038 | 0.199 ± 0.026 |
| bin_matmul_1024 | 56.977 ± 0.586 | **48.799 ± 0.286** | 75.359 ± 0.600 | 63.405 ± 4.724 | 74.501 ± 29.705 |
| bin_matmul_256 | 1.033 ± 0.012 | **0.808 ± 0.012** | 1.442 ± 0.021 | 1.163 ± 0.019 | 1.952 ± 0.889 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.011 ± 0.000 | **0.006 ± 0.005** | 0.037 ± 0.006 | 0.205 ± 0.026 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.066 ± 0.001 | **0.041 ± 0.002** | 0.148 ± 0.008 | 0.241 ± 0.034 | 0.082 ± 0.045 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.001 | **0.011 ± 0.000** | 0.034 ± 0.010 | 0.152 ± 0.046 | 0.024 ± 0.000 |
| bin_outer_product_4096 | 67.532 ± 0.559 | 67.187 ± 0.554 | **55.917 ± 0.563** | 59.150 ± 1.266 | 315.923 ± 1.754 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 59.014 ± 0.329 | **51.246 ± 0.259** | 65.354 ± 0.449 | 65.496 ± 0.755 | 72.726 ± 15.605 |
| gm_queen5_5_3.wcsp | 7124.582 ± 130.979 | 6869.580 ± 35.875 | **6408.370 ± 48.596** | 8253.497 ± 60.150 | 9494.326 ± 186.526 |
| lm_batch_likelihood_brackets_4_4d | 23.799 ± 0.523 | 34.986 ± 1.685 | 35.783 ± 1.219 | **14.895 ± 0.498** | 31.951 ± 8.293 |
| lm_batch_likelihood_sentence_3_12d | **56.142 ± 0.721** | 139.888 ± 3.096 | 110.027 ± 1.555 | 96.009 ± 1.264 | 109.222 ± 8.977 |
| lm_batch_likelihood_sentence_4_4d | 25.149 ± 0.434 | 39.656 ± 2.339 | 37.160 ± 0.884 | **16.856 ± 0.371** | 67.004 ± 25.854 |
| nary_matmul_chain_64 | 0.094 ± 0.003 | **0.046 ± 0.020** | 0.080 ± 0.013 | 0.200 ± 0.033 | 0.127 ± 0.049 |
| str_matrix_chain_multiplication_100 | **14.380 ± 0.197** | 15.229 ± 0.268 | 15.972 ± 0.197 | 15.231 ± 0.500 | 16.902 ± 0.645 |
| str_mps_varying_inner_product_200 | **20.279 ± 0.560** | 22.031 ± 0.274 | 26.004 ± 0.635 | 23.682 ± 0.565 | 25.811 ± 7.019 |
| str_nw_mera_closed_120 | **1659.215 ± 11.727** | 1842.133 ± 32.761 | 2015.972 ± 11.924 | 2514.191 ± 8.098 | 1848.338 ± 25.037 |
| str_nw_mera_open_26 | 1324.871 ± 4.926 | 1492.928 ± 8.089 | **1246.299 ± 11.615** | 1981.432 ± 9.123 | 1593.790 ± 46.910 |
| tensornetwork_permutation_focus_step409_316 | **486.025 ± 8.268** | 558.665 ± 3.370 | 891.078 ± 4.817 | 675.624 ± 7.387 | 535.616 ± 57.277 |
| tensornetwork_permutation_light_415 | **475.177 ± 3.400** | 498.486 ± 3.206 | 852.233 ± 2.831 | 709.332 ± 13.732 | 555.157 ± 60.565 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.945 ± 0.100 | **3.896 ± 0.034** | 4.103 ± 0.053 | 4.286 ± 0.112 | 13.431 ± 3.591 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.779 ± 0.048 | **0.600 ± 0.014** | 0.654 ± 0.010 | 0.842 ± 0.014 | 0.916 ± 0.116 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.762 ± 0.019 | 0.750 ± 0.008 | **0.200 ± 0.018** | 0.612 ± 0.037 | 0.796 ± 3.347 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.237 ± 0.016 | 0.745 ± 0.024 | **0.196 ± 0.016** | 0.615 ± 0.035 | 5.305 ± 6.826 |
| bin_elementwise_mul_2048x2048 | 1.898 ± 0.137 | **1.820 ± 0.095** | 16.754 ± 0.496 | 17.964 ± 0.189 | 19.345 ± 0.348 |
| bin_elementwise_mul_512x512 | 0.126 ± 0.003 | **0.122 ± 0.007** | 0.146 ± 0.007 | 0.394 ± 0.038 | 0.199 ± 0.026 |
| bin_matmul_1024 | 56.977 ± 0.586 | **48.799 ± 0.286** | 75.359 ± 0.600 | 63.405 ± 4.724 | 74.501 ± 29.705 |
| bin_matmul_256 | 1.033 ± 0.012 | **0.808 ± 0.012** | 1.442 ± 0.021 | 1.163 ± 0.019 | 1.952 ± 0.889 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.011 ± 0.000 | **0.006 ± 0.005** | 0.037 ± 0.006 | 0.205 ± 0.026 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.066 ± 0.001 | **0.041 ± 0.002** | 0.148 ± 0.008 | 0.241 ± 0.034 | 0.082 ± 0.045 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.001 | **0.011 ± 0.000** | 0.034 ± 0.010 | 0.152 ± 0.046 | 0.024 ± 0.000 |
| bin_outer_product_4096 | 67.532 ± 0.559 | 67.187 ± 0.554 | **55.917 ± 0.563** | 59.150 ± 1.266 | 315.923 ± 1.754 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 59.014 ± 0.329 | **51.246 ± 0.259** | 65.354 ± 0.449 | 65.496 ± 0.755 | 72.726 ± 15.605 |
| gm_queen5_5_3.wcsp | **2407.731 ± 32.949** | 2627.302 ± 11.825 | 2480.355 ± 15.860 | 2854.893 ± 16.732 | 3295.998 ± 52.903 |
| lm_batch_likelihood_brackets_4_4d | 23.735 ± 0.122 | 35.156 ± 0.901 | 36.405 ± 1.022 | **15.029 ± 0.395** | 70.071 ± 27.450 |
| lm_batch_likelihood_sentence_3_12d | **54.640 ± 1.679** | 161.657 ± 2.030 | 114.376 ± 0.766 | 98.872 ± 2.132 | 107.364 ± 32.760 |
| lm_batch_likelihood_sentence_4_4d | 24.921 ± 0.459 | 41.033 ± 1.522 | 37.997 ± 0.421 | **16.059 ± 0.705** | 70.404 ± 28.483 |
| nary_matmul_chain_64 | 0.094 ± 0.003 | **0.046 ± 0.020** | 0.080 ± 0.013 | 0.200 ± 0.033 | 0.127 ± 0.049 |
| str_matrix_chain_multiplication_100 | 15.276 ± 0.135 | 15.425 ± 0.498 | 16.106 ± 0.371 | **14.718 ± 0.287** | 16.606 ± 0.259 |
| str_mps_varying_inner_product_200 | **23.667 ± 0.259** | 24.062 ± 0.465 | 25.880 ± 0.424 | 24.037 ± 0.516 | 24.688 ± 8.266 |
| str_nw_mera_closed_120 | 1515.567 ± 6.921 | 1500.580 ± 17.862 | 1531.395 ± 12.025 | 2094.329 ± 18.333 | **1488.808 ± 34.156** |
| str_nw_mera_open_26 | 1334.130 ± 6.285 | 1519.049 ± 15.055 | **1258.166 ± 7.676** | 2044.946 ± 7.140 | 1569.614 ± 31.001 |
| tensornetwork_permutation_focus_step409_316 | **486.025 ± 8.268** | 558.665 ± 3.370 | 891.078 ± 4.817 | 675.624 ± 7.387 | 535.616 ± 57.277 |
| tensornetwork_permutation_light_415 | **475.177 ± 3.400** | 498.486 ± 3.206 | 852.233 ± 2.831 | 709.332 ± 13.732 | 555.157 ± 60.565 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261101_020627/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261101_020627/run.yaml`
- Timestamp: `2026-11-01T02:06:27.548364+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261101_020627`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

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

- `data/results/amd-cpu/cpu/einsum/20261101_020627/jax_cpu_t4_20261101_020627.log`
- `data/results/amd-cpu/cpu/einsum/20261101_020627/julia_omeinsum_t4_20261101_020627.log`
- `data/results/amd-cpu/cpu/einsum/20261101_020627/pytorch_cpu_t4_20261101_020627.log`
- `data/results/amd-cpu/cpu/einsum/20261101_020627/tenferro_eager_t4_20261101_020627.log`
- `data/results/amd-cpu/cpu/einsum/20261101_020627/tenferro_trace_t4_20261101_020627.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 17.985 ± 0.069 | 2.848 ± 0.015 | **2.345 ± 3.659** | 3.057 ± 0.670 | 13.568 ± 12.118 |
| bin_batched_matmul_b32_m64_n64_k64 | 9.000 ± 2.221 | 0.673 ± 0.026 | **0.387 ± 0.026** | 0.799 ± 0.034 | 0.988 ± 1.583 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.424 ± 0.030 | 0.436 ± 0.049 | **0.210 ± 0.012** | 0.471 ± 0.044 | 4.046 ± 2.384 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.057 ± 1.999 | 3.011 ± 0.054 | **0.209 ± 0.023** | 0.461 ± 0.041 | 5.733 ± 5.609 |
| bin_elementwise_mul_2048x2048 | 2.997 ± 0.050 | **0.863 ± 0.027** | 5.855 ± 0.134 | 9.791 ± 0.386 | 19.809 ± 0.296 |
| bin_elementwise_mul_512x512 | 0.139 ± 0.935 | **0.132 ± 0.001** | 0.139 ± 0.036 | 0.398 ± 0.033 | 0.171 ± 0.808 |
| bin_matmul_1024 | 38.753 ± 2.134 | **24.473 ± 0.119** | 35.827 ± 0.356 | 36.145 ± 0.442 | 56.856 ± 17.463 |
| bin_matmul_256 | 3.000 ± 0.013 | **0.472 ± 0.026** | 0.864 ± 0.010 | 0.903 ± 0.072 | 1.922 ± 0.688 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.018 ± 0.001 | **0.011 ± 0.000** | 0.052 ± 0.003 | 0.203 ± 0.026 | 0.014 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.075 ± 0.018 | 0.067 ± 0.001 | 0.226 ± 0.025 | 0.251 ± 0.040 | **0.054 ± 0.007** |
| bin_omeinsum_matmul_10x10 | 0.050 ± 0.004 | **0.020 ± 0.000** | 0.049 ± 0.003 | 0.152 ± 0.046 | 0.028 ± 0.012 |
| bin_outer_product_4096 | 22.726 ± 0.485 | 23.605 ± 0.572 | **19.098 ± 0.331** | 31.122 ± 0.700 | 306.957 ± 20.131 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 36.111 ± 2.248 | **27.082 ± 0.210** | 29.733 ± 0.398 | 38.808 ± 1.638 | 51.800 ± 28.530 |
| gm_queen5_5_3.wcsp | 6239.682 ± 65.598 | 5234.099 ± 64.224 | **2672.068 ± 22.401** | 5474.472 ± 49.770 | 8912.761 ± 330.013 |
| lm_batch_likelihood_brackets_4_4d | 121.929 ± 25.059 | 139.122 ± 24.604 | **17.094 ± 0.585** | 23.758 ± 1.278 | 60.702 ± 26.300 |
| lm_batch_likelihood_sentence_3_12d | 109.033 ± 3.010 | 174.839 ± 25.481 | **32.973 ± 1.444** | 50.939 ± 7.633 | 113.038 ± 31.787 |
| lm_batch_likelihood_sentence_4_4d | 105.030 ± 45.065 | 145.388 ± 20.513 | **18.064 ± 0.763** | 24.288 ± 1.038 | 54.552 ± 26.831 |
| nary_matmul_chain_64 | 0.078 ± 0.021 | **0.065 ± 0.009** | 0.100 ± 0.025 | 0.198 ± 0.021 | 0.148 ± 0.049 |
| str_matrix_chain_multiplication_100 | 9.301 ± 0.251 | **9.129 ± 0.183** | 10.975 ± 0.443 | 15.787 ± 0.537 | 15.953 ± 0.853 |
| str_mps_varying_inner_product_200 | **22.972 ± 0.420** | 26.338 ± 0.546 | 26.556 ± 1.034 | 31.374 ± 0.798 | 33.275 ± 23.640 |
| str_nw_mera_closed_120 | 1049.898 ± 6.207 | 1237.949 ± 12.226 | **959.666 ± 11.670** | 1193.491 ± 21.871 | 1281.120 ± 62.511 |
| str_nw_mera_open_26 | 959.807 ± 13.296 | 911.374 ± 40.989 | **594.802 ± 4.774** | 954.949 ± 16.225 | 1317.877 ± 30.945 |
| tensornetwork_permutation_focus_step409_316 | 486.221 ± 4.746 | 496.320 ± 25.671 | **349.957 ± 4.946** | 434.982 ± 45.121 | 622.521 ± 15.482 |
| tensornetwork_permutation_light_415 | 480.173 ± 4.359 | 474.730 ± 8.859 | **346.126 ± 1.644** | 449.221 ± 73.536 | 624.046 ± 18.879 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 17.985 ± 0.069 | 2.848 ± 0.015 | **2.345 ± 3.659** | 3.057 ± 0.670 | 13.568 ± 12.118 |
| bin_batched_matmul_b32_m64_n64_k64 | 9.000 ± 2.221 | 0.673 ± 0.026 | **0.387 ± 0.026** | 0.799 ± 0.034 | 0.988 ± 1.583 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.424 ± 0.030 | 0.436 ± 0.049 | **0.210 ± 0.012** | 0.471 ± 0.044 | 4.046 ± 2.384 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.057 ± 1.999 | 3.011 ± 0.054 | **0.209 ± 0.023** | 0.461 ± 0.041 | 5.733 ± 5.609 |
| bin_elementwise_mul_2048x2048 | 2.997 ± 0.050 | **0.863 ± 0.027** | 5.855 ± 0.134 | 9.791 ± 0.386 | 19.809 ± 0.296 |
| bin_elementwise_mul_512x512 | 0.139 ± 0.935 | **0.132 ± 0.001** | 0.139 ± 0.036 | 0.398 ± 0.033 | 0.171 ± 0.808 |
| bin_matmul_1024 | 38.753 ± 2.134 | **24.473 ± 0.119** | 35.827 ± 0.356 | 36.145 ± 0.442 | 56.856 ± 17.463 |
| bin_matmul_256 | 3.000 ± 0.013 | **0.472 ± 0.026** | 0.864 ± 0.010 | 0.903 ± 0.072 | 1.922 ± 0.688 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.018 ± 0.001 | **0.011 ± 0.000** | 0.052 ± 0.003 | 0.203 ± 0.026 | 0.014 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.075 ± 0.018 | 0.067 ± 0.001 | 0.226 ± 0.025 | 0.251 ± 0.040 | **0.054 ± 0.007** |
| bin_omeinsum_matmul_10x10 | 0.050 ± 0.004 | **0.020 ± 0.000** | 0.049 ± 0.003 | 0.152 ± 0.046 | 0.028 ± 0.012 |
| bin_outer_product_4096 | 22.726 ± 0.485 | 23.605 ± 0.572 | **19.098 ± 0.331** | 31.122 ± 0.700 | 306.957 ± 20.131 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 36.111 ± 2.248 | **27.082 ± 0.210** | 29.733 ± 0.398 | 38.808 ± 1.638 | 51.800 ± 28.530 |
| gm_queen5_5_3.wcsp | 2107.934 ± 21.015 | 1725.354 ± 43.895 | **1131.549 ± 11.534** | 1930.791 ± 42.232 | 3248.858 ± 47.249 |
| lm_batch_likelihood_brackets_4_4d | 122.970 ± 54.086 | 133.738 ± 18.907 | **18.527 ± 0.736** | 24.096 ± 1.074 | 60.654 ± 31.427 |
| lm_batch_likelihood_sentence_3_12d | 110.273 ± 23.021 | 214.812 ± 19.971 | **33.175 ± 0.500** | 49.831 ± 7.426 | 122.632 ± 19.883 |
| lm_batch_likelihood_sentence_4_4d | 119.025 ± 67.587 | 131.896 ± 27.783 | **18.973 ± 0.337** | 24.203 ± 0.776 | 57.782 ± 41.857 |
| nary_matmul_chain_64 | 0.078 ± 0.021 | **0.065 ± 0.009** | 0.100 ± 0.025 | 0.198 ± 0.021 | 0.148 ± 0.049 |
| str_matrix_chain_multiplication_100 | **9.448 ± 0.604** | 9.647 ± 0.197 | 11.043 ± 0.138 | 15.082 ± 1.068 | 15.092 ± 0.565 |
| str_mps_varying_inner_product_200 | 29.461 ± 0.574 | 29.879 ± 0.471 | 28.729 ± 0.788 | 31.427 ± 0.651 | **26.287 ± 12.063** |
| str_nw_mera_closed_120 | 947.640 ± 6.341 | 944.871 ± 19.369 | **769.587 ± 5.073** | 978.046 ± 19.608 | 988.897 ± 57.945 |
| str_nw_mera_open_26 | 962.725 ± 7.170 | 899.393 ± 68.230 | **593.840 ± 7.146** | 983.601 ± 12.840 | 1282.861 ± 47.537 |
| tensornetwork_permutation_focus_step409_316 | 486.221 ± 4.746 | 496.320 ± 25.671 | **349.957 ± 4.946** | 434.982 ± 45.121 | 622.521 ± 15.482 |
| tensornetwork_permutation_light_415 | 480.173 ± 4.359 | 474.730 ± 8.859 | **346.126 ± 1.644** | 449.221 ± 73.536 | 624.046 ± 18.879 |

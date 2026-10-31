# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`
- Benchmark commit: `41db23777fb8781bbb7be5c457ac47b9216e2e3e`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261031_220732/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261031_220732/run.yaml`
- Timestamp: `2026-10-31T22:07:32.514790+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261031_220732`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

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

- `data/results/amd-cpu/cpu/einsum/20261031_220732/jax_cpu_t1_20261031_220732.log`
- `data/results/amd-cpu/cpu/einsum/20261031_220732/julia_omeinsum_t1_20261031_220732.log`
- `data/results/amd-cpu/cpu/einsum/20261031_220732/pytorch_cpu_t1_20261031_220732.log`
- `data/results/amd-cpu/cpu/einsum/20261031_220732/tenferro_eager_t1_20261031_220732.log`
- `data/results/amd-cpu/cpu/einsum/20261031_220732/tenferro_trace_t1_20261031_220732.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.196 ± 0.264 | 3.332 ± 0.046 | 3.671 ± 0.123 | **3.142 ± 0.040** | 10.142 ± 6.244 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.625 ± 0.013 | **0.483 ± 0.011** | 0.548 ± 0.015 | 0.616 ± 0.011 | 2.138 ± 1.603 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.538 ± 0.012 | 0.504 ± 0.009 | **0.126 ± 0.005** | 0.484 ± 0.018 | 2.946 ± 1.010 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.929 ± 0.006 | 0.506 ± 0.009 | **0.125 ± 0.002** | 0.486 ± 0.013 | 4.744 ± 2.723 |
| bin_elementwise_mul_2048x2048 | **1.855 ± 0.054** | 1.880 ± 0.182 | 14.865 ± 0.591 | 16.210 ± 0.830 | 17.524 ± 2.625 |
| bin_elementwise_mul_512x512 | 0.076 ± 0.004 | **0.071 ± 0.001** | 0.096 ± 0.005 | 0.256 ± 0.018 | 0.352 ± 0.818 |
| bin_matmul_1024 | 50.786 ± 0.224 | **42.748 ± 0.418** | 52.299 ± 0.204 | 46.345 ± 0.251 | 66.088 ± 18.318 |
| bin_matmul_256 | 0.946 ± 0.035 | **0.694 ± 0.007** | 1.243 ± 0.012 | 0.876 ± 0.485 | 1.085 ± 0.015 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.000 | **0.007 ± 0.000** | 0.020 ± 0.001 | 0.135 ± 0.020 | 0.010 ± 0.000 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.032 ± 0.001 | **0.030 ± 0.000** | 0.094 ± 0.004 | 0.173 ± 0.029 | 0.036 ± 0.004 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.000 | **0.012 ± 0.001** | 0.030 ± 0.011 | 0.076 ± 0.017 | 0.018 ± 0.001 |
| bin_outer_product_4096 | 57.841 ± 0.469 | 57.519 ± 0.452 | **47.243 ± 0.995** | 48.845 ± 1.789 | 274.627 ± 54.456 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 48.928 ± 0.461 | **44.360 ± 0.258** | 54.028 ± 0.474 | 49.040 ± 0.224 | 51.998 ± 13.438 |
| gm_queen5_5_3.wcsp | 6184.511 ± 84.271 | 6106.887 ± 47.199 | **5843.496 ± 26.968** | 6063.234 ± 77.239 | 8404.302 ± 341.427 |
| lm_batch_likelihood_brackets_4_4d | 20.560 ± 0.448 | 26.647 ± 1.717 | 25.386 ± 0.122 | **11.917 ± 0.397** | 52.148 ± 28.447 |
| lm_batch_likelihood_sentence_3_12d | **50.895 ± 1.151** | 112.540 ± 26.499 | 59.226 ± 0.464 | 82.972 ± 1.219 | 97.827 ± 11.591 |
| lm_batch_likelihood_sentence_4_4d | 21.438 ± 1.230 | 31.565 ± 2.545 | 28.168 ± 0.640 | **13.959 ± 0.122** | 29.450 ± 25.923 |
| nary_matmul_chain_64 | **0.046 ± 0.002** | 0.049 ± 0.014 | 0.050 ± 0.003 | 0.122 ± 0.009 | 0.108 ± 0.016 |
| str_matrix_chain_multiplication_100 | 13.667 ± 0.285 | 13.558 ± 0.219 | 15.710 ± 0.156 | **12.789 ± 0.122** | 15.641 ± 0.762 |
| str_mps_varying_inner_product_200 | **18.164 ± 0.234** | 20.449 ± 0.229 | 20.941 ± 1.180 | 20.369 ± 0.743 | 25.836 ± 34.202 |
| str_nw_mera_closed_120 | **1500.560 ± 10.121** | 1631.862 ± 36.248 | 1779.067 ± 10.623 | 1904.191 ± 10.717 | 1762.918 ± 41.831 |
| str_nw_mera_open_26 | 1164.410 ± 8.453 | 1374.465 ± 13.893 | **1106.764 ± 8.746** | 1584.070 ± 34.116 | 1548.708 ± 65.870 |
| tensornetwork_permutation_focus_step409_316 | **430.771 ± 3.219** | 533.457 ± 2.446 | 763.128 ± 14.533 | 660.310 ± 11.122 | 524.805 ± 19.557 |
| tensornetwork_permutation_light_415 | **422.271 ± 3.587** | 469.777 ± 1.302 | 731.435 ± 2.030 | 691.798 ± 12.728 | 474.995 ± 51.796 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.196 ± 0.264 | 3.332 ± 0.046 | 3.671 ± 0.123 | **3.142 ± 0.040** | 10.142 ± 6.244 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.625 ± 0.013 | **0.483 ± 0.011** | 0.548 ± 0.015 | 0.616 ± 0.011 | 2.138 ± 1.603 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.538 ± 0.012 | 0.504 ± 0.009 | **0.126 ± 0.005** | 0.484 ± 0.018 | 2.946 ± 1.010 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.929 ± 0.006 | 0.506 ± 0.009 | **0.125 ± 0.002** | 0.486 ± 0.013 | 4.744 ± 2.723 |
| bin_elementwise_mul_2048x2048 | **1.855 ± 0.054** | 1.880 ± 0.182 | 14.865 ± 0.591 | 16.210 ± 0.830 | 17.524 ± 2.625 |
| bin_elementwise_mul_512x512 | 0.076 ± 0.004 | **0.071 ± 0.001** | 0.096 ± 0.005 | 0.256 ± 0.018 | 0.352 ± 0.818 |
| bin_matmul_1024 | 50.786 ± 0.224 | **42.748 ± 0.418** | 52.299 ± 0.204 | 46.345 ± 0.251 | 66.088 ± 18.318 |
| bin_matmul_256 | 0.946 ± 0.035 | **0.694 ± 0.007** | 1.243 ± 0.012 | 0.876 ± 0.485 | 1.085 ± 0.015 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.000 | **0.007 ± 0.000** | 0.020 ± 0.001 | 0.135 ± 0.020 | 0.010 ± 0.000 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.032 ± 0.001 | **0.030 ± 0.000** | 0.094 ± 0.004 | 0.173 ± 0.029 | 0.036 ± 0.004 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.000 | **0.012 ± 0.001** | 0.030 ± 0.011 | 0.076 ± 0.017 | 0.018 ± 0.001 |
| bin_outer_product_4096 | 57.841 ± 0.469 | 57.519 ± 0.452 | **47.243 ± 0.995** | 48.845 ± 1.789 | 274.627 ± 54.456 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 48.928 ± 0.461 | **44.360 ± 0.258** | 54.028 ± 0.474 | 49.040 ± 0.224 | 51.998 ± 13.438 |
| gm_queen5_5_3.wcsp | **2097.984 ± 9.590** | 2392.744 ± 16.957 | 2458.388 ± 21.025 | 2260.407 ± 28.530 | 2985.681 ± 22.228 |
| lm_batch_likelihood_brackets_4_4d | 21.793 ± 0.783 | 26.728 ± 1.520 | 27.509 ± 0.333 | **14.001 ± 2.143** | 57.606 ± 26.375 |
| lm_batch_likelihood_sentence_3_12d | **53.631 ± 0.725** | 133.087 ± 0.714 | 58.043 ± 0.489 | 107.954 ± 1.111 | 96.352 ± 27.653 |
| lm_batch_likelihood_sentence_4_4d | 22.768 ± 0.996 | 32.114 ± 1.540 | 29.494 ± 0.338 | **12.803 ± 0.200** | 32.186 ± 36.363 |
| nary_matmul_chain_64 | **0.046 ± 0.002** | 0.049 ± 0.014 | 0.050 ± 0.003 | 0.122 ± 0.009 | 0.108 ± 0.016 |
| str_matrix_chain_multiplication_100 | 14.689 ± 0.164 | 14.765 ± 0.184 | 15.613 ± 0.177 | **12.642 ± 0.124** | 15.488 ± 0.787 |
| str_mps_varying_inner_product_200 | 21.519 ± 0.226 | 20.303 ± 0.138 | **19.819 ± 0.125** | 22.659 ± 0.362 | 21.026 ± 10.048 |
| str_nw_mera_closed_120 | 1367.804 ± 7.316 | **1327.330 ± 9.094** | 1401.240 ± 10.101 | 1498.777 ± 21.427 | 1431.104 ± 58.359 |
| str_nw_mera_open_26 | 1156.239 ± 10.761 | 1373.018 ± 9.736 | **1125.177 ± 4.036** | 1622.171 ± 5.893 | 1527.837 ± 37.545 |
| tensornetwork_permutation_focus_step409_316 | **430.771 ± 3.219** | 533.457 ± 2.446 | 763.128 ± 14.533 | 660.310 ± 11.122 | 524.805 ± 19.557 |
| tensornetwork_permutation_light_415 | **422.271 ± 3.587** | 469.777 ± 1.302 | 731.435 ± 2.030 | 691.798 ± 12.728 | 474.995 ± 51.796 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261031_224329/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261031_224329/run.yaml`
- Timestamp: `2026-10-31T22:43:29.944486+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261031_224329`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

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

- `data/results/amd-cpu/cpu/einsum/20261031_224329/jax_cpu_t4_20261031_224329.log`
- `data/results/amd-cpu/cpu/einsum/20261031_224329/julia_omeinsum_t4_20261031_224329.log`
- `data/results/amd-cpu/cpu/einsum/20261031_224329/pytorch_cpu_t4_20261031_224329.log`
- `data/results/amd-cpu/cpu/einsum/20261031_224329/tenferro_eager_t4_20261031_224329.log`
- `data/results/amd-cpu/cpu/einsum/20261031_224329/tenferro_trace_t4_20261031_224329.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 15.999 ± 0.006 | 2.372 ± 0.027 | **1.831 ± 0.918** | 2.280 ± 0.680 | 9.343 ± 12.171 |
| bin_batched_matmul_b32_m64_n64_k64 | 11.998 ± 0.011 | 0.553 ± 0.015 | **0.290 ± 0.011** | 0.617 ± 0.024 | 0.580 ± 1.362 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 2.997 ± 0.042 | 2.998 ± 0.015 | **0.078 ± 0.007** | 0.362 ± 0.034 | 2.472 ± 2.541 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.999 ± 0.008 | 5.998 ± 0.004 | **0.078 ± 0.009** | 0.467 ± 0.020 | 4.725 ± 5.485 |
| bin_elementwise_mul_2048x2048 | 1.758 ± 3.081 | **0.782 ± 0.049** | 5.242 ± 0.213 | 8.696 ± 0.598 | 17.756 ± 0.282 |
| bin_elementwise_mul_512x512 | 0.089 ± 0.046 | **0.046 ± 0.000** | 0.064 ± 0.008 | 0.231 ± 0.023 | 0.109 ± 0.940 |
| bin_matmul_1024 | 37.999 ± 2.174 | **21.056 ± 0.178** | 30.846 ± 0.361 | 26.691 ± 0.503 | 41.789 ± 24.139 |
| bin_matmul_256 | 5.064 ± 4.715 | **0.396 ± 0.015** | 0.821 ± 0.010 | 0.609 ± 0.055 | 0.677 ± 0.027 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.013 ± 0.000 | **0.008 ± 0.000** | 0.029 ± 0.002 | 0.135 ± 0.019 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.051 ± 0.001 | 0.049 ± 0.002 | 0.139 ± 0.017 | 0.181 ± 0.018 | **0.042 ± 0.028** |
| bin_omeinsum_matmul_10x10 | 0.030 ± 0.040 | **0.015 ± 0.001** | 0.029 ± 0.001 | 0.077 ± 0.015 | 0.020 ± 0.001 |
| bin_outer_product_4096 | 19.465 ± 0.586 | 20.085 ± 0.522 | **15.545 ± 0.333** | 26.814 ± 1.895 | 261.621 ± 1.805 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 33.922 ± 6.985 | **24.151 ± 0.122** | 25.294 ± 0.276 | 28.236 ± 1.767 | 38.769 ± 17.118 |
| gm_queen5_5_3.wcsp | 5146.293 ± 65.037 | 4389.603 ± 62.023 | **2389.371 ± 54.710** | 3854.829 ± 76.963 | 7675.670 ± 166.274 |
| lm_batch_likelihood_brackets_4_4d | 132.005 ± 44.027 | 250.879 ± 179.406 | **12.924 ± 0.234** | 18.158 ± 1.214 | 30.015 ± 13.056 |
| lm_batch_likelihood_sentence_3_12d | 78.090 ± 27.860 | 148.270 ± 11.043 | **28.873 ± 0.804** | 38.969 ± 5.182 | 87.112 ± 23.644 |
| lm_batch_likelihood_sentence_4_4d | 125.932 ± 17.107 | 162.659 ± 31.362 | **13.681 ± 0.244** | 19.116 ± 4.036 | 51.261 ± 28.307 |
| nary_matmul_chain_64 | 0.061 ± 0.001 | **0.050 ± 0.005** | 0.060 ± 0.014 | 0.113 ± 0.020 | 0.105 ± 0.045 |
| str_matrix_chain_multiplication_100 | **7.376 ± 0.078** | 7.933 ± 0.069 | 9.605 ± 0.134 | 12.927 ± 0.633 | 13.347 ± 1.306 |
| str_mps_varying_inner_product_200 | **21.161 ± 0.239** | 23.578 ± 0.499 | 22.406 ± 0.650 | 25.394 ± 0.580 | 26.268 ± 17.004 |
| str_nw_mera_closed_120 | 930.934 ± 9.795 | 1056.074 ± 10.714 | **853.486 ± 9.916** | 924.662 ± 19.895 | 1110.428 ± 37.743 |
| str_nw_mera_open_26 | 834.339 ± 10.348 | 825.117 ± 78.898 | **521.997 ± 6.418** | 780.511 ± 10.403 | 1091.974 ± 34.910 |
| tensornetwork_permutation_focus_step409_316 | 405.031 ± 3.852 | 430.295 ± 8.763 | **301.963 ± 1.351** | 377.578 ± 10.249 | 481.732 ± 46.192 |
| tensornetwork_permutation_light_415 | 408.255 ± 8.808 | 399.483 ± 12.174 | **300.334 ± 5.509** | 394.488 ± 10.246 | 464.850 ± 66.420 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 15.999 ± 0.006 | 2.372 ± 0.027 | **1.831 ± 0.918** | 2.280 ± 0.680 | 9.343 ± 12.171 |
| bin_batched_matmul_b32_m64_n64_k64 | 11.998 ± 0.011 | 0.553 ± 0.015 | **0.290 ± 0.011** | 0.617 ± 0.024 | 0.580 ± 1.362 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 2.997 ± 0.042 | 2.998 ± 0.015 | **0.078 ± 0.007** | 0.362 ± 0.034 | 2.472 ± 2.541 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.999 ± 0.008 | 5.998 ± 0.004 | **0.078 ± 0.009** | 0.467 ± 0.020 | 4.725 ± 5.485 |
| bin_elementwise_mul_2048x2048 | 1.758 ± 3.081 | **0.782 ± 0.049** | 5.242 ± 0.213 | 8.696 ± 0.598 | 17.756 ± 0.282 |
| bin_elementwise_mul_512x512 | 0.089 ± 0.046 | **0.046 ± 0.000** | 0.064 ± 0.008 | 0.231 ± 0.023 | 0.109 ± 0.940 |
| bin_matmul_1024 | 37.999 ± 2.174 | **21.056 ± 0.178** | 30.846 ± 0.361 | 26.691 ± 0.503 | 41.789 ± 24.139 |
| bin_matmul_256 | 5.064 ± 4.715 | **0.396 ± 0.015** | 0.821 ± 0.010 | 0.609 ± 0.055 | 0.677 ± 0.027 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.013 ± 0.000 | **0.008 ± 0.000** | 0.029 ± 0.002 | 0.135 ± 0.019 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.051 ± 0.001 | 0.049 ± 0.002 | 0.139 ± 0.017 | 0.181 ± 0.018 | **0.042 ± 0.028** |
| bin_omeinsum_matmul_10x10 | 0.030 ± 0.040 | **0.015 ± 0.001** | 0.029 ± 0.001 | 0.077 ± 0.015 | 0.020 ± 0.001 |
| bin_outer_product_4096 | 19.465 ± 0.586 | 20.085 ± 0.522 | **15.545 ± 0.333** | 26.814 ± 1.895 | 261.621 ± 1.805 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 33.922 ± 6.985 | **24.151 ± 0.122** | 25.294 ± 0.276 | 28.236 ± 1.767 | 38.769 ± 17.118 |
| gm_queen5_5_3.wcsp | 1769.846 ± 17.800 | 1529.772 ± 73.421 | **1070.573 ± 17.018** | 1514.874 ± 14.787 | 2759.918 ± 46.959 |
| lm_batch_likelihood_brackets_4_4d | 144.874 ± 40.022 | 185.254 ± 40.185 | **13.623 ± 0.426** | 18.291 ± 0.788 | 40.541 ± 24.203 |
| lm_batch_likelihood_sentence_3_12d | 124.011 ± 45.997 | 190.927 ± 39.194 | **29.046 ± 0.525** | 44.090 ± 9.874 | 97.513 ± 36.688 |
| lm_batch_likelihood_sentence_4_4d | 124.927 ± 50.530 | 180.120 ± 44.688 | **14.812 ± 0.551** | 18.511 ± 1.390 | 39.009 ± 40.114 |
| nary_matmul_chain_64 | 0.061 ± 0.001 | **0.050 ± 0.005** | 0.060 ± 0.014 | 0.113 ± 0.020 | 0.105 ± 0.045 |
| str_matrix_chain_multiplication_100 | **7.887 ± 0.048** | 8.485 ± 0.115 | 9.467 ± 0.183 | 12.813 ± 0.506 | 12.587 ± 0.798 |
| str_mps_varying_inner_product_200 | 26.048 ± 0.319 | 26.416 ± 0.357 | 25.039 ± 0.664 | 26.736 ± 0.328 | **22.309 ± 6.271** |
| str_nw_mera_closed_120 | 838.595 ± 3.497 | 807.248 ± 11.624 | **680.722 ± 3.605** | 741.156 ± 14.921 | 907.035 ± 82.238 |
| str_nw_mera_open_26 | 832.811 ± 6.012 | 829.046 ± 45.437 | **528.966 ± 3.274** | 792.135 ± 9.083 | 1114.230 ± 32.279 |
| tensornetwork_permutation_focus_step409_316 | 405.031 ± 3.852 | 430.295 ± 8.763 | **301.963 ± 1.351** | 377.578 ± 10.249 | 481.732 ± 46.192 |
| tensornetwork_permutation_light_415 | 408.255 ± 8.808 | 399.483 ± 12.174 | **300.334 ± 5.509** | 394.488 ± 10.246 | 464.850 ± 66.420 |

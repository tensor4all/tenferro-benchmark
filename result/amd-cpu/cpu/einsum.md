# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`
- Benchmark commit: `ce2efd37d9e132703dfd2ebefbf4353249cf53ea`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261105_180410/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261105_180410/run.yaml`
- Timestamp: `2026-11-05T18:04:10.102563+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261105_180410`.

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

- `data/results/amd-cpu/cpu/einsum/20261105_180410/jax_cpu_t1_20261105_180410.log`
- `data/results/amd-cpu/cpu/einsum/20261105_180410/julia_omeinsum_t1_20261105_180410.log`
- `data/results/amd-cpu/cpu/einsum/20261105_180410/pytorch_cpu_t1_20261105_180410.log`
- `data/results/amd-cpu/cpu/einsum/20261105_180410/tenferro_eager_t1_20261105_180410.log`
- `data/results/amd-cpu/cpu/einsum/20261105_180410/tenferro_trace_t1_20261105_180410.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.253 ± 0.172 | **3.869 ± 0.032** | 4.325 ± 0.237 | 4.363 ± 0.132 | 12.293 ± 10.345 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.797 ± 0.046 | **0.599 ± 0.015** | 0.667 ± 0.029 | 0.798 ± 0.018 | 1.034 ± 0.301 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.742 ± 0.010 | 0.736 ± 0.015 | **0.227 ± 0.084** | 0.570 ± 0.028 | 0.909 ± 0.570 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.188 ± 0.015 | 0.738 ± 0.023 | **0.237 ± 0.018** | 0.575 ± 0.026 | 5.082 ± 9.559 |
| bin_elementwise_mul_2048x2048 | 2.193 ± 0.104 | **1.965 ± 0.186** | 17.671 ± 0.347 | 19.370 ± 0.485 | 20.784 ± 2.744 |
| bin_elementwise_mul_512x512 | 0.157 ± 0.019 | **0.124 ± 0.010** | 0.177 ± 0.028 | 0.364 ± 0.027 | 0.211 ± 0.057 |
| bin_matmul_1024 | 58.257 ± 0.625 | **49.002 ± 0.438** | 60.466 ± 0.551 | 63.458 ± 4.436 | 77.772 ± 25.769 |
| bin_matmul_256 | 1.041 ± 0.014 | **0.816 ± 0.019** | 1.359 ± 0.013 | 1.170 ± 0.027 | 1.285 ± 0.056 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.017 ± 0.007 | **0.010 ± 0.002** | 0.035 ± 0.003 | 0.203 ± 0.023 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.044 ± 0.029 | **0.040 ± 0.000** | 0.143 ± 0.007 | 0.293 ± 0.061 | 0.080 ± 0.059 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.001 | **0.018 ± 0.001** | 0.034 ± 0.002 | 0.178 ± 0.017 | 0.026 ± 0.002 |
| bin_outer_product_4096 | 69.147 ± 0.416 | 69.133 ± 0.664 | 61.014 ± 0.976 | **60.680 ± 0.805** | 327.370 ± 6.130 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 60.301 ± 0.376 | **51.400 ± 0.788** | 62.412 ± 0.744 | 65.558 ± 1.550 | 77.360 ± 34.278 |
| gm_queen5_5_3.wcsp | 7502.605 ± 118.799 | 7050.248 ± 48.020 | **6448.085 ± 42.826** | 8466.562 ± 123.347 | 9786.547 ± 253.573 |
| lm_batch_likelihood_brackets_4_4d | 26.058 ± 1.441 | 37.010 ± 3.221 | 35.438 ± 0.795 | **15.405 ± 0.467** | 31.627 ± 9.724 |
| lm_batch_likelihood_sentence_3_12d | **60.586 ± 1.039** | 160.085 ± 5.696 | 111.982 ± 0.813 | 98.686 ± 2.375 | 112.615 ± 14.513 |
| lm_batch_likelihood_sentence_4_4d | 27.216 ± 0.282 | 40.706 ± 3.012 | 37.198 ± 0.586 | **17.051 ± 0.762** | 68.470 ± 27.077 |
| nary_matmul_chain_64 | **0.059 ± 0.023** | 0.065 ± 0.022 | 0.080 ± 0.004 | 0.202 ± 0.017 | 0.118 ± 0.027 |
| str_matrix_chain_multiplication_100 | **14.972 ± 0.201** | 16.131 ± 0.291 | 16.398 ± 0.234 | 15.388 ± 0.301 | 17.140 ± 0.386 |
| str_mps_varying_inner_product_200 | **20.951 ± 0.386** | 22.222 ± 0.418 | 26.435 ± 0.616 | 23.919 ± 0.452 | 26.720 ± 33.239 |
| str_nw_mera_closed_120 | **1699.896 ± 13.957** | 1914.985 ± 18.944 | 2016.089 ± 4.792 | 2536.633 ± 6.349 | 1908.354 ± 16.531 |
| str_nw_mera_open_26 | 1366.765 ± 8.902 | 1532.816 ± 16.354 | **1288.160 ± 3.369** | 1983.401 ± 4.737 | 1718.944 ± 80.972 |
| tensornetwork_permutation_focus_step409_316 | **499.236 ± 7.081** | 579.741 ± 2.456 | 895.331 ± 7.592 | 736.185 ± 94.629 | 603.464 ± 41.457 |
| tensornetwork_permutation_light_415 | **487.050 ± 3.908** | 509.846 ± 2.304 | 864.832 ± 4.466 | 767.992 ± 19.827 | 620.946 ± 5.629 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.253 ± 0.172 | **3.869 ± 0.032** | 4.325 ± 0.237 | 4.363 ± 0.132 | 12.293 ± 10.345 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.797 ± 0.046 | **0.599 ± 0.015** | 0.667 ± 0.029 | 0.798 ± 0.018 | 1.034 ± 0.301 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.742 ± 0.010 | 0.736 ± 0.015 | **0.227 ± 0.084** | 0.570 ± 0.028 | 0.909 ± 0.570 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.188 ± 0.015 | 0.738 ± 0.023 | **0.237 ± 0.018** | 0.575 ± 0.026 | 5.082 ± 9.559 |
| bin_elementwise_mul_2048x2048 | 2.193 ± 0.104 | **1.965 ± 0.186** | 17.671 ± 0.347 | 19.370 ± 0.485 | 20.784 ± 2.744 |
| bin_elementwise_mul_512x512 | 0.157 ± 0.019 | **0.124 ± 0.010** | 0.177 ± 0.028 | 0.364 ± 0.027 | 0.211 ± 0.057 |
| bin_matmul_1024 | 58.257 ± 0.625 | **49.002 ± 0.438** | 60.466 ± 0.551 | 63.458 ± 4.436 | 77.772 ± 25.769 |
| bin_matmul_256 | 1.041 ± 0.014 | **0.816 ± 0.019** | 1.359 ± 0.013 | 1.170 ± 0.027 | 1.285 ± 0.056 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.017 ± 0.007 | **0.010 ± 0.002** | 0.035 ± 0.003 | 0.203 ± 0.023 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.044 ± 0.029 | **0.040 ± 0.000** | 0.143 ± 0.007 | 0.293 ± 0.061 | 0.080 ± 0.059 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.001 | **0.018 ± 0.001** | 0.034 ± 0.002 | 0.178 ± 0.017 | 0.026 ± 0.002 |
| bin_outer_product_4096 | 69.147 ± 0.416 | 69.133 ± 0.664 | 61.014 ± 0.976 | **60.680 ± 0.805** | 327.370 ± 6.130 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 60.301 ± 0.376 | **51.400 ± 0.788** | 62.412 ± 0.744 | 65.558 ± 1.550 | 77.360 ± 34.278 |
| gm_queen5_5_3.wcsp | 2529.631 ± 23.640 | 2748.801 ± 30.767 | **2499.451 ± 64.969** | 2852.689 ± 22.714 | 3467.810 ± 51.433 |
| lm_batch_likelihood_brackets_4_4d | 25.093 ± 0.864 | 36.264 ± 0.681 | 36.939 ± 0.595 | **15.531 ± 0.432** | 72.298 ± 31.785 |
| lm_batch_likelihood_sentence_3_12d | **57.970 ± 2.296** | 165.640 ± 7.921 | 67.967 ± 0.611 | 72.496 ± 1.659 | 121.841 ± 26.009 |
| lm_batch_likelihood_sentence_4_4d | 26.099 ± 0.488 | 42.054 ± 1.403 | 38.837 ± 0.259 | **16.588 ± 0.441** | 47.922 ± 39.514 |
| nary_matmul_chain_64 | **0.059 ± 0.023** | 0.065 ± 0.022 | 0.080 ± 0.004 | 0.202 ± 0.017 | 0.118 ± 0.027 |
| str_matrix_chain_multiplication_100 | 16.300 ± 0.403 | 16.124 ± 0.248 | 16.557 ± 0.283 | **14.872 ± 0.884** | 17.479 ± 0.811 |
| str_mps_varying_inner_product_200 | 24.784 ± 0.361 | 24.833 ± 0.505 | 26.212 ± 0.343 | **24.708 ± 0.724** | 25.866 ± 9.916 |
| str_nw_mera_closed_120 | 1549.154 ± 11.069 | **1539.325 ± 10.130** | 1617.343 ± 11.345 | 2079.308 ± 8.472 | 1555.836 ± 35.863 |
| str_nw_mera_open_26 | 1347.520 ± 17.545 | 1511.016 ± 12.824 | **1316.780 ± 10.744** | 2061.631 ± 16.448 | 1682.368 ± 80.681 |
| tensornetwork_permutation_focus_step409_316 | **499.236 ± 7.081** | 579.741 ± 2.456 | 895.331 ± 7.592 | 736.185 ± 94.629 | 603.464 ± 41.457 |
| tensornetwork_permutation_light_415 | **487.050 ± 3.908** | 509.846 ± 2.304 | 864.832 ± 4.466 | 767.992 ± 19.827 | 620.946 ± 5.629 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261105_184557/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261105_184557/run.yaml`
- Timestamp: `2026-11-05T18:45:57.792006+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261105_184557`.

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

- `data/results/amd-cpu/cpu/einsum/20261105_184557/jax_cpu_t4_20261105_184557.log`
- `data/results/amd-cpu/cpu/einsum/20261105_184557/julia_omeinsum_t4_20261105_184557.log`
- `data/results/amd-cpu/cpu/einsum/20261105_184557/pytorch_cpu_t4_20261105_184557.log`
- `data/results/amd-cpu/cpu/einsum/20261105_184557/tenferro_eager_t4_20261105_184557.log`
- `data/results/amd-cpu/cpu/einsum/20261105_184557/tenferro_trace_t4_20261105_184557.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 15.546 ± 0.583 | 2.830 ± 0.027 | **2.363 ± 3.134** | 2.932 ± 0.669 | 12.083 ± 12.533 |
| bin_batched_matmul_b32_m64_n64_k64 | 3.011 ± 2.918 | 0.671 ± 0.041 | **0.385 ± 0.011** | 0.796 ± 0.018 | 0.979 ± 1.693 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.331 ± 0.032 | 5.994 ± 2.999 | **0.204 ± 0.018** | 0.460 ± 0.040 | 3.997 ± 1.569 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.995 ± 0.977 | 2.999 ± 0.002 | **0.214 ± 0.018** | 0.463 ± 0.039 | 6.785 ± 4.118 |
| bin_elementwise_mul_2048x2048 | 3.005 ± 0.039 | **0.887 ± 0.014** | 5.983 ± 0.202 | 10.106 ± 0.183 | 19.684 ± 0.305 |
| bin_elementwise_mul_512x512 | 0.143 ± 4.577 | **0.131 ± 0.003** | 0.159 ± 0.013 | 0.395 ± 0.035 | 0.159 ± 0.241 |
| bin_matmul_1024 | 41.005 ± 2.050 | **24.691 ± 0.206** | 35.533 ± 0.277 | 36.084 ± 0.381 | 52.632 ± 27.268 |
| bin_matmul_256 | 3.836 ± 1.904 | **0.467 ± 0.019** | 0.859 ± 0.015 | 0.845 ± 0.135 | 1.220 ± 0.931 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.018 ± 0.001 | **0.011 ± 0.000** | 0.051 ± 0.007 | 0.201 ± 0.027 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.072 ± 0.003 | 0.065 ± 0.004 | 0.222 ± 0.021 | 0.250 ± 0.045 | **0.051 ± 0.002** |
| bin_omeinsum_matmul_10x10 | 0.048 ± 0.008 | 0.130 ± 0.008 | 0.051 ± 0.011 | 0.150 ± 0.020 | **0.027 ± 0.001** |
| bin_outer_product_4096 | 23.630 ± 0.303 | 23.540 ± 0.315 | **19.933 ± 0.397** | 32.174 ± 0.609 | 307.947 ± 14.961 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 42.012 ± 7.685 | **27.541 ± 0.254** | 30.750 ± 0.338 | 38.670 ± 1.176 | 47.189 ± 23.896 |
| gm_queen5_5_3.wcsp | 6426.140 ± 105.554 | 5300.596 ± 99.242 | **2708.782 ± 27.187** | 5523.960 ± 56.386 | 8857.291 ± 151.916 |
| lm_batch_likelihood_brackets_4_4d | 127.022 ± 36.143 | 131.835 ± 56.836 | **17.133 ± 0.873** | 23.423 ± 1.265 | 74.229 ± 44.699 |
| lm_batch_likelihood_sentence_3_12d | 76.999 ± 14.197 | 174.266 ± 35.798 | 59.233 ± 1.085 | **57.727 ± 4.613** | 123.787 ± 30.318 |
| lm_batch_likelihood_sentence_4_4d | 132.000 ± 11.699 | 143.877 ± 20.231 | **17.657 ± 0.311** | 24.498 ± 1.072 | 76.550 ± 35.923 |
| nary_matmul_chain_64 | 0.076 ± 0.002 | **0.057 ± 0.001** | 0.092 ± 0.022 | 0.280 ± 0.039 | 0.150 ± 0.076 |
| str_matrix_chain_multiplication_100 | **9.083 ± 0.304** | 9.470 ± 0.353 | 10.921 ± 0.417 | 15.271 ± 0.411 | 14.877 ± 0.634 |
| str_mps_varying_inner_product_200 | 25.035 ± 0.473 | **23.790 ± 0.373** | 26.347 ± 0.361 | 31.490 ± 1.213 | 32.684 ± 18.079 |
| str_nw_mera_closed_120 | 1093.084 ± 13.743 | 1206.562 ± 20.515 | **966.122 ± 11.373** | 1187.723 ± 27.023 | 1284.085 ± 38.414 |
| str_nw_mera_open_26 | 1103.720 ± 16.091 | 944.342 ± 42.238 | **607.692 ± 7.169** | 958.043 ± 12.750 | 1293.317 ± 49.221 |
| tensornetwork_permutation_focus_step409_316 | 485.733 ± 3.833 | 494.525 ± 19.501 | **352.821 ± 4.265** | 416.601 ± 8.333 | 605.772 ± 17.618 |
| tensornetwork_permutation_light_415 | 470.960 ± 3.790 | 467.877 ± 15.594 | **349.132 ± 3.668** | 438.290 ± 29.480 | 615.272 ± 23.646 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 15.546 ± 0.583 | 2.830 ± 0.027 | **2.363 ± 3.134** | 2.932 ± 0.669 | 12.083 ± 12.533 |
| bin_batched_matmul_b32_m64_n64_k64 | 3.011 ± 2.918 | 0.671 ± 0.041 | **0.385 ± 0.011** | 0.796 ± 0.018 | 0.979 ± 1.693 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.331 ± 0.032 | 5.994 ± 2.999 | **0.204 ± 0.018** | 0.460 ± 0.040 | 3.997 ± 1.569 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.995 ± 0.977 | 2.999 ± 0.002 | **0.214 ± 0.018** | 0.463 ± 0.039 | 6.785 ± 4.118 |
| bin_elementwise_mul_2048x2048 | 3.005 ± 0.039 | **0.887 ± 0.014** | 5.983 ± 0.202 | 10.106 ± 0.183 | 19.684 ± 0.305 |
| bin_elementwise_mul_512x512 | 0.143 ± 4.577 | **0.131 ± 0.003** | 0.159 ± 0.013 | 0.395 ± 0.035 | 0.159 ± 0.241 |
| bin_matmul_1024 | 41.005 ± 2.050 | **24.691 ± 0.206** | 35.533 ± 0.277 | 36.084 ± 0.381 | 52.632 ± 27.268 |
| bin_matmul_256 | 3.836 ± 1.904 | **0.467 ± 0.019** | 0.859 ± 0.015 | 0.845 ± 0.135 | 1.220 ± 0.931 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.018 ± 0.001 | **0.011 ± 0.000** | 0.051 ± 0.007 | 0.201 ± 0.027 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.072 ± 0.003 | 0.065 ± 0.004 | 0.222 ± 0.021 | 0.250 ± 0.045 | **0.051 ± 0.002** |
| bin_omeinsum_matmul_10x10 | 0.048 ± 0.008 | 0.130 ± 0.008 | 0.051 ± 0.011 | 0.150 ± 0.020 | **0.027 ± 0.001** |
| bin_outer_product_4096 | 23.630 ± 0.303 | 23.540 ± 0.315 | **19.933 ± 0.397** | 32.174 ± 0.609 | 307.947 ± 14.961 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 42.012 ± 7.685 | **27.541 ± 0.254** | 30.750 ± 0.338 | 38.670 ± 1.176 | 47.189 ± 23.896 |
| gm_queen5_5_3.wcsp | 2141.328 ± 21.784 | 1769.994 ± 69.421 | **1159.547 ± 10.358** | 1905.587 ± 29.442 | 3168.273 ± 27.263 |
| lm_batch_likelihood_brackets_4_4d | 124.961 ± 111.669 | 133.698 ± 32.536 | **18.058 ± 0.150** | 23.674 ± 1.905 | 81.808 ± 43.909 |
| lm_batch_likelihood_sentence_3_12d | 78.137 ± 10.262 | 200.557 ± 25.484 | 61.053 ± 0.896 | **51.919 ± 11.562** | 112.272 ± 32.438 |
| lm_batch_likelihood_sentence_4_4d | 122.710 ± 62.298 | 120.153 ± 44.086 | **18.347 ± 0.092** | 24.301 ± 1.899 | 66.154 ± 39.576 |
| nary_matmul_chain_64 | 0.076 ± 0.002 | **0.057 ± 0.001** | 0.092 ± 0.022 | 0.280 ± 0.039 | 0.150 ± 0.076 |
| str_matrix_chain_multiplication_100 | **9.411 ± 0.205** | 9.593 ± 0.309 | 10.869 ± 0.287 | 15.712 ± 0.821 | 17.789 ± 0.398 |
| str_mps_varying_inner_product_200 | 29.627 ± 0.292 | **27.365 ± 0.312** | 28.192 ± 0.251 | 32.440 ± 0.847 | 34.125 ± 22.061 |
| str_nw_mera_closed_120 | 981.596 ± 23.890 | 921.460 ± 17.440 | **768.082 ± 4.730** | 988.120 ± 12.574 | 964.880 ± 60.665 |
| str_nw_mera_open_26 | 1112.964 ± 8.955 | 931.163 ± 92.363 | **604.691 ± 3.975** | 979.111 ± 13.252 | 1275.492 ± 61.932 |
| tensornetwork_permutation_focus_step409_316 | 485.733 ± 3.833 | 494.525 ± 19.501 | **352.821 ± 4.265** | 416.601 ± 8.333 | 605.772 ± 17.618 |
| tensornetwork_permutation_light_415 | 470.960 ± 3.790 | 467.877 ± 15.594 | **349.132 ± 3.668** | 438.290 ± 29.480 | 615.272 ± 23.646 |

# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `5c1d5a382dd88d20901598318730d850d3568ac8`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261116_180206/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261116_180206/run.yaml`
- Timestamp: `2026-11-16T18:02:06.211309+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261116_180206`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Intel(R) Xeon(R) 6975P-C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1016-aws-x86_64-with-glibc2.39`
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

- `data/results/amd-cpu/cpu/einsum/20261116_180206/jax_cpu_t1_20261116_180206.log`
- `data/results/amd-cpu/cpu/einsum/20261116_180206/julia_omeinsum_t1_20261116_180206.log`
- `data/results/amd-cpu/cpu/einsum/20261116_180206/pytorch_cpu_t1_20261116_180206.log`
- `data/results/amd-cpu/cpu/einsum/20261116_180206/tenferro_eager_t1_20261116_180206.log`
- `data/results/amd-cpu/cpu/einsum/20261116_180206/tenferro_trace_t1_20261116_180206.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.104 ± 0.021 | **1.339 ± 0.069** | 1.379 ± 0.042 | 3.866 ± 0.080 | 7.532 ± 3.212 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.418 ± 0.021 | **0.178 ± 0.011** | 0.229 ± 0.035 | 0.762 ± 0.019 | 0.617 ± 0.929 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.432 ± 0.027 | 0.428 ± 0.019 | **0.345 ± 0.022** | 0.534 ± 0.029 | 2.563 ± 2.254 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.886 ± 0.015 | 0.426 ± 0.025 | **0.340 ± 0.022** | 0.541 ± 0.026 | 4.690 ± 4.471 |
| bin_elementwise_mul_2048x2048 | **1.483 ± 0.030** | 1.485 ± 0.031 | 9.157 ± 0.208 | 14.926 ± 0.552 | 13.297 ± 2.114 |
| bin_elementwise_mul_512x512 | 0.219 ± 0.010 | **0.217 ± 0.012** | 0.242 ± 0.021 | 0.511 ± 0.052 | 0.237 ± 0.339 |
| bin_matmul_1024 | 21.404 ± 0.126 | **19.853 ± 0.108** | 20.163 ± 0.069 | 55.427 ± 0.564 | 51.449 ± 5.084 |
| bin_matmul_256 | 0.446 ± 0.021 | **0.340 ± 0.022** | 0.377 ± 0.028 | 1.002 ± 0.037 | 1.328 ± 0.565 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.001 | **0.005 ± 0.000** | 0.020 ± 0.006 | 0.109 ± 0.015 | 0.010 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.025 ± 0.006 | **0.023 ± 0.001** | 0.075 ± 0.002 | 0.181 ± 0.035 | 0.058 ± 0.020 |
| bin_omeinsum_matmul_10x10 | 0.012 ± 0.000 | **0.006 ± 0.000** | 0.015 ± 0.001 | 0.073 ± 0.007 | 0.015 ± 0.001 |
| bin_outer_product_4096 | 42.019 ± 1.127 | 41.875 ± 0.385 | **36.664 ± 0.422** | 40.109 ± 0.567 | 276.072 ± 33.037 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 31.597 ± 0.647 | **21.612 ± 0.166** | 31.154 ± 0.242 | 60.947 ± 1.072 | 42.894 ± 7.739 |
| gm_queen5_5_3.wcsp | 6478.206 ± 35.999 | 4527.432 ± 49.480 | **4247.046 ± 57.240** | 4757.797 ± 71.946 | 6496.118 ± 121.266 |
| lm_batch_likelihood_brackets_4_4d | 16.888 ± 0.173 | 29.340 ± 1.127 | 20.651 ± 0.432 | **15.098 ± 0.144** | 27.101 ± 12.619 |
| lm_batch_likelihood_sentence_3_12d | 28.090 ± 0.416 | 96.430 ± 1.770 | **25.040 ± 0.311** | 74.550 ± 1.083 | 59.334 ± 8.913 |
| lm_batch_likelihood_sentence_4_4d | 18.326 ± 0.224 | 34.977 ± 1.165 | 23.660 ± 0.356 | **18.279 ± 0.244** | 32.214 ± 15.575 |
| nary_matmul_chain_64 | 0.023 ± 0.001 | **0.020 ± 0.003** | 0.039 ± 0.011 | 0.135 ± 0.012 | 0.050 ± 0.024 |
| str_matrix_chain_multiplication_100 | **5.020 ± 0.205** | 6.572 ± 0.727 | 6.700 ± 0.458 | 14.686 ± 0.266 | 12.889 ± 2.736 |
| str_mps_varying_inner_product_200 | 9.131 ± 0.119 | 11.377 ± 0.275 | **7.731 ± 0.185** | 20.887 ± 1.410 | 18.339 ± 3.685 |
| str_nw_mera_closed_120 | **840.573 ± 9.050** | 988.014 ± 24.625 | 1006.092 ± 7.558 | 2070.442 ± 13.912 | 1225.443 ± 15.465 |
| str_nw_mera_open_26 | 798.514 ± 8.182 | 885.577 ± 5.367 | **655.210 ± 6.129** | 1715.862 ± 18.036 | 1151.197 ± 38.814 |
| tensornetwork_permutation_focus_step409_316 | **299.703 ± 3.224** | 446.009 ± 27.532 | 447.344 ± 5.490 | 653.572 ± 15.954 | 403.397 ± 10.798 |
| tensornetwork_permutation_light_415 | **304.781 ± 3.300** | 316.937 ± 4.368 | 414.221 ± 2.125 | 672.477 ± 20.099 | 407.525 ± 8.289 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.104 ± 0.021 | **1.339 ± 0.069** | 1.379 ± 0.042 | 3.866 ± 0.080 | 7.532 ± 3.212 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.418 ± 0.021 | **0.178 ± 0.011** | 0.229 ± 0.035 | 0.762 ± 0.019 | 0.617 ± 0.929 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.432 ± 0.027 | 0.428 ± 0.019 | **0.345 ± 0.022** | 0.534 ± 0.029 | 2.563 ± 2.254 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.886 ± 0.015 | 0.426 ± 0.025 | **0.340 ± 0.022** | 0.541 ± 0.026 | 4.690 ± 4.471 |
| bin_elementwise_mul_2048x2048 | **1.483 ± 0.030** | 1.485 ± 0.031 | 9.157 ± 0.208 | 14.926 ± 0.552 | 13.297 ± 2.114 |
| bin_elementwise_mul_512x512 | 0.219 ± 0.010 | **0.217 ± 0.012** | 0.242 ± 0.021 | 0.511 ± 0.052 | 0.237 ± 0.339 |
| bin_matmul_1024 | 21.404 ± 0.126 | **19.853 ± 0.108** | 20.163 ± 0.069 | 55.427 ± 0.564 | 51.449 ± 5.084 |
| bin_matmul_256 | 0.446 ± 0.021 | **0.340 ± 0.022** | 0.377 ± 0.028 | 1.002 ± 0.037 | 1.328 ± 0.565 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.001 | **0.005 ± 0.000** | 0.020 ± 0.006 | 0.109 ± 0.015 | 0.010 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.025 ± 0.006 | **0.023 ± 0.001** | 0.075 ± 0.002 | 0.181 ± 0.035 | 0.058 ± 0.020 |
| bin_omeinsum_matmul_10x10 | 0.012 ± 0.000 | **0.006 ± 0.000** | 0.015 ± 0.001 | 0.073 ± 0.007 | 0.015 ± 0.001 |
| bin_outer_product_4096 | 42.019 ± 1.127 | 41.875 ± 0.385 | **36.664 ± 0.422** | 40.109 ± 0.567 | 276.072 ± 33.037 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 31.597 ± 0.647 | **21.612 ± 0.166** | 31.154 ± 0.242 | 60.947 ± 1.072 | 42.894 ± 7.739 |
| gm_queen5_5_3.wcsp | **1790.239 ± 40.362** | 1813.070 ± 18.555 | 1791.749 ± 35.170 | 1927.160 ± 27.261 | 2334.314 ± 43.232 |
| lm_batch_likelihood_brackets_4_4d | 20.527 ± 0.140 | 28.421 ± 1.268 | 19.967 ± 0.264 | **14.841 ± 0.224** | 31.412 ± 14.360 |
| lm_batch_likelihood_sentence_3_12d | 33.128 ± 0.854 | 109.815 ± 0.882 | **30.933 ± 0.847** | 82.762 ± 2.235 | 88.739 ± 48.758 |
| lm_batch_likelihood_sentence_4_4d | 19.528 ± 0.198 | 34.329 ± 1.160 | 20.543 ± 0.423 | **16.581 ± 0.120** | 30.184 ± 43.310 |
| nary_matmul_chain_64 | 0.023 ± 0.001 | **0.020 ± 0.003** | 0.039 ± 0.011 | 0.135 ± 0.012 | 0.050 ± 0.024 |
| str_matrix_chain_multiplication_100 | **5.333 ± 0.132** | 5.566 ± 0.244 | 7.517 ± 0.172 | 14.333 ± 0.304 | 14.087 ± 1.128 |
| str_mps_varying_inner_product_200 | 9.079 ± 0.356 | **8.932 ± 0.548** | 9.310 ± 0.831 | 24.146 ± 0.503 | 17.837 ± 1.107 |
| str_nw_mera_closed_120 | **764.353 ± 11.918** | 787.229 ± 11.892 | 776.074 ± 3.702 | 1729.557 ± 24.058 | 1002.078 ± 16.591 |
| str_nw_mera_open_26 | 781.926 ± 4.744 | 866.683 ± 4.105 | **677.472 ± 11.604** | 1748.222 ± 33.457 | 1142.922 ± 61.635 |
| tensornetwork_permutation_focus_step409_316 | **299.703 ± 3.224** | 446.009 ± 27.532 | 447.344 ± 5.490 | 653.572 ± 15.954 | 403.397 ± 10.798 |
| tensornetwork_permutation_light_415 | **304.781 ± 3.300** | 316.937 ± 4.368 | 414.221 ± 2.125 | 672.477 ± 20.099 | 407.525 ± 8.289 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261116_183058/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261116_183058/run.yaml`
- Timestamp: `2026-11-16T18:30:58.322744+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261116_183058`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Intel(R) Xeon(R) 6975P-C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1016-aws-x86_64-with-glibc2.39`
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

- `data/results/amd-cpu/cpu/einsum/20261116_183058/jax_cpu_t4_20261116_183058.log`
- `data/results/amd-cpu/cpu/einsum/20261116_183058/julia_omeinsum_t4_20261116_183058.log`
- `data/results/amd-cpu/cpu/einsum/20261116_183058/pytorch_cpu_t4_20261116_183058.log`
- `data/results/amd-cpu/cpu/einsum/20261116_183058/tenferro_eager_t4_20261116_183058.log`
- `data/results/amd-cpu/cpu/einsum/20261116_183058/tenferro_trace_t4_20261116_183058.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.510 ± 0.114 | 1.624 ± 0.036 | **0.788 ± 0.019** | 2.444 ± 0.496 | 9.410 ± 4.552 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.638 ± 0.020 | 0.354 ± 0.061 | **0.136 ± 0.071** | 0.822 ± 0.188 | 0.617 ± 0.852 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.204 ± 0.016 | 0.199 ± 0.008 | **0.191 ± 0.015** | 0.411 ± 0.036 | 1.831 ± 2.209 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.253 ± 0.022 | 0.200 ± 0.015 | **0.192 ± 0.027** | 0.435 ± 0.056 | 3.661 ± 3.685 |
| bin_elementwise_mul_2048x2048 | 0.601 ± 0.018 | **0.594 ± 0.015** | 4.463 ± 0.253 | 8.530 ± 0.281 | 15.091 ± 0.932 |
| bin_elementwise_mul_512x512 | 0.114 ± 0.006 | **0.109 ± 0.005** | 0.140 ± 0.015 | 0.363 ± 0.025 | 0.237 ± 0.458 |
| bin_matmul_1024 | 24.063 ± 0.195 | 21.856 ± 0.052 | **10.506 ± 0.092** | 27.677 ± 0.302 | 43.572 ± 6.040 |
| bin_matmul_256 | 0.543 ± 0.029 | 0.419 ± 0.036 | **0.214 ± 0.015** | 0.716 ± 0.051 | 0.642 ± 0.186 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.010 ± 0.000 | **0.005 ± 0.000** | 0.021 ± 0.001 | 0.123 ± 0.021 | 0.008 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.028 ± 0.001 | **0.027 ± 0.001** | 0.093 ± 0.004 | 0.172 ± 0.040 | 0.056 ± 0.023 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.008 ± 0.000** | 0.018 ± 0.001 | 0.073 ± 0.009 | 0.015 ± 0.001 |
| bin_outer_product_4096 | 16.735 ± 0.388 | 16.431 ± 0.246 | **14.264 ± 0.170** | 22.596 ± 0.334 | 267.045 ± 28.958 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 33.483 ± 0.324 | 23.199 ± 0.229 | **13.978 ± 0.241** | 31.786 ± 1.644 | 30.672 ± 10.498 |
| gm_queen5_5_3.wcsp | 6240.627 ± 52.905 | 4034.801 ± 33.296 | **1834.586 ± 30.223** | 3077.690 ± 63.412 | 6175.200 ± 106.345 |
| lm_batch_likelihood_brackets_4_4d | 15.723 ± 0.283 | 37.692 ± 2.880 | **10.255 ± 0.231** | 20.775 ± 0.837 | 30.590 ± 13.365 |
| lm_batch_likelihood_sentence_3_12d | 34.437 ± 0.492 | 83.738 ± 17.123 | **13.910 ± 0.097** | 43.614 ± 9.661 | 57.741 ± 11.219 |
| lm_batch_likelihood_sentence_4_4d | 16.924 ± 0.335 | 40.421 ± 1.505 | **11.401 ± 0.097** | 21.977 ± 0.914 | 26.116 ± 9.141 |
| nary_matmul_chain_64 | 0.042 ± 0.008 | **0.034 ± 0.001** | **0.034 ± 0.002** | 0.113 ± 0.012 | 0.051 ± 0.020 |
| str_matrix_chain_multiplication_100 | 5.539 ± 0.102 | 6.299 ± 0.649 | **4.523 ± 0.091** | 12.244 ± 1.270 | 16.622 ± 1.187 |
| str_mps_varying_inner_product_200 | 55.312 ± 0.850 | 56.156 ± 0.433 | **9.057 ± 0.124** | 24.892 ± 0.666 | 19.065 ± 3.092 |
| str_nw_mera_closed_120 | 51258.273 ± 95.955 | 51428.182 ± 130.694 | **544.393 ± 11.675** | 902.631 ± 9.758 | 871.426 ± 64.021 |
| str_nw_mera_open_26 | 17264.617 ± 59.464 | 17247.785 ± 20.776 | **349.770 ± 10.970** | 761.888 ± 13.597 | 890.270 ± 28.795 |
| tensornetwork_permutation_focus_step409_316 | 306.214 ± 2.697 | 360.371 ± 8.852 | **218.496 ± 2.603** | 352.056 ± 12.450 | 395.717 ± 36.699 |
| tensornetwork_permutation_light_415 | 311.694 ± 1.353 | 292.848 ± 3.317 | **214.785 ± 2.743** | 370.246 ± 19.291 | 398.301 ± 27.350 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.510 ± 0.114 | 1.624 ± 0.036 | **0.788 ± 0.019** | 2.444 ± 0.496 | 9.410 ± 4.552 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.638 ± 0.020 | 0.354 ± 0.061 | **0.136 ± 0.071** | 0.822 ± 0.188 | 0.617 ± 0.852 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.204 ± 0.016 | 0.199 ± 0.008 | **0.191 ± 0.015** | 0.411 ± 0.036 | 1.831 ± 2.209 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.253 ± 0.022 | 0.200 ± 0.015 | **0.192 ± 0.027** | 0.435 ± 0.056 | 3.661 ± 3.685 |
| bin_elementwise_mul_2048x2048 | 0.601 ± 0.018 | **0.594 ± 0.015** | 4.463 ± 0.253 | 8.530 ± 0.281 | 15.091 ± 0.932 |
| bin_elementwise_mul_512x512 | 0.114 ± 0.006 | **0.109 ± 0.005** | 0.140 ± 0.015 | 0.363 ± 0.025 | 0.237 ± 0.458 |
| bin_matmul_1024 | 24.063 ± 0.195 | 21.856 ± 0.052 | **10.506 ± 0.092** | 27.677 ± 0.302 | 43.572 ± 6.040 |
| bin_matmul_256 | 0.543 ± 0.029 | 0.419 ± 0.036 | **0.214 ± 0.015** | 0.716 ± 0.051 | 0.642 ± 0.186 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.010 ± 0.000 | **0.005 ± 0.000** | 0.021 ± 0.001 | 0.123 ± 0.021 | 0.008 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.028 ± 0.001 | **0.027 ± 0.001** | 0.093 ± 0.004 | 0.172 ± 0.040 | 0.056 ± 0.023 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.008 ± 0.000** | 0.018 ± 0.001 | 0.073 ± 0.009 | 0.015 ± 0.001 |
| bin_outer_product_4096 | 16.735 ± 0.388 | 16.431 ± 0.246 | **14.264 ± 0.170** | 22.596 ± 0.334 | 267.045 ± 28.958 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 33.483 ± 0.324 | 23.199 ± 0.229 | **13.978 ± 0.241** | 31.786 ± 1.644 | 30.672 ± 10.498 |
| gm_queen5_5_3.wcsp | 1676.611 ± 21.427 | 1442.072 ± 14.189 | **759.021 ± 20.065** | 1286.805 ± 12.393 | 2216.397 ± 118.926 |
| lm_batch_likelihood_brackets_4_4d | 15.684 ± 0.228 | 39.772 ± 1.914 | **10.385 ± 0.195** | 20.803 ± 1.689 | 34.787 ± 7.677 |
| lm_batch_likelihood_sentence_3_12d | **34.052 ± 0.383** | 122.137 ± 0.951 | 37.144 ± 2.609 | 56.253 ± 6.163 | 76.671 ± 48.689 |
| lm_batch_likelihood_sentence_4_4d | 14.460 ± 0.327 | 44.165 ± 1.769 | **10.573 ± 0.153** | 23.283 ± 1.338 | 29.504 ± 20.162 |
| nary_matmul_chain_64 | 0.042 ± 0.008 | **0.034 ± 0.001** | **0.034 ± 0.002** | 0.113 ± 0.012 | 0.051 ± 0.020 |
| str_matrix_chain_multiplication_100 | 6.502 ± 0.281 | 7.538 ± 0.220 | **4.846 ± 0.394** | 12.849 ± 0.684 | 14.475 ± 0.903 |
| str_mps_varying_inner_product_200 | 54.697 ± 0.384 | 56.025 ± 0.370 | **13.355 ± 0.540** | 29.562 ± 0.875 | 17.868 ± 1.898 |
| str_nw_mera_closed_120 | 46478.732 ± 94.385 | 46496.003 ± 148.441 | **443.026 ± 13.709** | 811.202 ± 13.691 | 666.731 ± 86.069 |
| str_nw_mera_open_26 | 17261.736 ± 19.152 | 17251.841 ± 74.302 | **354.628 ± 9.389** | 798.973 ± 21.133 | 862.567 ± 40.972 |
| tensornetwork_permutation_focus_step409_316 | 306.214 ± 2.697 | 360.371 ± 8.852 | **218.496 ± 2.603** | 352.056 ± 12.450 | 395.717 ± 36.699 |
| tensornetwork_permutation_light_415 | 311.694 ± 1.353 | 292.848 ± 3.317 | **214.785 ± 2.743** | 370.246 ± 19.291 | 398.301 ± 27.350 |

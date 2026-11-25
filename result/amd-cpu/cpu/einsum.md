# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `74775e4a2ad3480fb9ee9f859e0037b8cac731f2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261125_175248/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261125_175248/run.yaml`
- Timestamp: `2026-11-25T17:52:48.228690+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261125_175248`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
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

- tenferro-rs features: `native`

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

- `data/results/amd-cpu/cpu/einsum/20261125_175248/jax_cpu_t1_20261125_175248.log`
- `data/results/amd-cpu/cpu/einsum/20261125_175248/julia_omeinsum_t1_20261125_175248.log`
- `data/results/amd-cpu/cpu/einsum/20261125_175248/pytorch_cpu_t1_20261125_175248.log`
- `data/results/amd-cpu/cpu/einsum/20261125_175248/tenferro_eager_t1_20261125_175248.log`
- `data/results/amd-cpu/cpu/einsum/20261125_175248/tenferro_trace_t1_20261125_175248.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.144 ± 0.087 | **2.918 ± 0.008** | 4.518 ± 0.133 | 4.386 ± 0.189 | 9.968 ± 2.194 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.561 ± 0.017 | **0.410 ± 0.020** | 0.654 ± 0.013 | 0.838 ± 0.013 | 1.550 ± 1.153 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.750 ± 0.002 | 0.743 ± 0.008 | **0.209 ± 0.025** | 0.581 ± 0.023 | 3.905 ± 2.542 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.189 ± 0.015 | 0.759 ± 0.009 | **0.206 ± 0.009** | 0.566 ± 0.022 | 5.024 ± 5.771 |
| bin_elementwise_mul_2048x2048 | 2.179 ± 0.064 | **1.846 ± 0.154** | 17.545 ± 0.341 | 19.197 ± 0.360 | 18.941 ± 4.718 |
| bin_elementwise_mul_512x512 | 0.150 ± 0.011 | **0.137 ± 0.029** | 0.160 ± 0.025 | 0.359 ± 0.035 | 0.179 ± 0.748 |
| bin_matmul_1024 | 129.154 ± 0.477 | **48.557 ± 0.382** | 60.946 ± 0.333 | 64.258 ± 4.401 | 76.353 ± 5.592 |
| bin_matmul_256 | 1.125 ± 0.024 | **0.760 ± 0.010** | 1.359 ± 0.039 | 1.152 ± 0.027 | 1.283 ± 0.057 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.010 ± 0.000 | **0.006 ± 0.000** | 0.035 ± 0.005 | 0.201 ± 0.020 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.034 ± 0.001 | **0.033 ± 0.001** | 0.152 ± 0.018 | 0.288 ± 0.041 | 0.052 ± 0.003 |
| bin_omeinsum_matmul_10x10 | 0.017 ± 0.001 | **0.009 ± 0.000** | 0.034 ± 0.004 | 0.180 ± 0.030 | 0.025 ± 0.002 |
| bin_outer_product_4096 | 67.350 ± 0.472 | 66.314 ± 0.891 | 57.776 ± 2.020 | **56.997 ± 3.998** | 313.055 ± 40.151 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 51.561 ± 0.598 | **50.568 ± 0.467** | 62.228 ± 0.424 | 66.092 ± 2.121 | 67.480 ± 9.967 |
| gm_queen5_5_3.wcsp | **5477.378 ± 52.005** | 5677.330 ± 48.219 | 6413.588 ± 30.819 | 8271.269 ± 94.478 | 9318.608 ± 189.638 |
| lm_batch_likelihood_brackets_4_4d | 24.254 ± 0.239 | 31.504 ± 1.002 | 36.508 ± 1.828 | **15.967 ± 1.291** | 37.027 ± 24.456 |
| lm_batch_likelihood_sentence_3_12d | **54.615 ± 0.848** | 94.929 ± 1.919 | 67.826 ± 0.907 | 99.440 ± 2.214 | 78.363 ± 11.788 |
| lm_batch_likelihood_sentence_4_4d | 25.227 ± 0.314 | 32.500 ± 0.643 | 37.822 ± 0.549 | **17.699 ± 0.561** | 39.556 ± 4.960 |
| nary_matmul_chain_64 | 0.042 ± 0.001 | **0.034 ± 0.001** | 0.080 ± 0.015 | 0.197 ± 0.012 | 0.075 ± 0.010 |
| str_matrix_chain_multiplication_100 | **12.426 ± 0.132** | 13.446 ± 0.431 | 15.915 ± 0.394 | 14.974 ± 0.552 | 16.793 ± 4.349 |
| str_mps_varying_inner_product_200 | 15.375 ± 0.188 | **15.154 ± 0.201** | 26.428 ± 0.182 | 24.066 ± 0.710 | 25.642 ± 1.540 |
| str_nw_mera_closed_120 | **1351.404 ± 12.574** | 1426.955 ± 8.084 | 2029.699 ± 22.222 | 2512.561 ± 14.909 | 1852.196 ± 41.533 |
| str_nw_mera_open_26 | **1106.746 ± 5.805** | 1475.177 ± 9.898 | 1250.144 ± 4.902 | 1968.834 ± 21.941 | 1684.999 ± 15.306 |
| tensornetwork_permutation_focus_step409_316 | **298.111 ± 4.379** | 410.632 ± 2.691 | 884.453 ± 3.018 | 1507.543 ± 258.556 | 555.604 ± 29.921 |
| tensornetwork_permutation_light_415 | **295.184 ± 4.446** | 341.341 ± 3.002 | 847.175 ± 2.581 | 1186.080 ± 282.338 | 552.091 ± 6.256 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.144 ± 0.087 | **2.918 ± 0.008** | 4.518 ± 0.133 | 4.386 ± 0.189 | 9.968 ± 2.194 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.561 ± 0.017 | **0.410 ± 0.020** | 0.654 ± 0.013 | 0.838 ± 0.013 | 1.550 ± 1.153 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.750 ± 0.002 | 0.743 ± 0.008 | **0.209 ± 0.025** | 0.581 ± 0.023 | 3.905 ± 2.542 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.189 ± 0.015 | 0.759 ± 0.009 | **0.206 ± 0.009** | 0.566 ± 0.022 | 5.024 ± 5.771 |
| bin_elementwise_mul_2048x2048 | 2.179 ± 0.064 | **1.846 ± 0.154** | 17.545 ± 0.341 | 19.197 ± 0.360 | 18.941 ± 4.718 |
| bin_elementwise_mul_512x512 | 0.150 ± 0.011 | **0.137 ± 0.029** | 0.160 ± 0.025 | 0.359 ± 0.035 | 0.179 ± 0.748 |
| bin_matmul_1024 | 129.154 ± 0.477 | **48.557 ± 0.382** | 60.946 ± 0.333 | 64.258 ± 4.401 | 76.353 ± 5.592 |
| bin_matmul_256 | 1.125 ± 0.024 | **0.760 ± 0.010** | 1.359 ± 0.039 | 1.152 ± 0.027 | 1.283 ± 0.057 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.010 ± 0.000 | **0.006 ± 0.000** | 0.035 ± 0.005 | 0.201 ± 0.020 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.034 ± 0.001 | **0.033 ± 0.001** | 0.152 ± 0.018 | 0.288 ± 0.041 | 0.052 ± 0.003 |
| bin_omeinsum_matmul_10x10 | 0.017 ± 0.001 | **0.009 ± 0.000** | 0.034 ± 0.004 | 0.180 ± 0.030 | 0.025 ± 0.002 |
| bin_outer_product_4096 | 67.350 ± 0.472 | 66.314 ± 0.891 | 57.776 ± 2.020 | **56.997 ± 3.998** | 313.055 ± 40.151 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 51.561 ± 0.598 | **50.568 ± 0.467** | 62.228 ± 0.424 | 66.092 ± 2.121 | 67.480 ± 9.967 |
| gm_queen5_5_3.wcsp | **1875.126 ± 32.956** | 2208.444 ± 24.950 | 2605.890 ± 81.559 | 2844.679 ± 24.500 | 3363.375 ± 101.580 |
| lm_batch_likelihood_brackets_4_4d | 24.381 ± 0.691 | 28.832 ± 0.239 | 37.643 ± 0.566 | **16.251 ± 0.418** | 36.103 ± 44.327 |
| lm_batch_likelihood_sentence_3_12d | **57.492 ± 0.805** | 115.327 ± 1.624 | 72.167 ± 0.751 | 101.838 ± 3.340 | 117.221 ± 46.720 |
| lm_batch_likelihood_sentence_4_4d | 23.965 ± 0.895 | 32.788 ± 0.707 | 38.894 ± 0.380 | **16.860 ± 0.382** | 38.769 ± 35.607 |
| nary_matmul_chain_64 | 0.042 ± 0.001 | **0.034 ± 0.001** | 0.080 ± 0.015 | 0.197 ± 0.012 | 0.075 ± 0.010 |
| str_matrix_chain_multiplication_100 | **13.616 ± 0.299** | 14.163 ± 0.178 | 15.900 ± 0.239 | 15.134 ± 0.512 | 17.012 ± 1.055 |
| str_mps_varying_inner_product_200 | 16.893 ± 0.222 | **15.421 ± 0.322** | 26.082 ± 0.325 | 24.106 ± 0.490 | 25.520 ± 1.417 |
| str_nw_mera_closed_120 | 1326.123 ± 11.483 | **1282.160 ± 18.916** | 1596.305 ± 10.867 | 2073.611 ± 21.522 | 1534.656 ± 13.667 |
| str_nw_mera_open_26 | **1084.069 ± 8.728** | 1491.396 ± 10.406 | 1293.837 ± 10.550 | 2051.176 ± 26.123 | 1685.317 ± 17.003 |
| tensornetwork_permutation_focus_step409_316 | **298.111 ± 4.379** | 410.632 ± 2.691 | 884.453 ± 3.018 | 1507.543 ± 258.556 | 555.604 ± 29.921 |
| tensornetwork_permutation_light_415 | **295.184 ± 4.446** | 341.341 ± 3.002 | 847.175 ± 2.581 | 1186.080 ± 282.338 | 552.091 ± 6.256 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261125_183303/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261125_183303/run.yaml`
- Timestamp: `2026-11-25T18:33:03.871566+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261125_183303`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
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

- tenferro-rs features: `native`

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

- `data/results/amd-cpu/cpu/einsum/20261125_183303/jax_cpu_t4_20261125_183303.log`
- `data/results/amd-cpu/cpu/einsum/20261125_183303/julia_omeinsum_t4_20261125_183303.log`
- `data/results/amd-cpu/cpu/einsum/20261125_183303/pytorch_cpu_t4_20261125_183303.log`
- `data/results/amd-cpu/cpu/einsum/20261125_183303/tenferro_eager_t4_20261125_183303.log`
- `data/results/amd-cpu/cpu/einsum/20261125_183303/tenferro_trace_t4_20261125_183303.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.439 ± 0.028 | **1.738 ± 0.014** | 2.432 ± 2.641 | 3.472 ± 0.624 | 11.490 ± 4.885 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.597 ± 0.019 | 0.467 ± 0.016 | **0.385 ± 0.033** | 0.794 ± 0.007 | 0.919 ± 0.335 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.426 ± 0.014 | 0.425 ± 0.013 | **0.214 ± 0.027** | 0.453 ± 0.041 | 1.990 ± 3.523 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.306 ± 0.014 | 0.425 ± 0.015 | **0.192 ± 0.031** | 0.472 ± 0.037 | 5.458 ± 5.298 |
| bin_elementwise_mul_2048x2048 | **0.774 ± 0.121** | 0.927 ± 0.023 | 5.880 ± 0.236 | 9.657 ± 0.329 | 19.063 ± 2.472 |
| bin_elementwise_mul_512x512 | **0.132 ± 0.033** | 0.144 ± 0.019 | 0.196 ± 0.031 | 0.409 ± 0.033 | 0.156 ± 0.054 |
| bin_matmul_1024 | 51.411 ± 0.818 | **22.881 ± 0.164** | 36.172 ± 0.319 | 36.013 ± 0.532 | 56.351 ± 5.789 |
| bin_matmul_256 | 0.657 ± 0.016 | **0.449 ± 0.015** | 0.863 ± 0.014 | 0.856 ± 0.070 | 0.925 ± 0.034 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.000 | **0.009 ± 0.000** | 0.051 ± 0.010 | 0.199 ± 0.019 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.050 ± 0.003 | **0.047 ± 0.004** | 0.231 ± 0.042 | 0.309 ± 0.030 | 0.051 ± 0.004 |
| bin_omeinsum_matmul_10x10 | 0.027 ± 0.001 | **0.014 ± 0.001** | 0.059 ± 0.031 | 0.191 ± 0.027 | 0.024 ± 0.001 |
| bin_outer_product_4096 | 22.660 ± 0.347 | 22.366 ± 0.502 | **18.988 ± 0.297** | 35.972 ± 1.332 | 310.762 ± 34.939 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 24.586 ± 0.163 | **24.233 ± 0.211** | 31.024 ± 0.511 | 43.965 ± 5.932 | 48.102 ± 9.796 |
| gm_queen5_5_3.wcsp | 3352.288 ± 78.589 | 3989.514 ± 60.799 | **2680.536 ± 35.155** | 5433.909 ± 65.175 | 8723.727 ± 129.902 |
| lm_batch_likelihood_brackets_4_4d | 18.243 ± 0.354 | 22.807 ± 0.500 | **17.313 ± 0.445** | 23.635 ± 0.744 | 46.741 ± 20.564 |
| lm_batch_likelihood_sentence_3_12d | **30.662 ± 0.428** | 54.292 ± 0.966 | 35.683 ± 0.773 | 51.051 ± 4.810 | 70.616 ± 15.155 |
| lm_batch_likelihood_sentence_4_4d | 18.958 ± 0.560 | 22.869 ± 0.281 | **18.153 ± 0.213** | 24.061 ± 1.185 | 43.836 ± 5.682 |
| nary_matmul_chain_64 | 0.067 ± 0.001 | **0.063 ± 0.001** | 0.094 ± 0.023 | 0.274 ± 0.025 | 0.150 ± 0.047 |
| str_matrix_chain_multiplication_100 | **7.895 ± 0.215** | 9.770 ± 0.181 | 10.901 ± 0.113 | 15.542 ± 0.520 | 14.649 ± 1.205 |
| str_mps_varying_inner_product_200 | 27.513 ± 0.618 | 29.729 ± 0.549 | **26.303 ± 0.309** | 31.038 ± 1.118 | 29.777 ± 1.912 |
| str_nw_mera_closed_120 | **697.943 ± 13.960** | 754.516 ± 16.095 | 949.194 ± 7.545 | 1186.853 ± 35.037 | 1239.292 ± 109.163 |
| str_nw_mera_open_26 | 631.863 ± 13.833 | 748.970 ± 10.259 | **592.778 ± 12.383** | 943.723 ± 16.553 | 1225.811 ± 28.736 |
| tensornetwork_permutation_focus_step409_316 | **187.614 ± 1.573** | 200.106 ± 4.132 | 352.223 ± 47.081 | 459.980 ± 117.440 | 527.338 ± 71.663 |
| tensornetwork_permutation_light_415 | 186.663 ± 1.556 | **160.943 ± 2.866** | 345.328 ± 4.234 | 436.258 ± 46.935 | 537.561 ± 55.905 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.439 ± 0.028 | **1.738 ± 0.014** | 2.432 ± 2.641 | 3.472 ± 0.624 | 11.490 ± 4.885 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.597 ± 0.019 | 0.467 ± 0.016 | **0.385 ± 0.033** | 0.794 ± 0.007 | 0.919 ± 0.335 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.426 ± 0.014 | 0.425 ± 0.013 | **0.214 ± 0.027** | 0.453 ± 0.041 | 1.990 ± 3.523 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.306 ± 0.014 | 0.425 ± 0.015 | **0.192 ± 0.031** | 0.472 ± 0.037 | 5.458 ± 5.298 |
| bin_elementwise_mul_2048x2048 | **0.774 ± 0.121** | 0.927 ± 0.023 | 5.880 ± 0.236 | 9.657 ± 0.329 | 19.063 ± 2.472 |
| bin_elementwise_mul_512x512 | **0.132 ± 0.033** | 0.144 ± 0.019 | 0.196 ± 0.031 | 0.409 ± 0.033 | 0.156 ± 0.054 |
| bin_matmul_1024 | 51.411 ± 0.818 | **22.881 ± 0.164** | 36.172 ± 0.319 | 36.013 ± 0.532 | 56.351 ± 5.789 |
| bin_matmul_256 | 0.657 ± 0.016 | **0.449 ± 0.015** | 0.863 ± 0.014 | 0.856 ± 0.070 | 0.925 ± 0.034 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.000 | **0.009 ± 0.000** | 0.051 ± 0.010 | 0.199 ± 0.019 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.050 ± 0.003 | **0.047 ± 0.004** | 0.231 ± 0.042 | 0.309 ± 0.030 | 0.051 ± 0.004 |
| bin_omeinsum_matmul_10x10 | 0.027 ± 0.001 | **0.014 ± 0.001** | 0.059 ± 0.031 | 0.191 ± 0.027 | 0.024 ± 0.001 |
| bin_outer_product_4096 | 22.660 ± 0.347 | 22.366 ± 0.502 | **18.988 ± 0.297** | 35.972 ± 1.332 | 310.762 ± 34.939 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 24.586 ± 0.163 | **24.233 ± 0.211** | 31.024 ± 0.511 | 43.965 ± 5.932 | 48.102 ± 9.796 |
| gm_queen5_5_3.wcsp | **919.229 ± 16.705** | 1108.136 ± 7.034 | 1117.614 ± 16.957 | 1896.427 ± 27.345 | 3111.359 ± 93.574 |
| lm_batch_likelihood_brackets_4_4d | **17.616 ± 0.414** | 21.602 ± 0.195 | 17.876 ± 0.296 | 24.040 ± 1.230 | 45.942 ± 3.561 |
| lm_batch_likelihood_sentence_3_12d | **30.733 ± 0.324** | 65.681 ± 0.630 | 32.969 ± 0.192 | 60.310 ± 7.526 | 89.076 ± 50.023 |
| lm_batch_likelihood_sentence_4_4d | **16.571 ± 0.281** | 22.205 ± 0.455 | 18.785 ± 0.231 | 29.505 ± 2.949 | 46.959 ± 13.856 |
| nary_matmul_chain_64 | 0.067 ± 0.001 | **0.063 ± 0.001** | 0.094 ± 0.023 | 0.274 ± 0.025 | 0.150 ± 0.047 |
| str_matrix_chain_multiplication_100 | **8.420 ± 0.137** | 9.653 ± 0.118 | 10.978 ± 0.345 | 20.525 ± 1.591 | 14.264 ± 0.692 |
| str_mps_varying_inner_product_200 | 31.523 ± 0.461 | 31.220 ± 0.377 | **28.996 ± 1.130** | 31.930 ± 1.452 | 29.429 ± 2.957 |
| str_nw_mera_closed_120 | 725.762 ± 7.776 | **680.082 ± 11.858** | 766.943 ± 15.117 | 994.433 ± 22.958 | 927.735 ± 58.775 |
| str_nw_mera_open_26 | 642.011 ± 15.562 | 751.750 ± 13.194 | **596.263 ± 12.951** | 969.417 ± 28.873 | 1236.214 ± 50.082 |
| tensornetwork_permutation_focus_step409_316 | **187.614 ± 1.573** | 200.106 ± 4.132 | 352.223 ± 47.081 | 459.980 ± 117.440 | 527.338 ± 71.663 |
| tensornetwork_permutation_light_415 | 186.663 ± 1.556 | **160.943 ± 2.866** | 345.328 ± 4.234 | 436.258 ± 46.935 | 537.561 ± 55.905 |

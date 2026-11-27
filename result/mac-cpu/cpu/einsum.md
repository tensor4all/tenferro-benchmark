# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `7da876a214acb65eb9ea099adef67b41ef8efc98`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261127_172420/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261127_172420/run.yaml`
- Timestamp: `2026-11-27T17:24:20.827597+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261127_172420`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

### CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

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

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.4`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.0.3.30.dylib)`, probe threads `1`

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

- `data/results/mac-cpu/cpu/einsum/20261127_172420/jax_cpu_t1_20261127_172420.log`
- `data/results/mac-cpu/cpu/einsum/20261127_172420/julia_omeinsum_t1_20261127_172420.log`
- `data/results/mac-cpu/cpu/einsum/20261127_172420/pytorch_cpu_t1_20261127_172420.log`
- `data/results/mac-cpu/cpu/einsum/20261127_172420/tenferro_eager_t1_20261127_172420.log`
- `data/results/mac-cpu/cpu/einsum/20261127_172420/tenferro_trace_t1_20261127_172420.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.272 ± 4.022 | 3.576 ± 0.233 | **0.736 ± 0.081** | 3.882 ± 0.174 | 7.420 ± 8.338 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.330 ± 0.996 | 0.544 ± 0.238 | **0.132 ± 0.018** | 0.638 ± 0.047 | 1.038 ± 0.990 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.523 ± 0.035 | 0.244 ± 0.037 | **0.212 ± 0.012** | 0.368 ± 0.077 | 0.625 ± 1.159 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.364 ± 0.384 | 0.235 ± 0.016 | **0.221 ± 0.030** | 0.436 ± 0.044 | 1.687 ± 2.675 |
| bin_elementwise_mul_2048x2048 | 4.289 ± 2.756 | **2.099 ± 0.267** | 2.443 ± 0.282 | 3.158 ± 0.095 | 3.663 ± 3.514 |
| bin_elementwise_mul_512x512 | 0.283 ± 0.185 | 0.081 ± 0.010 | **0.078 ± 0.001** | 0.255 ± 0.047 | 0.082 ± 0.182 |
| bin_matmul_1024 | 83.032 ± 22.713 | 52.721 ± 7.437 | **12.060 ± 7.288** | 52.923 ± 1.769 | 62.538 ± 7.976 |
| bin_matmul_256 | 2.342 ± 1.458 | 0.940 ± 0.162 | **0.132 ± 0.018** | 0.957 ± 0.115 | 1.189 ± 0.510 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.004 | **0.005 ± 0.001** | 0.018 ± 0.001 | 0.088 ± 0.005 | 0.012 ± 0.009 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.072 ± 0.071 | **0.025 ± 0.005** | 0.080 ± 0.012 | 0.157 ± 0.038 | 0.086 ± 0.019 |
| bin_omeinsum_matmul_10x10 | 0.041 ± 0.011 | **0.008 ± 0.002** | 0.017 ± 0.001 | 0.065 ± 0.004 | 0.036 ± 0.007 |
| bin_outer_product_4096 | 10.435 ± 3.214 | 5.679 ± 1.154 | 3.472 ± 0.239 | **2.532 ± 0.191** | 271.460 ± 93.822 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 150.306 ± 26.869 | 99.947 ± 20.723 | **13.207 ± 4.686** | 56.629 ± 1.875 | 62.084 ± 9.255 |
| gm_queen5_5_3.wcsp | 7331.006 ± 3605.188 | 5455.438 ± 2574.336 | **4063.241 ± 230.355** | 5690.170 ± 211.362 | 8489.708 ± 1747.887 |
| lm_batch_likelihood_brackets_4_4d | 35.033 ± 18.139 | 40.273 ± 4.141 | 14.877 ± 2.074 | **13.644 ± 0.597** | 20.552 ± 10.023 |
| lm_batch_likelihood_sentence_3_12d | 105.428 ± 36.029 | 127.158 ± 24.738 | **26.701 ± 3.683** | 60.002 ± 3.047 | 64.021 ± 7.838 |
| lm_batch_likelihood_sentence_4_4d | 39.032 ± 9.180 | 50.820 ± 29.904 | 14.846 ± 2.042 | **14.347 ± 0.548** | 22.555 ± 3.870 |
| nary_matmul_chain_64 | 0.038 ± 0.006 | 0.065 ± 0.068 | **0.022 ± 0.001** | 0.097 ± 0.008 | 0.104 ± 0.022 |
| str_matrix_chain_multiplication_100 | 15.649 ± 4.546 | 21.267 ± 12.056 | **5.027 ± 0.334** | 14.743 ± 0.613 | 14.820 ± 3.520 |
| str_mps_varying_inner_product_200 | 22.846 ± 6.510 | 31.355 ± 9.115 | **6.825 ± 0.554** | 18.631 ± 0.912 | 17.854 ± 2.231 |
| str_nw_mera_closed_120 | 2659.108 ± 319.697 | 2286.359 ± 221.838 | **341.935 ± 6.051** | 1710.462 ± 104.870 | 1523.186 ± 229.686 |
| str_nw_mera_open_26 | 2265.357 ± 396.002 | 1647.950 ± 164.163 | **229.224 ± 24.837** | 1160.177 ± 243.802 | 1042.561 ± 319.012 |
| tensornetwork_permutation_focus_step409_316 | 346.322 ± 73.894 | 328.170 ± 9.929 | 238.705 ± 24.995 | 379.464 ± 37.773 | **219.716 ± 29.907** |
| tensornetwork_permutation_light_415 | 288.323 ± 29.225 | 297.626 ± 40.090 | 225.960 ± 27.619 | 386.179 ± 50.446 | **202.977 ± 7.484** |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.272 ± 4.022 | 3.576 ± 0.233 | **0.736 ± 0.081** | 3.882 ± 0.174 | 7.420 ± 8.338 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.330 ± 0.996 | 0.544 ± 0.238 | **0.132 ± 0.018** | 0.638 ± 0.047 | 1.038 ± 0.990 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.523 ± 0.035 | 0.244 ± 0.037 | **0.212 ± 0.012** | 0.368 ± 0.077 | 0.625 ± 1.159 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.364 ± 0.384 | 0.235 ± 0.016 | **0.221 ± 0.030** | 0.436 ± 0.044 | 1.687 ± 2.675 |
| bin_elementwise_mul_2048x2048 | 4.289 ± 2.756 | **2.099 ± 0.267** | 2.443 ± 0.282 | 3.158 ± 0.095 | 3.663 ± 3.514 |
| bin_elementwise_mul_512x512 | 0.283 ± 0.185 | 0.081 ± 0.010 | **0.078 ± 0.001** | 0.255 ± 0.047 | 0.082 ± 0.182 |
| bin_matmul_1024 | 83.032 ± 22.713 | 52.721 ± 7.437 | **12.060 ± 7.288** | 52.923 ± 1.769 | 62.538 ± 7.976 |
| bin_matmul_256 | 2.342 ± 1.458 | 0.940 ± 0.162 | **0.132 ± 0.018** | 0.957 ± 0.115 | 1.189 ± 0.510 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.004 | **0.005 ± 0.001** | 0.018 ± 0.001 | 0.088 ± 0.005 | 0.012 ± 0.009 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.072 ± 0.071 | **0.025 ± 0.005** | 0.080 ± 0.012 | 0.157 ± 0.038 | 0.086 ± 0.019 |
| bin_omeinsum_matmul_10x10 | 0.041 ± 0.011 | **0.008 ± 0.002** | 0.017 ± 0.001 | 0.065 ± 0.004 | 0.036 ± 0.007 |
| bin_outer_product_4096 | 10.435 ± 3.214 | 5.679 ± 1.154 | 3.472 ± 0.239 | **2.532 ± 0.191** | 271.460 ± 93.822 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 150.306 ± 26.869 | 99.947 ± 20.723 | **13.207 ± 4.686** | 56.629 ± 1.875 | 62.084 ± 9.255 |
| gm_queen5_5_3.wcsp | 2438.771 ± 334.286 | 1780.207 ± 349.636 | **817.599 ± 83.684** | 1296.633 ± 178.918 | 1982.396 ± 241.872 |
| lm_batch_likelihood_brackets_4_4d | 26.249 ± 5.273 | 37.459 ± 11.804 | 14.650 ± 0.410 | **12.540 ± 0.501** | 22.083 ± 3.908 |
| lm_batch_likelihood_sentence_3_12d | 108.036 ± 12.064 | 149.925 ± 28.089 | **25.393 ± 0.923** | 57.316 ± 5.507 | 106.844 ± 46.493 |
| lm_batch_likelihood_sentence_4_4d | 30.861 ± 4.028 | 36.515 ± 9.167 | 14.638 ± 0.261 | **14.314 ± 2.287** | 30.327 ± 43.505 |
| nary_matmul_chain_64 | 0.038 ± 0.006 | 0.065 ± 0.068 | **0.022 ± 0.001** | 0.097 ± 0.008 | 0.104 ± 0.022 |
| str_matrix_chain_multiplication_100 | 18.272 ± 3.137 | 24.834 ± 8.453 | **5.062 ± 0.194** | 15.023 ± 1.417 | 17.802 ± 4.082 |
| str_mps_varying_inner_product_200 | 31.994 ± 3.953 | 34.133 ± 8.238 | **8.239 ± 0.196** | 22.456 ± 0.623 | 18.249 ± 3.954 |
| str_nw_mera_closed_120 | 2257.226 ± 497.906 | 2336.393 ± 186.729 | **271.034 ± 3.085** | 1512.459 ± 102.739 | 1473.749 ± 244.493 |
| str_nw_mera_open_26 | 1546.413 ± 214.982 | 1530.782 ± 214.476 | **237.893 ± 16.719** | 1470.447 ± 402.498 | 1855.239 ± 415.708 |
| tensornetwork_permutation_focus_step409_316 | 346.322 ± 73.894 | 328.170 ± 9.929 | 238.705 ± 24.995 | 379.464 ± 37.773 | **219.716 ± 29.907** |
| tensornetwork_permutation_light_415 | 288.323 ± 29.225 | 297.626 ± 40.090 | 225.960 ± 27.619 | 386.179 ± 50.446 | **202.977 ± 7.484** |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261127_180034/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261127_180034/run.yaml`
- Timestamp: `2026-11-27T18:00:34.399034+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261127_180034`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

### CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

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

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.4`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.0.3.30.dylib)`, probe threads `4`

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

- `data/results/mac-cpu/cpu/einsum/20261127_180034/jax_cpu_t4_20261127_180034.log`
- `data/results/mac-cpu/cpu/einsum/20261127_180034/julia_omeinsum_t4_20261127_180034.log`
- `data/results/mac-cpu/cpu/einsum/20261127_180034/pytorch_cpu_t4_20261127_180034.log`
- `data/results/mac-cpu/cpu/einsum/20261127_180034/tenferro_eager_t4_20261127_180034.log`
- `data/results/mac-cpu/cpu/einsum/20261127_180034/tenferro_trace_t4_20261127_180034.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.012 ± 0.177 | 3.155 ± 1.061 | **0.835 ± 0.227** | 3.383 ± 2.015 | 93.111 ± 17.720 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.469 ± 0.043 | 0.463 ± 0.036 | **0.134 ± 0.004** | 0.680 ± 0.100 | 1.714 ± 1.694 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.164 ± 0.025** | 0.186 ± 0.025 | 0.412 ± 0.394 | 0.807 ± 0.496 | 0.871 ± 0.412 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.214 ± 0.032 | **0.186 ± 0.027** | 0.320 ± 0.196 | 0.362 ± 0.060 | 1.457 ± 0.380 |
| bin_elementwise_mul_2048x2048 | 2.779 ± 0.675 | **2.180 ± 0.183** | 2.216 ± 0.682 | 3.474 ± 0.147 | 2.784 ± 0.155 |
| bin_elementwise_mul_512x512 | 0.138 ± 0.162 | **0.056 ± 0.006** | 0.210 ± 0.182 | 0.753 ± 0.833 | 0.076 ± 0.026 |
| bin_matmul_1024 | 21.854 ± 2.936 | 19.511 ± 1.211 | **9.997 ± 1.898** | 29.177 ± 15.594 | 57.381 ± 38.305 |
| bin_matmul_256 | 0.510 ± 0.014 | 0.452 ± 0.030 | **0.119 ± 0.016** | 0.760 ± 0.238 | 2.907 ± 2.642 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.004 ± 0.000** | 0.018 ± 0.002 | 0.087 ± 0.007 | 0.011 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.022 ± 0.000 | **0.021 ± 0.001** | 0.073 ± 0.009 | 0.161 ± 0.060 | 0.117 ± 0.009 |
| bin_omeinsum_matmul_10x10 | 0.013 ± 0.000 | **0.007 ± 0.000** | 0.015 ± 0.001 | 0.174 ± 0.134 | 0.030 ± 0.007 |
| bin_outer_product_4096 | 2.399 ± 0.317 | **2.370 ± 0.219** | 2.793 ± 0.664 | 2.485 ± 0.158 | 230.456 ± 26.225 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 41.991 ± 3.712 | 43.345 ± 5.062 | **10.254 ± 1.150** | 20.995 ± 0.887 | 52.427 ± 25.443 |
| gm_queen5_5_3.wcsp | **3079.649 ± 1099.441** | 4588.611 ± 517.770 | 3715.036 ± 342.089 | 4826.574 ± 546.829 | 9681.470 ± 681.065 |
| lm_batch_likelihood_brackets_4_4d | 26.332 ± 8.401 | 18.647 ± 8.083 | **12.257 ± 0.670** | 17.233 ± 5.402 | 122.194 ± 106.558 |
| lm_batch_likelihood_sentence_3_12d | 59.784 ± 6.667 | 63.816 ± 14.801 | **26.484 ± 2.860** | 35.708 ± 4.857 | 105.460 ± 17.021 |
| lm_batch_likelihood_sentence_4_4d | 28.001 ± 5.409 | 25.063 ± 15.574 | **13.148 ± 0.971** | 17.550 ± 0.603 | 94.962 ± 117.963 |
| nary_matmul_chain_64 | 0.100 ± 0.015 | 0.104 ± 0.032 | **0.027 ± 0.016** | 0.137 ± 0.022 | 0.093 ± 0.041 |
| str_matrix_chain_multiplication_100 | 19.857 ± 2.689 | 10.034 ± 1.039 | **4.907 ± 0.396** | 13.219 ± 6.040 | 284.250 ± 57.990 |
| str_mps_varying_inner_product_200 | 42.439 ± 3.238 | 37.865 ± 20.908 | **7.591 ± 0.442** | 21.330 ± 3.741 | 22.968 ± 7.816 |
| str_nw_mera_closed_120 | 1261.789 ± 195.711 | 1120.701 ± 186.437 | **328.382 ± 58.455** | 561.324 ± 153.573 | 1000.521 ± 213.187 |
| str_nw_mera_open_26 | 675.885 ± 76.959 | 765.413 ± 73.277 | **258.083 ± 53.508** | 495.556 ± 75.010 | 878.798 ± 163.075 |
| tensornetwork_permutation_focus_step409_316 | 169.029 ± 50.782 | **164.993 ± 14.681** | 188.082 ± 24.047 | 253.876 ± 27.248 | 419.613 ± 49.387 |
| tensornetwork_permutation_light_415 | **124.048 ± 13.316** | 125.761 ± 11.863 | 219.451 ± 34.988 | 308.297 ± 94.408 | 440.544 ± 88.665 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.012 ± 0.177 | 3.155 ± 1.061 | **0.835 ± 0.227** | 3.383 ± 2.015 | 93.111 ± 17.720 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.469 ± 0.043 | 0.463 ± 0.036 | **0.134 ± 0.004** | 0.680 ± 0.100 | 1.714 ± 1.694 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.164 ± 0.025** | 0.186 ± 0.025 | 0.412 ± 0.394 | 0.807 ± 0.496 | 0.871 ± 0.412 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.214 ± 0.032 | **0.186 ± 0.027** | 0.320 ± 0.196 | 0.362 ± 0.060 | 1.457 ± 0.380 |
| bin_elementwise_mul_2048x2048 | 2.779 ± 0.675 | **2.180 ± 0.183** | 2.216 ± 0.682 | 3.474 ± 0.147 | 2.784 ± 0.155 |
| bin_elementwise_mul_512x512 | 0.138 ± 0.162 | **0.056 ± 0.006** | 0.210 ± 0.182 | 0.753 ± 0.833 | 0.076 ± 0.026 |
| bin_matmul_1024 | 21.854 ± 2.936 | 19.511 ± 1.211 | **9.997 ± 1.898** | 29.177 ± 15.594 | 57.381 ± 38.305 |
| bin_matmul_256 | 0.510 ± 0.014 | 0.452 ± 0.030 | **0.119 ± 0.016** | 0.760 ± 0.238 | 2.907 ± 2.642 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.004 ± 0.000** | 0.018 ± 0.002 | 0.087 ± 0.007 | 0.011 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.022 ± 0.000 | **0.021 ± 0.001** | 0.073 ± 0.009 | 0.161 ± 0.060 | 0.117 ± 0.009 |
| bin_omeinsum_matmul_10x10 | 0.013 ± 0.000 | **0.007 ± 0.000** | 0.015 ± 0.001 | 0.174 ± 0.134 | 0.030 ± 0.007 |
| bin_outer_product_4096 | 2.399 ± 0.317 | **2.370 ± 0.219** | 2.793 ± 0.664 | 2.485 ± 0.158 | 230.456 ± 26.225 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 41.991 ± 3.712 | 43.345 ± 5.062 | **10.254 ± 1.150** | 20.995 ± 0.887 | 52.427 ± 25.443 |
| gm_queen5_5_3.wcsp | 932.267 ± 146.671 | 915.356 ± 154.813 | **683.145 ± 103.691** | 847.951 ± 57.823 | 1955.203 ± 343.697 |
| lm_batch_likelihood_brackets_4_4d | 21.839 ± 7.944 | 18.235 ± 4.180 | 19.646 ± 7.618 | **17.335 ± 3.608** | 166.544 ± 75.157 |
| lm_batch_likelihood_sentence_3_12d | 51.236 ± 3.705 | 48.707 ± 28.508 | 43.226 ± 13.571 | **40.495 ± 2.055** | 127.356 ± 48.087 |
| lm_batch_likelihood_sentence_4_4d | 22.184 ± 5.704 | **17.491 ± 1.476** | 22.730 ± 7.177 | 22.685 ± 2.636 | 102.175 ± 42.687 |
| nary_matmul_chain_64 | 0.100 ± 0.015 | 0.104 ± 0.032 | **0.027 ± 0.016** | 0.137 ± 0.022 | 0.093 ± 0.041 |
| str_matrix_chain_multiplication_100 | 16.367 ± 8.299 | 10.227 ± 2.997 | **5.585 ± 0.493** | 14.773 ± 4.314 | 282.840 ± 49.559 |
| str_mps_varying_inner_product_200 | 44.846 ± 5.685 | 22.796 ± 12.052 | **16.684 ± 4.276** | 35.876 ± 6.759 | 54.739 ± 30.695 |
| str_nw_mera_closed_120 | 1384.041 ± 228.972 | 1225.914 ± 266.089 | **357.924 ± 50.582** | 582.309 ± 84.076 | 841.136 ± 161.178 |
| str_nw_mera_open_26 | 751.425 ± 114.949 | 1020.504 ± 119.093 | **247.758 ± 25.614** | 490.673 ± 67.710 | 881.069 ± 143.411 |
| tensornetwork_permutation_focus_step409_316 | 169.029 ± 50.782 | **164.993 ± 14.681** | 188.082 ± 24.047 | 253.876 ± 27.248 | 419.613 ± 49.387 |
| tensornetwork_permutation_light_415 | **124.048 ± 13.316** | 125.761 ± 11.863 | 219.451 ± 34.988 | 308.297 ± 94.408 | 440.544 ± 88.665 |

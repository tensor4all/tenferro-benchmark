# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `ec30d812749ef2bd6dc72245a56150aa77cb2f9c`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261108_172311/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261108_172311/run.yaml`
- Timestamp: `2026-11-08T17:23:11.781207+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261108_172311`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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

- tenferro-rs features: `blas-accelerate`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.3`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.0.3.30.dylib)`, probe threads `1`

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

- `data/results/mac-cpu/cpu/einsum/20261108_172311/jax_cpu_t1_20261108_172311.log`
- `data/results/mac-cpu/cpu/einsum/20261108_172311/julia_omeinsum_t1_20261108_172311.log`
- `data/results/mac-cpu/cpu/einsum/20261108_172311/pytorch_cpu_t1_20261108_172311.log`
- `data/results/mac-cpu/cpu/einsum/20261108_172311/tenferro_eager_t1_20261108_172311.log`
- `data/results/mac-cpu/cpu/einsum/20261108_172311/tenferro_trace_t1_20261108_172311.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.958 ± 2.194 | **0.943 ± 0.464** | 1.005 ± 0.218 | 5.116 ± 0.626 | 6.411 ± 2.962 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.172 ± 0.019 | **0.088 ± 0.009** | 0.151 ± 0.050 | 0.933 ± 0.344 | 2.291 ± 2.186 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.238 ± 0.025** | 0.381 ± 0.198 | 0.248 ± 0.120 | 0.565 ± 0.068 | 0.824 ± 0.386 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.936 ± 0.075 | 0.284 ± 0.158 | **0.225 ± 0.067** | 0.694 ± 0.171 | 1.474 ± 6.219 |
| bin_elementwise_mul_2048x2048 | 3.534 ± 1.256 | 2.820 ± 0.629 | 3.259 ± 0.718 | 4.078 ± 0.401 | **2.729 ± 0.706** |
| bin_elementwise_mul_512x512 | 0.110 ± 0.016 | **0.072 ± 0.014** | 0.094 ± 0.025 | 0.433 ± 0.120 | 0.082 ± 0.011 |
| bin_matmul_1024 | 28.695 ± 10.100 | **14.428 ± 5.212** | 15.424 ± 7.698 | 56.288 ± 3.558 | 62.157 ± 23.175 |
| bin_matmul_256 | 0.170 ± 0.034 | **0.133 ± 0.044** | 0.153 ± 0.025 | 0.914 ± 0.087 | 1.025 ± 0.212 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.005 ± 0.001** | 0.021 ± 0.005 | 0.093 ± 0.011 | 0.010 ± 0.004 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.010 | **0.028 ± 0.017** | 0.080 ± 0.002 | 0.156 ± 0.022 | 0.090 ± 0.013 |
| bin_omeinsum_matmul_10x10 | 0.014 ± 0.003 | **0.007 ± 0.000** | 0.017 ± 0.003 | 0.084 ± 0.027 | 0.038 ± 0.010 |
| bin_outer_product_4096 | 9.693 ± 1.957 | 6.633 ± 1.313 | 3.984 ± 1.966 | **3.014 ± 0.983** | 248.098 ± 11.185 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 29.085 ± 10.280 | 14.743 ± 14.055 | **13.639 ± 4.575** | 63.016 ± 6.403 | 61.914 ± 10.450 |
| gm_queen5_5_3.wcsp | 10353.604 ± 813.189 | 8160.190 ± 1804.224 | **4695.573 ± 621.295** | 7209.794 ± 268.491 | 10239.479 ± 1870.764 |
| lm_batch_likelihood_brackets_4_4d | 25.116 ± 6.484 | 21.936 ± 10.394 | **17.146 ± 3.612** | 18.107 ± 1.186 | 24.922 ± 17.350 |
| lm_batch_likelihood_sentence_3_12d | 38.373 ± 7.404 | 42.838 ± 5.299 | **34.539 ± 11.795** | 69.050 ± 4.101 | 92.297 ± 54.987 |
| lm_batch_likelihood_sentence_4_4d | 25.596 ± 5.719 | 26.377 ± 14.388 | **19.858 ± 3.495** | 25.113 ± 7.409 | 53.848 ± 26.299 |
| nary_matmul_chain_64 | 0.017 ± 0.001 | **0.012 ± 0.003** | 0.025 ± 0.001 | 0.446 ± 0.372 | 0.224 ± 0.103 |
| str_matrix_chain_multiplication_100 | 7.186 ± 2.711 | 8.697 ± 2.434 | **6.648 ± 0.519** | 21.230 ± 3.586 | 19.784 ± 6.963 |
| str_mps_varying_inner_product_200 | 14.155 ± 2.895 | 14.655 ± 5.039 | **8.790 ± 1.671** | 21.488 ± 4.207 | 19.045 ± 8.301 |
| str_nw_mera_closed_120 | 638.386 ± 165.350 | 486.041 ± 158.104 | **443.769 ± 213.479** | 2094.724 ± 587.256 | 2046.065 ± 538.096 |
| str_nw_mera_open_26 | 708.579 ± 102.812 | 631.417 ± 290.980 | **257.453 ± 35.884** | 1325.749 ± 193.207 | 1338.623 ± 219.002 |
| tensornetwork_permutation_focus_step409_316 | 269.537 ± 88.085 | 351.009 ± 33.167 | 325.163 ± 43.135 | 713.212 ± 142.207 | **251.098 ± 58.243** |
| tensornetwork_permutation_light_415 | 245.751 ± 79.634 | **149.939 ± 6.303** | 282.951 ± 29.853 | 668.455 ± 146.596 | 279.574 ± 100.179 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.958 ± 2.194 | **0.943 ± 0.464** | 1.005 ± 0.218 | 5.116 ± 0.626 | 6.411 ± 2.962 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.172 ± 0.019 | **0.088 ± 0.009** | 0.151 ± 0.050 | 0.933 ± 0.344 | 2.291 ± 2.186 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.238 ± 0.025** | 0.381 ± 0.198 | 0.248 ± 0.120 | 0.565 ± 0.068 | 0.824 ± 0.386 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.936 ± 0.075 | 0.284 ± 0.158 | **0.225 ± 0.067** | 0.694 ± 0.171 | 1.474 ± 6.219 |
| bin_elementwise_mul_2048x2048 | 3.534 ± 1.256 | 2.820 ± 0.629 | 3.259 ± 0.718 | 4.078 ± 0.401 | **2.729 ± 0.706** |
| bin_elementwise_mul_512x512 | 0.110 ± 0.016 | **0.072 ± 0.014** | 0.094 ± 0.025 | 0.433 ± 0.120 | 0.082 ± 0.011 |
| bin_matmul_1024 | 28.695 ± 10.100 | **14.428 ± 5.212** | 15.424 ± 7.698 | 56.288 ± 3.558 | 62.157 ± 23.175 |
| bin_matmul_256 | 0.170 ± 0.034 | **0.133 ± 0.044** | 0.153 ± 0.025 | 0.914 ± 0.087 | 1.025 ± 0.212 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.005 ± 0.001** | 0.021 ± 0.005 | 0.093 ± 0.011 | 0.010 ± 0.004 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.010 | **0.028 ± 0.017** | 0.080 ± 0.002 | 0.156 ± 0.022 | 0.090 ± 0.013 |
| bin_omeinsum_matmul_10x10 | 0.014 ± 0.003 | **0.007 ± 0.000** | 0.017 ± 0.003 | 0.084 ± 0.027 | 0.038 ± 0.010 |
| bin_outer_product_4096 | 9.693 ± 1.957 | 6.633 ± 1.313 | 3.984 ± 1.966 | **3.014 ± 0.983** | 248.098 ± 11.185 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 29.085 ± 10.280 | 14.743 ± 14.055 | **13.639 ± 4.575** | 63.016 ± 6.403 | 61.914 ± 10.450 |
| gm_queen5_5_3.wcsp | 1633.595 ± 312.166 | 1572.402 ± 595.240 | **964.068 ± 164.943** | 1614.489 ± 470.906 | 2467.349 ± 245.460 |
| lm_batch_likelihood_brackets_4_4d | 25.654 ± 10.286 | 36.434 ± 20.656 | 15.450 ± 1.894 | **14.432 ± 3.741** | 31.134 ± 30.422 |
| lm_batch_likelihood_sentence_3_12d | 46.756 ± 13.974 | 57.816 ± 11.008 | **28.985 ± 5.868** | 84.590 ± 34.625 | 119.068 ± 40.908 |
| lm_batch_likelihood_sentence_4_4d | 23.540 ± 6.666 | 30.279 ± 12.627 | **16.466 ± 1.023** | 19.868 ± 5.625 | 39.230 ± 26.502 |
| nary_matmul_chain_64 | 0.017 ± 0.001 | **0.012 ± 0.003** | 0.025 ± 0.001 | 0.446 ± 0.372 | 0.224 ± 0.103 |
| str_matrix_chain_multiplication_100 | 6.333 ± 2.793 | 8.305 ± 2.283 | **5.409 ± 1.234** | 18.384 ± 6.007 | 18.076 ± 2.259 |
| str_mps_varying_inner_product_200 | 16.141 ± 3.819 | 15.423 ± 5.272 | **8.937 ± 0.459** | 30.215 ± 7.779 | 19.027 ± 10.019 |
| str_nw_mera_closed_120 | 437.232 ± 74.630 | 569.535 ± 155.166 | **347.543 ± 58.048** | 1697.088 ± 291.962 | 1484.637 ± 92.742 |
| str_nw_mera_open_26 | 464.326 ± 100.258 | 548.100 ± 111.074 | **250.507 ± 21.350** | 1435.174 ± 209.780 | 1412.988 ± 139.492 |
| tensornetwork_permutation_focus_step409_316 | 269.537 ± 88.085 | 351.009 ± 33.167 | 325.163 ± 43.135 | 713.212 ± 142.207 | **251.098 ± 58.243** |
| tensornetwork_permutation_light_415 | 245.751 ± 79.634 | **149.939 ± 6.303** | 282.951 ± 29.853 | 668.455 ± 146.596 | 279.574 ± 100.179 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261108_175931/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261108_175931/run.yaml`
- Timestamp: `2026-11-08T17:59:31.505952+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261108_175931`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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

- tenferro-rs features: `blas-accelerate`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.3`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.0.3.30.dylib)`, probe threads `4`

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

- `data/results/mac-cpu/cpu/einsum/20261108_175931/jax_cpu_t4_20261108_175931.log`
- `data/results/mac-cpu/cpu/einsum/20261108_175931/julia_omeinsum_t4_20261108_175931.log`
- `data/results/mac-cpu/cpu/einsum/20261108_175931/pytorch_cpu_t4_20261108_175931.log`
- `data/results/mac-cpu/cpu/einsum/20261108_175931/tenferro_eager_t4_20261108_175931.log`
- `data/results/mac-cpu/cpu/einsum/20261108_175931/tenferro_trace_t4_20261108_175931.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.639 ± 0.240 | 0.735 ± 0.143 | **0.704 ± 0.037** | 2.709 ± 0.595 | 92.208 ± 16.854 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.301 ± 0.095 | **0.088 ± 0.008** | 0.130 ± 0.010 | 0.828 ± 0.159 | 0.806 ± 0.478 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.310 ± 0.085 | **0.174 ± 0.047** | 0.243 ± 0.079 | 0.358 ± 0.112 | 0.632 ± 0.480 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.351 ± 0.096 | **0.157 ± 0.016** | 0.261 ± 0.025 | 0.380 ± 0.051 | 1.341 ± 6.199 |
| bin_elementwise_mul_2048x2048 | 3.897 ± 5.879 | 2.643 ± 0.303 | 2.842 ± 0.293 | 3.750 ± 0.414 | **2.123 ± 0.097** |
| bin_elementwise_mul_512x512 | 0.147 ± 0.419 | 0.131 ± 0.037 | 0.157 ± 0.025 | 0.573 ± 0.159 | **0.076 ± 0.009** |
| bin_matmul_1024 | 17.898 ± 2.887 | 14.834 ± 2.985 | **14.654 ± 1.920** | 21.670 ± 2.312 | 48.884 ± 22.083 |
| bin_matmul_256 | 0.304 ± 0.363 | **0.134 ± 0.097** | 0.162 ± 0.023 | 0.736 ± 0.080 | 2.773 ± 0.740 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.025 ± 0.006 | **0.004 ± 0.000** | 0.020 ± 0.004 | 0.093 ± 0.021 | 0.032 ± 0.006 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.031 ± 0.004 | **0.028 ± 0.005** | 0.197 ± 0.003 | 0.198 ± 0.038 | 0.090 ± 0.009 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.002 | **0.008 ± 0.000** | 0.021 ± 0.042 | 0.165 ± 0.158 | 0.035 ± 0.003 |
| bin_outer_product_4096 | 3.611 ± 0.812 | 3.510 ± 1.043 | 3.634 ± 1.440 | **3.078 ± 0.326** | 233.974 ± 15.033 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 18.000 ± 0.953 | 14.871 ± 1.563 | **11.838 ± 2.012** | 23.270 ± 2.591 | 32.255 ± 12.765 |
| gm_queen5_5_3.wcsp | 6567.292 ± 616.167 | 7738.908 ± 628.710 | **3942.441 ± 388.337** | 5608.836 ± 2842.790 | 10846.730 ± 1929.821 |
| lm_batch_likelihood_brackets_4_4d | 18.812 ± 1.350 | 26.219 ± 8.478 | 22.981 ± 8.649 | **16.966 ± 1.942** | 162.000 ± 79.944 |
| lm_batch_likelihood_sentence_3_12d | **24.949 ± 3.479** | 62.322 ± 28.511 | 41.357 ± 9.060 | 38.913 ± 12.463 | 158.511 ± 28.433 |
| lm_batch_likelihood_sentence_4_4d | 22.631 ± 5.201 | 25.076 ± 3.481 | 30.196 ± 8.524 | **17.389 ± 5.859** | 165.712 ± 95.537 |
| nary_matmul_chain_64 | 0.020 ± 0.035 | **0.012 ± 0.000** | 0.092 ± 0.017 | 0.128 ± 0.021 | 0.115 ± 0.171 |
| str_matrix_chain_multiplication_100 | 6.801 ± 1.481 | **6.163 ± 0.332** | 9.560 ± 2.517 | 14.479 ± 6.614 | 285.216 ± 60.285 |
| str_mps_varying_inner_product_200 | 10.912 ± 0.607 | **10.278 ± 1.603** | 13.011 ± 3.384 | 29.894 ± 15.552 | 31.736 ± 25.845 |
| str_nw_mera_closed_120 | 435.441 ± 46.729 | 428.473 ± 35.950 | **340.314 ± 21.740** | 662.418 ± 230.409 | 904.033 ± 350.382 |
| str_nw_mera_open_26 | 347.412 ± 45.870 | 461.500 ± 48.573 | **246.121 ± 22.678** | 596.168 ± 194.412 | 1121.390 ± 272.022 |
| tensornetwork_permutation_focus_step409_316 | **162.456 ± 19.903** | 312.085 ± 35.125 | 173.571 ± 26.170 | 296.938 ± 53.428 | 526.855 ± 107.244 |
| tensornetwork_permutation_light_415 | 150.344 ± 3.947 | **132.268 ± 20.234** | 132.605 ± 18.130 | 280.116 ± 66.219 | 486.436 ± 89.719 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.639 ± 0.240 | 0.735 ± 0.143 | **0.704 ± 0.037** | 2.709 ± 0.595 | 92.208 ± 16.854 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.301 ± 0.095 | **0.088 ± 0.008** | 0.130 ± 0.010 | 0.828 ± 0.159 | 0.806 ± 0.478 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.310 ± 0.085 | **0.174 ± 0.047** | 0.243 ± 0.079 | 0.358 ± 0.112 | 0.632 ± 0.480 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.351 ± 0.096 | **0.157 ± 0.016** | 0.261 ± 0.025 | 0.380 ± 0.051 | 1.341 ± 6.199 |
| bin_elementwise_mul_2048x2048 | 3.897 ± 5.879 | 2.643 ± 0.303 | 2.842 ± 0.293 | 3.750 ± 0.414 | **2.123 ± 0.097** |
| bin_elementwise_mul_512x512 | 0.147 ± 0.419 | 0.131 ± 0.037 | 0.157 ± 0.025 | 0.573 ± 0.159 | **0.076 ± 0.009** |
| bin_matmul_1024 | 17.898 ± 2.887 | 14.834 ± 2.985 | **14.654 ± 1.920** | 21.670 ± 2.312 | 48.884 ± 22.083 |
| bin_matmul_256 | 0.304 ± 0.363 | **0.134 ± 0.097** | 0.162 ± 0.023 | 0.736 ± 0.080 | 2.773 ± 0.740 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.025 ± 0.006 | **0.004 ± 0.000** | 0.020 ± 0.004 | 0.093 ± 0.021 | 0.032 ± 0.006 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.031 ± 0.004 | **0.028 ± 0.005** | 0.197 ± 0.003 | 0.198 ± 0.038 | 0.090 ± 0.009 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.002 | **0.008 ± 0.000** | 0.021 ± 0.042 | 0.165 ± 0.158 | 0.035 ± 0.003 |
| bin_outer_product_4096 | 3.611 ± 0.812 | 3.510 ± 1.043 | 3.634 ± 1.440 | **3.078 ± 0.326** | 233.974 ± 15.033 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 18.000 ± 0.953 | 14.871 ± 1.563 | **11.838 ± 2.012** | 23.270 ± 2.591 | 32.255 ± 12.765 |
| gm_queen5_5_3.wcsp | 857.086 ± 93.061 | 792.981 ± 135.349 | **559.654 ± 43.250** | 930.455 ± 401.650 | 2243.608 ± 596.578 |
| lm_batch_likelihood_brackets_4_4d | 17.734 ± 2.498 | 25.302 ± 5.838 | **14.976 ± 0.962** | 26.956 ± 12.471 | 120.592 ± 52.161 |
| lm_batch_likelihood_sentence_3_12d | **24.400 ± 0.913** | 44.149 ± 4.798 | 28.262 ± 1.055 | 53.645 ± 7.629 | 128.912 ± 31.728 |
| lm_batch_likelihood_sentence_4_4d | 18.661 ± 5.357 | 26.633 ± 5.160 | **15.450 ± 1.005** | 22.494 ± 4.022 | 111.735 ± 44.741 |
| nary_matmul_chain_64 | 0.020 ± 0.035 | **0.012 ± 0.000** | 0.092 ± 0.017 | 0.128 ± 0.021 | 0.115 ± 0.171 |
| str_matrix_chain_multiplication_100 | **5.525 ± 0.630** | 5.873 ± 0.878 | 5.987 ± 0.579 | 17.859 ± 2.428 | 308.290 ± 226.035 |
| str_mps_varying_inner_product_200 | 12.863 ± 0.877 | **12.024 ± 1.326** | 14.229 ± 3.947 | 23.249 ± 7.784 | 24.317 ± 18.878 |
| str_nw_mera_closed_120 | 358.479 ± 32.616 | 419.974 ± 33.738 | **280.683 ± 7.289** | 586.724 ± 127.839 | 928.544 ± 169.451 |
| str_nw_mera_open_26 | 359.317 ± 30.978 | 634.752 ± 155.816 | **233.583 ± 20.378** | 532.285 ± 179.223 | 997.782 ± 276.052 |
| tensornetwork_permutation_focus_step409_316 | **162.456 ± 19.903** | 312.085 ± 35.125 | 173.571 ± 26.170 | 296.938 ± 53.428 | 526.855 ± 107.244 |
| tensornetwork_permutation_light_415 | 150.344 ± 3.947 | **132.268 ± 20.234** | 132.605 ± 18.130 | 280.116 ± 66.219 | 486.436 ± 89.719 |

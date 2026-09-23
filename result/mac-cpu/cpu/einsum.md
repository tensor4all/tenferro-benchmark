# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `f6279624e964c253ad21dec6521d82df6b00f7db`
- Benchmark commit: `8595ce266fa34a87659e72c9b14195383343df95`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260923_103427/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260923_103427/run.yaml`
- Timestamp: `20260923_103427`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260923_103427`.

- tenferro-rs commit: `f6279624e964c253ad21dec6521d82df6b00f7db`

### CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`

### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-accelerate`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

### Julia / OMEinsum.jl Backend

- Julia: version `1.12.5`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.dylib)`, probe threads `1`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20260923_103427/einsum_table_t1_20260923_103427.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260923_103427/tenferro_trace_t1_20260923_103427.log`
- `data/results/mac-cpu/cpu/einsum/20260923_103427/tenferro_eager_t1_20260923_103427.log`
- `data/results/mac-cpu/cpu/einsum/20260923_103427/pytorch_cpu_t1_20260923_103427.log`
- `data/results/mac-cpu/cpu/einsum/20260923_103427/jax_cpu_t1_20260923_103427.log`
- `data/results/mac-cpu/cpu/einsum/20260923_103427/julia_omeinsum_t1_20260923_103427.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.988 ± 0.194 | 0.496 ± 0.157 | **0.387 ± 0.004** | 2.223 ± 0.042 | 3.803 ± 0.717 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.200 ± 0.011 | **0.082 ± 0.001** | 0.089 ± 0.005 | 0.336 ± 0.011 | 0.517 ± 0.086 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.103 ± 0.004** | 0.130 ± 0.004 | 0.108 ± 0.003 | 0.177 ± 0.007 | 0.299 ± 0.019 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.438 ± 0.036 | **0.110 ± 0.001** | 0.113 ± 0.005 | 0.204 ± 0.008 | 0.625 ± 0.914 |
| bin_elementwise_mul_2048x2048 | 0.953 ± 0.050 | 1.038 ± 0.036 | **0.803 ± 0.039** | 1.439 ± 0.024 | 0.913 ± 0.429 |
| bin_matmul_1024 | 5.807 ± 0.157 | 11.527 ± 0.691 | **4.554 ± 0.096** | 34.783 ± 0.333 | 39.977 ± 2.420 |
| bin_matmul_256 | 0.129 ± 0.001 | 0.093 ± 0.009 | **0.089 ± 0.007** | 0.604 ± 0.018 | 0.715 ± 0.016 |
| bin_outer_product_4096 | **1.020 ± 0.044** | **1.020 ± 0.032** | 1.499 ± 0.067 | 1.051 ± 0.033 | 63.589 ± 39.539 |
| gm_queen5_5_3.wcsp | **650.197 ± 9.391** | 1097.394 ± 49.171 | 901.669 ± 5.377 | 1262.042 ± 10.558 | 1377.760 ± 92.222 |
| lm_batch_likelihood_brackets_4_4d | 11.014 ± 0.148 | 10.607 ± 0.336 | 8.535 ± 0.119 | **7.256 ± 0.116** | 20.875 ± 46.555 |
| lm_batch_likelihood_sentence_3_12d | 17.322 ± 0.239 | 20.902 ± 0.486 | **16.864 ± 0.567** | 28.848 ± 0.138 | 82.349 ± 45.433 |
| lm_batch_likelihood_sentence_4_4d | 11.600 ± 0.116 | 12.011 ± 0.819 | 8.909 ± 0.146 | **8.309 ± 0.075** | 17.982 ± 45.803 |
| nary_matmul_chain_64 | 0.011 ± 0.001 | **0.009 ± 0.000** | 0.013 ± 0.000 | 0.058 ± 0.002 | 0.040 ± 0.004 |
| str_matrix_chain_multiplication_100 | 1.913 ± 0.025 | **1.908 ± 0.037** | 2.142 ± 0.077 | 8.065 ± 0.402 | 9.032 ± 1.311 |
| str_mps_varying_inner_product_200 | 4.395 ± 0.043 | 4.236 ± 0.031 | **3.571 ± 0.083** | 12.087 ± 0.121 | 10.573 ± 5.640 |
| str_nw_mera_closed_120 | 313.091 ± 1.001 | 315.350 ± 1.082 | **157.815 ± 0.639** | 837.056 ± 0.954 | 962.856 ± 103.137 |
| str_nw_mera_open_26 | 238.444 ± 1.306 | 172.814 ± 1.094 | **128.402 ± 1.359** | 623.466 ± 2.293 | 710.839 ± 51.590 |
| tensornetwork_permutation_focus_step409_316 | **79.346 ± 3.147** | 184.479 ± 4.286 | 153.886 ± 0.718 | 201.480 ± 2.531 | 178.439 ± 50.007 |
| tensornetwork_permutation_light_415 | **79.490 ± 2.848** | 96.824 ± 1.215 | 133.281 ± 0.363 | 204.041 ± 2.082 | 175.878 ± 0.563 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.988 ± 0.194 | 0.496 ± 0.157 | **0.387 ± 0.004** | 2.223 ± 0.042 | 3.803 ± 0.717 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.200 ± 0.011 | **0.082 ± 0.001** | 0.089 ± 0.005 | 0.336 ± 0.011 | 0.517 ± 0.086 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.103 ± 0.004** | 0.130 ± 0.004 | 0.108 ± 0.003 | 0.177 ± 0.007 | 0.299 ± 0.019 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.438 ± 0.036 | **0.110 ± 0.001** | 0.113 ± 0.005 | 0.204 ± 0.008 | 0.625 ± 0.914 |
| bin_elementwise_mul_2048x2048 | 0.953 ± 0.050 | 1.038 ± 0.036 | **0.803 ± 0.039** | 1.439 ± 0.024 | 0.913 ± 0.429 |
| bin_matmul_1024 | 5.807 ± 0.157 | 11.527 ± 0.691 | **4.554 ± 0.096** | 34.783 ± 0.333 | 39.977 ± 2.420 |
| bin_matmul_256 | 0.129 ± 0.001 | 0.093 ± 0.009 | **0.089 ± 0.007** | 0.604 ± 0.018 | 0.715 ± 0.016 |
| bin_outer_product_4096 | **1.020 ± 0.044** | **1.020 ± 0.032** | 1.499 ± 0.067 | 1.051 ± 0.033 | 63.589 ± 39.539 |
| gm_queen5_5_3.wcsp | **145.531 ± 15.395** | 298.345 ± 1.664 | 316.351 ± 2.686 | 636.262 ± 3.523 | 764.442 ± 5.000 |
| lm_batch_likelihood_brackets_4_4d | 9.950 ± 0.112 | 9.828 ± 0.323 | 8.758 ± 0.192 | **6.981 ± 0.060** | 19.291 ± 46.318 |
| lm_batch_likelihood_sentence_3_12d | 18.366 ± 0.125 | 17.822 ± 0.450 | **17.240 ± 0.186** | 29.945 ± 0.225 | 83.168 ± 42.957 |
| lm_batch_likelihood_sentence_4_4d | 11.368 ± 0.073 | 10.870 ± 0.370 | 8.513 ± 0.127 | **7.765 ± 0.036** | 19.396 ± 46.058 |
| nary_matmul_chain_64 | 0.011 ± 0.001 | **0.009 ± 0.000** | 0.013 ± 0.000 | 0.058 ± 0.002 | 0.040 ± 0.004 |
| str_matrix_chain_multiplication_100 | **1.955 ± 0.027** | 2.021 ± 0.014 | 2.091 ± 0.015 | 7.818 ± 0.078 | 8.970 ± 0.706 |
| str_mps_varying_inner_product_200 | 5.496 ± 0.087 | 5.574 ± 0.084 | **4.333 ± 0.051** | 13.037 ± 0.204 | 9.585 ± 0.574 |
| str_nw_mera_closed_120 | 182.255 ± 0.595 | 181.632 ± 0.450 | **130.912 ± 0.389** | 816.709 ± 0.875 | 943.252 ± 54.750 |
| str_nw_mera_open_26 | 239.089 ± 2.620 | 174.194 ± 0.960 | **133.578 ± 0.367** | 633.235 ± 1.579 | 712.976 ± 54.684 |
| tensornetwork_permutation_focus_step409_316 | **79.346 ± 3.147** | 184.479 ± 4.286 | 153.886 ± 0.718 | 201.480 ± 2.531 | 178.439 ± 50.007 |
| tensornetwork_permutation_light_415 | **79.490 ± 2.848** | 96.824 ± 1.215 | 133.281 ± 0.363 | 204.041 ± 2.082 | 175.878 ± 0.563 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260923_105726/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260923_105726/run.yaml`
- Timestamp: `20260923_105726`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260923_105726`.

- tenferro-rs commit: `f6279624e964c253ad21dec6521d82df6b00f7db`

### CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4`

### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-accelerate`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

### Julia / OMEinsum.jl Backend

- Julia: version `1.12.5`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.dylib)`, probe threads `4`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20260923_105726/einsum_table_t4_20260923_105726.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260923_105726/tenferro_trace_t4_20260923_105726.log`
- `data/results/mac-cpu/cpu/einsum/20260923_105726/tenferro_eager_t4_20260923_105726.log`
- `data/results/mac-cpu/cpu/einsum/20260923_105726/pytorch_cpu_t4_20260923_105726.log`
- `data/results/mac-cpu/cpu/einsum/20260923_105726/jax_cpu_t4_20260923_105726.log`
- `data/results/mac-cpu/cpu/einsum/20260923_105726/julia_omeinsum_t4_20260923_105726.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.796 ± 0.175 | 0.530 ± 0.133 | **0.388 ± 0.009** | 0.643 ± 0.021 | 2.032 ± 3.363 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.204 ± 0.042 | **0.066 ± 0.019** | 0.086 ± 0.001 | 0.150 ± 0.007 | 0.521 ± 0.204 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.077 ± 0.016 | **0.050 ± 0.017** | 0.056 ± 0.006 | 0.114 ± 0.015 | 0.320 ± 0.022 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.180 ± 0.027 | 0.059 ± 0.013 | **0.050 ± 0.003** | 0.120 ± 0.005 | 0.718 ± 1.160 |
| bin_elementwise_mul_2048x2048 | 0.621 ± 0.029 | **0.590 ± 0.029** | 0.601 ± 0.056 | 0.819 ± 0.050 | 0.946 ± 0.569 |
| bin_matmul_1024 | 5.090 ± 0.229 | 9.311 ± 0.496 | **4.524 ± 0.031** | 9.097 ± 0.098 | 14.937 ± 4.816 |
| bin_matmul_256 | 0.159 ± 0.009 | **0.086 ± 0.000** | 0.087 ± 0.002 | 0.233 ± 0.006 | 0.305 ± 0.006 |
| bin_outer_product_4096 | 0.575 ± 0.010 | **0.564 ± 0.023** | 0.566 ± 0.025 | 0.631 ± 0.051 | 69.354 ± 13.078 |
| gm_queen5_5_3.wcsp | 672.106 ± 36.242 | 569.723 ± 33.158 | **448.550 ± 6.613** | 625.615 ± 43.584 | 1046.612 ± 14.268 |
| lm_batch_likelihood_brackets_4_4d | 6.460 ± 0.060 | 8.545 ± 0.229 | 9.218 ± 0.864 | **6.181 ± 0.318** | 9.410 ± 1.334 |
| lm_batch_likelihood_sentence_3_12d | **8.449 ± 0.172** | 13.413 ± 0.259 | 21.331 ± 1.494 | 11.286 ± 0.159 | 24.030 ± 16.921 |
| lm_batch_likelihood_sentence_4_4d | 6.855 ± 0.201 | 9.578 ± 0.120 | 9.943 ± 0.798 | **6.356 ± 0.135** | 13.604 ± 15.805 |
| nary_matmul_chain_64 | 0.037 ± 0.016 | **0.009 ± 0.000** | 0.013 ± 0.001 | 0.058 ± 0.009 | 0.044 ± 0.005 |
| str_matrix_chain_multiplication_100 | **1.911 ± 0.051** | 1.958 ± 0.037 | 2.148 ± 0.061 | 6.622 ± 0.424 | 5.578 ± 0.447 |
| str_mps_varying_inner_product_200 | 5.013 ± 0.201 | 4.950 ± 0.170 | **3.956 ± 0.211** | 14.373 ± 0.519 | 9.905 ± 4.259 |
| str_nw_mera_closed_120 | 145.440 ± 0.949 | **143.515 ± 1.646** | 144.331 ± 3.947 | 251.285 ± 1.349 | 313.077 ± 21.397 |
| str_nw_mera_open_26 | 151.504 ± 3.517 | 147.376 ± 2.216 | **119.968 ± 1.438** | 186.442 ± 1.371 | 263.674 ± 20.370 |
| tensornetwork_permutation_focus_step409_316 | 92.203 ± 0.668 | 117.674 ± 2.384 | **70.211 ± 7.037** | 90.727 ± 0.795 | 89.651 ± 16.674 |
| tensornetwork_permutation_light_415 | 95.998 ± 3.137 | 95.802 ± 20.074 | **71.260 ± 10.089** | 94.295 ± 0.780 | 90.570 ± 17.189 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.796 ± 0.175 | 0.530 ± 0.133 | **0.388 ± 0.009** | 0.643 ± 0.021 | 2.032 ± 3.363 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.204 ± 0.042 | **0.066 ± 0.019** | 0.086 ± 0.001 | 0.150 ± 0.007 | 0.521 ± 0.204 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.077 ± 0.016 | **0.050 ± 0.017** | 0.056 ± 0.006 | 0.114 ± 0.015 | 0.320 ± 0.022 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.180 ± 0.027 | 0.059 ± 0.013 | **0.050 ± 0.003** | 0.120 ± 0.005 | 0.718 ± 1.160 |
| bin_elementwise_mul_2048x2048 | 0.621 ± 0.029 | **0.590 ± 0.029** | 0.601 ± 0.056 | 0.819 ± 0.050 | 0.946 ± 0.569 |
| bin_matmul_1024 | 5.090 ± 0.229 | 9.311 ± 0.496 | **4.524 ± 0.031** | 9.097 ± 0.098 | 14.937 ± 4.816 |
| bin_matmul_256 | 0.159 ± 0.009 | **0.086 ± 0.000** | 0.087 ± 0.002 | 0.233 ± 0.006 | 0.305 ± 0.006 |
| bin_outer_product_4096 | 0.575 ± 0.010 | **0.564 ± 0.023** | 0.566 ± 0.025 | 0.631 ± 0.051 | 69.354 ± 13.078 |
| gm_queen5_5_3.wcsp | 188.875 ± 14.874 | **180.474 ± 17.377** | 188.902 ± 4.878 | 263.296 ± 3.565 | 398.309 ± 8.534 |
| lm_batch_likelihood_brackets_4_4d | 7.094 ± 0.061 | 9.352 ± 0.097 | 6.817 ± 0.276 | **6.755 ± 0.212** | 14.302 ± 15.890 |
| lm_batch_likelihood_sentence_3_12d | **9.368 ± 0.126** | 13.909 ± 0.584 | 16.288 ± 0.189 | 12.306 ± 0.235 | 34.343 ± 16.424 |
| lm_batch_likelihood_sentence_4_4d | 7.173 ± 0.059 | 9.502 ± 0.140 | 6.706 ± 0.059 | **6.493 ± 0.197** | 13.409 ± 15.422 |
| nary_matmul_chain_64 | 0.037 ± 0.016 | **0.009 ± 0.000** | 0.013 ± 0.001 | 0.058 ± 0.009 | 0.044 ± 0.005 |
| str_matrix_chain_multiplication_100 | **1.926 ± 0.025** | 2.003 ± 0.043 | 2.143 ± 0.038 | 8.133 ± 0.167 | 5.402 ± 0.887 |
| str_mps_varying_inner_product_200 | 5.997 ± 0.103 | 6.242 ± 0.097 | **4.771 ± 0.087** | 17.044 ± 0.401 | 8.808 ± 5.002 |
| str_nw_mera_closed_120 | 131.107 ± 0.524 | 127.272 ± 1.983 | **122.658 ± 0.491** | 250.967 ± 4.847 | 294.140 ± 21.251 |
| str_nw_mera_open_26 | 154.592 ± 5.141 | 146.705 ± 1.431 | **123.670 ± 1.469** | 193.527 ± 1.746 | 267.808 ± 6.956 |
| tensornetwork_permutation_focus_step409_316 | 92.203 ± 0.668 | 117.674 ± 2.384 | **70.211 ± 7.037** | 90.727 ± 0.795 | 89.651 ± 16.674 |
| tensornetwork_permutation_light_415 | 95.998 ± 3.137 | 95.802 ± 20.074 | **71.260 ± 10.089** | 94.295 ± 0.780 | 90.570 ± 17.189 |

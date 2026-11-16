# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `5c1d5a382dd88d20901598318730d850d3568ac8`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261116_172433/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261116_172433/run.yaml`
- Timestamp: `2026-11-16T17:24:33.485160+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261116_172433`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Apple M2 Pro (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `5`
- Physical CPUs: `5`
- Sockets: `1`
- Cores per socket: `5`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 5 physical / 5 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 5 CPUs/L2)`
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

- `data/results/mac-cpu/cpu/einsum/20261116_172433/jax_cpu_t1_20261116_172433.log`
- `data/results/mac-cpu/cpu/einsum/20261116_172433/julia_omeinsum_t1_20261116_172433.log`
- `data/results/mac-cpu/cpu/einsum/20261116_172433/pytorch_cpu_t1_20261116_172433.log`
- `data/results/mac-cpu/cpu/einsum/20261116_172433/tenferro_eager_t1_20261116_172433.log`
- `data/results/mac-cpu/cpu/einsum/20261116_172433/tenferro_trace_t1_20261116_172433.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.894 ± 0.149 | **0.436 ± 0.021** | 0.597 ± 0.040 | 3.971 ± 0.235 | 4.591 ± 0.956 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.122 ± 0.005 | **0.078 ± 0.004** | 0.130 ± 0.006 | 0.738 ± 0.203 | 0.827 ± 0.364 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.242 ± 0.016 | 0.207 ± 0.020 | **0.171 ± 0.017** | 0.419 ± 0.156 | 0.569 ± 0.170 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.933 ± 0.093 | 0.199 ± 0.017 | **0.168 ± 0.010** | 0.521 ± 0.039 | 0.946 ± 1.738 |
| bin_elementwise_mul_2048x2048 | **1.322 ± 0.254** | 1.377 ± 0.162 | 1.351 ± 0.144 | 2.842 ± 0.715 | 1.626 ± 0.451 |
| bin_elementwise_mul_512x512 | **0.064 ± 0.001** | 0.066 ± 0.000 | 0.074 ± 0.003 | 0.363 ± 0.177 | 0.096 ± 0.028 |
| bin_matmul_1024 | 9.215 ± 0.715 | **7.704 ± 0.762** | 8.114 ± 0.759 | 56.781 ± 5.021 | 54.320 ± 7.732 |
| bin_matmul_256 | 0.202 ± 0.070 | **0.114 ± 0.005** | 0.116 ± 0.005 | 1.010 ± 0.115 | 0.981 ± 0.079 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.006 ± 0.002 | **0.003 ± 0.000** | 0.021 ± 0.002 | 0.096 ± 0.019 | 0.011 ± 0.005 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.031 ± 0.004 | **0.026 ± 0.004** | 0.078 ± 0.006 | 0.156 ± 0.065 | 0.079 ± 0.179 |
| bin_omeinsum_matmul_10x10 | 0.014 ± 0.001 | **0.007 ± 0.001** | 0.015 ± 0.001 | 0.063 ± 0.003 | 0.032 ± 0.019 |
| bin_outer_product_4096 | 3.784 ± 0.464 | 3.408 ± 0.127 | **2.719 ± 0.117** | 2.834 ± 0.312 | 106.610 ± 22.888 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 14.066 ± 1.335 | **8.677 ± 1.629** | 10.733 ± 1.654 | 60.527 ± 3.768 | 47.454 ± 4.612 |
| gm_queen5_5_3.wcsp | 2979.717 ± 437.475 | 2659.811 ± 441.409 | **1570.438 ± 85.650** | 2683.794 ± 397.949 | 2488.304 ± 305.396 |
| lm_batch_likelihood_brackets_4_4d | 15.538 ± 1.906 | 17.490 ± 4.180 | **11.257 ± 0.143** | 14.460 ± 1.803 | 21.004 ± 2.187 |
| lm_batch_likelihood_sentence_3_12d | 21.452 ± 1.851 | 37.033 ± 2.894 | **20.276 ± 0.799** | 52.693 ± 5.374 | 59.477 ± 4.708 |
| lm_batch_likelihood_sentence_4_4d | 14.725 ± 0.742 | 16.381 ± 0.559 | **11.728 ± 0.496** | 12.716 ± 1.854 | 20.212 ± 2.610 |
| nary_matmul_chain_64 | 0.017 ± 0.001 | **0.010 ± 0.000** | 0.019 ± 0.002 | 0.159 ± 0.056 | 0.101 ± 0.018 |
| str_matrix_chain_multiplication_100 | 4.933 ± 0.292 | 4.576 ± 0.262 | **3.862 ± 0.023** | 13.041 ± 0.874 | 14.823 ± 5.366 |
| str_mps_varying_inner_product_200 | 8.859 ± 1.056 | 8.549 ± 0.267 | **5.405 ± 0.111** | 18.550 ± 0.967 | 18.409 ± 3.081 |
| str_nw_mera_closed_120 | 373.530 ± 54.672 | 340.785 ± 18.926 | **283.518 ± 25.045** | 1514.715 ± 88.554 | 1204.732 ± 137.217 |
| str_nw_mera_open_26 | 309.437 ± 63.624 | 342.488 ± 24.417 | **199.658 ± 14.204** | 1092.260 ± 59.485 | 983.247 ± 42.991 |
| tensornetwork_permutation_focus_step409_316 | **151.453 ± 33.700** | 285.278 ± 15.932 | 230.843 ± 9.327 | 337.217 ± 16.037 | 200.198 ± 20.500 |
| tensornetwork_permutation_light_415 | 142.144 ± 17.697 | **124.880 ± 10.013** | 238.356 ± 9.559 | 394.634 ± 57.568 | 203.306 ± 17.837 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.894 ± 0.149 | **0.436 ± 0.021** | 0.597 ± 0.040 | 3.971 ± 0.235 | 4.591 ± 0.956 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.122 ± 0.005 | **0.078 ± 0.004** | 0.130 ± 0.006 | 0.738 ± 0.203 | 0.827 ± 0.364 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.242 ± 0.016 | 0.207 ± 0.020 | **0.171 ± 0.017** | 0.419 ± 0.156 | 0.569 ± 0.170 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.933 ± 0.093 | 0.199 ± 0.017 | **0.168 ± 0.010** | 0.521 ± 0.039 | 0.946 ± 1.738 |
| bin_elementwise_mul_2048x2048 | **1.322 ± 0.254** | 1.377 ± 0.162 | 1.351 ± 0.144 | 2.842 ± 0.715 | 1.626 ± 0.451 |
| bin_elementwise_mul_512x512 | **0.064 ± 0.001** | 0.066 ± 0.000 | 0.074 ± 0.003 | 0.363 ± 0.177 | 0.096 ± 0.028 |
| bin_matmul_1024 | 9.215 ± 0.715 | **7.704 ± 0.762** | 8.114 ± 0.759 | 56.781 ± 5.021 | 54.320 ± 7.732 |
| bin_matmul_256 | 0.202 ± 0.070 | **0.114 ± 0.005** | 0.116 ± 0.005 | 1.010 ± 0.115 | 0.981 ± 0.079 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.006 ± 0.002 | **0.003 ± 0.000** | 0.021 ± 0.002 | 0.096 ± 0.019 | 0.011 ± 0.005 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.031 ± 0.004 | **0.026 ± 0.004** | 0.078 ± 0.006 | 0.156 ± 0.065 | 0.079 ± 0.179 |
| bin_omeinsum_matmul_10x10 | 0.014 ± 0.001 | **0.007 ± 0.001** | 0.015 ± 0.001 | 0.063 ± 0.003 | 0.032 ± 0.019 |
| bin_outer_product_4096 | 3.784 ± 0.464 | 3.408 ± 0.127 | **2.719 ± 0.117** | 2.834 ± 0.312 | 106.610 ± 22.888 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 14.066 ± 1.335 | **8.677 ± 1.629** | 10.733 ± 1.654 | 60.527 ± 3.768 | 47.454 ± 4.612 |
| gm_queen5_5_3.wcsp | 783.909 ± 67.656 | 840.967 ± 36.819 | **745.227 ± 63.665** | 1237.735 ± 45.427 | 910.764 ± 105.347 |
| lm_batch_likelihood_brackets_4_4d | 17.240 ± 3.809 | 16.165 ± 1.155 | 13.599 ± 1.243 | **12.338 ± 1.126** | 17.871 ± 12.122 |
| lm_batch_likelihood_sentence_3_12d | 27.135 ± 2.411 | 38.364 ± 3.306 | **22.880 ± 4.006** | 53.457 ± 3.668 | 59.966 ± 24.353 |
| lm_batch_likelihood_sentence_4_4d | 14.971 ± 2.042 | 18.930 ± 1.489 | **13.709 ± 1.865** | 14.430 ± 1.159 | 21.863 ± 33.756 |
| nary_matmul_chain_64 | 0.017 ± 0.001 | **0.010 ± 0.000** | 0.019 ± 0.002 | 0.159 ± 0.056 | 0.101 ± 0.018 |
| str_matrix_chain_multiplication_100 | 5.105 ± 0.524 | 4.825 ± 0.297 | **3.928 ± 0.170** | 13.274 ± 2.095 | 17.068 ± 2.594 |
| str_mps_varying_inner_product_200 | 10.095 ± 0.984 | 10.308 ± 0.357 | **7.441 ± 0.337** | 21.667 ± 1.285 | 18.935 ± 8.667 |
| str_nw_mera_closed_120 | 295.097 ± 56.118 | 271.033 ± 23.144 | **243.574 ± 6.469** | 1400.590 ± 94.895 | 1283.305 ± 66.022 |
| str_nw_mera_open_26 | 301.651 ± 26.490 | 377.753 ± 38.604 | **232.373 ± 5.080** | 1092.234 ± 126.943 | 990.316 ± 64.452 |
| tensornetwork_permutation_focus_step409_316 | **151.453 ± 33.700** | 285.278 ± 15.932 | 230.843 ± 9.327 | 337.217 ± 16.037 | 200.198 ± 20.500 |
| tensornetwork_permutation_light_415 | 142.144 ± 17.697 | **124.880 ± 10.013** | 238.356 ± 9.559 | 394.634 ± 57.568 | 203.306 ± 17.837 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261116_174236/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261116_174236/run.yaml`
- Timestamp: `2026-11-16T17:42:36.879712+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261116_174236`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Apple M2 Pro (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `5`
- Physical CPUs: `5`
- Sockets: `1`
- Cores per socket: `5`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 5 physical / 5 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 5 CPUs/L2)`
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

- `data/results/mac-cpu/cpu/einsum/20261116_174236/jax_cpu_t4_20261116_174236.log`
- `data/results/mac-cpu/cpu/einsum/20261116_174236/julia_omeinsum_t4_20261116_174236.log`
- `data/results/mac-cpu/cpu/einsum/20261116_174236/pytorch_cpu_t4_20261116_174236.log`
- `data/results/mac-cpu/cpu/einsum/20261116_174236/tenferro_eager_t4_20261116_174236.log`
- `data/results/mac-cpu/cpu/einsum/20261116_174236/tenferro_trace_t4_20261116_174236.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.715 ± 0.098 | **0.509 ± 0.044** | 0.655 ± 0.069 | 1.660 ± 0.424 | 4.513 ± 2.061 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.130 ± 0.014 | **0.077 ± 0.006** | 0.140 ± 0.009 | 0.681 ± 0.090 | 0.668 ± 0.286 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.090 ± 0.009** | 0.137 ± 0.015 | 0.174 ± 0.035 | 0.364 ± 0.053 | 0.621 ± 0.166 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.114 ± 0.003 | **0.090 ± 0.006** | 0.183 ± 0.023 | 0.361 ± 0.095 | 1.181 ± 2.120 |
| bin_elementwise_mul_2048x2048 | **0.837 ± 0.078** | 1.159 ± 0.295 | 0.963 ± 0.171 | 2.014 ± 0.044 | 1.796 ± 0.225 |
| bin_elementwise_mul_512x512 | 0.039 ± 0.017 | **0.031 ± 0.003** | 0.117 ± 0.021 | 0.243 ± 0.035 | 0.063 ± 0.019 |
| bin_matmul_1024 | 7.689 ± 0.851 | 7.582 ± 0.469 | **7.255 ± 0.530** | 12.434 ± 0.405 | 24.061 ± 4.896 |
| bin_matmul_256 | 0.142 ± 0.006 | **0.116 ± 0.014** | 0.123 ± 0.007 | 0.418 ± 0.023 | 0.567 ± 0.216 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.004 ± 0.003** | 0.021 ± 0.005 | 0.093 ± 0.009 | 0.009 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.003 | **0.028 ± 0.003** | 0.086 ± 0.005 | 0.176 ± 0.078 | 0.048 ± 0.011 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.008 ± 0.000** | 0.016 ± 0.001 | 0.070 ± 0.018 | 0.016 ± 0.000 |
| bin_outer_product_4096 | **1.368 ± 0.110** | 1.990 ± 0.746 | 1.735 ± 0.364 | 1.980 ± 0.779 | 103.778 ± 17.680 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 12.032 ± 1.053 | 9.091 ± 0.908 | **8.683 ± 0.448** | 16.611 ± 3.521 | 16.490 ± 1.014 |
| gm_queen5_5_3.wcsp | 1872.745 ± 166.680 | 1761.336 ± 288.109 | **676.936 ± 34.371** | 1259.812 ± 33.877 | 2115.914 ± 564.494 |
| lm_batch_likelihood_brackets_4_4d | **10.163 ± 1.022** | 14.115 ± 1.396 | 12.778 ± 3.178 | 14.693 ± 0.621 | 15.998 ± 7.020 |
| lm_batch_likelihood_sentence_3_12d | **17.413 ± 2.532** | 31.980 ± 1.955 | 26.435 ± 4.394 | 23.484 ± 0.943 | 28.735 ± 10.642 |
| lm_batch_likelihood_sentence_4_4d | **9.745 ± 1.084** | 16.415 ± 1.381 | 13.048 ± 1.752 | 16.089 ± 0.870 | 15.210 ± 7.027 |
| nary_matmul_chain_64 | 0.018 ± 0.001 | **0.012 ± 0.000** | 0.025 ± 0.001 | 0.139 ± 0.059 | 0.079 ± 0.024 |
| str_matrix_chain_multiplication_100 | **4.144 ± 0.363** | 4.177 ± 0.274 | 5.057 ± 0.973 | 13.679 ± 2.261 | 8.948 ± 2.663 |
| str_mps_varying_inner_product_200 | 8.288 ± 0.389 | 9.233 ± 0.435 | **7.917 ± 0.570** | 23.348 ± 1.132 | 15.498 ± 2.395 |
| str_nw_mera_closed_120 | 324.607 ± 15.523 | 292.643 ± 14.666 | **270.513 ± 21.299** | 378.346 ± 24.300 | 423.743 ± 75.057 |
| str_nw_mera_open_26 | 255.122 ± 16.063 | 307.027 ± 18.518 | **189.524 ± 10.420** | 285.729 ± 13.427 | 376.239 ± 57.729 |
| tensornetwork_permutation_focus_step409_316 | 129.424 ± 10.846 | 231.945 ± 14.543 | **113.413 ± 5.971** | 195.588 ± 8.631 | 120.754 ± 18.427 |
| tensornetwork_permutation_light_415 | 123.403 ± 10.898 | **107.607 ± 4.524** | 118.774 ± 9.748 | 199.932 ± 15.809 | 111.976 ± 11.398 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.715 ± 0.098 | **0.509 ± 0.044** | 0.655 ± 0.069 | 1.660 ± 0.424 | 4.513 ± 2.061 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.130 ± 0.014 | **0.077 ± 0.006** | 0.140 ± 0.009 | 0.681 ± 0.090 | 0.668 ± 0.286 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.090 ± 0.009** | 0.137 ± 0.015 | 0.174 ± 0.035 | 0.364 ± 0.053 | 0.621 ± 0.166 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.114 ± 0.003 | **0.090 ± 0.006** | 0.183 ± 0.023 | 0.361 ± 0.095 | 1.181 ± 2.120 |
| bin_elementwise_mul_2048x2048 | **0.837 ± 0.078** | 1.159 ± 0.295 | 0.963 ± 0.171 | 2.014 ± 0.044 | 1.796 ± 0.225 |
| bin_elementwise_mul_512x512 | 0.039 ± 0.017 | **0.031 ± 0.003** | 0.117 ± 0.021 | 0.243 ± 0.035 | 0.063 ± 0.019 |
| bin_matmul_1024 | 7.689 ± 0.851 | 7.582 ± 0.469 | **7.255 ± 0.530** | 12.434 ± 0.405 | 24.061 ± 4.896 |
| bin_matmul_256 | 0.142 ± 0.006 | **0.116 ± 0.014** | 0.123 ± 0.007 | 0.418 ± 0.023 | 0.567 ± 0.216 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.004 ± 0.003** | 0.021 ± 0.005 | 0.093 ± 0.009 | 0.009 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.003 | **0.028 ± 0.003** | 0.086 ± 0.005 | 0.176 ± 0.078 | 0.048 ± 0.011 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.008 ± 0.000** | 0.016 ± 0.001 | 0.070 ± 0.018 | 0.016 ± 0.000 |
| bin_outer_product_4096 | **1.368 ± 0.110** | 1.990 ± 0.746 | 1.735 ± 0.364 | 1.980 ± 0.779 | 103.778 ± 17.680 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 12.032 ± 1.053 | 9.091 ± 0.908 | **8.683 ± 0.448** | 16.611 ± 3.521 | 16.490 ± 1.014 |
| gm_queen5_5_3.wcsp | 553.085 ± 17.954 | 486.705 ± 35.161 | **387.614 ± 20.717** | 457.549 ± 24.512 | 606.593 ± 115.646 |
| lm_batch_likelihood_brackets_4_4d | **11.114 ± 1.266** | 18.545 ± 6.850 | 14.198 ± 0.375 | 14.749 ± 0.527 | 18.775 ± 12.816 |
| lm_batch_likelihood_sentence_3_12d | **18.329 ± 2.182** | 34.300 ± 3.658 | 25.259 ± 1.305 | 25.305 ± 1.874 | 50.411 ± 25.111 |
| lm_batch_likelihood_sentence_4_4d | **9.300 ± 1.522** | 16.539 ± 0.469 | 13.713 ± 0.408 | 14.875 ± 1.163 | 20.738 ± 22.743 |
| nary_matmul_chain_64 | 0.018 ± 0.001 | **0.012 ± 0.000** | 0.025 ± 0.001 | 0.139 ± 0.059 | 0.079 ± 0.024 |
| str_matrix_chain_multiplication_100 | **3.890 ± 0.494** | 4.326 ± 0.746 | 4.736 ± 0.235 | 12.855 ± 0.765 | 10.574 ± 3.592 |
| str_mps_varying_inner_product_200 | 9.514 ± 1.596 | **8.113 ± 0.269** | 11.342 ± 0.752 | 26.127 ± 1.119 | 15.532 ± 4.808 |
| str_nw_mera_closed_120 | 242.295 ± 9.209 | 255.499 ± 16.978 | **224.332 ± 20.238** | 341.858 ± 17.626 | 389.537 ± 28.576 |
| str_nw_mera_open_26 | 281.828 ± 8.441 | 310.928 ± 13.655 | **177.636 ± 8.406** | 287.514 ± 9.368 | 397.762 ± 45.233 |
| tensornetwork_permutation_focus_step409_316 | 129.424 ± 10.846 | 231.945 ± 14.543 | **113.413 ± 5.971** | 195.588 ± 8.631 | 120.754 ± 18.427 |
| tensornetwork_permutation_light_415 | 123.403 ± 10.898 | **107.607 ± 4.524** | 118.774 ± 9.748 | 199.932 ± 15.809 | 111.976 ± 11.398 |

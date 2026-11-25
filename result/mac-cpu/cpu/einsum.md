# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `74775e4a2ad3480fb9ee9f859e0037b8cac731f2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261125_172313/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261125_172313/run.yaml`
- Timestamp: `2026-11-25T17:23:13.361566+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261125_172313`.

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

- `data/results/mac-cpu/cpu/einsum/20261125_172313/jax_cpu_t1_20261125_172313.log`
- `data/results/mac-cpu/cpu/einsum/20261125_172313/julia_omeinsum_t1_20261125_172313.log`
- `data/results/mac-cpu/cpu/einsum/20261125_172313/pytorch_cpu_t1_20261125_172313.log`
- `data/results/mac-cpu/cpu/einsum/20261125_172313/tenferro_eager_t1_20261125_172313.log`
- `data/results/mac-cpu/cpu/einsum/20261125_172313/tenferro_trace_t1_20261125_172313.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.458 ± 3.204 | 3.746 ± 0.463 | **0.683 ± 0.071** | 3.853 ± 0.551 | 7.320 ± 13.508 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.600 ± 0.055 | 0.687 ± 0.193 | **0.129 ± 0.004** | 0.681 ± 0.035 | 2.161 ± 4.783 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.235 ± 0.016 | 0.286 ± 0.066 | **0.207 ± 0.034** | 0.473 ± 0.073 | 0.957 ± 1.806 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.971 ± 0.215 | 0.272 ± 0.036 | **0.193 ± 0.003** | 0.561 ± 0.070 | 1.887 ± 7.218 |
| bin_elementwise_mul_2048x2048 | 2.571 ± 0.558 | 2.478 ± 0.318 | **2.152 ± 0.157** | 3.517 ± 0.115 | 3.251 ± 2.635 |
| bin_elementwise_mul_512x512 | 0.089 ± 0.016 | 0.084 ± 0.030 | 0.121 ± 0.014 | 0.563 ± 0.163 | **0.078 ± 0.065** |
| bin_matmul_1024 | 61.470 ± 11.615 | 50.997 ± 3.428 | **12.254 ± 0.858** | 56.458 ± 1.458 | 83.957 ± 17.276 |
| bin_matmul_256 | 0.929 ± 0.083 | 0.818 ± 0.026 | **0.135 ± 0.026** | 0.965 ± 0.034 | 1.390 ± 0.514 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.014 ± 0.010 | **0.004 ± 0.000** | 0.018 ± 0.001 | 0.090 ± 0.014 | 0.013 ± 0.005 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.023 ± 0.002** | 0.068 ± 0.002 | 0.077 ± 0.002 | 0.196 ± 0.051 | 0.178 ± 0.096 |
| bin_omeinsum_matmul_10x10 | **0.013 ± 0.001** | 0.022 ± 0.001 | 0.017 ± 0.001 | 0.072 ± 0.014 | 0.033 ± 0.001 |
| bin_outer_product_4096 | 5.997 ± 1.875 | 5.386 ± 0.500 | 3.082 ± 0.409 | **2.918 ± 0.361** | 310.994 ± 69.169 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 111.394 ± 6.514 | 106.979 ± 12.257 | **11.762 ± 1.359** | 58.746 ± 6.603 | 85.451 ± 36.814 |
| gm_queen5_5_3.wcsp | 6604.604 ± 891.988 | 5918.589 ± 919.881 | **4001.773 ± 278.340** | 5657.568 ± 461.707 | 8400.580 ± 906.588 |
| lm_batch_likelihood_brackets_4_4d | 28.551 ± 1.823 | 28.834 ± 2.726 | 16.042 ± 6.183 | **13.133 ± 0.342** | 37.810 ± 39.157 |
| lm_batch_likelihood_sentence_3_12d | 107.732 ± 19.151 | 102.686 ± 14.053 | **30.226 ± 1.819** | 53.554 ± 1.947 | 123.700 ± 40.004 |
| lm_batch_likelihood_sentence_4_4d | 31.413 ± 3.122 | 31.139 ± 12.604 | 16.556 ± 0.782 | **15.962 ± 0.648** | 36.331 ± 35.274 |
| nary_matmul_chain_64 | 0.108 ± 0.079 | 0.034 ± 0.000 | **0.026 ± 0.011** | 0.134 ± 0.019 | 0.115 ± 0.017 |
| str_matrix_chain_multiplication_100 | 17.347 ± 4.204 | 16.022 ± 2.484 | **5.170 ± 0.120** | 13.431 ± 0.612 | 20.058 ± 6.148 |
| str_mps_varying_inner_product_200 | 22.952 ± 3.362 | 18.048 ± 1.252 | **7.296 ± 0.219** | 17.882 ± 0.473 | 21.882 ± 6.321 |
| str_nw_mera_closed_120 | 2443.351 ± 192.274 | 2253.339 ± 268.174 | **347.806 ± 13.671** | 1595.920 ± 45.028 | 1718.227 ± 322.622 |
| str_nw_mera_open_26 | 2138.993 ± 800.240 | 1893.235 ± 503.778 | **233.088 ± 20.051** | 1145.109 ± 70.403 | 1151.013 ± 154.310 |
| tensornetwork_permutation_focus_step409_316 | 410.431 ± 50.705 | 379.283 ± 31.124 | 259.334 ± 10.402 | 389.321 ± 42.699 | **253.724 ± 81.117** |
| tensornetwork_permutation_light_415 | 460.401 ± 50.460 | 316.026 ± 43.671 | **215.933 ± 7.151** | 398.616 ± 8.591 | 232.462 ± 24.258 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.458 ± 3.204 | 3.746 ± 0.463 | **0.683 ± 0.071** | 3.853 ± 0.551 | 7.320 ± 13.508 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.600 ± 0.055 | 0.687 ± 0.193 | **0.129 ± 0.004** | 0.681 ± 0.035 | 2.161 ± 4.783 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.235 ± 0.016 | 0.286 ± 0.066 | **0.207 ± 0.034** | 0.473 ± 0.073 | 0.957 ± 1.806 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.971 ± 0.215 | 0.272 ± 0.036 | **0.193 ± 0.003** | 0.561 ± 0.070 | 1.887 ± 7.218 |
| bin_elementwise_mul_2048x2048 | 2.571 ± 0.558 | 2.478 ± 0.318 | **2.152 ± 0.157** | 3.517 ± 0.115 | 3.251 ± 2.635 |
| bin_elementwise_mul_512x512 | 0.089 ± 0.016 | 0.084 ± 0.030 | 0.121 ± 0.014 | 0.563 ± 0.163 | **0.078 ± 0.065** |
| bin_matmul_1024 | 61.470 ± 11.615 | 50.997 ± 3.428 | **12.254 ± 0.858** | 56.458 ± 1.458 | 83.957 ± 17.276 |
| bin_matmul_256 | 0.929 ± 0.083 | 0.818 ± 0.026 | **0.135 ± 0.026** | 0.965 ± 0.034 | 1.390 ± 0.514 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.014 ± 0.010 | **0.004 ± 0.000** | 0.018 ± 0.001 | 0.090 ± 0.014 | 0.013 ± 0.005 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.023 ± 0.002** | 0.068 ± 0.002 | 0.077 ± 0.002 | 0.196 ± 0.051 | 0.178 ± 0.096 |
| bin_omeinsum_matmul_10x10 | **0.013 ± 0.001** | 0.022 ± 0.001 | 0.017 ± 0.001 | 0.072 ± 0.014 | 0.033 ± 0.001 |
| bin_outer_product_4096 | 5.997 ± 1.875 | 5.386 ± 0.500 | 3.082 ± 0.409 | **2.918 ± 0.361** | 310.994 ± 69.169 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 111.394 ± 6.514 | 106.979 ± 12.257 | **11.762 ± 1.359** | 58.746 ± 6.603 | 85.451 ± 36.814 |
| gm_queen5_5_3.wcsp | 2400.584 ± 536.580 | 1808.467 ± 316.644 | **811.419 ± 109.447** | 1273.833 ± 109.926 | 2457.266 ± 491.123 |
| lm_batch_likelihood_brackets_4_4d | 28.864 ± 1.525 | 24.215 ± 1.561 | 15.327 ± 0.856 | **12.987 ± 0.364** | 28.350 ± 12.041 |
| lm_batch_likelihood_sentence_3_12d | 87.601 ± 5.642 | 100.136 ± 3.774 | **25.779 ± 1.635** | 65.786 ± 6.037 | 113.495 ± 67.496 |
| lm_batch_likelihood_sentence_4_4d | 23.649 ± 2.658 | 27.517 ± 1.076 | 16.448 ± 1.508 | **13.870 ± 0.450** | 65.320 ± 55.204 |
| nary_matmul_chain_64 | 0.108 ± 0.079 | 0.034 ± 0.000 | **0.026 ± 0.011** | 0.134 ± 0.019 | 0.115 ± 0.017 |
| str_matrix_chain_multiplication_100 | 14.988 ± 5.093 | 14.225 ± 0.133 | **5.544 ± 0.413** | 14.053 ± 0.521 | 23.908 ± 13.067 |
| str_mps_varying_inner_product_200 | 23.758 ± 1.957 | 21.320 ± 3.017 | **8.955 ± 0.285** | 24.873 ± 6.085 | 27.982 ± 11.017 |
| str_nw_mera_closed_120 | 2422.116 ± 652.652 | 2229.998 ± 470.848 | **287.812 ± 10.695** | 1492.793 ± 32.137 | 1593.725 ± 195.618 |
| str_nw_mera_open_26 | 1697.934 ± 180.400 | 1604.450 ± 338.909 | **252.301 ± 15.729** | 1251.946 ± 262.593 | 1486.227 ± 456.802 |
| tensornetwork_permutation_focus_step409_316 | 410.431 ± 50.705 | 379.283 ± 31.124 | 259.334 ± 10.402 | 389.321 ± 42.699 | **253.724 ± 81.117** |
| tensornetwork_permutation_light_415 | 460.401 ± 50.460 | 316.026 ± 43.671 | **215.933 ± 7.151** | 398.616 ± 8.591 | 232.462 ± 24.258 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261125_180002/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261125_180002/run.yaml`
- Timestamp: `2026-11-25T18:00:02.301847+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261125_180002`.

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

- `data/results/mac-cpu/cpu/einsum/20261125_180002/jax_cpu_t4_20261125_180002.log`
- `data/results/mac-cpu/cpu/einsum/20261125_180002/julia_omeinsum_t4_20261125_180002.log`
- `data/results/mac-cpu/cpu/einsum/20261125_180002/pytorch_cpu_t4_20261125_180002.log`
- `data/results/mac-cpu/cpu/einsum/20261125_180002/tenferro_eager_t4_20261125_180002.log`
- `data/results/mac-cpu/cpu/einsum/20261125_180002/tenferro_trace_t4_20261125_180002.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.474 ± 1.972 | 4.079 ± 0.351 | **0.837 ± 0.087** | 1.890 ± 0.121 | 89.054 ± 27.728 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.084 ± 0.298 | 0.740 ± 0.147 | **0.513 ± 0.051** | 0.711 ± 0.079 | 1.376 ± 4.236 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.339 ± 0.446 | **0.196 ± 0.067** | 0.569 ± 0.184 | 0.482 ± 0.057 | 1.749 ± 2.414 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.833 ± 0.737 | **0.200 ± 0.012** | 0.691 ± 0.187 | 0.537 ± 0.111 | 3.230 ± 21.765 |
| bin_elementwise_mul_2048x2048 | 3.302 ± 0.361 | **2.643 ± 0.198** | 3.062 ± 0.260 | 3.897 ± 0.254 | 3.775 ± 1.301 |
| bin_elementwise_mul_512x512 | 0.169 ± 0.212 | 0.131 ± 0.026 | 0.199 ± 0.066 | 0.391 ± 0.032 | **0.088 ± 0.025** |
| bin_matmul_1024 | 30.700 ± 3.698 | 25.077 ± 2.046 | **15.001 ± 1.741** | 20.938 ± 4.243 | 53.507 ± 21.083 |
| bin_matmul_256 | 0.812 ± 0.106 | 0.479 ± 0.034 | **0.134 ± 0.014** | 0.526 ± 0.042 | 3.724 ± 2.735 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.004 ± 0.000** | 0.019 ± 0.004 | 0.088 ± 0.006 | 0.012 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.022 ± 0.000** | 0.024 ± 0.005 | 0.076 ± 0.007 | 0.257 ± 0.091 | 0.110 ± 0.043 |
| bin_omeinsum_matmul_10x10 | 0.013 ± 0.001 | **0.007 ± 0.000** | 0.019 ± 0.019 | 0.124 ± 0.059 | 0.019 ± 0.004 |
| bin_outer_product_4096 | 3.984 ± 1.256 | 3.559 ± 0.526 | 3.808 ± 1.162 | **3.171 ± 0.854** | 235.415 ± 17.554 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 50.561 ± 6.996 | 52.100 ± 4.494 | **12.375 ± 0.895** | 52.245 ± 9.562 | 40.195 ± 10.360 |
| gm_queen5_5_3.wcsp | **3112.171 ± 495.591** | 4681.901 ± 1207.892 | 5262.248 ± 411.619 | 5370.642 ± 1424.204 | 10906.326 ± 1785.462 |
| lm_batch_likelihood_brackets_4_4d | 24.988 ± 4.712 | 26.548 ± 5.732 | 25.866 ± 7.039 | **22.394 ± 4.043** | 170.022 ± 148.164 |
| lm_batch_likelihood_sentence_3_12d | 56.795 ± 5.452 | 71.458 ± 19.081 | **38.062 ± 5.774** | 56.782 ± 11.717 | 109.590 ± 7.801 |
| lm_batch_likelihood_sentence_4_4d | **24.995 ± 3.200** | 32.258 ± 9.008 | 27.247 ± 4.300 | 27.254 ± 4.638 | 71.917 ± 13.609 |
| nary_matmul_chain_64 | 0.103 ± 0.021 | 0.077 ± 0.060 | **0.024 ± 0.002** | 0.308 ± 0.006 | 0.059 ± 0.035 |
| str_matrix_chain_multiplication_100 | 15.597 ± 0.906 | 18.242 ± 2.321 | **6.930 ± 1.893** | 23.633 ± 9.940 | 329.996 ± 142.383 |
| str_mps_varying_inner_product_200 | 37.290 ± 2.609 | 44.428 ± 3.426 | **11.383 ± 3.613** | 36.005 ± 9.531 | 24.390 ± 16.315 |
| str_nw_mera_closed_120 | 1318.608 ± 97.236 | 1273.357 ± 87.429 | **374.939 ± 36.741** | 818.371 ± 179.948 | 1125.410 ± 137.496 |
| str_nw_mera_open_26 | 850.424 ± 44.444 | 885.847 ± 172.370 | **308.630 ± 34.327** | 674.067 ± 176.220 | 887.149 ± 120.923 |
| tensornetwork_permutation_focus_step409_316 | **159.185 ± 10.796** | 191.275 ± 15.561 | 187.674 ± 38.146 | 272.535 ± 85.962 | 441.679 ± 24.573 |
| tensornetwork_permutation_light_415 | 157.472 ± 18.340 | **146.942 ± 13.047** | 176.851 ± 29.497 | 283.977 ± 76.069 | 438.280 ± 20.358 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.474 ± 1.972 | 4.079 ± 0.351 | **0.837 ± 0.087** | 1.890 ± 0.121 | 89.054 ± 27.728 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.084 ± 0.298 | 0.740 ± 0.147 | **0.513 ± 0.051** | 0.711 ± 0.079 | 1.376 ± 4.236 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.339 ± 0.446 | **0.196 ± 0.067** | 0.569 ± 0.184 | 0.482 ± 0.057 | 1.749 ± 2.414 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.833 ± 0.737 | **0.200 ± 0.012** | 0.691 ± 0.187 | 0.537 ± 0.111 | 3.230 ± 21.765 |
| bin_elementwise_mul_2048x2048 | 3.302 ± 0.361 | **2.643 ± 0.198** | 3.062 ± 0.260 | 3.897 ± 0.254 | 3.775 ± 1.301 |
| bin_elementwise_mul_512x512 | 0.169 ± 0.212 | 0.131 ± 0.026 | 0.199 ± 0.066 | 0.391 ± 0.032 | **0.088 ± 0.025** |
| bin_matmul_1024 | 30.700 ± 3.698 | 25.077 ± 2.046 | **15.001 ± 1.741** | 20.938 ± 4.243 | 53.507 ± 21.083 |
| bin_matmul_256 | 0.812 ± 0.106 | 0.479 ± 0.034 | **0.134 ± 0.014** | 0.526 ± 0.042 | 3.724 ± 2.735 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.004 ± 0.000** | 0.019 ± 0.004 | 0.088 ± 0.006 | 0.012 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.022 ± 0.000** | 0.024 ± 0.005 | 0.076 ± 0.007 | 0.257 ± 0.091 | 0.110 ± 0.043 |
| bin_omeinsum_matmul_10x10 | 0.013 ± 0.001 | **0.007 ± 0.000** | 0.019 ± 0.019 | 0.124 ± 0.059 | 0.019 ± 0.004 |
| bin_outer_product_4096 | 3.984 ± 1.256 | 3.559 ± 0.526 | 3.808 ± 1.162 | **3.171 ± 0.854** | 235.415 ± 17.554 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 50.561 ± 6.996 | 52.100 ± 4.494 | **12.375 ± 0.895** | 52.245 ± 9.562 | 40.195 ± 10.360 |
| gm_queen5_5_3.wcsp | 1122.765 ± 85.931 | 1065.990 ± 116.704 | **700.008 ± 272.770** | 1052.310 ± 256.615 | 1941.627 ± 553.906 |
| lm_batch_likelihood_brackets_4_4d | 20.925 ± 3.290 | 22.448 ± 4.276 | 17.211 ± 8.598 | **16.193 ± 1.983** | 115.265 ± 54.807 |
| lm_batch_likelihood_sentence_3_12d | 48.201 ± 2.579 | 61.020 ± 6.296 | 31.187 ± 11.639 | **27.882 ± 1.351** | 138.567 ± 36.440 |
| lm_batch_likelihood_sentence_4_4d | 20.159 ± 0.895 | 20.680 ± 2.150 | 32.895 ± 4.176 | **17.370 ± 1.251** | 107.517 ± 44.403 |
| nary_matmul_chain_64 | 0.103 ± 0.021 | 0.077 ± 0.060 | **0.024 ± 0.002** | 0.308 ± 0.006 | 0.059 ± 0.035 |
| str_matrix_chain_multiplication_100 | 15.401 ± 1.934 | 13.896 ± 3.063 | **6.687 ± 0.693** | 14.338 ± 3.295 | 308.710 ± 90.689 |
| str_mps_varying_inner_product_200 | 44.514 ± 2.794 | 43.077 ± 6.569 | **12.731 ± 3.396** | 37.231 ± 10.746 | 34.893 ± 16.861 |
| str_nw_mera_closed_120 | 1315.577 ± 124.771 | 1288.118 ± 90.688 | **320.147 ± 63.895** | 789.723 ± 145.149 | 860.110 ± 123.224 |
| str_nw_mera_open_26 | 840.133 ± 79.447 | 1046.608 ± 292.275 | **319.526 ± 147.086** | 893.184 ± 165.408 | 972.770 ± 106.772 |
| tensornetwork_permutation_focus_step409_316 | **159.185 ± 10.796** | 191.275 ± 15.561 | 187.674 ± 38.146 | 272.535 ± 85.962 | 441.679 ± 24.573 |
| tensornetwork_permutation_light_415 | 157.472 ± 18.340 | **146.942 ± 13.047** | 176.851 ± 29.497 | 283.977 ± 76.069 | 438.280 ± 20.358 |

# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `89c207f1ca97cde3ae0f31586747a039ca4e5cc6`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261114_172310/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261114_172310/run.yaml`
- Timestamp: `2026-11-14T17:23:10.539309+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261114_172310`.

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

- `data/results/mac-cpu/cpu/einsum/20261114_172310/jax_cpu_t1_20261114_172310.log`
- `data/results/mac-cpu/cpu/einsum/20261114_172310/julia_omeinsum_t1_20261114_172310.log`
- `data/results/mac-cpu/cpu/einsum/20261114_172310/pytorch_cpu_t1_20261114_172310.log`
- `data/results/mac-cpu/cpu/einsum/20261114_172310/tenferro_eager_t1_20261114_172310.log`
- `data/results/mac-cpu/cpu/einsum/20261114_172310/tenferro_trace_t1_20261114_172310.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.447 ± 0.057 | **0.697 ± 0.282** | 1.067 ± 0.368 | 4.889 ± 0.930 | 5.940 ± 8.939 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.155 ± 0.006 | **0.094 ± 0.011** | 0.135 ± 0.007 | 0.854 ± 0.291 | 1.964 ± 2.584 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.225 ± 0.003** | 0.252 ± 0.028 | 0.236 ± 0.020 | 0.842 ± 0.580 | 0.867 ± 2.225 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.010 ± 0.063 | **0.225 ± 0.009** | 0.231 ± 0.045 | 0.659 ± 0.218 | 1.382 ± 2.607 |
| bin_elementwise_mul_2048x2048 | **2.416 ± 0.334** | 2.568 ± 0.616 | 2.758 ± 0.980 | 3.982 ± 0.953 | 3.223 ± 1.345 |
| bin_elementwise_mul_512x512 | **0.076 ± 0.000** | 0.081 ± 0.018 | 0.083 ± 0.026 | 0.279 ± 0.786 | 0.079 ± 0.020 |
| bin_matmul_1024 | 16.204 ± 2.445 | 14.772 ± 4.152 | **14.225 ± 2.921** | 82.906 ± 18.360 | 67.123 ± 17.914 |
| bin_matmul_256 | 0.159 ± 0.004 | 0.130 ± 0.483 | **0.129 ± 0.004** | 1.091 ± 0.535 | 1.093 ± 0.381 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.005 ± 0.001** | 0.018 ± 0.001 | 0.091 ± 0.015 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.028 ± 0.002 | **0.025 ± 0.003** | 0.084 ± 0.021 | 0.168 ± 0.043 | 0.088 ± 0.014 |
| bin_omeinsum_matmul_10x10 | 0.014 ± 0.000 | **0.007 ± 0.001** | 0.019 ± 0.009 | 0.069 ± 0.004 | 0.029 ± 0.014 |
| bin_outer_product_4096 | 4.699 ± 0.309 | 7.373 ± 3.151 | 4.311 ± 0.788 | **3.374 ± 0.776** | 256.244 ± 35.105 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 19.082 ± 5.280 | **16.769 ± 12.205** | 19.223 ± 6.786 | 69.063 ± 10.204 | 78.978 ± 4.760 |
| gm_queen5_5_3.wcsp | 10361.866 ± 2077.613 | 8772.093 ± 3376.388 | **5054.240 ± 337.642** | 7188.057 ± 389.324 | 9098.374 ± 2182.300 |
| lm_batch_likelihood_brackets_4_4d | 19.648 ± 5.600 | 22.200 ± 13.439 | 18.023 ± 3.400 | **16.151 ± 1.694** | 41.734 ± 51.875 |
| lm_batch_likelihood_sentence_3_12d | 33.640 ± 5.395 | 49.556 ± 22.294 | **32.739 ± 1.255** | 66.952 ± 5.725 | 129.053 ± 40.248 |
| lm_batch_likelihood_sentence_4_4d | 22.179 ± 4.615 | 30.146 ± 12.717 | 18.139 ± 1.460 | **17.851 ± 2.344** | 42.543 ± 50.898 |
| nary_matmul_chain_64 | **0.018 ± 0.008** | 0.030 ± 0.028 | 0.024 ± 0.003 | 0.419 ± 0.373 | 0.294 ± 0.270 |
| str_matrix_chain_multiplication_100 | **5.133 ± 0.533** | 5.496 ± 0.732 | 6.161 ± 1.329 | 16.916 ± 1.063 | 25.268 ± 6.923 |
| str_mps_varying_inner_product_200 | 9.994 ± 1.186 | 11.500 ± 4.150 | **7.684 ± 1.108** | 24.714 ± 5.924 | 30.524 ± 10.693 |
| str_nw_mera_closed_120 | 647.288 ± 104.489 | 532.692 ± 61.463 | **441.719 ± 77.641** | 1812.637 ± 219.936 | 2856.105 ± 178.301 |
| str_nw_mera_open_26 | 477.052 ± 158.385 | 861.187 ± 310.409 | **273.186 ± 20.826** | 1337.622 ± 165.102 | 1523.961 ± 143.775 |
| tensornetwork_permutation_focus_step409_316 | **212.206 ± 46.102** | 498.717 ± 75.179 | 297.973 ± 57.141 | 436.950 ± 24.041 | 329.316 ± 75.459 |
| tensornetwork_permutation_light_415 | **177.949 ± 23.627** | 204.267 ± 68.638 | 257.151 ± 29.632 | 453.113 ± 40.924 | 305.391 ± 28.719 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.447 ± 0.057 | **0.697 ± 0.282** | 1.067 ± 0.368 | 4.889 ± 0.930 | 5.940 ± 8.939 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.155 ± 0.006 | **0.094 ± 0.011** | 0.135 ± 0.007 | 0.854 ± 0.291 | 1.964 ± 2.584 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.225 ± 0.003** | 0.252 ± 0.028 | 0.236 ± 0.020 | 0.842 ± 0.580 | 0.867 ± 2.225 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.010 ± 0.063 | **0.225 ± 0.009** | 0.231 ± 0.045 | 0.659 ± 0.218 | 1.382 ± 2.607 |
| bin_elementwise_mul_2048x2048 | **2.416 ± 0.334** | 2.568 ± 0.616 | 2.758 ± 0.980 | 3.982 ± 0.953 | 3.223 ± 1.345 |
| bin_elementwise_mul_512x512 | **0.076 ± 0.000** | 0.081 ± 0.018 | 0.083 ± 0.026 | 0.279 ± 0.786 | 0.079 ± 0.020 |
| bin_matmul_1024 | 16.204 ± 2.445 | 14.772 ± 4.152 | **14.225 ± 2.921** | 82.906 ± 18.360 | 67.123 ± 17.914 |
| bin_matmul_256 | 0.159 ± 0.004 | 0.130 ± 0.483 | **0.129 ± 0.004** | 1.091 ± 0.535 | 1.093 ± 0.381 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.005 ± 0.001** | 0.018 ± 0.001 | 0.091 ± 0.015 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.028 ± 0.002 | **0.025 ± 0.003** | 0.084 ± 0.021 | 0.168 ± 0.043 | 0.088 ± 0.014 |
| bin_omeinsum_matmul_10x10 | 0.014 ± 0.000 | **0.007 ± 0.001** | 0.019 ± 0.009 | 0.069 ± 0.004 | 0.029 ± 0.014 |
| bin_outer_product_4096 | 4.699 ± 0.309 | 7.373 ± 3.151 | 4.311 ± 0.788 | **3.374 ± 0.776** | 256.244 ± 35.105 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 19.082 ± 5.280 | **16.769 ± 12.205** | 19.223 ± 6.786 | 69.063 ± 10.204 | 78.978 ± 4.760 |
| gm_queen5_5_3.wcsp | 1301.560 ± 137.096 | 1346.517 ± 169.203 | **1060.632 ± 139.155** | 1486.611 ± 342.172 | 3190.623 ± 533.528 |
| lm_batch_likelihood_brackets_4_4d | 26.094 ± 9.931 | 36.706 ± 18.343 | 16.519 ± 1.298 | **15.116 ± 1.745** | 35.097 ± 19.227 |
| lm_batch_likelihood_sentence_3_12d | 47.753 ± 11.390 | 49.879 ± 12.294 | **31.702 ± 3.515** | 79.894 ± 6.602 | 133.758 ± 55.211 |
| lm_batch_likelihood_sentence_4_4d | 21.365 ± 6.628 | 31.916 ± 8.327 | 19.511 ± 2.393 | **14.788 ± 1.215** | 41.304 ± 30.516 |
| nary_matmul_chain_64 | **0.018 ± 0.008** | 0.030 ± 0.028 | 0.024 ± 0.003 | 0.419 ± 0.373 | 0.294 ± 0.270 |
| str_matrix_chain_multiplication_100 | 7.127 ± 4.675 | 7.035 ± 1.212 | **6.836 ± 1.942** | 15.477 ± 1.046 | 27.790 ± 7.584 |
| str_mps_varying_inner_product_200 | 14.980 ± 5.326 | 13.968 ± 6.714 | **10.486 ± 1.974** | 24.096 ± 0.999 | 24.976 ± 10.269 |
| str_nw_mera_closed_120 | 500.635 ± 74.050 | 482.568 ± 124.868 | **354.961 ± 37.747** | 1701.333 ± 68.934 | 2151.858 ± 439.604 |
| str_nw_mera_open_26 | 510.347 ± 54.484 | 515.946 ± 65.985 | **314.851 ± 54.534** | 1395.389 ± 176.489 | 1836.511 ± 299.702 |
| tensornetwork_permutation_focus_step409_316 | **212.206 ± 46.102** | 498.717 ± 75.179 | 297.973 ± 57.141 | 436.950 ± 24.041 | 329.316 ± 75.459 |
| tensornetwork_permutation_light_415 | **177.949 ± 23.627** | 204.267 ± 68.638 | 257.151 ± 29.632 | 453.113 ± 40.924 | 305.391 ± 28.719 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261114_175950/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261114_175950/run.yaml`
- Timestamp: `2026-11-14T17:59:50.064820+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261114_175950`.

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

- `data/results/mac-cpu/cpu/einsum/20261114_175950/jax_cpu_t4_20261114_175950.log`
- `data/results/mac-cpu/cpu/einsum/20261114_175950/julia_omeinsum_t4_20261114_175950.log`
- `data/results/mac-cpu/cpu/einsum/20261114_175950/pytorch_cpu_t4_20261114_175950.log`
- `data/results/mac-cpu/cpu/einsum/20261114_175950/tenferro_eager_t4_20261114_175950.log`
- `data/results/mac-cpu/cpu/einsum/20261114_175950/tenferro_trace_t4_20261114_175950.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.601 ± 0.605 | **0.755 ± 0.134** | 0.928 ± 0.185 | 1.847 ± 0.466 | 93.513 ± 13.486 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.290 ± 0.223 | **0.083 ± 0.005** | 0.451 ± 0.628 | 0.750 ± 0.217 | 0.727 ± 0.618 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.347 ± 0.131 | **0.170 ± 0.027** | 0.446 ± 0.250 | 0.401 ± 0.164 | 0.737 ± 0.647 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.391 ± 0.067 | 0.254 ± 0.119 | **0.185 ± 0.033** | 0.393 ± 0.050 | 1.306 ± 0.228 |
| bin_elementwise_mul_2048x2048 | 3.038 ± 0.235 | 3.188 ± 0.632 | 3.246 ± 0.780 | 3.618 ± 0.521 | **2.784 ± 0.393** |
| bin_elementwise_mul_512x512 | 0.146 ± 0.064 | 0.082 ± 0.018 | 0.252 ± 0.345 | 0.302 ± 0.116 | **0.079 ± 0.022** |
| bin_matmul_1024 | 16.679 ± 1.118 | **15.672 ± 5.326** | 20.222 ± 4.912 | 22.158 ± 4.725 | 37.880 ± 8.891 |
| bin_matmul_256 | 0.166 ± 0.045 | **0.124 ± 0.008** | 0.140 ± 0.015 | 0.912 ± 0.213 | 1.997 ± 1.769 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.002 | **0.006 ± 0.003** | 0.018 ± 0.001 | 0.141 ± 0.110 | 0.009 ± 0.000 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.003 | **0.028 ± 0.003** | 0.090 ± 0.140 | 0.207 ± 0.052 | 0.105 ± 0.130 |
| bin_omeinsum_matmul_10x10 | 0.017 ± 0.001 | **0.008 ± 0.000** | 0.050 ± 0.004 | 0.146 ± 0.077 | 0.018 ± 0.014 |
| bin_outer_product_4096 | **3.256 ± 0.730** | 3.470 ± 0.605 | 4.564 ± 0.961 | 3.384 ± 0.591 | 243.728 ± 14.225 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 18.752 ± 2.290 | **16.366 ± 9.116** | 16.832 ± 4.746 | 25.517 ± 5.134 | 32.272 ± 4.247 |
| gm_queen5_5_3.wcsp | 5940.761 ± 282.679 | 7080.628 ± 284.262 | **2877.526 ± 398.703** | 5292.499 ± 628.129 | 8479.172 ± 580.582 |
| lm_batch_likelihood_brackets_4_4d | 18.728 ± 1.945 | 23.318 ± 9.130 | **15.639 ± 6.189** | 26.422 ± 8.569 | 201.937 ± 187.010 |
| lm_batch_likelihood_sentence_3_12d | **24.058 ± 2.755** | 45.488 ± 6.439 | 35.670 ± 4.808 | 39.093 ± 7.707 | 106.520 ± 31.231 |
| lm_batch_likelihood_sentence_4_4d | **18.505 ± 3.335** | 24.369 ± 2.783 | 23.759 ± 5.373 | 21.426 ± 4.463 | 92.938 ± 24.951 |
| nary_matmul_chain_64 | **0.019 ± 0.002** | 0.037 ± 0.026 | 0.024 ± 0.001 | 0.130 ± 0.095 | 0.084 ± 0.058 |
| str_matrix_chain_multiplication_100 | 5.545 ± 1.382 | **5.259 ± 0.500** | 6.151 ± 2.289 | 16.692 ± 2.765 | 370.519 ± 106.069 |
| str_mps_varying_inner_product_200 | 10.284 ± 1.156 | **10.212 ± 0.786** | 11.652 ± 3.178 | 20.837 ± 1.281 | 25.685 ± 12.950 |
| str_nw_mera_closed_120 | 425.822 ± 13.428 | 436.869 ± 104.224 | **365.722 ± 27.584** | 588.140 ± 56.500 | 1040.405 ± 131.572 |
| str_nw_mera_open_26 | 357.369 ± 38.173 | 523.643 ± 98.076 | **240.852 ± 21.364** | 491.936 ± 77.565 | 879.944 ± 76.776 |
| tensornetwork_permutation_focus_step409_316 | **156.930 ± 6.423** | 300.368 ± 26.702 | 165.068 ± 31.090 | 258.073 ± 12.940 | 441.657 ± 295.966 |
| tensornetwork_permutation_light_415 | 152.064 ± 20.889 | 153.987 ± 8.466 | **141.843 ± 19.624** | 300.486 ± 48.634 | 444.765 ± 71.618 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.601 ± 0.605 | **0.755 ± 0.134** | 0.928 ± 0.185 | 1.847 ± 0.466 | 93.513 ± 13.486 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.290 ± 0.223 | **0.083 ± 0.005** | 0.451 ± 0.628 | 0.750 ± 0.217 | 0.727 ± 0.618 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.347 ± 0.131 | **0.170 ± 0.027** | 0.446 ± 0.250 | 0.401 ± 0.164 | 0.737 ± 0.647 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.391 ± 0.067 | 0.254 ± 0.119 | **0.185 ± 0.033** | 0.393 ± 0.050 | 1.306 ± 0.228 |
| bin_elementwise_mul_2048x2048 | 3.038 ± 0.235 | 3.188 ± 0.632 | 3.246 ± 0.780 | 3.618 ± 0.521 | **2.784 ± 0.393** |
| bin_elementwise_mul_512x512 | 0.146 ± 0.064 | 0.082 ± 0.018 | 0.252 ± 0.345 | 0.302 ± 0.116 | **0.079 ± 0.022** |
| bin_matmul_1024 | 16.679 ± 1.118 | **15.672 ± 5.326** | 20.222 ± 4.912 | 22.158 ± 4.725 | 37.880 ± 8.891 |
| bin_matmul_256 | 0.166 ± 0.045 | **0.124 ± 0.008** | 0.140 ± 0.015 | 0.912 ± 0.213 | 1.997 ± 1.769 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.002 | **0.006 ± 0.003** | 0.018 ± 0.001 | 0.141 ± 0.110 | 0.009 ± 0.000 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.003 | **0.028 ± 0.003** | 0.090 ± 0.140 | 0.207 ± 0.052 | 0.105 ± 0.130 |
| bin_omeinsum_matmul_10x10 | 0.017 ± 0.001 | **0.008 ± 0.000** | 0.050 ± 0.004 | 0.146 ± 0.077 | 0.018 ± 0.014 |
| bin_outer_product_4096 | **3.256 ± 0.730** | 3.470 ± 0.605 | 4.564 ± 0.961 | 3.384 ± 0.591 | 243.728 ± 14.225 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 18.752 ± 2.290 | **16.366 ± 9.116** | 16.832 ± 4.746 | 25.517 ± 5.134 | 32.272 ± 4.247 |
| gm_queen5_5_3.wcsp | 809.718 ± 77.383 | 871.977 ± 100.859 | **586.717 ± 172.499** | 908.112 ± 481.036 | 1692.277 ± 213.626 |
| lm_batch_likelihood_brackets_4_4d | 20.902 ± 1.641 | 37.968 ± 19.358 | **14.773 ± 1.114** | 22.371 ± 4.231 | 106.506 ± 43.947 |
| lm_batch_likelihood_sentence_3_12d | **28.440 ± 2.699** | 50.954 ± 7.627 | 28.693 ± 2.596 | 38.358 ± 7.167 | 137.548 ± 33.834 |
| lm_batch_likelihood_sentence_4_4d | 18.087 ± 2.858 | 33.183 ± 6.287 | **14.539 ± 1.143** | 20.881 ± 3.736 | 122.017 ± 35.409 |
| nary_matmul_chain_64 | **0.019 ± 0.002** | 0.037 ± 0.026 | 0.024 ± 0.001 | 0.130 ± 0.095 | 0.084 ± 0.058 |
| str_matrix_chain_multiplication_100 | 5.764 ± 0.888 | 6.689 ± 0.841 | **4.972 ± 0.539** | 25.257 ± 3.287 | 307.549 ± 103.422 |
| str_mps_varying_inner_product_200 | 11.680 ± 0.726 | 13.742 ± 1.349 | **10.438 ± 0.578** | 30.758 ± 3.690 | 30.116 ± 15.740 |
| str_nw_mera_closed_120 | 361.855 ± 55.956 | 402.227 ± 40.888 | **291.019 ± 18.672** | 597.876 ± 87.566 | 838.648 ± 180.376 |
| str_nw_mera_open_26 | 357.949 ± 28.719 | 554.575 ± 92.816 | **243.407 ± 26.156** | 484.941 ± 120.087 | 795.653 ± 97.585 |
| tensornetwork_permutation_focus_step409_316 | **156.930 ± 6.423** | 300.368 ± 26.702 | 165.068 ± 31.090 | 258.073 ± 12.940 | 441.657 ± 295.966 |
| tensornetwork_permutation_light_415 | 152.064 ± 20.889 | 153.987 ± 8.466 | **141.843 ± 19.624** | 300.486 ± 48.634 | 444.765 ± 71.618 |

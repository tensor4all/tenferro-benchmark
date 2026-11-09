# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `ad9aa8a2fc06c36f8ce1d1dfb1ce9fb1cd9a9c46`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261109_172454/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261109_172454/run.yaml`
- Timestamp: `2026-11-09T17:24:54.736713+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261109_172454`.

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

- `data/results/mac-cpu/cpu/einsum/20261109_172454/jax_cpu_t1_20261109_172454.log`
- `data/results/mac-cpu/cpu/einsum/20261109_172454/julia_omeinsum_t1_20261109_172454.log`
- `data/results/mac-cpu/cpu/einsum/20261109_172454/pytorch_cpu_t1_20261109_172454.log`
- `data/results/mac-cpu/cpu/einsum/20261109_172454/tenferro_eager_t1_20261109_172454.log`
- `data/results/mac-cpu/cpu/einsum/20261109_172454/tenferro_trace_t1_20261109_172454.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.417 ± 0.124 | **0.684 ± 0.046** | 0.913 ± 0.334 | 4.430 ± 1.457 | 7.849 ± 6.049 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.160 ± 0.006 | **0.087 ± 0.004** | 0.555 ± 0.107 | 0.627 ± 0.033 | 0.889 ± 3.153 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.253 ± 0.008 | 0.241 ± 0.033 | **0.240 ± 0.025** | 0.409 ± 0.095 | 0.928 ± 0.823 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.983 ± 0.032 | 0.240 ± 0.046 | **0.232 ± 0.022** | 0.499 ± 0.104 | 1.558 ± 5.157 |
| bin_elementwise_mul_2048x2048 | **2.229 ± 0.176** | 3.920 ± 2.701 | 4.492 ± 2.178 | 4.267 ± 0.801 | 2.672 ± 0.460 |
| bin_elementwise_mul_512x512 | 0.086 ± 0.017 | 0.091 ± 0.004 | 0.108 ± 0.025 | 0.455 ± 0.176 | **0.079 ± 0.009** |
| bin_matmul_1024 | **16.023 ± 6.843** | 18.821 ± 9.455 | 23.560 ± 7.043 | 67.465 ± 10.385 | 67.454 ± 14.136 |
| bin_matmul_256 | 0.180 ± 0.020 | **0.129 ± 0.049** | 0.645 ± 0.417 | 1.186 ± 0.522 | 1.209 ± 1.737 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.004 ± 0.000** | 0.076 ± 0.013 | 0.098 ± 0.024 | 0.028 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.032 ± 0.007 | **0.026 ± 0.004** | 0.085 ± 0.012 | 0.201 ± 0.034 | 0.104 ± 0.033 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.025 | **0.007 ± 0.000** | 0.051 ± 0.018 | 0.184 ± 0.110 | 0.033 ± 0.009 |
| bin_outer_product_4096 | 5.586 ± 0.249 | 5.346 ± 0.351 | 5.208 ± 4.248 | **3.554 ± 0.293** | 254.642 ± 16.326 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 19.502 ± 7.338 | **12.833 ± 4.889** | 19.230 ± 4.673 | 82.764 ± 11.466 | 80.403 ± 16.061 |
| gm_queen5_5_3.wcsp | 11403.580 ± 1103.209 | 10522.909 ± 1879.419 | **5780.773 ± 1047.278** | 7073.831 ± 1216.354 | 9102.249 ± 1321.522 |
| lm_batch_likelihood_brackets_4_4d | 35.299 ± 18.153 | 21.421 ± 15.230 | 20.347 ± 12.141 | **17.489 ± 5.350** | 25.563 ± 19.987 |
| lm_batch_likelihood_sentence_3_12d | 38.771 ± 7.212 | 55.408 ± 8.782 | **37.629 ± 6.231** | 75.909 ± 8.293 | 84.748 ± 18.859 |
| lm_batch_likelihood_sentence_4_4d | 24.558 ± 5.283 | 35.496 ± 15.555 | 22.654 ± 3.594 | **16.285 ± 2.012** | 47.216 ± 24.991 |
| nary_matmul_chain_64 | **0.017 ± 0.001** | 0.030 ± 0.001 | 0.033 ± 0.015 | 0.113 ± 0.058 | 0.260 ± 0.283 |
| str_matrix_chain_multiplication_100 | **5.480 ± 1.666** | 6.444 ± 1.445 | 8.046 ± 1.871 | 14.749 ± 0.540 | 14.995 ± 1.932 |
| str_mps_varying_inner_product_200 | 11.461 ± 4.683 | **10.781 ± 1.878** | 13.236 ± 6.435 | 24.304 ± 8.595 | 21.098 ± 9.512 |
| str_nw_mera_closed_120 | 655.241 ± 121.717 | 642.814 ± 133.263 | **442.924 ± 47.931** | 1846.603 ± 253.717 | 1913.600 ± 623.731 |
| str_nw_mera_open_26 | 529.565 ± 105.946 | 565.220 ± 82.495 | **275.334 ± 33.925** | 1456.821 ± 413.528 | 1194.941 ± 96.249 |
| tensornetwork_permutation_focus_step409_316 | **241.486 ± 76.461** | 389.007 ± 98.012 | 316.110 ± 52.995 | 424.454 ± 94.492 | 253.133 ± 34.859 |
| tensornetwork_permutation_light_415 | 195.109 ± 31.976 | **160.872 ± 9.653** | 229.156 ± 16.583 | 424.551 ± 44.166 | 255.808 ± 27.275 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.417 ± 0.124 | **0.684 ± 0.046** | 0.913 ± 0.334 | 4.430 ± 1.457 | 7.849 ± 6.049 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.160 ± 0.006 | **0.087 ± 0.004** | 0.555 ± 0.107 | 0.627 ± 0.033 | 0.889 ± 3.153 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.253 ± 0.008 | 0.241 ± 0.033 | **0.240 ± 0.025** | 0.409 ± 0.095 | 0.928 ± 0.823 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.983 ± 0.032 | 0.240 ± 0.046 | **0.232 ± 0.022** | 0.499 ± 0.104 | 1.558 ± 5.157 |
| bin_elementwise_mul_2048x2048 | **2.229 ± 0.176** | 3.920 ± 2.701 | 4.492 ± 2.178 | 4.267 ± 0.801 | 2.672 ± 0.460 |
| bin_elementwise_mul_512x512 | 0.086 ± 0.017 | 0.091 ± 0.004 | 0.108 ± 0.025 | 0.455 ± 0.176 | **0.079 ± 0.009** |
| bin_matmul_1024 | **16.023 ± 6.843** | 18.821 ± 9.455 | 23.560 ± 7.043 | 67.465 ± 10.385 | 67.454 ± 14.136 |
| bin_matmul_256 | 0.180 ± 0.020 | **0.129 ± 0.049** | 0.645 ± 0.417 | 1.186 ± 0.522 | 1.209 ± 1.737 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.004 ± 0.000** | 0.076 ± 0.013 | 0.098 ± 0.024 | 0.028 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.032 ± 0.007 | **0.026 ± 0.004** | 0.085 ± 0.012 | 0.201 ± 0.034 | 0.104 ± 0.033 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.025 | **0.007 ± 0.000** | 0.051 ± 0.018 | 0.184 ± 0.110 | 0.033 ± 0.009 |
| bin_outer_product_4096 | 5.586 ± 0.249 | 5.346 ± 0.351 | 5.208 ± 4.248 | **3.554 ± 0.293** | 254.642 ± 16.326 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 19.502 ± 7.338 | **12.833 ± 4.889** | 19.230 ± 4.673 | 82.764 ± 11.466 | 80.403 ± 16.061 |
| gm_queen5_5_3.wcsp | 1436.748 ± 419.240 | 2052.333 ± 1119.030 | **1112.054 ± 220.860** | 1490.407 ± 224.969 | 2262.066 ± 211.205 |
| lm_batch_likelihood_brackets_4_4d | 21.134 ± 7.550 | 28.660 ± 29.830 | **18.318 ± 1.926** | 18.897 ± 2.134 | 49.817 ± 29.771 |
| lm_batch_likelihood_sentence_3_12d | 40.870 ± 8.179 | 49.182 ± 10.434 | **33.679 ± 4.006** | 57.632 ± 3.446 | 111.260 ± 31.359 |
| lm_batch_likelihood_sentence_4_4d | 20.958 ± 10.685 | 31.881 ± 19.175 | **17.417 ± 2.327** | 17.490 ± 2.453 | 42.641 ± 53.807 |
| nary_matmul_chain_64 | **0.017 ± 0.001** | 0.030 ± 0.001 | 0.033 ± 0.015 | 0.113 ± 0.058 | 0.260 ± 0.283 |
| str_matrix_chain_multiplication_100 | 9.562 ± 3.549 | 7.720 ± 7.077 | **5.005 ± 0.243** | 16.624 ± 1.341 | 24.520 ± 15.387 |
| str_mps_varying_inner_product_200 | 19.178 ± 6.898 | 11.842 ± 3.210 | **8.411 ± 0.901** | 24.227 ± 2.915 | 19.142 ± 8.668 |
| str_nw_mera_closed_120 | 630.689 ± 191.007 | 480.908 ± 107.187 | **316.356 ± 32.239** | 1678.825 ± 111.810 | 1467.170 ± 345.174 |
| str_nw_mera_open_26 | 524.481 ± 55.750 | 852.449 ± 499.737 | **317.778 ± 75.032** | 1355.657 ± 107.705 | 1360.926 ± 325.426 |
| tensornetwork_permutation_focus_step409_316 | **241.486 ± 76.461** | 389.007 ± 98.012 | 316.110 ± 52.995 | 424.454 ± 94.492 | 253.133 ± 34.859 |
| tensornetwork_permutation_light_415 | 195.109 ± 31.976 | **160.872 ± 9.653** | 229.156 ± 16.583 | 424.551 ± 44.166 | 255.808 ± 27.275 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261109_180327/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261109_180327/run.yaml`
- Timestamp: `2026-11-09T18:03:27.099105+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261109_180327`.

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

- `data/results/mac-cpu/cpu/einsum/20261109_180327/jax_cpu_t4_20261109_180327.log`
- `data/results/mac-cpu/cpu/einsum/20261109_180327/julia_omeinsum_t4_20261109_180327.log`
- `data/results/mac-cpu/cpu/einsum/20261109_180327/pytorch_cpu_t4_20261109_180327.log`
- `data/results/mac-cpu/cpu/einsum/20261109_180327/tenferro_eager_t4_20261109_180327.log`
- `data/results/mac-cpu/cpu/einsum/20261109_180327/tenferro_trace_t4_20261109_180327.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.497 ± 0.271 | 0.931 ± 0.835 | **0.706 ± 0.061** | 2.995 ± 0.362 | 96.129 ± 26.273 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.163 ± 0.008 | **0.120 ± 0.025** | 0.133 ± 0.007 | 0.798 ± 0.129 | 2.254 ± 1.877 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.168 ± 0.011** | 0.247 ± 0.067 | 0.194 ± 0.013 | 0.514 ± 0.094 | 0.994 ± 0.975 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.203 ± 0.005** | 0.217 ± 0.114 | **0.203 ± 0.022** | 0.525 ± 0.146 | 1.559 ± 0.609 |
| bin_elementwise_mul_2048x2048 | 2.722 ± 0.524 | 2.594 ± 0.319 | **2.487 ± 0.266** | 3.543 ± 0.518 | 3.639 ± 0.732 |
| bin_elementwise_mul_512x512 | **0.082 ± 0.060** | 0.171 ± 0.041 | 0.109 ± 0.042 | 0.595 ± 0.489 | 0.089 ± 0.024 |
| bin_matmul_1024 | 17.399 ± 2.997 | 13.434 ± 2.563 | **12.793 ± 1.194** | 22.134 ± 6.227 | 58.835 ± 24.870 |
| bin_matmul_256 | 0.169 ± 0.030 | **0.124 ± 0.023** | 0.133 ± 0.007 | 0.603 ± 0.105 | 3.719 ± 0.789 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | **0.008 ± 0.000** | 0.013 ± 0.000 | 0.019 ± 0.003 | 0.109 ± 0.017 | 0.038 ± 0.032 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.031 ± 0.003** | 0.046 ± 0.015 | 0.076 ± 0.007 | 0.212 ± 0.075 | 0.094 ± 0.008 |
| bin_omeinsum_matmul_10x10 | 0.018 ± 0.025 | 0.026 ± 0.007 | **0.017 ± 0.001** | 0.161 ± 0.172 | 0.019 ± 0.014 |
| bin_outer_product_4096 | 3.045 ± 0.416 | 3.134 ± 0.337 | 3.200 ± 0.350 | **2.931 ± 0.080** | 240.993 ± 26.173 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 14.784 ± 0.402 | 15.796 ± 2.079 | **10.771 ± 0.306** | 25.373 ± 2.130 | 62.203 ± 34.778 |
| gm_queen5_5_3.wcsp | 5931.955 ± 616.257 | 7140.668 ± 369.120 | **3265.906 ± 1912.345** | 5627.970 ± 1287.409 | 10332.686 ± 2128.992 |
| lm_batch_likelihood_brackets_4_4d | 25.242 ± 7.019 | 21.760 ± 13.511 | 18.836 ± 0.572 | **18.102 ± 1.011** | 83.383 ± 13.954 |
| lm_batch_likelihood_sentence_3_12d | **29.626 ± 3.201** | 45.992 ± 13.911 | 32.548 ± 1.320 | 42.155 ± 14.570 | 199.295 ± 58.234 |
| lm_batch_likelihood_sentence_4_4d | 26.709 ± 10.376 | 25.031 ± 8.481 | **21.239 ± 2.327** | 24.441 ± 2.381 | 121.841 ± 31.130 |
| nary_matmul_chain_64 | 0.055 ± 0.003 | **0.011 ± 0.001** | 0.029 ± 0.006 | 0.283 ± 0.127 | 0.100 ± 0.043 |
| str_matrix_chain_multiplication_100 | 7.644 ± 5.559 | **5.610 ± 0.170** | 7.972 ± 2.260 | 11.633 ± 4.687 | 263.890 ± 71.646 |
| str_mps_varying_inner_product_200 | 12.604 ± 7.686 | **11.573 ± 4.288** | 11.824 ± 1.339 | 25.862 ± 5.561 | 28.400 ± 27.207 |
| str_nw_mera_closed_120 | 453.486 ± 61.289 | 416.743 ± 67.807 | **410.240 ± 68.264** | 674.784 ± 35.752 | 1237.466 ± 563.053 |
| str_nw_mera_open_26 | 346.826 ± 29.678 | 436.409 ± 22.358 | **299.570 ± 36.770** | 554.304 ± 73.973 | 904.676 ± 131.990 |
| tensornetwork_permutation_focus_step409_316 | **156.621 ± 13.424** | 311.564 ± 31.509 | 214.052 ± 39.074 | 285.246 ± 37.281 | 485.917 ± 80.229 |
| tensornetwork_permutation_light_415 | **151.378 ± 15.546** | 158.383 ± 18.450 | 191.385 ± 37.905 | 275.162 ± 32.854 | 467.749 ± 73.252 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.497 ± 0.271 | 0.931 ± 0.835 | **0.706 ± 0.061** | 2.995 ± 0.362 | 96.129 ± 26.273 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.163 ± 0.008 | **0.120 ± 0.025** | 0.133 ± 0.007 | 0.798 ± 0.129 | 2.254 ± 1.877 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.168 ± 0.011** | 0.247 ± 0.067 | 0.194 ± 0.013 | 0.514 ± 0.094 | 0.994 ± 0.975 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.203 ± 0.005** | 0.217 ± 0.114 | **0.203 ± 0.022** | 0.525 ± 0.146 | 1.559 ± 0.609 |
| bin_elementwise_mul_2048x2048 | 2.722 ± 0.524 | 2.594 ± 0.319 | **2.487 ± 0.266** | 3.543 ± 0.518 | 3.639 ± 0.732 |
| bin_elementwise_mul_512x512 | **0.082 ± 0.060** | 0.171 ± 0.041 | 0.109 ± 0.042 | 0.595 ± 0.489 | 0.089 ± 0.024 |
| bin_matmul_1024 | 17.399 ± 2.997 | 13.434 ± 2.563 | **12.793 ± 1.194** | 22.134 ± 6.227 | 58.835 ± 24.870 |
| bin_matmul_256 | 0.169 ± 0.030 | **0.124 ± 0.023** | 0.133 ± 0.007 | 0.603 ± 0.105 | 3.719 ± 0.789 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | **0.008 ± 0.000** | 0.013 ± 0.000 | 0.019 ± 0.003 | 0.109 ± 0.017 | 0.038 ± 0.032 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.031 ± 0.003** | 0.046 ± 0.015 | 0.076 ± 0.007 | 0.212 ± 0.075 | 0.094 ± 0.008 |
| bin_omeinsum_matmul_10x10 | 0.018 ± 0.025 | 0.026 ± 0.007 | **0.017 ± 0.001** | 0.161 ± 0.172 | 0.019 ± 0.014 |
| bin_outer_product_4096 | 3.045 ± 0.416 | 3.134 ± 0.337 | 3.200 ± 0.350 | **2.931 ± 0.080** | 240.993 ± 26.173 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 14.784 ± 0.402 | 15.796 ± 2.079 | **10.771 ± 0.306** | 25.373 ± 2.130 | 62.203 ± 34.778 |
| gm_queen5_5_3.wcsp | 849.287 ± 66.954 | 829.083 ± 112.517 | **765.056 ± 125.663** | 1188.706 ± 275.808 | 1656.320 ± 242.537 |
| lm_batch_likelihood_brackets_4_4d | **19.174 ± 4.892** | 20.944 ± 14.665 | 25.007 ± 3.394 | 29.353 ± 11.316 | 120.482 ± 29.385 |
| lm_batch_likelihood_sentence_3_12d | **27.472 ± 2.223** | 42.634 ± 9.200 | 44.947 ± 10.510 | 52.750 ± 10.071 | 121.268 ± 32.908 |
| lm_batch_likelihood_sentence_4_4d | **17.866 ± 1.489** | 22.390 ± 1.346 | 29.407 ± 4.312 | 28.917 ± 4.899 | 127.174 ± 46.481 |
| nary_matmul_chain_64 | 0.055 ± 0.003 | **0.011 ± 0.001** | 0.029 ± 0.006 | 0.283 ± 0.127 | 0.100 ± 0.043 |
| str_matrix_chain_multiplication_100 | 5.314 ± 0.340 | **5.073 ± 0.508** | 8.562 ± 6.232 | 18.029 ± 5.787 | 258.800 ± 23.311 |
| str_mps_varying_inner_product_200 | 11.510 ± 1.314 | **10.672 ± 0.479** | 20.747 ± 4.838 | 39.279 ± 3.155 | 15.782 ± 26.271 |
| str_nw_mera_closed_120 | 331.551 ± 22.511 | 344.425 ± 31.387 | **327.014 ± 32.544** | 790.359 ± 155.331 | 933.688 ± 90.777 |
| str_nw_mera_open_26 | 363.944 ± 30.809 | 476.055 ± 30.567 | **248.203 ± 29.439** | 619.154 ± 184.182 | 1391.205 ± 394.969 |
| tensornetwork_permutation_focus_step409_316 | **156.621 ± 13.424** | 311.564 ± 31.509 | 214.052 ± 39.074 | 285.246 ± 37.281 | 485.917 ± 80.229 |
| tensornetwork_permutation_light_415 | **151.378 ± 15.546** | 158.383 ± 18.450 | 191.385 ± 37.905 | 275.162 ± 32.854 | 467.749 ± 73.252 |

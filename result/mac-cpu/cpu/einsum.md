# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `d8759f4320a337d2399f4a87dfec55af51d2ebf1`
- Benchmark commit: `64970fd383409685da2dbf874a4ad9dcdff4fec0`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260916_162342/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260916_162342/run.yaml`
- Timestamp: `20260916_162342`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260916_162342`.

- tenferro-rs commit: `d8759f4320a337d2399f4a87dfec55af51d2ebf1`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20260916_162342/einsum_table_t1_20260916_162342.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260916_162342/tenferro_trace_t1_20260916_162342.log`
- `data/results/mac-cpu/cpu/einsum/20260916_162342/tenferro_eager_t1_20260916_162342.log`
- `data/results/mac-cpu/cpu/einsum/20260916_162342/pytorch_cpu_t1_20260916_162342.log`
- `data/results/mac-cpu/cpu/einsum/20260916_162342/jax_cpu_t1_20260916_162342.log`
- `data/results/mac-cpu/cpu/einsum/20260916_162342/julia_omeinsum_t1_20260916_162342.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.795 ± 0.099 | 0.744 ± 0.142 | **0.388 ± 0.005** | 2.274 ± 0.073 | 3.944 ± 4.598 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.153 ± 0.003 | 0.126 ± 0.005 | **0.087 ± 0.003** | 0.347 ± 0.015 | 0.677 ± 0.119 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.077 ± 0.002** | 0.103 ± 0.005 | 0.118 ± 0.001 | 0.181 ± 0.011 | 0.308 ± 0.146 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.345 ± 0.007 | **0.099 ± 0.003** | 0.117 ± 0.002 | 0.203 ± 0.009 | 0.631 ± 0.812 |
| bin_elementwise_mul_2048x2048 | 0.876 ± 0.095 | 2.120 ± 0.167 | **0.841 ± 0.024** | 1.487 ± 0.039 | 0.921 ± 0.075 |
| bin_matmul_1024 | 8.405 ± 5.227 | 5.289 ± 0.421 | **5.184 ± 1.040** | 35.509 ± 0.314 | 40.721 ± 4.560 |
| bin_matmul_256 | 0.143 ± 0.005 | 0.100 ± 0.001 | **0.087 ± 0.001** | 0.621 ± 0.015 | 0.731 ± 0.029 |
| bin_outer_product_4096 | 1.183 ± 0.230 | **1.016 ± 0.028** | 1.488 ± 0.020 | 1.077 ± 0.033 | 101.152 ± 41.109 |
| gm_queen5_5_3.wcsp | 934.432 ± 27.219 | 1117.619 ± 63.508 | **928.000 ± 15.027** | 1341.182 ± 21.565 | 1409.980 ± 120.180 |
| lm_batch_likelihood_brackets_4_4d | 8.248 ± 0.074 | 14.240 ± 0.462 | 9.153 ± 0.273 | **7.540 ± 0.114** | 12.379 ± 1.465 |
| lm_batch_likelihood_sentence_3_12d | **10.437 ± 0.147** | 26.060 ± 0.737 | 18.395 ± 0.776 | 29.713 ± 0.158 | 85.796 ± 43.200 |
| lm_batch_likelihood_sentence_4_4d | 8.653 ± 0.058 | 15.840 ± 0.334 | 9.222 ± 0.159 | **8.565 ± 0.093** | 20.058 ± 48.114 |
| nary_matmul_chain_64 | 0.014 ± 0.001 | 0.017 ± 0.000 | **0.013 ± 0.000** | 0.058 ± 0.011 | 0.040 ± 0.009 |
| str_matrix_chain_multiplication_100 | **2.138 ± 0.029** | 2.822 ± 0.063 | 2.165 ± 0.053 | 8.524 ± 0.099 | 9.520 ± 0.163 |
| str_mps_varying_inner_product_200 | 5.556 ± 0.078 | 7.112 ± 0.089 | **3.602 ± 0.060** | 12.474 ± 0.089 | 11.010 ± 6.037 |
| str_nw_mera_closed_120 | 176.993 ± 12.800 | 189.582 ± 1.265 | **160.457 ± 1.215** | 847.810 ± 20.290 | 980.250 ± 18.827 |
| str_nw_mera_open_26 | 186.886 ± 0.602 | 201.661 ± 1.373 | **130.392 ± 1.171** | 626.349 ± 4.310 | 725.655 ± 48.887 |
| tensornetwork_permutation_focus_step409_316 | **122.752 ± 1.036** | 182.109 ± 1.292 | 157.218 ± 1.830 | 208.237 ± 10.429 | 185.343 ± 15.257 |
| tensornetwork_permutation_light_415 | **123.044 ± 0.970** | 155.961 ± 0.722 | 134.498 ± 0.908 | 211.771 ± 3.242 | 180.156 ± 2.053 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.795 ± 0.099 | 0.744 ± 0.142 | **0.388 ± 0.005** | 2.274 ± 0.073 | 3.944 ± 4.598 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.153 ± 0.003 | 0.126 ± 0.005 | **0.087 ± 0.003** | 0.347 ± 0.015 | 0.677 ± 0.119 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.077 ± 0.002** | 0.103 ± 0.005 | 0.118 ± 0.001 | 0.181 ± 0.011 | 0.308 ± 0.146 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.345 ± 0.007 | **0.099 ± 0.003** | 0.117 ± 0.002 | 0.203 ± 0.009 | 0.631 ± 0.812 |
| bin_elementwise_mul_2048x2048 | 0.876 ± 0.095 | 2.120 ± 0.167 | **0.841 ± 0.024** | 1.487 ± 0.039 | 0.921 ± 0.075 |
| bin_matmul_1024 | 8.405 ± 5.227 | 5.289 ± 0.421 | **5.184 ± 1.040** | 35.509 ± 0.314 | 40.721 ± 4.560 |
| bin_matmul_256 | 0.143 ± 0.005 | 0.100 ± 0.001 | **0.087 ± 0.001** | 0.621 ± 0.015 | 0.731 ± 0.029 |
| bin_outer_product_4096 | 1.183 ± 0.230 | **1.016 ± 0.028** | 1.488 ± 0.020 | 1.077 ± 0.033 | 101.152 ± 41.109 |
| gm_queen5_5_3.wcsp | **233.718 ± 0.515** | 405.281 ± 3.663 | 326.692 ± 6.358 | 641.161 ± 1.892 | 778.370 ± 4.301 |
| lm_batch_likelihood_brackets_4_4d | 9.355 ± 0.104 | 15.266 ± 0.290 | 9.009 ± 0.086 | **7.043 ± 0.096** | 19.625 ± 49.905 |
| lm_batch_likelihood_sentence_3_12d | **12.233 ± 0.261** | 28.815 ± 1.179 | 18.037 ± 0.445 | 30.214 ± 0.133 | 47.253 ± 44.158 |
| lm_batch_likelihood_sentence_4_4d | 9.548 ± 0.098 | 16.654 ± 0.180 | 8.920 ± 0.383 | **7.811 ± 0.062** | 19.373 ± 50.918 |
| nary_matmul_chain_64 | 0.014 ± 0.001 | 0.017 ± 0.000 | **0.013 ± 0.000** | 0.058 ± 0.011 | 0.040 ± 0.009 |
| str_matrix_chain_multiplication_100 | **2.135 ± 0.026** | 2.826 ± 0.059 | 2.162 ± 0.049 | 8.142 ± 0.109 | 9.482 ± 0.587 |
| str_mps_varying_inner_product_200 | 6.523 ± 0.094 | 7.533 ± 0.453 | **4.580 ± 0.140** | 13.967 ± 0.257 | 10.032 ± 1.005 |
| str_nw_mera_closed_120 | 151.838 ± 0.319 | 162.442 ± 2.054 | **132.272 ± 0.333** | 827.803 ± 8.095 | 965.654 ± 63.801 |
| str_nw_mera_open_26 | 189.963 ± 1.057 | 198.754 ± 0.739 | **135.343 ± 0.426** | 643.207 ± 2.001 | 680.232 ± 58.042 |
| tensornetwork_permutation_focus_step409_316 | **122.752 ± 1.036** | 182.109 ± 1.292 | 157.218 ± 1.830 | 208.237 ± 10.429 | 185.343 ± 15.257 |
| tensornetwork_permutation_light_415 | **123.044 ± 0.970** | 155.961 ± 0.722 | 134.498 ± 0.908 | 211.771 ± 3.242 | 180.156 ± 2.053 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260916_164245/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260916_164245/run.yaml`
- Timestamp: `20260916_164245`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260916_164245`.

- tenferro-rs commit: `d8759f4320a337d2399f4a87dfec55af51d2ebf1`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20260916_164245/einsum_table_t4_20260916_164245.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260916_164245/tenferro_trace_t4_20260916_164245.log`
- `data/results/mac-cpu/cpu/einsum/20260916_164245/tenferro_eager_t4_20260916_164245.log`
- `data/results/mac-cpu/cpu/einsum/20260916_164245/pytorch_cpu_t4_20260916_164245.log`
- `data/results/mac-cpu/cpu/einsum/20260916_164245/jax_cpu_t4_20260916_164245.log`
- `data/results/mac-cpu/cpu/einsum/20260916_164245/julia_omeinsum_t4_20260916_164245.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.875 ± 0.170 | 0.706 ± 0.125 | **0.382 ± 0.028** | 0.634 ± 0.073 | 2.232 ± 3.738 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.218 ± 0.065 | 0.150 ± 0.024 | **0.090 ± 0.001** | 0.165 ± 0.066 | 0.690 ± 0.267 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.087 ± 0.012 | 0.067 ± 0.017 | **0.054 ± 0.005** | 0.116 ± 0.025 | 0.263 ± 0.256 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.182 ± 0.035 | 0.082 ± 0.004 | **0.053 ± 0.002** | 0.127 ± 0.022 | 0.669 ± 0.597 |
| bin_elementwise_mul_2048x2048 | 0.629 ± 0.036 | 1.274 ± 0.101 | **0.585 ± 0.019** | 0.801 ± 0.059 | 0.965 ± 0.076 |
| bin_matmul_1024 | 5.641 ± 0.334 | 5.138 ± 0.423 | **5.094 ± 0.342** | 9.379 ± 0.080 | 16.722 ± 5.124 |
| bin_matmul_256 | 0.174 ± 0.017 | 0.116 ± 0.005 | **0.092 ± 0.001** | 0.225 ± 0.008 | 0.325 ± 0.071 |
| bin_outer_product_4096 | 0.604 ± 0.045 | **0.577 ± 0.030** | 0.589 ± 0.054 | 0.633 ± 0.048 | 72.159 ± 14.675 |
| gm_queen5_5_3.wcsp | 633.074 ± 163.041 | 600.616 ± 47.834 | **467.547 ± 7.650** | 707.645 ± 37.230 | 1078.723 ± 11.267 |
| lm_batch_likelihood_brackets_4_4d | **6.953 ± 0.197** | 12.280 ± 0.172 | 9.213 ± 0.708 | 7.110 ± 0.194 | 10.231 ± 0.358 |
| lm_batch_likelihood_sentence_3_12d | **9.817 ± 0.297** | 16.657 ± 0.900 | 20.317 ± 1.180 | 12.125 ± 0.180 | 31.080 ± 15.859 |
| lm_batch_likelihood_sentence_4_4d | 7.518 ± 0.208 | 13.221 ± 0.345 | 9.606 ± 0.497 | **7.260 ± 0.216** | 14.070 ± 14.260 |
| nary_matmul_chain_64 | 0.037 ± 0.021 | 0.036 ± 0.005 | **0.013 ± 0.001** | 0.056 ± 0.005 | 0.044 ± 0.004 |
| str_matrix_chain_multiplication_100 | 2.168 ± 0.081 | 3.831 ± 0.186 | **2.151 ± 0.110** | 8.092 ± 0.201 | 6.055 ± 0.171 |
| str_mps_varying_inner_product_200 | 5.839 ± 0.268 | 9.080 ± 0.133 | **4.510 ± 1.073** | 16.035 ± 0.202 | 10.500 ± 5.299 |
| str_nw_mera_closed_120 | 152.769 ± 3.447 | 157.995 ± 1.464 | **143.103 ± 1.675** | 254.884 ± 2.349 | 330.787 ± 39.602 |
| str_nw_mera_open_26 | 159.749 ± 12.948 | 155.877 ± 3.241 | **122.791 ± 4.379** | 188.289 ± 1.430 | 279.079 ± 20.928 |
| tensornetwork_permutation_focus_step409_316 | 87.808 ± 8.570 | 94.056 ± 10.793 | **75.843 ± 8.558** | 94.584 ± 2.408 | 104.109 ± 20.665 |
| tensornetwork_permutation_light_415 | **68.294 ± 20.645** | 85.954 ± 2.122 | 72.073 ± 7.629 | 98.265 ± 0.753 | 93.334 ± 17.391 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.875 ± 0.170 | 0.706 ± 0.125 | **0.382 ± 0.028** | 0.634 ± 0.073 | 2.232 ± 3.738 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.218 ± 0.065 | 0.150 ± 0.024 | **0.090 ± 0.001** | 0.165 ± 0.066 | 0.690 ± 0.267 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.087 ± 0.012 | 0.067 ± 0.017 | **0.054 ± 0.005** | 0.116 ± 0.025 | 0.263 ± 0.256 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.182 ± 0.035 | 0.082 ± 0.004 | **0.053 ± 0.002** | 0.127 ± 0.022 | 0.669 ± 0.597 |
| bin_elementwise_mul_2048x2048 | 0.629 ± 0.036 | 1.274 ± 0.101 | **0.585 ± 0.019** | 0.801 ± 0.059 | 0.965 ± 0.076 |
| bin_matmul_1024 | 5.641 ± 0.334 | 5.138 ± 0.423 | **5.094 ± 0.342** | 9.379 ± 0.080 | 16.722 ± 5.124 |
| bin_matmul_256 | 0.174 ± 0.017 | 0.116 ± 0.005 | **0.092 ± 0.001** | 0.225 ± 0.008 | 0.325 ± 0.071 |
| bin_outer_product_4096 | 0.604 ± 0.045 | **0.577 ± 0.030** | 0.589 ± 0.054 | 0.633 ± 0.048 | 72.159 ± 14.675 |
| gm_queen5_5_3.wcsp | **134.338 ± 5.013** | 214.945 ± 15.396 | 187.144 ± 5.739 | 267.048 ± 4.940 | 425.557 ± 5.731 |
| lm_batch_likelihood_brackets_4_4d | 7.465 ± 0.130 | 12.942 ± 0.434 | 10.251 ± 0.811 | **6.930 ± 0.242** | 15.653 ± 16.218 |
| lm_batch_likelihood_sentence_3_12d | **10.132 ± 0.404** | 16.998 ± 0.365 | 19.640 ± 1.069 | 12.495 ± 0.105 | 28.950 ± 12.092 |
| lm_batch_likelihood_sentence_4_4d | 7.404 ± 0.076 | 13.177 ± 0.201 | 10.053 ± 0.960 | **6.591 ± 0.314** | 14.431 ± 15.749 |
| nary_matmul_chain_64 | 0.037 ± 0.021 | 0.036 ± 0.005 | **0.013 ± 0.001** | 0.056 ± 0.005 | 0.044 ± 0.004 |
| str_matrix_chain_multiplication_100 | **2.136 ± 0.054** | 3.787 ± 0.160 | 2.204 ± 0.127 | 8.103 ± 0.188 | 6.044 ± 0.549 |
| str_mps_varying_inner_product_200 | **6.782 ± 0.167** | 9.811 ± 0.237 | 7.786 ± 0.322 | 17.212 ± 0.397 | 9.245 ± 5.040 |
| str_nw_mera_closed_120 | **130.161 ± 1.011** | 139.660 ± 1.493 | 130.794 ± 4.921 | 258.203 ± 8.322 | 305.344 ± 15.013 |
| str_nw_mera_open_26 | 146.538 ± 1.188 | 154.260 ± 1.431 | **125.758 ± 7.965** | 195.973 ± 1.735 | 281.050 ± 31.520 |
| tensornetwork_permutation_focus_step409_316 | 87.808 ± 8.570 | 94.056 ± 10.793 | **75.843 ± 8.558** | 94.584 ± 2.408 | 104.109 ± 20.665 |
| tensornetwork_permutation_light_415 | **68.294 ± 20.645** | 85.954 ± 2.122 | 72.073 ± 7.629 | 98.265 ± 0.753 | 93.334 ± 17.391 |

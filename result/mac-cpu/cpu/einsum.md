# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`
- Benchmark commit: `50a7dc9b3ea08f5dba14db07df1e09314cac56f0`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261024_172655/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261024_172655/run.yaml`
- Timestamp: `20261024_172655`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261024_172655`.

- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`

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
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`

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

- Julia: version `1.13.2`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.0.3.30.dylib)`, probe threads `1`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20261024_172655/einsum_table_t1_20261024_172655.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20261024_172655/tenferro_trace_t1_20261024_172655.log`
- `data/results/mac-cpu/cpu/einsum/20261024_172655/tenferro_eager_t1_20261024_172655.log`
- `data/results/mac-cpu/cpu/einsum/20261024_172655/pytorch_cpu_t1_20261024_172655.log`
- `data/results/mac-cpu/cpu/einsum/20261024_172655/jax_cpu_t1_20261024_172655.log`
- `data/results/mac-cpu/cpu/einsum/20261024_172655/julia_omeinsum_t1_20261024_172655.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.036 ± 0.212 | **0.511 ± 0.062** | 0.878 ± 0.274 | 3.980 ± 0.069 | 5.952 ± 14.915 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.129 ± 0.001 | **0.075 ± 0.006** | 0.333 ± 0.294 | 0.678 ± 0.035 | 1.974 ± 3.866 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.141 ± 0.008 | **0.121 ± 0.013** | 0.285 ± 0.507 | 0.566 ± 0.602 | 1.002 ± 1.376 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.393 ± 0.013 | **0.134 ± 0.017** | 0.234 ± 0.030 | 1.384 ± 0.449 | 2.941 ± 6.981 |
| bin_elementwise_mul_2048x2048 | **1.821 ± 0.266** | 2.563 ± 0.693 | 2.453 ± 0.369 | 3.836 ± 0.884 | 4.223 ± 2.733 |
| bin_elementwise_mul_512x512 | 0.103 ± 0.021 | **0.067 ± 0.000** | 0.088 ± 0.023 | 0.938 ± 0.248 | 0.087 ± 0.016 |
| bin_matmul_1024 | 10.972 ± 5.364 | **7.900 ± 0.305** | 14.167 ± 5.933 | 64.141 ± 12.082 | 99.287 ± 25.896 |
| bin_matmul_256 | 0.159 ± 0.010 | **0.102 ± 0.000** | 0.132 ± 0.034 | 1.010 ± 0.016 | 1.372 ± 0.478 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.002 | **0.004 ± 0.000** | 0.018 ± 0.006 | 0.094 ± 0.034 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.025 ± 0.002 | **0.022 ± 0.003** | 0.077 ± 0.005 | 0.145 ± 0.010 | 0.103 ± 0.016 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.006 ± 0.000** | 0.018 ± 0.008 | 0.137 ± 0.006 | 0.031 ± 0.010 |
| bin_outer_product_4096 | 2.307 ± 0.127 | **2.296 ± 0.141** | 3.236 ± 0.368 | 3.162 ± 0.080 | 321.404 ± 80.702 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 12.780 ± 1.472 | **9.186 ± 1.909** | 10.922 ± 1.603 | 57.825 ± 1.736 | 84.850 ± 16.718 |
| gm_queen5_5_3.wcsp | 7040.983 ± 567.631 | 6173.113 ± 1315.493 | **4424.669 ± 684.997** | 6163.893 ± 2597.695 | 7664.620 ± 545.706 |
| lm_batch_likelihood_brackets_4_4d | 15.299 ± 1.445 | 23.526 ± 4.676 | **14.007 ± 1.533** | 14.280 ± 2.432 | 59.834 ± 37.692 |
| lm_batch_likelihood_sentence_3_12d | **21.609 ± 1.939** | 44.650 ± 10.921 | 33.409 ± 11.219 | 54.715 ± 6.321 | 86.900 ± 13.462 |
| lm_batch_likelihood_sentence_4_4d | 14.678 ± 0.996 | 21.803 ± 6.708 | 18.849 ± 1.338 | **13.420 ± 0.558** | 43.440 ± 23.084 |
| nary_matmul_chain_64 | 0.019 ± 0.002 | **0.012 ± 0.006** | 0.024 ± 0.001 | 0.115 ± 0.006 | 0.063 ± 0.015 |
| str_matrix_chain_multiplication_100 | 5.153 ± 1.269 | **5.088 ± 0.398** | 5.634 ± 1.638 | 12.946 ± 1.513 | 14.208 ± 1.372 |
| str_mps_varying_inner_product_200 | 9.846 ± 0.582 | 10.268 ± 1.235 | **7.034 ± 0.413** | 19.744 ± 1.380 | 18.306 ± 6.665 |
| str_nw_mera_closed_120 | **341.746 ± 47.133** | 459.953 ± 127.072 | 375.888 ± 39.459 | 1631.376 ± 225.254 | 1319.004 ± 126.272 |
| str_nw_mera_open_26 | 251.723 ± 16.920 | 395.831 ± 39.763 | **244.499 ± 19.333** | 1344.987 ± 265.297 | 1007.722 ± 107.944 |
| tensornetwork_permutation_focus_step409_316 | **135.001 ± 16.918** | 306.832 ± 21.938 | 262.528 ± 22.524 | 428.815 ± 125.093 | 234.392 ± 38.848 |
| tensornetwork_permutation_light_415 | **115.370 ± 0.929** | 190.073 ± 36.912 | 227.411 ± 22.672 | 387.724 ± 73.810 | 202.364 ± 29.942 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.036 ± 0.212 | **0.511 ± 0.062** | 0.878 ± 0.274 | 3.980 ± 0.069 | 5.952 ± 14.915 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.129 ± 0.001 | **0.075 ± 0.006** | 0.333 ± 0.294 | 0.678 ± 0.035 | 1.974 ± 3.866 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.141 ± 0.008 | **0.121 ± 0.013** | 0.285 ± 0.507 | 0.566 ± 0.602 | 1.002 ± 1.376 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.393 ± 0.013 | **0.134 ± 0.017** | 0.234 ± 0.030 | 1.384 ± 0.449 | 2.941 ± 6.981 |
| bin_elementwise_mul_2048x2048 | **1.821 ± 0.266** | 2.563 ± 0.693 | 2.453 ± 0.369 | 3.836 ± 0.884 | 4.223 ± 2.733 |
| bin_elementwise_mul_512x512 | 0.103 ± 0.021 | **0.067 ± 0.000** | 0.088 ± 0.023 | 0.938 ± 0.248 | 0.087 ± 0.016 |
| bin_matmul_1024 | 10.972 ± 5.364 | **7.900 ± 0.305** | 14.167 ± 5.933 | 64.141 ± 12.082 | 99.287 ± 25.896 |
| bin_matmul_256 | 0.159 ± 0.010 | **0.102 ± 0.000** | 0.132 ± 0.034 | 1.010 ± 0.016 | 1.372 ± 0.478 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.002 | **0.004 ± 0.000** | 0.018 ± 0.006 | 0.094 ± 0.034 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.025 ± 0.002 | **0.022 ± 0.003** | 0.077 ± 0.005 | 0.145 ± 0.010 | 0.103 ± 0.016 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.006 ± 0.000** | 0.018 ± 0.008 | 0.137 ± 0.006 | 0.031 ± 0.010 |
| bin_outer_product_4096 | 2.307 ± 0.127 | **2.296 ± 0.141** | 3.236 ± 0.368 | 3.162 ± 0.080 | 321.404 ± 80.702 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 12.780 ± 1.472 | **9.186 ± 1.909** | 10.922 ± 1.603 | 57.825 ± 1.736 | 84.850 ± 16.718 |
| gm_queen5_5_3.wcsp | **766.144 ± 86.158** | 1064.341 ± 593.454 | 963.856 ± 109.632 | 1526.254 ± 266.290 | 1701.261 ± 245.810 |
| lm_batch_likelihood_brackets_4_4d | **14.084 ± 0.352** | 36.085 ± 21.779 | 15.641 ± 1.011 | 15.145 ± 2.101 | 42.139 ± 27.226 |
| lm_batch_likelihood_sentence_3_12d | **24.252 ± 0.337** | 54.842 ± 11.659 | 27.195 ± 1.196 | 61.486 ± 5.889 | 89.234 ± 30.385 |
| lm_batch_likelihood_sentence_4_4d | **10.237 ± 0.251** | 33.278 ± 11.721 | 15.668 ± 0.313 | 18.013 ± 1.007 | 39.272 ± 24.147 |
| nary_matmul_chain_64 | 0.019 ± 0.002 | **0.012 ± 0.006** | 0.024 ± 0.001 | 0.115 ± 0.006 | 0.063 ± 0.015 |
| str_matrix_chain_multiplication_100 | **4.135 ± 0.096** | 7.152 ± 1.962 | 5.396 ± 0.134 | 16.020 ± 3.040 | 14.592 ± 8.909 |
| str_mps_varying_inner_product_200 | 9.866 ± 0.286 | 14.825 ± 4.748 | **8.933 ± 0.213** | 29.003 ± 10.446 | 16.414 ± 3.491 |
| str_nw_mera_closed_120 | **273.651 ± 3.162** | 434.921 ± 223.551 | 293.652 ± 30.944 | 1672.208 ± 148.518 | 1296.088 ± 154.348 |
| str_nw_mera_open_26 | 266.607 ± 33.455 | 473.372 ± 96.947 | **248.753 ± 5.683** | 1387.917 ± 360.673 | 1152.803 ± 221.702 |
| tensornetwork_permutation_focus_step409_316 | **135.001 ± 16.918** | 306.832 ± 21.938 | 262.528 ± 22.524 | 428.815 ± 125.093 | 234.392 ± 38.848 |
| tensornetwork_permutation_light_415 | **115.370 ± 0.929** | 190.073 ± 36.912 | 227.411 ± 22.672 | 387.724 ± 73.810 | 202.364 ± 29.942 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261024_175546/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261024_175546/run.yaml`
- Timestamp: `20261024_175546`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261024_175546`.

- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`

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
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`

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

- Julia: version `1.13.2`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.0.3.30.dylib)`, probe threads `4`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20261024_175546/einsum_table_t4_20261024_175546.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20261024_175546/tenferro_trace_t4_20261024_175546.log`
- `data/results/mac-cpu/cpu/einsum/20261024_175546/tenferro_eager_t4_20261024_175546.log`
- `data/results/mac-cpu/cpu/einsum/20261024_175546/pytorch_cpu_t4_20261024_175546.log`
- `data/results/mac-cpu/cpu/einsum/20261024_175546/jax_cpu_t4_20261024_175546.log`
- `data/results/mac-cpu/cpu/einsum/20261024_175546/julia_omeinsum_t4_20261024_175546.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.261 ± 0.193 | **0.587 ± 0.099** | 0.599 ± 0.013 | 1.446 ± 0.152 | 90.290 ± 10.447 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.221 ± 0.127 | **0.076 ± 0.003** | 0.115 ± 0.003 | 0.677 ± 0.057 | 1.638 ± 5.470 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.174 ± 0.026 | 0.208 ± 0.056 | **0.163 ± 0.002** | 0.376 ± 0.075 | 0.774 ± 0.363 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.241 ± 0.040 | 0.176 ± 0.014 | **0.167 ± 0.001** | 0.359 ± 0.008 | 1.206 ± 0.401 |
| bin_elementwise_mul_2048x2048 | 2.295 ± 0.220 | 2.522 ± 0.255 | **2.053 ± 0.200** | 2.647 ± 0.032 | 2.724 ± 0.247 |
| bin_elementwise_mul_512x512 | 0.078 ± 0.015 | 0.163 ± 0.048 | **0.074 ± 0.004** | 0.265 ± 0.019 | 0.090 ± 0.254 |
| bin_matmul_1024 | 12.926 ± 0.987 | **9.756 ± 0.632** | 13.557 ± 4.314 | 17.681 ± 1.347 | 43.649 ± 24.629 |
| bin_matmul_256 | 0.406 ± 0.196 | **0.110 ± 0.000** | **0.110 ± 0.001** | 0.506 ± 0.113 | 3.629 ± 1.783 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.021 ± 0.001 | **0.004 ± 0.000** | 0.018 ± 0.001 | 0.089 ± 0.007 | 0.011 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.002 | **0.025 ± 0.001** | 0.075 ± 0.001 | 0.166 ± 0.017 | 0.269 ± 0.178 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.002 | **0.007 ± 0.000** | 0.016 ± 0.002 | 0.077 ± 0.014 | 0.042 ± 0.023 |
| bin_outer_product_4096 | 2.532 ± 0.204 | 3.693 ± 1.188 | 2.309 ± 0.223 | **2.220 ± 0.038** | 253.249 ± 55.059 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 14.619 ± 0.270 | 14.786 ± 4.772 | **9.289 ± 0.301** | 19.340 ± 0.267 | 60.786 ± 22.608 |
| gm_queen5_5_3.wcsp | 4102.023 ± 2350.319 | 6490.998 ± 3201.175 | **2465.787 ± 178.167** | 3646.392 ± 2203.541 | 9327.017 ± 1325.464 |
| lm_batch_likelihood_brackets_4_4d | 16.781 ± 3.226 | 29.862 ± 23.415 | **13.822 ± 4.890** | 33.643 ± 3.612 | 99.292 ± 21.920 |
| lm_batch_likelihood_sentence_3_12d | 31.634 ± 12.631 | 34.534 ± 10.445 | **26.072 ± 3.384** | 39.659 ± 7.064 | 134.555 ± 46.036 |
| lm_batch_likelihood_sentence_4_4d | 19.135 ± 4.429 | 22.171 ± 2.632 | **18.731 ± 3.198** | 34.493 ± 6.439 | 101.990 ± 34.141 |
| nary_matmul_chain_64 | 0.020 ± 0.001 | **0.010 ± 0.001** | 0.023 ± 0.002 | 0.432 ± 0.441 | 0.124 ± 0.020 |
| str_matrix_chain_multiplication_100 | 6.203 ± 1.639 | **4.827 ± 0.976** | 5.418 ± 0.994 | 19.574 ± 3.573 | 270.673 ± 59.923 |
| str_mps_varying_inner_product_200 | 16.778 ± 4.128 | **9.938 ± 2.592** | 12.146 ± 3.754 | 39.814 ± 6.244 | 27.223 ± 22.192 |
| str_nw_mera_closed_120 | 524.989 ± 76.733 | 377.523 ± 50.600 | **356.382 ± 61.293** | 825.587 ± 267.663 | 983.940 ± 518.334 |
| str_nw_mera_open_26 | 415.748 ± 42.024 | 358.106 ± 50.214 | **221.879 ± 46.346** | 451.726 ± 148.128 | 710.984 ± 125.347 |
| tensornetwork_permutation_focus_step409_316 | **131.052 ± 20.816** | 159.702 ± 5.328 | 157.732 ± 28.826 | 235.131 ± 80.109 | 443.212 ± 68.644 |
| tensornetwork_permutation_light_415 | 116.606 ± 7.623 | **92.409 ± 7.413** | 147.369 ± 48.400 | 258.670 ± 39.730 | 434.440 ± 41.837 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.261 ± 0.193 | **0.587 ± 0.099** | 0.599 ± 0.013 | 1.446 ± 0.152 | 90.290 ± 10.447 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.221 ± 0.127 | **0.076 ± 0.003** | 0.115 ± 0.003 | 0.677 ± 0.057 | 1.638 ± 5.470 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.174 ± 0.026 | 0.208 ± 0.056 | **0.163 ± 0.002** | 0.376 ± 0.075 | 0.774 ± 0.363 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.241 ± 0.040 | 0.176 ± 0.014 | **0.167 ± 0.001** | 0.359 ± 0.008 | 1.206 ± 0.401 |
| bin_elementwise_mul_2048x2048 | 2.295 ± 0.220 | 2.522 ± 0.255 | **2.053 ± 0.200** | 2.647 ± 0.032 | 2.724 ± 0.247 |
| bin_elementwise_mul_512x512 | 0.078 ± 0.015 | 0.163 ± 0.048 | **0.074 ± 0.004** | 0.265 ± 0.019 | 0.090 ± 0.254 |
| bin_matmul_1024 | 12.926 ± 0.987 | **9.756 ± 0.632** | 13.557 ± 4.314 | 17.681 ± 1.347 | 43.649 ± 24.629 |
| bin_matmul_256 | 0.406 ± 0.196 | **0.110 ± 0.000** | **0.110 ± 0.001** | 0.506 ± 0.113 | 3.629 ± 1.783 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.021 ± 0.001 | **0.004 ± 0.000** | 0.018 ± 0.001 | 0.089 ± 0.007 | 0.011 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.002 | **0.025 ± 0.001** | 0.075 ± 0.001 | 0.166 ± 0.017 | 0.269 ± 0.178 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.002 | **0.007 ± 0.000** | 0.016 ± 0.002 | 0.077 ± 0.014 | 0.042 ± 0.023 |
| bin_outer_product_4096 | 2.532 ± 0.204 | 3.693 ± 1.188 | 2.309 ± 0.223 | **2.220 ± 0.038** | 253.249 ± 55.059 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 14.619 ± 0.270 | 14.786 ± 4.772 | **9.289 ± 0.301** | 19.340 ± 0.267 | 60.786 ± 22.608 |
| gm_queen5_5_3.wcsp | 622.852 ± 93.032 | **491.237 ± 33.095** | 511.179 ± 113.792 | 909.502 ± 226.694 | 1506.037 ± 241.420 |
| lm_batch_likelihood_brackets_4_4d | **12.322 ± 2.779** | 15.909 ± 10.196 | 13.716 ± 0.462 | 14.240 ± 0.651 | 104.225 ± 37.710 |
| lm_batch_likelihood_sentence_3_12d | **22.984 ± 1.916** | 30.608 ± 9.918 | 31.271 ± 3.097 | 33.988 ± 7.264 | 143.755 ± 44.161 |
| lm_batch_likelihood_sentence_4_4d | **10.438 ± 1.930** | 20.694 ± 4.407 | 16.294 ± 1.364 | 21.137 ± 0.799 | 95.151 ± 22.163 |
| nary_matmul_chain_64 | 0.020 ± 0.001 | **0.010 ± 0.001** | 0.023 ± 0.002 | 0.432 ± 0.441 | 0.124 ± 0.020 |
| str_matrix_chain_multiplication_100 | **4.431 ± 0.756** | 5.038 ± 0.126 | 5.756 ± 0.527 | 18.777 ± 0.752 | 272.260 ± 58.740 |
| str_mps_varying_inner_product_200 | 12.099 ± 3.073 | 11.964 ± 1.239 | **11.787 ± 0.877** | 34.754 ± 8.892 | 20.650 ± 19.858 |
| str_nw_mera_closed_120 | 348.494 ± 56.925 | 296.734 ± 26.094 | **254.810 ± 18.554** | 586.587 ± 57.719 | 626.139 ± 59.051 |
| str_nw_mera_open_26 | 293.047 ± 52.420 | 370.381 ± 41.604 | **210.659 ± 20.523** | 496.204 ± 63.279 | 681.617 ± 121.599 |
| tensornetwork_permutation_focus_step409_316 | **131.052 ± 20.816** | 159.702 ± 5.328 | 157.732 ± 28.826 | 235.131 ± 80.109 | 443.212 ± 68.644 |
| tensornetwork_permutation_light_415 | 116.606 ± 7.623 | **92.409 ± 7.413** | 147.369 ± 48.400 | 258.670 ± 39.730 | 434.440 ± 41.837 |

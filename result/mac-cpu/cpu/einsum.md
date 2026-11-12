# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `4edf89377c13d5c8a287dd562b529de6bb85f4dc`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261112_172325/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261112_172325/run.yaml`
- Timestamp: `2026-11-12T17:23:25.841634+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261112_172325`.

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

- `data/results/mac-cpu/cpu/einsum/20261112_172325/jax_cpu_t1_20261112_172325.log`
- `data/results/mac-cpu/cpu/einsum/20261112_172325/julia_omeinsum_t1_20261112_172325.log`
- `data/results/mac-cpu/cpu/einsum/20261112_172325/pytorch_cpu_t1_20261112_172325.log`
- `data/results/mac-cpu/cpu/einsum/20261112_172325/tenferro_eager_t1_20261112_172325.log`
- `data/results/mac-cpu/cpu/einsum/20261112_172325/tenferro_trace_t1_20261112_172325.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.644 ± 1.223 | 0.878 ± 0.479 | **0.824 ± 0.096** | 4.536 ± 0.547 | 7.238 ± 5.281 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.152 ± 0.007 | **0.088 ± 0.006** | 0.134 ± 0.004 | 1.961 ± 0.753 | 1.130 ± 2.332 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.432 ± 0.426 | 0.248 ± 0.021 | **0.222 ± 0.128** | 0.859 ± 0.596 | 0.771 ± 0.352 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.116 ± 0.878 | 0.505 ± 0.287 | **0.242 ± 0.235** | 1.507 ± 0.299 | 1.386 ± 0.704 |
| bin_elementwise_mul_2048x2048 | 4.704 ± 1.568 | 2.855 ± 2.423 | 3.139 ± 1.695 | 3.711 ± 2.270 | **2.776 ± 0.381** |
| bin_elementwise_mul_512x512 | 0.087 ± 0.002 | **0.060 ± 0.010** | 0.099 ± 0.013 | 0.529 ± 0.063 | 0.069 ± 0.014 |
| bin_matmul_1024 | 24.469 ± 12.413 | **14.621 ± 7.589** | 27.602 ± 11.056 | 62.624 ± 12.824 | 59.440 ± 19.098 |
| bin_matmul_256 | 0.157 ± 0.004 | **0.119 ± 0.005** | 0.152 ± 0.023 | 0.967 ± 0.026 | 0.970 ± 0.038 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.006 ± 0.000 | **0.004 ± 0.000** | 0.018 ± 0.001 | 0.257 ± 0.005 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.029 ± 0.002** | 0.059 ± 0.042 | 0.078 ± 0.009 | 0.199 ± 0.024 | 0.095 ± 0.021 |
| bin_omeinsum_matmul_10x10 | 0.014 ± 0.001 | **0.007 ± 0.000** | 0.017 ± 0.001 | 0.152 ± 0.008 | 0.018 ± 0.001 |
| bin_outer_product_4096 | 8.301 ± 4.939 | 6.749 ± 3.211 | 3.596 ± 1.151 | **2.961 ± 0.311** | 279.167 ± 39.578 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 28.176 ± 14.206 | **17.097 ± 5.216** | 17.616 ± 3.410 | 58.207 ± 4.721 | 54.752 ± 9.894 |
| gm_queen5_5_3.wcsp | 11589.903 ± 2271.649 | 8563.856 ± 3770.474 | **5467.107 ± 3239.768** | 7374.628 ± 583.569 | 8765.751 ± 1191.682 |
| lm_batch_likelihood_brackets_4_4d | 29.418 ± 12.083 | 35.603 ± 14.217 | **16.183 ± 3.038** | 17.000 ± 7.484 | 60.870 ± 50.399 |
| lm_batch_likelihood_sentence_3_12d | 38.518 ± 8.083 | 64.053 ± 5.178 | **35.190 ± 11.309** | 68.398 ± 6.427 | 97.102 ± 13.602 |
| lm_batch_likelihood_sentence_4_4d | 25.787 ± 5.736 | 34.080 ± 6.982 | **18.631 ± 2.031** | 18.658 ± 4.246 | 53.008 ± 32.079 |
| nary_matmul_chain_64 | **0.017 ± 0.001** | 0.030 ± 0.019 | 0.024 ± 0.001 | 0.195 ± 0.078 | 0.100 ± 0.040 |
| str_matrix_chain_multiplication_100 | **6.677 ± 5.080** | 7.596 ± 4.210 | 6.722 ± 0.848 | 17.109 ± 4.169 | 19.193 ± 9.017 |
| str_mps_varying_inner_product_200 | 15.528 ± 6.871 | 16.437 ± 4.747 | **8.121 ± 0.492** | 19.578 ± 1.936 | 26.352 ± 31.188 |
| str_nw_mera_closed_120 | 582.961 ± 109.115 | 954.287 ± 340.711 | **371.330 ± 33.569** | 1916.854 ± 370.264 | 1744.649 ± 354.292 |
| str_nw_mera_open_26 | 505.538 ± 113.452 | 910.374 ± 178.827 | **224.524 ± 42.924** | 1451.468 ± 265.894 | 1010.488 ± 174.832 |
| tensornetwork_permutation_focus_step409_316 | **196.474 ± 28.249** | 381.599 ± 61.709 | 282.262 ± 29.255 | 359.805 ± 49.132 | 293.472 ± 41.810 |
| tensornetwork_permutation_light_415 | 203.468 ± 29.974 | **183.202 ± 48.024** | 245.989 ± 37.861 | 364.497 ± 33.571 | 334.653 ± 99.203 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.644 ± 1.223 | 0.878 ± 0.479 | **0.824 ± 0.096** | 4.536 ± 0.547 | 7.238 ± 5.281 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.152 ± 0.007 | **0.088 ± 0.006** | 0.134 ± 0.004 | 1.961 ± 0.753 | 1.130 ± 2.332 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.432 ± 0.426 | 0.248 ± 0.021 | **0.222 ± 0.128** | 0.859 ± 0.596 | 0.771 ± 0.352 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.116 ± 0.878 | 0.505 ± 0.287 | **0.242 ± 0.235** | 1.507 ± 0.299 | 1.386 ± 0.704 |
| bin_elementwise_mul_2048x2048 | 4.704 ± 1.568 | 2.855 ± 2.423 | 3.139 ± 1.695 | 3.711 ± 2.270 | **2.776 ± 0.381** |
| bin_elementwise_mul_512x512 | 0.087 ± 0.002 | **0.060 ± 0.010** | 0.099 ± 0.013 | 0.529 ± 0.063 | 0.069 ± 0.014 |
| bin_matmul_1024 | 24.469 ± 12.413 | **14.621 ± 7.589** | 27.602 ± 11.056 | 62.624 ± 12.824 | 59.440 ± 19.098 |
| bin_matmul_256 | 0.157 ± 0.004 | **0.119 ± 0.005** | 0.152 ± 0.023 | 0.967 ± 0.026 | 0.970 ± 0.038 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.006 ± 0.000 | **0.004 ± 0.000** | 0.018 ± 0.001 | 0.257 ± 0.005 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.029 ± 0.002** | 0.059 ± 0.042 | 0.078 ± 0.009 | 0.199 ± 0.024 | 0.095 ± 0.021 |
| bin_omeinsum_matmul_10x10 | 0.014 ± 0.001 | **0.007 ± 0.000** | 0.017 ± 0.001 | 0.152 ± 0.008 | 0.018 ± 0.001 |
| bin_outer_product_4096 | 8.301 ± 4.939 | 6.749 ± 3.211 | 3.596 ± 1.151 | **2.961 ± 0.311** | 279.167 ± 39.578 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 28.176 ± 14.206 | **17.097 ± 5.216** | 17.616 ± 3.410 | 58.207 ± 4.721 | 54.752 ± 9.894 |
| gm_queen5_5_3.wcsp | 1281.438 ± 214.221 | 2472.425 ± 305.409 | **1010.426 ± 130.220** | 1293.727 ± 593.081 | 2022.340 ± 414.803 |
| lm_batch_likelihood_brackets_4_4d | 25.572 ± 13.154 | 38.094 ± 14.057 | 15.764 ± 2.595 | **13.454 ± 0.747** | 37.661 ± 26.994 |
| lm_batch_likelihood_sentence_3_12d | 39.610 ± 5.111 | 58.937 ± 9.170 | **34.261 ± 10.427** | 55.925 ± 1.686 | 96.936 ± 27.389 |
| lm_batch_likelihood_sentence_4_4d | 17.605 ± 5.903 | 32.062 ± 7.151 | 16.236 ± 1.254 | **13.688 ± 0.687** | 36.929 ± 29.923 |
| nary_matmul_chain_64 | **0.017 ± 0.001** | 0.030 ± 0.019 | 0.024 ± 0.001 | 0.195 ± 0.078 | 0.100 ± 0.040 |
| str_matrix_chain_multiplication_100 | **5.272 ± 2.268** | 9.473 ± 1.613 | 6.349 ± 3.440 | 16.634 ± 2.515 | 16.855 ± 4.648 |
| str_mps_varying_inner_product_200 | 15.506 ± 4.159 | 18.345 ± 7.479 | **9.157 ± 0.667** | 20.999 ± 0.865 | 20.626 ± 10.655 |
| str_nw_mera_closed_120 | 498.293 ± 45.967 | 342.803 ± 56.960 | **334.762 ± 78.122** | 1636.384 ± 103.616 | 1491.635 ± 252.425 |
| str_nw_mera_open_26 | 550.824 ± 178.900 | 958.719 ± 380.140 | **302.765 ± 38.590** | 1251.502 ± 115.069 | 1613.591 ± 395.224 |
| tensornetwork_permutation_focus_step409_316 | **196.474 ± 28.249** | 381.599 ± 61.709 | 282.262 ± 29.255 | 359.805 ± 49.132 | 293.472 ± 41.810 |
| tensornetwork_permutation_light_415 | 203.468 ± 29.974 | **183.202 ± 48.024** | 245.989 ± 37.861 | 364.497 ± 33.571 | 334.653 ± 99.203 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261112_175944/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261112_175944/run.yaml`
- Timestamp: `2026-11-12T17:59:44.081935+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261112_175944`.

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

- `data/results/mac-cpu/cpu/einsum/20261112_175944/jax_cpu_t4_20261112_175944.log`
- `data/results/mac-cpu/cpu/einsum/20261112_175944/julia_omeinsum_t4_20261112_175944.log`
- `data/results/mac-cpu/cpu/einsum/20261112_175944/pytorch_cpu_t4_20261112_175944.log`
- `data/results/mac-cpu/cpu/einsum/20261112_175944/tenferro_eager_t4_20261112_175944.log`
- `data/results/mac-cpu/cpu/einsum/20261112_175944/tenferro_trace_t4_20261112_175944.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.345 ± 0.155 | **0.667 ± 0.045** | 0.850 ± 0.183 | 2.842 ± 0.385 | 92.864 ± 29.344 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.192 ± 0.031 | **0.083 ± 0.023** | 0.133 ± 0.007 | 0.798 ± 0.135 | 1.328 ± 3.600 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.177 ± 0.044 | **0.170 ± 0.064** | 0.243 ± 0.054 | 0.701 ± 0.250 | 0.937 ± 0.641 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.325 ± 0.080 | **0.164 ± 0.022** | 0.263 ± 0.135 | 0.766 ± 0.145 | 1.421 ± 3.640 |
| bin_elementwise_mul_2048x2048 | **2.741 ± 0.187** | 2.916 ± 0.381 | 3.486 ± 2.037 | 3.588 ± 0.345 | 3.309 ± 0.663 |
| bin_elementwise_mul_512x512 | **0.084 ± 0.039** | 0.120 ± 0.096 | 0.151 ± 0.183 | 0.393 ± 0.069 | 0.089 ± 0.317 |
| bin_matmul_1024 | 15.421 ± 0.580 | **12.038 ± 2.152** | 14.690 ± 5.003 | 26.604 ± 2.822 | 41.548 ± 39.834 |
| bin_matmul_256 | 0.181 ± 0.025 | **0.127 ± 0.007** | 0.131 ± 0.004 | 1.292 ± 0.400 | 3.353 ± 2.698 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.005 ± 0.002** | 0.060 ± 0.004 | 0.134 ± 0.070 | 0.012 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.003 | **0.027 ± 0.004** | 0.202 ± 0.013 | 0.357 ± 0.215 | 0.099 ± 0.017 |
| bin_omeinsum_matmul_10x10 | **0.017 ± 0.001** | 0.020 ± 0.000 | 0.049 ± 0.003 | 0.230 ± 0.171 | 0.029 ± 0.014 |
| bin_outer_product_4096 | 4.775 ± 1.805 | 3.303 ± 1.325 | **3.281 ± 1.006** | 4.119 ± 1.753 | 240.837 ± 30.186 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 16.053 ± 1.493 | 17.357 ± 4.245 | **11.482 ± 6.030** | 44.147 ± 7.663 | 30.137 ± 8.777 |
| gm_queen5_5_3.wcsp | 4889.256 ± 1385.425 | 6857.515 ± 483.750 | **2685.311 ± 229.306** | 5549.423 ± 861.035 | 10489.708 ± 873.977 |
| lm_batch_likelihood_brackets_4_4d | 18.451 ± 0.918 | 33.127 ± 8.264 | **16.164 ± 3.383** | 26.272 ± 5.130 | 138.901 ± 90.550 |
| lm_batch_likelihood_sentence_3_12d | **19.819 ± 4.059** | 50.809 ± 11.349 | 30.214 ± 7.881 | 39.011 ± 1.052 | 142.048 ± 34.091 |
| lm_batch_likelihood_sentence_4_4d | 17.924 ± 5.578 | 21.807 ± 2.363 | **17.241 ± 4.827** | 19.648 ± 5.683 | 114.539 ± 26.905 |
| nary_matmul_chain_64 | **0.019 ± 0.002** | 0.036 ± 0.001 | 0.024 ± 0.001 | 0.136 ± 0.033 | 0.071 ± 0.036 |
| str_matrix_chain_multiplication_100 | 5.377 ± 0.619 | **5.035 ± 0.416** | 5.868 ± 1.964 | 14.181 ± 6.617 | 290.558 ± 121.564 |
| str_mps_varying_inner_product_200 | **8.880 ± 0.780** | 9.030 ± 0.523 | 14.126 ± 8.236 | 22.067 ± 14.399 | 27.403 ± 25.707 |
| str_nw_mera_closed_120 | 404.664 ± 62.872 | 408.268 ± 32.323 | **382.960 ± 38.855** | 653.881 ± 55.713 | 977.255 ± 153.545 |
| str_nw_mera_open_26 | 297.012 ± 20.815 | 421.500 ± 99.515 | **243.579 ± 26.656** | 533.917 ± 60.607 | 1267.441 ± 214.408 |
| tensornetwork_permutation_focus_step409_316 | **140.114 ± 8.085** | 279.380 ± 35.478 | 191.619 ± 39.776 | 267.643 ± 13.601 | 470.218 ± 125.467 |
| tensornetwork_permutation_light_415 | **127.693 ± 0.987** | 136.175 ± 15.458 | 163.259 ± 60.135 | 297.358 ± 44.433 | 439.257 ± 53.649 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.345 ± 0.155 | **0.667 ± 0.045** | 0.850 ± 0.183 | 2.842 ± 0.385 | 92.864 ± 29.344 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.192 ± 0.031 | **0.083 ± 0.023** | 0.133 ± 0.007 | 0.798 ± 0.135 | 1.328 ± 3.600 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.177 ± 0.044 | **0.170 ± 0.064** | 0.243 ± 0.054 | 0.701 ± 0.250 | 0.937 ± 0.641 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.325 ± 0.080 | **0.164 ± 0.022** | 0.263 ± 0.135 | 0.766 ± 0.145 | 1.421 ± 3.640 |
| bin_elementwise_mul_2048x2048 | **2.741 ± 0.187** | 2.916 ± 0.381 | 3.486 ± 2.037 | 3.588 ± 0.345 | 3.309 ± 0.663 |
| bin_elementwise_mul_512x512 | **0.084 ± 0.039** | 0.120 ± 0.096 | 0.151 ± 0.183 | 0.393 ± 0.069 | 0.089 ± 0.317 |
| bin_matmul_1024 | 15.421 ± 0.580 | **12.038 ± 2.152** | 14.690 ± 5.003 | 26.604 ± 2.822 | 41.548 ± 39.834 |
| bin_matmul_256 | 0.181 ± 0.025 | **0.127 ± 0.007** | 0.131 ± 0.004 | 1.292 ± 0.400 | 3.353 ± 2.698 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.005 ± 0.002** | 0.060 ± 0.004 | 0.134 ± 0.070 | 0.012 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.003 | **0.027 ± 0.004** | 0.202 ± 0.013 | 0.357 ± 0.215 | 0.099 ± 0.017 |
| bin_omeinsum_matmul_10x10 | **0.017 ± 0.001** | 0.020 ± 0.000 | 0.049 ± 0.003 | 0.230 ± 0.171 | 0.029 ± 0.014 |
| bin_outer_product_4096 | 4.775 ± 1.805 | 3.303 ± 1.325 | **3.281 ± 1.006** | 4.119 ± 1.753 | 240.837 ± 30.186 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 16.053 ± 1.493 | 17.357 ± 4.245 | **11.482 ± 6.030** | 44.147 ± 7.663 | 30.137 ± 8.777 |
| gm_queen5_5_3.wcsp | 686.189 ± 55.985 | 741.879 ± 28.564 | **668.107 ± 68.192** | 888.256 ± 177.111 | 2273.667 ± 499.068 |
| lm_batch_likelihood_brackets_4_4d | **14.542 ± 0.934** | 22.011 ± 14.093 | 23.134 ± 6.904 | 18.186 ± 3.450 | 123.657 ± 23.403 |
| lm_batch_likelihood_sentence_3_12d | **20.462 ± 0.660** | 42.494 ± 0.523 | 30.354 ± 6.214 | 38.088 ± 7.920 | 140.594 ± 20.511 |
| lm_batch_likelihood_sentence_4_4d | **10.639 ± 0.460** | 24.044 ± 1.446 | 14.499 ± 0.382 | 14.902 ± 0.580 | 167.302 ± 49.709 |
| nary_matmul_chain_64 | **0.019 ± 0.002** | 0.036 ± 0.001 | 0.024 ± 0.001 | 0.136 ± 0.033 | 0.071 ± 0.036 |
| str_matrix_chain_multiplication_100 | **4.699 ± 0.280** | 5.146 ± 0.239 | 5.553 ± 0.105 | 11.300 ± 4.720 | 284.572 ± 24.585 |
| str_mps_varying_inner_product_200 | 10.904 ± 0.441 | 11.082 ± 0.898 | **10.009 ± 1.431** | 34.190 ± 10.090 | 27.289 ± 21.343 |
| str_nw_mera_closed_120 | 318.784 ± 34.314 | 331.837 ± 76.473 | **275.241 ± 12.378** | 619.238 ± 73.609 | 910.087 ± 197.129 |
| str_nw_mera_open_26 | 321.738 ± 38.987 | 426.902 ± 13.679 | **255.508 ± 18.977** | 490.941 ± 115.389 | 1042.932 ± 458.657 |
| tensornetwork_permutation_focus_step409_316 | **140.114 ± 8.085** | 279.380 ± 35.478 | 191.619 ± 39.776 | 267.643 ± 13.601 | 470.218 ± 125.467 |
| tensornetwork_permutation_light_415 | **127.693 ± 0.987** | 136.175 ± 15.458 | 163.259 ± 60.135 | 297.358 ± 44.433 | 439.257 ± 53.649 |

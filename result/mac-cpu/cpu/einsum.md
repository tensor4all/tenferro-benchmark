# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`
- Benchmark commit: `8eb4f087ca7cfae8e8f724cdb0b344ad5123e372`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261106_172313/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261106_172313/run.yaml`
- Timestamp: `2026-11-06T17:23:13.669553+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261106_172313`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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

- `data/results/mac-cpu/cpu/einsum/20261106_172313/jax_cpu_t1_20261106_172313.log`
- `data/results/mac-cpu/cpu/einsum/20261106_172313/julia_omeinsum_t1_20261106_172313.log`
- `data/results/mac-cpu/cpu/einsum/20261106_172313/pytorch_cpu_t1_20261106_172313.log`
- `data/results/mac-cpu/cpu/einsum/20261106_172313/tenferro_eager_t1_20261106_172313.log`
- `data/results/mac-cpu/cpu/einsum/20261106_172313/tenferro_trace_t1_20261106_172313.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.525 ± 0.274 | **0.739 ± 0.094** | 0.872 ± 0.111 | 4.334 ± 0.449 | 7.715 ± 8.640 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.156 ± 0.014** | 0.382 ± 0.051 | 0.278 ± 0.398 | 0.636 ± 0.036 | 1.441 ± 3.384 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.248 ± 0.024** | **0.248 ± 0.021** | 0.374 ± 0.210 | 0.545 ± 0.077 | 0.865 ± 0.396 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.981 ± 0.366 | 0.244 ± 0.158 | **0.230 ± 0.052** | 0.497 ± 0.132 | 1.294 ± 6.027 |
| bin_elementwise_mul_2048x2048 | **2.470 ± 0.527** | 2.551 ± 0.115 | 3.270 ± 0.915 | 3.220 ± 0.104 | 2.996 ± 1.060 |
| bin_elementwise_mul_512x512 | 0.154 ± 0.171 | 0.113 ± 0.163 | **0.088 ± 0.018** | 0.352 ± 0.066 | 0.089 ± 0.016 |
| bin_matmul_1024 | 19.485 ± 11.461 | **15.737 ± 8.515** | 17.480 ± 4.302 | 58.445 ± 4.720 | 74.893 ± 23.643 |
| bin_matmul_256 | 0.246 ± 0.514 | 0.582 ± 0.048 | **0.159 ± 0.082** | 1.166 ± 0.127 | 1.140 ± 0.318 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | **0.008 ± 0.002** | 0.017 ± 0.008 | 0.018 ± 0.004 | 0.106 ± 0.012 | 0.010 ± 0.005 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.032 ± 0.005** | 0.064 ± 0.014 | 0.078 ± 0.012 | 0.294 ± 0.180 | 0.094 ± 0.012 |
| bin_omeinsum_matmul_10x10 | **0.016 ± 0.001** | **0.016 ± 0.003** | 0.017 ± 0.001 | 0.140 ± 0.047 | 0.033 ± 0.004 |
| bin_outer_product_4096 | 4.829 ± 0.406 | 9.956 ± 2.094 | 3.619 ± 0.371 | **3.403 ± 0.627** | 240.535 ± 19.904 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 29.641 ± 7.225 | 16.131 ± 3.476 | **12.407 ± 2.765** | 61.326 ± 6.389 | 74.543 ± 22.386 |
| gm_queen5_5_3.wcsp | 8726.590 ± 1970.648 | 9128.717 ± 856.004 | **5974.455 ± 1661.502** | 7114.628 ± 1542.094 | 9510.030 ± 4169.822 |
| lm_batch_likelihood_brackets_4_4d | 19.587 ± 7.787 | 30.121 ± 12.964 | **19.565 ± 4.442** | 20.961 ± 6.566 | 23.293 ± 10.459 |
| lm_batch_likelihood_sentence_3_12d | **31.429 ± 4.720** | 56.786 ± 14.153 | 42.296 ± 11.368 | 65.817 ± 20.644 | 99.227 ± 34.853 |
| lm_batch_likelihood_sentence_4_4d | **15.632 ± 3.613** | 30.317 ± 10.081 | 21.630 ± 4.920 | 18.568 ± 5.155 | 50.231 ± 26.837 |
| nary_matmul_chain_64 | 0.058 ± 0.009 | **0.010 ± 0.000** | 0.024 ± 0.004 | 0.223 ± 0.052 | 0.103 ± 0.036 |
| str_matrix_chain_multiplication_100 | **4.879 ± 0.703** | 7.280 ± 2.441 | 6.066 ± 1.350 | 15.882 ± 1.686 | 15.252 ± 1.662 |
| str_mps_varying_inner_product_200 | 10.479 ± 2.135 | 14.104 ± 5.992 | **8.120 ± 1.439** | 24.669 ± 7.421 | 19.924 ± 10.534 |
| str_nw_mera_closed_120 | 551.168 ± 71.659 | 682.177 ± 162.066 | **470.179 ± 94.189** | 1953.484 ± 375.622 | 1815.520 ± 447.414 |
| str_nw_mera_open_26 | 450.433 ± 123.021 | 522.519 ± 107.646 | **264.520 ± 44.504** | 1229.031 ± 179.699 | 1715.072 ± 307.546 |
| tensornetwork_permutation_focus_step409_316 | **182.354 ± 27.690** | 348.296 ± 42.108 | 349.773 ± 43.384 | 419.634 ± 49.066 | 346.824 ± 61.691 |
| tensornetwork_permutation_light_415 | 154.759 ± 25.261 | **148.106 ± 8.255** | 286.745 ± 77.400 | 416.700 ± 73.794 | 267.622 ± 25.565 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.525 ± 0.274 | **0.739 ± 0.094** | 0.872 ± 0.111 | 4.334 ± 0.449 | 7.715 ± 8.640 |
| bin_batched_matmul_b32_m64_n64_k64 | **0.156 ± 0.014** | 0.382 ± 0.051 | 0.278 ± 0.398 | 0.636 ± 0.036 | 1.441 ± 3.384 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.248 ± 0.024** | **0.248 ± 0.021** | 0.374 ± 0.210 | 0.545 ± 0.077 | 0.865 ± 0.396 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.981 ± 0.366 | 0.244 ± 0.158 | **0.230 ± 0.052** | 0.497 ± 0.132 | 1.294 ± 6.027 |
| bin_elementwise_mul_2048x2048 | **2.470 ± 0.527** | 2.551 ± 0.115 | 3.270 ± 0.915 | 3.220 ± 0.104 | 2.996 ± 1.060 |
| bin_elementwise_mul_512x512 | 0.154 ± 0.171 | 0.113 ± 0.163 | **0.088 ± 0.018** | 0.352 ± 0.066 | 0.089 ± 0.016 |
| bin_matmul_1024 | 19.485 ± 11.461 | **15.737 ± 8.515** | 17.480 ± 4.302 | 58.445 ± 4.720 | 74.893 ± 23.643 |
| bin_matmul_256 | 0.246 ± 0.514 | 0.582 ± 0.048 | **0.159 ± 0.082** | 1.166 ± 0.127 | 1.140 ± 0.318 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | **0.008 ± 0.002** | 0.017 ± 0.008 | 0.018 ± 0.004 | 0.106 ± 0.012 | 0.010 ± 0.005 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.032 ± 0.005** | 0.064 ± 0.014 | 0.078 ± 0.012 | 0.294 ± 0.180 | 0.094 ± 0.012 |
| bin_omeinsum_matmul_10x10 | **0.016 ± 0.001** | **0.016 ± 0.003** | 0.017 ± 0.001 | 0.140 ± 0.047 | 0.033 ± 0.004 |
| bin_outer_product_4096 | 4.829 ± 0.406 | 9.956 ± 2.094 | 3.619 ± 0.371 | **3.403 ± 0.627** | 240.535 ± 19.904 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 29.641 ± 7.225 | 16.131 ± 3.476 | **12.407 ± 2.765** | 61.326 ± 6.389 | 74.543 ± 22.386 |
| gm_queen5_5_3.wcsp | 1247.118 ± 194.342 | 1289.629 ± 290.242 | **1041.420 ± 616.625** | 1898.320 ± 499.448 | 2924.692 ± 371.467 |
| lm_batch_likelihood_brackets_4_4d | 18.718 ± 3.685 | 25.185 ± 16.273 | 15.449 ± 1.994 | **14.881 ± 3.316** | 59.378 ± 33.184 |
| lm_batch_likelihood_sentence_3_12d | 41.349 ± 16.598 | 58.797 ± 13.461 | **27.037 ± 1.895** | 68.973 ± 9.853 | 132.957 ± 50.394 |
| lm_batch_likelihood_sentence_4_4d | 20.119 ± 5.238 | 29.418 ± 6.870 | **14.411 ± 1.472** | 16.616 ± 1.630 | 44.031 ± 23.427 |
| nary_matmul_chain_64 | 0.058 ± 0.009 | **0.010 ± 0.000** | 0.024 ± 0.004 | 0.223 ± 0.052 | 0.103 ± 0.036 |
| str_matrix_chain_multiplication_100 | 7.353 ± 4.229 | 5.578 ± 0.819 | **5.573 ± 0.297** | 15.992 ± 1.784 | 20.637 ± 12.680 |
| str_mps_varying_inner_product_200 | 19.105 ± 6.139 | 11.130 ± 1.518 | **8.615 ± 0.596** | 24.284 ± 2.536 | 30.730 ± 10.778 |
| str_nw_mera_closed_120 | 392.582 ± 85.614 | 332.315 ± 30.299 | **311.109 ± 47.711** | 2086.605 ± 210.038 | 2001.126 ± 1475.570 |
| str_nw_mera_open_26 | 456.020 ± 97.735 | 539.045 ± 68.998 | **250.708 ± 21.605** | 1927.162 ± 500.779 | 1936.154 ± 317.383 |
| tensornetwork_permutation_focus_step409_316 | **182.354 ± 27.690** | 348.296 ± 42.108 | 349.773 ± 43.384 | 419.634 ± 49.066 | 346.824 ± 61.691 |
| tensornetwork_permutation_light_415 | 154.759 ± 25.261 | **148.106 ± 8.255** | 286.745 ± 77.400 | 416.700 ± 73.794 | 267.622 ± 25.565 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261106_175840/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261106_175840/run.yaml`
- Timestamp: `2026-11-06T17:58:40.229051+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261106_175840`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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

- `data/results/mac-cpu/cpu/einsum/20261106_175840/jax_cpu_t4_20261106_175840.log`
- `data/results/mac-cpu/cpu/einsum/20261106_175840/julia_omeinsum_t4_20261106_175840.log`
- `data/results/mac-cpu/cpu/einsum/20261106_175840/pytorch_cpu_t4_20261106_175840.log`
- `data/results/mac-cpu/cpu/einsum/20261106_175840/tenferro_eager_t4_20261106_175840.log`
- `data/results/mac-cpu/cpu/einsum/20261106_175840/tenferro_trace_t4_20261106_175840.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.504 ± 0.152 | **0.700 ± 0.068** | 0.888 ± 0.238 | 2.751 ± 0.119 | 105.568 ± 23.530 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.207 ± 0.041 | **0.086 ± 0.006** | 0.149 ± 0.026 | 0.772 ± 0.087 | 1.117 ± 2.780 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.226 ± 0.125 | **0.185 ± 0.035** | 0.485 ± 0.088 | 0.524 ± 0.047 | 1.179 ± 1.110 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.363 ± 0.094 | **0.205 ± 0.060** | 0.334 ± 0.189 | 0.526 ± 0.049 | 2.906 ± 7.806 |
| bin_elementwise_mul_2048x2048 | **2.827 ± 0.393** | 3.175 ± 1.035 | 3.209 ± 0.366 | 3.447 ± 0.188 | 3.636 ± 2.168 |
| bin_elementwise_mul_512x512 | **0.065 ± 0.041** | 0.272 ± 0.138 | 0.406 ± 0.218 | 0.376 ± 0.035 | 0.337 ± 0.135 |
| bin_matmul_1024 | **10.132 ± 5.919** | 10.919 ± 0.779 | 19.242 ± 9.686 | 30.296 ± 6.832 | 79.986 ± 25.192 |
| bin_matmul_256 | 0.136 ± 0.010 | **0.120 ± 0.007** | 0.134 ± 0.002 | 0.868 ± 0.024 | 7.555 ± 3.535 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.001 | **0.005 ± 0.001** | 0.018 ± 0.003 | 0.086 ± 0.006 | 0.010 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.002 | **0.027 ± 0.004** | 0.203 ± 0.009 | 0.194 ± 0.028 | 0.087 ± 0.012 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.001 | **0.008 ± 0.000** | 0.048 ± 0.001 | 0.170 ± 0.058 | 0.054 ± 0.009 |
| bin_outer_product_4096 | 3.218 ± 0.419 | **2.797 ± 0.280** | 3.853 ± 1.089 | 3.255 ± 0.220 | 231.298 ± 20.052 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 17.846 ± 4.231 | 11.796 ± 1.507 | **11.392 ± 0.803** | 33.227 ± 6.332 | 43.722 ± 22.631 |
| gm_queen5_5_3.wcsp | 5769.628 ± 2650.469 | 7308.118 ± 289.748 | **3076.809 ± 303.968** | 5083.746 ± 701.031 | 10203.474 ± 1760.615 |
| lm_batch_likelihood_brackets_4_4d | **14.298 ± 1.549** | 24.064 ± 15.950 | 22.481 ± 7.104 | 22.446 ± 2.014 | 113.709 ± 53.185 |
| lm_batch_likelihood_sentence_3_12d | **22.579 ± 7.650** | 45.383 ± 6.935 | 29.660 ± 2.539 | 39.032 ± 4.439 | 135.691 ± 28.406 |
| lm_batch_likelihood_sentence_4_4d | 14.248 ± 0.731 | 25.366 ± 4.177 | **14.005 ± 2.545** | 22.781 ± 2.202 | 121.363 ± 32.279 |
| nary_matmul_chain_64 | 0.021 ± 0.002 | **0.012 ± 0.000** | 0.024 ± 0.002 | 0.190 ± 0.061 | 0.239 ± 0.276 |
| str_matrix_chain_multiplication_100 | **5.326 ± 0.534** | 5.534 ± 0.406 | 5.448 ± 0.950 | 19.705 ± 4.302 | 276.605 ± 63.413 |
| str_mps_varying_inner_product_200 | **9.904 ± 0.730** | 10.083 ± 0.298 | 12.244 ± 4.672 | 33.136 ± 3.587 | 56.550 ± 34.476 |
| str_nw_mera_closed_120 | 531.229 ± 113.763 | 463.917 ± 104.038 | **337.535 ± 39.387** | 705.497 ± 143.067 | 1276.364 ± 530.901 |
| str_nw_mera_open_26 | 426.011 ± 145.120 | 508.946 ± 114.205 | **233.972 ± 5.696** | 588.480 ± 165.977 | 765.193 ± 188.974 |
| tensornetwork_permutation_focus_step409_316 | **173.751 ± 26.812** | 305.598 ± 53.117 | 185.826 ± 59.907 | 313.588 ± 63.294 | 417.705 ± 43.012 |
| tensornetwork_permutation_light_415 | 202.151 ± 25.567 | 163.858 ± 17.237 | **134.990 ± 27.982** | 301.091 ± 38.062 | 473.727 ± 63.800 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.504 ± 0.152 | **0.700 ± 0.068** | 0.888 ± 0.238 | 2.751 ± 0.119 | 105.568 ± 23.530 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.207 ± 0.041 | **0.086 ± 0.006** | 0.149 ± 0.026 | 0.772 ± 0.087 | 1.117 ± 2.780 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.226 ± 0.125 | **0.185 ± 0.035** | 0.485 ± 0.088 | 0.524 ± 0.047 | 1.179 ± 1.110 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.363 ± 0.094 | **0.205 ± 0.060** | 0.334 ± 0.189 | 0.526 ± 0.049 | 2.906 ± 7.806 |
| bin_elementwise_mul_2048x2048 | **2.827 ± 0.393** | 3.175 ± 1.035 | 3.209 ± 0.366 | 3.447 ± 0.188 | 3.636 ± 2.168 |
| bin_elementwise_mul_512x512 | **0.065 ± 0.041** | 0.272 ± 0.138 | 0.406 ± 0.218 | 0.376 ± 0.035 | 0.337 ± 0.135 |
| bin_matmul_1024 | **10.132 ± 5.919** | 10.919 ± 0.779 | 19.242 ± 9.686 | 30.296 ± 6.832 | 79.986 ± 25.192 |
| bin_matmul_256 | 0.136 ± 0.010 | **0.120 ± 0.007** | 0.134 ± 0.002 | 0.868 ± 0.024 | 7.555 ± 3.535 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.001 | **0.005 ± 0.001** | 0.018 ± 0.003 | 0.086 ± 0.006 | 0.010 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.002 | **0.027 ± 0.004** | 0.203 ± 0.009 | 0.194 ± 0.028 | 0.087 ± 0.012 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.001 | **0.008 ± 0.000** | 0.048 ± 0.001 | 0.170 ± 0.058 | 0.054 ± 0.009 |
| bin_outer_product_4096 | 3.218 ± 0.419 | **2.797 ± 0.280** | 3.853 ± 1.089 | 3.255 ± 0.220 | 231.298 ± 20.052 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 17.846 ± 4.231 | 11.796 ± 1.507 | **11.392 ± 0.803** | 33.227 ± 6.332 | 43.722 ± 22.631 |
| gm_queen5_5_3.wcsp | 921.907 ± 183.661 | 1030.695 ± 246.494 | **577.410 ± 107.605** | 1221.287 ± 301.581 | 1542.322 ± 454.171 |
| lm_batch_likelihood_brackets_4_4d | **11.546 ± 0.956** | 32.295 ± 19.153 | 17.674 ± 0.831 | 23.981 ± 2.307 | 130.608 ± 33.918 |
| lm_batch_likelihood_sentence_3_12d | **21.200 ± 4.025** | 50.426 ± 3.651 | 29.046 ± 2.393 | 36.692 ± 11.691 | 118.857 ± 26.305 |
| lm_batch_likelihood_sentence_4_4d | **13.655 ± 5.798** | 37.215 ± 6.363 | 18.722 ± 1.108 | 19.810 ± 8.337 | 113.216 ± 31.572 |
| nary_matmul_chain_64 | 0.021 ± 0.002 | **0.012 ± 0.000** | 0.024 ± 0.002 | 0.190 ± 0.061 | 0.239 ± 0.276 |
| str_matrix_chain_multiplication_100 | **4.845 ± 0.237** | 6.852 ± 1.951 | 5.536 ± 0.288 | 17.402 ± 10.434 | 273.919 ± 45.645 |
| str_mps_varying_inner_product_200 | **10.797 ± 0.876** | 13.024 ± 2.760 | 12.369 ± 1.175 | 34.663 ± 2.816 | 23.958 ± 20.754 |
| str_nw_mera_closed_120 | 374.092 ± 74.849 | 369.321 ± 98.986 | **292.104 ± 65.070** | 624.059 ± 98.650 | 735.498 ± 71.558 |
| str_nw_mera_open_26 | 337.087 ± 19.148 | 590.261 ± 143.937 | **241.672 ± 4.537** | 639.240 ± 139.241 | 1017.555 ± 372.818 |
| tensornetwork_permutation_focus_step409_316 | **173.751 ± 26.812** | 305.598 ± 53.117 | 185.826 ± 59.907 | 313.588 ± 63.294 | 417.705 ± 43.012 |
| tensornetwork_permutation_light_415 | 202.151 ± 25.567 | 163.858 ± 17.237 | **134.990 ± 27.982** | 301.091 ± 38.062 | 473.727 ± 63.800 |

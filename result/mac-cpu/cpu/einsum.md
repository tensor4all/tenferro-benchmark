# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `d6655b567e09aa29c3201655e1de418716934876`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261107_172300/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261107_172300/run.yaml`
- Timestamp: `2026-11-07T17:23:00.301852+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261107_172300`.

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

- `data/results/mac-cpu/cpu/einsum/20261107_172300/jax_cpu_t1_20261107_172300.log`
- `data/results/mac-cpu/cpu/einsum/20261107_172300/julia_omeinsum_t1_20261107_172300.log`
- `data/results/mac-cpu/cpu/einsum/20261107_172300/pytorch_cpu_t1_20261107_172300.log`
- `data/results/mac-cpu/cpu/einsum/20261107_172300/tenferro_eager_t1_20261107_172300.log`
- `data/results/mac-cpu/cpu/einsum/20261107_172300/tenferro_trace_t1_20261107_172300.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.634 ± 0.816 | **0.655 ± 0.104** | 0.911 ± 0.125 | 4.587 ± 0.168 | 5.213 ± 12.062 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.200 ± 0.626 | **0.088 ± 0.003** | 0.156 ± 0.016 | 0.752 ± 0.363 | 2.402 ± 4.650 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.409 ± 0.269 | **0.240 ± 0.034** | 0.397 ± 0.046 | 0.472 ± 0.092 | 0.706 ± 3.879 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.468 ± 1.215 | 0.267 ± 0.042 | **0.241 ± 0.176** | 0.637 ± 0.187 | 1.393 ± 0.373 |
| bin_elementwise_mul_2048x2048 | 4.857 ± 2.489 | **2.582 ± 0.365** | 4.373 ± 2.534 | 3.876 ± 0.525 | 3.327 ± 1.551 |
| bin_elementwise_mul_512x512 | 0.175 ± 0.160 | **0.074 ± 0.001** | 0.212 ± 0.109 | 0.422 ± 0.148 | 0.418 ± 0.095 |
| bin_matmul_1024 | 26.176 ± 6.211 | **13.939 ± 5.923** | 25.813 ± 8.195 | 64.128 ± 2.490 | 68.229 ± 13.701 |
| bin_matmul_256 | 0.177 ± 0.013 | **0.124 ± 0.003** | 0.138 ± 0.038 | 1.181 ± 0.145 | 1.159 ± 0.155 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.002 | **0.004 ± 0.000** | 0.019 ± 0.001 | 0.094 ± 0.040 | 0.010 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.038 ± 0.005 | **0.026 ± 0.002** | 0.088 ± 0.008 | 0.196 ± 0.061 | 0.089 ± 0.012 |
| bin_omeinsum_matmul_10x10 | 0.064 ± 0.119 | **0.007 ± 0.000** | 0.017 ± 0.001 | 0.160 ± 0.039 | 0.021 ± 0.018 |
| bin_outer_product_4096 | 8.971 ± 4.093 | 5.453 ± 1.079 | 5.259 ± 3.566 | **3.838 ± 0.383** | 235.996 ± 10.514 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 31.229 ± 15.268 | **12.269 ± 4.024** | 18.030 ± 2.589 | 84.421 ± 11.468 | 60.027 ± 22.836 |
| gm_queen5_5_3.wcsp | 10400.052 ± 1358.411 | 10172.019 ± 2042.950 | **5095.547 ± 657.807** | 6943.583 ± 2784.025 | 8344.491 ± 630.594 |
| lm_batch_likelihood_brackets_4_4d | 21.451 ± 2.027 | 19.451 ± 9.930 | 30.026 ± 9.448 | **15.844 ± 2.345** | 28.294 ± 6.131 |
| lm_batch_likelihood_sentence_3_12d | **33.991 ± 9.548** | 42.865 ± 20.665 | 41.244 ± 11.283 | 68.381 ± 7.899 | 128.260 ± 51.069 |
| lm_batch_likelihood_sentence_4_4d | **20.643 ± 7.919** | 24.939 ± 28.446 | 23.206 ± 8.362 | 21.349 ± 4.929 | 53.821 ± 26.026 |
| nary_matmul_chain_64 | **0.017 ± 0.001** | 0.030 ± 0.001 | 0.025 ± 0.002 | 0.337 ± 0.262 | 0.112 ± 0.022 |
| str_matrix_chain_multiplication_100 | **5.100 ± 1.527** | 7.200 ± 3.626 | 8.781 ± 3.211 | 15.449 ± 0.503 | 19.898 ± 8.566 |
| str_mps_varying_inner_product_200 | 11.396 ± 6.059 | 15.833 ± 5.789 | **9.442 ± 2.007** | 18.896 ± 1.894 | 24.549 ± 15.407 |
| str_nw_mera_closed_120 | 458.574 ± 53.948 | 727.175 ± 255.700 | **386.259 ± 73.027** | 1816.258 ± 208.698 | 1545.372 ± 343.608 |
| str_nw_mera_open_26 | 400.807 ± 139.168 | 644.299 ± 51.030 | **259.383 ± 16.415** | 1327.634 ± 238.035 | 1497.864 ± 263.999 |
| tensornetwork_permutation_focus_step409_316 | **166.113 ± 19.265** | 346.330 ± 100.938 | 299.631 ± 41.707 | 456.863 ± 43.394 | 257.688 ± 47.107 |
| tensornetwork_permutation_light_415 | 151.900 ± 26.918 | **146.451 ± 8.636** | 256.029 ± 45.064 | 463.346 ± 52.496 | 249.454 ± 41.421 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.634 ± 0.816 | **0.655 ± 0.104** | 0.911 ± 0.125 | 4.587 ± 0.168 | 5.213 ± 12.062 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.200 ± 0.626 | **0.088 ± 0.003** | 0.156 ± 0.016 | 0.752 ± 0.363 | 2.402 ± 4.650 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.409 ± 0.269 | **0.240 ± 0.034** | 0.397 ± 0.046 | 0.472 ± 0.092 | 0.706 ± 3.879 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.468 ± 1.215 | 0.267 ± 0.042 | **0.241 ± 0.176** | 0.637 ± 0.187 | 1.393 ± 0.373 |
| bin_elementwise_mul_2048x2048 | 4.857 ± 2.489 | **2.582 ± 0.365** | 4.373 ± 2.534 | 3.876 ± 0.525 | 3.327 ± 1.551 |
| bin_elementwise_mul_512x512 | 0.175 ± 0.160 | **0.074 ± 0.001** | 0.212 ± 0.109 | 0.422 ± 0.148 | 0.418 ± 0.095 |
| bin_matmul_1024 | 26.176 ± 6.211 | **13.939 ± 5.923** | 25.813 ± 8.195 | 64.128 ± 2.490 | 68.229 ± 13.701 |
| bin_matmul_256 | 0.177 ± 0.013 | **0.124 ± 0.003** | 0.138 ± 0.038 | 1.181 ± 0.145 | 1.159 ± 0.155 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.002 | **0.004 ± 0.000** | 0.019 ± 0.001 | 0.094 ± 0.040 | 0.010 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.038 ± 0.005 | **0.026 ± 0.002** | 0.088 ± 0.008 | 0.196 ± 0.061 | 0.089 ± 0.012 |
| bin_omeinsum_matmul_10x10 | 0.064 ± 0.119 | **0.007 ± 0.000** | 0.017 ± 0.001 | 0.160 ± 0.039 | 0.021 ± 0.018 |
| bin_outer_product_4096 | 8.971 ± 4.093 | 5.453 ± 1.079 | 5.259 ± 3.566 | **3.838 ± 0.383** | 235.996 ± 10.514 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 31.229 ± 15.268 | **12.269 ± 4.024** | 18.030 ± 2.589 | 84.421 ± 11.468 | 60.027 ± 22.836 |
| gm_queen5_5_3.wcsp | **919.537 ± 97.341** | 1652.455 ± 872.951 | 931.919 ± 275.618 | 1444.565 ± 389.676 | 2073.667 ± 794.123 |
| lm_batch_likelihood_brackets_4_4d | 18.266 ± 1.555 | 20.628 ± 20.314 | 23.937 ± 6.188 | **16.510 ± 1.429** | 43.171 ± 46.578 |
| lm_batch_likelihood_sentence_3_12d | **31.914 ± 0.923** | 58.115 ± 25.604 | 42.974 ± 7.243 | 68.257 ± 6.725 | 100.983 ± 32.164 |
| lm_batch_likelihood_sentence_4_4d | **17.213 ± 1.010** | 21.749 ± 8.168 | 29.509 ± 13.425 | 20.542 ± 6.043 | 57.146 ± 27.122 |
| nary_matmul_chain_64 | **0.017 ± 0.001** | 0.030 ± 0.001 | 0.025 ± 0.002 | 0.337 ± 0.262 | 0.112 ± 0.022 |
| str_matrix_chain_multiplication_100 | **5.028 ± 0.272** | 9.089 ± 5.178 | 5.907 ± 1.779 | 15.406 ± 0.486 | 19.149 ± 4.542 |
| str_mps_varying_inner_product_200 | 11.552 ± 0.121 | 14.191 ± 5.264 | **9.661 ± 1.103** | 26.765 ± 1.361 | 20.534 ± 8.329 |
| str_nw_mera_closed_120 | 393.312 ± 126.379 | **320.107 ± 86.512** | 366.516 ± 44.202 | 1754.772 ± 120.116 | 1667.110 ± 154.743 |
| str_nw_mera_open_26 | 363.519 ± 53.194 | 593.580 ± 167.955 | **320.342 ± 58.096** | 1367.424 ± 122.058 | 1436.195 ± 344.454 |
| tensornetwork_permutation_focus_step409_316 | **166.113 ± 19.265** | 346.330 ± 100.938 | 299.631 ± 41.707 | 456.863 ± 43.394 | 257.688 ± 47.107 |
| tensornetwork_permutation_light_415 | 151.900 ± 26.918 | **146.451 ± 8.636** | 256.029 ± 45.064 | 463.346 ± 52.496 | 249.454 ± 41.421 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261107_175904/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261107_175904/run.yaml`
- Timestamp: `2026-11-07T17:59:04.170477+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261107_175904`.

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

- `data/results/mac-cpu/cpu/einsum/20261107_175904/jax_cpu_t4_20261107_175904.log`
- `data/results/mac-cpu/cpu/einsum/20261107_175904/julia_omeinsum_t4_20261107_175904.log`
- `data/results/mac-cpu/cpu/einsum/20261107_175904/pytorch_cpu_t4_20261107_175904.log`
- `data/results/mac-cpu/cpu/einsum/20261107_175904/tenferro_eager_t4_20261107_175904.log`
- `data/results/mac-cpu/cpu/einsum/20261107_175904/tenferro_trace_t4_20261107_175904.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.183 ± 0.656 | **0.667 ± 0.236** | 0.701 ± 0.049 | 1.670 ± 0.075 | 88.292 ± 32.477 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.165 ± 0.003 | **0.079 ± 0.005** | 0.123 ± 0.007 | 0.690 ± 0.043 | 1.019 ± 1.507 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.187 ± 0.016** | 0.190 ± 0.017 | 0.215 ± 0.030 | 0.404 ± 0.065 | 0.702 ± 0.282 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.232 ± 0.002 | **0.172 ± 0.008** | 0.191 ± 0.003 | 0.619 ± 0.245 | 1.274 ± 3.327 |
| bin_elementwise_mul_2048x2048 | **1.921 ± 0.074** | 2.288 ± 0.154 | 2.227 ± 0.246 | 3.277 ± 0.258 | 2.361 ± 0.030 |
| bin_elementwise_mul_512x512 | **0.062 ± 0.019** | 0.112 ± 0.033 | 0.087 ± 0.014 | 0.356 ± 0.097 | 0.076 ± 0.016 |
| bin_matmul_1024 | 11.694 ± 1.364 | **9.783 ± 0.968** | 13.699 ± 5.601 | 23.388 ± 10.849 | 33.070 ± 15.762 |
| bin_matmul_256 | 0.151 ± 0.010 | **0.124 ± 0.007** | 0.126 ± 0.005 | 0.505 ± 0.034 | 3.799 ± 2.235 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.002 | **0.005 ± 0.001** | 0.018 ± 0.001 | 0.085 ± 0.006 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.035 ± 0.006 | **0.026 ± 0.001** | 0.083 ± 0.030 | 0.190 ± 0.045 | 0.083 ± 0.011 |
| bin_omeinsum_matmul_10x10 | 0.017 ± 0.002 | **0.008 ± 0.000** | 0.017 ± 0.001 | 0.090 ± 0.021 | 0.018 ± 0.002 |
| bin_outer_product_4096 | 3.032 ± 0.647 | 2.434 ± 0.171 | 2.893 ± 0.600 | **2.415 ± 0.104** | 240.911 ± 73.451 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 14.619 ± 0.835 | **11.041 ± 1.009** | 11.631 ± 2.162 | 21.879 ± 6.809 | 29.129 ± 18.062 |
| gm_queen5_5_3.wcsp | 3551.215 ± 678.124 | 6199.678 ± 1634.661 | **2798.537 ± 327.153** | 4854.713 ± 859.112 | 8818.883 ± 3546.613 |
| lm_batch_likelihood_brackets_4_4d | 14.098 ± 1.084 | 17.921 ± 2.131 | 19.018 ± 1.704 | **13.823 ± 0.232** | 72.340 ± 31.510 |
| lm_batch_likelihood_sentence_3_12d | **19.314 ± 0.387** | 38.824 ± 2.046 | 34.617 ± 5.402 | 26.698 ± 0.464 | 124.779 ± 29.965 |
| lm_batch_likelihood_sentence_4_4d | **12.587 ± 0.993** | 19.410 ± 0.326 | 18.458 ± 8.422 | 19.270 ± 3.648 | 113.036 ± 42.506 |
| nary_matmul_chain_64 | 0.019 ± 0.002 | **0.011 ± 0.000** | 0.025 ± 0.001 | 0.246 ± 0.094 | 0.172 ± 0.179 |
| str_matrix_chain_multiplication_100 | **4.546 ± 0.384** | 4.867 ± 0.234 | 5.884 ± 0.335 | 10.177 ± 0.244 | 262.580 ± 61.863 |
| str_mps_varying_inner_product_200 | **8.691 ± 0.036** | 9.241 ± 0.349 | 11.985 ± 7.832 | 18.595 ± 0.146 | 30.487 ± 20.141 |
| str_nw_mera_closed_120 | **373.303 ± 21.714** | 474.081 ± 72.699 | 404.769 ± 26.901 | 515.170 ± 98.655 | 926.437 ± 455.067 |
| str_nw_mera_open_26 | 310.248 ± 42.382 | 501.625 ± 199.927 | **224.299 ± 27.537** | 451.050 ± 263.596 | 949.744 ± 105.265 |
| tensornetwork_permutation_focus_step409_316 | **146.596 ± 14.369** | 338.346 ± 46.773 | 165.311 ± 37.287 | 230.612 ± 11.049 | 461.726 ± 48.378 |
| tensornetwork_permutation_light_415 | **127.252 ± 4.183** | 171.702 ± 43.448 | 178.503 ± 70.743 | 229.485 ± 45.243 | 477.127 ± 78.742 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.183 ± 0.656 | **0.667 ± 0.236** | 0.701 ± 0.049 | 1.670 ± 0.075 | 88.292 ± 32.477 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.165 ± 0.003 | **0.079 ± 0.005** | 0.123 ± 0.007 | 0.690 ± 0.043 | 1.019 ± 1.507 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.187 ± 0.016** | 0.190 ± 0.017 | 0.215 ± 0.030 | 0.404 ± 0.065 | 0.702 ± 0.282 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.232 ± 0.002 | **0.172 ± 0.008** | 0.191 ± 0.003 | 0.619 ± 0.245 | 1.274 ± 3.327 |
| bin_elementwise_mul_2048x2048 | **1.921 ± 0.074** | 2.288 ± 0.154 | 2.227 ± 0.246 | 3.277 ± 0.258 | 2.361 ± 0.030 |
| bin_elementwise_mul_512x512 | **0.062 ± 0.019** | 0.112 ± 0.033 | 0.087 ± 0.014 | 0.356 ± 0.097 | 0.076 ± 0.016 |
| bin_matmul_1024 | 11.694 ± 1.364 | **9.783 ± 0.968** | 13.699 ± 5.601 | 23.388 ± 10.849 | 33.070 ± 15.762 |
| bin_matmul_256 | 0.151 ± 0.010 | **0.124 ± 0.007** | 0.126 ± 0.005 | 0.505 ± 0.034 | 3.799 ± 2.235 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.002 | **0.005 ± 0.001** | 0.018 ± 0.001 | 0.085 ± 0.006 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.035 ± 0.006 | **0.026 ± 0.001** | 0.083 ± 0.030 | 0.190 ± 0.045 | 0.083 ± 0.011 |
| bin_omeinsum_matmul_10x10 | 0.017 ± 0.002 | **0.008 ± 0.000** | 0.017 ± 0.001 | 0.090 ± 0.021 | 0.018 ± 0.002 |
| bin_outer_product_4096 | 3.032 ± 0.647 | 2.434 ± 0.171 | 2.893 ± 0.600 | **2.415 ± 0.104** | 240.911 ± 73.451 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 14.619 ± 0.835 | **11.041 ± 1.009** | 11.631 ± 2.162 | 21.879 ± 6.809 | 29.129 ± 18.062 |
| gm_queen5_5_3.wcsp | 665.416 ± 21.837 | 1038.155 ± 301.263 | 651.832 ± 107.934 | **637.808 ± 55.816** | 1737.548 ± 291.939 |
| lm_batch_likelihood_brackets_4_4d | 14.524 ± 1.395 | 23.092 ± 4.709 | 24.721 ± 4.528 | **13.928 ± 1.610** | 121.789 ± 30.455 |
| lm_batch_likelihood_sentence_3_12d | **21.684 ± 1.631** | 58.443 ± 6.898 | 32.927 ± 5.367 | 28.569 ± 0.691 | 124.228 ± 44.276 |
| lm_batch_likelihood_sentence_4_4d | **10.373 ± 0.364** | 44.468 ± 12.850 | 16.213 ± 5.093 | 14.271 ± 0.239 | 141.317 ± 99.140 |
| nary_matmul_chain_64 | 0.019 ± 0.002 | **0.011 ± 0.000** | 0.025 ± 0.001 | 0.246 ± 0.094 | 0.172 ± 0.179 |
| str_matrix_chain_multiplication_100 | **4.276 ± 0.076** | 7.357 ± 3.419 | 7.417 ± 1.223 | 10.260 ± 0.233 | 264.850 ± 48.633 |
| str_mps_varying_inner_product_200 | **10.169 ± 0.224** | 13.600 ± 1.778 | 23.973 ± 4.486 | 21.216 ± 0.321 | 15.321 ± 17.932 |
| str_nw_mera_closed_120 | 308.789 ± 19.044 | 338.032 ± 50.444 | **302.110 ± 22.735** | 446.509 ± 141.077 | 774.362 ± 268.943 |
| str_nw_mera_open_26 | 303.104 ± 29.199 | 507.971 ± 93.868 | **267.028 ± 38.326** | 430.673 ± 98.291 | 1207.116 ± 405.919 |
| tensornetwork_permutation_focus_step409_316 | **146.596 ± 14.369** | 338.346 ± 46.773 | 165.311 ± 37.287 | 230.612 ± 11.049 | 461.726 ± 48.378 |
| tensornetwork_permutation_light_415 | **127.252 ± 4.183** | 171.702 ± 43.448 | 178.503 ± 70.743 | 229.485 ± 45.243 | 477.127 ± 78.742 |

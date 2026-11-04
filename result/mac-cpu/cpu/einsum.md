# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`
- Benchmark commit: `ef930a2e0f075f6b42bbbf9608afa9ea69ed412b`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261104_172845/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261104_172845/run.yaml`
- Timestamp: `2026-11-04T17:28:45.718036+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261104_172845`.

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

- `data/results/mac-cpu/cpu/einsum/20261104_172845/jax_cpu_t1_20261104_172845.log`
- `data/results/mac-cpu/cpu/einsum/20261104_172845/julia_omeinsum_t1_20261104_172845.log`
- `data/results/mac-cpu/cpu/einsum/20261104_172845/pytorch_cpu_t1_20261104_172845.log`
- `data/results/mac-cpu/cpu/einsum/20261104_172845/tenferro_eager_t1_20261104_172845.log`
- `data/results/mac-cpu/cpu/einsum/20261104_172845/tenferro_trace_t1_20261104_172845.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.357 ± 0.116 | **0.522 ± 0.076** | 0.584 ± 0.029 | 3.368 ± 0.052 | 4.455 ± 5.284 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.143 ± 0.022 | **0.083 ± 0.008** | 0.113 ± 0.001 | 0.545 ± 0.024 | 0.677 ± 0.912 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.254 ± 0.025 | **0.206 ± 0.008** | 0.224 ± 0.034 | 0.343 ± 0.100 | 0.877 ± 0.314 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.971 ± 0.083 | 0.229 ± 0.010 | **0.209 ± 0.039** | 0.403 ± 0.022 | 1.352 ± 4.952 |
| bin_elementwise_mul_2048x2048 | 2.667 ± 0.227 | 2.227 ± 0.125 | **1.845 ± 0.102** | 3.264 ± 0.134 | 3.111 ± 0.436 |
| bin_elementwise_mul_512x512 | 0.081 ± 0.010 | **0.072 ± 0.000** | 0.075 ± 0.001 | 0.231 ± 0.024 | 0.098 ± 0.145 |
| bin_matmul_1024 | 15.551 ± 4.268 | 10.360 ± 0.331 | **8.283 ± 1.862** | 46.705 ± 1.921 | 55.417 ± 20.770 |
| bin_matmul_256 | 0.156 ± 0.003 | 0.129 ± 0.019 | **0.127 ± 0.009** | 0.839 ± 0.007 | 1.157 ± 0.255 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.004 ± 0.000** | 0.019 ± 0.002 | 0.079 ± 0.005 | 0.009 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.002 | **0.027 ± 0.005** | 0.072 ± 0.001 | 0.146 ± 0.038 | 0.085 ± 0.010 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.007 ± 0.000** | 0.015 ± 0.002 | 0.064 ± 0.007 | 0.023 ± 0.009 |
| bin_outer_product_4096 | 5.296 ± 0.573 | 4.945 ± 0.567 | 2.932 ± 0.569 | **2.238 ± 0.290** | 235.318 ± 18.740 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 18.269 ± 2.699 | 11.181 ± 1.098 | **11.000 ± 0.896** | 51.654 ± 2.262 | 53.041 ± 15.122 |
| gm_queen5_5_3.wcsp | 4991.605 ± 542.517 | 4542.922 ± 758.801 | **2485.902 ± 461.246** | 3699.801 ± 407.009 | 3780.047 ± 818.130 |
| lm_batch_likelihood_brackets_4_4d | 14.705 ± 1.782 | 17.249 ± 3.556 | **12.659 ± 2.369** | 13.371 ± 0.844 | 24.398 ± 9.713 |
| lm_batch_likelihood_sentence_3_12d | **23.311 ± 2.674** | 41.542 ± 13.164 | 24.044 ± 2.297 | 48.579 ± 2.296 | 78.158 ± 23.946 |
| lm_batch_likelihood_sentence_4_4d | 15.124 ± 0.768 | 20.685 ± 3.695 | 13.703 ± 1.499 | **13.002 ± 0.482** | 40.166 ± 17.534 |
| nary_matmul_chain_64 | 0.024 ± 0.006 | **0.009 ± 0.002** | 0.022 ± 0.001 | 0.114 ± 0.004 | 0.081 ± 0.017 |
| str_matrix_chain_multiplication_100 | 4.559 ± 0.237 | **4.455 ± 0.599** | 5.082 ± 0.732 | 14.130 ± 0.469 | 13.063 ± 1.387 |
| str_mps_varying_inner_product_200 | 9.744 ± 0.344 | 7.890 ± 5.517 | **6.413 ± 0.418** | 16.844 ± 0.169 | 17.210 ± 5.069 |
| str_nw_mera_closed_120 | 408.045 ± 61.771 | 423.779 ± 46.606 | **327.523 ± 21.309** | 1453.915 ± 49.015 | 1205.361 ± 59.082 |
| str_nw_mera_open_26 | 321.502 ± 57.145 | 416.223 ± 30.932 | **208.381 ± 20.891** | 1049.752 ± 117.887 | 901.212 ± 34.074 |
| tensornetwork_permutation_focus_step409_316 | **148.521 ± 4.851** | 294.828 ± 17.469 | 236.098 ± 2.825 | 334.189 ± 21.204 | 207.329 ± 12.539 |
| tensornetwork_permutation_light_415 | 136.489 ± 9.218 | **128.878 ± 4.780** | 195.731 ± 2.327 | 348.218 ± 36.389 | 186.569 ± 29.881 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.357 ± 0.116 | **0.522 ± 0.076** | 0.584 ± 0.029 | 3.368 ± 0.052 | 4.455 ± 5.284 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.143 ± 0.022 | **0.083 ± 0.008** | 0.113 ± 0.001 | 0.545 ± 0.024 | 0.677 ± 0.912 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.254 ± 0.025 | **0.206 ± 0.008** | 0.224 ± 0.034 | 0.343 ± 0.100 | 0.877 ± 0.314 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.971 ± 0.083 | 0.229 ± 0.010 | **0.209 ± 0.039** | 0.403 ± 0.022 | 1.352 ± 4.952 |
| bin_elementwise_mul_2048x2048 | 2.667 ± 0.227 | 2.227 ± 0.125 | **1.845 ± 0.102** | 3.264 ± 0.134 | 3.111 ± 0.436 |
| bin_elementwise_mul_512x512 | 0.081 ± 0.010 | **0.072 ± 0.000** | 0.075 ± 0.001 | 0.231 ± 0.024 | 0.098 ± 0.145 |
| bin_matmul_1024 | 15.551 ± 4.268 | 10.360 ± 0.331 | **8.283 ± 1.862** | 46.705 ± 1.921 | 55.417 ± 20.770 |
| bin_matmul_256 | 0.156 ± 0.003 | 0.129 ± 0.019 | **0.127 ± 0.009** | 0.839 ± 0.007 | 1.157 ± 0.255 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.004 ± 0.000** | 0.019 ± 0.002 | 0.079 ± 0.005 | 0.009 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.002 | **0.027 ± 0.005** | 0.072 ± 0.001 | 0.146 ± 0.038 | 0.085 ± 0.010 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.007 ± 0.000** | 0.015 ± 0.002 | 0.064 ± 0.007 | 0.023 ± 0.009 |
| bin_outer_product_4096 | 5.296 ± 0.573 | 4.945 ± 0.567 | 2.932 ± 0.569 | **2.238 ± 0.290** | 235.318 ± 18.740 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 18.269 ± 2.699 | 11.181 ± 1.098 | **11.000 ± 0.896** | 51.654 ± 2.262 | 53.041 ± 15.122 |
| gm_queen5_5_3.wcsp | 836.297 ± 48.335 | 826.237 ± 100.654 | **656.902 ± 136.701** | 1114.618 ± 73.347 | 921.337 ± 34.214 |
| lm_batch_likelihood_brackets_4_4d | 14.841 ± 1.031 | 20.697 ± 8.787 | 13.497 ± 1.306 | **12.177 ± 0.789** | 29.336 ± 16.010 |
| lm_batch_likelihood_sentence_3_12d | 26.691 ± 1.126 | 41.714 ± 3.787 | **22.796 ± 0.621** | 53.610 ± 2.520 | 74.276 ± 19.029 |
| lm_batch_likelihood_sentence_4_4d | **11.692 ± 0.347** | 21.890 ± 4.100 | 13.449 ± 0.223 | 14.525 ± 1.166 | 32.367 ± 21.927 |
| nary_matmul_chain_64 | 0.024 ± 0.006 | **0.009 ± 0.002** | 0.022 ± 0.001 | 0.114 ± 0.004 | 0.081 ± 0.017 |
| str_matrix_chain_multiplication_100 | 4.601 ± 0.050 | 4.915 ± 0.433 | **4.521 ± 0.046** | 14.623 ± 1.656 | 15.560 ± 3.015 |
| str_mps_varying_inner_product_200 | 10.684 ± 0.233 | 9.979 ± 0.800 | **7.530 ± 0.476** | 21.415 ± 0.607 | 15.105 ± 6.848 |
| str_nw_mera_closed_120 | 329.573 ± 16.213 | 308.252 ± 54.067 | **251.259 ± 27.894** | 1498.595 ± 47.966 | 1160.664 ± 49.795 |
| str_nw_mera_open_26 | 332.795 ± 22.250 | 456.904 ± 35.212 | **215.066 ± 2.593** | 1166.343 ± 48.723 | 921.553 ± 41.973 |
| tensornetwork_permutation_focus_step409_316 | **148.521 ± 4.851** | 294.828 ± 17.469 | 236.098 ± 2.825 | 334.189 ± 21.204 | 207.329 ± 12.539 |
| tensornetwork_permutation_light_415 | 136.489 ± 9.218 | **128.878 ± 4.780** | 195.731 ± 2.327 | 348.218 ± 36.389 | 186.569 ± 29.881 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261104_175043/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261104_175043/run.yaml`
- Timestamp: `2026-11-04T17:50:43.176758+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261104_175043`.

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

- `data/results/mac-cpu/cpu/einsum/20261104_175043/jax_cpu_t4_20261104_175043.log`
- `data/results/mac-cpu/cpu/einsum/20261104_175043/julia_omeinsum_t4_20261104_175043.log`
- `data/results/mac-cpu/cpu/einsum/20261104_175043/pytorch_cpu_t4_20261104_175043.log`
- `data/results/mac-cpu/cpu/einsum/20261104_175043/tenferro_eager_t4_20261104_175043.log`
- `data/results/mac-cpu/cpu/einsum/20261104_175043/tenferro_trace_t4_20261104_175043.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.229 ± 0.061 | 0.672 ± 0.087 | **0.609 ± 0.011** | 1.656 ± 0.082 | 91.504 ± 18.757 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.165 ± 0.060 | **0.091 ± 0.010** | 0.113 ± 0.001 | 0.578 ± 0.017 | 0.834 ± 0.974 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.157 ± 0.018 | **0.128 ± 0.014** | 0.162 ± 0.001 | 0.327 ± 0.065 | 0.706 ± 0.212 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.243 ± 0.019 | **0.146 ± 0.019** | 0.159 ± 0.008 | 0.421 ± 0.039 | 1.201 ± 3.700 |
| bin_elementwise_mul_2048x2048 | 2.428 ± 0.271 | 2.038 ± 0.207 | **1.929 ± 0.013** | 2.669 ± 0.191 | 2.164 ± 0.476 |
| bin_elementwise_mul_512x512 | 0.073 ± 0.013 | **0.057 ± 0.006** | 0.073 ± 0.019 | 0.261 ± 0.011 | 0.074 ± 0.005 |
| bin_matmul_1024 | 13.015 ± 3.105 | **8.087 ± 1.890** | 8.282 ± 3.790 | 17.755 ± 0.339 | 33.515 ± 8.202 |
| bin_matmul_256 | 0.157 ± 0.010 | **0.103 ± 0.001** | 0.109 ± 0.001 | 0.495 ± 0.074 | 3.549 ± 2.695 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.005 ± 0.002** | 0.016 ± 0.001 | 0.080 ± 0.007 | 0.011 ± 0.006 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.002 | **0.026 ± 0.003** | 0.072 ± 0.001 | 0.168 ± 0.016 | 0.101 ± 0.079 |
| bin_omeinsum_matmul_10x10 | 0.017 ± 0.001 | **0.007 ± 0.001** | 0.015 ± 0.002 | 0.077 ± 0.010 | 0.032 ± 0.013 |
| bin_outer_product_4096 | 2.225 ± 0.050 | **2.217 ± 0.472** | 2.257 ± 0.116 | 2.280 ± 0.482 | 459.960 ± 177.347 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 16.522 ± 3.055 | **9.210 ± 2.236** | 9.260 ± 1.215 | 20.986 ± 1.488 | 32.749 ± 9.275 |
| gm_queen5_5_3.wcsp | 2589.658 ± 349.238 | 3397.555 ± 462.866 | **1010.881 ± 200.340** | 2596.134 ± 393.787 | 6132.888 ± 330.282 |
| lm_batch_likelihood_brackets_4_4d | **10.269 ± 0.950** | 14.772 ± 7.098 | 11.516 ± 1.520 | 13.281 ± 1.212 | 79.691 ± 57.513 |
| lm_batch_likelihood_sentence_3_12d | **16.108 ± 2.488** | 32.357 ± 4.334 | 22.724 ± 2.051 | 25.693 ± 2.111 | 111.668 ± 26.208 |
| lm_batch_likelihood_sentence_4_4d | 11.611 ± 1.220 | 16.986 ± 2.454 | **11.399 ± 0.360** | 13.702 ± 1.529 | 102.163 ± 23.620 |
| nary_matmul_chain_64 | 0.020 ± 0.000 | **0.012 ± 0.001** | 0.022 ± 0.002 | 0.115 ± 0.012 | 0.086 ± 0.028 |
| str_matrix_chain_multiplication_100 | **4.216 ± 0.131** | 4.228 ± 0.045 | 5.727 ± 1.612 | 10.343 ± 0.104 | 246.721 ± 50.155 |
| str_mps_varying_inner_product_200 | 8.017 ± 0.668 | 7.959 ± 0.990 | **7.082 ± 0.743** | 18.810 ± 1.825 | 33.254 ± 23.676 |
| str_nw_mera_closed_120 | 349.639 ± 37.885 | 337.114 ± 20.206 | **306.998 ± 14.672** | 496.683 ± 140.015 | 723.333 ± 39.412 |
| str_nw_mera_open_26 | 263.338 ± 8.984 | 355.561 ± 22.132 | **204.049 ± 18.559** | 373.277 ± 26.211 | 608.890 ± 67.473 |
| tensornetwork_permutation_focus_step409_316 | **127.648 ± 3.838** | 243.780 ± 23.451 | 136.313 ± 21.283 | 206.902 ± 6.810 | 396.386 ± 49.043 |
| tensornetwork_permutation_light_415 | 117.812 ± 3.700 | 113.364 ± 3.314 | **108.403 ± 14.058** | 217.324 ± 14.283 | 378.029 ± 32.066 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.229 ± 0.061 | 0.672 ± 0.087 | **0.609 ± 0.011** | 1.656 ± 0.082 | 91.504 ± 18.757 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.165 ± 0.060 | **0.091 ± 0.010** | 0.113 ± 0.001 | 0.578 ± 0.017 | 0.834 ± 0.974 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.157 ± 0.018 | **0.128 ± 0.014** | 0.162 ± 0.001 | 0.327 ± 0.065 | 0.706 ± 0.212 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.243 ± 0.019 | **0.146 ± 0.019** | 0.159 ± 0.008 | 0.421 ± 0.039 | 1.201 ± 3.700 |
| bin_elementwise_mul_2048x2048 | 2.428 ± 0.271 | 2.038 ± 0.207 | **1.929 ± 0.013** | 2.669 ± 0.191 | 2.164 ± 0.476 |
| bin_elementwise_mul_512x512 | 0.073 ± 0.013 | **0.057 ± 0.006** | 0.073 ± 0.019 | 0.261 ± 0.011 | 0.074 ± 0.005 |
| bin_matmul_1024 | 13.015 ± 3.105 | **8.087 ± 1.890** | 8.282 ± 3.790 | 17.755 ± 0.339 | 33.515 ± 8.202 |
| bin_matmul_256 | 0.157 ± 0.010 | **0.103 ± 0.001** | 0.109 ± 0.001 | 0.495 ± 0.074 | 3.549 ± 2.695 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.005 ± 0.002** | 0.016 ± 0.001 | 0.080 ± 0.007 | 0.011 ± 0.006 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.002 | **0.026 ± 0.003** | 0.072 ± 0.001 | 0.168 ± 0.016 | 0.101 ± 0.079 |
| bin_omeinsum_matmul_10x10 | 0.017 ± 0.001 | **0.007 ± 0.001** | 0.015 ± 0.002 | 0.077 ± 0.010 | 0.032 ± 0.013 |
| bin_outer_product_4096 | 2.225 ± 0.050 | **2.217 ± 0.472** | 2.257 ± 0.116 | 2.280 ± 0.482 | 459.960 ± 177.347 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 16.522 ± 3.055 | **9.210 ± 2.236** | 9.260 ± 1.215 | 20.986 ± 1.488 | 32.749 ± 9.275 |
| gm_queen5_5_3.wcsp | 635.225 ± 15.975 | 619.010 ± 36.985 | **462.241 ± 40.037** | 585.379 ± 41.296 | 761.008 ± 75.586 |
| lm_batch_likelihood_brackets_4_4d | **12.021 ± 0.572** | 18.268 ± 9.067 | 12.520 ± 1.544 | 13.014 ± 1.293 | 114.892 ± 29.286 |
| lm_batch_likelihood_sentence_3_12d | **20.964 ± 2.366** | 32.328 ± 3.599 | 22.304 ± 2.600 | 27.004 ± 1.973 | 109.950 ± 13.566 |
| lm_batch_likelihood_sentence_4_4d | **8.247 ± 0.884** | 18.066 ± 1.934 | 11.680 ± 1.351 | 13.717 ± 1.221 | 98.705 ± 22.072 |
| nary_matmul_chain_64 | 0.020 ± 0.000 | **0.012 ± 0.001** | 0.022 ± 0.002 | 0.115 ± 0.012 | 0.086 ± 0.028 |
| str_matrix_chain_multiplication_100 | **4.163 ± 1.008** | 4.865 ± 0.579 | 4.483 ± 0.082 | 9.490 ± 0.628 | 274.674 ± 68.738 |
| str_mps_varying_inner_product_200 | 9.851 ± 0.105 | **9.024 ± 1.201** | 9.241 ± 0.142 | 22.043 ± 0.307 | 21.309 ± 20.333 |
| str_nw_mera_closed_120 | 281.708 ± 3.976 | 269.453 ± 23.074 | **248.196 ± 18.392** | 437.282 ± 18.146 | 641.420 ± 65.918 |
| str_nw_mera_open_26 | 277.371 ± 42.025 | 382.290 ± 34.977 | **207.050 ± 18.870** | 383.053 ± 5.172 | 668.084 ± 32.276 |
| tensornetwork_permutation_focus_step409_316 | **127.648 ± 3.838** | 243.780 ± 23.451 | 136.313 ± 21.283 | 206.902 ± 6.810 | 396.386 ± 49.043 |
| tensornetwork_permutation_light_415 | 117.812 ± 3.700 | 113.364 ± 3.314 | **108.403 ± 14.058** | 217.324 ± 14.283 | 378.029 ± 32.066 |

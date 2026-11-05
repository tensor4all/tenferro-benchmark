# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`
- Benchmark commit: `ce2efd37d9e132703dfd2ebefbf4353249cf53ea`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261105_172310/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261105_172310/run.yaml`
- Timestamp: `2026-11-05T17:23:10.950764+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261105_172310`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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

- `data/results/mac-cpu/cpu/einsum/20261105_172310/jax_cpu_t1_20261105_172310.log`
- `data/results/mac-cpu/cpu/einsum/20261105_172310/julia_omeinsum_t1_20261105_172310.log`
- `data/results/mac-cpu/cpu/einsum/20261105_172310/pytorch_cpu_t1_20261105_172310.log`
- `data/results/mac-cpu/cpu/einsum/20261105_172310/tenferro_eager_t1_20261105_172310.log`
- `data/results/mac-cpu/cpu/einsum/20261105_172310/tenferro_trace_t1_20261105_172310.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.138 ± 0.236 | **0.448 ± 0.065** | 0.509 ± 0.038 | 3.153 ± 0.142 | 5.090 ± 4.070 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.159 ± 0.029 | **0.075 ± 0.004** | 0.113 ± 0.003 | 0.537 ± 0.172 | 0.672 ± 0.226 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.212 ± 0.020 | 0.196 ± 0.009 | **0.166 ± 0.002** | 0.269 ± 0.078 | 0.608 ± 0.156 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.806 ± 0.043 | 0.221 ± 0.023 | **0.187 ± 0.033** | 0.320 ± 0.015 | 1.005 ± 3.917 |
| bin_elementwise_mul_2048x2048 | **1.562 ± 0.194** | 1.574 ± 0.192 | 2.014 ± 0.286 | 2.010 ± 0.130 | 1.696 ± 0.165 |
| bin_elementwise_mul_512x512 | 0.077 ± 0.010 | **0.069 ± 0.013** | 0.075 ± 0.002 | 0.219 ± 0.062 | 0.071 ± 0.016 |
| bin_matmul_1024 | 9.813 ± 5.432 | 8.257 ± 2.832 | **6.429 ± 0.359** | 44.976 ± 8.114 | 53.517 ± 8.799 |
| bin_matmul_256 | 0.154 ± 0.032 | 0.113 ± 0.002 | **0.103 ± 0.004** | 0.830 ± 0.159 | 0.843 ± 0.046 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.002 | **0.005 ± 0.001** | 0.017 ± 0.003 | 0.078 ± 0.010 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.002 | **0.026 ± 0.004** | 0.070 ± 0.002 | 0.215 ± 0.046 | 0.039 ± 0.005 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.007 ± 0.000** | 0.015 ± 0.001 | 0.071 ± 0.020 | 0.017 ± 0.002 |
| bin_outer_product_4096 | 3.864 ± 0.457 | 3.556 ± 0.497 | 2.525 ± 0.151 | **2.467 ± 0.290** | 114.411 ± 15.741 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 20.904 ± 8.783 | 9.731 ± 1.682 | **7.658 ± 0.714** | 52.605 ± 4.187 | 51.702 ± 3.212 |
| gm_queen5_5_3.wcsp | 2519.466 ± 362.755 | 2695.472 ± 177.264 | **1535.952 ± 104.834** | 2705.879 ± 212.563 | 2240.794 ± 292.464 |
| lm_batch_likelihood_brackets_4_4d | 13.904 ± 0.822 | 23.557 ± 7.087 | 14.360 ± 1.141 | **12.696 ± 0.814** | 19.138 ± 6.130 |
| lm_batch_likelihood_sentence_3_12d | **19.547 ± 3.969** | 43.528 ± 3.798 | 24.534 ± 1.828 | 51.316 ± 2.414 | 79.401 ± 22.244 |
| lm_batch_likelihood_sentence_4_4d | **13.117 ± 0.413** | 20.748 ± 1.842 | 14.227 ± 0.595 | 15.369 ± 1.218 | 42.460 ± 21.928 |
| nary_matmul_chain_64 | 0.019 ± 0.001 | **0.011 ± 0.000** | 0.021 ± 0.001 | 0.104 ± 0.013 | 0.069 ± 0.013 |
| str_matrix_chain_multiplication_100 | **4.094 ± 0.321** | 4.743 ± 0.395 | 5.038 ± 0.310 | 14.169 ± 1.010 | 15.496 ± 1.139 |
| str_mps_varying_inner_product_200 | 9.099 ± 1.507 | 8.875 ± 0.815 | **6.693 ± 0.404** | 18.787 ± 0.701 | 17.384 ± 22.641 |
| str_nw_mera_closed_120 | 356.218 ± 43.872 | 372.666 ± 26.746 | **305.620 ± 19.230** | 1427.221 ± 177.735 | 1216.727 ± 27.638 |
| str_nw_mera_open_26 | 295.629 ± 23.209 | 404.534 ± 48.971 | **203.169 ± 19.218** | 1098.000 ± 142.571 | 893.130 ± 43.035 |
| tensornetwork_permutation_focus_step409_316 | **136.456 ± 21.026** | 360.544 ± 36.494 | 254.226 ± 11.286 | 409.580 ± 25.508 | 219.329 ± 25.992 |
| tensornetwork_permutation_light_415 | **132.398 ± 15.870** | 142.338 ± 14.427 | 193.988 ± 16.801 | 344.632 ± 37.824 | 218.799 ± 59.070 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.138 ± 0.236 | **0.448 ± 0.065** | 0.509 ± 0.038 | 3.153 ± 0.142 | 5.090 ± 4.070 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.159 ± 0.029 | **0.075 ± 0.004** | 0.113 ± 0.003 | 0.537 ± 0.172 | 0.672 ± 0.226 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.212 ± 0.020 | 0.196 ± 0.009 | **0.166 ± 0.002** | 0.269 ± 0.078 | 0.608 ± 0.156 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.806 ± 0.043 | 0.221 ± 0.023 | **0.187 ± 0.033** | 0.320 ± 0.015 | 1.005 ± 3.917 |
| bin_elementwise_mul_2048x2048 | **1.562 ± 0.194** | 1.574 ± 0.192 | 2.014 ± 0.286 | 2.010 ± 0.130 | 1.696 ± 0.165 |
| bin_elementwise_mul_512x512 | 0.077 ± 0.010 | **0.069 ± 0.013** | 0.075 ± 0.002 | 0.219 ± 0.062 | 0.071 ± 0.016 |
| bin_matmul_1024 | 9.813 ± 5.432 | 8.257 ± 2.832 | **6.429 ± 0.359** | 44.976 ± 8.114 | 53.517 ± 8.799 |
| bin_matmul_256 | 0.154 ± 0.032 | 0.113 ± 0.002 | **0.103 ± 0.004** | 0.830 ± 0.159 | 0.843 ± 0.046 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.002 | **0.005 ± 0.001** | 0.017 ± 0.003 | 0.078 ± 0.010 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.002 | **0.026 ± 0.004** | 0.070 ± 0.002 | 0.215 ± 0.046 | 0.039 ± 0.005 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.007 ± 0.000** | 0.015 ± 0.001 | 0.071 ± 0.020 | 0.017 ± 0.002 |
| bin_outer_product_4096 | 3.864 ± 0.457 | 3.556 ± 0.497 | 2.525 ± 0.151 | **2.467 ± 0.290** | 114.411 ± 15.741 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 20.904 ± 8.783 | 9.731 ± 1.682 | **7.658 ± 0.714** | 52.605 ± 4.187 | 51.702 ± 3.212 |
| gm_queen5_5_3.wcsp | 754.410 ± 34.259 | 912.816 ± 80.887 | **653.387 ± 72.080** | 1087.168 ± 149.345 | 991.552 ± 107.995 |
| lm_batch_likelihood_brackets_4_4d | 15.374 ± 0.539 | 19.634 ± 4.301 | 15.443 ± 0.751 | **10.194 ± 0.258** | 29.046 ± 28.541 |
| lm_batch_likelihood_sentence_3_12d | 28.510 ± 3.597 | 43.803 ± 6.498 | **25.022 ± 1.238** | 43.050 ± 1.088 | 79.868 ± 24.767 |
| lm_batch_likelihood_sentence_4_4d | 11.947 ± 2.129 | 20.775 ± 2.185 | 14.543 ± 0.480 | **11.329 ± 0.673** | 26.747 ± 27.644 |
| nary_matmul_chain_64 | 0.019 ± 0.001 | **0.011 ± 0.000** | 0.021 ± 0.001 | 0.104 ± 0.013 | 0.069 ± 0.013 |
| str_matrix_chain_multiplication_100 | **4.593 ± 0.635** | 5.381 ± 0.223 | 4.730 ± 0.305 | 12.446 ± 1.231 | 13.588 ± 1.647 |
| str_mps_varying_inner_product_200 | 11.607 ± 0.746 | 10.424 ± 0.432 | **8.011 ± 0.381** | 20.546 ± 1.786 | 15.270 ± 7.145 |
| str_nw_mera_closed_120 | 309.749 ± 33.328 | 273.943 ± 14.194 | **221.676 ± 8.230** | 1424.003 ± 114.302 | 1161.809 ± 40.819 |
| str_nw_mera_open_26 | 296.054 ± 19.190 | 366.102 ± 10.530 | **200.220 ± 7.461** | 1110.606 ± 45.614 | 886.163 ± 34.477 |
| tensornetwork_permutation_focus_step409_316 | **136.456 ± 21.026** | 360.544 ± 36.494 | 254.226 ± 11.286 | 409.580 ± 25.508 | 219.329 ± 25.992 |
| tensornetwork_permutation_light_415 | **132.398 ± 15.870** | 142.338 ± 14.427 | 193.988 ± 16.801 | 344.632 ± 37.824 | 218.799 ± 59.070 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261105_174004/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261105_174004/run.yaml`
- Timestamp: `2026-11-05T17:40:04.858371+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261105_174004`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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

- `data/results/mac-cpu/cpu/einsum/20261105_174004/jax_cpu_t4_20261105_174004.log`
- `data/results/mac-cpu/cpu/einsum/20261105_174004/julia_omeinsum_t4_20261105_174004.log`
- `data/results/mac-cpu/cpu/einsum/20261105_174004/pytorch_cpu_t4_20261105_174004.log`
- `data/results/mac-cpu/cpu/einsum/20261105_174004/tenferro_eager_t4_20261105_174004.log`
- `data/results/mac-cpu/cpu/einsum/20261105_174004/tenferro_trace_t4_20261105_174004.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.753 ± 0.212 | **0.466 ± 0.010** | 0.606 ± 0.034 | 1.789 ± 0.173 | 4.883 ± 2.865 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.119 ± 0.011 | **0.078 ± 0.002** | 0.129 ± 0.005 | 0.688 ± 0.115 | 0.805 ± 0.655 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.102 ± 0.030 | **0.094 ± 0.002** | 0.171 ± 0.046 | 0.309 ± 0.053 | 0.640 ± 0.124 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.197 ± 0.047 | **0.094 ± 0.002** | 0.157 ± 0.015 | 0.291 ± 0.058 | 1.212 ± 2.140 |
| bin_elementwise_mul_2048x2048 | 1.119 ± 0.228 | 1.065 ± 0.197 | **0.968 ± 0.326** | 1.690 ± 0.106 | 1.760 ± 0.250 |
| bin_elementwise_mul_512x512 | 0.051 ± 0.012 | **0.044 ± 0.000** | 0.101 ± 0.024 | 0.435 ± 0.111 | 0.073 ± 0.044 |
| bin_matmul_1024 | 9.516 ± 2.187 | **7.022 ± 0.968** | 7.529 ± 0.599 | 14.898 ± 1.058 | 22.110 ± 10.340 |
| bin_matmul_256 | 0.153 ± 0.013 | **0.109 ± 0.003** | 0.119 ± 0.003 | 0.617 ± 0.068 | 0.387 ± 0.016 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.010 ± 0.002 | **0.004 ± 0.000** | 0.017 ± 0.001 | 0.082 ± 0.011 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.056 ± 0.004 | **0.027 ± 0.003** | 0.074 ± 0.005 | 0.159 ± 0.041 | 0.076 ± 0.016 |
| bin_omeinsum_matmul_10x10 | 0.018 ± 0.001 | **0.008 ± 0.000** | 0.021 ± 0.008 | 0.063 ± 0.006 | 0.018 ± 0.005 |
| bin_outer_product_4096 | 1.592 ± 0.723 | **1.172 ± 0.076** | 1.446 ± 0.225 | 2.524 ± 0.613 | 108.895 ± 5.929 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 12.531 ± 1.390 | 9.129 ± 0.455 | **8.754 ± 0.863** | 17.501 ± 0.429 | 29.180 ± 6.373 |
| gm_queen5_5_3.wcsp | 1873.148 ± 149.944 | 1653.693 ± 66.827 | **726.647 ± 40.512** | 1243.853 ± 61.697 | 2022.638 ± 118.552 |
| lm_batch_likelihood_brackets_4_4d | **6.960 ± 0.237** | 15.217 ± 4.880 | 14.604 ± 0.529 | 14.738 ± 1.318 | 15.180 ± 4.831 |
| lm_batch_likelihood_sentence_3_12d | **13.530 ± 0.883** | 34.836 ± 7.378 | 23.297 ± 1.873 | 22.851 ± 1.265 | 60.713 ± 29.336 |
| lm_batch_likelihood_sentence_4_4d | **7.383 ± 0.274** | 17.130 ± 2.176 | 14.263 ± 0.671 | 15.721 ± 2.519 | 33.554 ± 20.351 |
| nary_matmul_chain_64 | 0.020 ± 0.002 | **0.013 ± 0.002** | 0.022 ± 0.001 | 0.119 ± 0.036 | 0.070 ± 0.013 |
| str_matrix_chain_multiplication_100 | **3.924 ± 0.186** | 4.987 ± 0.171 | 5.256 ± 0.331 | 13.116 ± 5.130 | 12.329 ± 5.003 |
| str_mps_varying_inner_product_200 | **7.371 ± 0.269** | 8.768 ± 0.639 | 8.997 ± 0.743 | 21.105 ± 1.135 | 23.698 ± 11.752 |
| str_nw_mera_closed_120 | 311.065 ± 20.211 | 338.702 ± 13.279 | **289.670 ± 21.941** | 369.838 ± 11.224 | 389.616 ± 34.751 |
| str_nw_mera_open_26 | 241.200 ± 19.374 | 327.326 ± 18.209 | **203.641 ± 9.366** | 280.396 ± 39.411 | 340.340 ± 11.503 |
| tensornetwork_permutation_focus_step409_316 | 133.211 ± 4.626 | 246.226 ± 10.744 | 132.441 ± 9.789 | 192.398 ± 16.727 | **125.331 ± 25.730** |
| tensornetwork_permutation_light_415 | 127.523 ± 5.909 | 117.911 ± 9.160 | 124.437 ± 18.791 | 200.814 ± 14.103 | **110.221 ± 33.942** |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.753 ± 0.212 | **0.466 ± 0.010** | 0.606 ± 0.034 | 1.789 ± 0.173 | 4.883 ± 2.865 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.119 ± 0.011 | **0.078 ± 0.002** | 0.129 ± 0.005 | 0.688 ± 0.115 | 0.805 ± 0.655 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.102 ± 0.030 | **0.094 ± 0.002** | 0.171 ± 0.046 | 0.309 ± 0.053 | 0.640 ± 0.124 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.197 ± 0.047 | **0.094 ± 0.002** | 0.157 ± 0.015 | 0.291 ± 0.058 | 1.212 ± 2.140 |
| bin_elementwise_mul_2048x2048 | 1.119 ± 0.228 | 1.065 ± 0.197 | **0.968 ± 0.326** | 1.690 ± 0.106 | 1.760 ± 0.250 |
| bin_elementwise_mul_512x512 | 0.051 ± 0.012 | **0.044 ± 0.000** | 0.101 ± 0.024 | 0.435 ± 0.111 | 0.073 ± 0.044 |
| bin_matmul_1024 | 9.516 ± 2.187 | **7.022 ± 0.968** | 7.529 ± 0.599 | 14.898 ± 1.058 | 22.110 ± 10.340 |
| bin_matmul_256 | 0.153 ± 0.013 | **0.109 ± 0.003** | 0.119 ± 0.003 | 0.617 ± 0.068 | 0.387 ± 0.016 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.010 ± 0.002 | **0.004 ± 0.000** | 0.017 ± 0.001 | 0.082 ± 0.011 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.056 ± 0.004 | **0.027 ± 0.003** | 0.074 ± 0.005 | 0.159 ± 0.041 | 0.076 ± 0.016 |
| bin_omeinsum_matmul_10x10 | 0.018 ± 0.001 | **0.008 ± 0.000** | 0.021 ± 0.008 | 0.063 ± 0.006 | 0.018 ± 0.005 |
| bin_outer_product_4096 | 1.592 ± 0.723 | **1.172 ± 0.076** | 1.446 ± 0.225 | 2.524 ± 0.613 | 108.895 ± 5.929 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 12.531 ± 1.390 | 9.129 ± 0.455 | **8.754 ± 0.863** | 17.501 ± 0.429 | 29.180 ± 6.373 |
| gm_queen5_5_3.wcsp | 597.834 ± 53.901 | 557.235 ± 18.787 | **439.024 ± 91.656** | 450.031 ± 15.391 | 524.366 ± 103.055 |
| lm_batch_likelihood_brackets_4_4d | **11.013 ± 3.356** | 18.273 ± 4.448 | 14.177 ± 0.190 | 13.799 ± 0.828 | 22.665 ± 16.701 |
| lm_batch_likelihood_sentence_3_12d | **16.102 ± 2.153** | 40.206 ± 4.327 | 26.030 ± 0.762 | 23.231 ± 0.435 | 51.283 ± 13.609 |
| lm_batch_likelihood_sentence_4_4d | **8.324 ± 3.200** | 20.487 ± 1.233 | 15.345 ± 0.410 | 14.926 ± 1.074 | 26.603 ± 22.577 |
| nary_matmul_chain_64 | 0.020 ± 0.002 | **0.013 ± 0.002** | 0.022 ± 0.001 | 0.119 ± 0.036 | 0.070 ± 0.013 |
| str_matrix_chain_multiplication_100 | **4.257 ± 0.384** | 5.124 ± 0.141 | 4.956 ± 0.562 | 11.881 ± 1.178 | 12.038 ± 5.093 |
| str_mps_varying_inner_product_200 | **10.025 ± 1.406** | 10.524 ± 1.683 | 12.405 ± 0.780 | 23.088 ± 0.522 | 14.316 ± 8.940 |
| str_nw_mera_closed_120 | 288.210 ± 28.107 | 264.905 ± 29.842 | **241.370 ± 10.777** | 313.987 ± 23.803 | 417.274 ± 30.653 |
| str_nw_mera_open_26 | 278.574 ± 14.416 | 325.321 ± 15.495 | **208.418 ± 7.713** | 245.702 ± 4.255 | 416.333 ± 21.773 |
| tensornetwork_permutation_focus_step409_316 | 133.211 ± 4.626 | 246.226 ± 10.744 | 132.441 ± 9.789 | 192.398 ± 16.727 | **125.331 ± 25.730** |
| tensornetwork_permutation_light_415 | 127.523 ± 5.909 | 117.911 ± 9.160 | 124.437 ± 18.791 | 200.814 ± 14.103 | **110.221 ± 33.942** |

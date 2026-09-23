# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `8a1839febeb3c868a502e26397ca10761bbc568d`
- Benchmark commit: `5df994b8aee8e926b52e885ab89a5732e4e2ff8f`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260923_201832/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260923_201832/run.yaml`
- Timestamp: `20260923_201832`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260923_201832`.

- tenferro-rs commit: `8a1839febeb3c868a502e26397ca10761bbc568d`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20260923_201832/einsum_table_t1_20260923_201832.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260923_201832/tenferro_trace_t1_20260923_201832.log`
- `data/results/mac-cpu/cpu/einsum/20260923_201832/tenferro_eager_t1_20260923_201832.log`
- `data/results/mac-cpu/cpu/einsum/20260923_201832/pytorch_cpu_t1_20260923_201832.log`
- `data/results/mac-cpu/cpu/einsum/20260923_201832/jax_cpu_t1_20260923_201832.log`
- `data/results/mac-cpu/cpu/einsum/20260923_201832/julia_omeinsum_t1_20260923_201832.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.039 ± 0.112 | 0.562 ± 0.060 | **0.386 ± 0.006** | 2.248 ± 0.137 | 3.757 ± 0.828 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.192 ± 0.004 | **0.084 ± 0.004** | 0.087 ± 0.003 | 0.342 ± 0.008 | 0.875 ± 1.220 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.106 ± 0.003** | 0.113 ± 0.004 | 0.112 ± 0.006 | 0.172 ± 0.007 | 0.335 ± 0.510 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.434 ± 0.035 | **0.107 ± 0.003** | 0.115 ± 0.001 | 0.200 ± 0.005 | 0.629 ± 0.210 |
| bin_elementwise_mul_2048x2048 | 0.939 ± 0.056 | 1.024 ± 0.060 | **0.854 ± 0.046** | 1.505 ± 0.061 | 0.913 ± 0.309 |
| bin_matmul_1024 | 6.168 ± 0.389 | 10.008 ± 0.113 | **4.577 ± 0.128** | 34.879 ± 0.056 | 39.903 ± 5.234 |
| bin_matmul_256 | 0.127 ± 0.006 | **0.085 ± 0.001** | 0.088 ± 0.001 | 0.611 ± 0.016 | 0.812 ± 0.136 |
| bin_outer_product_4096 | 1.034 ± 0.055 | **1.007 ± 0.007** | 1.521 ± 0.054 | 1.080 ± 0.067 | 60.365 ± 41.064 |
| gm_queen5_5_3.wcsp | **639.581 ± 4.074** | 1051.874 ± 2.493 | 915.479 ± 6.651 | 1260.510 ± 3.906 | 1340.976 ± 91.828 |
| lm_batch_likelihood_brackets_4_4d | 11.372 ± 0.052 | 9.396 ± 0.152 | 8.547 ± 0.147 | **7.285 ± 0.204** | 22.080 ± 42.448 |
| lm_batch_likelihood_sentence_3_12d | 17.676 ± 0.365 | 17.045 ± 0.183 | **16.731 ± 0.166** | 29.002 ± 0.190 | 82.941 ± 42.494 |
| lm_batch_likelihood_sentence_4_4d | 12.099 ± 0.091 | 10.619 ± 0.089 | 8.814 ± 0.084 | **8.351 ± 0.084** | 17.950 ± 41.107 |
| nary_matmul_chain_64 | 0.011 ± 0.000 | **0.009 ± 0.000** | 0.013 ± 0.001 | 0.056 ± 0.004 | 0.042 ± 0.007 |
| str_matrix_chain_multiplication_100 | **1.939 ± 0.075** | 1.957 ± 0.059 | 2.086 ± 0.026 | 7.980 ± 0.086 | 8.955 ± 1.718 |
| str_mps_varying_inner_product_200 | 4.445 ± 0.052 | 4.516 ± 0.060 | **3.595 ± 0.046** | 12.096 ± 0.071 | 10.657 ± 5.224 |
| str_nw_mera_closed_120 | 312.151 ± 1.509 | 311.697 ± 1.539 | **156.906 ± 1.423** | 836.560 ± 2.478 | 962.492 ± 51.812 |
| str_nw_mera_open_26 | 233.749 ± 1.977 | 173.928 ± 1.629 | **128.312 ± 0.814** | 622.674 ± 1.517 | 709.730 ± 50.941 |
| tensornetwork_permutation_focus_step409_316 | **75.547 ± 0.412** | 181.905 ± 3.202 | 154.007 ± 0.806 | 201.909 ± 1.716 | 179.243 ± 47.183 |
| tensornetwork_permutation_light_415 | **76.334 ± 1.055** | 96.242 ± 0.895 | 132.721 ± 1.150 | 205.045 ± 2.252 | 175.879 ± 2.299 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.039 ± 0.112 | 0.562 ± 0.060 | **0.386 ± 0.006** | 2.248 ± 0.137 | 3.757 ± 0.828 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.192 ± 0.004 | **0.084 ± 0.004** | 0.087 ± 0.003 | 0.342 ± 0.008 | 0.875 ± 1.220 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.106 ± 0.003** | 0.113 ± 0.004 | 0.112 ± 0.006 | 0.172 ± 0.007 | 0.335 ± 0.510 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.434 ± 0.035 | **0.107 ± 0.003** | 0.115 ± 0.001 | 0.200 ± 0.005 | 0.629 ± 0.210 |
| bin_elementwise_mul_2048x2048 | 0.939 ± 0.056 | 1.024 ± 0.060 | **0.854 ± 0.046** | 1.505 ± 0.061 | 0.913 ± 0.309 |
| bin_matmul_1024 | 6.168 ± 0.389 | 10.008 ± 0.113 | **4.577 ± 0.128** | 34.879 ± 0.056 | 39.903 ± 5.234 |
| bin_matmul_256 | 0.127 ± 0.006 | **0.085 ± 0.001** | 0.088 ± 0.001 | 0.611 ± 0.016 | 0.812 ± 0.136 |
| bin_outer_product_4096 | 1.034 ± 0.055 | **1.007 ± 0.007** | 1.521 ± 0.054 | 1.080 ± 0.067 | 60.365 ± 41.064 |
| gm_queen5_5_3.wcsp | **138.828 ± 1.977** | 285.326 ± 0.972 | 315.241 ± 1.736 | 636.726 ± 1.232 | 762.801 ± 8.017 |
| lm_batch_likelihood_brackets_4_4d | 10.176 ± 0.251 | 9.902 ± 0.234 | 8.923 ± 0.322 | **7.007 ± 0.040** | 18.619 ± 47.830 |
| lm_batch_likelihood_sentence_3_12d | 18.486 ± 0.242 | 18.684 ± 0.228 | **17.125 ± 0.217** | 29.967 ± 0.166 | 83.347 ± 43.194 |
| lm_batch_likelihood_sentence_4_4d | 11.768 ± 0.198 | 11.378 ± 0.586 | 8.552 ± 0.157 | **7.672 ± 0.090** | 18.502 ± 47.634 |
| nary_matmul_chain_64 | 0.011 ± 0.000 | **0.009 ± 0.000** | 0.013 ± 0.001 | 0.056 ± 0.004 | 0.042 ± 0.007 |
| str_matrix_chain_multiplication_100 | **1.933 ± 0.042** | 2.011 ± 0.075 | 2.109 ± 0.038 | 8.086 ± 0.222 | 9.291 ± 0.723 |
| str_mps_varying_inner_product_200 | 5.409 ± 0.127 | 5.345 ± 0.124 | **4.301 ± 0.164** | 13.006 ± 0.206 | 9.419 ± 6.195 |
| str_nw_mera_closed_120 | 182.926 ± 0.403 | 181.400 ± 0.545 | **130.586 ± 0.168** | 814.695 ± 1.236 | 939.781 ± 46.818 |
| str_nw_mera_open_26 | 237.855 ± 2.384 | 174.520 ± 0.730 | **133.354 ± 0.505** | 631.340 ± 0.616 | 711.559 ± 53.936 |
| tensornetwork_permutation_focus_step409_316 | **75.547 ± 0.412** | 181.905 ± 3.202 | 154.007 ± 0.806 | 201.909 ± 1.716 | 179.243 ± 47.183 |
| tensornetwork_permutation_light_415 | **76.334 ± 1.055** | 96.242 ± 0.895 | 132.721 ± 1.150 | 205.045 ± 2.252 | 175.879 ± 2.299 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260923_204226/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260923_204226/run.yaml`
- Timestamp: `20260923_204226`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260923_204226`.

- tenferro-rs commit: `8a1839febeb3c868a502e26397ca10761bbc568d`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20260923_204226/einsum_table_t4_20260923_204226.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260923_204226/tenferro_trace_t4_20260923_204226.log`
- `data/results/mac-cpu/cpu/einsum/20260923_204226/tenferro_eager_t4_20260923_204226.log`
- `data/results/mac-cpu/cpu/einsum/20260923_204226/pytorch_cpu_t4_20260923_204226.log`
- `data/results/mac-cpu/cpu/einsum/20260923_204226/jax_cpu_t4_20260923_204226.log`
- `data/results/mac-cpu/cpu/einsum/20260923_204226/julia_omeinsum_t4_20260923_204226.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.859 ± 0.125 | 0.594 ± 0.064 | **0.388 ± 0.004** | 0.649 ± 0.065 | 2.011 ± 3.362 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.200 ± 0.012 | **0.084 ± 0.002** | 0.087 ± 0.001 | 0.154 ± 0.016 | 0.630 ± 0.161 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.083 ± 0.012 | 0.059 ± 0.015 | **0.057 ± 0.003** | 0.115 ± 0.009 | 0.320 ± 0.025 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.163 ± 0.016 | **0.049 ± 0.014** | 0.054 ± 0.002 | 0.120 ± 0.014 | 0.666 ± 0.460 |
| bin_elementwise_mul_2048x2048 | 0.619 ± 0.027 | **0.587 ± 0.038** | 0.593 ± 0.015 | 0.755 ± 0.051 | 0.977 ± 0.583 |
| bin_matmul_1024 | 5.075 ± 0.242 | 10.191 ± 1.154 | **4.925 ± 0.148** | 9.093 ± 0.079 | 13.825 ± 5.640 |
| bin_matmul_256 | 0.155 ± 0.009 | **0.083 ± 0.001** | 0.088 ± 0.000 | 0.228 ± 0.008 | 0.311 ± 0.042 |
| bin_outer_product_4096 | 0.587 ± 0.022 | **0.564 ± 0.037** | 0.566 ± 0.016 | 0.633 ± 0.024 | 68.300 ± 13.330 |
| gm_queen5_5_3.wcsp | 482.433 ± 18.987 | 486.183 ± 35.972 | **452.716 ± 5.144** | 666.748 ± 12.080 | 1040.648 ± 6.179 |
| lm_batch_likelihood_brackets_4_4d | **6.304 ± 0.132** | 8.400 ± 0.119 | 6.604 ± 0.112 | 7.059 ± 0.169 | 9.857 ± 1.503 |
| lm_batch_likelihood_sentence_3_12d | **8.354 ± 0.085** | 13.037 ± 0.487 | 16.141 ± 0.115 | 11.912 ± 0.107 | 24.729 ± 17.118 |
| lm_batch_likelihood_sentence_4_4d | **6.635 ± 0.094** | 9.361 ± 0.095 | 6.746 ± 0.076 | 7.243 ± 0.204 | 14.795 ± 15.192 |
| nary_matmul_chain_64 | 0.032 ± 0.004 | **0.009 ± 0.000** | 0.013 ± 0.001 | 0.064 ± 0.008 | 0.043 ± 0.002 |
| str_matrix_chain_multiplication_100 | 2.015 ± 0.061 | **1.975 ± 0.046** | 2.127 ± 0.082 | 8.007 ± 0.221 | 5.320 ± 0.381 |
| str_mps_varying_inner_product_200 | 5.143 ± 0.138 | 5.115 ± 0.070 | **3.682 ± 0.046** | 16.241 ± 0.252 | 9.878 ± 4.126 |
| str_nw_mera_closed_120 | 146.582 ± 0.871 | 146.526 ± 1.173 | **136.376 ± 1.174** | 251.913 ± 3.334 | 309.995 ± 19.416 |
| str_nw_mera_open_26 | 146.411 ± 2.972 | 144.520 ± 1.882 | **128.892 ± 9.454** | 188.749 ± 1.249 | 266.296 ± 19.420 |
| tensornetwork_permutation_focus_step409_316 | **54.930 ± 0.426** | 79.518 ± 15.640 | 74.771 ± 9.410 | 90.391 ± 0.647 | 100.567 ± 21.491 |
| tensornetwork_permutation_light_415 | **55.268 ± 0.310** | 64.407 ± 1.953 | 70.258 ± 8.290 | 92.014 ± 1.542 | 92.139 ± 19.609 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.859 ± 0.125 | 0.594 ± 0.064 | **0.388 ± 0.004** | 0.649 ± 0.065 | 2.011 ± 3.362 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.200 ± 0.012 | **0.084 ± 0.002** | 0.087 ± 0.001 | 0.154 ± 0.016 | 0.630 ± 0.161 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.083 ± 0.012 | 0.059 ± 0.015 | **0.057 ± 0.003** | 0.115 ± 0.009 | 0.320 ± 0.025 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.163 ± 0.016 | **0.049 ± 0.014** | 0.054 ± 0.002 | 0.120 ± 0.014 | 0.666 ± 0.460 |
| bin_elementwise_mul_2048x2048 | 0.619 ± 0.027 | **0.587 ± 0.038** | 0.593 ± 0.015 | 0.755 ± 0.051 | 0.977 ± 0.583 |
| bin_matmul_1024 | 5.075 ± 0.242 | 10.191 ± 1.154 | **4.925 ± 0.148** | 9.093 ± 0.079 | 13.825 ± 5.640 |
| bin_matmul_256 | 0.155 ± 0.009 | **0.083 ± 0.001** | 0.088 ± 0.000 | 0.228 ± 0.008 | 0.311 ± 0.042 |
| bin_outer_product_4096 | 0.587 ± 0.022 | **0.564 ± 0.037** | 0.566 ± 0.016 | 0.633 ± 0.024 | 68.300 ± 13.330 |
| gm_queen5_5_3.wcsp | **126.826 ± 6.214** | 168.141 ± 16.897 | 185.303 ± 3.062 | 258.400 ± 4.531 | 412.850 ± 10.933 |
| lm_batch_likelihood_brackets_4_4d | 7.023 ± 0.156 | 8.843 ± 0.252 | 7.676 ± 0.106 | **6.752 ± 0.182** | 14.811 ± 15.598 |
| lm_batch_likelihood_sentence_3_12d | **9.250 ± 0.156** | 13.043 ± 0.099 | 16.606 ± 0.161 | 12.315 ± 0.071 | 24.631 ± 11.714 |
| lm_batch_likelihood_sentence_4_4d | 7.038 ± 0.058 | 9.195 ± 0.103 | 7.557 ± 0.146 | **6.649 ± 0.106** | 13.838 ± 16.106 |
| nary_matmul_chain_64 | 0.032 ± 0.004 | **0.009 ± 0.000** | 0.013 ± 0.001 | 0.064 ± 0.008 | 0.043 ± 0.002 |
| str_matrix_chain_multiplication_100 | **1.971 ± 0.039** | 2.017 ± 0.022 | 2.153 ± 0.035 | 7.973 ± 0.292 | 5.464 ± 0.825 |
| str_mps_varying_inner_product_200 | 6.088 ± 0.141 | 6.278 ± 0.090 | **5.032 ± 0.098** | 17.155 ± 0.255 | 8.877 ± 1.091 |
| str_nw_mera_closed_120 | 131.790 ± 2.030 | 128.959 ± 0.601 | **122.679 ± 0.444** | 250.847 ± 4.155 | 296.146 ± 17.596 |
| str_nw_mera_open_26 | 145.692 ± 4.680 | 141.752 ± 1.516 | **130.691 ± 5.601** | 192.897 ± 0.757 | 249.873 ± 22.842 |
| tensornetwork_permutation_focus_step409_316 | **54.930 ± 0.426** | 79.518 ± 15.640 | 74.771 ± 9.410 | 90.391 ± 0.647 | 100.567 ± 21.491 |
| tensornetwork_permutation_light_415 | **55.268 ± 0.310** | 64.407 ± 1.953 | 70.258 ± 8.290 | 92.014 ± 1.542 | 92.139 ± 19.609 |

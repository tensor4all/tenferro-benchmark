# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `9b03dd718025b3350d93eaab13a8b93dcc66cab2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261126_172406/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261126_172406/run.yaml`
- Timestamp: `2026-11-26T17:24:06.847382+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261126_172406`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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

- tenferro-rs features: `native`

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

- `data/results/mac-cpu/cpu/einsum/20261126_172406/jax_cpu_t1_20261126_172406.log`
- `data/results/mac-cpu/cpu/einsum/20261126_172406/julia_omeinsum_t1_20261126_172406.log`
- `data/results/mac-cpu/cpu/einsum/20261126_172406/pytorch_cpu_t1_20261126_172406.log`
- `data/results/mac-cpu/cpu/einsum/20261126_172406/tenferro_eager_t1_20261126_172406.log`
- `data/results/mac-cpu/cpu/einsum/20261126_172406/tenferro_trace_t1_20261126_172406.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.767 ± 2.246 | 3.580 ± 3.055 | **0.728 ± 0.088** | 4.109 ± 0.232 | 4.705 ± 0.720 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.579 ± 0.045 | 0.472 ± 0.029 | **0.128 ± 0.008** | 0.818 ± 0.086 | 1.096 ± 3.482 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.254 ± 0.517 | 0.224 ± 0.012 | **0.205 ± 0.022** | 0.466 ± 0.140 | 1.143 ± 1.180 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.927 ± 0.506 | 0.239 ± 0.038 | **0.217 ± 0.025** | 0.494 ± 0.138 | 1.487 ± 4.662 |
| bin_elementwise_mul_2048x2048 | 3.314 ± 2.344 | **2.206 ± 0.216** | 2.282 ± 0.122 | 3.288 ± 0.313 | 3.058 ± 1.109 |
| bin_elementwise_mul_512x512 | 0.102 ± 0.215 | **0.077 ± 0.005** | 0.084 ± 0.012 | 0.307 ± 0.037 | 0.288 ± 0.093 |
| bin_matmul_1024 | 91.826 ± 18.189 | 58.501 ± 6.301 | **12.187 ± 1.670** | 54.827 ± 3.496 | 84.418 ± 16.562 |
| bin_matmul_256 | 0.925 ± 0.911 | 0.810 ± 0.041 | **0.120 ± 0.004** | 0.958 ± 0.091 | 1.011 ± 0.408 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | **0.009 ± 0.002** | 0.011 ± 0.000 | 0.019 ± 0.002 | 0.091 ± 0.081 | **0.009 ± 0.003** |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.025 ± 0.004** | 0.067 ± 0.003 | 0.079 ± 0.007 | 0.147 ± 0.025 | 0.097 ± 0.257 |
| bin_omeinsum_matmul_10x10 | 0.045 ± 0.010 | 0.021 ± 0.000 | **0.017 ± 0.001** | 0.082 ± 0.019 | 0.032 ± 0.017 |
| bin_outer_product_4096 | 9.458 ± 4.808 | 5.452 ± 3.212 | 3.473 ± 0.154 | **2.508 ± 0.251** | 262.959 ± 43.851 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 171.530 ± 30.433 | 104.056 ± 6.500 | **10.988 ± 0.247** | 56.931 ± 0.377 | 93.302 ± 25.904 |
| gm_queen5_5_3.wcsp | 9062.914 ± 1253.825 | 5794.859 ± 297.013 | **4071.033 ± 283.876** | 6270.074 ± 409.771 | 7436.320 ± 1092.483 |
| lm_batch_likelihood_brackets_4_4d | 37.945 ± 17.711 | 41.185 ± 15.258 | **14.166 ± 1.183** | 14.266 ± 2.291 | 28.811 ± 10.252 |
| lm_batch_likelihood_sentence_3_12d | 130.173 ± 24.494 | 106.727 ± 30.157 | **26.794 ± 1.968** | 59.903 ± 3.064 | 73.561 ± 28.460 |
| lm_batch_likelihood_sentence_4_4d | 38.488 ± 10.349 | 29.429 ± 3.999 | 15.185 ± 2.361 | **14.264 ± 0.658** | 20.335 ± 1.294 |
| nary_matmul_chain_64 | 0.040 ± 0.009 | 0.047 ± 0.006 | **0.022 ± 0.002** | 0.127 ± 0.008 | 0.101 ± 0.164 |
| str_matrix_chain_multiplication_100 | 22.681 ± 7.466 | 14.084 ± 2.891 | **5.414 ± 1.373** | 14.977 ± 1.083 | 14.009 ± 1.057 |
| str_mps_varying_inner_product_200 | 29.975 ± 13.427 | 18.503 ± 1.969 | **8.220 ± 2.357** | 18.563 ± 0.555 | 16.337 ± 0.644 |
| str_nw_mera_closed_120 | 3653.543 ± 446.483 | 2409.964 ± 742.260 | **375.070 ± 62.448** | 1725.422 ± 181.959 | 2175.187 ± 823.005 |
| str_nw_mera_open_26 | 2150.697 ± 194.962 | 1837.643 ± 279.885 | **235.860 ± 11.265** | 1262.796 ± 189.945 | 1134.999 ± 230.697 |
| tensornetwork_permutation_focus_step409_316 | 410.484 ± 101.779 | 476.217 ± 114.573 | 263.253 ± 24.845 | 378.899 ± 31.602 | **229.897 ± 42.173** |
| tensornetwork_permutation_light_415 | 421.961 ± 79.953 | 286.649 ± 35.888 | 223.997 ± 42.213 | 414.196 ± 23.402 | **211.421 ± 6.726** |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.767 ± 2.246 | 3.580 ± 3.055 | **0.728 ± 0.088** | 4.109 ± 0.232 | 4.705 ± 0.720 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.579 ± 0.045 | 0.472 ± 0.029 | **0.128 ± 0.008** | 0.818 ± 0.086 | 1.096 ± 3.482 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.254 ± 0.517 | 0.224 ± 0.012 | **0.205 ± 0.022** | 0.466 ± 0.140 | 1.143 ± 1.180 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.927 ± 0.506 | 0.239 ± 0.038 | **0.217 ± 0.025** | 0.494 ± 0.138 | 1.487 ± 4.662 |
| bin_elementwise_mul_2048x2048 | 3.314 ± 2.344 | **2.206 ± 0.216** | 2.282 ± 0.122 | 3.288 ± 0.313 | 3.058 ± 1.109 |
| bin_elementwise_mul_512x512 | 0.102 ± 0.215 | **0.077 ± 0.005** | 0.084 ± 0.012 | 0.307 ± 0.037 | 0.288 ± 0.093 |
| bin_matmul_1024 | 91.826 ± 18.189 | 58.501 ± 6.301 | **12.187 ± 1.670** | 54.827 ± 3.496 | 84.418 ± 16.562 |
| bin_matmul_256 | 0.925 ± 0.911 | 0.810 ± 0.041 | **0.120 ± 0.004** | 0.958 ± 0.091 | 1.011 ± 0.408 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | **0.009 ± 0.002** | 0.011 ± 0.000 | 0.019 ± 0.002 | 0.091 ± 0.081 | **0.009 ± 0.003** |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.025 ± 0.004** | 0.067 ± 0.003 | 0.079 ± 0.007 | 0.147 ± 0.025 | 0.097 ± 0.257 |
| bin_omeinsum_matmul_10x10 | 0.045 ± 0.010 | 0.021 ± 0.000 | **0.017 ± 0.001** | 0.082 ± 0.019 | 0.032 ± 0.017 |
| bin_outer_product_4096 | 9.458 ± 4.808 | 5.452 ± 3.212 | 3.473 ± 0.154 | **2.508 ± 0.251** | 262.959 ± 43.851 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 171.530 ± 30.433 | 104.056 ± 6.500 | **10.988 ± 0.247** | 56.931 ± 0.377 | 93.302 ± 25.904 |
| gm_queen5_5_3.wcsp | 2444.414 ± 658.455 | 1720.553 ± 220.387 | **886.557 ± 100.879** | 1242.574 ± 91.598 | 2527.670 ± 554.879 |
| lm_batch_likelihood_brackets_4_4d | 36.444 ± 5.968 | 28.012 ± 7.872 | 15.300 ± 1.405 | **13.371 ± 1.813** | 29.385 ± 10.870 |
| lm_batch_likelihood_sentence_3_12d | 124.431 ± 50.008 | 102.039 ± 19.220 | **26.722 ± 2.299** | 57.028 ± 1.848 | 98.413 ± 46.261 |
| lm_batch_likelihood_sentence_4_4d | 35.654 ± 8.302 | 26.791 ± 0.339 | 15.229 ± 1.608 | **13.805 ± 0.386** | 37.955 ± 39.575 |
| nary_matmul_chain_64 | 0.040 ± 0.009 | 0.047 ± 0.006 | **0.022 ± 0.002** | 0.127 ± 0.008 | 0.101 ± 0.164 |
| str_matrix_chain_multiplication_100 | 19.938 ± 5.652 | 14.235 ± 0.422 | **6.432 ± 1.037** | 14.517 ± 1.052 | 17.695 ± 6.475 |
| str_mps_varying_inner_product_200 | 33.728 ± 7.837 | 22.142 ± 1.024 | **10.055 ± 0.776** | 22.466 ± 0.774 | 18.904 ± 3.622 |
| str_nw_mera_closed_120 | 2757.901 ± 364.052 | 2224.700 ± 215.514 | **288.254 ± 12.518** | 1619.042 ± 345.165 | 1370.268 ± 197.549 |
| str_nw_mera_open_26 | 1950.425 ± 144.074 | 1621.016 ± 72.995 | **243.589 ± 17.669** | 1729.956 ± 299.666 | 2052.523 ± 829.709 |
| tensornetwork_permutation_focus_step409_316 | 410.484 ± 101.779 | 476.217 ± 114.573 | 263.253 ± 24.845 | 378.899 ± 31.602 | **229.897 ± 42.173** |
| tensornetwork_permutation_light_415 | 421.961 ± 79.953 | 286.649 ± 35.888 | 223.997 ± 42.213 | 414.196 ± 23.402 | **211.421 ± 6.726** |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261126_180245/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261126_180245/run.yaml`
- Timestamp: `2026-11-26T18:02:45.132485+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261126_180245`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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

- tenferro-rs features: `native`

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

- `data/results/mac-cpu/cpu/einsum/20261126_180245/jax_cpu_t4_20261126_180245.log`
- `data/results/mac-cpu/cpu/einsum/20261126_180245/julia_omeinsum_t4_20261126_180245.log`
- `data/results/mac-cpu/cpu/einsum/20261126_180245/pytorch_cpu_t4_20261126_180245.log`
- `data/results/mac-cpu/cpu/einsum/20261126_180245/tenferro_eager_t4_20261126_180245.log`
- `data/results/mac-cpu/cpu/einsum/20261126_180245/tenferro_trace_t4_20261126_180245.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.815 ± 0.897 | 2.555 ± 0.113 | **0.739 ± 0.035** | 2.809 ± 0.119 | 93.984 ± 32.290 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.906 ± 0.329 | **0.400 ± 0.019** | 0.466 ± 0.345 | 0.747 ± 0.047 | 1.475 ± 1.988 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.243 ± 0.093 | **0.158 ± 0.011** | 0.240 ± 0.023 | 0.534 ± 0.036 | 0.878 ± 0.422 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.229 ± 0.023 | **0.162 ± 0.005** | 0.233 ± 0.106 | 0.552 ± 0.109 | 1.769 ± 0.708 |
| bin_elementwise_mul_2048x2048 | 3.522 ± 1.180 | **2.029 ± 0.246** | 2.764 ± 0.097 | 3.624 ± 0.108 | 2.921 ± 0.964 |
| bin_elementwise_mul_512x512 | 0.242 ± 0.142 | **0.058 ± 0.015** | 0.132 ± 0.024 | 0.427 ± 0.050 | 0.079 ± 0.013 |
| bin_matmul_1024 | 31.502 ± 5.516 | 18.178 ± 1.027 | **13.497 ± 0.236** | 26.293 ± 2.500 | 46.713 ± 5.926 |
| bin_matmul_256 | 0.673 ± 0.209 | 0.478 ± 0.053 | **0.128 ± 0.002** | 0.901 ± 0.047 | 3.043 ± 1.326 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.002 | **0.004 ± 0.000** | 0.018 ± 0.002 | 0.086 ± 0.008 | 0.010 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.027 ± 0.002 | **0.022 ± 0.002** | 0.073 ± 0.002 | 0.184 ± 0.004 | 0.088 ± 0.013 |
| bin_omeinsum_matmul_10x10 | 0.014 ± 0.003 | **0.007 ± 0.000** | 0.016 ± 0.000 | 0.152 ± 0.006 | 0.039 ± 0.007 |
| bin_outer_product_4096 | 3.101 ± 2.636 | **2.181 ± 0.367** | 3.327 ± 0.097 | 3.270 ± 0.145 | 219.126 ± 16.304 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 59.701 ± 9.252 | 42.669 ± 8.930 | **10.738 ± 0.483** | 32.632 ± 8.138 | 37.486 ± 4.179 |
| gm_queen5_5_3.wcsp | **2395.439 ± 189.052** | 5072.070 ± 616.165 | 3518.133 ± 399.713 | 4420.928 ± 659.459 | 10651.092 ± 2572.649 |
| lm_batch_likelihood_brackets_4_4d | 15.448 ± 5.854 | 29.290 ± 17.187 | **13.926 ± 1.143** | 23.010 ± 1.295 | 152.729 ± 79.054 |
| lm_batch_likelihood_sentence_3_12d | 42.064 ± 12.895 | 68.341 ± 10.358 | **26.567 ± 0.986** | 44.510 ± 9.520 | 93.856 ± 12.411 |
| lm_batch_likelihood_sentence_4_4d | 21.509 ± 1.733 | 37.654 ± 10.074 | **14.086 ± 0.836** | 23.100 ± 1.311 | 81.256 ± 14.900 |
| nary_matmul_chain_64 | 0.049 ± 0.009 | 0.075 ± 0.016 | **0.024 ± 0.004** | 0.198 ± 0.033 | 0.060 ± 0.014 |
| str_matrix_chain_multiplication_100 | 9.013 ± 1.991 | 16.521 ± 2.111 | **6.566 ± 0.920** | 17.760 ± 1.103 | 305.127 ± 108.081 |
| str_mps_varying_inner_product_200 | 24.154 ± 11.992 | 40.122 ± 1.678 | **13.891 ± 1.596** | 32.338 ± 3.392 | 21.376 ± 8.277 |
| str_nw_mera_closed_120 | 1279.720 ± 96.073 | 1346.928 ± 123.616 | **367.253 ± 39.843** | 664.828 ± 68.125 | 978.278 ± 257.042 |
| str_nw_mera_open_26 | 785.916 ± 52.785 | 887.718 ± 164.605 | **233.183 ± 6.770** | 558.280 ± 59.419 | 782.334 ± 82.670 |
| tensornetwork_permutation_focus_step409_316 | **133.399 ± 10.089** | 173.530 ± 5.745 | 184.701 ± 43.737 | 285.496 ± 50.979 | 438.628 ± 70.956 |
| tensornetwork_permutation_light_415 | 172.857 ± 46.652 | **137.398 ± 10.451** | 148.432 ± 16.581 | 276.873 ± 35.743 | 454.789 ± 96.721 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.815 ± 0.897 | 2.555 ± 0.113 | **0.739 ± 0.035** | 2.809 ± 0.119 | 93.984 ± 32.290 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.906 ± 0.329 | **0.400 ± 0.019** | 0.466 ± 0.345 | 0.747 ± 0.047 | 1.475 ± 1.988 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.243 ± 0.093 | **0.158 ± 0.011** | 0.240 ± 0.023 | 0.534 ± 0.036 | 0.878 ± 0.422 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.229 ± 0.023 | **0.162 ± 0.005** | 0.233 ± 0.106 | 0.552 ± 0.109 | 1.769 ± 0.708 |
| bin_elementwise_mul_2048x2048 | 3.522 ± 1.180 | **2.029 ± 0.246** | 2.764 ± 0.097 | 3.624 ± 0.108 | 2.921 ± 0.964 |
| bin_elementwise_mul_512x512 | 0.242 ± 0.142 | **0.058 ± 0.015** | 0.132 ± 0.024 | 0.427 ± 0.050 | 0.079 ± 0.013 |
| bin_matmul_1024 | 31.502 ± 5.516 | 18.178 ± 1.027 | **13.497 ± 0.236** | 26.293 ± 2.500 | 46.713 ± 5.926 |
| bin_matmul_256 | 0.673 ± 0.209 | 0.478 ± 0.053 | **0.128 ± 0.002** | 0.901 ± 0.047 | 3.043 ± 1.326 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.002 | **0.004 ± 0.000** | 0.018 ± 0.002 | 0.086 ± 0.008 | 0.010 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.027 ± 0.002 | **0.022 ± 0.002** | 0.073 ± 0.002 | 0.184 ± 0.004 | 0.088 ± 0.013 |
| bin_omeinsum_matmul_10x10 | 0.014 ± 0.003 | **0.007 ± 0.000** | 0.016 ± 0.000 | 0.152 ± 0.006 | 0.039 ± 0.007 |
| bin_outer_product_4096 | 3.101 ± 2.636 | **2.181 ± 0.367** | 3.327 ± 0.097 | 3.270 ± 0.145 | 219.126 ± 16.304 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 59.701 ± 9.252 | 42.669 ± 8.930 | **10.738 ± 0.483** | 32.632 ± 8.138 | 37.486 ± 4.179 |
| gm_queen5_5_3.wcsp | 1039.249 ± 94.628 | 1087.856 ± 185.660 | **599.645 ± 73.103** | 801.883 ± 189.107 | 1976.440 ± 370.476 |
| lm_batch_likelihood_brackets_4_4d | **15.113 ± 1.492** | 24.913 ± 2.402 | 17.681 ± 0.309 | 20.520 ± 0.925 | 196.944 ± 112.955 |
| lm_batch_likelihood_sentence_3_12d | 40.954 ± 0.975 | 74.714 ± 8.735 | **29.561 ± 1.328** | 42.405 ± 21.855 | 122.280 ± 46.865 |
| lm_batch_likelihood_sentence_4_4d | **14.231 ± 0.539** | 25.623 ± 2.766 | 18.349 ± 0.939 | 21.916 ± 3.037 | 108.889 ± 44.610 |
| nary_matmul_chain_64 | 0.049 ± 0.009 | 0.075 ± 0.016 | **0.024 ± 0.004** | 0.198 ± 0.033 | 0.060 ± 0.014 |
| str_matrix_chain_multiplication_100 | 10.101 ± 7.449 | 16.690 ± 1.985 | **5.765 ± 0.473** | 17.637 ± 0.374 | 262.456 ± 14.317 |
| str_mps_varying_inner_product_200 | 34.107 ± 11.194 | 47.347 ± 7.753 | **12.238 ± 1.542** | 34.079 ± 3.138 | 45.725 ± 20.026 |
| str_nw_mera_closed_120 | 1095.432 ± 136.204 | 1273.690 ± 133.051 | **295.092 ± 38.078** | 617.178 ± 83.150 | 786.785 ± 184.449 |
| str_nw_mera_open_26 | 635.119 ± 30.409 | 939.705 ± 158.930 | **243.684 ± 4.872** | 544.513 ± 51.013 | 855.030 ± 203.652 |
| tensornetwork_permutation_focus_step409_316 | **133.399 ± 10.089** | 173.530 ± 5.745 | 184.701 ± 43.737 | 285.496 ± 50.979 | 438.628 ± 70.956 |
| tensornetwork_permutation_light_415 | 172.857 ± 46.652 | **137.398 ± 10.451** | 148.432 ± 16.581 | 276.873 ± 35.743 | 454.789 ± 96.721 |

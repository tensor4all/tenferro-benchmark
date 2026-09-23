# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `412852a8cfb13b14246443fc09b1c0c34f5ce1cd`
- Benchmark commit: `9f63fb7704feec767fda97251865741714f3926f`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260923_173045/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260923_173045/run.yaml`
- Timestamp: `20260923_173045`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260923_173045`.

- tenferro-rs commit: `412852a8cfb13b14246443fc09b1c0c34f5ce1cd`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20260923_173045/einsum_table_t1_20260923_173045.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260923_173045/tenferro_trace_t1_20260923_173045.log`
- `data/results/mac-cpu/cpu/einsum/20260923_173045/tenferro_eager_t1_20260923_173045.log`
- `data/results/mac-cpu/cpu/einsum/20260923_173045/pytorch_cpu_t1_20260923_173045.log`
- `data/results/mac-cpu/cpu/einsum/20260923_173045/jax_cpu_t1_20260923_173045.log`
- `data/results/mac-cpu/cpu/einsum/20260923_173045/julia_omeinsum_t1_20260923_173045.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.121 ± 0.157 | 0.579 ± 0.107 | **0.387 ± 0.001** | 2.202 ± 0.019 | 3.923 ± 0.908 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.202 ± 0.012 | **0.084 ± 0.005** | 0.087 ± 0.004 | 0.343 ± 0.012 | 0.489 ± 0.111 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.105 ± 0.003** | 0.113 ± 0.004 | 0.111 ± 0.006 | 0.175 ± 0.004 | 0.410 ± 0.433 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.459 ± 0.037 | **0.109 ± 0.003** | 0.114 ± 0.001 | 0.201 ± 0.006 | 0.615 ± 0.670 |
| bin_elementwise_mul_2048x2048 | 0.958 ± 0.069 | 1.012 ± 0.095 | **0.833 ± 0.026** | 1.512 ± 0.072 | 0.917 ± 0.567 |
| bin_matmul_1024 | 6.412 ± 0.590 | 10.665 ± 0.769 | **4.558 ± 0.379** | 34.775 ± 0.157 | 41.507 ± 4.756 |
| bin_matmul_256 | 0.127 ± 0.004 | **0.085 ± 0.001** | 0.088 ± 0.001 | 0.604 ± 0.012 | 0.716 ± 0.099 |
| bin_outer_product_4096 | 1.046 ± 0.050 | **1.019 ± 0.056** | 1.479 ± 0.012 | 1.064 ± 0.022 | 97.935 ± 39.365 |
| gm_queen5_5_3.wcsp | **650.701 ± 11.043** | 1085.425 ± 8.523 | 933.170 ± 37.737 | 1269.945 ± 8.773 | 1380.441 ± 92.521 |
| lm_batch_likelihood_brackets_4_4d | 11.716 ± 0.147 | 9.908 ± 0.370 | 8.390 ± 0.084 | **7.238 ± 0.140** | 11.965 ± 2.361 |
| lm_batch_likelihood_sentence_3_12d | 18.549 ± 0.399 | 17.268 ± 0.449 | **17.091 ± 0.278** | 28.702 ± 0.109 | 82.741 ± 43.686 |
| lm_batch_likelihood_sentence_4_4d | 12.436 ± 0.133 | 10.593 ± 0.567 | 8.715 ± 0.175 | **8.231 ± 0.080** | 18.107 ± 46.463 |
| nary_matmul_chain_64 | 0.012 ± 0.000 | **0.009 ± 0.001** | 0.013 ± 0.001 | 0.057 ± 0.001 | 0.040 ± 0.008 |
| str_matrix_chain_multiplication_100 | **2.017 ± 0.080** | 2.021 ± 0.064 | 2.173 ± 0.038 | 8.000 ± 0.131 | 9.014 ± 0.278 |
| str_mps_varying_inner_product_200 | 4.490 ± 0.026 | 4.506 ± 0.062 | **3.559 ± 0.043** | 12.080 ± 0.065 | 10.715 ± 0.952 |
| str_nw_mera_closed_120 | 330.156 ± 6.677 | 328.572 ± 9.275 | **158.039 ± 1.575** | 832.828 ± 1.285 | 959.707 ± 50.009 |
| str_nw_mera_open_26 | 238.009 ± 5.837 | 176.710 ± 2.777 | **127.827 ± 0.513** | 622.904 ± 1.184 | 709.139 ± 51.280 |
| tensornetwork_permutation_focus_step409_316 | **78.325 ± 0.894** | 156.726 ± 5.859 | 151.955 ± 1.440 | 204.284 ± 3.180 | 179.523 ± 46.714 |
| tensornetwork_permutation_light_415 | **78.040 ± 1.870** | 98.143 ± 0.534 | 132.248 ± 1.060 | 207.940 ± 3.220 | 176.227 ± 2.115 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.121 ± 0.157 | 0.579 ± 0.107 | **0.387 ± 0.001** | 2.202 ± 0.019 | 3.923 ± 0.908 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.202 ± 0.012 | **0.084 ± 0.005** | 0.087 ± 0.004 | 0.343 ± 0.012 | 0.489 ± 0.111 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.105 ± 0.003** | 0.113 ± 0.004 | 0.111 ± 0.006 | 0.175 ± 0.004 | 0.410 ± 0.433 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.459 ± 0.037 | **0.109 ± 0.003** | 0.114 ± 0.001 | 0.201 ± 0.006 | 0.615 ± 0.670 |
| bin_elementwise_mul_2048x2048 | 0.958 ± 0.069 | 1.012 ± 0.095 | **0.833 ± 0.026** | 1.512 ± 0.072 | 0.917 ± 0.567 |
| bin_matmul_1024 | 6.412 ± 0.590 | 10.665 ± 0.769 | **4.558 ± 0.379** | 34.775 ± 0.157 | 41.507 ± 4.756 |
| bin_matmul_256 | 0.127 ± 0.004 | **0.085 ± 0.001** | 0.088 ± 0.001 | 0.604 ± 0.012 | 0.716 ± 0.099 |
| bin_outer_product_4096 | 1.046 ± 0.050 | **1.019 ± 0.056** | 1.479 ± 0.012 | 1.064 ± 0.022 | 97.935 ± 39.365 |
| gm_queen5_5_3.wcsp | **141.550 ± 0.922** | 287.647 ± 2.108 | 315.341 ± 8.493 | 633.885 ± 2.953 | 760.990 ± 6.669 |
| lm_batch_likelihood_brackets_4_4d | 10.251 ± 0.182 | 10.655 ± 0.136 | 8.708 ± 0.190 | **6.939 ± 0.040** | 19.436 ± 47.995 |
| lm_batch_likelihood_sentence_3_12d | 18.453 ± 0.214 | 20.565 ± 0.439 | **17.150 ± 0.211** | 29.965 ± 0.134 | 46.531 ± 48.174 |
| lm_batch_likelihood_sentence_4_4d | 11.801 ± 0.350 | 11.517 ± 0.156 | 8.383 ± 0.145 | **7.722 ± 0.048** | 19.194 ± 48.494 |
| nary_matmul_chain_64 | 0.012 ± 0.000 | **0.009 ± 0.001** | 0.013 ± 0.001 | 0.057 ± 0.001 | 0.040 ± 0.008 |
| str_matrix_chain_multiplication_100 | **1.947 ± 0.035** | 2.078 ± 0.090 | 2.166 ± 0.035 | 7.952 ± 0.170 | 9.065 ± 0.700 |
| str_mps_varying_inner_product_200 | 5.683 ± 0.086 | 5.858 ± 0.107 | **4.420 ± 0.190** | 13.157 ± 0.129 | 9.339 ± 7.134 |
| str_nw_mera_closed_120 | 186.569 ± 3.173 | 185.159 ± 1.095 | **130.759 ± 0.557** | 812.532 ± 0.586 | 934.149 ± 55.304 |
| str_nw_mera_open_26 | 243.436 ± 3.648 | 177.413 ± 0.714 | **133.410 ± 0.912** | 634.110 ± 1.111 | 711.312 ± 2.290 |
| tensornetwork_permutation_focus_step409_316 | **78.325 ± 0.894** | 156.726 ± 5.859 | 151.955 ± 1.440 | 204.284 ± 3.180 | 179.523 ± 46.714 |
| tensornetwork_permutation_light_415 | **78.040 ± 1.870** | 98.143 ± 0.534 | 132.248 ± 1.060 | 207.940 ± 3.220 | 176.227 ± 2.115 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260923_175502/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260923_175502/run.yaml`
- Timestamp: `20260923_175502`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260923_175502`.

- tenferro-rs commit: `412852a8cfb13b14246443fc09b1c0c34f5ce1cd`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20260923_175502/einsum_table_t4_20260923_175502.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260923_175502/tenferro_trace_t4_20260923_175502.log`
- `data/results/mac-cpu/cpu/einsum/20260923_175502/tenferro_eager_t4_20260923_175502.log`
- `data/results/mac-cpu/cpu/einsum/20260923_175502/pytorch_cpu_t4_20260923_175502.log`
- `data/results/mac-cpu/cpu/einsum/20260923_175502/jax_cpu_t4_20260923_175502.log`
- `data/results/mac-cpu/cpu/einsum/20260923_175502/julia_omeinsum_t4_20260923_175502.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.851 ± 0.102 | 0.519 ± 0.101 | **0.387 ± 0.003** | 0.653 ± 0.073 | 2.052 ± 3.280 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.204 ± 0.033 | **0.085 ± 0.002** | 0.086 ± 0.001 | 0.151 ± 0.034 | 0.678 ± 0.277 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.074 ± 0.016 | 0.057 ± 0.020 | **0.055 ± 0.005** | 0.110 ± 0.017 | 0.336 ± 0.084 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.175 ± 0.022 | **0.053 ± 0.008** | **0.053 ± 0.001** | 0.118 ± 0.011 | 0.903 ± 1.018 |
| bin_elementwise_mul_2048x2048 | 0.598 ± 0.047 | 0.582 ± 0.032 | **0.536 ± 0.008** | 0.768 ± 0.073 | 0.974 ± 0.474 |
| bin_matmul_1024 | 7.392 ± 0.918 | 10.469 ± 0.273 | **4.523 ± 0.059** | 9.158 ± 0.079 | 16.818 ± 6.061 |
| bin_matmul_256 | 0.161 ± 0.009 | **0.084 ± 0.002** | 0.087 ± 0.001 | 0.233 ± 0.019 | 0.315 ± 0.034 |
| bin_outer_product_4096 | 0.587 ± 0.016 | **0.555 ± 0.018** | 0.563 ± 0.007 | 0.647 ± 0.037 | 72.860 ± 12.055 |
| gm_queen5_5_3.wcsp | 553.252 ± 35.986 | 527.750 ± 31.850 | **466.196 ± 29.429** | 690.214 ± 26.124 | 1152.544 ± 187.648 |
| lm_batch_likelihood_brackets_4_4d | **6.302 ± 0.100** | 8.318 ± 0.149 | 8.407 ± 0.390 | 7.031 ± 0.349 | 11.148 ± 1.715 |
| lm_batch_likelihood_sentence_3_12d | **9.429 ± 0.537** | 13.235 ± 0.761 | 19.192 ± 2.418 | 11.781 ± 0.190 | 40.674 ± 16.719 |
| lm_batch_likelihood_sentence_4_4d | 6.951 ± 0.106 | 9.360 ± 0.134 | 8.741 ± 0.377 | **6.915 ± 0.161** | 15.747 ± 15.595 |
| nary_matmul_chain_64 | 0.033 ± 0.004 | **0.009 ± 0.000** | 0.013 ± 0.001 | 0.064 ± 0.007 | 0.047 ± 0.007 |
| str_matrix_chain_multiplication_100 | **1.975 ± 0.043** | **1.975 ± 0.017** | 2.093 ± 0.070 | 7.773 ± 0.214 | 6.653 ± 0.598 |
| str_mps_varying_inner_product_200 | 4.825 ± 0.104 | 5.106 ± 0.102 | **3.799 ± 0.035** | 16.423 ± 0.417 | 11.656 ± 5.566 |
| str_nw_mera_closed_120 | 149.755 ± 2.632 | 148.627 ± 1.450 | **143.850 ± 9.224** | 254.568 ± 5.238 | 345.168 ± 29.086 |
| str_nw_mera_open_26 | 148.322 ± 6.236 | 148.641 ± 2.598 | **125.204 ± 6.064** | 184.780 ± 2.988 | 278.052 ± 23.240 |
| tensornetwork_permutation_focus_step409_316 | **59.423 ± 10.444** | 74.740 ± 14.601 | 70.907 ± 8.164 | 89.324 ± 1.981 | 105.240 ± 19.982 |
| tensornetwork_permutation_light_415 | **57.916 ± 15.079** | 65.075 ± 2.266 | 74.322 ± 13.773 | 97.716 ± 4.759 | 99.662 ± 20.119 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.851 ± 0.102 | 0.519 ± 0.101 | **0.387 ± 0.003** | 0.653 ± 0.073 | 2.052 ± 3.280 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.204 ± 0.033 | **0.085 ± 0.002** | 0.086 ± 0.001 | 0.151 ± 0.034 | 0.678 ± 0.277 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.074 ± 0.016 | 0.057 ± 0.020 | **0.055 ± 0.005** | 0.110 ± 0.017 | 0.336 ± 0.084 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.175 ± 0.022 | **0.053 ± 0.008** | **0.053 ± 0.001** | 0.118 ± 0.011 | 0.903 ± 1.018 |
| bin_elementwise_mul_2048x2048 | 0.598 ± 0.047 | 0.582 ± 0.032 | **0.536 ± 0.008** | 0.768 ± 0.073 | 0.974 ± 0.474 |
| bin_matmul_1024 | 7.392 ± 0.918 | 10.469 ± 0.273 | **4.523 ± 0.059** | 9.158 ± 0.079 | 16.818 ± 6.061 |
| bin_matmul_256 | 0.161 ± 0.009 | **0.084 ± 0.002** | 0.087 ± 0.001 | 0.233 ± 0.019 | 0.315 ± 0.034 |
| bin_outer_product_4096 | 0.587 ± 0.016 | **0.555 ± 0.018** | 0.563 ± 0.007 | 0.647 ± 0.037 | 72.860 ± 12.055 |
| gm_queen5_5_3.wcsp | **141.998 ± 25.515** | 193.875 ± 12.574 | 184.421 ± 10.465 | 270.027 ± 5.008 | 409.957 ± 13.559 |
| lm_batch_likelihood_brackets_4_4d | 6.735 ± 0.159 | 8.847 ± 0.154 | 9.808 ± 0.468 | **6.609 ± 0.138** | 14.843 ± 15.937 |
| lm_batch_likelihood_sentence_3_12d | **9.548 ± 0.394** | 13.297 ± 0.188 | 18.764 ± 0.340 | 12.218 ± 0.144 | 35.748 ± 15.852 |
| lm_batch_likelihood_sentence_4_4d | 6.912 ± 0.140 | 9.085 ± 0.137 | 9.743 ± 0.603 | **6.390 ± 0.242** | 13.243 ± 15.676 |
| nary_matmul_chain_64 | 0.033 ± 0.004 | **0.009 ± 0.000** | 0.013 ± 0.001 | 0.064 ± 0.007 | 0.047 ± 0.007 |
| str_matrix_chain_multiplication_100 | **1.957 ± 0.038** | 2.000 ± 0.017 | 2.128 ± 0.045 | 7.702 ± 0.264 | 5.726 ± 0.504 |
| str_mps_varying_inner_product_200 | **5.996 ± 0.101** | 6.021 ± 0.058 | 8.318 ± 0.317 | 17.042 ± 0.409 | 8.695 ± 0.743 |
| str_nw_mera_closed_120 | 129.488 ± 2.759 | 129.796 ± 0.280 | **128.619 ± 3.688** | 254.798 ± 4.426 | 298.744 ± 17.845 |
| str_nw_mera_open_26 | 146.695 ± 3.737 | 143.745 ± 4.275 | **124.586 ± 1.179** | 193.097 ± 1.111 | 270.215 ± 20.545 |
| tensornetwork_permutation_focus_step409_316 | **59.423 ± 10.444** | 74.740 ± 14.601 | 70.907 ± 8.164 | 89.324 ± 1.981 | 105.240 ± 19.982 |
| tensornetwork_permutation_light_415 | **57.916 ± 15.079** | 65.075 ± 2.266 | 74.322 ± 13.773 | 97.716 ± 4.759 | 99.662 ± 20.119 |

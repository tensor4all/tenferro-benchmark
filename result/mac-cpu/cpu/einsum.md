# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Benchmark commit: `d9f6406557a0d1066f37785ec85f9b9afdbb4781`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261123_172252/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261123_172252/run.yaml`
- Timestamp: `2026-11-23T17:22:52.769897+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261123_172252`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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

- `data/results/mac-cpu/cpu/einsum/20261123_172252/jax_cpu_t1_20261123_172252.log`
- `data/results/mac-cpu/cpu/einsum/20261123_172252/julia_omeinsum_t1_20261123_172252.log`
- `data/results/mac-cpu/cpu/einsum/20261123_172252/pytorch_cpu_t1_20261123_172252.log`
- `data/results/mac-cpu/cpu/einsum/20261123_172252/tenferro_eager_t1_20261123_172252.log`
- `data/results/mac-cpu/cpu/einsum/20261123_172252/tenferro_trace_t1_20261123_172252.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.409 ± 0.361 | 3.209 ± 0.166 | **0.758 ± 0.052** | 3.608 ± 0.131 | 4.571 ± 0.423 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.595 ± 0.062 | 0.446 ± 0.001 | **0.140 ± 0.014** | 0.613 ± 0.054 | 0.628 ± 0.363 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.249 ± 0.046 | 0.221 ± 0.027 | **0.199 ± 0.030** | 0.406 ± 0.011 | 0.683 ± 0.292 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.900 ± 0.069 | 0.215 ± 0.003 | **0.207 ± 0.020** | 0.545 ± 0.070 | 1.176 ± 0.275 |
| bin_elementwise_mul_2048x2048 | 2.188 ± 0.215 | **1.902 ± 0.335** | 2.432 ± 0.123 | 2.984 ± 0.020 | 2.301 ± 0.233 |
| bin_elementwise_mul_512x512 | 0.083 ± 0.010 | **0.071 ± 0.001** | 0.092 ± 0.013 | 0.311 ± 0.018 | 0.075 ± 0.009 |
| bin_matmul_1024 | 58.607 ± 4.209 | 49.233 ± 5.945 | **13.360 ± 0.177** | 51.772 ± 0.728 | 50.711 ± 1.545 |
| bin_matmul_256 | 1.070 ± 0.179 | 0.892 ± 0.085 | **0.133 ± 0.007** | 0.927 ± 0.030 | 0.912 ± 0.019 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.002 | **0.004 ± 0.000** | 0.021 ± 0.004 | 0.085 ± 0.005 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.024 ± 0.005 | **0.023 ± 0.002** | 0.087 ± 0.006 | 0.189 ± 0.022 | 0.063 ± 0.013 |
| bin_omeinsum_matmul_10x10 | 0.013 ± 0.001 | **0.008 ± 0.000** | 0.017 ± 0.001 | 0.073 ± 0.020 | 0.017 ± 0.001 |
| bin_outer_product_4096 | 5.449 ± 0.975 | 4.720 ± 0.236 | 3.311 ± 0.136 | **2.547 ± 0.072** | 294.147 ± 136.213 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 106.125 ± 10.114 | 96.855 ± 9.014 | **11.301 ± 0.235** | 54.607 ± 0.409 | 52.733 ± 1.922 |
| gm_queen5_5_3.wcsp | 3448.458 ± 117.782 | 2968.180 ± 259.857 | **2037.585 ± 40.328** | 3318.042 ± 91.661 | 3320.191 ± 109.511 |
| lm_batch_likelihood_brackets_4_4d | 24.595 ± 2.486 | 26.494 ± 3.011 | 13.320 ± 0.345 | **12.464 ± 0.067** | 24.277 ± 22.097 |
| lm_batch_likelihood_sentence_3_12d | 84.290 ± 2.079 | 88.914 ± 1.639 | **25.544 ± 1.024** | 53.607 ± 5.944 | 69.997 ± 31.160 |
| lm_batch_likelihood_sentence_4_4d | 27.187 ± 2.774 | 26.663 ± 0.343 | 14.488 ± 0.339 | **13.633 ± 0.111** | 23.972 ± 23.609 |
| nary_matmul_chain_64 | 0.039 ± 0.000 | 0.034 ± 0.002 | **0.024 ± 0.004** | 0.121 ± 0.004 | 0.060 ± 0.005 |
| str_matrix_chain_multiplication_100 | 13.815 ± 0.581 | 13.775 ± 0.833 | **4.932 ± 0.059** | 13.005 ± 0.049 | 13.834 ± 0.744 |
| str_mps_varying_inner_product_200 | 18.701 ± 0.546 | 18.893 ± 1.831 | **6.890 ± 0.075** | 18.990 ± 0.240 | 17.461 ± 0.595 |
| str_nw_mera_closed_120 | 2288.561 ± 100.841 | 2076.760 ± 46.553 | **340.932 ± 31.155** | 1511.676 ± 37.520 | 1232.474 ± 72.821 |
| str_nw_mera_open_26 | 1490.041 ± 30.842 | 1460.623 ± 76.542 | **226.009 ± 7.103** | 1084.876 ± 32.468 | 931.043 ± 26.216 |
| tensornetwork_permutation_focus_step409_316 | 274.866 ± 20.740 | 337.339 ± 28.998 | 252.940 ± 7.918 | 335.842 ± 6.566 | **195.115 ± 17.680** |
| tensornetwork_permutation_light_415 | 271.020 ± 12.452 | 261.537 ± 9.611 | 212.784 ± 19.569 | 344.935 ± 20.130 | **197.212 ± 9.894** |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.409 ± 0.361 | 3.209 ± 0.166 | **0.758 ± 0.052** | 3.608 ± 0.131 | 4.571 ± 0.423 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.595 ± 0.062 | 0.446 ± 0.001 | **0.140 ± 0.014** | 0.613 ± 0.054 | 0.628 ± 0.363 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.249 ± 0.046 | 0.221 ± 0.027 | **0.199 ± 0.030** | 0.406 ± 0.011 | 0.683 ± 0.292 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.900 ± 0.069 | 0.215 ± 0.003 | **0.207 ± 0.020** | 0.545 ± 0.070 | 1.176 ± 0.275 |
| bin_elementwise_mul_2048x2048 | 2.188 ± 0.215 | **1.902 ± 0.335** | 2.432 ± 0.123 | 2.984 ± 0.020 | 2.301 ± 0.233 |
| bin_elementwise_mul_512x512 | 0.083 ± 0.010 | **0.071 ± 0.001** | 0.092 ± 0.013 | 0.311 ± 0.018 | 0.075 ± 0.009 |
| bin_matmul_1024 | 58.607 ± 4.209 | 49.233 ± 5.945 | **13.360 ± 0.177** | 51.772 ± 0.728 | 50.711 ± 1.545 |
| bin_matmul_256 | 1.070 ± 0.179 | 0.892 ± 0.085 | **0.133 ± 0.007** | 0.927 ± 0.030 | 0.912 ± 0.019 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.002 | **0.004 ± 0.000** | 0.021 ± 0.004 | 0.085 ± 0.005 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.024 ± 0.005 | **0.023 ± 0.002** | 0.087 ± 0.006 | 0.189 ± 0.022 | 0.063 ± 0.013 |
| bin_omeinsum_matmul_10x10 | 0.013 ± 0.001 | **0.008 ± 0.000** | 0.017 ± 0.001 | 0.073 ± 0.020 | 0.017 ± 0.001 |
| bin_outer_product_4096 | 5.449 ± 0.975 | 4.720 ± 0.236 | 3.311 ± 0.136 | **2.547 ± 0.072** | 294.147 ± 136.213 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 106.125 ± 10.114 | 96.855 ± 9.014 | **11.301 ± 0.235** | 54.607 ± 0.409 | 52.733 ± 1.922 |
| gm_queen5_5_3.wcsp | 1619.064 ± 40.334 | 1585.297 ± 115.086 | **746.862 ± 86.186** | 1116.397 ± 64.682 | 928.817 ± 30.662 |
| lm_batch_likelihood_brackets_4_4d | 25.202 ± 0.773 | 22.848 ± 0.588 | 14.952 ± 0.365 | **11.863 ± 0.084** | 22.747 ± 7.251 |
| lm_batch_likelihood_sentence_3_12d | 88.619 ± 6.259 | 91.135 ± 0.952 | **25.504 ± 0.750** | 54.135 ± 0.409 | 68.927 ± 25.995 |
| lm_batch_likelihood_sentence_4_4d | 21.943 ± 0.268 | 24.978 ± 0.182 | 15.148 ± 0.216 | **13.674 ± 0.073** | 21.519 ± 34.428 |
| nary_matmul_chain_64 | 0.039 ± 0.000 | 0.034 ± 0.002 | **0.024 ± 0.004** | 0.121 ± 0.004 | 0.060 ± 0.005 |
| str_matrix_chain_multiplication_100 | 13.006 ± 0.141 | 12.799 ± 0.109 | **4.964 ± 0.072** | 13.445 ± 0.336 | 15.167 ± 1.694 |
| str_mps_varying_inner_product_200 | 22.031 ± 0.503 | 19.763 ± 0.295 | **8.203 ± 0.062** | 21.477 ± 1.032 | 15.025 ± 1.469 |
| str_nw_mera_closed_120 | 2128.180 ± 62.865 | 1954.661 ± 43.989 | **273.616 ± 13.724** | 1424.509 ± 20.826 | 1194.078 ± 28.356 |
| str_nw_mera_open_26 | 1521.209 ± 105.463 | 1441.023 ± 39.855 | **234.271 ± 9.363** | 1133.979 ± 25.299 | 947.310 ± 33.855 |
| tensornetwork_permutation_focus_step409_316 | 274.866 ± 20.740 | 337.339 ± 28.998 | 252.940 ± 7.918 | 335.842 ± 6.566 | **195.115 ± 17.680** |
| tensornetwork_permutation_light_415 | 271.020 ± 12.452 | 261.537 ± 9.611 | 212.784 ± 19.569 | 344.935 ± 20.130 | **197.212 ± 9.894** |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261123_174752/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261123_174752/run.yaml`
- Timestamp: `2026-11-23T17:47:52.092097+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261123_174752`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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

- `data/results/mac-cpu/cpu/einsum/20261123_174752/jax_cpu_t4_20261123_174752.log`
- `data/results/mac-cpu/cpu/einsum/20261123_174752/julia_omeinsum_t4_20261123_174752.log`
- `data/results/mac-cpu/cpu/einsum/20261123_174752/pytorch_cpu_t4_20261123_174752.log`
- `data/results/mac-cpu/cpu/einsum/20261123_174752/tenferro_eager_t4_20261123_174752.log`
- `data/results/mac-cpu/cpu/einsum/20261123_174752/tenferro_trace_t4_20261123_174752.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.011 ± 0.072 | 2.545 ± 0.078 | **0.671 ± 0.004** | 1.552 ± 0.064 | 83.608 ± 29.827 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.512 ± 0.024 | 0.414 ± 0.008 | **0.126 ± 0.006** | 0.660 ± 0.039 | 0.650 ± 0.346 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.145 ± 0.002** | 0.157 ± 0.008 | 0.198 ± 0.051 | 0.342 ± 0.068 | 0.699 ± 0.289 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.202 ± 0.006 | **0.163 ± 0.004** | 0.193 ± 0.008 | 0.404 ± 0.026 | 1.225 ± 0.094 |
| bin_elementwise_mul_2048x2048 | **1.885 ± 0.008** | 1.987 ± 0.007 | 2.015 ± 0.036 | 3.056 ± 0.084 | 2.343 ± 0.029 |
| bin_elementwise_mul_512x512 | 0.066 ± 0.008 | **0.060 ± 0.011** | 0.076 ± 0.003 | 0.264 ± 0.025 | 0.074 ± 0.011 |
| bin_matmul_1024 | 24.561 ± 6.007 | 17.805 ± 1.208 | **10.319 ± 0.754** | 17.970 ± 0.276 | 30.886 ± 6.062 |
| bin_matmul_256 | 0.526 ± 0.052 | 0.457 ± 0.012 | **0.124 ± 0.003** | 0.472 ± 0.043 | 2.444 ± 1.489 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.001 | **0.005 ± 0.000** | 0.019 ± 0.001 | 0.085 ± 0.006 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.027 ± 0.006 | **0.024 ± 0.001** | 0.079 ± 0.001 | 0.200 ± 0.027 | 0.040 ± 0.010 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.002 | **0.008 ± 0.000** | 0.017 ± 0.001 | 0.087 ± 0.021 | 0.017 ± 0.001 |
| bin_outer_product_4096 | 2.302 ± 0.064 | **2.219 ± 0.018** | 2.433 ± 0.019 | 2.410 ± 0.108 | 305.753 ± 161.038 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 43.725 ± 0.953 | 40.352 ± 5.842 | **9.817 ± 0.205** | 19.845 ± 0.601 | 28.577 ± 7.285 |
| gm_queen5_5_3.wcsp | 2078.398 ± 88.423 | 2044.816 ± 50.325 | **1480.835 ± 94.661** | 2358.182 ± 111.395 | 4606.709 ± 284.760 |
| lm_batch_likelihood_brackets_4_4d | 14.953 ± 0.143 | 17.668 ± 0.140 | 14.458 ± 0.487 | **14.230 ± 0.158** | 84.462 ± 41.870 |
| lm_batch_likelihood_sentence_3_12d | 37.374 ± 0.218 | 45.561 ± 0.487 | **26.308 ± 1.027** | 26.517 ± 0.375 | 78.471 ± 18.236 |
| lm_batch_likelihood_sentence_4_4d | 17.037 ± 0.411 | 18.561 ± 1.213 | **12.931 ± 0.182** | 14.584 ± 0.112 | 66.061 ± 17.200 |
| nary_matmul_chain_64 | 0.043 ± 0.009 | 0.049 ± 0.004 | **0.023 ± 0.003** | 0.125 ± 0.013 | 0.050 ± 0.005 |
| str_matrix_chain_multiplication_100 | 8.506 ± 0.073 | 9.636 ± 0.520 | **4.900 ± 0.088** | 10.059 ± 0.110 | 264.361 ± 82.797 |
| str_mps_varying_inner_product_200 | 19.520 ± 0.165 | 22.042 ± 0.580 | **7.835 ± 0.100** | 18.366 ± 0.222 | 15.943 ± 3.844 |
| str_nw_mera_closed_120 | 964.285 ± 42.829 | 965.114 ± 18.319 | **314.837 ± 10.923** | 461.887 ± 5.034 | 662.409 ± 52.037 |
| str_nw_mera_open_26 | 594.687 ± 14.549 | 680.811 ± 38.745 | **220.863 ± 8.247** | 375.381 ± 3.011 | 597.373 ± 30.906 |
| tensornetwork_permutation_focus_step409_316 | **125.131 ± 8.461** | 145.766 ± 11.402 | 129.827 ± 30.091 | 222.799 ± 31.197 | 374.700 ± 58.935 |
| tensornetwork_permutation_light_415 | 128.055 ± 4.468 | 127.838 ± 24.098 | **122.891 ± 20.584** | 228.933 ± 20.214 | 365.459 ± 21.843 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.011 ± 0.072 | 2.545 ± 0.078 | **0.671 ± 0.004** | 1.552 ± 0.064 | 83.608 ± 29.827 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.512 ± 0.024 | 0.414 ± 0.008 | **0.126 ± 0.006** | 0.660 ± 0.039 | 0.650 ± 0.346 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.145 ± 0.002** | 0.157 ± 0.008 | 0.198 ± 0.051 | 0.342 ± 0.068 | 0.699 ± 0.289 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.202 ± 0.006 | **0.163 ± 0.004** | 0.193 ± 0.008 | 0.404 ± 0.026 | 1.225 ± 0.094 |
| bin_elementwise_mul_2048x2048 | **1.885 ± 0.008** | 1.987 ± 0.007 | 2.015 ± 0.036 | 3.056 ± 0.084 | 2.343 ± 0.029 |
| bin_elementwise_mul_512x512 | 0.066 ± 0.008 | **0.060 ± 0.011** | 0.076 ± 0.003 | 0.264 ± 0.025 | 0.074 ± 0.011 |
| bin_matmul_1024 | 24.561 ± 6.007 | 17.805 ± 1.208 | **10.319 ± 0.754** | 17.970 ± 0.276 | 30.886 ± 6.062 |
| bin_matmul_256 | 0.526 ± 0.052 | 0.457 ± 0.012 | **0.124 ± 0.003** | 0.472 ± 0.043 | 2.444 ± 1.489 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.001 | **0.005 ± 0.000** | 0.019 ± 0.001 | 0.085 ± 0.006 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.027 ± 0.006 | **0.024 ± 0.001** | 0.079 ± 0.001 | 0.200 ± 0.027 | 0.040 ± 0.010 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.002 | **0.008 ± 0.000** | 0.017 ± 0.001 | 0.087 ± 0.021 | 0.017 ± 0.001 |
| bin_outer_product_4096 | 2.302 ± 0.064 | **2.219 ± 0.018** | 2.433 ± 0.019 | 2.410 ± 0.108 | 305.753 ± 161.038 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 43.725 ± 0.953 | 40.352 ± 5.842 | **9.817 ± 0.205** | 19.845 ± 0.601 | 28.577 ± 7.285 |
| gm_queen5_5_3.wcsp | 832.438 ± 12.801 | 797.114 ± 21.462 | **508.003 ± 37.734** | 602.990 ± 13.071 | 720.630 ± 16.183 |
| lm_batch_likelihood_brackets_4_4d | 14.790 ± 0.082 | 16.290 ± 0.120 | **14.052 ± 0.681** | 14.073 ± 0.404 | 78.461 ± 18.208 |
| lm_batch_likelihood_sentence_3_12d | 39.191 ± 0.609 | 48.867 ± 0.345 | **28.140 ± 0.322** | 28.230 ± 0.303 | 104.452 ± 29.325 |
| lm_batch_likelihood_sentence_4_4d | 15.010 ± 0.325 | 17.101 ± 0.668 | **14.246 ± 0.210** | 14.378 ± 0.269 | 81.371 ± 20.845 |
| nary_matmul_chain_64 | 0.043 ± 0.009 | 0.049 ± 0.004 | **0.023 ± 0.003** | 0.125 ± 0.013 | 0.050 ± 0.005 |
| str_matrix_chain_multiplication_100 | 9.290 ± 0.572 | 9.384 ± 0.660 | **4.915 ± 0.128** | 9.986 ± 0.155 | 275.845 ± 36.529 |
| str_mps_varying_inner_product_200 | 27.044 ± 1.436 | 23.281 ± 0.249 | **10.296 ± 0.361** | 20.995 ± 0.294 | 14.438 ± 24.015 |
| str_nw_mera_closed_120 | 957.561 ± 23.468 | 967.748 ± 25.050 | **272.948 ± 4.412** | 431.385 ± 6.829 | 608.136 ± 47.922 |
| str_nw_mera_open_26 | 600.328 ± 58.855 | 669.404 ± 18.421 | **230.702 ± 11.660** | 387.286 ± 4.700 | 658.721 ± 45.418 |
| tensornetwork_permutation_focus_step409_316 | **125.131 ± 8.461** | 145.766 ± 11.402 | 129.827 ± 30.091 | 222.799 ± 31.197 | 374.700 ± 58.935 |
| tensornetwork_permutation_light_415 | 128.055 ± 4.468 | 127.838 ± 24.098 | **122.891 ± 20.584** | 228.933 ± 20.214 | 365.459 ± 21.843 |

# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `bca2d54a5884586b1aeae8006caa9b8c0ba44a26`
- Benchmark commit: `f983e88b9d07e558989427d047f1fa4676df2ea8`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260916_102753/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260916_102753/run.yaml`
- Timestamp: `20260916_102753`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260916_102753`.

- tenferro-rs commit: `bca2d54a5884586b1aeae8006caa9b8c0ba44a26`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20260916_102753/einsum_table_t1_20260916_102753.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260916_102753/tenferro_trace_t1_20260916_102753.log`
- `data/results/mac-cpu/cpu/einsum/20260916_102753/tenferro_eager_t1_20260916_102753.log`
- `data/results/mac-cpu/cpu/einsum/20260916_102753/pytorch_cpu_t1_20260916_102753.log`
- `data/results/mac-cpu/cpu/einsum/20260916_102753/jax_cpu_t1_20260916_102753.log`
- `data/results/mac-cpu/cpu/einsum/20260916_102753/julia_omeinsum_t1_20260916_102753.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.072 ± 0.176 | 0.812 ± 0.073 | **0.365 ± 0.021** | 2.314 ± 0.108 | 3.840 ± 0.919 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.200 ± 0.006 | 0.134 ± 0.001 | **0.086 ± 0.001** | 0.357 ± 0.027 | 0.610 ± 0.187 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.104 ± 0.001** | 0.113 ± 0.002 | 0.118 ± 0.003 | 0.185 ± 0.063 | 0.617 ± 0.278 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.436 ± 0.021 | **0.111 ± 0.007** | 0.118 ± 0.001 | 0.208 ± 0.009 | 0.741 ± 0.708 |
| bin_elementwise_mul_2048x2048 | 0.953 ± 0.063 | 2.061 ± 0.138 | **0.849 ± 0.056** | 1.559 ± 0.042 | 0.909 ± 0.410 |
| bin_matmul_1024 | 6.351 ± 0.350 | **5.178 ± 0.828** | 6.669 ± 0.213 | 37.069 ± 3.845 | 40.243 ± 4.629 |
| bin_matmul_256 | 0.136 ± 0.003 | 0.101 ± 0.001 | **0.092 ± 0.013** | 0.616 ± 0.026 | 0.722 ± 0.047 |
| bin_outer_product_4096 | 1.037 ± 0.064 | **1.027 ± 0.064** | 1.528 ± 0.062 | 1.094 ± 0.068 | 100.231 ± 40.725 |
| gm_queen5_5_3.wcsp | **894.996 ± 7.782** | 1088.143 ± 6.306 | 917.798 ± 7.764 | 1335.024 ± 48.497 | 1401.680 ± 98.680 |
| lm_batch_likelihood_brackets_4_4d | 8.163 ± 0.100 | 14.134 ± 0.253 | 8.757 ± 0.303 | **7.896 ± 0.270** | 12.506 ± 4.117 |
| lm_batch_likelihood_sentence_3_12d | **10.281 ± 0.157** | 25.442 ± 0.385 | 19.397 ± 0.828 | 30.078 ± 1.761 | 85.379 ± 43.269 |
| lm_batch_likelihood_sentence_4_4d | 8.556 ± 0.132 | 15.582 ± 0.166 | 9.247 ± 0.273 | **8.460 ± 0.158** | 18.664 ± 48.806 |
| nary_matmul_chain_64 | 0.014 ± 0.001 | 0.017 ± 0.001 | **0.013 ± 0.001** | 0.060 ± 0.003 | 0.040 ± 0.001 |
| str_matrix_chain_multiplication_100 | **2.168 ± 0.047** | 2.852 ± 0.036 | 2.174 ± 0.088 | 8.339 ± 0.235 | 9.468 ± 0.230 |
| str_mps_varying_inner_product_200 | 5.557 ± 0.041 | 6.804 ± 0.355 | **3.687 ± 0.077** | 12.372 ± 0.085 | 10.734 ± 5.975 |
| str_nw_mera_closed_120 | 174.748 ± 3.477 | 187.319 ± 1.337 | **159.574 ± 1.860** | 842.198 ± 8.755 | 970.228 ± 2.931 |
| str_nw_mera_open_26 | 181.146 ± 0.819 | 202.448 ± 0.695 | **129.361 ± 0.537** | 626.816 ± 2.075 | 717.374 ± 54.543 |
| tensornetwork_permutation_focus_step409_316 | **119.377 ± 0.403** | 182.239 ± 26.393 | 157.381 ± 5.778 | 209.356 ± 5.348 | 184.560 ± 50.537 |
| tensornetwork_permutation_light_415 | **120.611 ± 0.679** | 157.468 ± 2.951 | 132.650 ± 1.328 | 212.166 ± 3.461 | 178.962 ± 1.498 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.072 ± 0.176 | 0.812 ± 0.073 | **0.365 ± 0.021** | 2.314 ± 0.108 | 3.840 ± 0.919 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.200 ± 0.006 | 0.134 ± 0.001 | **0.086 ± 0.001** | 0.357 ± 0.027 | 0.610 ± 0.187 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.104 ± 0.001** | 0.113 ± 0.002 | 0.118 ± 0.003 | 0.185 ± 0.063 | 0.617 ± 0.278 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.436 ± 0.021 | **0.111 ± 0.007** | 0.118 ± 0.001 | 0.208 ± 0.009 | 0.741 ± 0.708 |
| bin_elementwise_mul_2048x2048 | 0.953 ± 0.063 | 2.061 ± 0.138 | **0.849 ± 0.056** | 1.559 ± 0.042 | 0.909 ± 0.410 |
| bin_matmul_1024 | 6.351 ± 0.350 | **5.178 ± 0.828** | 6.669 ± 0.213 | 37.069 ± 3.845 | 40.243 ± 4.629 |
| bin_matmul_256 | 0.136 ± 0.003 | 0.101 ± 0.001 | **0.092 ± 0.013** | 0.616 ± 0.026 | 0.722 ± 0.047 |
| bin_outer_product_4096 | 1.037 ± 0.064 | **1.027 ± 0.064** | 1.528 ± 0.062 | 1.094 ± 0.068 | 100.231 ± 40.725 |
| gm_queen5_5_3.wcsp | **235.162 ± 0.437** | 405.009 ± 2.394 | 342.627 ± 77.153 | 641.892 ± 3.637 | 772.089 ± 3.634 |
| lm_batch_likelihood_brackets_4_4d | 9.102 ± 0.120 | 14.941 ± 0.201 | 8.980 ± 0.134 | **7.173 ± 0.099** | 19.473 ± 49.597 |
| lm_batch_likelihood_sentence_3_12d | **12.049 ± 0.105** | 26.728 ± 0.309 | 18.042 ± 0.550 | 30.457 ± 0.199 | 43.413 ± 44.624 |
| lm_batch_likelihood_sentence_4_4d | 9.511 ± 0.077 | 16.608 ± 0.177 | 8.789 ± 0.177 | **7.807 ± 0.063** | 19.112 ± 49.509 |
| nary_matmul_chain_64 | 0.014 ± 0.001 | 0.017 ± 0.001 | **0.013 ± 0.001** | 0.060 ± 0.003 | 0.040 ± 0.001 |
| str_matrix_chain_multiplication_100 | **2.130 ± 0.033** | 2.851 ± 0.052 | 2.170 ± 0.035 | 8.291 ± 0.136 | 9.459 ± 0.555 |
| str_mps_varying_inner_product_200 | 6.601 ± 0.112 | 7.697 ± 0.087 | **4.708 ± 0.092** | 13.639 ± 0.095 | 9.722 ± 7.387 |
| str_nw_mera_closed_120 | 150.444 ± 0.237 | 158.832 ± 0.667 | **131.607 ± 0.176** | 821.944 ± 2.295 | 943.967 ± 55.919 |
| str_nw_mera_open_26 | 184.175 ± 0.693 | 198.509 ± 0.770 | **137.719 ± 3.331** | 637.576 ± 1.912 | 666.348 ± 57.365 |
| tensornetwork_permutation_focus_step409_316 | **119.377 ± 0.403** | 182.239 ± 26.393 | 157.381 ± 5.778 | 209.356 ± 5.348 | 184.560 ± 50.537 |
| tensornetwork_permutation_light_415 | **120.611 ± 0.679** | 157.468 ± 2.951 | 132.650 ± 1.328 | 212.166 ± 3.461 | 178.962 ± 1.498 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260916_103859/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260916_103859/run.yaml`
- Timestamp: `20260916_103859`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260916_103859`.

- tenferro-rs commit: `bca2d54a5884586b1aeae8006caa9b8c0ba44a26`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20260916_103859/einsum_table_t4_20260916_103859.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260916_103859/tenferro_trace_t4_20260916_103859.log`
- `data/results/mac-cpu/cpu/einsum/20260916_103859/tenferro_eager_t4_20260916_103859.log`
- `data/results/mac-cpu/cpu/einsum/20260916_103859/pytorch_cpu_t4_20260916_103859.log`
- `data/results/mac-cpu/cpu/einsum/20260916_103859/jax_cpu_t4_20260916_103859.log`
- `data/results/mac-cpu/cpu/einsum/20260916_103859/julia_omeinsum_t4_20260916_103859.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.898 ± 0.144 | 0.741 ± 0.125 | **0.385 ± 0.016** | 0.631 ± 0.025 | 2.653 ± 3.493 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.198 ± 0.027 | 0.149 ± 0.016 | **0.086 ± 0.001** | 0.147 ± 0.005 | 0.546 ± 0.213 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.086 ± 0.026 | 0.071 ± 0.024 | **0.057 ± 0.008** | 0.113 ± 0.014 | 0.321 ± 0.036 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.179 ± 0.006 | 0.074 ± 0.021 | **0.057 ± 0.003** | 0.119 ± 0.012 | 0.705 ± 0.672 |
| bin_elementwise_mul_2048x2048 | 0.640 ± 0.036 | 1.323 ± 0.114 | **0.607 ± 0.024** | 0.740 ± 0.066 | 0.995 ± 0.161 |
| bin_matmul_1024 | 5.498 ± 0.259 | 5.128 ± 0.652 | **5.073 ± 0.474** | 9.070 ± 0.078 | 15.832 ± 4.828 |
| bin_matmul_256 | 0.158 ± 0.012 | 0.118 ± 0.003 | **0.098 ± 0.018** | 0.229 ± 0.006 | 0.314 ± 0.034 |
| bin_outer_product_4096 | 0.596 ± 0.052 | 0.604 ± 0.054 | **0.585 ± 0.031** | 0.628 ± 0.018 | 71.510 ± 13.708 |
| gm_queen5_5_3.wcsp | 502.866 ± 21.253 | 653.757 ± 52.278 | **451.694 ± 14.722** | 642.579 ± 20.484 | 1116.243 ± 62.606 |
| lm_batch_likelihood_brackets_4_4d | **6.802 ± 0.233** | 12.817 ± 0.455 | 8.787 ± 0.369 | 7.018 ± 0.190 | 10.155 ± 0.707 |
| lm_batch_likelihood_sentence_3_12d | **9.414 ± 0.112** | 16.719 ± 0.728 | 20.355 ± 2.065 | 11.791 ± 0.162 | 24.816 ± 19.565 |
| lm_batch_likelihood_sentence_4_4d | **7.034 ± 0.126** | 13.220 ± 0.214 | 8.916 ± 0.496 | 7.179 ± 0.167 | 17.440 ± 17.275 |
| nary_matmul_chain_64 | 0.035 ± 0.007 | 0.037 ± 0.005 | **0.013 ± 0.001** | 0.062 ± 0.004 | 0.044 ± 0.006 |
| str_matrix_chain_multiplication_100 | **2.202 ± 0.090** | 3.889 ± 0.246 | 2.220 ± 0.050 | 7.477 ± 0.293 | 7.417 ± 1.904 |
| str_mps_varying_inner_product_200 | 5.759 ± 0.138 | 9.046 ± 0.137 | **4.110 ± 0.515** | 16.657 ± 0.250 | 10.421 ± 5.867 |
| str_nw_mera_closed_120 | 144.342 ± 2.194 | 161.568 ± 4.528 | **142.190 ± 4.529** | 250.666 ± 1.543 | 333.548 ± 20.890 |
| str_nw_mera_open_26 | 145.373 ± 4.800 | 160.406 ± 3.974 | **125.307 ± 1.753** | 187.793 ± 3.043 | 270.803 ± 23.620 |
| tensornetwork_permutation_focus_step409_316 | **58.522 ± 0.508** | 122.216 ± 4.097 | 74.642 ± 9.822 | 88.130 ± 0.927 | 101.495 ± 19.760 |
| tensornetwork_permutation_light_415 | 85.431 ± 0.327 | 83.610 ± 11.973 | **70.690 ± 1.583** | 94.278 ± 5.034 | 92.117 ± 18.867 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.898 ± 0.144 | 0.741 ± 0.125 | **0.385 ± 0.016** | 0.631 ± 0.025 | 2.653 ± 3.493 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.198 ± 0.027 | 0.149 ± 0.016 | **0.086 ± 0.001** | 0.147 ± 0.005 | 0.546 ± 0.213 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.086 ± 0.026 | 0.071 ± 0.024 | **0.057 ± 0.008** | 0.113 ± 0.014 | 0.321 ± 0.036 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.179 ± 0.006 | 0.074 ± 0.021 | **0.057 ± 0.003** | 0.119 ± 0.012 | 0.705 ± 0.672 |
| bin_elementwise_mul_2048x2048 | 0.640 ± 0.036 | 1.323 ± 0.114 | **0.607 ± 0.024** | 0.740 ± 0.066 | 0.995 ± 0.161 |
| bin_matmul_1024 | 5.498 ± 0.259 | 5.128 ± 0.652 | **5.073 ± 0.474** | 9.070 ± 0.078 | 15.832 ± 4.828 |
| bin_matmul_256 | 0.158 ± 0.012 | 0.118 ± 0.003 | **0.098 ± 0.018** | 0.229 ± 0.006 | 0.314 ± 0.034 |
| bin_outer_product_4096 | 0.596 ± 0.052 | 0.604 ± 0.054 | **0.585 ± 0.031** | 0.628 ± 0.018 | 71.510 ± 13.708 |
| gm_queen5_5_3.wcsp | **147.289 ± 15.371** | 221.259 ± 9.902 | 195.239 ± 17.605 | 256.624 ± 3.847 | 412.811 ± 15.712 |
| lm_batch_likelihood_brackets_4_4d | 7.462 ± 0.137 | 12.777 ± 0.233 | 10.414 ± 0.375 | **6.768 ± 1.217** | 14.185 ± 17.186 |
| lm_batch_likelihood_sentence_3_12d | **10.083 ± 0.424** | 16.817 ± 0.171 | 19.381 ± 0.466 | 12.293 ± 0.170 | 34.634 ± 12.445 |
| lm_batch_likelihood_sentence_4_4d | 7.489 ± 0.218 | 13.090 ± 0.450 | 10.138 ± 0.334 | **6.656 ± 0.411** | 13.482 ± 14.026 |
| nary_matmul_chain_64 | 0.035 ± 0.007 | 0.037 ± 0.005 | **0.013 ± 0.001** | 0.062 ± 0.004 | 0.044 ± 0.006 |
| str_matrix_chain_multiplication_100 | 2.169 ± 0.033 | 3.858 ± 0.191 | **2.154 ± 0.063** | 7.742 ± 0.190 | 5.401 ± 0.290 |
| str_mps_varying_inner_product_200 | **6.769 ± 0.071** | 9.976 ± 0.075 | 9.129 ± 0.461 | 17.290 ± 0.532 | 8.949 ± 4.685 |
| str_nw_mera_closed_120 | 130.146 ± 1.632 | 140.104 ± 0.942 | **130.052 ± 5.470** | 249.652 ± 1.014 | 299.896 ± 16.387 |
| str_nw_mera_open_26 | 148.892 ± 4.143 | 147.649 ± 1.143 | **125.765 ± 3.360** | 196.044 ± 1.647 | 267.281 ± 24.037 |
| tensornetwork_permutation_focus_step409_316 | **58.522 ± 0.508** | 122.216 ± 4.097 | 74.642 ± 9.822 | 88.130 ± 0.927 | 101.495 ± 19.760 |
| tensornetwork_permutation_light_415 | 85.431 ± 0.327 | 83.610 ± 11.973 | **70.690 ± 1.583** | 94.278 ± 5.034 | 92.117 ± 18.867 |

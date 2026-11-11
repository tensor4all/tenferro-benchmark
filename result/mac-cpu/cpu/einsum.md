# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `35979dcd7110a29d0728b2e128c72914f4c3f67b`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261111_172341/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261111_172341/run.yaml`
- Timestamp: `2026-11-11T17:23:41.760374+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261111_172341`.

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

- `data/results/mac-cpu/cpu/einsum/20261111_172341/jax_cpu_t1_20261111_172341.log`
- `data/results/mac-cpu/cpu/einsum/20261111_172341/julia_omeinsum_t1_20261111_172341.log`
- `data/results/mac-cpu/cpu/einsum/20261111_172341/pytorch_cpu_t1_20261111_172341.log`
- `data/results/mac-cpu/cpu/einsum/20261111_172341/tenferro_eager_t1_20261111_172341.log`
- `data/results/mac-cpu/cpu/einsum/20261111_172341/tenferro_trace_t1_20261111_172341.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.583 ± 0.365 | 0.843 ± 0.097 | **0.669 ± 0.007** | 4.103 ± 0.095 | 8.365 ± 17.352 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.205 ± 0.406 | **0.087 ± 0.012** | 0.134 ± 0.001 | 0.922 ± 0.108 | 3.340 ± 2.083 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.280 ± 0.103 | 0.244 ± 0.042 | **0.193 ± 0.001** | 0.436 ± 0.073 | 0.674 ± 0.521 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.058 ± 0.211 | 0.245 ± 0.057 | **0.193 ± 0.000** | 0.833 ± 0.138 | 2.067 ± 2.168 |
| bin_elementwise_mul_2048x2048 | 3.299 ± 0.982 | 2.996 ± 0.341 | 3.190 ± 1.216 | 4.407 ± 0.175 | **2.331 ± 0.259** |
| bin_elementwise_mul_512x512 | 0.096 ± 0.017 | 0.092 ± 0.042 | 0.111 ± 0.115 | 0.544 ± 0.110 | **0.087 ± 0.020** |
| bin_matmul_1024 | 22.671 ± 7.046 | 17.812 ± 3.905 | **14.070 ± 5.777** | 60.383 ± 10.831 | 67.954 ± 19.148 |
| bin_matmul_256 | 0.191 ± 0.399 | **0.136 ± 0.014** | 0.178 ± 0.013 | 1.505 ± 0.301 | 1.667 ± 0.219 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.024 ± 0.006 | **0.004 ± 0.000** | 0.020 ± 0.005 | 0.096 ± 0.019 | 0.016 ± 0.016 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.028 ± 0.002 | **0.025 ± 0.001** | 0.086 ± 0.005 | 0.204 ± 0.142 | 0.247 ± 0.146 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.007 ± 0.000** | 0.025 ± 0.011 | 0.327 ± 0.258 | 0.028 ± 0.003 |
| bin_outer_product_4096 | 7.258 ± 1.689 | 6.592 ± 1.277 | **2.915 ± 0.242** | 4.343 ± 0.899 | 245.290 ± 32.948 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 26.187 ± 7.642 | 17.490 ± 8.566 | **12.437 ± 4.701** | 64.523 ± 2.908 | 75.594 ± 14.312 |
| gm_queen5_5_3.wcsp | 8782.659 ± 1454.137 | 7960.522 ± 449.880 | **4672.776 ± 816.668** | 7797.055 ± 808.665 | 9511.094 ± 1356.838 |
| lm_batch_likelihood_brackets_4_4d | 20.162 ± 4.352 | 24.547 ± 7.705 | **16.166 ± 1.034** | 19.878 ± 2.837 | 46.147 ± 27.254 |
| lm_batch_likelihood_sentence_3_12d | **29.771 ± 8.090** | 58.946 ± 10.383 | 33.030 ± 9.022 | 59.085 ± 3.502 | 94.466 ± 20.434 |
| lm_batch_likelihood_sentence_4_4d | 21.157 ± 4.696 | 33.988 ± 10.644 | 32.259 ± 11.626 | **14.436 ± 0.223** | 43.059 ± 21.930 |
| nary_matmul_chain_64 | 0.033 ± 0.032 | **0.010 ± 0.002** | 0.025 ± 0.002 | 0.140 ± 0.022 | 0.105 ± 0.019 |
| str_matrix_chain_multiplication_100 | 7.202 ± 3.808 | **6.454 ± 3.181** | 6.485 ± 0.937 | 14.015 ± 0.762 | 16.743 ± 18.358 |
| str_mps_varying_inner_product_200 | 10.171 ± 2.526 | 10.685 ± 1.490 | **7.380 ± 0.604** | 17.532 ± 2.288 | 24.071 ± 24.720 |
| str_nw_mera_closed_120 | 550.521 ± 48.575 | 581.789 ± 54.924 | **371.045 ± 21.610** | 1612.412 ± 82.158 | 1886.212 ± 558.121 |
| str_nw_mera_open_26 | 415.445 ± 102.165 | 504.122 ± 96.579 | **236.381 ± 26.112** | 1282.604 ± 189.892 | 1182.287 ± 157.576 |
| tensornetwork_permutation_focus_step409_316 | **202.659 ± 14.023** | 405.468 ± 70.999 | 268.428 ± 19.050 | 424.403 ± 69.558 | 302.282 ± 42.288 |
| tensornetwork_permutation_light_415 | **182.162 ± 40.046** | 185.166 ± 38.039 | 214.051 ± 12.180 | 474.620 ± 39.841 | 266.516 ± 56.071 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.583 ± 0.365 | 0.843 ± 0.097 | **0.669 ± 0.007** | 4.103 ± 0.095 | 8.365 ± 17.352 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.205 ± 0.406 | **0.087 ± 0.012** | 0.134 ± 0.001 | 0.922 ± 0.108 | 3.340 ± 2.083 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.280 ± 0.103 | 0.244 ± 0.042 | **0.193 ± 0.001** | 0.436 ± 0.073 | 0.674 ± 0.521 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.058 ± 0.211 | 0.245 ± 0.057 | **0.193 ± 0.000** | 0.833 ± 0.138 | 2.067 ± 2.168 |
| bin_elementwise_mul_2048x2048 | 3.299 ± 0.982 | 2.996 ± 0.341 | 3.190 ± 1.216 | 4.407 ± 0.175 | **2.331 ± 0.259** |
| bin_elementwise_mul_512x512 | 0.096 ± 0.017 | 0.092 ± 0.042 | 0.111 ± 0.115 | 0.544 ± 0.110 | **0.087 ± 0.020** |
| bin_matmul_1024 | 22.671 ± 7.046 | 17.812 ± 3.905 | **14.070 ± 5.777** | 60.383 ± 10.831 | 67.954 ± 19.148 |
| bin_matmul_256 | 0.191 ± 0.399 | **0.136 ± 0.014** | 0.178 ± 0.013 | 1.505 ± 0.301 | 1.667 ± 0.219 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.024 ± 0.006 | **0.004 ± 0.000** | 0.020 ± 0.005 | 0.096 ± 0.019 | 0.016 ± 0.016 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.028 ± 0.002 | **0.025 ± 0.001** | 0.086 ± 0.005 | 0.204 ± 0.142 | 0.247 ± 0.146 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.007 ± 0.000** | 0.025 ± 0.011 | 0.327 ± 0.258 | 0.028 ± 0.003 |
| bin_outer_product_4096 | 7.258 ± 1.689 | 6.592 ± 1.277 | **2.915 ± 0.242** | 4.343 ± 0.899 | 245.290 ± 32.948 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 26.187 ± 7.642 | 17.490 ± 8.566 | **12.437 ± 4.701** | 64.523 ± 2.908 | 75.594 ± 14.312 |
| gm_queen5_5_3.wcsp | 1208.666 ± 202.595 | 1419.401 ± 381.420 | **960.973 ± 51.044** | 1482.980 ± 438.848 | 2344.621 ± 355.587 |
| lm_batch_likelihood_brackets_4_4d | 20.581 ± 3.722 | 33.723 ± 14.490 | 16.465 ± 0.804 | **13.635 ± 1.011** | 43.244 ± 18.837 |
| lm_batch_likelihood_sentence_3_12d | 44.159 ± 7.579 | 58.311 ± 6.363 | **30.436 ± 2.358** | 58.356 ± 2.924 | 85.872 ± 25.595 |
| lm_batch_likelihood_sentence_4_4d | 20.918 ± 4.352 | 33.763 ± 5.538 | 18.455 ± 2.236 | **14.004 ± 0.447** | 35.506 ± 22.727 |
| nary_matmul_chain_64 | 0.033 ± 0.032 | **0.010 ± 0.002** | 0.025 ± 0.002 | 0.140 ± 0.022 | 0.105 ± 0.019 |
| str_matrix_chain_multiplication_100 | **5.322 ± 0.547** | 6.739 ± 1.346 | 6.512 ± 1.045 | 14.069 ± 0.355 | 15.449 ± 8.667 |
| str_mps_varying_inner_product_200 | 16.395 ± 2.132 | 12.823 ± 1.122 | **10.334 ± 1.227** | 23.176 ± 0.444 | 16.753 ± 7.637 |
| str_nw_mera_closed_120 | 440.270 ± 45.296 | 376.775 ± 6.170 | **342.207 ± 116.132** | 2108.553 ± 597.208 | 1396.496 ± 275.451 |
| str_nw_mera_open_26 | 456.680 ± 88.302 | 639.664 ± 93.129 | **269.749 ± 25.338** | 1660.966 ± 195.703 | 1131.127 ± 208.807 |
| tensornetwork_permutation_focus_step409_316 | **202.659 ± 14.023** | 405.468 ± 70.999 | 268.428 ± 19.050 | 424.403 ± 69.558 | 302.282 ± 42.288 |
| tensornetwork_permutation_light_415 | **182.162 ± 40.046** | 185.166 ± 38.039 | 214.051 ± 12.180 | 474.620 ± 39.841 | 266.516 ± 56.071 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261111_175728/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261111_175728/run.yaml`
- Timestamp: `2026-11-11T17:57:28.429255+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261111_175728`.

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

- `data/results/mac-cpu/cpu/einsum/20261111_175728/jax_cpu_t4_20261111_175728.log`
- `data/results/mac-cpu/cpu/einsum/20261111_175728/julia_omeinsum_t4_20261111_175728.log`
- `data/results/mac-cpu/cpu/einsum/20261111_175728/pytorch_cpu_t4_20261111_175728.log`
- `data/results/mac-cpu/cpu/einsum/20261111_175728/tenferro_eager_t4_20261111_175728.log`
- `data/results/mac-cpu/cpu/einsum/20261111_175728/tenferro_trace_t4_20261111_175728.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.194 ± 0.178 | **0.673 ± 0.034** | 0.685 ± 0.072 | 3.094 ± 0.725 | 136.213 ± 132.544 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.253 ± 0.077 | **0.086 ± 0.002** | 0.137 ± 0.005 | 0.854 ± 0.384 | 0.712 ± 2.382 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.162 ± 0.012** | 0.186 ± 0.015 | 0.233 ± 0.077 | 0.912 ± 0.468 | 0.481 ± 0.367 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.252 ± 0.034 | 0.219 ± 0.075 | **0.192 ± 0.015** | 0.748 ± 0.377 | 1.269 ± 4.318 |
| bin_elementwise_mul_2048x2048 | 2.074 ± 0.238 | **2.021 ± 0.114** | 2.211 ± 0.041 | 3.767 ± 0.132 | 2.569 ± 0.469 |
| bin_elementwise_mul_512x512 | **0.055 ± 0.007** | 0.059 ± 0.015 | 0.088 ± 0.023 | 0.479 ± 0.118 | 0.071 ± 0.004 |
| bin_matmul_1024 | 11.615 ± 1.366 | **9.431 ± 0.325** | 12.039 ± 0.381 | 27.189 ± 3.626 | 49.801 ± 13.462 |
| bin_matmul_256 | 0.155 ± 0.006 | **0.119 ± 0.002** | 0.131 ± 0.007 | 1.348 ± 0.161 | 3.979 ± 4.380 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.005 ± 0.001** | 0.019 ± 0.001 | 0.132 ± 0.121 | 0.010 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.028 ± 0.003 | **0.027 ± 0.001** | 0.078 ± 0.005 | 0.344 ± 0.173 | 0.098 ± 0.021 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.001 | **0.008 ± 0.000** | 0.017 ± 0.001 | 0.218 ± 0.007 | 0.031 ± 0.014 |
| bin_outer_product_4096 | 2.681 ± 0.393 | **2.257 ± 0.231** | 2.733 ± 0.282 | 3.684 ± 0.327 | 257.555 ± 14.564 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 15.805 ± 0.191 | 10.605 ± 0.372 | **10.251 ± 0.202** | 45.092 ± 12.062 | 46.450 ± 18.802 |
| gm_queen5_5_3.wcsp | 4528.866 ± 1267.361 | 5556.360 ± 428.599 | **2535.584 ± 325.540** | 5345.864 ± 1288.723 | 10192.870 ± 768.688 |
| lm_batch_likelihood_brackets_4_4d | **15.206 ± 0.898** | 15.919 ± 9.525 | 20.162 ± 2.696 | 26.777 ± 6.095 | 87.844 ± 28.569 |
| lm_batch_likelihood_sentence_3_12d | **22.417 ± 1.989** | 33.711 ± 2.069 | 38.216 ± 10.894 | 43.289 ± 3.853 | 151.319 ± 24.620 |
| lm_batch_likelihood_sentence_4_4d | **16.113 ± 3.461** | 18.931 ± 1.054 | 25.901 ± 8.123 | 26.734 ± 3.879 | 130.786 ± 31.085 |
| nary_matmul_chain_64 | 0.024 ± 0.008 | **0.011 ± 0.001** | 0.025 ± 0.002 | 0.242 ± 0.285 | 0.225 ± 0.160 |
| str_matrix_chain_multiplication_100 | 5.332 ± 0.549 | **4.688 ± 0.248** | 5.368 ± 0.253 | 16.801 ± 4.090 | 281.713 ± 224.264 |
| str_mps_varying_inner_product_200 | 8.911 ± 0.093 | 8.830 ± 0.139 | **8.751 ± 2.841** | 30.635 ± 8.733 | 36.528 ± 32.645 |
| str_nw_mera_closed_120 | 377.926 ± 30.602 | 378.180 ± 54.557 | **332.685 ± 26.074** | 670.486 ± 33.226 | 1270.418 ± 580.943 |
| str_nw_mera_open_26 | 306.149 ± 15.877 | 374.701 ± 24.573 | **227.499 ± 24.491** | 530.304 ± 23.411 | 856.059 ± 249.293 |
| tensornetwork_permutation_focus_step409_316 | **159.655 ± 19.813** | 294.802 ± 47.926 | 180.028 ± 47.360 | 269.778 ± 23.014 | 445.486 ± 79.382 |
| tensornetwork_permutation_light_415 | 128.430 ± 7.040 | **126.499 ± 4.924** | 174.014 ± 16.788 | 278.670 ± 64.232 | 467.395 ± 52.001 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.194 ± 0.178 | **0.673 ± 0.034** | 0.685 ± 0.072 | 3.094 ± 0.725 | 136.213 ± 132.544 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.253 ± 0.077 | **0.086 ± 0.002** | 0.137 ± 0.005 | 0.854 ± 0.384 | 0.712 ± 2.382 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.162 ± 0.012** | 0.186 ± 0.015 | 0.233 ± 0.077 | 0.912 ± 0.468 | 0.481 ± 0.367 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.252 ± 0.034 | 0.219 ± 0.075 | **0.192 ± 0.015** | 0.748 ± 0.377 | 1.269 ± 4.318 |
| bin_elementwise_mul_2048x2048 | 2.074 ± 0.238 | **2.021 ± 0.114** | 2.211 ± 0.041 | 3.767 ± 0.132 | 2.569 ± 0.469 |
| bin_elementwise_mul_512x512 | **0.055 ± 0.007** | 0.059 ± 0.015 | 0.088 ± 0.023 | 0.479 ± 0.118 | 0.071 ± 0.004 |
| bin_matmul_1024 | 11.615 ± 1.366 | **9.431 ± 0.325** | 12.039 ± 0.381 | 27.189 ± 3.626 | 49.801 ± 13.462 |
| bin_matmul_256 | 0.155 ± 0.006 | **0.119 ± 0.002** | 0.131 ± 0.007 | 1.348 ± 0.161 | 3.979 ± 4.380 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.005 ± 0.001** | 0.019 ± 0.001 | 0.132 ± 0.121 | 0.010 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.028 ± 0.003 | **0.027 ± 0.001** | 0.078 ± 0.005 | 0.344 ± 0.173 | 0.098 ± 0.021 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.001 | **0.008 ± 0.000** | 0.017 ± 0.001 | 0.218 ± 0.007 | 0.031 ± 0.014 |
| bin_outer_product_4096 | 2.681 ± 0.393 | **2.257 ± 0.231** | 2.733 ± 0.282 | 3.684 ± 0.327 | 257.555 ± 14.564 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 15.805 ± 0.191 | 10.605 ± 0.372 | **10.251 ± 0.202** | 45.092 ± 12.062 | 46.450 ± 18.802 |
| gm_queen5_5_3.wcsp | 723.899 ± 58.575 | **632.102 ± 45.962** | 721.328 ± 106.537 | 899.929 ± 139.163 | 1721.750 ± 525.521 |
| lm_batch_likelihood_brackets_4_4d | **17.967 ± 4.298** | 18.729 ± 6.204 | 25.392 ± 4.118 | 22.722 ± 4.696 | 105.313 ± 21.799 |
| lm_batch_likelihood_sentence_3_12d | **23.094 ± 1.190** | 33.817 ± 1.290 | 39.270 ± 12.772 | 41.740 ± 4.599 | 117.576 ± 12.469 |
| lm_batch_likelihood_sentence_4_4d | **13.078 ± 1.515** | 20.456 ± 1.461 | 23.327 ± 9.269 | 23.875 ± 6.627 | 118.827 ± 32.498 |
| nary_matmul_chain_64 | 0.024 ± 0.008 | **0.011 ± 0.001** | 0.025 ± 0.002 | 0.242 ± 0.285 | 0.225 ± 0.160 |
| str_matrix_chain_multiplication_100 | 4.697 ± 0.261 | **4.609 ± 0.043** | 5.637 ± 0.340 | 17.930 ± 3.302 | 284.792 ± 44.755 |
| str_mps_varying_inner_product_200 | 10.613 ± 0.257 | **9.832 ± 0.169** | 15.011 ± 4.471 | 35.641 ± 6.162 | 28.561 ± 17.777 |
| str_nw_mera_closed_120 | 317.541 ± 12.096 | **296.810 ± 73.090** | 361.501 ± 46.154 | 641.591 ± 44.157 | 1095.002 ± 198.088 |
| str_nw_mera_open_26 | **297.505 ± 14.383** | 400.683 ± 19.339 | 327.457 ± 43.455 | 549.706 ± 87.975 | 1114.089 ± 357.287 |
| tensornetwork_permutation_focus_step409_316 | **159.655 ± 19.813** | 294.802 ± 47.926 | 180.028 ± 47.360 | 269.778 ± 23.014 | 445.486 ± 79.382 |
| tensornetwork_permutation_light_415 | 128.430 ± 7.040 | **126.499 ± 4.924** | 174.014 ± 16.788 | 278.670 ± 64.232 | 467.395 ± 52.001 |

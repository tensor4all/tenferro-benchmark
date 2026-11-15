# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `9e383937df4e365bd68bf5b16dbfcaaa269bcc82`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261115_172247/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261115_172247/run.yaml`
- Timestamp: `2026-11-15T17:22:47.463268+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261115_172247`.

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

- `data/results/mac-cpu/cpu/einsum/20261115_172247/jax_cpu_t1_20261115_172247.log`
- `data/results/mac-cpu/cpu/einsum/20261115_172247/julia_omeinsum_t1_20261115_172247.log`
- `data/results/mac-cpu/cpu/einsum/20261115_172247/pytorch_cpu_t1_20261115_172247.log`
- `data/results/mac-cpu/cpu/einsum/20261115_172247/tenferro_eager_t1_20261115_172247.log`
- `data/results/mac-cpu/cpu/einsum/20261115_172247/tenferro_trace_t1_20261115_172247.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.647 ± 0.268 | 3.304 ± 1.768 | **0.881 ± 0.102** | 4.079 ± 0.157 | 4.795 ± 1.163 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.188 ± 0.088 | **0.084 ± 0.006** | 0.138 ± 0.018 | 0.833 ± 0.218 | 1.062 ± 1.630 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.267 ± 0.089 | 0.710 ± 0.484 | **0.216 ± 0.039** | 0.448 ± 0.090 | 0.816 ± 0.345 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.179 ± 0.648 | 0.250 ± 0.094 | **0.223 ± 0.089** | 0.649 ± 0.140 | 1.440 ± 6.665 |
| bin_elementwise_mul_2048x2048 | 3.112 ± 0.669 | 3.239 ± 1.778 | **2.619 ± 0.192** | 3.871 ± 0.408 | 2.780 ± 0.347 |
| bin_elementwise_mul_512x512 | 0.104 ± 0.188 | 0.081 ± 0.014 | 0.083 ± 0.064 | 0.453 ± 0.117 | **0.078 ± 0.014** |
| bin_matmul_1024 | 24.741 ± 7.253 | 14.151 ± 4.871 | **13.631 ± 1.571** | 55.628 ± 6.304 | 63.800 ± 2.652 |
| bin_matmul_256 | 0.163 ± 0.006 | **0.120 ± 0.026** | 0.129 ± 0.005 | 1.023 ± 0.077 | 1.148 ± 0.280 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.020 ± 0.001 | **0.004 ± 0.000** | 0.019 ± 0.003 | 0.094 ± 0.014 | 0.011 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.090 ± 0.030 | **0.025 ± 0.001** | 0.079 ± 0.005 | 0.209 ± 0.151 | 0.085 ± 0.012 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.007 ± 0.000** | 0.017 ± 0.008 | 0.113 ± 0.082 | 0.035 ± 0.005 |
| bin_outer_product_4096 | 7.527 ± 3.598 | 5.493 ± 0.221 | **3.023 ± 0.063** | 3.061 ± 0.338 | 240.558 ± 34.913 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 17.915 ± 2.362 | 15.145 ± 4.556 | **10.841 ± 0.558** | 59.272 ± 5.316 | 63.255 ± 10.862 |
| gm_queen5_5_3.wcsp | 9358.704 ± 1299.713 | 8346.842 ± 3134.669 | **4928.614 ± 2184.433** | 7344.193 ± 1384.403 | 9489.814 ± 1040.604 |
| lm_batch_likelihood_brackets_4_4d | 27.768 ± 18.676 | 35.920 ± 15.829 | 22.721 ± 4.892 | **17.391 ± 4.192** | 53.266 ± 95.949 |
| lm_batch_likelihood_sentence_3_12d | **31.903 ± 7.563** | 74.406 ± 22.069 | 41.948 ± 9.696 | 69.742 ± 6.493 | 69.214 ± 10.281 |
| lm_batch_likelihood_sentence_4_4d | 22.404 ± 3.539 | 37.901 ± 15.479 | 23.676 ± 7.581 | **17.682 ± 0.876** | 26.962 ± 5.066 |
| nary_matmul_chain_64 | 0.018 ± 0.002 | **0.011 ± 0.000** | 0.025 ± 0.002 | 0.222 ± 0.113 | 0.089 ± 0.043 |
| str_matrix_chain_multiplication_100 | **5.968 ± 3.819** | 7.530 ± 3.662 | 9.260 ± 2.808 | 17.747 ± 2.385 | 22.012 ± 8.167 |
| str_mps_varying_inner_product_200 | 11.685 ± 0.916 | 14.115 ± 5.365 | **9.422 ± 1.028** | 18.695 ± 0.596 | 21.143 ± 1.563 |
| str_nw_mera_closed_120 | **518.552 ± 68.864** | 772.548 ± 157.107 | 560.989 ± 62.197 | 2518.840 ± 392.458 | 1824.705 ± 300.629 |
| str_nw_mera_open_26 | 403.409 ± 76.828 | 744.130 ± 165.763 | **398.756 ± 114.038** | 1623.012 ± 256.968 | 1113.115 ± 241.580 |
| tensornetwork_permutation_focus_step409_316 | **203.437 ± 46.221** | 454.542 ± 67.620 | 380.029 ± 63.989 | 524.835 ± 182.724 | 284.980 ± 43.108 |
| tensornetwork_permutation_light_415 | **180.225 ± 13.935** | 194.621 ± 44.067 | 266.087 ± 40.244 | 465.178 ± 60.795 | 280.490 ± 45.823 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.647 ± 0.268 | 3.304 ± 1.768 | **0.881 ± 0.102** | 4.079 ± 0.157 | 4.795 ± 1.163 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.188 ± 0.088 | **0.084 ± 0.006** | 0.138 ± 0.018 | 0.833 ± 0.218 | 1.062 ± 1.630 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.267 ± 0.089 | 0.710 ± 0.484 | **0.216 ± 0.039** | 0.448 ± 0.090 | 0.816 ± 0.345 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.179 ± 0.648 | 0.250 ± 0.094 | **0.223 ± 0.089** | 0.649 ± 0.140 | 1.440 ± 6.665 |
| bin_elementwise_mul_2048x2048 | 3.112 ± 0.669 | 3.239 ± 1.778 | **2.619 ± 0.192** | 3.871 ± 0.408 | 2.780 ± 0.347 |
| bin_elementwise_mul_512x512 | 0.104 ± 0.188 | 0.081 ± 0.014 | 0.083 ± 0.064 | 0.453 ± 0.117 | **0.078 ± 0.014** |
| bin_matmul_1024 | 24.741 ± 7.253 | 14.151 ± 4.871 | **13.631 ± 1.571** | 55.628 ± 6.304 | 63.800 ± 2.652 |
| bin_matmul_256 | 0.163 ± 0.006 | **0.120 ± 0.026** | 0.129 ± 0.005 | 1.023 ± 0.077 | 1.148 ± 0.280 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.020 ± 0.001 | **0.004 ± 0.000** | 0.019 ± 0.003 | 0.094 ± 0.014 | 0.011 ± 0.003 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.090 ± 0.030 | **0.025 ± 0.001** | 0.079 ± 0.005 | 0.209 ± 0.151 | 0.085 ± 0.012 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.007 ± 0.000** | 0.017 ± 0.008 | 0.113 ± 0.082 | 0.035 ± 0.005 |
| bin_outer_product_4096 | 7.527 ± 3.598 | 5.493 ± 0.221 | **3.023 ± 0.063** | 3.061 ± 0.338 | 240.558 ± 34.913 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 17.915 ± 2.362 | 15.145 ± 4.556 | **10.841 ± 0.558** | 59.272 ± 5.316 | 63.255 ± 10.862 |
| gm_queen5_5_3.wcsp | 1218.148 ± 128.256 | 1873.692 ± 588.660 | **1057.572 ± 240.977** | 1400.882 ± 658.260 | 2826.247 ± 592.747 |
| lm_batch_likelihood_brackets_4_4d | 22.703 ± 5.032 | 22.301 ± 9.683 | 17.083 ± 2.381 | **14.272 ± 1.626** | 34.641 ± 13.678 |
| lm_batch_likelihood_sentence_3_12d | 39.459 ± 5.330 | 56.507 ± 8.543 | **26.808 ± 2.706** | 62.864 ± 11.107 | 100.710 ± 47.986 |
| lm_batch_likelihood_sentence_4_4d | 21.331 ± 3.658 | 30.575 ± 16.137 | **15.257 ± 1.504** | 17.254 ± 2.513 | 40.424 ± 40.986 |
| nary_matmul_chain_64 | 0.018 ± 0.002 | **0.011 ± 0.000** | 0.025 ± 0.002 | 0.222 ± 0.113 | 0.089 ± 0.043 |
| str_matrix_chain_multiplication_100 | 6.544 ± 2.385 | 5.753 ± 1.242 | **5.214 ± 0.401** | 17.749 ± 1.350 | 23.184 ± 8.452 |
| str_mps_varying_inner_product_200 | 13.798 ± 1.812 | 18.340 ± 5.780 | **8.803 ± 2.000** | 23.823 ± 0.409 | 24.616 ± 10.137 |
| str_nw_mera_closed_120 | 459.064 ± 105.033 | 397.887 ± 82.261 | **339.754 ± 89.924** | 1747.490 ± 144.042 | 1958.686 ± 307.781 |
| str_nw_mera_open_26 | 415.381 ± 77.451 | 676.155 ± 88.929 | **253.462 ± 19.298** | 1268.991 ± 132.532 | 1604.599 ± 159.970 |
| tensornetwork_permutation_focus_step409_316 | **203.437 ± 46.221** | 454.542 ± 67.620 | 380.029 ± 63.989 | 524.835 ± 182.724 | 284.980 ± 43.108 |
| tensornetwork_permutation_light_415 | **180.225 ± 13.935** | 194.621 ± 44.067 | 266.087 ± 40.244 | 465.178 ± 60.795 | 280.490 ± 45.823 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261115_175655/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261115_175655/run.yaml`
- Timestamp: `2026-11-15T17:56:55.332197+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261115_175655`.

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

- `data/results/mac-cpu/cpu/einsum/20261115_175655/jax_cpu_t4_20261115_175655.log`
- `data/results/mac-cpu/cpu/einsum/20261115_175655/julia_omeinsum_t4_20261115_175655.log`
- `data/results/mac-cpu/cpu/einsum/20261115_175655/pytorch_cpu_t4_20261115_175655.log`
- `data/results/mac-cpu/cpu/einsum/20261115_175655/tenferro_eager_t4_20261115_175655.log`
- `data/results/mac-cpu/cpu/einsum/20261115_175655/tenferro_trace_t4_20261115_175655.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.187 ± 0.149 | **0.648 ± 0.053** | 1.178 ± 0.502 | 1.910 ± 0.322 | 97.270 ± 21.771 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.224 ± 0.013 | 0.378 ± 0.040 | **0.180 ± 0.103** | 0.771 ± 0.122 | 1.633 ± 4.317 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.209 ± 0.047** | 0.242 ± 0.076 | 0.276 ± 0.101 | 0.426 ± 0.097 | 1.587 ± 1.617 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.304 ± 0.029 | **0.245 ± 0.084** | 0.259 ± 0.064 | 0.535 ± 0.129 | 1.918 ± 2.063 |
| bin_elementwise_mul_2048x2048 | **2.366 ± 0.166** | 2.524 ± 0.184 | 2.987 ± 0.478 | 3.788 ± 0.414 | 3.604 ± 2.113 |
| bin_elementwise_mul_512x512 | 0.091 ± 0.033 | **0.073 ± 0.103** | 0.197 ± 0.128 | 0.461 ± 0.165 | 0.091 ± 0.109 |
| bin_matmul_1024 | 12.706 ± 1.261 | **10.032 ± 0.231** | 20.869 ± 4.937 | 23.129 ± 6.043 | 58.301 ± 32.612 |
| bin_matmul_256 | 0.156 ± 0.015 | **0.120 ± 0.002** | 0.139 ± 0.011 | 0.773 ± 0.185 | 7.115 ± 3.510 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | **0.008 ± 0.001** | 0.014 ± 0.001 | 0.022 ± 0.014 | 0.116 ± 0.048 | 0.012 ± 0.004 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.002 | **0.027 ± 0.002** | 0.087 ± 0.061 | 0.288 ± 0.241 | 0.306 ± 0.284 |
| bin_omeinsum_matmul_10x10 | **0.016 ± 0.002** | 0.019 ± 0.001 | 0.017 ± 0.001 | 0.173 ± 0.169 | 0.069 ± 0.020 |
| bin_outer_product_4096 | **2.536 ± 0.214** | 2.964 ± 0.773 | 4.738 ± 2.628 | 4.158 ± 1.398 | 221.762 ± 23.553 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 14.604 ± 0.320 | **11.157 ± 3.306** | 11.997 ± 1.749 | 33.998 ± 3.392 | 43.801 ± 16.589 |
| gm_queen5_5_3.wcsp | 5335.480 ± 1501.173 | 6120.539 ± 630.271 | **1482.794 ± 753.312** | 5292.968 ± 1073.844 | 9150.939 ± 1108.954 |
| lm_batch_likelihood_brackets_4_4d | 18.356 ± 2.057 | 21.131 ± 10.606 | 30.626 ± 5.607 | **17.994 ± 1.272** | 99.885 ± 16.143 |
| lm_batch_likelihood_sentence_3_12d | **23.696 ± 1.692** | 48.889 ± 6.361 | 43.604 ± 5.122 | 29.020 ± 2.872 | 108.568 ± 20.907 |
| lm_batch_likelihood_sentence_4_4d | 18.129 ± 4.818 | 25.631 ± 1.248 | 25.555 ± 5.971 | **15.730 ± 1.061** | 96.937 ± 13.703 |
| nary_matmul_chain_64 | 0.019 ± 0.001 | **0.012 ± 0.004** | 0.026 ± 0.009 | 0.137 ± 0.050 | 0.104 ± 0.054 |
| str_matrix_chain_multiplication_100 | **4.852 ± 0.283** | 5.106 ± 1.279 | 8.170 ± 1.842 | 15.713 ± 0.832 | 303.218 ± 64.040 |
| str_mps_varying_inner_product_200 | **9.485 ± 0.239** | 9.797 ± 1.012 | 14.425 ± 1.891 | 24.260 ± 10.611 | 29.204 ± 15.220 |
| str_nw_mera_closed_120 | 400.097 ± 15.737 | **369.305 ± 24.426** | 479.983 ± 139.397 | 684.460 ± 292.568 | 1407.200 ± 498.080 |
| str_nw_mera_open_26 | 309.844 ± 90.141 | 433.443 ± 92.841 | **296.083 ± 51.874** | 670.126 ± 136.520 | 743.161 ± 196.268 |
| tensornetwork_permutation_focus_step409_316 | **144.507 ± 11.786** | 293.243 ± 35.516 | 248.660 ± 35.064 | 278.617 ± 48.053 | 471.789 ± 107.922 |
| tensornetwork_permutation_light_415 | **139.356 ± 11.658** | 143.620 ± 31.296 | 184.100 ± 26.206 | 267.303 ± 26.703 | 488.311 ± 65.287 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.187 ± 0.149 | **0.648 ± 0.053** | 1.178 ± 0.502 | 1.910 ± 0.322 | 97.270 ± 21.771 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.224 ± 0.013 | 0.378 ± 0.040 | **0.180 ± 0.103** | 0.771 ± 0.122 | 1.633 ± 4.317 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.209 ± 0.047** | 0.242 ± 0.076 | 0.276 ± 0.101 | 0.426 ± 0.097 | 1.587 ± 1.617 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.304 ± 0.029 | **0.245 ± 0.084** | 0.259 ± 0.064 | 0.535 ± 0.129 | 1.918 ± 2.063 |
| bin_elementwise_mul_2048x2048 | **2.366 ± 0.166** | 2.524 ± 0.184 | 2.987 ± 0.478 | 3.788 ± 0.414 | 3.604 ± 2.113 |
| bin_elementwise_mul_512x512 | 0.091 ± 0.033 | **0.073 ± 0.103** | 0.197 ± 0.128 | 0.461 ± 0.165 | 0.091 ± 0.109 |
| bin_matmul_1024 | 12.706 ± 1.261 | **10.032 ± 0.231** | 20.869 ± 4.937 | 23.129 ± 6.043 | 58.301 ± 32.612 |
| bin_matmul_256 | 0.156 ± 0.015 | **0.120 ± 0.002** | 0.139 ± 0.011 | 0.773 ± 0.185 | 7.115 ± 3.510 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | **0.008 ± 0.001** | 0.014 ± 0.001 | 0.022 ± 0.014 | 0.116 ± 0.048 | 0.012 ± 0.004 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.002 | **0.027 ± 0.002** | 0.087 ± 0.061 | 0.288 ± 0.241 | 0.306 ± 0.284 |
| bin_omeinsum_matmul_10x10 | **0.016 ± 0.002** | 0.019 ± 0.001 | 0.017 ± 0.001 | 0.173 ± 0.169 | 0.069 ± 0.020 |
| bin_outer_product_4096 | **2.536 ± 0.214** | 2.964 ± 0.773 | 4.738 ± 2.628 | 4.158 ± 1.398 | 221.762 ± 23.553 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 14.604 ± 0.320 | **11.157 ± 3.306** | 11.997 ± 1.749 | 33.998 ± 3.392 | 43.801 ± 16.589 |
| gm_queen5_5_3.wcsp | 764.522 ± 88.295 | **651.579 ± 32.355** | 675.782 ± 113.404 | 1163.829 ± 184.594 | 2134.011 ± 696.888 |
| lm_batch_likelihood_brackets_4_4d | 19.335 ± 0.622 | 20.394 ± 19.590 | 16.856 ± 3.086 | **13.928 ± 0.497** | 119.000 ± 53.078 |
| lm_batch_likelihood_sentence_3_12d | **27.164 ± 5.667** | 39.446 ± 3.029 | 33.585 ± 9.877 | 31.305 ± 0.773 | 133.298 ± 42.514 |
| lm_batch_likelihood_sentence_4_4d | 19.429 ± 2.969 | 23.011 ± 1.464 | **14.504 ± 8.807** | 15.263 ± 0.295 | 145.927 ± 61.570 |
| nary_matmul_chain_64 | 0.019 ± 0.001 | **0.012 ± 0.004** | 0.026 ± 0.009 | 0.137 ± 0.050 | 0.104 ± 0.054 |
| str_matrix_chain_multiplication_100 | **5.308 ± 0.876** | 6.760 ± 1.273 | 6.622 ± 0.401 | 9.556 ± 0.302 | 271.338 ± 71.077 |
| str_mps_varying_inner_product_200 | 12.871 ± 3.184 | 11.297 ± 1.656 | **10.557 ± 0.517** | 22.471 ± 1.286 | 18.189 ± 22.540 |
| str_nw_mera_closed_120 | 332.194 ± 12.727 | 349.995 ± 96.802 | **291.405 ± 19.407** | 668.928 ± 20.177 | 912.857 ± 191.428 |
| str_nw_mera_open_26 | 326.722 ± 110.631 | 454.155 ± 36.655 | **240.144 ± 37.503** | 834.115 ± 250.962 | 1101.137 ± 303.482 |
| tensornetwork_permutation_focus_step409_316 | **144.507 ± 11.786** | 293.243 ± 35.516 | 248.660 ± 35.064 | 278.617 ± 48.053 | 471.789 ± 107.922 |
| tensornetwork_permutation_light_415 | **139.356 ± 11.658** | 143.620 ± 31.296 | 184.100 ± 26.206 | 267.303 ± 26.703 | 488.311 ± 65.287 |

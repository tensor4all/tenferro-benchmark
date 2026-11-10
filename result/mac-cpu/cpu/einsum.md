# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `15cf8f674203aeca079999fdb339cf4d35801b88`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261110_172255/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261110_172255/run.yaml`
- Timestamp: `2026-11-10T17:22:55.246162+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261110_172255`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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

- `data/results/mac-cpu/cpu/einsum/20261110_172255/jax_cpu_t1_20261110_172255.log`
- `data/results/mac-cpu/cpu/einsum/20261110_172255/julia_omeinsum_t1_20261110_172255.log`
- `data/results/mac-cpu/cpu/einsum/20261110_172255/pytorch_cpu_t1_20261110_172255.log`
- `data/results/mac-cpu/cpu/einsum/20261110_172255/tenferro_eager_t1_20261110_172255.log`
- `data/results/mac-cpu/cpu/einsum/20261110_172255/tenferro_trace_t1_20261110_172255.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.795 ± 0.034 | **0.473 ± 0.041** | 0.585 ± 0.018 | 3.196 ± 0.155 | 5.921 ± 4.378 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.124 ± 0.005 | **0.075 ± 0.003** | 0.124 ± 0.004 | 0.532 ± 0.018 | 0.933 ± 0.423 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.185 ± 0.001** | 0.208 ± 0.037 | 0.227 ± 0.018 | 0.267 ± 0.017 | 0.518 ± 0.069 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.775 ± 0.041 | **0.212 ± 0.022** | 0.222 ± 0.028 | 0.324 ± 0.048 | 0.899 ± 1.015 |
| bin_elementwise_mul_2048x2048 | **1.286 ± 0.207** | 1.557 ± 0.197 | 1.536 ± 0.147 | 2.041 ± 0.128 | 1.437 ± 0.088 |
| bin_elementwise_mul_512x512 | **0.064 ± 0.002** | 0.074 ± 0.020 | 0.088 ± 0.015 | 0.215 ± 0.006 | 0.068 ± 0.011 |
| bin_matmul_1024 | 8.252 ± 0.767 | 7.855 ± 1.081 | **6.246 ± 0.102** | 42.635 ± 0.562 | 52.438 ± 10.808 |
| bin_matmul_256 | 0.129 ± 0.003 | 0.113 ± 0.003 | **0.106 ± 0.005** | 0.806 ± 0.014 | 0.806 ± 0.016 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.006 ± 0.000 | **0.004 ± 0.000** | 0.019 ± 0.005 | 0.074 ± 0.005 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.027 ± 0.003** | 0.028 ± 0.004 | 0.076 ± 0.006 | 0.180 ± 0.074 | 0.063 ± 0.033 |
| bin_omeinsum_matmul_10x10 | 0.012 ± 0.001 | **0.007 ± 0.001** | 0.015 ± 0.001 | 0.082 ± 0.032 | 0.021 ± 0.003 |
| bin_outer_product_4096 | 3.451 ± 0.190 | 4.070 ± 0.336 | 3.182 ± 0.191 | **2.477 ± 0.046** | 118.246 ± 23.383 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 14.167 ± 2.204 | 9.943 ± 0.449 | **8.687 ± 0.492** | 45.223 ± 0.638 | 50.815 ± 15.111 |
| gm_queen5_5_3.wcsp | 2829.034 ± 381.860 | 2722.359 ± 329.371 | **1346.626 ± 19.445** | 2238.683 ± 71.651 | 2296.254 ± 276.430 |
| lm_batch_likelihood_brackets_4_4d | 14.264 ± 2.220 | 20.867 ± 6.696 | 11.125 ± 0.045 | **10.213 ± 0.285** | 18.130 ± 6.035 |
| lm_batch_likelihood_sentence_3_12d | **19.638 ± 3.421** | 47.232 ± 9.618 | 19.823 ± 0.096 | 39.986 ± 0.370 | 84.275 ± 20.254 |
| lm_batch_likelihood_sentence_4_4d | 15.134 ± 1.368 | 20.316 ± 2.066 | **11.566 ± 0.049** | 11.601 ± 0.334 | 42.466 ± 22.129 |
| nary_matmul_chain_64 | 0.018 ± 0.005 | **0.011 ± 0.003** | 0.019 ± 0.001 | 0.133 ± 0.005 | 0.074 ± 0.028 |
| str_matrix_chain_multiplication_100 | 4.433 ± 0.326 | 4.748 ± 0.472 | **3.874 ± 0.016** | 13.899 ± 1.108 | 14.272 ± 1.189 |
| str_mps_varying_inner_product_200 | 8.015 ± 1.149 | 8.414 ± 1.661 | **5.506 ± 0.024** | 17.584 ± 0.723 | 20.194 ± 24.295 |
| str_nw_mera_closed_120 | 349.515 ± 36.843 | 362.200 ± 14.145 | **271.006 ± 17.335** | 1235.691 ± 64.311 | 1292.851 ± 51.197 |
| str_nw_mera_open_26 | 298.846 ± 42.686 | 358.635 ± 21.556 | **187.865 ± 6.497** | 864.923 ± 27.723 | 936.543 ± 109.110 |
| tensornetwork_permutation_focus_step409_316 | **146.500 ± 17.532** | 295.238 ± 25.653 | 226.759 ± 20.316 | 342.151 ± 47.198 | 237.896 ± 48.012 |
| tensornetwork_permutation_light_415 | **133.400 ± 12.994** | 137.591 ± 11.606 | 193.837 ± 23.851 | 410.878 ± 32.212 | 220.804 ± 35.857 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.795 ± 0.034 | **0.473 ± 0.041** | 0.585 ± 0.018 | 3.196 ± 0.155 | 5.921 ± 4.378 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.124 ± 0.005 | **0.075 ± 0.003** | 0.124 ± 0.004 | 0.532 ± 0.018 | 0.933 ± 0.423 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.185 ± 0.001** | 0.208 ± 0.037 | 0.227 ± 0.018 | 0.267 ± 0.017 | 0.518 ± 0.069 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.775 ± 0.041 | **0.212 ± 0.022** | 0.222 ± 0.028 | 0.324 ± 0.048 | 0.899 ± 1.015 |
| bin_elementwise_mul_2048x2048 | **1.286 ± 0.207** | 1.557 ± 0.197 | 1.536 ± 0.147 | 2.041 ± 0.128 | 1.437 ± 0.088 |
| bin_elementwise_mul_512x512 | **0.064 ± 0.002** | 0.074 ± 0.020 | 0.088 ± 0.015 | 0.215 ± 0.006 | 0.068 ± 0.011 |
| bin_matmul_1024 | 8.252 ± 0.767 | 7.855 ± 1.081 | **6.246 ± 0.102** | 42.635 ± 0.562 | 52.438 ± 10.808 |
| bin_matmul_256 | 0.129 ± 0.003 | 0.113 ± 0.003 | **0.106 ± 0.005** | 0.806 ± 0.014 | 0.806 ± 0.016 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.006 ± 0.000 | **0.004 ± 0.000** | 0.019 ± 0.005 | 0.074 ± 0.005 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.027 ± 0.003** | 0.028 ± 0.004 | 0.076 ± 0.006 | 0.180 ± 0.074 | 0.063 ± 0.033 |
| bin_omeinsum_matmul_10x10 | 0.012 ± 0.001 | **0.007 ± 0.001** | 0.015 ± 0.001 | 0.082 ± 0.032 | 0.021 ± 0.003 |
| bin_outer_product_4096 | 3.451 ± 0.190 | 4.070 ± 0.336 | 3.182 ± 0.191 | **2.477 ± 0.046** | 118.246 ± 23.383 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 14.167 ± 2.204 | 9.943 ± 0.449 | **8.687 ± 0.492** | 45.223 ± 0.638 | 50.815 ± 15.111 |
| gm_queen5_5_3.wcsp | 821.852 ± 178.962 | 981.954 ± 134.985 | **603.609 ± 45.778** | 1256.322 ± 92.381 | 957.066 ± 140.530 |
| lm_batch_likelihood_brackets_4_4d | 16.852 ± 1.455 | 16.213 ± 2.707 | 15.197 ± 1.220 | **13.155 ± 0.950** | 32.620 ± 22.461 |
| lm_batch_likelihood_sentence_3_12d | 33.307 ± 3.934 | 34.890 ± 2.596 | **24.073 ± 0.900** | 57.025 ± 4.991 | 75.762 ± 21.463 |
| lm_batch_likelihood_sentence_4_4d | 14.976 ± 5.245 | 17.699 ± 2.428 | **12.481 ± 0.910** | 14.141 ± 1.460 | 32.548 ± 23.803 |
| nary_matmul_chain_64 | 0.018 ± 0.005 | **0.011 ± 0.003** | 0.019 ± 0.001 | 0.133 ± 0.005 | 0.074 ± 0.028 |
| str_matrix_chain_multiplication_100 | 5.348 ± 0.678 | 4.358 ± 0.525 | **3.881 ± 0.023** | 16.745 ± 0.566 | 16.032 ± 16.267 |
| str_mps_varying_inner_product_200 | 12.811 ± 2.808 | 11.433 ± 0.476 | **6.483 ± 0.052** | 26.274 ± 0.950 | 17.178 ± 9.460 |
| str_nw_mera_closed_120 | 317.824 ± 47.832 | 276.711 ± 25.055 | **214.496 ± 6.539** | 1480.979 ± 82.127 | 1211.189 ± 61.259 |
| str_nw_mera_open_26 | 310.572 ± 43.213 | 372.707 ± 10.308 | **195.536 ± 1.909** | 1139.714 ± 54.516 | 965.829 ± 73.257 |
| tensornetwork_permutation_focus_step409_316 | **146.500 ± 17.532** | 295.238 ± 25.653 | 226.759 ± 20.316 | 342.151 ± 47.198 | 237.896 ± 48.012 |
| tensornetwork_permutation_light_415 | **133.400 ± 12.994** | 137.591 ± 11.606 | 193.837 ± 23.851 | 410.878 ± 32.212 | 220.804 ± 35.857 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261110_173954/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261110_173954/run.yaml`
- Timestamp: `2026-11-10T17:39:54.268002+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261110_173954`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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

- `data/results/mac-cpu/cpu/einsum/20261110_173954/jax_cpu_t4_20261110_173954.log`
- `data/results/mac-cpu/cpu/einsum/20261110_173954/julia_omeinsum_t4_20261110_173954.log`
- `data/results/mac-cpu/cpu/einsum/20261110_173954/pytorch_cpu_t4_20261110_173954.log`
- `data/results/mac-cpu/cpu/einsum/20261110_173954/tenferro_eager_t4_20261110_173954.log`
- `data/results/mac-cpu/cpu/einsum/20261110_173954/tenferro_trace_t4_20261110_173954.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.660 ± 0.055 | **0.455 ± 0.119** | 0.577 ± 0.011 | 1.879 ± 0.155 | 3.000 ± 2.912 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.125 ± 0.025 | **0.076 ± 0.002** | 0.113 ± 0.007 | 0.731 ± 0.124 | 0.783 ± 0.958 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.066 ± 0.002** | 0.102 ± 0.010 | 0.161 ± 0.017 | 0.314 ± 0.157 | 0.596 ± 0.221 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.087 ± 0.002** | 0.103 ± 0.004 | 0.156 ± 0.026 | 0.329 ± 0.078 | 1.225 ± 2.750 |
| bin_elementwise_mul_2048x2048 | **0.744 ± 0.077** | 0.884 ± 0.031 | 0.966 ± 0.162 | 1.659 ± 0.131 | 1.528 ± 0.226 |
| bin_elementwise_mul_512x512 | 0.029 ± 0.011 | **0.022 ± 0.001** | 0.091 ± 0.011 | 0.310 ± 0.033 | 0.065 ± 0.005 |
| bin_matmul_1024 | 7.356 ± 0.325 | 7.164 ± 0.650 | **6.916 ± 0.620** | 14.495 ± 1.251 | 18.311 ± 8.479 |
| bin_matmul_256 | 0.147 ± 0.014 | **0.110 ± 0.002** | 0.115 ± 0.001 | 0.537 ± 0.108 | 0.348 ± 0.071 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.004 ± 0.000** | 0.016 ± 0.002 | 0.102 ± 0.012 | 0.008 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.003 | **0.027 ± 0.002** | 0.065 ± 0.002 | 0.145 ± 0.028 | 0.075 ± 0.013 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.002 | **0.008 ± 0.000** | 0.016 ± 0.001 | 0.080 ± 0.037 | 0.016 ± 0.004 |
| bin_outer_product_4096 | 1.897 ± 0.374 | **1.296 ± 0.468** | 1.416 ± 0.330 | 2.034 ± 0.214 | 106.199 ± 8.762 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 12.166 ± 1.817 | 8.637 ± 0.672 | **7.740 ± 0.621** | 16.043 ± 0.574 | 20.183 ± 11.248 |
| gm_queen5_5_3.wcsp | 1911.121 ± 116.322 | 1506.628 ± 213.567 | **697.803 ± 87.321** | 1223.328 ± 93.227 | 1758.308 ± 223.875 |
| lm_batch_likelihood_brackets_4_4d | **9.763 ± 1.516** | 12.850 ± 2.187 | 13.420 ± 0.664 | 14.546 ± 0.595 | 14.168 ± 1.815 |
| lm_batch_likelihood_sentence_3_12d | **15.534 ± 0.543** | 32.280 ± 1.727 | 23.034 ± 1.626 | 24.333 ± 1.630 | 44.377 ± 14.486 |
| lm_batch_likelihood_sentence_4_4d | **9.489 ± 2.281** | 18.779 ± 4.269 | 14.156 ± 0.902 | 15.994 ± 1.127 | 26.010 ± 17.928 |
| nary_matmul_chain_64 | 0.018 ± 0.001 | **0.012 ± 0.000** | 0.023 ± 0.002 | 0.152 ± 0.049 | 0.145 ± 0.099 |
| str_matrix_chain_multiplication_100 | **4.535 ± 0.138** | 4.933 ± 0.248 | 4.871 ± 0.169 | 14.028 ± 1.342 | 8.541 ± 3.892 |
| str_mps_varying_inner_product_200 | 8.879 ± 0.672 | 8.982 ± 0.638 | **8.386 ± 1.148** | 18.885 ± 3.026 | 19.647 ± 5.139 |
| str_nw_mera_closed_120 | 312.024 ± 46.039 | 334.528 ± 25.968 | **263.527 ± 12.960** | 339.910 ± 85.136 | 419.332 ± 57.517 |
| str_nw_mera_open_26 | 272.809 ± 31.253 | 347.537 ± 36.831 | **192.034 ± 4.751** | 266.783 ± 18.220 | 377.809 ± 59.490 |
| tensornetwork_permutation_focus_step409_316 | 129.800 ± 12.230 | 266.169 ± 34.704 | 123.359 ± 10.476 | 191.417 ± 9.022 | **108.236 ± 20.011** |
| tensornetwork_permutation_light_415 | 120.348 ± 5.869 | 118.758 ± 13.549 | 111.137 ± 5.390 | 212.293 ± 32.006 | **109.233 ± 25.518** |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.660 ± 0.055 | **0.455 ± 0.119** | 0.577 ± 0.011 | 1.879 ± 0.155 | 3.000 ± 2.912 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.125 ± 0.025 | **0.076 ± 0.002** | 0.113 ± 0.007 | 0.731 ± 0.124 | 0.783 ± 0.958 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.066 ± 0.002** | 0.102 ± 0.010 | 0.161 ± 0.017 | 0.314 ± 0.157 | 0.596 ± 0.221 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.087 ± 0.002** | 0.103 ± 0.004 | 0.156 ± 0.026 | 0.329 ± 0.078 | 1.225 ± 2.750 |
| bin_elementwise_mul_2048x2048 | **0.744 ± 0.077** | 0.884 ± 0.031 | 0.966 ± 0.162 | 1.659 ± 0.131 | 1.528 ± 0.226 |
| bin_elementwise_mul_512x512 | 0.029 ± 0.011 | **0.022 ± 0.001** | 0.091 ± 0.011 | 0.310 ± 0.033 | 0.065 ± 0.005 |
| bin_matmul_1024 | 7.356 ± 0.325 | 7.164 ± 0.650 | **6.916 ± 0.620** | 14.495 ± 1.251 | 18.311 ± 8.479 |
| bin_matmul_256 | 0.147 ± 0.014 | **0.110 ± 0.002** | 0.115 ± 0.001 | 0.537 ± 0.108 | 0.348 ± 0.071 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.004 ± 0.000** | 0.016 ± 0.002 | 0.102 ± 0.012 | 0.008 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.003 | **0.027 ± 0.002** | 0.065 ± 0.002 | 0.145 ± 0.028 | 0.075 ± 0.013 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.002 | **0.008 ± 0.000** | 0.016 ± 0.001 | 0.080 ± 0.037 | 0.016 ± 0.004 |
| bin_outer_product_4096 | 1.897 ± 0.374 | **1.296 ± 0.468** | 1.416 ± 0.330 | 2.034 ± 0.214 | 106.199 ± 8.762 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 12.166 ± 1.817 | 8.637 ± 0.672 | **7.740 ± 0.621** | 16.043 ± 0.574 | 20.183 ± 11.248 |
| gm_queen5_5_3.wcsp | 526.104 ± 61.861 | 479.854 ± 68.043 | **409.913 ± 21.135** | 456.881 ± 18.015 | 533.972 ± 81.445 |
| lm_batch_likelihood_brackets_4_4d | **9.764 ± 0.375** | 14.358 ± 3.939 | 14.411 ± 1.725 | 13.605 ± 1.074 | 20.117 ± 11.914 |
| lm_batch_likelihood_sentence_3_12d | **19.832 ± 3.383** | 38.223 ± 3.968 | 22.990 ± 1.426 | 22.056 ± 1.612 | 42.706 ± 16.891 |
| lm_batch_likelihood_sentence_4_4d | **10.257 ± 3.021** | 19.683 ± 3.914 | 15.131 ± 0.909 | 13.924 ± 1.085 | 19.400 ± 14.352 |
| nary_matmul_chain_64 | 0.018 ± 0.001 | **0.012 ± 0.000** | 0.023 ± 0.002 | 0.152 ± 0.049 | 0.145 ± 0.099 |
| str_matrix_chain_multiplication_100 | 4.759 ± 0.317 | **4.196 ± 0.234** | 5.018 ± 0.315 | 12.184 ± 3.345 | 9.773 ± 3.989 |
| str_mps_varying_inner_product_200 | 10.640 ± 0.949 | **9.992 ± 0.606** | 11.114 ± 1.462 | 22.156 ± 1.685 | 14.351 ± 5.377 |
| str_nw_mera_closed_120 | 245.565 ± 40.491 | 244.048 ± 10.660 | **229.209 ± 19.316** | 328.555 ± 40.037 | 384.740 ± 80.486 |
| str_nw_mera_open_26 | 284.455 ± 19.369 | 295.558 ± 15.984 | **194.239 ± 7.305** | 260.530 ± 8.657 | 382.160 ± 37.133 |
| tensornetwork_permutation_focus_step409_316 | 129.800 ± 12.230 | 266.169 ± 34.704 | 123.359 ± 10.476 | 191.417 ± 9.022 | **108.236 ± 20.011** |
| tensornetwork_permutation_light_415 | 120.348 ± 5.869 | 118.758 ± 13.549 | 111.137 ± 5.390 | 212.293 ± 32.006 | **109.233 ± 25.518** |

# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `f3a05ec9a2a7262482348fd5bddccbe90389866b`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261113_172248/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261113_172248/run.yaml`
- Timestamp: `2026-11-13T17:22:48.940279+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261113_172248`.

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

- `data/results/mac-cpu/cpu/einsum/20261113_172248/jax_cpu_t1_20261113_172248.log`
- `data/results/mac-cpu/cpu/einsum/20261113_172248/julia_omeinsum_t1_20261113_172248.log`
- `data/results/mac-cpu/cpu/einsum/20261113_172248/pytorch_cpu_t1_20261113_172248.log`
- `data/results/mac-cpu/cpu/einsum/20261113_172248/tenferro_eager_t1_20261113_172248.log`
- `data/results/mac-cpu/cpu/einsum/20261113_172248/tenferro_trace_t1_20261113_172248.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.753 ± 1.029 | **0.754 ± 0.148** | 0.961 ± 0.193 | 4.580 ± 1.204 | 8.050 ± 7.220 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.218 ± 0.037 | **0.093 ± 0.013** | 0.135 ± 0.011 | 0.750 ± 0.191 | 1.074 ± 1.178 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.566 ± 0.589 | 0.297 ± 0.257 | **0.252 ± 0.055** | 0.524 ± 0.188 | 1.460 ± 1.431 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.189 ± 1.400 | 0.261 ± 0.035 | **0.245 ± 0.115** | 0.582 ± 0.198 | 2.052 ± 3.003 |
| bin_elementwise_mul_2048x2048 | **2.835 ± 0.316** | 3.258 ± 0.948 | 3.605 ± 1.784 | 3.619 ± 0.184 | **2.835 ± 0.389** |
| bin_elementwise_mul_512x512 | 0.081 ± 0.008 | 0.199 ± 0.160 | 0.412 ± 0.093 | 0.376 ± 0.063 | **0.077 ± 0.016** |
| bin_matmul_1024 | 29.865 ± 9.643 | 17.433 ± 6.847 | **17.337 ± 3.794** | 58.718 ± 7.650 | 69.525 ± 22.712 |
| bin_matmul_256 | 0.173 ± 0.152 | **0.134 ± 0.031** | 0.177 ± 0.063 | 1.043 ± 0.173 | 1.037 ± 0.143 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.001 | **0.004 ± 0.000** | 0.056 ± 0.022 | 0.089 ± 0.016 | 0.012 ± 0.006 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.064 ± 0.002 | **0.042 ± 0.036** | 0.220 ± 0.239 | 0.163 ± 0.036 | 0.092 ± 0.014 |
| bin_omeinsum_matmul_10x10 | 0.014 ± 0.002 | **0.010 ± 0.006** | 0.018 ± 0.004 | 0.131 ± 0.011 | 0.034 ± 0.017 |
| bin_outer_product_4096 | 12.081 ± 5.979 | 7.627 ± 3.090 | 4.731 ± 3.208 | **3.525 ± 0.418** | 272.972 ± 35.852 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 26.859 ± 6.671 | **21.230 ± 8.691** | 24.191 ± 10.029 | 83.329 ± 10.648 | 75.304 ± 29.112 |
| gm_queen5_5_3.wcsp | 10922.372 ± 2498.195 | 11279.203 ± 3887.515 | **5438.793 ± 1683.360** | 7386.127 ± 1013.464 | 11349.298 ± 2487.945 |
| lm_batch_likelihood_brackets_4_4d | 19.820 ± 3.949 | 25.151 ± 8.840 | 20.682 ± 9.277 | **14.755 ± 5.165** | 26.265 ± 15.141 |
| lm_batch_likelihood_sentence_3_12d | **30.356 ± 16.544** | 79.911 ± 44.877 | 52.443 ± 12.078 | 56.648 ± 9.425 | 128.014 ± 51.763 |
| lm_batch_likelihood_sentence_4_4d | 22.996 ± 4.776 | 31.777 ± 7.210 | 27.094 ± 7.053 | **21.523 ± 2.233** | 60.914 ± 45.546 |
| nary_matmul_chain_64 | 0.018 ± 0.002 | **0.010 ± 0.000** | 0.024 ± 0.007 | 0.238 ± 0.130 | 0.125 ± 0.053 |
| str_matrix_chain_multiplication_100 | **6.664 ± 3.106** | 8.510 ± 1.804 | 7.558 ± 1.690 | 16.503 ± 1.090 | 33.032 ± 29.367 |
| str_mps_varying_inner_product_200 | 9.737 ± 1.585 | 12.588 ± 1.872 | **9.597 ± 3.432** | 22.530 ± 7.184 | 32.799 ± 29.459 |
| str_nw_mera_closed_120 | 581.474 ± 80.121 | 645.130 ± 317.626 | **565.581 ± 124.629** | 2186.709 ± 262.397 | 3025.734 ± 394.949 |
| str_nw_mera_open_26 | 565.037 ± 132.880 | 661.132 ± 95.827 | **294.774 ± 55.331** | 1866.644 ± 519.923 | 2421.193 ± 410.380 |
| tensornetwork_permutation_focus_step409_316 | **199.870 ± 67.704** | 473.279 ± 45.405 | 334.425 ± 51.629 | 494.236 ± 55.999 | 336.398 ± 34.096 |
| tensornetwork_permutation_light_415 | **190.979 ± 36.833** | 213.560 ± 48.566 | 266.779 ± 38.705 | 626.861 ± 78.367 | 373.644 ± 71.269 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.753 ± 1.029 | **0.754 ± 0.148** | 0.961 ± 0.193 | 4.580 ± 1.204 | 8.050 ± 7.220 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.218 ± 0.037 | **0.093 ± 0.013** | 0.135 ± 0.011 | 0.750 ± 0.191 | 1.074 ± 1.178 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.566 ± 0.589 | 0.297 ± 0.257 | **0.252 ± 0.055** | 0.524 ± 0.188 | 1.460 ± 1.431 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.189 ± 1.400 | 0.261 ± 0.035 | **0.245 ± 0.115** | 0.582 ± 0.198 | 2.052 ± 3.003 |
| bin_elementwise_mul_2048x2048 | **2.835 ± 0.316** | 3.258 ± 0.948 | 3.605 ± 1.784 | 3.619 ± 0.184 | **2.835 ± 0.389** |
| bin_elementwise_mul_512x512 | 0.081 ± 0.008 | 0.199 ± 0.160 | 0.412 ± 0.093 | 0.376 ± 0.063 | **0.077 ± 0.016** |
| bin_matmul_1024 | 29.865 ± 9.643 | 17.433 ± 6.847 | **17.337 ± 3.794** | 58.718 ± 7.650 | 69.525 ± 22.712 |
| bin_matmul_256 | 0.173 ± 0.152 | **0.134 ± 0.031** | 0.177 ± 0.063 | 1.043 ± 0.173 | 1.037 ± 0.143 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.001 | **0.004 ± 0.000** | 0.056 ± 0.022 | 0.089 ± 0.016 | 0.012 ± 0.006 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.064 ± 0.002 | **0.042 ± 0.036** | 0.220 ± 0.239 | 0.163 ± 0.036 | 0.092 ± 0.014 |
| bin_omeinsum_matmul_10x10 | 0.014 ± 0.002 | **0.010 ± 0.006** | 0.018 ± 0.004 | 0.131 ± 0.011 | 0.034 ± 0.017 |
| bin_outer_product_4096 | 12.081 ± 5.979 | 7.627 ± 3.090 | 4.731 ± 3.208 | **3.525 ± 0.418** | 272.972 ± 35.852 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 26.859 ± 6.671 | **21.230 ± 8.691** | 24.191 ± 10.029 | 83.329 ± 10.648 | 75.304 ± 29.112 |
| gm_queen5_5_3.wcsp | **1334.129 ± 220.080** | 1408.994 ± 326.301 | 1434.234 ± 313.626 | 2422.102 ± 450.677 | 3132.229 ± 460.178 |
| lm_batch_likelihood_brackets_4_4d | 22.772 ± 3.223 | 28.027 ± 24.118 | 24.392 ± 10.059 | **20.562 ± 10.320** | 65.685 ± 50.188 |
| lm_batch_likelihood_sentence_3_12d | 44.346 ± 7.748 | 48.193 ± 9.407 | **36.045 ± 17.618** | 75.302 ± 9.031 | 140.435 ± 52.272 |
| lm_batch_likelihood_sentence_4_4d | 24.801 ± 7.286 | 25.932 ± 6.142 | 23.329 ± 9.374 | **18.282 ± 2.243** | 52.058 ± 40.582 |
| nary_matmul_chain_64 | 0.018 ± 0.002 | **0.010 ± 0.000** | 0.024 ± 0.007 | 0.238 ± 0.130 | 0.125 ± 0.053 |
| str_matrix_chain_multiplication_100 | 8.071 ± 2.979 | **6.270 ± 3.417** | 8.518 ± 5.948 | 18.158 ± 0.844 | 28.976 ± 15.653 |
| str_mps_varying_inner_product_200 | 12.376 ± 6.449 | 14.000 ± 6.431 | **10.520 ± 2.706** | 27.442 ± 1.526 | 29.959 ± 23.140 |
| str_nw_mera_closed_120 | 454.559 ± 121.231 | **323.379 ± 74.025** | 325.524 ± 34.325 | 2243.316 ± 453.387 | 2561.783 ± 359.458 |
| str_nw_mera_open_26 | 556.025 ± 100.908 | 513.886 ± 39.616 | **293.740 ± 96.582** | 1950.942 ± 334.852 | 1526.280 ± 226.380 |
| tensornetwork_permutation_focus_step409_316 | **199.870 ± 67.704** | 473.279 ± 45.405 | 334.425 ± 51.629 | 494.236 ± 55.999 | 336.398 ± 34.096 |
| tensornetwork_permutation_light_415 | **190.979 ± 36.833** | 213.560 ± 48.566 | 266.779 ± 38.705 | 626.861 ± 78.367 | 373.644 ± 71.269 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261113_180121/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261113_180121/run.yaml`
- Timestamp: `2026-11-13T18:01:21.706892+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261113_180121`.

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

- `data/results/mac-cpu/cpu/einsum/20261113_180121/jax_cpu_t4_20261113_180121.log`
- `data/results/mac-cpu/cpu/einsum/20261113_180121/julia_omeinsum_t4_20261113_180121.log`
- `data/results/mac-cpu/cpu/einsum/20261113_180121/pytorch_cpu_t4_20261113_180121.log`
- `data/results/mac-cpu/cpu/einsum/20261113_180121/tenferro_eager_t4_20261113_180121.log`
- `data/results/mac-cpu/cpu/einsum/20261113_180121/tenferro_trace_t4_20261113_180121.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.296 ± 0.128 | **0.945 ± 0.653** | 2.186 ± 1.658 | 3.019 ± 0.822 | 187.421 ± 95.193 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.368 ± 0.385 | **0.088 ± 0.003** | 0.133 ± 0.004 | 1.476 ± 0.759 | 2.333 ± 2.332 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.281 ± 0.050 | **0.249 ± 0.083** | 0.457 ± 0.165 | 0.852 ± 0.267 | 1.081 ± 0.790 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.522 ± 0.268 | **0.319 ± 0.168** | 0.400 ± 0.220 | 1.048 ± 0.415 | 1.570 ± 1.626 |
| bin_elementwise_mul_2048x2048 | **2.191 ± 0.589** | 2.952 ± 0.262 | 2.967 ± 0.320 | 4.406 ± 1.580 | 2.962 ± 0.724 |
| bin_elementwise_mul_512x512 | 0.130 ± 0.033 | 0.192 ± 0.109 | 0.321 ± 0.253 | 0.782 ± 0.457 | **0.086 ± 0.026** |
| bin_matmul_1024 | 13.290 ± 6.572 | **11.840 ± 0.924** | 15.854 ± 1.762 | 40.618 ± 6.736 | 63.572 ± 26.889 |
| bin_matmul_256 | 0.159 ± 0.064 | **0.127 ± 0.008** | 0.137 ± 0.003 | 2.130 ± 0.845 | 3.335 ± 1.422 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.005 | **0.005 ± 0.001** | 0.066 ± 0.098 | 0.144 ± 0.158 | 0.012 ± 0.007 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.032 ± 0.023 | **0.027 ± 0.003** | 0.082 ± 0.007 | 0.416 ± 0.346 | 0.414 ± 0.334 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.001 | **0.008 ± 0.001** | 0.048 ± 0.006 | 0.338 ± 0.425 | 0.096 ± 0.077 |
| bin_outer_product_4096 | **2.884 ± 2.155** | 3.864 ± 1.251 | 3.405 ± 1.181 | 5.695 ± 1.357 | 236.439 ± 20.840 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 17.455 ± 3.941 | **13.209 ± 1.258** | 15.916 ± 5.581 | 40.621 ± 4.522 | 56.294 ± 28.229 |
| gm_queen5_5_3.wcsp | 7916.990 ± 773.526 | 7784.440 ± 684.301 | **4378.545 ± 883.657** | 5855.872 ± 1222.794 | 11025.715 ± 1121.006 |
| lm_batch_likelihood_brackets_4_4d | 29.276 ± 5.467 | **21.437 ± 13.603** | 22.163 ± 5.320 | 27.367 ± 8.550 | 89.466 ± 82.712 |
| lm_batch_likelihood_sentence_3_12d | 40.200 ± 8.168 | 50.088 ± 9.716 | **31.567 ± 3.984** | 44.859 ± 10.422 | 128.694 ± 16.588 |
| lm_batch_likelihood_sentence_4_4d | 20.768 ± 5.189 | 30.753 ± 8.298 | 23.185 ± 7.649 | **18.862 ± 6.852** | 134.972 ± 41.664 |
| nary_matmul_chain_64 | 0.019 ± 0.003 | **0.012 ± 0.001** | 0.024 ± 0.002 | 0.200 ± 0.048 | 0.132 ± 0.056 |
| str_matrix_chain_multiplication_100 | 8.629 ± 3.735 | 6.893 ± 2.329 | **6.275 ± 0.433** | 16.716 ± 5.870 | 301.318 ± 114.510 |
| str_mps_varying_inner_product_200 | 11.945 ± 3.347 | 11.637 ± 1.257 | **10.857 ± 1.251** | 38.940 ± 4.632 | 31.980 ± 15.337 |
| str_nw_mera_closed_120 | 509.519 ± 84.371 | 468.672 ± 170.729 | **406.737 ± 55.991** | 783.812 ± 208.478 | 1500.337 ± 555.701 |
| str_nw_mera_open_26 | 432.980 ± 81.736 | 500.518 ± 109.805 | **257.672 ± 64.761** | 595.324 ± 202.711 | 1246.395 ± 426.141 |
| tensornetwork_permutation_focus_step409_316 | **165.477 ± 18.013** | 311.034 ± 62.585 | 213.599 ± 41.276 | 374.544 ± 118.780 | 479.871 ± 59.369 |
| tensornetwork_permutation_light_415 | 184.164 ± 28.484 | **151.503 ± 5.870** | 186.841 ± 48.508 | 399.782 ± 161.461 | 483.795 ± 104.333 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.296 ± 0.128 | **0.945 ± 0.653** | 2.186 ± 1.658 | 3.019 ± 0.822 | 187.421 ± 95.193 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.368 ± 0.385 | **0.088 ± 0.003** | 0.133 ± 0.004 | 1.476 ± 0.759 | 2.333 ± 2.332 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.281 ± 0.050 | **0.249 ± 0.083** | 0.457 ± 0.165 | 0.852 ± 0.267 | 1.081 ± 0.790 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.522 ± 0.268 | **0.319 ± 0.168** | 0.400 ± 0.220 | 1.048 ± 0.415 | 1.570 ± 1.626 |
| bin_elementwise_mul_2048x2048 | **2.191 ± 0.589** | 2.952 ± 0.262 | 2.967 ± 0.320 | 4.406 ± 1.580 | 2.962 ± 0.724 |
| bin_elementwise_mul_512x512 | 0.130 ± 0.033 | 0.192 ± 0.109 | 0.321 ± 0.253 | 0.782 ± 0.457 | **0.086 ± 0.026** |
| bin_matmul_1024 | 13.290 ± 6.572 | **11.840 ± 0.924** | 15.854 ± 1.762 | 40.618 ± 6.736 | 63.572 ± 26.889 |
| bin_matmul_256 | 0.159 ± 0.064 | **0.127 ± 0.008** | 0.137 ± 0.003 | 2.130 ± 0.845 | 3.335 ± 1.422 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.005 | **0.005 ± 0.001** | 0.066 ± 0.098 | 0.144 ± 0.158 | 0.012 ± 0.007 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.032 ± 0.023 | **0.027 ± 0.003** | 0.082 ± 0.007 | 0.416 ± 0.346 | 0.414 ± 0.334 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.001 | **0.008 ± 0.001** | 0.048 ± 0.006 | 0.338 ± 0.425 | 0.096 ± 0.077 |
| bin_outer_product_4096 | **2.884 ± 2.155** | 3.864 ± 1.251 | 3.405 ± 1.181 | 5.695 ± 1.357 | 236.439 ± 20.840 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 17.455 ± 3.941 | **13.209 ± 1.258** | 15.916 ± 5.581 | 40.621 ± 4.522 | 56.294 ± 28.229 |
| gm_queen5_5_3.wcsp | 905.654 ± 270.089 | 891.854 ± 581.036 | **656.203 ± 153.130** | 1239.545 ± 716.840 | 2586.789 ± 637.408 |
| lm_batch_likelihood_brackets_4_4d | 20.897 ± 2.271 | 43.181 ± 7.988 | **20.283 ± 2.075** | 26.972 ± 12.562 | 176.761 ± 60.888 |
| lm_batch_likelihood_sentence_3_12d | **27.217 ± 1.176** | 59.736 ± 7.593 | 29.476 ± 3.621 | 53.672 ± 11.425 | 161.320 ± 34.998 |
| lm_batch_likelihood_sentence_4_4d | **18.528 ± 0.693** | 28.138 ± 7.090 | 19.078 ± 7.692 | 32.476 ± 4.571 | 128.465 ± 38.658 |
| nary_matmul_chain_64 | 0.019 ± 0.003 | **0.012 ± 0.001** | 0.024 ± 0.002 | 0.200 ± 0.048 | 0.132 ± 0.056 |
| str_matrix_chain_multiplication_100 | 5.564 ± 0.518 | 6.141 ± 1.310 | **5.322 ± 0.488** | 21.370 ± 6.417 | 288.692 ± 54.028 |
| str_mps_varying_inner_product_200 | 12.265 ± 1.407 | **10.968 ± 0.931** | 14.800 ± 3.556 | 37.187 ± 3.799 | 70.758 ± 18.630 |
| str_nw_mera_closed_120 | 393.120 ± 83.049 | 387.526 ± 118.309 | **328.785 ± 208.895** | 702.846 ± 58.078 | 985.847 ± 236.416 |
| str_nw_mera_open_26 | 345.282 ± 73.393 | 542.098 ± 133.505 | **264.005 ± 31.262** | 587.581 ± 24.199 | 1057.590 ± 260.770 |
| tensornetwork_permutation_focus_step409_316 | **165.477 ± 18.013** | 311.034 ± 62.585 | 213.599 ± 41.276 | 374.544 ± 118.780 | 479.871 ± 59.369 |
| tensornetwork_permutation_light_415 | 184.164 ± 28.484 | **151.503 ± 5.870** | 186.841 ± 48.508 | 399.782 ± 161.461 | 483.795 ± 104.333 |

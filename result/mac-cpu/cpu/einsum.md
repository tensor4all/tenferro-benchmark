# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`
- Benchmark commit: `12e0b50f983ca3253ff46841eb15912448ae3e20`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261022_172907/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261022_172907/run.yaml`
- Timestamp: `20261022_172907`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261022_172907`.

- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`

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
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`

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

- Julia: version `1.13.2`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.0.3.30.dylib)`, probe threads `1`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20261022_172907/einsum_table_t1_20261022_172907.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20261022_172907/tenferro_trace_t1_20261022_172907.log`
- `data/results/mac-cpu/cpu/einsum/20261022_172907/tenferro_eager_t1_20261022_172907.log`
- `data/results/mac-cpu/cpu/einsum/20261022_172907/pytorch_cpu_t1_20261022_172907.log`
- `data/results/mac-cpu/cpu/einsum/20261022_172907/jax_cpu_t1_20261022_172907.log`
- `data/results/mac-cpu/cpu/einsum/20261022_172907/julia_omeinsum_t1_20261022_172907.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.592 ± 0.253 | **0.635 ± 0.038** | 0.674 ± 0.049 | 3.913 ± 0.062 | 5.121 ± 5.992 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.176 ± 0.069 | **0.088 ± 0.008** | 0.123 ± 0.002 | 0.696 ± 0.059 | 1.985 ± 4.263 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.157 ± 0.028 | **0.141 ± 0.010** | 0.194 ± 0.013 | 0.449 ± 0.033 | 0.864 ± 0.609 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.445 ± 0.105 | **0.150 ± 0.020** | 0.193 ± 0.001 | 0.489 ± 0.037 | 1.512 ± 9.446 |
| bin_elementwise_mul_2048x2048 | 2.302 ± 0.109 | 2.237 ± 0.116 | **2.127 ± 0.092** | 3.132 ± 0.202 | 3.349 ± 1.551 |
| bin_elementwise_mul_512x512 | **0.071 ± 0.012** | 0.074 ± 0.003 | 0.084 ± 0.002 | 0.273 ± 0.044 | 0.084 ± 0.030 |
| bin_matmul_1024 | 16.448 ± 7.326 | **10.980 ± 0.851** | 11.199 ± 4.115 | 54.160 ± 2.650 | 66.305 ± 11.054 |
| bin_matmul_256 | 0.207 ± 0.062 | **0.133 ± 0.007** | 0.144 ± 0.038 | 1.060 ± 0.098 | 0.992 ± 0.300 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.004 ± 0.000** | 0.019 ± 0.001 | 0.110 ± 0.011 | 0.013 ± 0.005 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.004 | **0.026 ± 0.002** | 0.080 ± 0.002 | 0.255 ± 0.101 | 0.079 ± 0.006 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.007 ± 0.000** | 0.018 ± 0.001 | 0.138 ± 0.054 | 0.035 ± 0.002 |
| bin_outer_product_4096 | 2.853 ± 1.735 | **2.494 ± 0.562** | 3.313 ± 0.571 | 2.522 ± 0.257 | 258.284 ± 40.041 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 18.039 ± 4.095 | 12.039 ± 1.769 | **11.341 ± 1.263** | 55.814 ± 2.606 | 63.829 ± 16.826 |
| gm_queen5_5_3.wcsp | 7751.510 ± 552.583 | 6806.786 ± 822.027 | **3881.001 ± 827.038** | 5562.704 ± 4575.591 | 10961.112 ± 1079.354 |
| lm_batch_likelihood_brackets_4_4d | 19.846 ± 4.559 | 30.076 ± 20.503 | 17.141 ± 2.787 | **15.754 ± 1.679** | 30.031 ± 14.844 |
| lm_batch_likelihood_sentence_3_12d | **23.951 ± 1.950** | 51.682 ± 8.303 | 29.834 ± 6.267 | 60.136 ± 15.212 | 137.064 ± 43.461 |
| lm_batch_likelihood_sentence_4_4d | 14.430 ± 2.350 | 20.568 ± 1.079 | **14.411 ± 0.151** | 20.045 ± 5.021 | 64.817 ± 30.465 |
| nary_matmul_chain_64 | 0.056 ± 0.006 | **0.010 ± 0.001** | 0.024 ± 0.003 | 0.134 ± 0.065 | 0.113 ± 0.021 |
| str_matrix_chain_multiplication_100 | 5.491 ± 0.584 | 5.037 ± 0.538 | **4.906 ± 0.468** | 15.895 ± 0.723 | 22.297 ± 7.929 |
| str_mps_varying_inner_product_200 | 10.503 ± 3.757 | 10.255 ± 0.818 | **7.218 ± 0.400** | 19.392 ± 1.013 | 30.627 ± 16.317 |
| str_nw_mera_closed_120 | 506.243 ± 73.843 | 413.580 ± 42.672 | **396.391 ± 90.041** | 2009.564 ± 321.243 | 2534.950 ± 395.978 |
| str_nw_mera_open_26 | 405.335 ± 54.048 | 448.367 ± 46.024 | **263.754 ± 34.539** | 1447.461 ± 285.149 | 1841.176 ± 578.126 |
| tensornetwork_permutation_focus_step409_316 | **176.745 ± 32.040** | 307.084 ± 18.898 | 280.844 ± 27.941 | 422.678 ± 37.809 | 303.230 ± 65.602 |
| tensornetwork_permutation_light_415 | **143.584 ± 7.344** | 146.844 ± 11.140 | 219.680 ± 6.703 | 422.659 ± 73.328 | 271.658 ± 75.883 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.592 ± 0.253 | **0.635 ± 0.038** | 0.674 ± 0.049 | 3.913 ± 0.062 | 5.121 ± 5.992 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.176 ± 0.069 | **0.088 ± 0.008** | 0.123 ± 0.002 | 0.696 ± 0.059 | 1.985 ± 4.263 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.157 ± 0.028 | **0.141 ± 0.010** | 0.194 ± 0.013 | 0.449 ± 0.033 | 0.864 ± 0.609 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.445 ± 0.105 | **0.150 ± 0.020** | 0.193 ± 0.001 | 0.489 ± 0.037 | 1.512 ± 9.446 |
| bin_elementwise_mul_2048x2048 | 2.302 ± 0.109 | 2.237 ± 0.116 | **2.127 ± 0.092** | 3.132 ± 0.202 | 3.349 ± 1.551 |
| bin_elementwise_mul_512x512 | **0.071 ± 0.012** | 0.074 ± 0.003 | 0.084 ± 0.002 | 0.273 ± 0.044 | 0.084 ± 0.030 |
| bin_matmul_1024 | 16.448 ± 7.326 | **10.980 ± 0.851** | 11.199 ± 4.115 | 54.160 ± 2.650 | 66.305 ± 11.054 |
| bin_matmul_256 | 0.207 ± 0.062 | **0.133 ± 0.007** | 0.144 ± 0.038 | 1.060 ± 0.098 | 0.992 ± 0.300 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.004 ± 0.000** | 0.019 ± 0.001 | 0.110 ± 0.011 | 0.013 ± 0.005 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.004 | **0.026 ± 0.002** | 0.080 ± 0.002 | 0.255 ± 0.101 | 0.079 ± 0.006 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.007 ± 0.000** | 0.018 ± 0.001 | 0.138 ± 0.054 | 0.035 ± 0.002 |
| bin_outer_product_4096 | 2.853 ± 1.735 | **2.494 ± 0.562** | 3.313 ± 0.571 | 2.522 ± 0.257 | 258.284 ± 40.041 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 18.039 ± 4.095 | 12.039 ± 1.769 | **11.341 ± 1.263** | 55.814 ± 2.606 | 63.829 ± 16.826 |
| gm_queen5_5_3.wcsp | 998.592 ± 94.669 | 818.941 ± 177.643 | **812.127 ± 95.152** | 1411.363 ± 268.180 | 2355.953 ± 183.721 |
| lm_batch_likelihood_brackets_4_4d | 18.139 ± 1.021 | 22.135 ± 12.673 | 15.859 ± 0.373 | **13.354 ± 1.012** | 56.275 ± 31.375 |
| lm_batch_likelihood_sentence_3_12d | 34.374 ± 3.944 | 56.166 ± 8.005 | **30.398 ± 0.934** | 83.673 ± 30.260 | 115.555 ± 44.572 |
| lm_batch_likelihood_sentence_4_4d | 15.055 ± 1.270 | 24.872 ± 8.671 | 17.263 ± 0.459 | **14.047 ± 0.510** | 42.119 ± 34.778 |
| nary_matmul_chain_64 | 0.056 ± 0.006 | **0.010 ± 0.001** | 0.024 ± 0.003 | 0.134 ± 0.065 | 0.113 ± 0.021 |
| str_matrix_chain_multiplication_100 | 5.752 ± 0.559 | **4.771 ± 0.132** | 6.129 ± 0.882 | 14.300 ± 0.542 | 21.834 ± 10.376 |
| str_mps_varying_inner_product_200 | 12.527 ± 3.543 | **10.776 ± 0.461** | 12.349 ± 1.713 | 23.924 ± 0.592 | 20.621 ± 7.858 |
| str_nw_mera_closed_120 | 380.454 ± 25.349 | 321.730 ± 33.281 | **283.943 ± 6.799** | 2411.816 ± 600.011 | 1673.975 ± 515.282 |
| str_nw_mera_open_26 | 354.067 ± 48.276 | 443.618 ± 33.208 | **243.913 ± 17.250** | 1512.250 ± 100.211 | 1279.676 ± 393.570 |
| tensornetwork_permutation_focus_step409_316 | **176.745 ± 32.040** | 307.084 ± 18.898 | 280.844 ± 27.941 | 422.678 ± 37.809 | 303.230 ± 65.602 |
| tensornetwork_permutation_light_415 | **143.584 ± 7.344** | 146.844 ± 11.140 | 219.680 ± 6.703 | 422.659 ± 73.328 | 271.658 ± 75.883 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261022_180005/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261022_180005/run.yaml`
- Timestamp: `20261022_180005`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261022_180005`.

- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`

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
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`

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

- Julia: version `1.13.2`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.0.3.30.dylib)`, probe threads `4`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20261022_180005/einsum_table_t4_20261022_180005.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20261022_180005/tenferro_trace_t4_20261022_180005.log`
- `data/results/mac-cpu/cpu/einsum/20261022_180005/tenferro_eager_t4_20261022_180005.log`
- `data/results/mac-cpu/cpu/einsum/20261022_180005/pytorch_cpu_t4_20261022_180005.log`
- `data/results/mac-cpu/cpu/einsum/20261022_180005/jax_cpu_t4_20261022_180005.log`
- `data/results/mac-cpu/cpu/einsum/20261022_180005/julia_omeinsum_t4_20261022_180005.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.176 ± 0.093 | **0.699 ± 0.163** | 3.127 ± 0.956 | 2.996 ± 0.758 | 102.467 ± 18.492 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.179 ± 0.025 | **0.087 ± 0.194** | 0.622 ± 0.529 | 0.665 ± 0.068 | 2.354 ± 3.990 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.213 ± 0.123 | **0.177 ± 0.157** | 0.431 ± 0.185 | 0.648 ± 0.118 | 1.540 ± 6.178 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.386 ± 0.172 | **0.184 ± 0.030** | 0.618 ± 0.417 | 0.434 ± 0.026 | 2.279 ± 6.108 |
| bin_elementwise_mul_2048x2048 | **2.796 ± 0.699** | 2.840 ± 0.246 | 3.845 ± 0.711 | 3.225 ± 0.380 | 4.220 ± 2.175 |
| bin_elementwise_mul_512x512 | 0.158 ± 0.060 | 0.095 ± 0.046 | 0.387 ± 0.092 | 0.485 ± 0.142 | **0.079 ± 0.012** |
| bin_matmul_1024 | 17.786 ± 3.943 | **12.795 ± 1.700** | 38.115 ± 14.269 | 21.018 ± 2.668 | 60.463 ± 22.110 |
| bin_matmul_256 | 0.254 ± 0.247 | **0.122 ± 0.043** | 0.146 ± 0.041 | 1.045 ± 0.253 | 3.721 ± 1.815 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.022 ± 0.020 | **0.004 ± 0.000** | 0.020 ± 0.004 | 0.086 ± 0.006 | 0.012 ± 0.006 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.002 | **0.025 ± 0.001** | 0.088 ± 0.053 | 0.149 ± 0.016 | 0.098 ± 0.012 |
| bin_omeinsum_matmul_10x10 | 0.038 ± 0.008 | **0.007 ± 0.002** | 0.018 ± 0.002 | 0.135 ± 0.090 | 0.018 ± 0.001 |
| bin_outer_product_4096 | 3.279 ± 0.670 | 3.495 ± 1.017 | 12.920 ± 3.935 | **3.252 ± 0.903** | 241.880 ± 12.099 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 15.522 ± 0.855 | **13.066 ± 1.837** | 36.455 ± 13.783 | 20.852 ± 6.788 | 53.438 ± 16.965 |
| gm_queen5_5_3.wcsp | 4901.702 ± 373.199 | 6649.551 ± 266.291 | **1397.816 ± 1531.613** | 4696.918 ± 1784.700 | 10607.257 ± 1631.008 |
| lm_batch_likelihood_brackets_4_4d | **17.354 ± 6.539** | 30.627 ± 12.905 | 27.513 ± 4.635 | 25.688 ± 4.969 | 108.293 ± 40.832 |
| lm_batch_likelihood_sentence_3_12d | **31.847 ± 10.154** | 43.978 ± 4.855 | 36.835 ± 4.807 | 43.915 ± 9.662 | 170.127 ± 29.578 |
| lm_batch_likelihood_sentence_4_4d | **15.013 ± 3.320** | 28.351 ± 2.582 | 25.341 ± 5.459 | 27.552 ± 4.141 | 127.456 ± 50.633 |
| nary_matmul_chain_64 | 0.020 ± 0.003 | **0.010 ± 0.000** | 0.026 ± 0.005 | 0.363 ± 0.126 | 0.119 ± 0.077 |
| str_matrix_chain_multiplication_100 | **5.505 ± 0.457** | 5.912 ± 0.878 | 7.064 ± 1.543 | 20.916 ± 5.429 | 292.593 ± 63.305 |
| str_mps_varying_inner_product_200 | **13.256 ± 3.040** | 14.339 ± 0.933 | 13.436 ± 1.625 | 39.110 ± 7.570 | 36.375 ± 26.115 |
| str_nw_mera_closed_120 | **385.284 ± 46.789** | 450.968 ± 55.100 | 499.189 ± 83.368 | 726.296 ± 50.359 | 1184.690 ± 623.096 |
| str_nw_mera_open_26 | 291.649 ± 17.112 | 449.789 ± 98.867 | **286.913 ± 26.598** | 525.182 ± 63.522 | 916.488 ± 303.740 |
| tensornetwork_permutation_focus_step409_316 | **126.512 ± 15.528** | 197.375 ± 10.828 | 220.917 ± 31.853 | 310.926 ± 37.899 | 489.600 ± 103.874 |
| tensornetwork_permutation_light_415 | 129.087 ± 28.511 | **128.863 ± 18.024** | 210.135 ± 35.140 | 302.120 ± 24.780 | 478.001 ± 71.491 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.176 ± 0.093 | **0.699 ± 0.163** | 3.127 ± 0.956 | 2.996 ± 0.758 | 102.467 ± 18.492 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.179 ± 0.025 | **0.087 ± 0.194** | 0.622 ± 0.529 | 0.665 ± 0.068 | 2.354 ± 3.990 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.213 ± 0.123 | **0.177 ± 0.157** | 0.431 ± 0.185 | 0.648 ± 0.118 | 1.540 ± 6.178 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.386 ± 0.172 | **0.184 ± 0.030** | 0.618 ± 0.417 | 0.434 ± 0.026 | 2.279 ± 6.108 |
| bin_elementwise_mul_2048x2048 | **2.796 ± 0.699** | 2.840 ± 0.246 | 3.845 ± 0.711 | 3.225 ± 0.380 | 4.220 ± 2.175 |
| bin_elementwise_mul_512x512 | 0.158 ± 0.060 | 0.095 ± 0.046 | 0.387 ± 0.092 | 0.485 ± 0.142 | **0.079 ± 0.012** |
| bin_matmul_1024 | 17.786 ± 3.943 | **12.795 ± 1.700** | 38.115 ± 14.269 | 21.018 ± 2.668 | 60.463 ± 22.110 |
| bin_matmul_256 | 0.254 ± 0.247 | **0.122 ± 0.043** | 0.146 ± 0.041 | 1.045 ± 0.253 | 3.721 ± 1.815 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.022 ± 0.020 | **0.004 ± 0.000** | 0.020 ± 0.004 | 0.086 ± 0.006 | 0.012 ± 0.006 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.002 | **0.025 ± 0.001** | 0.088 ± 0.053 | 0.149 ± 0.016 | 0.098 ± 0.012 |
| bin_omeinsum_matmul_10x10 | 0.038 ± 0.008 | **0.007 ± 0.002** | 0.018 ± 0.002 | 0.135 ± 0.090 | 0.018 ± 0.001 |
| bin_outer_product_4096 | 3.279 ± 0.670 | 3.495 ± 1.017 | 12.920 ± 3.935 | **3.252 ± 0.903** | 241.880 ± 12.099 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 15.522 ± 0.855 | **13.066 ± 1.837** | 36.455 ± 13.783 | 20.852 ± 6.788 | 53.438 ± 16.965 |
| gm_queen5_5_3.wcsp | 698.040 ± 99.859 | 740.910 ± 52.300 | **658.189 ± 89.772** | 852.872 ± 161.421 | 2456.641 ± 352.010 |
| lm_batch_likelihood_brackets_4_4d | **16.553 ± 3.023** | 29.734 ± 6.857 | 20.955 ± 5.361 | 23.406 ± 3.707 | 126.320 ± 34.494 |
| lm_batch_likelihood_sentence_3_12d | **26.531 ± 2.338** | 55.214 ± 14.847 | 32.643 ± 6.095 | 40.926 ± 3.015 | 148.260 ± 28.148 |
| lm_batch_likelihood_sentence_4_4d | **12.923 ± 0.921** | 39.486 ± 7.644 | 20.052 ± 2.429 | 26.352 ± 8.010 | 137.399 ± 28.744 |
| nary_matmul_chain_64 | 0.020 ± 0.003 | **0.010 ± 0.000** | 0.026 ± 0.005 | 0.363 ± 0.126 | 0.119 ± 0.077 |
| str_matrix_chain_multiplication_100 | 5.707 ± 0.326 | 6.944 ± 3.040 | **5.431 ± 0.295** | 19.324 ± 2.879 | 300.136 ± 253.991 |
| str_mps_varying_inner_product_200 | 17.936 ± 1.215 | 16.163 ± 4.346 | **13.438 ± 3.260** | 36.797 ± 3.854 | 34.414 ± 14.924 |
| str_nw_mera_closed_120 | 343.646 ± 58.080 | 471.105 ± 52.975 | **304.502 ± 44.461** | 612.492 ± 56.978 | 1052.619 ± 336.809 |
| str_nw_mera_open_26 | 321.272 ± 7.044 | 538.984 ± 102.372 | **259.059 ± 29.643** | 590.105 ± 175.670 | 893.905 ± 179.666 |
| tensornetwork_permutation_focus_step409_316 | **126.512 ± 15.528** | 197.375 ± 10.828 | 220.917 ± 31.853 | 310.926 ± 37.899 | 489.600 ± 103.874 |
| tensornetwork_permutation_light_415 | 129.087 ± 28.511 | **128.863 ± 18.024** | 210.135 ± 35.140 | 302.120 ± 24.780 | 478.001 ± 71.491 |

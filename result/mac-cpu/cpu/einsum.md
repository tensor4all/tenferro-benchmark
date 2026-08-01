# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260801_221400/run.yaml`
- Timestamp: `20260801_221400`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260801_221400`.

- tenferro-rs commit: `0ee2d0dc2f8d21ff62ea682f90f34e4319108ace`

## CPU Information

- Model: `Apple M4`
- Vendor: `Apple`
- Logical CPUs: `10`
- Physical CPUs: `10`
- Sockets: `1`
- Cores per socket: `10`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Performance: 4 physical / 4 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 4 CPUs/L2); Efficiency: 6 physical / 6 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.2-arm64-arm-64bit`

## Thread Environment

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

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-accelerate`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Julia / OMEinsum.jl Backend

- Julia: version `1.12.6`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.dylib)`, probe threads `4`

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

## Threads: 4

- Source table: `data/results/mac-cpu/cpu/einsum/20260801_221400/einsum_table_t4_20260801_221400.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260801_221400/tenferro_trace_t4_20260801_221400.log`
- `data/results/mac-cpu/cpu/einsum/20260801_221400/tenferro_eager_t4_20260801_221400.log`
- `data/results/mac-cpu/cpu/einsum/20260801_221400/pytorch_cpu_t4_20260801_221400.log`
- `data/results/mac-cpu/cpu/einsum/20260801_221400/jax_cpu_t4_20260801_221400.log`
- `data/results/mac-cpu/cpu/einsum/20260801_221400/julia_omeinsum_t4_20260801_221400.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.835 ± 0.095 | 0.667 ± 0.092 | **0.407 ± 0.004** | 0.619 ± 0.024 | 2.018 ± 0.724 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.174 ± 0.024 | 0.116 ± 0.015 | **0.102 ± 0.003** | 0.188 ± 0.033 | 0.596 ± 0.177 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.120 ± 0.019 | 0.117 ± 0.015 | **0.094 ± 0.007** | 0.141 ± 0.029 | 0.367 ± 0.097 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.169 ± 0.008 | 0.118 ± 0.004 | **0.095 ± 0.003** | 0.140 ± 0.027 | 0.755 ± 0.551 |
| bin_elementwise_mul_2048x2048 | 1.108 ± 0.033 | **1.092 ± 0.013** | **1.092 ± 0.012** | 1.525 ± 0.026 | 1.224 ± 0.113 |
| bin_matmul_1024 | 5.243 ± 0.042 | 4.646 ± 0.039 | **4.342 ± 0.010** | 7.396 ± 0.368 | 17.000 ± 6.127 |
| bin_matmul_256 | 0.136 ± 0.005 | 0.091 ± 0.002 | **0.086 ± 0.001** | 0.288 ± 0.039 | 0.390 ± 0.048 |
| bin_outer_product_4096 | **1.764 ± 0.005** | 1.782 ± 0.028 | 1.768 ± 0.008 | 1.846 ± 0.018 | 78.934 ± 16.739 |
| gm_queen5_5_3.wcsp | 898.057 ± 60.010 | 835.821 ± 74.044 | **598.933 ± 69.958** | 694.957 ± 168.138 | 1694.200 ± 24.485 |
| lm_batch_likelihood_brackets_4_4d | 10.305 ± 0.164 | 15.054 ± 1.500 | 9.460 ± 0.415 | **6.969 ± 0.146** | 15.703 ± 0.785 |
| lm_batch_likelihood_sentence_3_12d | 17.683 ± 3.277 | 31.504 ± 3.687 | 24.336 ± 0.827 | **11.542 ± 0.145** | 47.730 ± 27.005 |
| lm_batch_likelihood_sentence_4_4d | 9.595 ± 0.246 | 15.221 ± 0.632 | 7.963 ± 0.243 | **7.025 ± 0.143** | 25.931 ± 23.008 |
| nary_matmul_chain_64 | 0.039 ± 0.003 | 0.039 ± 0.007 | **0.031 ± 0.001** | 0.086 ± 0.003 | 0.073 ± 0.002 |
| str_matrix_chain_multiplication_100 | 3.210 ± 0.018 | 4.039 ± 0.121 | **3.103 ± 0.047** | 6.994 ± 0.088 | 9.262 ± 0.226 |
| str_mps_varying_inner_product_200 | 7.948 ± 0.198 | 12.503 ± 0.213 | **6.285 ± 0.073** | 14.810 ± 0.096 | 18.028 ± 7.115 |
| str_nw_mera_closed_120 | 200.892 ± 4.294 | 186.886 ± 9.613 | **151.330 ± 1.423** | 347.413 ± 122.168 | 664.842 ± 68.918 |
| str_nw_mera_open_26 | 191.758 ± 5.087 | 185.050 ± 20.015 | **121.476 ± 2.389** | 268.099 ± 13.144 | 492.029 ± 35.856 |
| tensornetwork_permutation_focus_step409_316 | 127.657 ± 5.529 | 169.401 ± 7.945 | **95.969 ± 7.917** | 128.407 ± 7.866 | 168.921 ± 35.372 |
| tensornetwork_permutation_light_415 | 121.765 ± 2.610 | 155.344 ± 2.963 | **102.850 ± 2.003** | 147.454 ± 23.102 | 170.688 ± 35.554 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.835 ± 0.095 | 0.667 ± 0.092 | **0.407 ± 0.004** | 0.619 ± 0.024 | 2.018 ± 0.724 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.174 ± 0.024 | 0.116 ± 0.015 | **0.102 ± 0.003** | 0.188 ± 0.033 | 0.596 ± 0.177 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.120 ± 0.019 | 0.117 ± 0.015 | **0.094 ± 0.007** | 0.141 ± 0.029 | 0.367 ± 0.097 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.169 ± 0.008 | 0.118 ± 0.004 | **0.095 ± 0.003** | 0.140 ± 0.027 | 0.755 ± 0.551 |
| bin_elementwise_mul_2048x2048 | 1.108 ± 0.033 | **1.092 ± 0.013** | **1.092 ± 0.012** | 1.525 ± 0.026 | 1.224 ± 0.113 |
| bin_matmul_1024 | 5.243 ± 0.042 | 4.646 ± 0.039 | **4.342 ± 0.010** | 7.396 ± 0.368 | 17.000 ± 6.127 |
| bin_matmul_256 | 0.136 ± 0.005 | 0.091 ± 0.002 | **0.086 ± 0.001** | 0.288 ± 0.039 | 0.390 ± 0.048 |
| bin_outer_product_4096 | **1.764 ± 0.005** | 1.782 ± 0.028 | 1.768 ± 0.008 | 1.846 ± 0.018 | 78.934 ± 16.739 |
| gm_queen5_5_3.wcsp | **231.025 ± 10.652** | 362.513 ± 13.787 | 261.188 ± 17.256 | 366.043 ± 17.497 | 717.034 ± 21.156 |
| lm_batch_likelihood_brackets_4_4d | 11.090 ± 0.083 | 16.198 ± 0.219 | 8.974 ± 0.104 | **7.677 ± 0.199** | 25.803 ± 30.696 |
| lm_batch_likelihood_sentence_3_12d | 21.072 ± 3.475 | 30.236 ± 4.913 | 22.545 ± 0.445 | **13.456 ± 1.335** | 63.991 ± 22.132 |
| lm_batch_likelihood_sentence_4_4d | 10.699 ± 0.476 | 14.441 ± 0.468 | 8.144 ± 0.216 | **6.868 ± 0.261** | 26.311 ± 29.366 |
| nary_matmul_chain_64 | 0.039 ± 0.003 | 0.039 ± 0.007 | **0.031 ± 0.001** | 0.086 ± 0.003 | 0.073 ± 0.002 |
| str_matrix_chain_multiplication_100 | 3.182 ± 0.025 | 4.050 ± 0.087 | **3.175 ± 0.027** | 8.210 ± 0.099 | 9.328 ± 0.765 |
| str_mps_varying_inner_product_200 | 8.407 ± 0.193 | 13.091 ± 0.187 | **7.368 ± 0.120** | 16.108 ± 0.129 | 15.594 ± 8.334 |
| str_nw_mera_closed_120 | 156.093 ± 3.704 | 147.247 ± 1.768 | **125.935 ± 3.976** | 375.531 ± 47.382 | 579.498 ± 32.584 |
| str_nw_mera_open_26 | 191.540 ± 6.343 | 182.439 ± 9.507 | **124.755 ± 3.930** | 279.209 ± 6.751 | 481.779 ± 41.434 |
| tensornetwork_permutation_focus_step409_316 | 127.657 ± 5.529 | 169.401 ± 7.945 | **95.969 ± 7.917** | 128.407 ± 7.866 | 168.921 ± 35.372 |
| tensornetwork_permutation_light_415 | 121.765 ± 2.610 | 155.344 ± 2.963 | **102.850 ± 2.003** | 147.454 ± 23.102 | 170.688 ± 35.554 |

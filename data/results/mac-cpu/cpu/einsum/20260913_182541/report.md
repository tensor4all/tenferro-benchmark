# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260913_182541/run.yaml`
- Timestamp: `20260913_182541`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260913_182541`.

- tenferro-rs commit: `28dfc7e383a653e1461d38b73ced3d563b4efdad`

## CPU Information

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

## Thread Environment

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

## Threads: 1

- Source table: `data/results/mac-cpu/cpu/einsum/20260913_182541/einsum_table_t1_20260913_182541.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260913_182541/tenferro_trace_t1_20260913_182541.log`
- `data/results/mac-cpu/cpu/einsum/20260913_182541/tenferro_eager_t1_20260913_182541.log`
- `data/results/mac-cpu/cpu/einsum/20260913_182541/pytorch_cpu_t1_20260913_182541.log`
- `data/results/mac-cpu/cpu/einsum/20260913_182541/jax_cpu_t1_20260913_182541.log`
- `data/results/mac-cpu/cpu/einsum/20260913_182541/julia_omeinsum_t1_20260913_182541.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.058 ± 0.155 | 0.775 ± 0.185 | **0.388 ± 0.007** | 2.236 ± 0.042 | 3.974 ± 4.818 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.196 ± 0.012 | 0.108 ± 0.002 | **0.090 ± 0.004** | 0.343 ± 0.015 | 0.598 ± 0.166 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.105 ± 0.005** | 0.114 ± 0.029 | 0.108 ± 0.006 | 0.177 ± 0.022 | 0.541 ± 0.355 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.439 ± 0.035 | **0.110 ± 0.019** | 0.112 ± 0.005 | 0.198 ± 0.010 | 0.657 ± 0.480 |
| bin_elementwise_mul_2048x2048 | 0.969 ± 0.063 | 2.173 ± 0.148 | **0.852 ± 0.056** | 1.483 ± 0.067 | 0.947 ± 0.471 |
| bin_matmul_1024 | 6.091 ± 0.319 | 5.244 ± 0.288 | **4.746 ± 0.601** | 35.163 ± 0.243 | 40.696 ± 4.765 |
| bin_matmul_256 | 0.135 ± 0.011 | 0.100 ± 0.001 | **0.087 ± 0.004** | 0.607 ± 0.016 | 0.752 ± 0.113 |
| bin_outer_product_4096 | **1.029 ± 0.052** | 1.050 ± 0.041 | 1.520 ± 0.069 | 1.142 ± 0.061 | 73.300 ± 41.124 |
| gm_queen5_5_3.wcsp | **894.607 ± 7.267** | 1101.627 ± 4.676 | 911.826 ± 5.959 | 1294.004 ± 8.836 | 1410.048 ± 98.563 |
| lm_batch_likelihood_brackets_4_4d | 8.165 ± 0.047 | 14.017 ± 0.292 | 8.703 ± 0.083 | **7.319 ± 0.071** | 13.058 ± 3.737 |
| lm_batch_likelihood_sentence_3_12d | **10.326 ± 0.119** | 25.960 ± 0.292 | 17.882 ± 0.193 | 29.233 ± 0.147 | 85.893 ± 44.515 |
| lm_batch_likelihood_sentence_4_4d | 8.544 ± 0.076 | 15.678 ± 0.122 | 8.942 ± 0.100 | **8.377 ± 0.066** | 19.186 ± 48.990 |
| nary_matmul_chain_64 | 0.014 ± 0.001 | 0.018 ± 0.001 | **0.013 ± 0.001** | 0.056 ± 0.002 | 0.043 ± 0.007 |
| str_matrix_chain_multiplication_100 | **2.159 ± 0.047** | 2.844 ± 0.060 | 2.219 ± 0.069 | 8.154 ± 0.084 | 9.524 ± 0.180 |
| str_mps_varying_inner_product_200 | 5.510 ± 0.104 | 7.136 ± 0.040 | **3.578 ± 0.053** | 12.184 ± 0.059 | 10.722 ± 6.155 |
| str_nw_mera_closed_120 | 175.138 ± 1.238 | 188.769 ± 0.518 | **160.414 ± 1.505** | 843.303 ± 2.553 | 978.963 ± 6.412 |
| str_nw_mera_open_26 | 182.014 ± 0.678 | 203.238 ± 0.677 | **128.900 ± 0.324** | 628.857 ± 2.976 | 723.511 ± 56.905 |
| tensornetwork_permutation_focus_step409_316 | **119.047 ± 0.608** | 209.933 ± 23.728 | 153.626 ± 1.340 | 214.490 ± 1.907 | 180.054 ± 22.399 |
| tensornetwork_permutation_light_415 | **120.314 ± 0.886** | 157.178 ± 0.848 | 133.868 ± 1.009 | 218.705 ± 2.753 | 179.517 ± 0.873 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.058 ± 0.155 | 0.775 ± 0.185 | **0.388 ± 0.007** | 2.236 ± 0.042 | 3.974 ± 4.818 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.196 ± 0.012 | 0.108 ± 0.002 | **0.090 ± 0.004** | 0.343 ± 0.015 | 0.598 ± 0.166 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.105 ± 0.005** | 0.114 ± 0.029 | 0.108 ± 0.006 | 0.177 ± 0.022 | 0.541 ± 0.355 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.439 ± 0.035 | **0.110 ± 0.019** | 0.112 ± 0.005 | 0.198 ± 0.010 | 0.657 ± 0.480 |
| bin_elementwise_mul_2048x2048 | 0.969 ± 0.063 | 2.173 ± 0.148 | **0.852 ± 0.056** | 1.483 ± 0.067 | 0.947 ± 0.471 |
| bin_matmul_1024 | 6.091 ± 0.319 | 5.244 ± 0.288 | **4.746 ± 0.601** | 35.163 ± 0.243 | 40.696 ± 4.765 |
| bin_matmul_256 | 0.135 ± 0.011 | 0.100 ± 0.001 | **0.087 ± 0.004** | 0.607 ± 0.016 | 0.752 ± 0.113 |
| bin_outer_product_4096 | **1.029 ± 0.052** | 1.050 ± 0.041 | 1.520 ± 0.069 | 1.142 ± 0.061 | 73.300 ± 41.124 |
| gm_queen5_5_3.wcsp | **235.303 ± 1.239** | 404.715 ± 1.851 | 325.317 ± 8.863 | 645.419 ± 9.080 | 777.293 ± 5.023 |
| lm_batch_likelihood_brackets_4_4d | 9.167 ± 0.172 | 15.120 ± 0.274 | 8.882 ± 0.084 | **7.073 ± 0.033** | 19.392 ± 49.057 |
| lm_batch_likelihood_sentence_3_12d | **12.120 ± 0.306** | 27.414 ± 0.201 | 17.043 ± 0.148 | 30.739 ± 0.281 | 85.957 ± 45.326 |
| lm_batch_likelihood_sentence_4_4d | 9.462 ± 0.042 | 16.580 ± 0.242 | 8.686 ± 0.101 | **7.827 ± 0.046** | 19.051 ± 50.334 |
| nary_matmul_chain_64 | 0.014 ± 0.001 | 0.018 ± 0.001 | **0.013 ± 0.001** | 0.056 ± 0.002 | 0.043 ± 0.007 |
| str_matrix_chain_multiplication_100 | **2.078 ± 0.065** | 2.836 ± 0.029 | 2.175 ± 0.054 | 8.561 ± 0.117 | 9.460 ± 1.344 |
| str_mps_varying_inner_product_200 | 6.540 ± 0.098 | 7.686 ± 0.097 | **4.474 ± 0.110** | 13.609 ± 0.123 | 9.555 ± 0.558 |
| str_nw_mera_closed_120 | 151.359 ± 0.285 | 160.090 ± 0.431 | **130.905 ± 0.504** | 825.694 ± 4.008 | 956.048 ± 4.723 |
| str_nw_mera_open_26 | 185.211 ± 0.835 | 200.653 ± 0.885 | **133.415 ± 0.561** | 641.076 ± 1.510 | 725.294 ± 54.894 |
| tensornetwork_permutation_focus_step409_316 | **119.047 ± 0.608** | 209.933 ± 23.728 | 153.626 ± 1.340 | 214.490 ± 1.907 | 180.054 ± 22.399 |
| tensornetwork_permutation_light_415 | **120.314 ± 0.886** | 157.178 ± 0.848 | 133.868 ± 1.009 | 218.705 ± 2.753 | 179.517 ± 0.873 |

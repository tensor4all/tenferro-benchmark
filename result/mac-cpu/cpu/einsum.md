# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260913_124035/run.yaml`
- Timestamp: `20260913_124035`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260913_124035`.

- tenferro-rs commit: `a48866b1a0bb52e6f9712c485925105b14ea30b9`

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

## Threads: 4

- Source table: `data/results/mac-cpu/cpu/einsum/20260913_124035/einsum_table_t4_20260913_124035.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260913_124035/tenferro_trace_t4_20260913_124035.log`
- `data/results/mac-cpu/cpu/einsum/20260913_124035/tenferro_eager_t4_20260913_124035.log`
- `data/results/mac-cpu/cpu/einsum/20260913_124035/pytorch_cpu_t4_20260913_124035.log`
- `data/results/mac-cpu/cpu/einsum/20260913_124035/jax_cpu_t4_20260913_124035.log`
- `data/results/mac-cpu/cpu/einsum/20260913_124035/julia_omeinsum_t4_20260913_124035.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.885 ± 0.120 | 0.500 ± 0.048 | 0.395 ± 0.025 | **0.366 ± 0.036** | 2.766 ± 3.239 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.216 ± 0.016 | 0.128 ± 0.016 | **0.094 ± 0.001** | 0.188 ± 0.039 | 1.202 ± 0.461 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.088 ± 0.015 | 0.069 ± 0.015 | **0.063 ± 0.007** | 0.121 ± 0.029 | 0.355 ± 0.175 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.178 ± 0.016 | 0.068 ± 0.017 | **0.060 ± 0.002** | 0.136 ± 0.013 | 0.718 ± 0.560 |
| bin_elementwise_mul_2048x2048 | **0.564 ± 0.012** | 1.259 ± 0.067 | 0.584 ± 0.030 | 0.843 ± 0.058 | 0.956 ± 0.056 |
| bin_matmul_1024 | 5.359 ± 0.617 | 6.645 ± 0.631 | 4.774 ± 0.284 | **3.358 ± 0.206** | 16.126 ± 4.958 |
| bin_matmul_256 | 0.166 ± 0.004 | 0.146 ± 0.022 | **0.093 ± 0.002** | 0.220 ± 0.039 | 0.407 ± 0.088 |
| bin_outer_product_4096 | **0.585 ± 0.005** | 0.601 ± 0.052 | 0.604 ± 0.018 | 0.671 ± 0.054 | 69.820 ± 8.923 |
| gm_queen5_5_3.wcsp | 512.693 ± 62.760 | 554.701 ± 69.657 | 465.017 ± 8.455 | **457.721 ± 5.926** | 1059.701 ± 27.162 |
| lm_batch_likelihood_brackets_4_4d | 7.509 ± 0.114 | 13.787 ± 0.322 | 7.903 ± 0.376 | **5.877 ± 0.166** | 9.221 ± 0.404 |
| lm_batch_likelihood_sentence_3_12d | 9.170 ± 0.172 | 16.849 ± 0.336 | 18.111 ± 0.408 | **6.751 ± 0.188** | 25.309 ± 17.905 |
| lm_batch_likelihood_sentence_4_4d | 8.002 ± 0.079 | 14.891 ± 0.239 | 7.966 ± 0.387 | **5.970 ± 0.329** | 13.898 ± 14.498 |
| nary_matmul_chain_64 | 0.050 ± 0.008 | 0.069 ± 0.004 | **0.025 ± 0.001** | 0.072 ± 0.014 | 0.048 ± 0.006 |
| str_matrix_chain_multiplication_100 | 2.996 ± 0.044 | 5.630 ± 0.121 | **2.712 ± 0.054** | 6.485 ± 0.077 | 5.713 ± 1.258 |
| str_mps_varying_inner_product_200 | 7.445 ± 0.045 | 14.430 ± 1.279 | **5.377 ± 0.104** | 14.063 ± 0.166 | 9.952 ± 4.004 |
| str_nw_mera_closed_120 | 141.307 ± 0.631 | 166.084 ± 2.775 | 144.116 ± 2.760 | **90.552 ± 2.047** | 319.004 ± 19.331 |
| str_nw_mera_open_26 | 141.807 ± 4.906 | 155.629 ± 9.318 | 122.859 ± 2.357 | **69.896 ± 1.851** | 272.281 ± 10.348 |
| tensornetwork_permutation_focus_step409_316 | 90.793 ± 0.337 | 127.407 ± 13.699 | **75.117 ± 13.793** | 78.816 ± 5.233 | 90.122 ± 16.780 |
| tensornetwork_permutation_light_415 | **63.823 ± 0.619** | 90.236 ± 3.152 | 76.944 ± 10.077 | 84.964 ± 5.481 | 89.913 ± 18.326 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.885 ± 0.120 | 0.500 ± 0.048 | 0.395 ± 0.025 | **0.366 ± 0.036** | 2.766 ± 3.239 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.216 ± 0.016 | 0.128 ± 0.016 | **0.094 ± 0.001** | 0.188 ± 0.039 | 1.202 ± 0.461 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.088 ± 0.015 | 0.069 ± 0.015 | **0.063 ± 0.007** | 0.121 ± 0.029 | 0.355 ± 0.175 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.178 ± 0.016 | 0.068 ± 0.017 | **0.060 ± 0.002** | 0.136 ± 0.013 | 0.718 ± 0.560 |
| bin_elementwise_mul_2048x2048 | **0.564 ± 0.012** | 1.259 ± 0.067 | 0.584 ± 0.030 | 0.843 ± 0.058 | 0.956 ± 0.056 |
| bin_matmul_1024 | 5.359 ± 0.617 | 6.645 ± 0.631 | 4.774 ± 0.284 | **3.358 ± 0.206** | 16.126 ± 4.958 |
| bin_matmul_256 | 0.166 ± 0.004 | 0.146 ± 0.022 | **0.093 ± 0.002** | 0.220 ± 0.039 | 0.407 ± 0.088 |
| bin_outer_product_4096 | **0.585 ± 0.005** | 0.601 ± 0.052 | 0.604 ± 0.018 | 0.671 ± 0.054 | 69.820 ± 8.923 |
| gm_queen5_5_3.wcsp | **136.030 ± 7.270** | 216.780 ± 15.299 | 189.713 ± 2.941 | 158.712 ± 1.653 | 394.652 ± 5.061 |
| lm_batch_likelihood_brackets_4_4d | 8.241 ± 0.192 | 15.952 ± 0.333 | 10.362 ± 0.848 | **6.337 ± 0.586** | 14.622 ± 15.516 |
| lm_batch_likelihood_sentence_3_12d | 10.146 ± 0.624 | 19.524 ± 0.327 | 19.823 ± 1.243 | **7.755 ± 0.144** | 34.451 ± 13.614 |
| lm_batch_likelihood_sentence_4_4d | 8.499 ± 0.224 | 15.687 ± 0.559 | 10.514 ± 0.987 | **6.524 ± 0.653** | 14.213 ± 15.840 |
| nary_matmul_chain_64 | 0.050 ± 0.008 | 0.069 ± 0.004 | **0.025 ± 0.001** | 0.072 ± 0.014 | 0.048 ± 0.006 |
| str_matrix_chain_multiplication_100 | 3.015 ± 0.074 | 5.678 ± 0.122 | **2.695 ± 0.075** | 6.864 ± 0.921 | 5.484 ± 0.269 |
| str_mps_varying_inner_product_200 | **8.036 ± 0.128** | 14.956 ± 0.148 | 9.629 ± 0.391 | 15.917 ± 0.294 | 8.647 ± 0.652 |
| str_nw_mera_closed_120 | 128.879 ± 1.940 | 143.225 ± 0.849 | 129.680 ± 4.753 | **93.007 ± 19.662** | 287.983 ± 20.404 |
| str_nw_mera_open_26 | 148.598 ± 6.499 | 152.592 ± 0.765 | 124.562 ± 2.108 | **80.922 ± 6.990** | 262.048 ± 17.192 |
| tensornetwork_permutation_focus_step409_316 | 90.793 ± 0.337 | 127.407 ± 13.699 | **75.117 ± 13.793** | 78.816 ± 5.233 | 90.122 ± 16.780 |
| tensornetwork_permutation_light_415 | **63.823 ± 0.619** | 90.236 ± 3.152 | 76.944 ± 10.077 | 84.964 ± 5.481 | 89.913 ± 18.326 |

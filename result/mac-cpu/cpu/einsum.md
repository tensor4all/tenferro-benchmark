# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260806_150841/run.yaml`
- Timestamp: `20260806_150841`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260806_150841`.

- tenferro-rs commit: `11a5b5a3c30b6258919557b4b69b429b6e686d75`

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

- Source table: `data/results/mac-cpu/cpu/einsum/20260806_150841/einsum_table_t4_20260806_150841.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260806_150841/tenferro_trace_t4_20260806_150841.log`
- `data/results/mac-cpu/cpu/einsum/20260806_150841/tenferro_eager_t4_20260806_150841.log`
- `data/results/mac-cpu/cpu/einsum/20260806_150841/pytorch_cpu_t4_20260806_150841.log`
- `data/results/mac-cpu/cpu/einsum/20260806_150841/jax_cpu_t4_20260806_150841.log`
- `data/results/mac-cpu/cpu/einsum/20260806_150841/julia_omeinsum_t4_20260806_150841.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.965 ± 0.157 | 0.852 ± 0.165 | **0.407 ± 0.007** | 0.642 ± 0.060 | 2.086 ± 5.044 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.184 ± 0.007 | 0.162 ± 0.012 | **0.099 ± 0.008** | 0.183 ± 0.014 | 0.670 ± 0.276 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.126 ± 0.023 | **0.104 ± 0.016** | 0.108 ± 0.005 | 0.140 ± 0.033 | 0.373 ± 0.121 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.173 ± 0.007 | 0.117 ± 0.012 | **0.109 ± 0.003** | 0.140 ± 0.014 | 0.726 ± 0.496 |
| bin_elementwise_mul_2048x2048 | 1.436 ± 0.053 | 3.321 ± 0.529 | **1.125 ± 0.036** | 1.536 ± 0.031 | 1.228 ± 0.051 |
| bin_matmul_1024 | 5.537 ± 0.233 | 5.331 ± 0.133 | **4.337 ± 0.050** | 7.561 ± 0.187 | 16.947 ± 6.411 |
| bin_matmul_256 | 0.150 ± 0.007 | 0.123 ± 0.015 | **0.086 ± 0.001** | 0.267 ± 0.036 | 0.365 ± 0.002 |
| bin_outer_product_4096 | 1.789 ± 0.015 | 1.796 ± 0.032 | **1.773 ± 0.093** | 1.860 ± 0.023 | 73.420 ± 16.124 |
| gm_queen5_5_3.wcsp | 740.794 ± 45.761 | 802.928 ± 16.523 | **723.218 ± 143.029** | 894.599 ± 52.566 | 2009.950 ± 579.999 |
| lm_batch_likelihood_brackets_4_4d | 9.485 ± 0.344 | 19.465 ± 2.524 | 10.790 ± 1.609 | **7.167 ± 0.105** | 14.514 ± 1.213 |
| lm_batch_likelihood_sentence_3_12d | 14.474 ± 0.416 | 28.015 ± 0.713 | 28.429 ± 0.811 | **11.807 ± 0.309** | 46.597 ± 27.075 |
| lm_batch_likelihood_sentence_4_4d | 9.405 ± 0.283 | 20.187 ± 0.451 | 8.418 ± 0.245 | **7.241 ± 0.155** | 25.064 ± 28.237 |
| nary_matmul_chain_64 | 0.047 ± 0.005 | 0.064 ± 0.010 | **0.031 ± 0.001** | 0.083 ± 0.003 | 0.074 ± 0.003 |
| str_matrix_chain_multiplication_100 | 3.397 ± 0.119 | 6.777 ± 0.871 | **3.147 ± 0.049** | 7.049 ± 0.088 | 8.717 ± 1.173 |
| str_mps_varying_inner_product_200 | 8.211 ± 0.056 | 19.314 ± 0.546 | **6.399 ± 0.133** | 14.982 ± 0.143 | 18.260 ± 7.601 |
| str_nw_mera_closed_120 | **165.546 ± 1.206** | 188.760 ± 9.602 | 184.495 ± 51.570 | 276.413 ± 128.286 | 584.683 ± 6.463 |
| str_nw_mera_open_26 | 169.338 ± 2.821 | 190.289 ± 3.802 | **138.352 ± 10.306** | 254.509 ± 14.581 | 467.042 ± 48.259 |
| tensornetwork_permutation_focus_step409_316 | **92.408 ± 3.964** | 139.578 ± 14.111 | 110.276 ± 4.564 | 133.848 ± 11.595 | 168.142 ± 34.548 |
| tensornetwork_permutation_light_415 | **96.726 ± 1.072** | 156.527 ± 2.748 | 105.377 ± 12.983 | 131.246 ± 9.590 | 180.774 ± 41.317 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.965 ± 0.157 | 0.852 ± 0.165 | **0.407 ± 0.007** | 0.642 ± 0.060 | 2.086 ± 5.044 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.184 ± 0.007 | 0.162 ± 0.012 | **0.099 ± 0.008** | 0.183 ± 0.014 | 0.670 ± 0.276 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.126 ± 0.023 | **0.104 ± 0.016** | 0.108 ± 0.005 | 0.140 ± 0.033 | 0.373 ± 0.121 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.173 ± 0.007 | 0.117 ± 0.012 | **0.109 ± 0.003** | 0.140 ± 0.014 | 0.726 ± 0.496 |
| bin_elementwise_mul_2048x2048 | 1.436 ± 0.053 | 3.321 ± 0.529 | **1.125 ± 0.036** | 1.536 ± 0.031 | 1.228 ± 0.051 |
| bin_matmul_1024 | 5.537 ± 0.233 | 5.331 ± 0.133 | **4.337 ± 0.050** | 7.561 ± 0.187 | 16.947 ± 6.411 |
| bin_matmul_256 | 0.150 ± 0.007 | 0.123 ± 0.015 | **0.086 ± 0.001** | 0.267 ± 0.036 | 0.365 ± 0.002 |
| bin_outer_product_4096 | 1.789 ± 0.015 | 1.796 ± 0.032 | **1.773 ± 0.093** | 1.860 ± 0.023 | 73.420 ± 16.124 |
| gm_queen5_5_3.wcsp | **165.717 ± 6.718** | 363.837 ± 16.482 | 288.482 ± 19.016 | 339.969 ± 36.952 | 721.267 ± 45.936 |
| lm_batch_likelihood_brackets_4_4d | 10.110 ± 0.166 | 20.656 ± 1.179 | 10.005 ± 0.204 | **7.307 ± 0.100** | 25.480 ± 28.986 |
| lm_batch_likelihood_sentence_3_12d | 16.057 ± 2.323 | 40.695 ± 4.970 | 26.058 ± 0.826 | **13.024 ± 0.530** | 60.070 ± 21.245 |
| lm_batch_likelihood_sentence_4_4d | 10.402 ± 0.343 | 19.575 ± 0.969 | 9.007 ± 0.145 | **6.645 ± 0.199** | 24.628 ± 30.981 |
| nary_matmul_chain_64 | 0.047 ± 0.005 | 0.064 ± 0.010 | **0.031 ± 0.001** | 0.083 ± 0.003 | 0.074 ± 0.003 |
| str_matrix_chain_multiplication_100 | 3.467 ± 0.100 | 6.796 ± 0.753 | **3.208 ± 0.017** | 7.396 ± 0.100 | 8.807 ± 0.772 |
| str_mps_varying_inner_product_200 | 8.588 ± 0.119 | 19.001 ± 0.639 | **7.039 ± 0.122** | 16.283 ± 0.084 | 15.343 ± 8.313 |
| str_nw_mera_closed_120 | 138.925 ± 2.744 | 161.830 ± 4.988 | **132.839 ± 0.509** | 339.294 ± 32.967 | 555.466 ± 22.275 |
| str_nw_mera_open_26 | 173.812 ± 8.910 | 223.442 ± 7.341 | **134.309 ± 2.867** | 265.684 ± 6.135 | 464.534 ± 34.942 |
| tensornetwork_permutation_focus_step409_316 | **92.408 ± 3.964** | 139.578 ± 14.111 | 110.276 ± 4.564 | 133.848 ± 11.595 | 168.142 ± 34.548 |
| tensornetwork_permutation_light_415 | **96.726 ± 1.072** | 156.527 ± 2.748 | 105.377 ± 12.983 | 131.246 ± 9.590 | 180.774 ± 41.317 |

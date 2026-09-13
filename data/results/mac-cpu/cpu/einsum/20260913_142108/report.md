# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260913_142108/run.yaml`
- Timestamp: `20260913_142108`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260913_142108`.

- tenferro-rs commit: `a793c2e95693f053722fbff8d0db25722336c21f`

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

- Source table: `data/results/mac-cpu/cpu/einsum/20260913_142108/einsum_table_t4_20260913_142108.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260913_142108/tenferro_trace_t4_20260913_142108.log`
- `data/results/mac-cpu/cpu/einsum/20260913_142108/tenferro_eager_t4_20260913_142108.log`
- `data/results/mac-cpu/cpu/einsum/20260913_142108/pytorch_cpu_t4_20260913_142108.log`
- `data/results/mac-cpu/cpu/einsum/20260913_142108/jax_cpu_t4_20260913_142108.log`
- `data/results/mac-cpu/cpu/einsum/20260913_142108/julia_omeinsum_t4_20260913_142108.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.912 ± 0.181 | 0.727 ± 0.200 | 0.381 ± 0.023 | **0.365 ± 0.033** | 2.488 ± 3.797 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.212 ± 0.020 | 0.168 ± 0.028 | **0.091 ± 0.004** | 0.161 ± 0.031 | 0.625 ± 0.154 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.081 ± 0.012 | 0.070 ± 0.020 | **0.053 ± 0.005** | 0.130 ± 0.026 | 0.337 ± 0.063 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.177 ± 0.011 | 0.075 ± 0.011 | **0.055 ± 0.007** | 0.149 ± 0.035 | 0.724 ± 0.449 |
| bin_elementwise_mul_2048x2048 | 0.614 ± 0.044 | 1.322 ± 0.101 | **0.594 ± 0.045** | 0.854 ± 0.085 | 1.040 ± 0.563 |
| bin_matmul_1024 | 5.447 ± 0.135 | 5.762 ± 0.690 | 5.200 ± 0.359 | **3.404 ± 0.069** | 16.388 ± 3.459 |
| bin_matmul_256 | 0.162 ± 0.006 | 0.144 ± 0.015 | **0.089 ± 0.010** | 0.202 ± 0.026 | 0.331 ± 0.064 |
| bin_outer_product_4096 | 0.608 ± 0.035 | 0.592 ± 0.023 | **0.591 ± 0.025** | 0.654 ± 0.040 | 72.577 ± 14.203 |
| gm_queen5_5_3.wcsp | 557.089 ± 30.678 | 549.771 ± 20.867 | 487.361 ± 10.806 | **472.430 ± 3.723** | 1066.673 ± 30.747 |
| lm_batch_likelihood_brackets_4_4d | 7.903 ± 0.153 | 13.932 ± 0.467 | 8.035 ± 0.179 | **5.553 ± 0.190** | 9.990 ± 1.383 |
| lm_batch_likelihood_sentence_3_12d | 9.705 ± 0.168 | 17.396 ± 0.425 | 19.069 ± 0.260 | **6.736 ± 0.153** | 24.311 ± 18.382 |
| lm_batch_likelihood_sentence_4_4d | 8.140 ± 0.209 | 14.939 ± 0.735 | 8.101 ± 0.643 | **5.519 ± 0.128** | 21.607 ± 12.276 |
| nary_matmul_chain_64 | 0.049 ± 0.009 | 0.064 ± 0.004 | **0.013 ± 0.000** | 0.060 ± 0.009 | 0.042 ± 0.003 |
| str_matrix_chain_multiplication_100 | 3.180 ± 0.127 | 5.297 ± 0.211 | **2.179 ± 0.167** | 6.221 ± 0.088 | 5.829 ± 0.354 |
| str_mps_varying_inner_product_200 | 7.555 ± 0.196 | 12.431 ± 0.171 | **3.746 ± 0.196** | 12.628 ± 0.205 | 10.290 ± 3.696 |
| str_nw_mera_closed_120 | 151.128 ± 1.981 | 161.914 ± 1.505 | 144.000 ± 1.340 | **90.390 ± 2.614** | 315.088 ± 2.556 |
| str_nw_mera_open_26 | 148.260 ± 5.243 | 154.064 ± 1.187 | 122.755 ± 1.595 | **69.502 ± 0.712** | 271.698 ± 4.512 |
| tensornetwork_permutation_focus_step409_316 | 94.444 ± 1.266 | 120.796 ± 18.423 | 79.998 ± 2.273 | **74.272 ± 3.541** | 101.135 ± 20.045 |
| tensornetwork_permutation_light_415 | 89.787 ± 19.258 | 96.520 ± 27.398 | 76.923 ± 0.270 | **76.323 ± 4.428** | 91.401 ± 17.182 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.912 ± 0.181 | 0.727 ± 0.200 | 0.381 ± 0.023 | **0.365 ± 0.033** | 2.488 ± 3.797 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.212 ± 0.020 | 0.168 ± 0.028 | **0.091 ± 0.004** | 0.161 ± 0.031 | 0.625 ± 0.154 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.081 ± 0.012 | 0.070 ± 0.020 | **0.053 ± 0.005** | 0.130 ± 0.026 | 0.337 ± 0.063 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.177 ± 0.011 | 0.075 ± 0.011 | **0.055 ± 0.007** | 0.149 ± 0.035 | 0.724 ± 0.449 |
| bin_elementwise_mul_2048x2048 | 0.614 ± 0.044 | 1.322 ± 0.101 | **0.594 ± 0.045** | 0.854 ± 0.085 | 1.040 ± 0.563 |
| bin_matmul_1024 | 5.447 ± 0.135 | 5.762 ± 0.690 | 5.200 ± 0.359 | **3.404 ± 0.069** | 16.388 ± 3.459 |
| bin_matmul_256 | 0.162 ± 0.006 | 0.144 ± 0.015 | **0.089 ± 0.010** | 0.202 ± 0.026 | 0.331 ± 0.064 |
| bin_outer_product_4096 | 0.608 ± 0.035 | 0.592 ± 0.023 | **0.591 ± 0.025** | 0.654 ± 0.040 | 72.577 ± 14.203 |
| gm_queen5_5_3.wcsp | **139.551 ± 4.741** | 216.093 ± 12.967 | 209.720 ± 6.304 | 161.836 ± 9.091 | 406.154 ± 5.977 |
| lm_batch_likelihood_brackets_4_4d | 8.407 ± 0.093 | 14.216 ± 0.388 | 10.198 ± 0.446 | **5.695 ± 0.189** | 15.012 ± 15.437 |
| lm_batch_likelihood_sentence_3_12d | 10.369 ± 0.198 | 18.395 ± 0.399 | 19.575 ± 0.339 | **7.437 ± 0.111** | 35.092 ± 16.891 |
| lm_batch_likelihood_sentence_4_4d | 8.673 ± 0.130 | 16.369 ± 0.754 | 10.135 ± 0.500 | **5.056 ± 0.159** | 13.877 ± 15.562 |
| nary_matmul_chain_64 | 0.049 ± 0.009 | 0.064 ± 0.004 | **0.013 ± 0.000** | 0.060 ± 0.009 | 0.042 ± 0.003 |
| str_matrix_chain_multiplication_100 | 3.182 ± 0.200 | 5.779 ± 0.807 | **2.204 ± 0.044** | 6.184 ± 0.482 | 5.779 ± 0.256 |
| str_mps_varying_inner_product_200 | **8.505 ± 0.201** | 13.551 ± 0.705 | 9.072 ± 0.502 | 13.675 ± 0.214 | 9.121 ± 5.060 |
| str_nw_mera_closed_120 | 133.568 ± 1.731 | 144.149 ± 1.964 | 129.622 ± 7.123 | **84.799 ± 1.427** | 297.164 ± 2.485 |
| str_nw_mera_open_26 | 151.924 ± 3.871 | 152.709 ± 0.800 | 124.311 ± 0.824 | **76.932 ± 1.012** | 267.621 ± 19.769 |
| tensornetwork_permutation_focus_step409_316 | 94.444 ± 1.266 | 120.796 ± 18.423 | 79.998 ± 2.273 | **74.272 ± 3.541** | 101.135 ± 20.045 |
| tensornetwork_permutation_light_415 | 89.787 ± 19.258 | 96.520 ± 27.398 | 76.923 ± 0.270 | **76.323 ± 4.428** | 91.401 ± 17.182 |

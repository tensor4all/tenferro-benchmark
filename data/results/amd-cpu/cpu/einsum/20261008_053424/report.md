# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261008_053424/run.yaml`
- Timestamp: `20261008_053424`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261008_053424`.

- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`

## CPU Information

- Model: `AMD Ryzen 9 9955HX 16-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `32`
- Sockets: `1`
- Cores per socket: `16`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=4`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Julia / OMEinsum.jl Backend

- Julia: version `1.13.1`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `4`

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

- Source table: `data/results/amd-cpu/cpu/einsum/20261008_053424/einsum_table_t4_20261008_053424.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20261008_053424/tenferro_trace_t4_20261008_053424.log`
- `data/results/amd-cpu/cpu/einsum/20261008_053424/tenferro_eager_t4_20261008_053424.log`
- `data/results/amd-cpu/cpu/einsum/20261008_053424/pytorch_cpu_t4_20261008_053424.log`
- `data/results/amd-cpu/cpu/einsum/20261008_053424/jax_cpu_t4_20261008_053424.log`
- `data/results/amd-cpu/cpu/einsum/20261008_053424/julia_omeinsum_t4_20261008_053424.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.336 ± 0.817 | 2.098 ± 0.012 | **1.156 ± 0.004** | 1.462 ± 0.060 | 5.514 ± 5.856 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.973 ± 0.117 | 0.558 ± 0.013 | **0.187 ± 0.003** | 0.691 ± 0.021 | 0.532 ± 0.317 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.172 ± 0.177 | 0.139 ± 0.073 | **0.057 ± 0.002** | 0.303 ± 0.072 | 0.500 ± 1.517 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.193 ± 0.010 | 0.088 ± 0.004 | **0.057 ± 0.002** | 0.280 ± 0.088 | 5.171 ± 5.319 |
| bin_elementwise_mul_2048x2048 | 3.237 ± 0.203 | **3.125 ± 0.606** | 3.146 ± 0.254 | 5.125 ± 0.339 | 9.641 ± 0.241 |
| bin_elementwise_mul_512x512 | 0.092 ± 0.031 | 0.046 ± 0.013 | **0.042 ± 0.003** | 0.204 ± 0.078 | 0.095 ± 0.307 |
| bin_matmul_1024 | 28.080 ± 0.669 | 17.526 ± 0.065 | 25.707 ± 0.127 | **16.242 ± 0.625** | 31.283 ± 8.470 |
| bin_matmul_256 | 0.646 ± 0.005 | **0.322 ± 0.008** | 0.712 ± 0.005 | 0.472 ± 0.050 | 0.531 ± 0.004 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.043 ± 0.016 | 0.016 ± 0.001 | 0.024 ± 0.001 | 0.133 ± 0.008 | **0.012 ± 0.001** |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.139 ± 0.101 | 0.063 ± 0.002 | 0.111 ± 0.001 | 0.180 ± 0.057 | **0.038 ± 0.001** |
| bin_omeinsum_matmul_10x10 | 0.056 ± 0.006 | **0.016 ± 0.001** | 0.023 ± 0.001 | 0.097 ± 0.015 | 0.020 ± 0.001 |
| bin_outer_product_4096 | 11.289 ± 5.566 | 10.359 ± 0.476 | **8.632 ± 0.148** | 10.395 ± 0.208 | 236.050 ± 3.601 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 19.830 ± 0.517 | 19.187 ± 0.337 | 20.217 ± 0.103 | **18.428 ± 1.013** | 22.307 ± 5.293 |
| gm_queen5_5_3.wcsp | 1883.194 ± 18.585 | 1871.317 ± 20.567 | 1686.025 ± 10.116 | **1494.468 ± 49.425** | 5507.456 ± 45.829 |
| lm_batch_likelihood_brackets_4_4d | 14.103 ± 1.650 | 21.766 ± 1.049 | **8.017 ± 0.276** | 14.823 ± 0.767 | 17.815 ± 4.619 |
| lm_batch_likelihood_sentence_3_12d | 32.334 ± 6.084 | 49.351 ± 4.432 | **19.719 ± 0.265** | 31.113 ± 3.536 | 56.072 ± 6.914 |
| lm_batch_likelihood_sentence_4_4d | 15.228 ± 2.173 | 22.535 ± 2.141 | **8.701 ± 0.070** | 15.979 ± 2.602 | 33.554 ± 16.101 |
| nary_matmul_chain_64 | 0.066 ± 0.024 | **0.032 ± 0.002** | 0.047 ± 0.001 | 0.114 ± 0.023 | 0.071 ± 0.027 |
| str_matrix_chain_multiplication_100 | 5.220 ± 0.317 | **4.974 ± 0.873** | 5.760 ± 0.044 | 5.737 ± 0.213 | 8.881 ± 0.743 |
| str_mps_varying_inner_product_200 | 15.989 ± 4.644 | 15.559 ± 1.342 | **14.009 ± 0.091** | 17.653 ± 0.664 | 21.768 ± 2.104 |
| str_nw_mera_closed_120 | 571.352 ± 10.271 | 521.686 ± 14.381 | 559.056 ± 1.791 | **501.404 ± 4.278** | 630.384 ± 11.938 |
| str_nw_mera_open_26 | 598.579 ± 4.306 | 453.522 ± 7.202 | **355.086 ± 1.786** | 398.911 ± 7.203 | 696.408 ± 37.167 |
| tensornetwork_permutation_focus_step409_316 | **142.288 ± 5.471** | 184.822 ± 9.267 | 234.708 ± 0.795 | 205.838 ± 4.333 | 330.805 ± 28.403 |
| tensornetwork_permutation_light_415 | **139.924 ± 6.792** | 184.155 ± 6.678 | 228.366 ± 0.985 | 218.839 ± 3.603 | 337.892 ± 8.323 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.336 ± 0.817 | 2.098 ± 0.012 | **1.156 ± 0.004** | 1.462 ± 0.060 | 5.514 ± 5.856 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.973 ± 0.117 | 0.558 ± 0.013 | **0.187 ± 0.003** | 0.691 ± 0.021 | 0.532 ± 0.317 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.172 ± 0.177 | 0.139 ± 0.073 | **0.057 ± 0.002** | 0.303 ± 0.072 | 0.500 ± 1.517 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.193 ± 0.010 | 0.088 ± 0.004 | **0.057 ± 0.002** | 0.280 ± 0.088 | 5.171 ± 5.319 |
| bin_elementwise_mul_2048x2048 | 3.237 ± 0.203 | **3.125 ± 0.606** | 3.146 ± 0.254 | 5.125 ± 0.339 | 9.641 ± 0.241 |
| bin_elementwise_mul_512x512 | 0.092 ± 0.031 | 0.046 ± 0.013 | **0.042 ± 0.003** | 0.204 ± 0.078 | 0.095 ± 0.307 |
| bin_matmul_1024 | 28.080 ± 0.669 | 17.526 ± 0.065 | 25.707 ± 0.127 | **16.242 ± 0.625** | 31.283 ± 8.470 |
| bin_matmul_256 | 0.646 ± 0.005 | **0.322 ± 0.008** | 0.712 ± 0.005 | 0.472 ± 0.050 | 0.531 ± 0.004 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.043 ± 0.016 | 0.016 ± 0.001 | 0.024 ± 0.001 | 0.133 ± 0.008 | **0.012 ± 0.001** |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.139 ± 0.101 | 0.063 ± 0.002 | 0.111 ± 0.001 | 0.180 ± 0.057 | **0.038 ± 0.001** |
| bin_omeinsum_matmul_10x10 | 0.056 ± 0.006 | **0.016 ± 0.001** | 0.023 ± 0.001 | 0.097 ± 0.015 | 0.020 ± 0.001 |
| bin_outer_product_4096 | 11.289 ± 5.566 | 10.359 ± 0.476 | **8.632 ± 0.148** | 10.395 ± 0.208 | 236.050 ± 3.601 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 19.830 ± 0.517 | 19.187 ± 0.337 | 20.217 ± 0.103 | **18.428 ± 1.013** | 22.307 ± 5.293 |
| gm_queen5_5_3.wcsp | 701.870 ± 8.029 | 820.139 ± 9.067 | 777.598 ± 6.752 | **641.971 ± 11.748** | 1952.444 ± 17.190 |
| lm_batch_likelihood_brackets_4_4d | 15.318 ± 2.300 | 21.135 ± 2.440 | **8.843 ± 0.177** | 15.073 ± 1.663 | 36.189 ± 10.632 |
| lm_batch_likelihood_sentence_3_12d | 36.430 ± 5.499 | 44.406 ± 2.326 | **20.311 ± 0.080** | 32.371 ± 2.677 | 59.792 ± 7.669 |
| lm_batch_likelihood_sentence_4_4d | 15.968 ± 2.777 | 22.272 ± 3.487 | **9.557 ± 0.106** | 14.571 ± 0.891 | 25.270 ± 11.051 |
| nary_matmul_chain_64 | 0.066 ± 0.024 | **0.032 ± 0.002** | 0.047 ± 0.001 | 0.114 ± 0.023 | 0.071 ± 0.027 |
| str_matrix_chain_multiplication_100 | 5.373 ± 0.455 | **5.080 ± 0.043** | 5.739 ± 0.055 | 6.515 ± 0.396 | 8.863 ± 0.559 |
| str_mps_varying_inner_product_200 | 16.067 ± 2.413 | 16.288 ± 2.914 | **13.993 ± 0.140** | 16.297 ± 0.885 | 18.255 ± 7.422 |
| str_nw_mera_closed_120 | 461.765 ± 5.849 | 445.002 ± 8.336 | 457.249 ± 2.970 | **430.970 ± 14.292** | 503.279 ± 16.666 |
| str_nw_mera_open_26 | 583.995 ± 3.814 | 446.992 ± 6.155 | **355.250 ± 1.833** | 413.274 ± 14.026 | 666.321 ± 42.274 |
| tensornetwork_permutation_focus_step409_316 | **142.288 ± 5.471** | 184.822 ± 9.267 | 234.708 ± 0.795 | 205.838 ± 4.333 | 330.805 ± 28.403 |
| tensornetwork_permutation_light_415 | **139.924 ± 6.792** | 184.155 ± 6.678 | 228.366 ± 0.985 | 218.839 ± 3.603 | 337.892 ± 8.323 |

# CPU Einsum Benchmark Results

- Target profile: `zen5-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `2ee56deba06eaad06c590db40867fcef00165e86`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/zen5-cpu/cpu/einsum/20261113_020537/report.md`.


- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_020537/run.yaml`
- Timestamp: `2026-11-12T17:05:37.788001+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261113_020537`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `disabled`
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

- tenferro-rs features: `native`
- BLAS implementation: `none`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.1`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `1`

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

- `data/results/zen5-cpu/cpu/einsum/20261113_020537/jax_cpu_t1_20261113_020537.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_020537/julia_omeinsum_t1_20261113_020537.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_020537/pytorch_cpu_t1_20261113_020537.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_020537/tenferro_eager_t1_20261113_020537.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_020537/tenferro_trace_t1_20261113_020537.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.229 ± 0.013 | **4.351 ± 0.009** | 5.982 ± 0.030 | 5.338 ± 0.035 | 13.307 ± 6.928 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.770 ± 0.004 | **0.553 ± 0.008** | 0.877 ± 0.002 | 0.895 ± 0.021 | 0.967 ± 1.216 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.990 ± 0.014 | 0.979 ± 0.008 | **0.185 ± 0.002** | 0.516 ± 0.037 | 2.572 ± 1.121 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.654 ± 0.010 | 0.981 ± 0.006 | **0.182 ± 0.004** | 0.522 ± 0.025 | 6.964 ± 7.608 |
| bin_elementwise_mul_2048x2048 | 1.926 ± 0.036 | **1.924 ± 0.029** | 11.422 ± 0.416 | 15.592 ± 0.144 | 12.903 ± 0.939 |
| bin_elementwise_mul_512x512 | 0.135 ± 0.030 | 0.124 ± 0.006 | 0.151 ± 0.001 | 0.339 ± 0.022 | **0.120 ± 0.561** |
| bin_matmul_1024 | 75.076 ± 0.283 | **68.761 ± 0.111** | 89.362 ± 0.155 | 77.286 ± 0.182 | 110.540 ± 24.786 |
| bin_matmul_256 | 1.216 ± 0.035 | **1.073 ± 0.003** | 1.838 ± 0.051 | 1.389 ± 0.015 | 1.771 ± 0.014 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.004 | **0.007 ± 0.000** | 0.029 ± 0.001 | 0.172 ± 0.013 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.036 ± 0.001 | **0.033 ± 0.001** | 0.145 ± 0.004 | 0.228 ± 0.025 | 0.051 ± 0.001 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.001 | **0.011 ± 0.000** | 0.029 ± 0.001 | 0.117 ± 0.016 | 0.027 ± 0.001 |
| bin_outer_product_4096 | 49.518 ± 0.813 | 49.535 ± 1.057 | **39.480 ± 0.578** | 40.985 ± 7.934 | 315.983 ± 2.170 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 73.671 ± 0.236 | **71.482 ± 0.322** | 85.061 ± 0.102 | 81.833 ± 0.301 | 101.781 ± 14.438 |
| gm_queen5_5_3.wcsp | 5454.580 ± 87.034 | 5378.205 ± 29.854 | 6498.289 ± 19.755 | **4906.584 ± 10.641** | 7871.360 ± 63.234 |
| lm_batch_likelihood_brackets_4_4d | **30.344 ± 0.104** | 35.429 ± 0.187 | 37.232 ± 0.294 | 30.817 ± 0.177 | 37.084 ± 15.839 |
| lm_batch_likelihood_sentence_3_12d | **72.000 ± 0.230** | 115.791 ± 1.578 | 87.096 ± 0.235 | 107.335 ± 0.550 | 143.858 ± 38.769 |
| lm_batch_likelihood_sentence_4_4d | **32.253 ± 0.235** | 37.312 ± 0.279 | 41.586 ± 0.231 | 32.580 ± 0.137 | 74.895 ± 31.939 |
| nary_matmul_chain_64 | 0.053 ± 0.000 | **0.043 ± 0.000** | 0.080 ± 0.000 | 0.192 ± 0.012 | 0.107 ± 0.035 |
| str_matrix_chain_multiplication_100 | **13.065 ± 0.054** | 13.145 ± 0.092 | 19.549 ± 0.024 | 23.229 ± 0.294 | 21.353 ± 0.790 |
| str_mps_varying_inner_product_200 | **17.811 ± 0.061** | 18.158 ± 0.227 | 27.434 ± 0.083 | 39.949 ± 0.194 | 27.641 ± 8.526 |
| str_nw_mera_closed_120 | **1793.395 ± 4.505** | 1842.398 ± 4.307 | 2314.945 ± 3.806 | 2541.785 ± 6.283 | 2579.359 ± 17.562 |
| str_nw_mera_open_26 | **1313.768 ± 2.120** | 1577.358 ± 1.758 | 1466.816 ± 2.351 | 1975.463 ± 2.525 | 2107.425 ± 41.968 |
| tensornetwork_permutation_focus_step409_316 | **313.199 ± 1.922** | 444.561 ± 1.585 | 1024.017 ± 1.522 | 756.701 ± 1.974 | 678.009 ± 31.741 |
| tensornetwork_permutation_light_415 | **312.373 ± 0.854** | 343.792 ± 1.652 | 980.970 ± 0.745 | 770.725 ± 3.352 | 684.016 ± 16.280 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.229 ± 0.013 | **4.351 ± 0.009** | 5.982 ± 0.030 | 5.338 ± 0.035 | 13.307 ± 6.928 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.770 ± 0.004 | **0.553 ± 0.008** | 0.877 ± 0.002 | 0.895 ± 0.021 | 0.967 ± 1.216 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.990 ± 0.014 | 0.979 ± 0.008 | **0.185 ± 0.002** | 0.516 ± 0.037 | 2.572 ± 1.121 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.654 ± 0.010 | 0.981 ± 0.006 | **0.182 ± 0.004** | 0.522 ± 0.025 | 6.964 ± 7.608 |
| bin_elementwise_mul_2048x2048 | 1.926 ± 0.036 | **1.924 ± 0.029** | 11.422 ± 0.416 | 15.592 ± 0.144 | 12.903 ± 0.939 |
| bin_elementwise_mul_512x512 | 0.135 ± 0.030 | 0.124 ± 0.006 | 0.151 ± 0.001 | 0.339 ± 0.022 | **0.120 ± 0.561** |
| bin_matmul_1024 | 75.076 ± 0.283 | **68.761 ± 0.111** | 89.362 ± 0.155 | 77.286 ± 0.182 | 110.540 ± 24.786 |
| bin_matmul_256 | 1.216 ± 0.035 | **1.073 ± 0.003** | 1.838 ± 0.051 | 1.389 ± 0.015 | 1.771 ± 0.014 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.004 | **0.007 ± 0.000** | 0.029 ± 0.001 | 0.172 ± 0.013 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.036 ± 0.001 | **0.033 ± 0.001** | 0.145 ± 0.004 | 0.228 ± 0.025 | 0.051 ± 0.001 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.001 | **0.011 ± 0.000** | 0.029 ± 0.001 | 0.117 ± 0.016 | 0.027 ± 0.001 |
| bin_outer_product_4096 | 49.518 ± 0.813 | 49.535 ± 1.057 | **39.480 ± 0.578** | 40.985 ± 7.934 | 315.983 ± 2.170 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 73.671 ± 0.236 | **71.482 ± 0.322** | 85.061 ± 0.102 | 81.833 ± 0.301 | 101.781 ± 14.438 |
| gm_queen5_5_3.wcsp | **1869.616 ± 10.094** | 2314.439 ± 4.806 | 2805.429 ± 12.682 | 2110.599 ± 4.758 | 3352.583 ± 45.833 |
| lm_batch_likelihood_brackets_4_4d | **30.304 ± 0.103** | 33.180 ± 0.392 | 40.804 ± 0.461 | 30.654 ± 0.281 | 46.216 ± 33.822 |
| lm_batch_likelihood_sentence_3_12d | **81.446 ± 0.381** | 117.039 ± 1.468 | 86.463 ± 0.388 | 131.192 ± 0.738 | 155.634 ± 39.550 |
| lm_batch_likelihood_sentence_4_4d | **29.334 ± 0.066** | 35.831 ± 0.282 | 43.263 ± 0.139 | 30.971 ± 0.147 | 47.703 ± 33.460 |
| nary_matmul_chain_64 | 0.053 ± 0.000 | **0.043 ± 0.000** | 0.080 ± 0.000 | 0.192 ± 0.012 | 0.107 ± 0.035 |
| str_matrix_chain_multiplication_100 | 13.199 ± 0.067 | **13.145 ± 0.051** | 19.456 ± 0.123 | 22.979 ± 0.271 | 21.176 ± 0.355 |
| str_mps_varying_inner_product_200 | 21.890 ± 0.110 | **19.079 ± 0.168** | 25.729 ± 0.108 | 38.447 ± 0.336 | 24.139 ± 12.407 |
| str_nw_mera_closed_120 | 1756.935 ± 3.204 | **1718.794 ± 3.389** | 1962.542 ± 1.554 | 2193.382 ± 1.885 | 2330.125 ± 76.026 |
| str_nw_mera_open_26 | **1329.213 ± 2.072** | 1589.296 ± 2.025 | 1482.108 ± 2.209 | 2014.623 ± 4.061 | 2110.799 ± 30.527 |
| tensornetwork_permutation_focus_step409_316 | **313.199 ± 1.922** | 444.561 ± 1.585 | 1024.017 ± 1.522 | 756.701 ± 1.974 | 678.009 ± 31.741 |
| tensornetwork_permutation_light_415 | **312.373 ± 0.854** | 343.792 ± 1.652 | 980.970 ± 0.745 | 770.725 ± 3.352 | 684.016 ± 16.280 |

## Threads: 4

Source report: `data/results/zen5-cpu/cpu/einsum/20261113_031144/report.md`.


- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_031144/run.yaml`
- Timestamp: `2026-11-12T18:11:44.458524+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261113_031144`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `disabled`
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

- tenferro-rs features: `native`
- BLAS implementation: `none`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

### Julia / OMEinsum.jl Backend

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


Logs:

- `data/results/zen5-cpu/cpu/einsum/20261113_031144/jax_cpu_t4_20261113_031144.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_031144/julia_omeinsum_t4_20261113_031144.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_031144/pytorch_cpu_t4_20261113_031144.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_031144/tenferro_eager_t4_20261113_031144.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_031144/tenferro_trace_t4_20261113_031144.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.783 ± 0.005 | **1.469 ± 0.008** | 1.705 ± 4.297 | 2.378 ± 0.496 | 10.056 ± 6.054 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.478 ± 0.009 | 0.363 ± 0.007 | **0.273 ± 0.006** | 1.038 ± 0.032 | 1.097 ± 1.141 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.245 ± 0.004 | 0.243 ± 0.003 | **0.069 ± 0.001** | 0.396 ± 0.186 | 0.924 ± 1.997 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.203 ± 0.004 | 0.240 ± 0.004 | **0.070 ± 0.002** | 0.485 ± 0.151 | 6.738 ± 2.837 |
| bin_elementwise_mul_2048x2048 | 0.806 ± 0.075 | **0.785 ± 0.011** | 3.902 ± 0.283 | 8.020 ± 0.079 | 13.216 ± 0.324 |
| bin_elementwise_mul_512x512 | 0.042 ± 0.001 | **0.038 ± 0.001** | 0.052 ± 0.002 | 0.412 ± 0.177 | 0.119 ± 0.462 |
| bin_matmul_1024 | 19.375 ± 0.066 | **17.324 ± 0.030** | 35.340 ± 0.443 | 20.230 ± 0.159 | 51.507 ± 17.816 |
| bin_matmul_256 | 0.372 ± 0.007 | **0.296 ± 0.003** | 0.921 ± 0.004 | 0.645 ± 0.172 | 0.833 ± 0.017 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.007 ± 0.000** | 0.030 ± 0.002 | 0.167 ± 0.015 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.031 ± 0.001 | **0.028 ± 0.000** | 0.143 ± 0.002 | 0.225 ± 0.049 | 0.049 ± 0.001 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.001 | **0.012 ± 0.001** | 0.029 ± 0.002 | 0.099 ± 0.011 | 0.027 ± 0.001 |
| bin_outer_product_4096 | 13.648 ± 0.141 | 13.885 ± 0.358 | **11.497 ± 0.231** | 21.768 ± 0.764 | 296.433 ± 2.438 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 19.910 ± 0.056 | **18.501 ± 0.035** | 26.404 ± 0.283 | 23.847 ± 0.590 | 44.905 ± 16.396 |
| gm_queen5_5_3.wcsp | 2747.871 ± 6.929 | 2710.089 ± 10.592 | **2194.640 ± 14.282** | 2984.680 ± 55.678 | 7147.330 ± 96.589 |
| lm_batch_likelihood_brackets_4_4d | 14.050 ± 0.136 | 14.422 ± 0.111 | **11.000 ± 0.074** | 21.071 ± 0.511 | 27.837 ± 8.390 |
| lm_batch_likelihood_sentence_3_12d | **22.677 ± 0.024** | 38.656 ± 0.552 | 42.116 ± 0.862 | 36.661 ± 6.745 | 101.528 ± 42.595 |
| lm_batch_likelihood_sentence_4_4d | 14.193 ± 0.051 | 14.375 ± 0.200 | **11.824 ± 0.130** | 21.564 ± 0.727 | 43.170 ± 31.092 |
| nary_matmul_chain_64 | 0.042 ± 0.001 | **0.032 ± 0.001** | 0.062 ± 0.001 | 0.159 ± 0.029 | 0.109 ± 0.019 |
| str_matrix_chain_multiplication_100 | **6.114 ± 0.067** | 6.332 ± 0.190 | 7.525 ± 0.027 | 10.905 ± 0.433 | 12.647 ± 0.640 |
| str_mps_varying_inner_product_200 | 10.931 ± 0.088 | **10.137 ± 0.290** | 18.135 ± 0.107 | 22.309 ± 0.293 | 29.156 ± 19.849 |
| str_nw_mera_closed_120 | **541.084 ± 1.434** | 552.667 ± 2.014 | 694.159 ± 3.565 | 737.012 ± 5.057 | 1016.287 ± 14.951 |
| str_nw_mera_open_26 | 465.598 ± 1.393 | 508.464 ± 2.490 | **435.378 ± 1.260** | 599.459 ± 5.829 | 1024.447 ± 49.401 |
| tensornetwork_permutation_focus_step409_316 | **147.729 ± 0.958** | 160.473 ± 3.353 | 297.606 ± 0.711 | 369.550 ± 5.136 | 480.513 ± 17.920 |
| tensornetwork_permutation_light_415 | 153.372 ± 1.396 | **114.628 ± 0.647** | 289.275 ± 4.637 | 381.054 ± 6.776 | 481.028 ± 71.652 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.783 ± 0.005 | **1.469 ± 0.008** | 1.705 ± 4.297 | 2.378 ± 0.496 | 10.056 ± 6.054 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.478 ± 0.009 | 0.363 ± 0.007 | **0.273 ± 0.006** | 1.038 ± 0.032 | 1.097 ± 1.141 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.245 ± 0.004 | 0.243 ± 0.003 | **0.069 ± 0.001** | 0.396 ± 0.186 | 0.924 ± 1.997 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.203 ± 0.004 | 0.240 ± 0.004 | **0.070 ± 0.002** | 0.485 ± 0.151 | 6.738 ± 2.837 |
| bin_elementwise_mul_2048x2048 | 0.806 ± 0.075 | **0.785 ± 0.011** | 3.902 ± 0.283 | 8.020 ± 0.079 | 13.216 ± 0.324 |
| bin_elementwise_mul_512x512 | 0.042 ± 0.001 | **0.038 ± 0.001** | 0.052 ± 0.002 | 0.412 ± 0.177 | 0.119 ± 0.462 |
| bin_matmul_1024 | 19.375 ± 0.066 | **17.324 ± 0.030** | 35.340 ± 0.443 | 20.230 ± 0.159 | 51.507 ± 17.816 |
| bin_matmul_256 | 0.372 ± 0.007 | **0.296 ± 0.003** | 0.921 ± 0.004 | 0.645 ± 0.172 | 0.833 ± 0.017 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.007 ± 0.000** | 0.030 ± 0.002 | 0.167 ± 0.015 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.031 ± 0.001 | **0.028 ± 0.000** | 0.143 ± 0.002 | 0.225 ± 0.049 | 0.049 ± 0.001 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.001 | **0.012 ± 0.001** | 0.029 ± 0.002 | 0.099 ± 0.011 | 0.027 ± 0.001 |
| bin_outer_product_4096 | 13.648 ± 0.141 | 13.885 ± 0.358 | **11.497 ± 0.231** | 21.768 ± 0.764 | 296.433 ± 2.438 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 19.910 ± 0.056 | **18.501 ± 0.035** | 26.404 ± 0.283 | 23.847 ± 0.590 | 44.905 ± 16.396 |
| gm_queen5_5_3.wcsp | **712.250 ± 4.538** | 765.541 ± 0.960 | 965.135 ± 10.877 | 1161.872 ± 12.288 | 2620.191 ± 29.083 |
| lm_batch_likelihood_brackets_4_4d | 14.034 ± 0.073 | 13.546 ± 0.126 | **11.586 ± 0.183** | 21.145 ± 0.352 | 45.943 ± 34.425 |
| lm_batch_likelihood_sentence_3_12d | **25.244 ± 0.098** | 36.825 ± 0.401 | 43.463 ± 0.540 | 41.320 ± 0.764 | 98.894 ± 35.310 |
| lm_batch_likelihood_sentence_4_4d | **12.739 ± 0.038** | 14.490 ± 0.089 | 12.786 ± 0.122 | 21.023 ± 0.510 | 42.587 ± 28.604 |
| nary_matmul_chain_64 | 0.042 ± 0.001 | **0.032 ± 0.001** | 0.062 ± 0.001 | 0.159 ± 0.029 | 0.109 ± 0.019 |
| str_matrix_chain_multiplication_100 | 6.492 ± 0.050 | **6.309 ± 0.049** | 7.490 ± 0.069 | 10.890 ± 0.340 | 12.624 ± 1.144 |
| str_mps_varying_inner_product_200 | 13.415 ± 0.059 | **10.934 ± 0.115** | 17.849 ± 0.048 | 21.680 ± 0.265 | 24.750 ± 5.982 |
| str_nw_mera_closed_120 | 565.553 ± 1.769 | **527.749 ± 4.927** | 565.849 ± 1.549 | 593.195 ± 10.107 | 877.502 ± 35.219 |
| str_nw_mera_open_26 | 469.735 ± 0.770 | 508.566 ± 1.624 | **442.513 ± 1.748** | 619.900 ± 10.620 | 1044.841 ± 49.863 |
| tensornetwork_permutation_focus_step409_316 | **147.729 ± 0.958** | 160.473 ± 3.353 | 297.606 ± 0.711 | 369.550 ± 5.136 | 480.513 ± 17.920 |
| tensornetwork_permutation_light_415 | 153.372 ± 1.396 | **114.628 ± 0.647** | 289.275 ± 4.637 | 381.054 ± 6.776 | 481.028 ± 71.652 |

## Threads: 8

Source report: `data/results/zen5-cpu/cpu/einsum/20261113_035559/report.md`.


- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_035559/run.yaml`
- Timestamp: `2026-11-12T18:55:59.554485+00:00`

Latest run: `./scripts/run_all.sh 8`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261113_035559`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `disabled`
- Container: `none`

### Thread Environment

- OMP_NUM_THREADS: `8`
- OMP_THREAD_LIMIT: `8`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `8`
- OPENBLAS_NUM_THREADS: `8`
- GOTO_NUM_THREADS: `8`
- MKL_NUM_THREADS: `8`
- VECLIB_MAXIMUM_THREADS: `8`
- VECLIB_NUM_THREADS: `8`
- NUMEXPR_NUM_THREADS: `8`
- BLIS_NUM_THREADS: `8`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=8`

### Tenferro CPU BLAS Backend

- tenferro-rs features: `native`
- BLAS implementation: `none`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.1`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `8`

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

- `data/results/zen5-cpu/cpu/einsum/20261113_035559/jax_cpu_t8_20261113_035559.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_035559/julia_omeinsum_t8_20261113_035559.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_035559/pytorch_cpu_t8_20261113_035559.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_035559/tenferro_eager_t8_20261113_035559.log`
- `data/results/zen5-cpu/cpu/einsum/20261113_035559/tenferro_trace_t8_20261113_035559.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=8, OMP_NUM_THREADS=8, RAYON_NUM_THREADS=8.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.503 ± 0.027 | **1.065 ± 0.010** | 5.997 ± 0.806 | 1.470 ± 0.034 | 11.321 ± 9.333 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.469 ± 0.009 | **0.318 ± 0.004** | 5.998 ± 0.012 | 1.053 ± 0.007 | 1.110 ± 1.150 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.131 ± 0.001 | **0.129 ± 0.025** | 5.999 ± 0.017 | 0.562 ± 0.245 | 1.120 ± 0.351 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.110 ± 0.001** | 0.157 ± 0.001 | 5.998 ± 0.003 | 0.573 ± 0.184 | 5.738 ± 5.004 |
| bin_elementwise_mul_2048x2048 | 1.139 ± 0.161 | **1.000 ± 0.241** | 5.112 ± 0.869 | 5.984 ± 0.416 | 13.803 ± 0.355 |
| bin_elementwise_mul_512x512 | 0.027 ± 0.001 | **0.020 ± 0.001** | 5.999 ± 0.002 | 0.405 ± 0.126 | 0.138 ± 0.591 |
| bin_matmul_1024 | 10.611 ± 0.042 | **8.932 ± 0.031** | 19.068 ± 0.231 | 13.780 ± 1.665 | 41.607 ± 9.865 |
| bin_matmul_256 | 0.236 ± 0.004 | **0.169 ± 0.001** | 0.459 ± 0.004 | 0.453 ± 0.116 | 0.671 ± 0.012 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.008 ± 0.000** | 0.030 ± 0.001 | 0.169 ± 0.009 | 0.015 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.001 | **0.028 ± 0.001** | 0.145 ± 0.001 | 0.232 ± 0.055 | 0.053 ± 0.004 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.000 | **0.013 ± 0.000** | 0.028 ± 0.001 | 0.099 ± 0.010 | 0.028 ± 0.003 |
| bin_outer_product_4096 | 8.945 ± 0.573 | 8.403 ± 0.303 | **7.533 ± 0.261** | 16.672 ± 0.208 | 296.975 ± 2.278 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 11.647 ± 0.074 | **10.479 ± 0.016** | 14.781 ± 0.179 | 13.560 ± 0.381 | 34.008 ± 10.037 |
| gm_queen5_5_3.wcsp | 2304.747 ± 8.647 | 1722.025 ± 6.888 | **1498.003 ± 10.183** | 2011.325 ± 34.284 | 7195.208 ± 118.102 |
| lm_batch_likelihood_brackets_4_4d | 11.145 ± 0.169 | 12.173 ± 0.215 | **8.754 ± 0.220** | 19.615 ± 0.364 | 48.479 ± 37.945 |
| lm_batch_likelihood_sentence_3_12d | **15.731 ± 0.180** | 29.639 ± 0.505 | 32.477 ± 0.525 | 32.773 ± 5.615 | 87.204 ± 30.087 |
| lm_batch_likelihood_sentence_4_4d | 11.776 ± 0.150 | 11.999 ± 0.288 | **9.201 ± 0.260** | 19.453 ± 0.180 | 45.997 ± 33.653 |
| nary_matmul_chain_64 | 0.041 ± 0.000 | **0.031 ± 0.000** | 0.051 ± 0.001 | 0.140 ± 0.021 | 0.110 ± 0.037 |
| str_matrix_chain_multiplication_100 | 5.755 ± 0.026 | 5.593 ± 0.206 | **5.517 ± 0.044** | 7.531 ± 0.248 | 11.107 ± 4.215 |
| str_mps_varying_inner_product_200 | 10.539 ± 0.083 | **9.791 ± 0.118** | 15.227 ± 0.128 | 22.173 ± 0.235 | 29.346 ± 19.074 |
| str_nw_mera_closed_120 | **332.194 ± 1.815** | 380.764 ± 1.195 | 451.545 ± 2.463 | 451.160 ± 19.682 | 789.496 ± 27.406 |
| str_nw_mera_open_26 | 339.213 ± 1.421 | 366.468 ± 1.554 | **281.542 ± 1.445** | 364.311 ± 9.437 | 900.171 ± 51.206 |
| tensornetwork_permutation_focus_step409_316 | **95.871 ± 0.844** | 190.690 ± 1.074 | 177.489 ± 0.578 | 352.515 ± 2.566 | 445.029 ± 44.251 |
| tensornetwork_permutation_light_415 | **99.303 ± 0.484** | 137.426 ± 1.116 | 175.056 ± 0.778 | 365.836 ± 5.811 | 424.774 ± 41.120 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=8, OMP_NUM_THREADS=8, RAYON_NUM_THREADS=8.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.503 ± 0.027 | **1.065 ± 0.010** | 5.997 ± 0.806 | 1.470 ± 0.034 | 11.321 ± 9.333 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.469 ± 0.009 | **0.318 ± 0.004** | 5.998 ± 0.012 | 1.053 ± 0.007 | 1.110 ± 1.150 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.131 ± 0.001 | **0.129 ± 0.025** | 5.999 ± 0.017 | 0.562 ± 0.245 | 1.120 ± 0.351 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | **0.110 ± 0.001** | 0.157 ± 0.001 | 5.998 ± 0.003 | 0.573 ± 0.184 | 5.738 ± 5.004 |
| bin_elementwise_mul_2048x2048 | 1.139 ± 0.161 | **1.000 ± 0.241** | 5.112 ± 0.869 | 5.984 ± 0.416 | 13.803 ± 0.355 |
| bin_elementwise_mul_512x512 | 0.027 ± 0.001 | **0.020 ± 0.001** | 5.999 ± 0.002 | 0.405 ± 0.126 | 0.138 ± 0.591 |
| bin_matmul_1024 | 10.611 ± 0.042 | **8.932 ± 0.031** | 19.068 ± 0.231 | 13.780 ± 1.665 | 41.607 ± 9.865 |
| bin_matmul_256 | 0.236 ± 0.004 | **0.169 ± 0.001** | 0.459 ± 0.004 | 0.453 ± 0.116 | 0.671 ± 0.012 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.008 ± 0.000** | 0.030 ± 0.001 | 0.169 ± 0.009 | 0.015 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.001 | **0.028 ± 0.001** | 0.145 ± 0.001 | 0.232 ± 0.055 | 0.053 ± 0.004 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.000 | **0.013 ± 0.000** | 0.028 ± 0.001 | 0.099 ± 0.010 | 0.028 ± 0.003 |
| bin_outer_product_4096 | 8.945 ± 0.573 | 8.403 ± 0.303 | **7.533 ± 0.261** | 16.672 ± 0.208 | 296.975 ± 2.278 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 11.647 ± 0.074 | **10.479 ± 0.016** | 14.781 ± 0.179 | 13.560 ± 0.381 | 34.008 ± 10.037 |
| gm_queen5_5_3.wcsp | 566.941 ± 2.808 | **505.853 ± 2.288** | 662.734 ± 6.290 | 766.268 ± 24.962 | 2535.796 ± 18.163 |
| lm_batch_likelihood_brackets_4_4d | 11.489 ± 0.086 | 11.764 ± 0.095 | **8.689 ± 0.346** | 19.242 ± 0.434 | 59.152 ± 27.429 |
| lm_batch_likelihood_sentence_3_12d | **16.151 ± 0.232** | 31.376 ± 3.102 | 18.539 ± 0.443 | 37.775 ± 6.246 | 92.395 ± 21.588 |
| lm_batch_likelihood_sentence_4_4d | 10.565 ± 0.118 | 12.348 ± 0.293 | **8.996 ± 0.174** | 19.142 ± 0.334 | 40.433 ± 36.840 |
| nary_matmul_chain_64 | 0.041 ± 0.000 | **0.031 ± 0.000** | 0.051 ± 0.001 | 0.140 ± 0.021 | 0.110 ± 0.037 |
| str_matrix_chain_multiplication_100 | 5.943 ± 0.097 | 5.654 ± 0.104 | **5.569 ± 0.119** | 7.588 ± 0.228 | 10.975 ± 0.704 |
| str_mps_varying_inner_product_200 | 13.486 ± 0.134 | **10.753 ± 0.123** | 15.174 ± 0.078 | 21.444 ± 0.266 | 25.335 ± 18.593 |
| str_nw_mera_closed_120 | 419.920 ± 1.951 | 374.475 ± 3.876 | 345.865 ± 1.424 | **345.495 ± 16.665** | 644.041 ± 53.031 |
| str_nw_mera_open_26 | 344.670 ± 1.329 | 365.728 ± 1.108 | **282.947 ± 1.101** | 388.286 ± 6.008 | 912.801 ± 48.881 |
| tensornetwork_permutation_focus_step409_316 | **95.871 ± 0.844** | 190.690 ± 1.074 | 177.489 ± 0.578 | 352.515 ± 2.566 | 445.029 ± 44.251 |
| tensornetwork_permutation_light_415 | **99.303 ± 0.484** | 137.426 ± 1.116 | 175.056 ± 0.778 | 365.836 ± 5.811 | 424.774 ± 41.120 |

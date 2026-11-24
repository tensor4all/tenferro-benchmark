# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Benchmark commit: `9185e8ca26c3fc600458361d0f477495a717daf3`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261124_175224/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261124_175224/run.yaml`
- Timestamp: `2026-11-24T17:52:24.640647+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261124_175224`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

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

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.4`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `1`

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

- `data/results/amd-cpu/cpu/einsum/20261124_175224/jax_cpu_t1_20261124_175224.log`
- `data/results/amd-cpu/cpu/einsum/20261124_175224/julia_omeinsum_t1_20261124_175224.log`
- `data/results/amd-cpu/cpu/einsum/20261124_175224/pytorch_cpu_t1_20261124_175224.log`
- `data/results/amd-cpu/cpu/einsum/20261124_175224/tenferro_eager_t1_20261124_175224.log`
- `data/results/amd-cpu/cpu/einsum/20261124_175224/tenferro_trace_t1_20261124_175224.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.212 ± 0.052 | **2.561 ± 0.035** | 4.119 ± 0.219 | 3.538 ± 0.110 | 8.981 ± 4.153 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.483 ± 0.017 | **0.333 ± 0.010** | 0.610 ± 0.009 | 0.683 ± 0.023 | 0.607 ± 1.270 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.684 ± 0.011 | 0.674 ± 0.019 | **0.135 ± 0.017** | 0.558 ± 0.025 | 3.276 ± 1.701 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.346 ± 0.016 | 0.694 ± 0.020 | **0.135 ± 0.011** | 0.551 ± 0.020 | 5.275 ± 6.521 |
| bin_elementwise_mul_2048x2048 | 2.223 ± 0.134 | **1.785 ± 0.094** | 15.058 ± 0.645 | 17.774 ± 1.211 | 18.230 ± 4.505 |
| bin_elementwise_mul_512x512 | **0.076 ± 0.001** | **0.076 ± 0.001** | 0.087 ± 0.002 | 0.267 ± 0.043 | 0.121 ± 0.954 |
| bin_matmul_1024 | 46.463 ± 4.342 | **40.770 ± 1.092** | 57.551 ± 1.129 | 52.025 ± 1.712 | 63.348 ± 10.856 |
| bin_matmul_256 | 0.751 ± 0.017 | **0.665 ± 0.028** | 1.355 ± 0.028 | 0.946 ± 0.013 | 1.213 ± 0.029 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.000 | **0.005 ± 0.000** | 0.022 ± 0.002 | 0.144 ± 0.022 | 0.011 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.026 ± 0.001 | **0.024 ± 0.001** | 0.104 ± 0.003 | 0.236 ± 0.028 | 0.037 ± 0.003 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.001 | **0.011 ± 0.001** | 0.021 ± 0.001 | 0.116 ± 0.018 | 0.021 ± 0.002 |
| bin_outer_product_4096 | 57.614 ± 1.628 | 55.640 ± 0.855 | **48.984 ± 1.374** | 53.707 ± 0.864 | 262.274 ± 42.925 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 45.332 ± 1.029 | **44.470 ± 0.803** | 59.283 ± 1.598 | 54.947 ± 0.760 | 60.629 ± 9.567 |
| gm_queen5_5_3.wcsp | 5881.662 ± 126.553 | **5480.579 ± 124.343** | 6068.345 ± 137.102 | 6519.456 ± 187.519 | 8770.751 ± 212.843 |
| lm_batch_likelihood_brackets_4_4d | 24.080 ± 0.435 | 28.430 ± 1.550 | 27.629 ± 0.827 | **16.921 ± 5.058** | 25.785 ± 22.391 |
| lm_batch_likelihood_sentence_3_12d | **51.888 ± 1.948** | 85.012 ± 2.853 | 63.315 ± 0.737 | 92.817 ± 2.229 | 74.977 ± 6.070 |
| lm_batch_likelihood_sentence_4_4d | 24.310 ± 0.750 | 30.010 ± 0.606 | 29.615 ± 0.601 | **16.348 ± 0.482** | 31.708 ± 20.710 |
| nary_matmul_chain_64 | 0.034 ± 0.001 | **0.033 ± 0.006** | 0.054 ± 0.002 | 0.132 ± 0.011 | 0.072 ± 0.023 |
| str_matrix_chain_multiplication_100 | **10.775 ± 0.189** | 10.791 ± 0.238 | 15.814 ± 0.281 | 14.745 ± 2.908 | 16.503 ± 0.934 |
| str_mps_varying_inner_product_200 | 14.043 ± 0.306 | **13.802 ± 0.426** | 22.051 ± 0.287 | 22.383 ± 0.588 | 26.003 ± 6.359 |
| str_nw_mera_closed_120 | **1253.929 ± 22.325** | 1337.645 ± 39.003 | 1894.267 ± 55.161 | 2042.303 ± 40.305 | 1890.258 ± 19.617 |
| str_nw_mera_open_26 | **997.446 ± 15.020** | 1403.221 ± 32.639 | 1138.616 ± 23.773 | 1639.998 ± 19.995 | 1681.419 ± 28.387 |
| tensornetwork_permutation_focus_step409_316 | **277.634 ± 3.941** | 391.468 ± 15.310 | 781.502 ± 13.146 | 1005.371 ± 345.881 | 539.041 ± 22.519 |
| tensornetwork_permutation_light_415 | **279.879 ± 4.918** | 329.052 ± 2.298 | 739.840 ± 9.411 | 1047.169 ± 403.875 | 547.233 ± 21.427 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.212 ± 0.052 | **2.561 ± 0.035** | 4.119 ± 0.219 | 3.538 ± 0.110 | 8.981 ± 4.153 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.483 ± 0.017 | **0.333 ± 0.010** | 0.610 ± 0.009 | 0.683 ± 0.023 | 0.607 ± 1.270 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.684 ± 0.011 | 0.674 ± 0.019 | **0.135 ± 0.017** | 0.558 ± 0.025 | 3.276 ± 1.701 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.346 ± 0.016 | 0.694 ± 0.020 | **0.135 ± 0.011** | 0.551 ± 0.020 | 5.275 ± 6.521 |
| bin_elementwise_mul_2048x2048 | 2.223 ± 0.134 | **1.785 ± 0.094** | 15.058 ± 0.645 | 17.774 ± 1.211 | 18.230 ± 4.505 |
| bin_elementwise_mul_512x512 | **0.076 ± 0.001** | **0.076 ± 0.001** | 0.087 ± 0.002 | 0.267 ± 0.043 | 0.121 ± 0.954 |
| bin_matmul_1024 | 46.463 ± 4.342 | **40.770 ± 1.092** | 57.551 ± 1.129 | 52.025 ± 1.712 | 63.348 ± 10.856 |
| bin_matmul_256 | 0.751 ± 0.017 | **0.665 ± 0.028** | 1.355 ± 0.028 | 0.946 ± 0.013 | 1.213 ± 0.029 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.000 | **0.005 ± 0.000** | 0.022 ± 0.002 | 0.144 ± 0.022 | 0.011 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.026 ± 0.001 | **0.024 ± 0.001** | 0.104 ± 0.003 | 0.236 ± 0.028 | 0.037 ± 0.003 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.001 | **0.011 ± 0.001** | 0.021 ± 0.001 | 0.116 ± 0.018 | 0.021 ± 0.002 |
| bin_outer_product_4096 | 57.614 ± 1.628 | 55.640 ± 0.855 | **48.984 ± 1.374** | 53.707 ± 0.864 | 262.274 ± 42.925 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 45.332 ± 1.029 | **44.470 ± 0.803** | 59.283 ± 1.598 | 54.947 ± 0.760 | 60.629 ± 9.567 |
| gm_queen5_5_3.wcsp | **1731.337 ± 38.049** | 2154.663 ± 75.420 | 2669.549 ± 110.587 | 2351.275 ± 34.762 | 3144.346 ± 112.737 |
| lm_batch_likelihood_brackets_4_4d | 22.207 ± 0.804 | 26.349 ± 0.329 | 29.462 ± 0.703 | **16.264 ± 0.218** | 23.966 ± 14.999 |
| lm_batch_likelihood_sentence_3_12d | **54.820 ± 2.399** | 109.739 ± 1.409 | 70.588 ± 1.565 | 59.587 ± 0.882 | 105.648 ± 44.061 |
| lm_batch_likelihood_sentence_4_4d | 22.686 ± 1.523 | 28.759 ± 0.293 | 32.475 ± 1.055 | **15.433 ± 0.510** | 25.216 ± 37.412 |
| nary_matmul_chain_64 | 0.034 ± 0.001 | **0.033 ± 0.006** | 0.054 ± 0.002 | 0.132 ± 0.011 | 0.072 ± 0.023 |
| str_matrix_chain_multiplication_100 | **11.829 ± 0.342** | 12.489 ± 0.269 | 16.003 ± 0.256 | 14.744 ± 0.351 | 16.200 ± 0.883 |
| str_mps_varying_inner_product_200 | **15.858 ± 0.499** | 16.061 ± 0.338 | 21.698 ± 0.452 | 25.830 ± 0.457 | 22.866 ± 1.209 |
| str_nw_mera_closed_120 | 1233.588 ± 20.589 | **1174.328 ± 15.791** | 1497.070 ± 32.445 | 1658.065 ± 32.065 | 1500.639 ± 41.408 |
| str_nw_mera_open_26 | **1010.723 ± 23.983** | 1362.582 ± 15.671 | 1200.098 ± 16.216 | 1758.569 ± 44.381 | 1680.564 ± 70.648 |
| tensornetwork_permutation_focus_step409_316 | **277.634 ± 3.941** | 391.468 ± 15.310 | 781.502 ± 13.146 | 1005.371 ± 345.881 | 539.041 ± 22.519 |
| tensornetwork_permutation_light_415 | **279.879 ± 4.918** | 329.052 ± 2.298 | 739.840 ± 9.411 | 1047.169 ± 403.875 | 547.233 ± 21.427 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261124_182830/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261124_182830/run.yaml`
- Timestamp: `2026-11-24T18:28:30.406333+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261124_182830`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

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

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.4`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `4`

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

- `data/results/amd-cpu/cpu/einsum/20261124_182830/jax_cpu_t4_20261124_182830.log`
- `data/results/amd-cpu/cpu/einsum/20261124_182830/julia_omeinsum_t4_20261124_182830.log`
- `data/results/amd-cpu/cpu/einsum/20261124_182830/pytorch_cpu_t4_20261124_182830.log`
- `data/results/amd-cpu/cpu/einsum/20261124_182830/tenferro_eager_t4_20261124_182830.log`
- `data/results/amd-cpu/cpu/einsum/20261124_182830/tenferro_trace_t4_20261124_182830.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.199 ± 0.038 | **1.720 ± 0.025** | 2.378 ± 2.681 | 2.537 ± 0.599 | 11.449 ± 5.832 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.528 ± 0.021 | 0.376 ± 0.014 | **0.344 ± 0.019** | 0.624 ± 0.015 | 0.681 ± 0.327 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.280 ± 0.017 | 0.260 ± 0.012 | **0.086 ± 0.008** | 0.389 ± 0.017 | 3.255 ± 1.205 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.281 ± 0.011 | 0.266 ± 0.016 | **0.086 ± 0.005** | 0.372 ± 0.013 | 5.665 ± 3.355 |
| bin_elementwise_mul_2048x2048 | **0.911 ± 0.096** | 0.959 ± 0.072 | 6.046 ± 0.309 | 8.837 ± 0.285 | 17.824 ± 0.782 |
| bin_elementwise_mul_512x512 | 0.058 ± 0.006 | **0.051 ± 0.008** | 0.064 ± 0.001 | 0.252 ± 0.019 | 0.137 ± 0.878 |
| bin_matmul_1024 | 24.201 ± 0.387 | **20.754 ± 0.205** | 35.344 ± 0.331 | 28.344 ± 0.809 | 43.640 ± 10.568 |
| bin_matmul_256 | 0.475 ± 0.018 | **0.362 ± 0.013** | 0.895 ± 0.011 | 0.675 ± 0.052 | 0.822 ± 0.020 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.013 ± 0.000 | **0.008 ± 0.000** | 0.031 ± 0.003 | 0.133 ± 0.015 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.030 ± 0.001** | **0.030 ± 0.000** | 0.151 ± 0.013 | 0.231 ± 0.021 | 0.038 ± 0.002 |
| bin_omeinsum_matmul_10x10 | 0.022 ± 0.002 | **0.011 ± 0.001** | 0.033 ± 0.002 | 0.112 ± 0.023 | 0.020 ± 0.001 |
| bin_outer_product_4096 | 20.309 ± 0.733 | 19.085 ± 0.341 | **17.943 ± 0.643** | 26.634 ± 1.476 | 276.723 ± 35.145 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 23.441 ± 0.315 | **22.130 ± 0.321** | 29.968 ± 0.680 | 30.694 ± 1.728 | 41.472 ± 12.194 |
| gm_queen5_5_3.wcsp | 3207.768 ± 35.929 | 3471.706 ± 51.942 | **2603.381 ± 75.059** | 4192.596 ± 92.543 | 8449.540 ± 372.021 |
| lm_batch_likelihood_brackets_4_4d | 15.594 ± 0.183 | 18.756 ± 0.460 | **15.409 ± 0.295** | 20.964 ± 1.493 | 37.699 ± 19.040 |
| lm_batch_likelihood_sentence_3_12d | **27.574 ± 0.481** | 55.730 ± 1.771 | 55.758 ± 0.540 | 50.287 ± 8.219 | 67.546 ± 12.320 |
| lm_batch_likelihood_sentence_4_4d | 16.471 ± 0.270 | 19.852 ± 0.346 | **16.307 ± 0.698** | 20.222 ± 1.354 | 27.479 ± 6.348 |
| nary_matmul_chain_64 | 0.044 ± 0.006 | **0.037 ± 0.001** | 0.065 ± 0.003 | 0.116 ± 0.016 | 0.129 ± 0.040 |
| str_matrix_chain_multiplication_100 | **7.387 ± 0.144** | 7.746 ± 0.266 | 10.661 ± 0.227 | 14.791 ± 0.475 | 13.323 ± 1.246 |
| str_mps_varying_inner_product_200 | 13.321 ± 0.250 | **12.300 ± 0.148** | 24.072 ± 0.830 | 27.691 ± 0.744 | 26.932 ± 2.046 |
| str_nw_mera_closed_120 | **629.178 ± 16.136** | 681.698 ± 13.557 | 952.580 ± 11.701 | 1031.648 ± 25.821 | 1166.971 ± 106.369 |
| str_nw_mera_open_26 | **570.570 ± 14.232** | 685.495 ± 14.043 | 581.992 ± 17.978 | 908.888 ± 21.732 | 1202.766 ± 55.065 |
| tensornetwork_permutation_focus_step409_316 | **164.818 ± 2.039** | 171.535 ± 4.683 | 329.121 ± 12.162 | 456.730 ± 35.662 | 508.212 ± 44.119 |
| tensornetwork_permutation_light_415 | 189.039 ± 5.905 | **137.678 ± 3.724** | 331.630 ± 15.570 | 471.326 ± 30.395 | 494.809 ± 36.279 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.199 ± 0.038 | **1.720 ± 0.025** | 2.378 ± 2.681 | 2.537 ± 0.599 | 11.449 ± 5.832 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.528 ± 0.021 | 0.376 ± 0.014 | **0.344 ± 0.019** | 0.624 ± 0.015 | 0.681 ± 0.327 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.280 ± 0.017 | 0.260 ± 0.012 | **0.086 ± 0.008** | 0.389 ± 0.017 | 3.255 ± 1.205 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.281 ± 0.011 | 0.266 ± 0.016 | **0.086 ± 0.005** | 0.372 ± 0.013 | 5.665 ± 3.355 |
| bin_elementwise_mul_2048x2048 | **0.911 ± 0.096** | 0.959 ± 0.072 | 6.046 ± 0.309 | 8.837 ± 0.285 | 17.824 ± 0.782 |
| bin_elementwise_mul_512x512 | 0.058 ± 0.006 | **0.051 ± 0.008** | 0.064 ± 0.001 | 0.252 ± 0.019 | 0.137 ± 0.878 |
| bin_matmul_1024 | 24.201 ± 0.387 | **20.754 ± 0.205** | 35.344 ± 0.331 | 28.344 ± 0.809 | 43.640 ± 10.568 |
| bin_matmul_256 | 0.475 ± 0.018 | **0.362 ± 0.013** | 0.895 ± 0.011 | 0.675 ± 0.052 | 0.822 ± 0.020 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.013 ± 0.000 | **0.008 ± 0.000** | 0.031 ± 0.003 | 0.133 ± 0.015 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.030 ± 0.001** | **0.030 ± 0.000** | 0.151 ± 0.013 | 0.231 ± 0.021 | 0.038 ± 0.002 |
| bin_omeinsum_matmul_10x10 | 0.022 ± 0.002 | **0.011 ± 0.001** | 0.033 ± 0.002 | 0.112 ± 0.023 | 0.020 ± 0.001 |
| bin_outer_product_4096 | 20.309 ± 0.733 | 19.085 ± 0.341 | **17.943 ± 0.643** | 26.634 ± 1.476 | 276.723 ± 35.145 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 23.441 ± 0.315 | **22.130 ± 0.321** | 29.968 ± 0.680 | 30.694 ± 1.728 | 41.472 ± 12.194 |
| gm_queen5_5_3.wcsp | **849.173 ± 16.895** | 1008.167 ± 20.857 | 1165.389 ± 16.206 | 1661.694 ± 64.928 | 3012.432 ± 55.995 |
| lm_batch_likelihood_brackets_4_4d | **15.068 ± 0.442** | 17.930 ± 0.186 | 15.595 ± 1.458 | 21.240 ± 2.313 | 33.001 ± 35.099 |
| lm_batch_likelihood_sentence_3_12d | **28.137 ± 0.295** | 57.999 ± 1.149 | 34.042 ± 1.788 | 47.006 ± 3.151 | 75.750 ± 40.330 |
| lm_batch_likelihood_sentence_4_4d | **13.897 ± 0.281** | 19.428 ± 0.313 | 16.404 ± 1.217 | 20.957 ± 2.406 | 31.186 ± 5.127 |
| nary_matmul_chain_64 | 0.044 ± 0.006 | **0.037 ± 0.001** | 0.065 ± 0.003 | 0.116 ± 0.016 | 0.129 ± 0.040 |
| str_matrix_chain_multiplication_100 | **7.644 ± 0.182** | 8.365 ± 0.586 | 10.674 ± 0.136 | 14.958 ± 0.570 | 13.903 ± 1.587 |
| str_mps_varying_inner_product_200 | 16.950 ± 0.298 | **14.690 ± 0.272** | 27.483 ± 1.005 | 30.909 ± 1.003 | 24.216 ± 2.480 |
| str_nw_mera_closed_120 | 678.950 ± 12.925 | **644.914 ± 7.637** | 765.534 ± 21.030 | 826.950 ± 32.316 | 944.374 ± 86.315 |
| str_nw_mera_open_26 | **563.100 ± 8.162** | 708.758 ± 18.134 | 570.074 ± 12.445 | 886.431 ± 19.653 | 1215.700 ± 74.331 |
| tensornetwork_permutation_focus_step409_316 | **164.818 ± 2.039** | 171.535 ± 4.683 | 329.121 ± 12.162 | 456.730 ± 35.662 | 508.212 ± 44.119 |
| tensornetwork_permutation_light_415 | 189.039 ± 5.905 | **137.678 ± 3.724** | 331.630 ± 15.570 | 471.326 ± 30.395 | 494.809 ± 36.279 |

# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`
- Benchmark commit: `ab8e941ebccb8606e3781e9d8f6bddeeeab3d2a2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261101_180030/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261101_180030/run.yaml`
- Timestamp: `2026-11-01T18:00:30.835405+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261101_180030`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

### CPU Information

- Model: `Intel(R) Xeon(R) 6973P-C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`

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

- tenferro-rs features: `blas-mkl`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.2`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.3`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `1`

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

- `data/results/amd-cpu/cpu/einsum/20261101_180030/jax_cpu_t1_20261101_180030.log`
- `data/results/amd-cpu/cpu/einsum/20261101_180030/julia_omeinsum_t1_20261101_180030.log`
- `data/results/amd-cpu/cpu/einsum/20261101_180030/pytorch_cpu_t1_20261101_180030.log`
- `data/results/amd-cpu/cpu/einsum/20261101_180030/tenferro_eager_t1_20261101_180030.log`
- `data/results/amd-cpu/cpu/einsum/20261101_180030/tenferro_trace_t1_20261101_180030.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.429 ± 0.058 | **1.324 ± 0.039** | 1.393 ± 0.032 | 3.992 ± 0.068 | 8.204 ± 3.911 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.486 ± 0.037 | **0.175 ± 0.008** | 0.231 ± 0.030 | 0.784 ± 0.027 | 1.958 ± 1.236 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.465 ± 0.016 | 0.418 ± 0.026 | **0.334 ± 0.026** | 0.533 ± 0.038 | 1.086 ± 2.138 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.949 ± 0.021 | 0.429 ± 0.020 | **0.333 ± 0.022** | 0.536 ± 0.028 | 4.024 ± 2.702 |
| bin_elementwise_mul_2048x2048 | 1.504 ± 0.090 | **1.403 ± 0.020** | 11.152 ± 0.628 | 17.183 ± 0.389 | 13.733 ± 2.336 |
| bin_elementwise_mul_512x512 | 0.220 ± 0.010 | **0.215 ± 0.019** | 0.239 ± 0.020 | 0.592 ± 0.046 | 0.262 ± 0.453 |
| bin_matmul_1024 | 28.535 ± 0.948 | **20.055 ± 0.580** | 25.916 ± 5.239 | 57.483 ± 0.382 | 59.759 ± 25.741 |
| bin_matmul_256 | 0.546 ± 0.031 | **0.343 ± 0.020** | 0.481 ± 0.029 | 1.008 ± 0.027 | 0.888 ± 0.027 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.000 | **0.004 ± 0.002** | 0.020 ± 0.007 | 0.110 ± 0.015 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.027 ± 0.001 | **0.022 ± 0.001** | 0.088 ± 0.002 | 0.175 ± 0.022 | 0.040 ± 0.005 |
| bin_omeinsum_matmul_10x10 | 0.013 ± 0.000 | **0.008 ± 0.000** | 0.019 ± 0.001 | 0.077 ± 0.008 | 0.020 ± 0.002 |
| bin_outer_product_4096 | 52.298 ± 1.431 | 46.605 ± 0.260 | **41.614 ± 0.265** | 45.212 ± 0.900 | 315.178 ± 13.299 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 40.511 ± 2.679 | **21.664 ± 1.728** | 31.469 ± 0.658 | 68.363 ± 4.824 | 47.114 ± 17.603 |
| gm_queen5_5_3.wcsp | 7935.124 ± 414.853 | 5312.191 ± 210.782 | **4660.040 ± 161.039** | 5116.540 ± 94.023 | 7395.333 ± 44.518 |
| lm_batch_likelihood_brackets_4_4d | **14.811 ± 0.149** | 35.393 ± 1.363 | 21.084 ± 0.495 | 14.873 ± 0.318 | 33.166 ± 13.555 |
| lm_batch_likelihood_sentence_3_12d | 27.209 ± 0.509 | 132.759 ± 8.480 | **25.591 ± 8.065** | 75.680 ± 3.506 | 118.478 ± 52.362 |
| lm_batch_likelihood_sentence_4_4d | 18.771 ± 0.395 | 42.580 ± 1.234 | 24.387 ± 0.358 | **17.630 ± 0.184** | 47.953 ± 29.897 |
| nary_matmul_chain_64 | 0.032 ± 0.006 | **0.022 ± 0.004** | 0.038 ± 0.011 | 0.137 ± 0.004 | 0.085 ± 0.029 |
| str_matrix_chain_multiplication_100 | **5.703 ± 0.390** | 5.732 ± 0.261 | 6.594 ± 0.332 | 14.887 ± 1.244 | 16.004 ± 3.323 |
| str_mps_varying_inner_product_200 | 8.781 ± 0.333 | 10.783 ± 0.958 | **8.033 ± 0.119** | 21.895 ± 1.200 | 23.673 ± 12.669 |
| str_nw_mera_closed_120 | **995.617 ± 74.936** | 1064.868 ± 75.449 | 1098.711 ± 65.269 | 2156.898 ± 28.466 | 1424.753 ± 48.861 |
| str_nw_mera_open_26 | 912.992 ± 60.361 | 1050.415 ± 57.519 | **740.487 ± 30.679** | 1770.609 ± 101.259 | 1255.515 ± 45.362 |
| tensornetwork_permutation_focus_step409_316 | **319.663 ± 4.659** | 516.644 ± 36.280 | 466.076 ± 6.086 | 687.488 ± 16.427 | 491.367 ± 72.638 |
| tensornetwork_permutation_light_415 | **326.260 ± 6.136** | 385.747 ± 34.036 | 432.884 ± 6.495 | 724.729 ± 63.027 | 502.433 ± 118.219 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.429 ± 0.058 | **1.324 ± 0.039** | 1.393 ± 0.032 | 3.992 ± 0.068 | 8.204 ± 3.911 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.486 ± 0.037 | **0.175 ± 0.008** | 0.231 ± 0.030 | 0.784 ± 0.027 | 1.958 ± 1.236 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.465 ± 0.016 | 0.418 ± 0.026 | **0.334 ± 0.026** | 0.533 ± 0.038 | 1.086 ± 2.138 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.949 ± 0.021 | 0.429 ± 0.020 | **0.333 ± 0.022** | 0.536 ± 0.028 | 4.024 ± 2.702 |
| bin_elementwise_mul_2048x2048 | 1.504 ± 0.090 | **1.403 ± 0.020** | 11.152 ± 0.628 | 17.183 ± 0.389 | 13.733 ± 2.336 |
| bin_elementwise_mul_512x512 | 0.220 ± 0.010 | **0.215 ± 0.019** | 0.239 ± 0.020 | 0.592 ± 0.046 | 0.262 ± 0.453 |
| bin_matmul_1024 | 28.535 ± 0.948 | **20.055 ± 0.580** | 25.916 ± 5.239 | 57.483 ± 0.382 | 59.759 ± 25.741 |
| bin_matmul_256 | 0.546 ± 0.031 | **0.343 ± 0.020** | 0.481 ± 0.029 | 1.008 ± 0.027 | 0.888 ± 0.027 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.009 ± 0.000 | **0.004 ± 0.002** | 0.020 ± 0.007 | 0.110 ± 0.015 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.027 ± 0.001 | **0.022 ± 0.001** | 0.088 ± 0.002 | 0.175 ± 0.022 | 0.040 ± 0.005 |
| bin_omeinsum_matmul_10x10 | 0.013 ± 0.000 | **0.008 ± 0.000** | 0.019 ± 0.001 | 0.077 ± 0.008 | 0.020 ± 0.002 |
| bin_outer_product_4096 | 52.298 ± 1.431 | 46.605 ± 0.260 | **41.614 ± 0.265** | 45.212 ± 0.900 | 315.178 ± 13.299 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 40.511 ± 2.679 | **21.664 ± 1.728** | 31.469 ± 0.658 | 68.363 ± 4.824 | 47.114 ± 17.603 |
| gm_queen5_5_3.wcsp | 2110.789 ± 55.440 | 2000.448 ± 210.490 | **1977.532 ± 62.885** | 2105.762 ± 95.730 | 2571.526 ± 141.312 |
| lm_batch_likelihood_brackets_4_4d | 18.335 ± 0.314 | 38.933 ± 1.468 | 20.626 ± 0.372 | **14.545 ± 0.314** | 53.136 ± 47.793 |
| lm_batch_likelihood_sentence_3_12d | **32.919 ± 0.952** | 134.145 ± 2.659 | 32.940 ± 0.967 | 108.822 ± 1.110 | 113.731 ± 45.674 |
| lm_batch_likelihood_sentence_4_4d | 17.556 ± 0.307 | 44.566 ± 2.003 | 21.040 ± 0.527 | **17.180 ± 0.451** | 50.430 ± 46.152 |
| nary_matmul_chain_64 | 0.032 ± 0.006 | **0.022 ± 0.004** | 0.038 ± 0.011 | 0.137 ± 0.004 | 0.085 ± 0.029 |
| str_matrix_chain_multiplication_100 | **5.683 ± 0.366** | 7.472 ± 0.445 | 9.387 ± 2.279 | 16.852 ± 0.627 | 14.048 ± 1.726 |
| str_mps_varying_inner_product_200 | **9.439 ± 0.561** | 10.751 ± 0.276 | 11.317 ± 0.401 | 26.178 ± 4.419 | 19.754 ± 15.416 |
| str_nw_mera_closed_120 | **813.260 ± 43.618** | 949.040 ± 67.211 | 861.433 ± 35.763 | 1813.298 ± 106.904 | 1125.252 ± 33.274 |
| str_nw_mera_open_26 | 896.528 ± 29.928 | 1042.210 ± 40.664 | **755.775 ± 48.433** | 1810.728 ± 20.191 | 1277.405 ± 53.026 |
| tensornetwork_permutation_focus_step409_316 | **319.663 ± 4.659** | 516.644 ± 36.280 | 466.076 ± 6.086 | 687.488 ± 16.427 | 491.367 ± 72.638 |
| tensornetwork_permutation_light_415 | **326.260 ± 6.136** | 385.747 ± 34.036 | 432.884 ± 6.495 | 724.729 ± 63.027 | 502.433 ± 118.219 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261101_183239/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261101_183239/run.yaml`
- Timestamp: `2026-11-01T18:32:39.800639+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261101_183239`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

### CPU Information

- Model: `Intel(R) Xeon(R) 6973P-C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`

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

- tenferro-rs features: `blas-mkl`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.2`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.3`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `4`

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

- `data/results/amd-cpu/cpu/einsum/20261101_183239/jax_cpu_t4_20261101_183239.log`
- `data/results/amd-cpu/cpu/einsum/20261101_183239/julia_omeinsum_t4_20261101_183239.log`
- `data/results/amd-cpu/cpu/einsum/20261101_183239/pytorch_cpu_t4_20261101_183239.log`
- `data/results/amd-cpu/cpu/einsum/20261101_183239/tenferro_eager_t4_20261101_183239.log`
- `data/results/amd-cpu/cpu/einsum/20261101_183239/tenferro_trace_t4_20261101_183239.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.701 ± 3.195 | 0.958 ± 0.015 | **0.772 ± 0.051** | 2.556 ± 0.536 | 8.671 ± 6.242 |
| bin_batched_matmul_b32_m64_n64_k64 | 9.000 ± 0.863 | 0.245 ± 0.014 | **0.121 ± 0.011** | 0.683 ± 0.026 | 0.891 ± 1.369 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.239 ± 0.023 | 5.999 ± 5.995 | **0.186 ± 0.013** | 0.434 ± 0.037 | 0.829 ± 2.049 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.996 ± 0.014 | 1.153 ± 5.763 | **0.193 ± 0.017** | 0.436 ± 0.022 | 5.242 ± 4.767 |
| bin_elementwise_mul_2048x2048 | 5.999 ± 0.001 | **0.609 ± 0.016** | 4.531 ± 0.197 | 9.152 ± 0.139 | 13.501 ± 1.009 |
| bin_elementwise_mul_512x512 | 0.120 ± 0.007 | **0.113 ± 0.006** | 0.136 ± 0.012 | 0.431 ± 0.039 | 0.349 ± 0.244 |
| bin_matmul_1024 | 19.985 ± 0.997 | 10.949 ± 0.397 | **10.333 ± 0.773** | 27.862 ± 0.518 | 40.089 ± 14.433 |
| bin_matmul_256 | 5.992 ± 4.992 | **0.206 ± 0.019** | 0.220 ± 0.024 | 0.727 ± 0.049 | 0.595 ± 0.090 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.010 ± 0.000 | **0.008 ± 0.000** | 0.020 ± 0.001 | 0.109 ± 0.031 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.001 | **0.028 ± 0.000** | 0.093 ± 0.004 | 0.191 ± 0.014 | 0.035 ± 0.002 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.000 | **0.011 ± 0.000** | 0.018 ± 0.001 | 0.082 ± 0.007 | 0.020 ± 0.004 |
| bin_outer_product_4096 | 17.805 ± 0.348 | 17.773 ± 0.979 | **15.017 ± 0.171** | 24.549 ± 0.394 | 295.306 ± 7.866 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 20.661 ± 0.805 | 14.805 ± 1.480 | **14.798 ± 0.449** | 32.849 ± 1.655 | 38.670 ± 10.943 |
| gm_queen5_5_3.wcsp | 5645.355 ± 147.693 | 3845.385 ± 102.886 | **1955.497 ± 50.033** | 3288.620 ± 70.452 | 6673.411 ± 245.992 |
| lm_batch_likelihood_brackets_4_4d | 85.093 ± 50.919 | 117.340 ± 47.559 | **10.024 ± 0.188** | 22.431 ± 0.826 | 43.866 ± 35.390 |
| lm_batch_likelihood_sentence_3_12d | 42.145 ± 16.486 | 131.546 ± 2.854 | **14.160 ± 0.333** | 45.208 ± 12.581 | 106.158 ± 28.011 |
| lm_batch_likelihood_sentence_4_4d | 125.044 ± 59.147 | 140.475 ± 71.116 | **11.350 ± 0.197** | 22.717 ± 1.443 | 46.266 ± 41.347 |
| nary_matmul_chain_64 | 0.040 ± 0.002 | **0.029 ± 0.002** | 0.033 ± 0.003 | 0.127 ± 0.005 | 0.093 ± 0.025 |
| str_matrix_chain_multiplication_100 | **3.519 ± 0.133** | 3.759 ± 0.128 | 4.315 ± 0.127 | 13.429 ± 0.633 | 12.982 ± 2.421 |
| str_mps_varying_inner_product_200 | 10.966 ± 0.242 | 12.861 ± 1.039 | **8.930 ± 0.161** | 27.014 ± 0.514 | 21.259 ± 22.990 |
| str_nw_mera_closed_120 | 664.577 ± 9.702 | 808.225 ± 49.037 | **577.788 ± 32.765** | 964.673 ± 54.941 | 959.822 ± 45.612 |
| str_nw_mera_open_26 | 798.447 ± 20.526 | 711.389 ± 49.775 | **358.134 ± 5.825** | 804.327 ± 12.111 | 993.942 ± 35.617 |
| tensornetwork_permutation_focus_step409_316 | 317.541 ± 11.011 | 431.456 ± 41.686 | **220.088 ± 20.229** | 366.321 ± 15.805 | 446.290 ± 42.153 |
| tensornetwork_permutation_light_415 | 330.811 ± 14.392 | 364.454 ± 69.095 | **218.864 ± 4.589** | 367.775 ± 5.960 | 444.028 ± 55.478 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.701 ± 3.195 | 0.958 ± 0.015 | **0.772 ± 0.051** | 2.556 ± 0.536 | 8.671 ± 6.242 |
| bin_batched_matmul_b32_m64_n64_k64 | 9.000 ± 0.863 | 0.245 ± 0.014 | **0.121 ± 0.011** | 0.683 ± 0.026 | 0.891 ± 1.369 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.239 ± 0.023 | 5.999 ± 5.995 | **0.186 ± 0.013** | 0.434 ± 0.037 | 0.829 ± 2.049 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.996 ± 0.014 | 1.153 ± 5.763 | **0.193 ± 0.017** | 0.436 ± 0.022 | 5.242 ± 4.767 |
| bin_elementwise_mul_2048x2048 | 5.999 ± 0.001 | **0.609 ± 0.016** | 4.531 ± 0.197 | 9.152 ± 0.139 | 13.501 ± 1.009 |
| bin_elementwise_mul_512x512 | 0.120 ± 0.007 | **0.113 ± 0.006** | 0.136 ± 0.012 | 0.431 ± 0.039 | 0.349 ± 0.244 |
| bin_matmul_1024 | 19.985 ± 0.997 | 10.949 ± 0.397 | **10.333 ± 0.773** | 27.862 ± 0.518 | 40.089 ± 14.433 |
| bin_matmul_256 | 5.992 ± 4.992 | **0.206 ± 0.019** | 0.220 ± 0.024 | 0.727 ± 0.049 | 0.595 ± 0.090 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.010 ± 0.000 | **0.008 ± 0.000** | 0.020 ± 0.001 | 0.109 ± 0.031 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.001 | **0.028 ± 0.000** | 0.093 ± 0.004 | 0.191 ± 0.014 | 0.035 ± 0.002 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.000 | **0.011 ± 0.000** | 0.018 ± 0.001 | 0.082 ± 0.007 | 0.020 ± 0.004 |
| bin_outer_product_4096 | 17.805 ± 0.348 | 17.773 ± 0.979 | **15.017 ± 0.171** | 24.549 ± 0.394 | 295.306 ± 7.866 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 20.661 ± 0.805 | 14.805 ± 1.480 | **14.798 ± 0.449** | 32.849 ± 1.655 | 38.670 ± 10.943 |
| gm_queen5_5_3.wcsp | 1814.500 ± 65.310 | 1449.717 ± 59.672 | **763.478 ± 44.908** | 1347.745 ± 15.416 | 2324.569 ± 59.756 |
| lm_batch_likelihood_brackets_4_4d | 94.958 ± 13.877 | 119.675 ± 44.702 | **9.998 ± 0.274** | 20.794 ± 0.962 | 70.971 ± 33.373 |
| lm_batch_likelihood_sentence_3_12d | 85.994 ± 23.686 | 132.670 ± 27.735 | **14.182 ± 0.832** | 53.590 ± 10.935 | 104.035 ± 31.644 |
| lm_batch_likelihood_sentence_4_4d | 98.994 ± 8.554 | 156.149 ± 132.124 | **10.499 ± 0.151** | 23.826 ± 1.799 | 50.591 ± 51.629 |
| nary_matmul_chain_64 | 0.040 ± 0.002 | **0.029 ± 0.002** | 0.033 ± 0.003 | 0.127 ± 0.005 | 0.093 ± 0.025 |
| str_matrix_chain_multiplication_100 | **3.643 ± 0.059** | 3.655 ± 0.122 | 4.510 ± 0.213 | 13.814 ± 0.818 | 14.286 ± 1.280 |
| str_mps_varying_inner_product_200 | 12.724 ± 0.835 | 14.714 ± 0.641 | **10.581 ± 0.391** | 30.190 ± 0.676 | 19.773 ± 6.439 |
| str_nw_mera_closed_120 | 647.741 ± 35.772 | 608.763 ± 14.257 | **460.148 ± 11.949** | 828.465 ± 21.312 | 750.735 ± 51.464 |
| str_nw_mera_open_26 | 812.034 ± 34.119 | 694.749 ± 47.202 | **362.697 ± 3.618** | 827.628 ± 48.661 | 997.143 ± 88.270 |
| tensornetwork_permutation_focus_step409_316 | 317.541 ± 11.011 | 431.456 ± 41.686 | **220.088 ± 20.229** | 366.321 ± 15.805 | 446.290 ± 42.153 |
| tensornetwork_permutation_light_415 | 330.811 ± 14.392 | 364.454 ± 69.095 | **218.864 ± 4.589** | 367.775 ± 5.960 | 444.028 ± 55.478 |

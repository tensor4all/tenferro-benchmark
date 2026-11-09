# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `ad9aa8a2fc06c36f8ce1d1dfb1ce9fb1cd9a9c46`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261109_180601/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261109_180601/run.yaml`
- Timestamp: `2026-11-09T18:06:01.414278+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261109_180601`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
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

- `data/results/amd-cpu/cpu/einsum/20261109_180601/jax_cpu_t1_20261109_180601.log`
- `data/results/amd-cpu/cpu/einsum/20261109_180601/julia_omeinsum_t1_20261109_180601.log`
- `data/results/amd-cpu/cpu/einsum/20261109_180601/pytorch_cpu_t1_20261109_180601.log`
- `data/results/amd-cpu/cpu/einsum/20261109_180601/tenferro_eager_t1_20261109_180601.log`
- `data/results/amd-cpu/cpu/einsum/20261109_180601/tenferro_trace_t1_20261109_180601.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.853 ± 0.127 | **3.322 ± 0.029** | 3.528 ± 0.018 | 3.366 ± 0.103 | 9.766 ± 5.448 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.626 ± 0.008 | **0.501 ± 0.010** | 0.553 ± 0.011 | 0.612 ± 0.015 | 0.611 ± 1.437 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.625 ± 0.011 | 0.624 ± 0.009 | **0.128 ± 0.009** | 0.485 ± 0.023 | 0.656 ± 2.335 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.948 ± 0.006 | 0.624 ± 0.015 | **0.127 ± 0.005** | 0.482 ± 0.022 | 3.251 ± 3.257 |
| bin_elementwise_mul_2048x2048 | 1.864 ± 0.071 | **1.745 ± 0.074** | 13.816 ± 0.404 | 16.136 ± 0.627 | 16.891 ± 2.506 |
| bin_elementwise_mul_512x512 | 0.076 ± 0.002 | **0.072 ± 0.002** | 0.082 ± 0.005 | 0.261 ± 0.015 | 0.122 ± 0.020 |
| bin_matmul_1024 | 50.543 ± 0.092 | **42.141 ± 0.178** | 52.092 ± 0.312 | 46.823 ± 0.493 | 60.134 ± 22.537 |
| bin_matmul_256 | 0.932 ± 0.014 | **0.695 ± 0.006** | 1.237 ± 0.007 | 0.857 ± 0.029 | 1.088 ± 0.723 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.007 ± 0.000** | 0.020 ± 0.001 | 0.129 ± 0.012 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.032 ± 0.001 | **0.030 ± 0.001** | 0.095 ± 0.003 | 0.205 ± 0.016 | 0.078 ± 0.048 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.000 | **0.013 ± 0.001** | 0.019 ± 0.002 | 0.104 ± 0.012 | 0.018 ± 0.001 |
| bin_outer_product_4096 | 51.903 ± 0.652 | 51.239 ± 0.682 | **45.581 ± 0.658** | 46.446 ± 0.781 | 264.491 ± 3.677 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 49.084 ± 0.609 | **43.432 ± 0.311** | 53.416 ± 0.431 | 48.778 ± 0.454 | 49.334 ± 7.561 |
| gm_queen5_5_3.wcsp | 5957.189 ± 67.942 | 5898.441 ± 45.917 | **5524.885 ± 49.507** | 5771.406 ± 35.352 | 8142.597 ± 93.022 |
| lm_batch_likelihood_brackets_4_4d | 22.248 ± 0.120 | 25.626 ± 1.208 | 26.602 ± 1.294 | **11.459 ± 0.185** | 51.990 ± 26.262 |
| lm_batch_likelihood_sentence_3_12d | **50.310 ± 0.145** | 121.029 ± 1.219 | 60.139 ± 1.540 | 80.911 ± 0.885 | 105.189 ± 9.741 |
| lm_batch_likelihood_sentence_4_4d | 23.259 ± 0.307 | 29.887 ± 0.917 | 30.216 ± 0.968 | **13.583 ± 0.086** | 56.293 ± 27.233 |
| nary_matmul_chain_64 | 0.045 ± 0.001 | **0.037 ± 0.001** | 0.051 ± 0.002 | 0.117 ± 0.006 | 0.103 ± 0.047 |
| str_matrix_chain_multiplication_100 | 12.820 ± 0.213 | 14.996 ± 0.291 | 16.022 ± 1.119 | **12.634 ± 0.124** | 15.238 ± 1.249 |
| str_mps_varying_inner_product_200 | **17.081 ± 0.139** | 19.696 ± 0.124 | 21.562 ± 0.252 | 19.689 ± 0.280 | 28.184 ± 7.145 |
| str_nw_mera_closed_120 | **1435.562 ± 6.886** | 1567.082 ± 28.795 | 1744.481 ± 10.547 | 1857.873 ± 14.553 | 1762.774 ± 66.967 |
| str_nw_mera_open_26 | 1107.067 ± 5.198 | 1308.304 ± 4.800 | **1058.703 ± 9.925** | 1539.974 ± 26.163 | 1568.347 ± 60.102 |
| tensornetwork_permutation_focus_step409_316 | **397.594 ± 2.732** | 510.833 ± 6.172 | 722.327 ± 3.537 | 649.211 ± 9.167 | 571.666 ± 42.024 |
| tensornetwork_permutation_light_415 | **394.144 ± 3.863** | 443.724 ± 2.575 | 698.917 ± 6.077 | 698.195 ± 17.117 | 585.235 ± 23.848 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.853 ± 0.127 | **3.322 ± 0.029** | 3.528 ± 0.018 | 3.366 ± 0.103 | 9.766 ± 5.448 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.626 ± 0.008 | **0.501 ± 0.010** | 0.553 ± 0.011 | 0.612 ± 0.015 | 0.611 ± 1.437 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.625 ± 0.011 | 0.624 ± 0.009 | **0.128 ± 0.009** | 0.485 ± 0.023 | 0.656 ± 2.335 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.948 ± 0.006 | 0.624 ± 0.015 | **0.127 ± 0.005** | 0.482 ± 0.022 | 3.251 ± 3.257 |
| bin_elementwise_mul_2048x2048 | 1.864 ± 0.071 | **1.745 ± 0.074** | 13.816 ± 0.404 | 16.136 ± 0.627 | 16.891 ± 2.506 |
| bin_elementwise_mul_512x512 | 0.076 ± 0.002 | **0.072 ± 0.002** | 0.082 ± 0.005 | 0.261 ± 0.015 | 0.122 ± 0.020 |
| bin_matmul_1024 | 50.543 ± 0.092 | **42.141 ± 0.178** | 52.092 ± 0.312 | 46.823 ± 0.493 | 60.134 ± 22.537 |
| bin_matmul_256 | 0.932 ± 0.014 | **0.695 ± 0.006** | 1.237 ± 0.007 | 0.857 ± 0.029 | 1.088 ± 0.723 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.007 ± 0.000** | 0.020 ± 0.001 | 0.129 ± 0.012 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.032 ± 0.001 | **0.030 ± 0.001** | 0.095 ± 0.003 | 0.205 ± 0.016 | 0.078 ± 0.048 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.000 | **0.013 ± 0.001** | 0.019 ± 0.002 | 0.104 ± 0.012 | 0.018 ± 0.001 |
| bin_outer_product_4096 | 51.903 ± 0.652 | 51.239 ± 0.682 | **45.581 ± 0.658** | 46.446 ± 0.781 | 264.491 ± 3.677 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 49.084 ± 0.609 | **43.432 ± 0.311** | 53.416 ± 0.431 | 48.778 ± 0.454 | 49.334 ± 7.561 |
| gm_queen5_5_3.wcsp | **1970.283 ± 5.179** | 2302.180 ± 20.522 | 2511.469 ± 190.599 | 2161.186 ± 8.120 | 3033.197 ± 50.180 |
| lm_batch_likelihood_brackets_4_4d | 22.536 ± 0.162 | 25.719 ± 1.275 | 27.182 ± 0.287 | **12.873 ± 0.378** | 59.457 ± 35.589 |
| lm_batch_likelihood_sentence_3_12d | **49.268 ± 0.241** | 130.837 ± 3.010 | 65.447 ± 2.020 | 104.172 ± 0.502 | 124.535 ± 28.002 |
| lm_batch_likelihood_sentence_4_4d | 23.328 ± 0.204 | 30.695 ± 1.798 | 29.843 ± 0.393 | **12.604 ± 0.176** | 42.640 ± 34.878 |
| nary_matmul_chain_64 | 0.045 ± 0.001 | **0.037 ± 0.001** | 0.051 ± 0.002 | 0.117 ± 0.006 | 0.103 ± 0.047 |
| str_matrix_chain_multiplication_100 | 13.649 ± 0.195 | 14.589 ± 0.263 | 15.651 ± 0.265 | **12.651 ± 0.121** | 15.526 ± 0.963 |
| str_mps_varying_inner_product_200 | 19.872 ± 0.072 | **19.747 ± 0.116** | 20.096 ± 0.280 | 22.520 ± 0.651 | 26.301 ± 10.941 |
| str_nw_mera_closed_120 | 1306.855 ± 9.910 | **1295.571 ± 11.647** | 1391.853 ± 15.255 | 1560.309 ± 21.064 | 1434.658 ± 58.014 |
| str_nw_mera_open_26 | 1110.210 ± 12.323 | 1294.037 ± 9.043 | **1094.351 ± 10.448** | 1599.891 ± 5.814 | 1527.946 ± 95.687 |
| tensornetwork_permutation_focus_step409_316 | **397.594 ± 2.732** | 510.833 ± 6.172 | 722.327 ± 3.537 | 649.211 ± 9.167 | 571.666 ± 42.024 |
| tensornetwork_permutation_light_415 | **394.144 ± 3.863** | 443.724 ± 2.575 | 698.917 ± 6.077 | 698.195 ± 17.117 | 585.235 ± 23.848 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261109_184059/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261109_184059/run.yaml`
- Timestamp: `2026-11-09T18:40:59.908465+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261109_184059`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
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

- `data/results/amd-cpu/cpu/einsum/20261109_184059/jax_cpu_t4_20261109_184059.log`
- `data/results/amd-cpu/cpu/einsum/20261109_184059/julia_omeinsum_t4_20261109_184059.log`
- `data/results/amd-cpu/cpu/einsum/20261109_184059/pytorch_cpu_t4_20261109_184059.log`
- `data/results/amd-cpu/cpu/einsum/20261109_184059/tenferro_eager_t4_20261109_184059.log`
- `data/results/amd-cpu/cpu/einsum/20261109_184059/tenferro_trace_t4_20261109_184059.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 9.211 ± 0.101 | 4.731 ± 0.035 | **1.924 ± 1.556** | 2.566 ± 0.646 | 9.826 ± 10.400 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.647 ± 0.010 | 1.202 ± 0.019 | **0.290 ± 0.018** | 0.623 ± 0.019 | 0.616 ± 0.060 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.240 ± 0.014 | 0.239 ± 0.009 | **0.076 ± 0.002** | 0.354 ± 0.018 | 0.727 ± 2.162 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.242 ± 0.009 | 0.248 ± 0.055 | **0.079 ± 0.030** | 0.352 ± 0.015 | 5.403 ± 4.653 |
| bin_elementwise_mul_2048x2048 | **0.908 ± 0.032** | 0.935 ± 0.067 | 5.424 ± 0.256 | 8.453 ± 0.230 | 16.983 ± 0.313 |
| bin_elementwise_mul_512x512 | 0.050 ± 0.008 | **0.047 ± 0.006** | 0.072 ± 0.010 | 0.328 ± 0.144 | 0.144 ± 0.776 |
| bin_matmul_1024 | 57.203 ± 0.493 | 47.687 ± 0.584 | 31.574 ± 0.493 | **27.075 ± 1.264** | 41.479 ± 16.637 |
| bin_matmul_256 | 1.152 ± 0.012 | 0.816 ± 0.157 | 0.836 ± 0.009 | **0.612 ± 0.030** | 0.690 ± 0.010 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.007 ± 0.000** | 0.028 ± 0.002 | 0.130 ± 0.018 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.045 ± 0.002 | 0.043 ± 0.002 | 0.139 ± 0.014 | 0.226 ± 0.020 | **0.036 ± 0.001** |
| bin_omeinsum_matmul_10x10 | 0.048 ± 0.002 | 0.035 ± 0.002 | 0.029 ± 0.001 | 0.079 ± 0.017 | **0.018 ± 0.001** |
| bin_outer_product_4096 | 18.432 ± 0.244 | 17.911 ± 0.272 | **15.805 ± 0.711** | 26.708 ± 0.207 | 252.707 ± 4.382 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 55.013 ± 0.947 | 49.808 ± 1.164 | **26.602 ± 0.222** | 28.926 ± 1.245 | 37.848 ± 9.772 |
| gm_queen5_5_3.wcsp | 5531.150 ± 46.514 | 5046.002 ± 58.897 | **2372.179 ± 43.050** | 3848.744 ± 74.589 | 7418.956 ± 138.675 |
| lm_batch_likelihood_brackets_4_4d | 25.844 ± 0.534 | 39.604 ± 1.993 | **13.836 ± 0.331** | 18.939 ± 0.638 | 23.909 ± 8.945 |
| lm_batch_likelihood_sentence_3_12d | 58.672 ± 1.669 | 161.388 ± 33.281 | 52.550 ± 1.255 | **43.432 ± 5.201** | 99.771 ± 36.395 |
| lm_batch_likelihood_sentence_4_4d | 28.255 ± 0.472 | 44.424 ± 1.269 | **15.218 ± 0.378** | 19.662 ± 0.889 | 44.720 ± 43.456 |
| nary_matmul_chain_64 | 0.111 ± 0.006 | 0.102 ± 0.001 | **0.061 ± 0.003** | 0.162 ± 0.024 | 0.108 ± 0.018 |
| str_matrix_chain_multiplication_100 | 15.302 ± 0.113 | 16.107 ± 0.089 | **9.743 ± 0.249** | 13.420 ± 0.427 | 15.082 ± 0.856 |
| str_mps_varying_inner_product_200 | 23.246 ± 0.235 | 25.095 ± 0.356 | **22.169 ± 0.225** | 25.628 ± 0.426 | 25.462 ± 17.641 |
| str_nw_mera_closed_120 | 1549.653 ± 13.362 | 1674.938 ± 16.587 | **842.028 ± 8.376** | 916.484 ± 30.440 | 1152.521 ± 59.766 |
| str_nw_mera_open_26 | 1174.920 ± 7.320 | 1196.371 ± 2.964 | **527.568 ± 5.170** | 773.478 ± 20.281 | 1206.548 ± 43.201 |
| tensornetwork_permutation_focus_step409_316 | 416.257 ± 5.332 | 441.060 ± 2.772 | **298.127 ± 2.283** | 397.868 ± 14.241 | 538.216 ± 28.471 |
| tensornetwork_permutation_light_415 | 415.867 ± 2.986 | 411.006 ± 6.899 | **292.192 ± 4.217** | 397.955 ± 9.334 | 525.505 ± 31.405 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 9.211 ± 0.101 | 4.731 ± 0.035 | **1.924 ± 1.556** | 2.566 ± 0.646 | 9.826 ± 10.400 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.647 ± 0.010 | 1.202 ± 0.019 | **0.290 ± 0.018** | 0.623 ± 0.019 | 0.616 ± 0.060 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.240 ± 0.014 | 0.239 ± 0.009 | **0.076 ± 0.002** | 0.354 ± 0.018 | 0.727 ± 2.162 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.242 ± 0.009 | 0.248 ± 0.055 | **0.079 ± 0.030** | 0.352 ± 0.015 | 5.403 ± 4.653 |
| bin_elementwise_mul_2048x2048 | **0.908 ± 0.032** | 0.935 ± 0.067 | 5.424 ± 0.256 | 8.453 ± 0.230 | 16.983 ± 0.313 |
| bin_elementwise_mul_512x512 | 0.050 ± 0.008 | **0.047 ± 0.006** | 0.072 ± 0.010 | 0.328 ± 0.144 | 0.144 ± 0.776 |
| bin_matmul_1024 | 57.203 ± 0.493 | 47.687 ± 0.584 | 31.574 ± 0.493 | **27.075 ± 1.264** | 41.479 ± 16.637 |
| bin_matmul_256 | 1.152 ± 0.012 | 0.816 ± 0.157 | 0.836 ± 0.009 | **0.612 ± 0.030** | 0.690 ± 0.010 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.007 ± 0.000** | 0.028 ± 0.002 | 0.130 ± 0.018 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.045 ± 0.002 | 0.043 ± 0.002 | 0.139 ± 0.014 | 0.226 ± 0.020 | **0.036 ± 0.001** |
| bin_omeinsum_matmul_10x10 | 0.048 ± 0.002 | 0.035 ± 0.002 | 0.029 ± 0.001 | 0.079 ± 0.017 | **0.018 ± 0.001** |
| bin_outer_product_4096 | 18.432 ± 0.244 | 17.911 ± 0.272 | **15.805 ± 0.711** | 26.708 ± 0.207 | 252.707 ± 4.382 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 55.013 ± 0.947 | 49.808 ± 1.164 | **26.602 ± 0.222** | 28.926 ± 1.245 | 37.848 ± 9.772 |
| gm_queen5_5_3.wcsp | 2052.799 ± 34.948 | 1876.306 ± 54.213 | **1054.466 ± 13.158** | 1504.746 ± 20.848 | 2724.772 ± 54.591 |
| lm_batch_likelihood_brackets_4_4d | 26.352 ± 0.380 | 43.085 ± 3.124 | **13.518 ± 0.254** | 19.419 ± 1.172 | 59.480 ± 27.589 |
| lm_batch_likelihood_sentence_3_12d | 57.425 ± 3.063 | 194.294 ± 0.911 | **30.671 ± 0.766** | 43.894 ± 5.972 | 107.736 ± 16.999 |
| lm_batch_likelihood_sentence_4_4d | 28.285 ± 0.617 | 48.611 ± 2.251 | **14.828 ± 0.211** | 19.367 ± 0.512 | 44.007 ± 58.231 |
| nary_matmul_chain_64 | 0.111 ± 0.006 | 0.102 ± 0.001 | **0.061 ± 0.003** | 0.162 ± 0.024 | 0.108 ± 0.018 |
| str_matrix_chain_multiplication_100 | 16.787 ± 0.143 | 17.606 ± 0.138 | **9.743 ± 0.689** | 13.154 ± 0.946 | 14.755 ± 1.840 |
| str_mps_varying_inner_product_200 | 26.205 ± 0.167 | 26.409 ± 0.182 | 25.710 ± 0.564 | 28.063 ± 0.354 | **21.285 ± 5.019** |
| str_nw_mera_closed_120 | 1378.942 ± 9.740 | 1358.605 ± 33.023 | **673.846 ± 3.147** | 740.107 ± 19.732 | 909.005 ± 63.350 |
| str_nw_mera_open_26 | 1172.841 ± 12.069 | 1209.240 ± 41.134 | **525.850 ± 5.339** | 791.278 ± 14.401 | 1182.423 ± 33.032 |
| tensornetwork_permutation_focus_step409_316 | 416.257 ± 5.332 | 441.060 ± 2.772 | **298.127 ± 2.283** | 397.868 ± 14.241 | 538.216 ± 28.471 |
| tensornetwork_permutation_light_415 | 415.867 ± 2.986 | 411.006 ± 6.899 | **292.192 ± 4.217** | 397.955 ± 9.334 | 525.505 ± 31.405 |

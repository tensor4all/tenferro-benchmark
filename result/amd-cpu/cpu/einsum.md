# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `7da876a214acb65eb9ea099adef67b41ef8efc98`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261127_175648/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261127_175648/run.yaml`
- Timestamp: `2026-11-27T17:56:48.092254+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261127_175648`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

### CPU Information

- Model: `INTEL(R) XEON(R) PLATINUM 8573C`
- Vendor: `GenuineIntel`
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

- `data/results/amd-cpu/cpu/einsum/20261127_175648/jax_cpu_t1_20261127_175648.log`
- `data/results/amd-cpu/cpu/einsum/20261127_175648/julia_omeinsum_t1_20261127_175648.log`
- `data/results/amd-cpu/cpu/einsum/20261127_175648/pytorch_cpu_t1_20261127_175648.log`
- `data/results/amd-cpu/cpu/einsum/20261127_175648/tenferro_eager_t1_20261127_175648.log`
- `data/results/amd-cpu/cpu/einsum/20261127_175648/tenferro_trace_t1_20261127_175648.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.290 ± 0.043 | **1.611 ± 0.019** | 1.636 ± 0.017 | 4.819 ± 0.082 | 8.074 ± 4.977 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.343 ± 0.025 | **0.222 ± 0.003** | 0.250 ± 0.023 | 0.948 ± 0.036 | 0.683 ± 0.270 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 1.158 ± 0.013 | 1.161 ± 0.013 | **0.373 ± 0.025** | 0.623 ± 0.053 | 1.055 ± 2.846 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.089 ± 0.019 | 1.187 ± 0.019 | **0.370 ± 0.013** | 0.625 ± 0.042 | 3.706 ± 2.289 |
| bin_elementwise_mul_2048x2048 | **1.960 ± 0.254** | 3.068 ± 0.805 | 15.545 ± 0.230 | 18.811 ± 0.180 | 16.869 ± 3.326 |
| bin_elementwise_mul_512x512 | 0.232 ± 0.015 | **0.227 ± 0.005** | 0.251 ± 0.021 | 0.521 ± 0.017 | 0.317 ± 0.541 |
| bin_matmul_1024 | 30.469 ± 0.961 | 26.794 ± 0.263 | **24.804 ± 0.205** | 67.813 ± 0.626 | 50.550 ± 7.345 |
| bin_matmul_256 | 0.462 ± 0.033 | **0.377 ± 0.009** | 0.443 ± 0.021 | 1.169 ± 0.025 | 0.867 ± 0.214 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.006 ± 0.000** | 0.025 ± 0.010 | 0.152 ± 0.015 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.027 ± 0.001 | **0.026 ± 0.001** | 0.094 ± 0.008 | 0.205 ± 0.029 | 0.070 ± 0.030 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.000 | **0.014 ± 0.007** | 0.019 ± 0.002 | 0.101 ± 0.012 | 0.020 ± 0.008 |
| bin_outer_product_4096 | 59.132 ± 0.596 | 58.579 ± 0.684 | **51.043 ± 0.804** | 54.279 ± 0.701 | 329.265 ± 34.513 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 29.632 ± 0.359 | **26.602 ± 0.201** | 37.158 ± 0.317 | 75.014 ± 1.297 | 45.427 ± 7.512 |
| gm_queen5_5_3.wcsp | 6568.567 ± 85.366 | 5170.583 ± 56.316 | **4997.046 ± 71.410** | 5509.767 ± 157.794 | 7676.194 ± 86.430 |
| lm_batch_likelihood_brackets_4_4d | 21.181 ± 0.307 | 38.081 ± 0.884 | 26.699 ± 0.500 | **20.005 ± 0.450** | 31.337 ± 20.853 |
| lm_batch_likelihood_sentence_3_12d | 40.780 ± 0.637 | 92.041 ± 1.242 | **32.396 ± 1.265** | 103.393 ± 0.905 | 67.426 ± 13.208 |
| lm_batch_likelihood_sentence_4_4d | **21.620 ± 0.381** | 40.506 ± 1.153 | 28.569 ± 0.499 | 21.829 ± 0.294 | 35.008 ± 7.787 |
| nary_matmul_chain_64 | 0.032 ± 0.001 | **0.026 ± 0.007** | 0.050 ± 0.020 | 0.155 ± 0.010 | 0.052 ± 0.030 |
| str_matrix_chain_multiplication_100 | 8.484 ± 0.188 | 9.169 ± 0.102 | **8.129 ± 0.142** | 16.968 ± 1.899 | 13.780 ± 0.230 |
| str_mps_varying_inner_product_200 | 11.766 ± 0.135 | 13.246 ± 0.215 | **10.046 ± 0.380** | 25.649 ± 0.325 | 20.846 ± 1.727 |
| str_nw_mera_closed_120 | **852.314 ± 7.140** | 913.473 ± 13.203 | 1218.198 ± 23.820 | 2412.316 ± 41.993 | 1418.548 ± 22.464 |
| str_nw_mera_open_26 | **731.935 ± 16.906** | 1106.112 ± 12.656 | 807.939 ± 6.233 | 1967.996 ± 33.825 | 1288.825 ± 35.444 |
| tensornetwork_permutation_focus_step409_316 | **239.587 ± 1.436** | 375.186 ± 3.008 | 560.455 ± 2.835 | 744.966 ± 15.072 | 486.466 ± 19.786 |
| tensornetwork_permutation_light_415 | **250.078 ± 1.299** | 286.136 ± 1.474 | 526.531 ± 6.579 | 763.457 ± 10.321 | 490.158 ± 16.531 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.290 ± 0.043 | **1.611 ± 0.019** | 1.636 ± 0.017 | 4.819 ± 0.082 | 8.074 ± 4.977 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.343 ± 0.025 | **0.222 ± 0.003** | 0.250 ± 0.023 | 0.948 ± 0.036 | 0.683 ± 0.270 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 1.158 ± 0.013 | 1.161 ± 0.013 | **0.373 ± 0.025** | 0.623 ± 0.053 | 1.055 ± 2.846 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.089 ± 0.019 | 1.187 ± 0.019 | **0.370 ± 0.013** | 0.625 ± 0.042 | 3.706 ± 2.289 |
| bin_elementwise_mul_2048x2048 | **1.960 ± 0.254** | 3.068 ± 0.805 | 15.545 ± 0.230 | 18.811 ± 0.180 | 16.869 ± 3.326 |
| bin_elementwise_mul_512x512 | 0.232 ± 0.015 | **0.227 ± 0.005** | 0.251 ± 0.021 | 0.521 ± 0.017 | 0.317 ± 0.541 |
| bin_matmul_1024 | 30.469 ± 0.961 | 26.794 ± 0.263 | **24.804 ± 0.205** | 67.813 ± 0.626 | 50.550 ± 7.345 |
| bin_matmul_256 | 0.462 ± 0.033 | **0.377 ± 0.009** | 0.443 ± 0.021 | 1.169 ± 0.025 | 0.867 ± 0.214 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.006 ± 0.000** | 0.025 ± 0.010 | 0.152 ± 0.015 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.027 ± 0.001 | **0.026 ± 0.001** | 0.094 ± 0.008 | 0.205 ± 0.029 | 0.070 ± 0.030 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.000 | **0.014 ± 0.007** | 0.019 ± 0.002 | 0.101 ± 0.012 | 0.020 ± 0.008 |
| bin_outer_product_4096 | 59.132 ± 0.596 | 58.579 ± 0.684 | **51.043 ± 0.804** | 54.279 ± 0.701 | 329.265 ± 34.513 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 29.632 ± 0.359 | **26.602 ± 0.201** | 37.158 ± 0.317 | 75.014 ± 1.297 | 45.427 ± 7.512 |
| gm_queen5_5_3.wcsp | **1694.818 ± 17.067** | 1839.533 ± 20.836 | 2315.913 ± 21.737 | 2223.839 ± 12.302 | 2748.803 ± 42.263 |
| lm_batch_likelihood_brackets_4_4d | 29.707 ± 1.781 | 39.839 ± 0.772 | 24.820 ± 0.640 | **18.627 ± 0.415** | 33.183 ± 14.915 |
| lm_batch_likelihood_sentence_3_12d | 50.115 ± 1.707 | 120.540 ± 1.206 | **42.268 ± 1.550** | 128.250 ± 1.471 | 104.446 ± 43.381 |
| lm_batch_likelihood_sentence_4_4d | 20.921 ± 1.719 | 46.336 ± 1.925 | 25.830 ± 0.478 | **20.584 ± 0.337** | 33.387 ± 15.039 |
| nary_matmul_chain_64 | 0.032 ± 0.001 | **0.026 ± 0.007** | 0.050 ± 0.020 | 0.155 ± 0.010 | 0.052 ± 0.030 |
| str_matrix_chain_multiplication_100 | 9.603 ± 0.183 | 9.754 ± 0.151 | **8.619 ± 0.279** | 16.835 ± 0.252 | 13.979 ± 0.668 |
| str_mps_varying_inner_product_200 | 15.673 ± 0.400 | 14.331 ± 0.234 | **11.122 ± 0.177** | 26.063 ± 1.505 | 19.852 ± 13.036 |
| str_nw_mera_closed_120 | 826.284 ± 7.842 | **769.533 ± 5.046** | 925.042 ± 5.803 | 2085.501 ± 21.575 | 1131.289 ± 52.237 |
| str_nw_mera_open_26 | **743.308 ± 4.918** | 1120.979 ± 19.507 | 815.279 ± 4.797 | 1979.870 ± 11.804 | 1259.350 ± 46.804 |
| tensornetwork_permutation_focus_step409_316 | **239.587 ± 1.436** | 375.186 ± 3.008 | 560.455 ± 2.835 | 744.966 ± 15.072 | 486.466 ± 19.786 |
| tensornetwork_permutation_light_415 | **250.078 ± 1.299** | 286.136 ± 1.474 | 526.531 ± 6.579 | 763.457 ± 10.321 | 490.158 ± 16.531 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261127_183055/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261127_183055/run.yaml`
- Timestamp: `2026-11-27T18:30:55.648913+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261127_183055`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

### CPU Information

- Model: `INTEL(R) XEON(R) PLATINUM 8573C`
- Vendor: `GenuineIntel`
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

- `data/results/amd-cpu/cpu/einsum/20261127_183055/jax_cpu_t4_20261127_183055.log`
- `data/results/amd-cpu/cpu/einsum/20261127_183055/julia_omeinsum_t4_20261127_183055.log`
- `data/results/amd-cpu/cpu/einsum/20261127_183055/pytorch_cpu_t4_20261127_183055.log`
- `data/results/amd-cpu/cpu/einsum/20261127_183055/tenferro_eager_t4_20261127_183055.log`
- `data/results/amd-cpu/cpu/einsum/20261127_183055/tenferro_trace_t4_20261127_183055.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.859 ± 0.015 | 1.386 ± 0.021 | **0.847 ± 0.024** | 3.384 ± 0.640 | 13.706 ± 7.448 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.440 ± 0.030 | 0.325 ± 0.020 | **0.134 ± 0.116** | 0.944 ± 0.057 | 0.678 ± 1.054 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.422 ± 0.016 | 0.414 ± 0.012 | **0.202 ± 0.014** | 0.474 ± 0.026 | 3.330 ± 1.340 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.335 ± 0.015 | 0.418 ± 0.012 | **0.213 ± 0.020** | 0.494 ± 0.049 | 5.536 ± 5.288 |
| bin_elementwise_mul_2048x2048 | 0.716 ± 0.081 | **0.675 ± 0.011** | 5.819 ± 0.172 | 10.401 ± 0.227 | 16.311 ± 0.424 |
| bin_elementwise_mul_512x512 | 0.152 ± 0.015 | **0.129 ± 0.003** | 0.146 ± 0.026 | 0.407 ± 0.021 | 0.723 ± 0.538 |
| bin_matmul_1024 | 16.969 ± 1.718 | 15.108 ± 0.149 | **12.247 ± 0.042** | 33.513 ± 1.864 | 33.039 ± 6.353 |
| bin_matmul_256 | 0.326 ± 0.013 | **0.237 ± 0.010** | **0.237 ± 0.015** | 0.838 ± 0.101 | 0.695 ± 0.047 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.011 ± 0.000 | **0.007 ± 0.000** | 0.030 ± 0.001 | 0.156 ± 0.022 | 0.016 ± 0.005 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.034 ± 0.001** | 0.035 ± 0.001 | 0.137 ± 0.007 | 0.218 ± 0.033 | 0.050 ± 0.029 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.001 | **0.020 ± 0.001** | 0.026 ± 0.000 | 0.100 ± 0.011 | **0.020 ± 0.008** |
| bin_outer_product_4096 | 21.585 ± 0.355 | 21.765 ± 0.169 | **17.859 ± 0.205** | 28.200 ± 0.301 | 314.113 ± 29.706 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 14.179 ± 0.073 | **13.829 ± 0.072** | 16.175 ± 0.222 | 39.072 ± 2.882 | 36.417 ± 7.529 |
| gm_queen5_5_3.wcsp | 3603.282 ± 30.837 | 3604.362 ± 28.052 | **2168.166 ± 45.329** | 3555.154 ± 72.574 | 7125.587 ± 167.358 |
| lm_batch_likelihood_brackets_4_4d | 12.658 ± 0.210 | 20.354 ± 0.926 | **11.985 ± 0.117** | 25.431 ± 2.204 | 36.604 ± 22.630 |
| lm_batch_likelihood_sentence_3_12d | 19.229 ± 0.329 | 76.104 ± 0.433 | **16.017 ± 0.207** | 60.971 ± 10.979 | 70.271 ± 15.000 |
| lm_batch_likelihood_sentence_4_4d | 13.139 ± 0.408 | 20.593 ± 0.581 | **12.745 ± 0.158** | 26.631 ± 1.271 | 35.247 ± 7.031 |
| nary_matmul_chain_64 | 0.059 ± 0.004 | 0.049 ± 0.004 | **0.044 ± 0.019** | 0.159 ± 0.012 | 0.067 ± 0.017 |
| str_matrix_chain_multiplication_100 | 5.636 ± 0.222 | 6.882 ± 0.431 | **5.060 ± 0.045** | 15.239 ± 0.575 | 14.031 ± 0.289 |
| str_mps_varying_inner_product_200 | 11.797 ± 0.451 | 12.277 ± 0.330 | **11.540 ± 0.409** | 30.389 ± 0.982 | 20.975 ± 2.227 |
| str_nw_mera_closed_120 | **471.038 ± 5.168** | 528.445 ± 10.301 | 635.165 ± 19.262 | 1096.872 ± 14.390 | 949.243 ± 46.495 |
| str_nw_mera_open_26 | 427.181 ± 3.702 | 566.813 ± 13.561 | **388.984 ± 9.726** | 896.804 ± 12.755 | 971.663 ± 67.733 |
| tensornetwork_permutation_focus_step409_316 | **143.430 ± 1.295** | 195.157 ± 2.420 | 244.692 ± 8.056 | 403.063 ± 13.172 | 462.300 ± 18.345 |
| tensornetwork_permutation_light_415 | 167.823 ± 1.251 | **125.101 ± 0.991** | 236.923 ± 5.440 | 413.734 ± 13.728 | 473.540 ± 18.111 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.859 ± 0.015 | 1.386 ± 0.021 | **0.847 ± 0.024** | 3.384 ± 0.640 | 13.706 ± 7.448 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.440 ± 0.030 | 0.325 ± 0.020 | **0.134 ± 0.116** | 0.944 ± 0.057 | 0.678 ± 1.054 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.422 ± 0.016 | 0.414 ± 0.012 | **0.202 ± 0.014** | 0.474 ± 0.026 | 3.330 ± 1.340 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.335 ± 0.015 | 0.418 ± 0.012 | **0.213 ± 0.020** | 0.494 ± 0.049 | 5.536 ± 5.288 |
| bin_elementwise_mul_2048x2048 | 0.716 ± 0.081 | **0.675 ± 0.011** | 5.819 ± 0.172 | 10.401 ± 0.227 | 16.311 ± 0.424 |
| bin_elementwise_mul_512x512 | 0.152 ± 0.015 | **0.129 ± 0.003** | 0.146 ± 0.026 | 0.407 ± 0.021 | 0.723 ± 0.538 |
| bin_matmul_1024 | 16.969 ± 1.718 | 15.108 ± 0.149 | **12.247 ± 0.042** | 33.513 ± 1.864 | 33.039 ± 6.353 |
| bin_matmul_256 | 0.326 ± 0.013 | **0.237 ± 0.010** | **0.237 ± 0.015** | 0.838 ± 0.101 | 0.695 ± 0.047 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.011 ± 0.000 | **0.007 ± 0.000** | 0.030 ± 0.001 | 0.156 ± 0.022 | 0.016 ± 0.005 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.034 ± 0.001** | 0.035 ± 0.001 | 0.137 ± 0.007 | 0.218 ± 0.033 | 0.050 ± 0.029 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.001 | **0.020 ± 0.001** | 0.026 ± 0.000 | 0.100 ± 0.011 | **0.020 ± 0.008** |
| bin_outer_product_4096 | 21.585 ± 0.355 | 21.765 ± 0.169 | **17.859 ± 0.205** | 28.200 ± 0.301 | 314.113 ± 29.706 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 14.179 ± 0.073 | **13.829 ± 0.072** | 16.175 ± 0.222 | 39.072 ± 2.882 | 36.417 ± 7.529 |
| gm_queen5_5_3.wcsp | **827.171 ± 5.110** | 914.399 ± 4.310 | 927.053 ± 12.029 | 1467.416 ± 15.188 | 2598.040 ± 52.254 |
| lm_batch_likelihood_brackets_4_4d | 14.262 ± 0.497 | 20.691 ± 0.771 | **12.010 ± 0.212** | 24.584 ± 0.554 | 39.702 ± 4.426 |
| lm_batch_likelihood_sentence_3_12d | 20.878 ± 0.648 | 77.531 ± 0.518 | **17.455 ± 0.327** | 52.071 ± 2.734 | 72.205 ± 47.640 |
| lm_batch_likelihood_sentence_4_4d | **11.792 ± 0.229** | 22.766 ± 0.521 | 12.150 ± 0.256 | 26.189 ± 1.405 | 36.623 ± 11.074 |
| nary_matmul_chain_64 | 0.059 ± 0.004 | 0.049 ± 0.004 | **0.044 ± 0.019** | 0.159 ± 0.012 | 0.067 ± 0.017 |
| str_matrix_chain_multiplication_100 | 5.943 ± 0.101 | 7.249 ± 0.396 | **5.132 ± 0.203** | 15.387 ± 1.102 | 14.428 ± 0.269 |
| str_mps_varying_inner_product_200 | 16.563 ± 0.173 | 15.862 ± 0.204 | **14.369 ± 0.296** | 30.649 ± 1.042 | 20.199 ± 2.103 |
| str_nw_mera_closed_120 | 498.824 ± 4.624 | **470.066 ± 7.614** | 502.481 ± 12.875 | 950.060 ± 12.857 | 726.379 ± 39.397 |
| str_nw_mera_open_26 | 437.806 ± 7.377 | 577.508 ± 14.397 | **386.082 ± 5.631** | 910.757 ± 11.288 | 956.932 ± 37.590 |
| tensornetwork_permutation_focus_step409_316 | **143.430 ± 1.295** | 195.157 ± 2.420 | 244.692 ± 8.056 | 403.063 ± 13.172 | 462.300 ± 18.345 |
| tensornetwork_permutation_light_415 | 167.823 ± 1.251 | **125.101 ± 0.991** | 236.923 ± 5.440 | 413.734 ± 13.728 | 473.540 ± 18.111 |

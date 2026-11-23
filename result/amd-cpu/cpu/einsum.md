# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Benchmark commit: `d9f6406557a0d1066f37785ec85f9b9afdbb4781`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261123_175252/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261123_175252/run.yaml`
- Timestamp: `2026-11-23T17:52:52.587828+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261123_175252`.

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

- `data/results/amd-cpu/cpu/einsum/20261123_175252/jax_cpu_t1_20261123_175252.log`
- `data/results/amd-cpu/cpu/einsum/20261123_175252/julia_omeinsum_t1_20261123_175252.log`
- `data/results/amd-cpu/cpu/einsum/20261123_175252/pytorch_cpu_t1_20261123_175252.log`
- `data/results/amd-cpu/cpu/einsum/20261123_175252/tenferro_eager_t1_20261123_175252.log`
- `data/results/amd-cpu/cpu/einsum/20261123_175252/tenferro_trace_t1_20261123_175252.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.921 ± 0.077 | **3.214 ± 0.025** | 4.614 ± 0.139 | 4.143 ± 0.042 | 11.969 ± 9.246 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.599 ± 0.021 | **0.438 ± 0.015** | 0.697 ± 0.018 | 0.800 ± 0.023 | 1.029 ± 1.693 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.662 ± 0.013 | 0.656 ± 0.022 | **0.153 ± 0.009** | 0.637 ± 0.022 | 4.029 ± 2.308 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.240 ± 0.012 | 0.683 ± 0.011 | **0.160 ± 0.005** | 0.629 ± 0.025 | 6.076 ± 7.655 |
| bin_elementwise_mul_2048x2048 | 2.473 ± 0.168 | **2.273 ± 0.132** | 17.555 ± 0.489 | 19.314 ± 0.573 | 19.937 ± 4.484 |
| bin_elementwise_mul_512x512 | 0.094 ± 0.001 | **0.089 ± 0.002** | 0.104 ± 0.005 | 0.347 ± 0.020 | 0.131 ± 0.922 |
| bin_matmul_1024 | 135.341 ± 0.668 | **50.838 ± 0.154** | 66.388 ± 0.315 | 59.945 ± 0.972 | 76.207 ± 7.974 |
| bin_matmul_256 | 0.994 ± 0.015 | **0.846 ± 0.006** | 1.586 ± 0.015 | 1.076 ± 0.022 | 1.454 ± 0.025 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.011 ± 0.000 | **0.006 ± 0.000** | 0.026 ± 0.002 | 0.173 ± 0.023 | 0.012 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.032 ± 0.001 | **0.030 ± 0.001** | 0.122 ± 0.007 | 0.219 ± 0.032 | 0.097 ± 0.057 |
| bin_omeinsum_matmul_10x10 | 0.017 ± 0.001 | **0.013 ± 0.000** | 0.039 ± 0.017 | 0.101 ± 0.020 | 0.024 ± 0.002 |
| bin_outer_product_4096 | 69.581 ± 0.705 | 69.360 ± 0.992 | **58.298 ± 1.416** | 60.659 ± 2.304 | 343.514 ± 47.835 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 57.422 ± 0.811 | **53.774 ± 0.622** | 68.453 ± 0.743 | 62.376 ± 0.584 | 72.474 ± 8.449 |
| gm_queen5_5_3.wcsp | 6460.961 ± 74.934 | **6350.592 ± 81.207** | 6983.672 ± 61.495 | 9562.303 ± 98.758 | 10107.008 ± 110.893 |
| lm_batch_likelihood_brackets_4_4d | 25.928 ± 0.192 | 32.345 ± 0.568 | 32.197 ± 0.331 | **14.848 ± 0.699** | 31.089 ± 19.887 |
| lm_batch_likelihood_sentence_3_12d | **59.222 ± 0.890** | 135.626 ± 1.194 | 115.785 ± 1.421 | 101.576 ± 2.188 | 84.725 ± 6.264 |
| lm_batch_likelihood_sentence_4_4d | 27.450 ± 0.749 | 35.051 ± 0.547 | 35.228 ± 0.710 | **16.835 ± 0.955** | 41.040 ± 13.162 |
| nary_matmul_chain_64 | 0.045 ± 0.006 | **0.035 ± 0.002** | 0.064 ± 0.005 | 0.153 ± 0.017 | 0.078 ± 0.005 |
| str_matrix_chain_multiplication_100 | **13.991 ± 0.180** | 16.190 ± 0.199 | 18.532 ± 0.110 | 16.149 ± 0.201 | 20.110 ± 6.378 |
| str_mps_varying_inner_product_200 | 17.983 ± 0.224 | **17.629 ± 0.430** | 25.951 ± 0.266 | 25.859 ± 1.018 | 29.393 ± 6.184 |
| str_nw_mera_closed_120 | **1488.052 ± 6.594** | 1545.515 ± 10.420 | 2128.267 ± 14.523 | 2378.841 ± 24.820 | 2092.213 ± 34.340 |
| str_nw_mera_open_26 | **1175.629 ± 6.030** | 1664.151 ± 17.115 | 1340.686 ± 10.010 | 1933.058 ± 32.722 | 1843.742 ± 29.207 |
| tensornetwork_permutation_focus_step409_316 | **335.163 ± 6.294** | 461.501 ± 4.300 | 949.570 ± 4.362 | 949.527 ± 381.542 | 653.171 ± 111.218 |
| tensornetwork_permutation_light_415 | **332.023 ± 1.498** | 380.376 ± 3.953 | 911.982 ± 25.050 | 775.499 ± 79.513 | 659.965 ± 13.474 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.921 ± 0.077 | **3.214 ± 0.025** | 4.614 ± 0.139 | 4.143 ± 0.042 | 11.969 ± 9.246 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.599 ± 0.021 | **0.438 ± 0.015** | 0.697 ± 0.018 | 0.800 ± 0.023 | 1.029 ± 1.693 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.662 ± 0.013 | 0.656 ± 0.022 | **0.153 ± 0.009** | 0.637 ± 0.022 | 4.029 ± 2.308 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.240 ± 0.012 | 0.683 ± 0.011 | **0.160 ± 0.005** | 0.629 ± 0.025 | 6.076 ± 7.655 |
| bin_elementwise_mul_2048x2048 | 2.473 ± 0.168 | **2.273 ± 0.132** | 17.555 ± 0.489 | 19.314 ± 0.573 | 19.937 ± 4.484 |
| bin_elementwise_mul_512x512 | 0.094 ± 0.001 | **0.089 ± 0.002** | 0.104 ± 0.005 | 0.347 ± 0.020 | 0.131 ± 0.922 |
| bin_matmul_1024 | 135.341 ± 0.668 | **50.838 ± 0.154** | 66.388 ± 0.315 | 59.945 ± 0.972 | 76.207 ± 7.974 |
| bin_matmul_256 | 0.994 ± 0.015 | **0.846 ± 0.006** | 1.586 ± 0.015 | 1.076 ± 0.022 | 1.454 ± 0.025 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.011 ± 0.000 | **0.006 ± 0.000** | 0.026 ± 0.002 | 0.173 ± 0.023 | 0.012 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.032 ± 0.001 | **0.030 ± 0.001** | 0.122 ± 0.007 | 0.219 ± 0.032 | 0.097 ± 0.057 |
| bin_omeinsum_matmul_10x10 | 0.017 ± 0.001 | **0.013 ± 0.000** | 0.039 ± 0.017 | 0.101 ± 0.020 | 0.024 ± 0.002 |
| bin_outer_product_4096 | 69.581 ± 0.705 | 69.360 ± 0.992 | **58.298 ± 1.416** | 60.659 ± 2.304 | 343.514 ± 47.835 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 57.422 ± 0.811 | **53.774 ± 0.622** | 68.453 ± 0.743 | 62.376 ± 0.584 | 72.474 ± 8.449 |
| gm_queen5_5_3.wcsp | **2022.626 ± 20.825** | 2445.594 ± 17.261 | 2925.889 ± 37.738 | 3122.245 ± 58.092 | 3765.439 ± 97.728 |
| lm_batch_likelihood_brackets_4_4d | 26.583 ± 0.292 | 30.735 ± 0.589 | 35.414 ± 0.381 | **15.014 ± 0.441** | 37.012 ± 54.061 |
| lm_batch_likelihood_sentence_3_12d | **61.603 ± 0.760** | 117.916 ± 0.828 | 71.793 ± 0.986 | 100.635 ± 1.808 | 127.677 ± 41.912 |
| lm_batch_likelihood_sentence_4_4d | 26.031 ± 0.709 | 33.608 ± 0.297 | 38.146 ± 1.420 | **16.336 ± 2.998** | 37.085 ± 12.638 |
| nary_matmul_chain_64 | 0.045 ± 0.006 | **0.035 ± 0.002** | 0.064 ± 0.005 | 0.153 ± 0.017 | 0.078 ± 0.005 |
| str_matrix_chain_multiplication_100 | 16.223 ± 0.221 | **16.145 ± 0.082** | 18.688 ± 0.247 | 16.526 ± 0.337 | 20.092 ± 1.247 |
| str_mps_varying_inner_product_200 | 19.256 ± 0.148 | **17.224 ± 0.147** | 24.895 ± 0.223 | 30.314 ± 1.307 | 29.217 ± 2.242 |
| str_nw_mera_closed_120 | 1465.915 ± 4.841 | **1407.225 ± 12.763** | 1714.426 ± 12.506 | 1946.631 ± 25.863 | 1735.200 ± 40.643 |
| str_nw_mera_open_26 | **1185.709 ± 8.752** | 1686.799 ± 14.186 | 1397.148 ± 14.075 | 2016.802 ± 34.803 | 1843.774 ± 45.397 |
| tensornetwork_permutation_focus_step409_316 | **335.163 ± 6.294** | 461.501 ± 4.300 | 949.570 ± 4.362 | 949.527 ± 381.542 | 653.171 ± 111.218 |
| tensornetwork_permutation_light_415 | **332.023 ± 1.498** | 380.376 ± 3.953 | 911.982 ± 25.050 | 775.499 ± 79.513 | 659.965 ± 13.474 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261123_183450/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261123_183450/run.yaml`
- Timestamp: `2026-11-23T18:34:50.691833+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261123_183450`.

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

- `data/results/amd-cpu/cpu/einsum/20261123_183450/jax_cpu_t4_20261123_183450.log`
- `data/results/amd-cpu/cpu/einsum/20261123_183450/julia_omeinsum_t4_20261123_183450.log`
- `data/results/amd-cpu/cpu/einsum/20261123_183450/pytorch_cpu_t4_20261123_183450.log`
- `data/results/amd-cpu/cpu/einsum/20261123_183450/tenferro_eager_t4_20261123_183450.log`
- `data/results/amd-cpu/cpu/einsum/20261123_183450/tenferro_trace_t4_20261123_183450.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.791 ± 0.048 | **1.997 ± 0.017** | 2.239 ± 0.404 | 2.791 ± 0.804 | 12.729 ± 5.185 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.674 ± 0.019 | 0.479 ± 0.015 | **0.366 ± 0.026** | 0.800 ± 0.015 | 1.003 ± 0.976 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.312 ± 0.026 | 0.309 ± 0.016 | **0.102 ± 0.003** | 0.464 ± 0.022 | 3.986 ± 2.396 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.324 ± 0.016 | 0.312 ± 0.015 | **0.103 ± 0.030** | 0.462 ± 0.039 | 6.084 ± 4.879 |
| bin_elementwise_mul_2048x2048 | 0.998 ± 0.060 | **0.965 ± 0.028** | 6.793 ± 0.292 | 10.496 ± 0.394 | 20.662 ± 0.480 |
| bin_elementwise_mul_512x512 | 0.063 ± 0.003 | **0.058 ± 0.004** | 0.076 ± 0.005 | 0.301 ± 0.022 | 0.131 ± 0.037 |
| bin_matmul_1024 | 57.361 ± 1.149 | **24.960 ± 0.124** | 39.396 ± 0.464 | 33.472 ± 0.430 | 49.701 ± 10.327 |
| bin_matmul_256 | 0.720 ± 0.017 | **0.497 ± 0.022** | 1.077 ± 0.007 | 0.812 ± 0.153 | 1.562 ± 0.975 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.001 | **0.010 ± 0.000** | 0.036 ± 0.003 | 0.168 ± 0.026 | 0.013 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.045 ± 0.003 | **0.042 ± 0.002** | 0.178 ± 0.016 | 0.340 ± 0.089 | 0.082 ± 0.045 |
| bin_omeinsum_matmul_10x10 | 0.025 ± 0.001 | **0.014 ± 0.000** | 0.038 ± 0.003 | 0.145 ± 0.024 | 0.026 ± 0.001 |
| bin_outer_product_4096 | 22.622 ± 0.848 | 22.368 ± 0.728 | **20.429 ± 1.015** | 31.355 ± 1.056 | 337.233 ± 36.803 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 27.875 ± 0.162 | **26.491 ± 0.215** | 33.688 ± 1.185 | 37.372 ± 5.047 | 43.539 ± 9.411 |
| gm_queen5_5_3.wcsp | 3701.081 ± 56.540 | 4167.664 ± 87.636 | **2967.740 ± 25.631** | 6167.128 ± 89.466 | 9452.310 ± 163.831 |
| lm_batch_likelihood_brackets_4_4d | 20.430 ± 0.531 | 23.964 ± 0.459 | **16.221 ± 0.517** | 23.897 ± 6.804 | 37.979 ± 19.187 |
| lm_batch_likelihood_sentence_3_12d | **34.113 ± 1.691** | 66.050 ± 1.561 | 35.262 ± 0.626 | 57.486 ± 4.061 | 74.132 ± 11.502 |
| lm_batch_likelihood_sentence_4_4d | 21.165 ± 1.246 | 24.696 ± 0.777 | **17.196 ± 0.224** | 25.936 ± 6.396 | 43.761 ± 8.040 |
| nary_matmul_chain_64 | 0.051 ± 0.001 | **0.042 ± 0.000** | 0.077 ± 0.004 | 0.156 ± 0.028 | 0.144 ± 0.038 |
| str_matrix_chain_multiplication_100 | **8.028 ± 0.373** | 8.740 ± 0.114 | 11.367 ± 0.083 | 15.975 ± 0.837 | 16.977 ± 1.493 |
| str_mps_varying_inner_product_200 | 28.648 ± 0.714 | 28.254 ± 0.857 | **26.429 ± 0.305** | 32.658 ± 1.237 | 29.473 ± 4.412 |
| str_nw_mera_closed_120 | **763.518 ± 11.255** | 827.388 ± 10.216 | 1056.886 ± 11.180 | 1170.882 ± 24.788 | 1290.390 ± 59.652 |
| str_nw_mera_open_26 | 686.489 ± 20.395 | 828.210 ± 22.864 | **656.479 ± 18.873** | 966.477 ± 24.571 | 1370.514 ± 34.628 |
| tensornetwork_permutation_focus_step409_316 | **206.591 ± 1.928** | 219.565 ± 14.085 | 372.072 ± 23.272 | 418.248 ± 9.648 | 574.353 ± 36.439 |
| tensornetwork_permutation_light_415 | 206.967 ± 1.139 | **174.871 ± 8.062** | 365.581 ± 9.268 | 428.941 ± 21.157 | 571.188 ± 21.374 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.791 ± 0.048 | **1.997 ± 0.017** | 2.239 ± 0.404 | 2.791 ± 0.804 | 12.729 ± 5.185 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.674 ± 0.019 | 0.479 ± 0.015 | **0.366 ± 0.026** | 0.800 ± 0.015 | 1.003 ± 0.976 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.312 ± 0.026 | 0.309 ± 0.016 | **0.102 ± 0.003** | 0.464 ± 0.022 | 3.986 ± 2.396 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.324 ± 0.016 | 0.312 ± 0.015 | **0.103 ± 0.030** | 0.462 ± 0.039 | 6.084 ± 4.879 |
| bin_elementwise_mul_2048x2048 | 0.998 ± 0.060 | **0.965 ± 0.028** | 6.793 ± 0.292 | 10.496 ± 0.394 | 20.662 ± 0.480 |
| bin_elementwise_mul_512x512 | 0.063 ± 0.003 | **0.058 ± 0.004** | 0.076 ± 0.005 | 0.301 ± 0.022 | 0.131 ± 0.037 |
| bin_matmul_1024 | 57.361 ± 1.149 | **24.960 ± 0.124** | 39.396 ± 0.464 | 33.472 ± 0.430 | 49.701 ± 10.327 |
| bin_matmul_256 | 0.720 ± 0.017 | **0.497 ± 0.022** | 1.077 ± 0.007 | 0.812 ± 0.153 | 1.562 ± 0.975 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.001 | **0.010 ± 0.000** | 0.036 ± 0.003 | 0.168 ± 0.026 | 0.013 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.045 ± 0.003 | **0.042 ± 0.002** | 0.178 ± 0.016 | 0.340 ± 0.089 | 0.082 ± 0.045 |
| bin_omeinsum_matmul_10x10 | 0.025 ± 0.001 | **0.014 ± 0.000** | 0.038 ± 0.003 | 0.145 ± 0.024 | 0.026 ± 0.001 |
| bin_outer_product_4096 | 22.622 ± 0.848 | 22.368 ± 0.728 | **20.429 ± 1.015** | 31.355 ± 1.056 | 337.233 ± 36.803 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 27.875 ± 0.162 | **26.491 ± 0.215** | 33.688 ± 1.185 | 37.372 ± 5.047 | 43.539 ± 9.411 |
| gm_queen5_5_3.wcsp | **1034.342 ± 21.746** | 1186.019 ± 10.024 | 1297.668 ± 20.171 | 2147.936 ± 46.650 | 3377.240 ± 46.845 |
| lm_batch_likelihood_brackets_4_4d | 19.930 ± 0.136 | 22.757 ± 0.129 | **17.356 ± 0.266** | 23.619 ± 3.063 | 45.648 ± 3.068 |
| lm_batch_likelihood_sentence_3_12d | **34.327 ± 0.297** | 65.630 ± 1.111 | 35.549 ± 0.450 | 60.486 ± 13.285 | 105.548 ± 59.929 |
| lm_batch_likelihood_sentence_4_4d | 18.634 ± 0.203 | 24.160 ± 0.352 | **18.604 ± 0.334** | 23.053 ± 1.482 | 48.989 ± 16.629 |
| nary_matmul_chain_64 | 0.051 ± 0.001 | **0.042 ± 0.000** | 0.077 ± 0.004 | 0.156 ± 0.028 | 0.144 ± 0.038 |
| str_matrix_chain_multiplication_100 | **8.597 ± 0.224** | 9.158 ± 0.330 | 11.566 ± 0.126 | 16.189 ± 0.698 | 16.687 ± 1.662 |
| str_mps_varying_inner_product_200 | 32.368 ± 1.018 | 31.831 ± 6.199 | **30.420 ± 0.628** | 34.746 ± 1.858 | 33.840 ± 7.279 |
| str_nw_mera_closed_120 | 803.083 ± 17.886 | **748.665 ± 25.413** | 846.194 ± 12.094 | 953.403 ± 23.007 | 1056.806 ± 48.302 |
| str_nw_mera_open_26 | 687.019 ± 12.830 | 847.647 ± 22.941 | **648.704 ± 14.919** | 988.464 ± 16.374 | 1339.429 ± 45.672 |
| tensornetwork_permutation_focus_step409_316 | **206.591 ± 1.928** | 219.565 ± 14.085 | 372.072 ± 23.272 | 418.248 ± 9.648 | 574.353 ± 36.439 |
| tensornetwork_permutation_light_415 | 206.967 ± 1.139 | **174.871 ± 8.062** | 365.581 ± 9.268 | 428.941 ± 21.157 | 571.188 ± 21.374 |

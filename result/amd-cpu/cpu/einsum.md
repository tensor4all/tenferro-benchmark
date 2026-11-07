# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `d6655b567e09aa29c3201655e1de418716934876`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261107_180711/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261107_180711/run.yaml`
- Timestamp: `2026-11-07T18:07:11.532995+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261107_180711`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
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

- `data/results/amd-cpu/cpu/einsum/20261107_180711/jax_cpu_t1_20261107_180711.log`
- `data/results/amd-cpu/cpu/einsum/20261107_180711/julia_omeinsum_t1_20261107_180711.log`
- `data/results/amd-cpu/cpu/einsum/20261107_180711/pytorch_cpu_t1_20261107_180711.log`
- `data/results/amd-cpu/cpu/einsum/20261107_180711/tenferro_eager_t1_20261107_180711.log`
- `data/results/amd-cpu/cpu/einsum/20261107_180711/tenferro_trace_t1_20261107_180711.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.901 ± 0.046 | **3.857 ± 0.019** | 4.100 ± 0.058 | 4.223 ± 0.063 | 12.735 ± 3.403 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.769 ± 0.019 | **0.594 ± 0.010** | 0.664 ± 0.021 | 0.837 ± 0.016 | 0.944 ± 0.254 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.736 ± 0.014 | 0.731 ± 0.008 | **0.206 ± 0.012** | 0.612 ± 0.025 | 0.695 ± 2.305 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.248 ± 0.008 | 0.733 ± 0.012 | **0.202 ± 0.023** | 0.610 ± 0.025 | 7.739 ± 5.519 |
| bin_elementwise_mul_2048x2048 | 1.833 ± 0.082 | **1.794 ± 0.116** | 15.814 ± 0.322 | 17.044 ± 0.410 | 17.970 ± 1.259 |
| bin_elementwise_mul_512x512 | 0.132 ± 0.022 | **0.122 ± 0.008** | 0.147 ± 0.006 | 0.386 ± 0.022 | 0.166 ± 0.037 |
| bin_matmul_1024 | 55.942 ± 0.120 | **48.853 ± 0.204** | 60.354 ± 0.445 | 62.414 ± 0.442 | 76.576 ± 22.689 |
| bin_matmul_256 | 1.062 ± 0.011 | **0.824 ± 0.006** | 1.369 ± 0.018 | 1.183 ± 0.030 | 1.268 ± 0.829 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.016 ± 0.007 | **0.010 ± 0.001** | 0.037 ± 0.005 | 0.203 ± 0.016 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.042 ± 0.003 | **0.039 ± 0.028** | 0.151 ± 0.016 | 0.231 ± 0.019 | 0.049 ± 0.001 |
| bin_omeinsum_matmul_10x10 | 0.022 ± 0.001 | **0.011 ± 0.007** | 0.036 ± 0.007 | 0.148 ± 0.020 | 0.026 ± 0.001 |
| bin_outer_product_4096 | 63.861 ± 0.195 | 63.646 ± 0.608 | **53.411 ± 0.527** | 55.735 ± 0.785 | 306.887 ± 3.900 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 55.880 ± 0.260 | **50.500 ± 0.385** | 61.209 ± 0.116 | 65.682 ± 4.140 | 73.246 ± 14.773 |
| gm_queen5_5_3.wcsp | 6845.372 ± 110.822 | 6624.438 ± 24.411 | **6119.655 ± 35.191** | 7937.122 ± 87.367 | 9103.056 ± 290.786 |
| lm_batch_likelihood_brackets_4_4d | 26.870 ± 0.324 | 33.357 ± 1.028 | 34.337 ± 0.217 | **15.057 ± 0.243** | 30.290 ± 8.242 |
| lm_batch_likelihood_sentence_3_12d | **57.064 ± 0.228** | 113.471 ± 26.061 | 104.393 ± 0.347 | 91.472 ± 2.403 | 107.226 ± 11.989 |
| lm_batch_likelihood_sentence_4_4d | 28.874 ± 0.231 | 37.988 ± 1.986 | 35.916 ± 0.243 | **16.611 ± 0.235** | 64.903 ± 24.501 |
| nary_matmul_chain_64 | 0.066 ± 0.005 | **0.045 ± 0.020** | 0.080 ± 0.006 | 0.195 ± 0.022 | 0.125 ± 0.046 |
| str_matrix_chain_multiplication_100 | **14.020 ± 0.151** | 14.099 ± 0.111 | 15.663 ± 0.096 | 14.490 ± 0.221 | 16.431 ± 0.603 |
| str_mps_varying_inner_product_200 | **19.540 ± 0.164** | 20.969 ± 0.130 | 25.440 ± 0.156 | 23.085 ± 0.303 | 26.046 ± 5.995 |
| str_nw_mera_closed_120 | **1624.701 ± 6.977** | 1820.879 ± 28.114 | 1921.480 ± 3.249 | 2467.637 ± 4.716 | 1806.582 ± 21.004 |
| str_nw_mera_open_26 | 1289.226 ± 11.740 | 1460.050 ± 3.302 | **1210.937 ± 2.478** | 1956.311 ± 4.896 | 1561.320 ± 48.682 |
| tensornetwork_permutation_focus_step409_316 | **467.979 ± 1.913** | 531.130 ± 1.991 | 901.953 ± 1.554 | 839.942 ± 146.748 | 544.154 ± 79.989 |
| tensornetwork_permutation_light_415 | **464.874 ± 3.430** | 518.295 ± 2.111 | 867.982 ± 1.700 | 742.702 ± 35.595 | 590.757 ± 35.462 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.901 ± 0.046 | **3.857 ± 0.019** | 4.100 ± 0.058 | 4.223 ± 0.063 | 12.735 ± 3.403 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.769 ± 0.019 | **0.594 ± 0.010** | 0.664 ± 0.021 | 0.837 ± 0.016 | 0.944 ± 0.254 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.736 ± 0.014 | 0.731 ± 0.008 | **0.206 ± 0.012** | 0.612 ± 0.025 | 0.695 ± 2.305 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.248 ± 0.008 | 0.733 ± 0.012 | **0.202 ± 0.023** | 0.610 ± 0.025 | 7.739 ± 5.519 |
| bin_elementwise_mul_2048x2048 | 1.833 ± 0.082 | **1.794 ± 0.116** | 15.814 ± 0.322 | 17.044 ± 0.410 | 17.970 ± 1.259 |
| bin_elementwise_mul_512x512 | 0.132 ± 0.022 | **0.122 ± 0.008** | 0.147 ± 0.006 | 0.386 ± 0.022 | 0.166 ± 0.037 |
| bin_matmul_1024 | 55.942 ± 0.120 | **48.853 ± 0.204** | 60.354 ± 0.445 | 62.414 ± 0.442 | 76.576 ± 22.689 |
| bin_matmul_256 | 1.062 ± 0.011 | **0.824 ± 0.006** | 1.369 ± 0.018 | 1.183 ± 0.030 | 1.268 ± 0.829 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.016 ± 0.007 | **0.010 ± 0.001** | 0.037 ± 0.005 | 0.203 ± 0.016 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.042 ± 0.003 | **0.039 ± 0.028** | 0.151 ± 0.016 | 0.231 ± 0.019 | 0.049 ± 0.001 |
| bin_omeinsum_matmul_10x10 | 0.022 ± 0.001 | **0.011 ± 0.007** | 0.036 ± 0.007 | 0.148 ± 0.020 | 0.026 ± 0.001 |
| bin_outer_product_4096 | 63.861 ± 0.195 | 63.646 ± 0.608 | **53.411 ± 0.527** | 55.735 ± 0.785 | 306.887 ± 3.900 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 55.880 ± 0.260 | **50.500 ± 0.385** | 61.209 ± 0.116 | 65.682 ± 4.140 | 73.246 ± 14.773 |
| gm_queen5_5_3.wcsp | **2331.357 ± 11.675** | 2562.121 ± 18.445 | 2361.431 ± 26.311 | 2776.030 ± 9.150 | 3173.591 ± 51.748 |
| lm_batch_likelihood_brackets_4_4d | 26.860 ± 0.326 | 33.140 ± 1.000 | 35.741 ± 0.286 | **14.815 ± 0.334** | 68.118 ± 25.633 |
| lm_batch_likelihood_sentence_3_12d | **55.179 ± 0.230** | 149.656 ± 1.874 | 65.461 ± 0.432 | 123.490 ± 2.198 | 106.136 ± 33.734 |
| lm_batch_likelihood_sentence_4_4d | 27.853 ± 0.323 | 40.018 ± 2.872 | 38.016 ± 0.393 | **16.156 ± 0.730** | 45.394 ± 26.206 |
| nary_matmul_chain_64 | 0.066 ± 0.005 | **0.045 ± 0.020** | 0.080 ± 0.006 | 0.195 ± 0.022 | 0.125 ± 0.046 |
| str_matrix_chain_multiplication_100 | 14.998 ± 0.141 | 15.162 ± 0.075 | 15.651 ± 0.100 | **14.576 ± 0.452** | 16.377 ± 0.309 |
| str_mps_varying_inner_product_200 | **22.755 ± 0.160** | 23.303 ± 0.988 | 25.146 ± 0.350 | 23.063 ± 0.383 | 24.604 ± 9.276 |
| str_nw_mera_closed_120 | 1477.060 ± 4.205 | 1484.448 ± 11.600 | 1514.925 ± 1.314 | 2062.211 ± 12.840 | **1468.739 ± 40.523** |
| str_nw_mera_open_26 | 1291.920 ± 3.741 | 1453.197 ± 5.568 | **1252.286 ± 5.071** | 1991.376 ± 3.919 | 1565.419 ± 34.536 |
| tensornetwork_permutation_focus_step409_316 | **467.979 ± 1.913** | 531.130 ± 1.991 | 901.953 ± 1.554 | 839.942 ± 146.748 | 544.154 ± 79.989 |
| tensornetwork_permutation_light_415 | **464.874 ± 3.430** | 518.295 ± 2.111 | 867.982 ± 1.700 | 742.702 ± 35.595 | 590.757 ± 35.462 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261107_184743/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261107_184743/run.yaml`
- Timestamp: `2026-11-07T18:47:43.753266+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261107_184743`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
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

- `data/results/amd-cpu/cpu/einsum/20261107_184743/jax_cpu_t4_20261107_184743.log`
- `data/results/amd-cpu/cpu/einsum/20261107_184743/julia_omeinsum_t4_20261107_184743.log`
- `data/results/amd-cpu/cpu/einsum/20261107_184743/pytorch_cpu_t4_20261107_184743.log`
- `data/results/amd-cpu/cpu/einsum/20261107_184743/tenferro_eager_t4_20261107_184743.log`
- `data/results/amd-cpu/cpu/einsum/20261107_184743/tenferro_trace_t4_20261107_184743.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 9.695 ± 0.046 | 5.209 ± 0.026 | **2.276 ± 0.977** | 2.801 ± 0.624 | 9.926 ± 9.856 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.963 ± 0.016 | 1.368 ± 0.031 | **0.385 ± 0.032** | 0.838 ± 0.016 | 0.870 ± 1.136 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.309 ± 0.030 | 0.331 ± 0.031 | **0.203 ± 0.016** | 0.445 ± 0.026 | 3.598 ± 3.224 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.287 ± 0.017 | 0.306 ± 0.028 | **0.201 ± 0.031** | 0.458 ± 0.040 | 7.754 ± 3.145 |
| bin_elementwise_mul_2048x2048 | 1.007 ± 0.023 | **0.878 ± 0.031** | 5.632 ± 0.441 | 9.304 ± 0.213 | 18.024 ± 0.391 |
| bin_elementwise_mul_512x512 | 0.136 ± 0.000 | **0.130 ± 0.002** | 0.153 ± 0.019 | 0.416 ± 0.035 | 0.152 ± 0.229 |
| bin_matmul_1024 | 61.278 ± 0.179 | 51.284 ± 0.485 | 36.450 ± 0.325 | **35.890 ± 0.291** | 53.389 ± 29.669 |
| bin_matmul_256 | 1.267 ± 0.021 | 0.930 ± 0.174 | 0.859 ± 0.006 | **0.842 ± 0.059** | 0.920 ± 0.077 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.000 | **0.010 ± 0.000** | 0.054 ± 0.008 | 0.204 ± 0.022 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.065 ± 0.003 | **0.062 ± 0.003** | 0.224 ± 0.015 | 0.247 ± 0.029 | 0.087 ± 0.036 |
| bin_omeinsum_matmul_10x10 | 0.070 ± 0.003 | 0.048 ± 0.003 | 0.049 ± 0.008 | 0.150 ± 0.017 | **0.026 ± 0.001** |
| bin_outer_product_4096 | 23.702 ± 0.582 | 23.677 ± 0.569 | **18.150 ± 0.216** | 28.397 ± 0.560 | 298.530 ± 5.446 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 59.979 ± 0.271 | 53.664 ± 0.234 | **29.657 ± 0.306** | 37.908 ± 0.932 | 48.673 ± 6.388 |
| gm_queen5_5_3.wcsp | 6268.828 ± 88.380 | 5751.586 ± 68.740 | **2606.378 ± 19.665** | 5188.497 ± 108.527 | 8313.977 ± 253.360 |
| lm_batch_likelihood_brackets_4_4d | 29.444 ± 0.134 | 47.267 ± 2.465 | **16.542 ± 0.451** | 23.170 ± 0.696 | 43.034 ± 20.120 |
| lm_batch_likelihood_sentence_3_12d | 61.128 ± 0.469 | 166.316 ± 27.044 | 57.247 ± 1.480 | **53.721 ± 7.695** | 97.863 ± 25.647 |
| lm_batch_likelihood_sentence_4_4d | 32.010 ± 0.416 | 53.553 ± 2.001 | **17.460 ± 0.302** | 23.235 ± 1.268 | 73.236 ± 28.000 |
| nary_matmul_chain_64 | 0.148 ± 0.006 | 0.132 ± 0.014 | **0.093 ± 0.011** | 0.275 ± 0.025 | 0.160 ± 0.050 |
| str_matrix_chain_multiplication_100 | 17.037 ± 0.073 | 18.301 ± 0.061 | **10.908 ± 0.141** | 15.420 ± 0.710 | 14.612 ± 1.022 |
| str_mps_varying_inner_product_200 | **25.889 ± 0.209** | 27.846 ± 0.093 | 26.066 ± 0.368 | 30.824 ± 0.668 | 30.939 ± 5.374 |
| str_nw_mera_closed_120 | 1653.096 ± 5.573 | 1848.306 ± 16.255 | **930.623 ± 5.704** | 1167.514 ± 25.755 | 1217.669 ± 55.702 |
| str_nw_mera_open_26 | 1303.656 ± 6.007 | 1322.679 ± 2.362 | **582.456 ± 4.336** | 916.626 ± 9.451 | 1227.053 ± 43.673 |
| tensornetwork_permutation_focus_step409_316 | 473.298 ± 4.759 | 502.248 ± 5.379 | **341.058 ± 1.975** | 391.604 ± 8.451 | 573.652 ± 29.178 |
| tensornetwork_permutation_light_415 | 468.172 ± 2.819 | 465.727 ± 1.031 | **340.732 ± 4.815** | 408.050 ± 6.199 | 539.394 ± 93.499 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 9.695 ± 0.046 | 5.209 ± 0.026 | **2.276 ± 0.977** | 2.801 ± 0.624 | 9.926 ± 9.856 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.963 ± 0.016 | 1.368 ± 0.031 | **0.385 ± 0.032** | 0.838 ± 0.016 | 0.870 ± 1.136 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.309 ± 0.030 | 0.331 ± 0.031 | **0.203 ± 0.016** | 0.445 ± 0.026 | 3.598 ± 3.224 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.287 ± 0.017 | 0.306 ± 0.028 | **0.201 ± 0.031** | 0.458 ± 0.040 | 7.754 ± 3.145 |
| bin_elementwise_mul_2048x2048 | 1.007 ± 0.023 | **0.878 ± 0.031** | 5.632 ± 0.441 | 9.304 ± 0.213 | 18.024 ± 0.391 |
| bin_elementwise_mul_512x512 | 0.136 ± 0.000 | **0.130 ± 0.002** | 0.153 ± 0.019 | 0.416 ± 0.035 | 0.152 ± 0.229 |
| bin_matmul_1024 | 61.278 ± 0.179 | 51.284 ± 0.485 | 36.450 ± 0.325 | **35.890 ± 0.291** | 53.389 ± 29.669 |
| bin_matmul_256 | 1.267 ± 0.021 | 0.930 ± 0.174 | 0.859 ± 0.006 | **0.842 ± 0.059** | 0.920 ± 0.077 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.015 ± 0.000 | **0.010 ± 0.000** | 0.054 ± 0.008 | 0.204 ± 0.022 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.065 ± 0.003 | **0.062 ± 0.003** | 0.224 ± 0.015 | 0.247 ± 0.029 | 0.087 ± 0.036 |
| bin_omeinsum_matmul_10x10 | 0.070 ± 0.003 | 0.048 ± 0.003 | 0.049 ± 0.008 | 0.150 ± 0.017 | **0.026 ± 0.001** |
| bin_outer_product_4096 | 23.702 ± 0.582 | 23.677 ± 0.569 | **18.150 ± 0.216** | 28.397 ± 0.560 | 298.530 ± 5.446 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 59.979 ± 0.271 | 53.664 ± 0.234 | **29.657 ± 0.306** | 37.908 ± 0.932 | 48.673 ± 6.388 |
| gm_queen5_5_3.wcsp | 2307.878 ± 10.509 | 2048.779 ± 10.918 | **1097.423 ± 10.431** | 1814.732 ± 13.749 | 2973.998 ± 10.566 |
| lm_batch_likelihood_brackets_4_4d | 30.046 ± 0.372 | 49.804 ± 0.961 | **17.952 ± 0.715** | 23.570 ± 1.799 | 53.572 ± 31.005 |
| lm_batch_likelihood_sentence_3_12d | 56.426 ± 0.194 | 224.319 ± 1.152 | **32.690 ± 0.277** | 57.312 ± 7.821 | 106.905 ± 47.166 |
| lm_batch_likelihood_sentence_4_4d | 31.360 ± 0.079 | 56.329 ± 2.269 | **18.148 ± 0.188** | 23.570 ± 1.921 | 55.791 ± 33.303 |
| nary_matmul_chain_64 | 0.148 ± 0.006 | 0.132 ± 0.014 | **0.093 ± 0.011** | 0.275 ± 0.025 | 0.160 ± 0.050 |
| str_matrix_chain_multiplication_100 | 16.531 ± 0.074 | 18.216 ± 0.076 | **10.842 ± 0.139** | 14.921 ± 0.717 | 14.578 ± 0.349 |
| str_mps_varying_inner_product_200 | 28.197 ± 0.278 | 30.091 ± 0.218 | 28.750 ± 0.598 | 31.049 ± 1.003 | **23.789 ± 12.755** |
| str_nw_mera_closed_120 | 1519.610 ± 1.967 | 1515.224 ± 24.594 | **747.679 ± 6.351** | 972.911 ± 23.038 | 950.447 ± 41.063 |
| str_nw_mera_open_26 | 1305.378 ± 4.036 | 1316.701 ± 4.363 | **582.114 ± 2.819** | 943.232 ± 12.159 | 1185.588 ± 40.590 |
| tensornetwork_permutation_focus_step409_316 | 473.298 ± 4.759 | 502.248 ± 5.379 | **341.058 ± 1.975** | 391.604 ± 8.451 | 573.652 ± 29.178 |
| tensornetwork_permutation_light_415 | 468.172 ± 2.819 | 465.727 ± 1.031 | **340.732 ± 4.815** | 408.050 ± 6.199 | 539.394 ± 93.499 |

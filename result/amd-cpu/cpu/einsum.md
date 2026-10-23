# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a24fccea2dc7cee09df442104201b7c52f9b943b`
- Benchmark commit: `cc2153b9548e1542f955c011dccfb77a98196fed`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261023_175936/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261023_175936/run.yaml`
- Timestamp: `20261023_175936`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261023_175936`.

- tenferro-rs commit: `a24fccea2dc7cee09df442104201b7c52f9b943b`

### CPU Information

- Model: `AMD EPYC 9V45 96-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

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

- Julia: version `1.13.2`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `1`

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


- Source table: `data/results/amd-cpu/cpu/einsum/20261023_175936/einsum_table_t1_20261023_175936.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20261023_175936/tenferro_trace_t1_20261023_175936.log`
- `data/results/amd-cpu/cpu/einsum/20261023_175936/tenferro_eager_t1_20261023_175936.log`
- `data/results/amd-cpu/cpu/einsum/20261023_175936/pytorch_cpu_t1_20261023_175936.log`
- `data/results/amd-cpu/cpu/einsum/20261023_175936/jax_cpu_t1_20261023_175936.log`
- `data/results/amd-cpu/cpu/einsum/20261023_175936/julia_omeinsum_t1_20261023_175936.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.991 ± 0.024 | **2.451 ± 0.017** | 3.234 ± 0.072 | 2.591 ± 0.018 | 5.573 ± 5.033 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.425 ± 0.014 | 0.350 ± 0.012 | 0.524 ± 0.088 | 0.443 ± 0.037 | **0.316 ± 0.020** |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.114 ± 0.001 | 0.114 ± 0.002 | **0.096 ± 0.008** | 0.244 ± 0.017 | 0.941 ± 0.671 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.206 ± 0.003 | 0.112 ± 0.001 | **0.094 ± 0.002** | 0.241 ± 0.012 | 2.176 ± 1.637 |
| bin_elementwise_mul_2048x2048 | 1.082 ± 0.059 | **1.058 ± 0.027** | 3.470 ± 0.050 | 4.160 ± 0.132 | 3.931 ± 0.162 |
| bin_elementwise_mul_512x512 | 0.061 ± 0.001 | **0.059 ± 0.000** | 0.071 ± 0.005 | 0.156 ± 0.025 | 0.095 ± 0.556 |
| bin_matmul_1024 | 42.334 ± 0.221 | **35.685 ± 2.164** | 57.893 ± 2.401 | 36.552 ± 0.919 | 39.555 ± 5.248 |
| bin_matmul_256 | 0.685 ± 0.026 | **0.573 ± 0.014** | 0.844 ± 0.040 | 0.643 ± 0.018 | 1.127 ± 0.056 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.006 ± 0.000 | **0.003 ± 0.001** | 0.021 ± 0.004 | 0.079 ± 0.011 | 0.008 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.021 ± 0.001 | **0.020 ± 0.000** | 0.066 ± 0.001 | 0.115 ± 0.008 | 0.024 ± 0.002 |
| bin_omeinsum_matmul_10x10 | 0.011 ± 0.000 | **0.010 ± 0.001** | 0.013 ± 0.000 | 0.047 ± 0.007 | 0.013 ± 0.001 |
| bin_outer_product_4096 | **7.108 ± 0.107** | 7.532 ± 0.112 | 7.915 ± 0.111 | 7.617 ± 0.185 | 421.113 ± 9.103 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 41.282 ± 0.341 | 35.674 ± 0.222 | 46.845 ± 0.781 | 36.612 ± 0.519 | **34.325 ± 5.373** |
| gm_queen5_5_3.wcsp | 4799.927 ± 151.142 | 3531.472 ± 51.174 | 3394.082 ± 280.708 | **2172.192 ± 76.384** | 3609.897 ± 339.783 |
| lm_batch_likelihood_brackets_4_4d | 14.002 ± 0.248 | 17.526 ± 0.909 | 16.586 ± 0.355 | **9.372 ± 0.335** | 45.837 ± 34.154 |
| lm_batch_likelihood_sentence_3_12d | **40.471 ± 0.297** | 58.529 ± 0.887 | 47.592 ± 1.141 | 42.495 ± 0.863 | 68.352 ± 27.777 |
| lm_batch_likelihood_sentence_4_4d | 15.208 ± 0.134 | 20.773 ± 1.064 | 18.142 ± 0.233 | **10.467 ± 0.564** | 20.170 ± 32.508 |
| nary_matmul_chain_64 | 0.033 ± 0.001 | 0.031 ± 0.001 | 0.035 ± 0.011 | 0.081 ± 0.005 | **0.028 ± 0.028** |
| str_matrix_chain_multiplication_100 | **9.195 ± 0.051** | 9.275 ± 0.055 | 10.038 ± 0.313 | 9.953 ± 0.198 | 10.499 ± 1.304 |
| str_mps_varying_inner_product_200 | 13.314 ± 0.254 | **13.096 ± 0.190** | 14.099 ± 0.318 | 15.606 ± 0.109 | 18.159 ± 6.420 |
| str_nw_mera_closed_120 | 996.799 ± 18.459 | 905.230 ± 5.791 | 932.331 ± 10.098 | 1079.005 ± 10.686 | **700.820 ± 11.432** |
| str_nw_mera_open_26 | 727.800 ± 6.747 | 700.863 ± 10.956 | **615.123 ± 11.836** | 876.175 ± 16.979 | 711.953 ± 8.474 |
| tensornetwork_permutation_focus_step409_316 | **218.673 ± 3.360** | 266.851 ± 5.298 | 481.666 ± 8.855 | 1422.634 ± 24.517 | 259.871 ± 32.099 |
| tensornetwork_permutation_light_415 | **198.665 ± 3.129** | 202.442 ± 1.329 | 448.903 ± 6.544 | 1406.775 ± 28.257 | 250.124 ± 61.912 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.991 ± 0.024 | **2.451 ± 0.017** | 3.234 ± 0.072 | 2.591 ± 0.018 | 5.573 ± 5.033 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.425 ± 0.014 | 0.350 ± 0.012 | 0.524 ± 0.088 | 0.443 ± 0.037 | **0.316 ± 0.020** |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.114 ± 0.001 | 0.114 ± 0.002 | **0.096 ± 0.008** | 0.244 ± 0.017 | 0.941 ± 0.671 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.206 ± 0.003 | 0.112 ± 0.001 | **0.094 ± 0.002** | 0.241 ± 0.012 | 2.176 ± 1.637 |
| bin_elementwise_mul_2048x2048 | 1.082 ± 0.059 | **1.058 ± 0.027** | 3.470 ± 0.050 | 4.160 ± 0.132 | 3.931 ± 0.162 |
| bin_elementwise_mul_512x512 | 0.061 ± 0.001 | **0.059 ± 0.000** | 0.071 ± 0.005 | 0.156 ± 0.025 | 0.095 ± 0.556 |
| bin_matmul_1024 | 42.334 ± 0.221 | **35.685 ± 2.164** | 57.893 ± 2.401 | 36.552 ± 0.919 | 39.555 ± 5.248 |
| bin_matmul_256 | 0.685 ± 0.026 | **0.573 ± 0.014** | 0.844 ± 0.040 | 0.643 ± 0.018 | 1.127 ± 0.056 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.006 ± 0.000 | **0.003 ± 0.001** | 0.021 ± 0.004 | 0.079 ± 0.011 | 0.008 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.021 ± 0.001 | **0.020 ± 0.000** | 0.066 ± 0.001 | 0.115 ± 0.008 | 0.024 ± 0.002 |
| bin_omeinsum_matmul_10x10 | 0.011 ± 0.000 | **0.010 ± 0.001** | 0.013 ± 0.000 | 0.047 ± 0.007 | 0.013 ± 0.001 |
| bin_outer_product_4096 | **7.108 ± 0.107** | 7.532 ± 0.112 | 7.915 ± 0.111 | 7.617 ± 0.185 | 421.113 ± 9.103 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 41.282 ± 0.341 | 35.674 ± 0.222 | 46.845 ± 0.781 | 36.612 ± 0.519 | **34.325 ± 5.373** |
| gm_queen5_5_3.wcsp | 1272.090 ± 18.552 | 1240.664 ± 14.685 | 1317.442 ± 36.992 | **932.081 ± 40.438** | 1355.522 ± 23.897 |
| lm_batch_likelihood_brackets_4_4d | 11.947 ± 0.161 | 17.850 ± 0.616 | 18.515 ± 0.644 | **9.347 ± 0.517** | 22.878 ± 28.355 |
| lm_batch_likelihood_sentence_3_12d | **37.829 ± 1.359** | 65.324 ± 0.737 | 41.877 ± 0.631 | 47.789 ± 1.278 | 76.013 ± 34.199 |
| lm_batch_likelihood_sentence_4_4d | 12.581 ± 0.140 | 21.713 ± 1.068 | 19.922 ± 0.128 | **9.118 ± 0.368** | 22.860 ± 35.386 |
| nary_matmul_chain_64 | 0.033 ± 0.001 | 0.031 ± 0.001 | 0.035 ± 0.011 | 0.081 ± 0.005 | **0.028 ± 0.028** |
| str_matrix_chain_multiplication_100 | **9.327 ± 0.152** | 10.074 ± 3.389 | 10.041 ± 0.093 | 9.393 ± 0.256 | 10.160 ± 0.523 |
| str_mps_varying_inner_product_200 | 13.339 ± 0.376 | **13.054 ± 0.156** | 13.062 ± 0.115 | 14.253 ± 0.113 | 14.667 ± 1.320 |
| str_nw_mera_closed_120 | 857.652 ± 17.797 | 859.092 ± 11.096 | 868.145 ± 33.517 | 956.232 ± 8.075 | **688.842 ± 54.129** |
| str_nw_mera_open_26 | 658.484 ± 11.168 | 695.998 ± 11.926 | **615.155 ± 11.349** | 878.690 ± 25.733 | 720.234 ± 40.959 |
| tensornetwork_permutation_focus_step409_316 | **218.673 ± 3.360** | 266.851 ± 5.298 | 481.666 ± 8.855 | 1422.634 ± 24.517 | 259.871 ± 32.099 |
| tensornetwork_permutation_light_415 | **198.665 ± 3.129** | 202.442 ± 1.329 | 448.903 ± 6.544 | 1406.775 ± 28.257 | 250.124 ± 61.912 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261023_182242/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261023_182242/run.yaml`
- Timestamp: `20261023_182242`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261023_182242`.

- tenferro-rs commit: `a24fccea2dc7cee09df442104201b7c52f9b943b`

### CPU Information

- Model: `AMD EPYC 9V45 96-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

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

- Julia: version `1.13.2`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `4`

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


- Source table: `data/results/amd-cpu/cpu/einsum/20261023_182242/einsum_table_t4_20261023_182242.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20261023_182242/tenferro_trace_t4_20261023_182242.log`
- `data/results/amd-cpu/cpu/einsum/20261023_182242/tenferro_eager_t4_20261023_182242.log`
- `data/results/amd-cpu/cpu/einsum/20261023_182242/pytorch_cpu_t4_20261023_182242.log`
- `data/results/amd-cpu/cpu/einsum/20261023_182242/jax_cpu_t4_20261023_182242.log`
- `data/results/amd-cpu/cpu/einsum/20261023_182242/julia_omeinsum_t4_20261023_182242.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.087 ± 0.051 | 1.568 ± 0.030 | **1.315 ± 4.675** | 1.614 ± 0.293 | 6.492 ± 5.974 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.440 ± 0.010 | 0.352 ± 0.013 | **0.205 ± 0.003** | 0.421 ± 0.013 | 0.353 ± 1.264 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.111 ± 0.018 | 0.106 ± 0.006 | **0.073 ± 0.007** | 0.211 ± 0.038 | 0.931 ± 0.766 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.187 ± 0.015 | 0.114 ± 0.013 | **0.067 ± 0.016** | 0.205 ± 0.020 | 2.546 ± 2.962 |
| bin_elementwise_mul_2048x2048 | **0.745 ± 0.037** | 1.052 ± 0.008 | 2.389 ± 0.042 | 3.526 ± 0.113 | 3.981 ± 0.071 |
| bin_elementwise_mul_512x512 | 0.078 ± 0.034 | 0.085 ± 0.006 | **0.045 ± 0.002** | 0.191 ± 0.061 | 0.082 ± 0.019 |
| bin_matmul_1024 | 27.237 ± 1.246 | **17.710 ± 0.088** | 35.397 ± 2.571 | 18.321 ± 0.243 | 26.084 ± 12.175 |
| bin_matmul_256 | 0.420 ± 0.024 | **0.322 ± 0.013** | 0.522 ± 0.016 | 0.421 ± 0.042 | 0.479 ± 0.006 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.005 ± 0.000** | 0.020 ± 0.002 | 0.075 ± 0.010 | 0.008 ± 0.000 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.001 | 0.028 ± 0.001 | 0.094 ± 0.001 | 0.114 ± 0.008 | **0.024 ± 0.001** |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.000 | **0.009 ± 0.000** | 0.019 ± 0.001 | 0.065 ± 0.008 | 0.012 ± 0.001 |
| bin_outer_product_4096 | 6.274 ± 0.228 | 5.297 ± 0.514 | **4.120 ± 0.199** | 4.703 ± 0.252 | 316.730 ± 24.526 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 25.297 ± 2.196 | **18.705 ± 0.486** | 24.325 ± 0.660 | 20.239 ± 2.590 | 23.865 ± 7.341 |
| gm_queen5_5_3.wcsp | 2982.352 ± 150.781 | 2375.345 ± 144.231 | **1296.113 ± 50.997** | 1438.541 ± 25.057 | 3640.916 ± 301.519 |
| lm_batch_likelihood_brackets_4_4d | 10.510 ± 1.231 | 20.179 ± 2.834 | **8.183 ± 0.078** | 12.018 ± 0.620 | 21.552 ± 15.011 |
| lm_batch_likelihood_sentence_3_12d | **19.917 ± 1.272** | 47.796 ± 6.079 | 30.092 ± 0.274 | 22.719 ± 1.068 | 49.356 ± 33.491 |
| lm_batch_likelihood_sentence_4_4d | 12.088 ± 0.313 | 22.930 ± 7.988 | **8.944 ± 0.138** | 12.694 ± 0.740 | 27.968 ± 35.308 |
| nary_matmul_chain_64 | 0.176 ± 0.006 | **0.026 ± 0.000** | 0.042 ± 0.000 | 0.069 ± 0.005 | 0.068 ± 0.045 |
| str_matrix_chain_multiplication_100 | **4.687 ± 0.041** | 5.873 ± 0.081 | 5.899 ± 0.078 | 8.319 ± 0.207 | 8.274 ± 1.241 |
| str_mps_varying_inner_product_200 | **12.039 ± 0.309** | 13.943 ± 0.910 | 12.239 ± 0.220 | 16.538 ± 0.347 | 17.231 ± 17.137 |
| str_nw_mera_closed_120 | **456.869 ± 28.470** | 496.756 ± 24.409 | 471.717 ± 13.403 | 460.053 ± 13.379 | 478.857 ± 13.085 |
| str_nw_mera_open_26 | 404.759 ± 39.451 | 438.959 ± 37.040 | **312.983 ± 10.054** | 427.337 ± 13.264 | 616.395 ± 19.922 |
| tensornetwork_permutation_focus_step409_316 | **138.525 ± 9.101** | 169.878 ± 21.780 | 182.441 ± 2.113 | 681.659 ± 12.136 | 258.752 ± 39.945 |
| tensornetwork_permutation_light_415 | **133.396 ± 12.061** | 138.872 ± 20.339 | 182.959 ± 1.880 | 687.495 ± 33.764 | 239.067 ± 35.911 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.087 ± 0.051 | 1.568 ± 0.030 | **1.315 ± 4.675** | 1.614 ± 0.293 | 6.492 ± 5.974 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.440 ± 0.010 | 0.352 ± 0.013 | **0.205 ± 0.003** | 0.421 ± 0.013 | 0.353 ± 1.264 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.111 ± 0.018 | 0.106 ± 0.006 | **0.073 ± 0.007** | 0.211 ± 0.038 | 0.931 ± 0.766 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.187 ± 0.015 | 0.114 ± 0.013 | **0.067 ± 0.016** | 0.205 ± 0.020 | 2.546 ± 2.962 |
| bin_elementwise_mul_2048x2048 | **0.745 ± 0.037** | 1.052 ± 0.008 | 2.389 ± 0.042 | 3.526 ± 0.113 | 3.981 ± 0.071 |
| bin_elementwise_mul_512x512 | 0.078 ± 0.034 | 0.085 ± 0.006 | **0.045 ± 0.002** | 0.191 ± 0.061 | 0.082 ± 0.019 |
| bin_matmul_1024 | 27.237 ± 1.246 | **17.710 ± 0.088** | 35.397 ± 2.571 | 18.321 ± 0.243 | 26.084 ± 12.175 |
| bin_matmul_256 | 0.420 ± 0.024 | **0.322 ± 0.013** | 0.522 ± 0.016 | 0.421 ± 0.042 | 0.479 ± 0.006 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.005 ± 0.000** | 0.020 ± 0.002 | 0.075 ± 0.010 | 0.008 ± 0.000 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.001 | 0.028 ± 0.001 | 0.094 ± 0.001 | 0.114 ± 0.008 | **0.024 ± 0.001** |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.000 | **0.009 ± 0.000** | 0.019 ± 0.001 | 0.065 ± 0.008 | 0.012 ± 0.001 |
| bin_outer_product_4096 | 6.274 ± 0.228 | 5.297 ± 0.514 | **4.120 ± 0.199** | 4.703 ± 0.252 | 316.730 ± 24.526 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 25.297 ± 2.196 | **18.705 ± 0.486** | 24.325 ± 0.660 | 20.239 ± 2.590 | 23.865 ± 7.341 |
| gm_queen5_5_3.wcsp | 809.217 ± 83.007 | 818.917 ± 86.378 | 598.243 ± 13.486 | **572.730 ± 12.293** | 1235.617 ± 10.973 |
| lm_batch_likelihood_brackets_4_4d | 14.265 ± 4.419 | 20.994 ± 3.658 | **8.992 ± 0.060** | 12.592 ± 1.046 | 25.964 ± 25.297 |
| lm_batch_likelihood_sentence_3_12d | 32.978 ± 6.350 | 59.621 ± 10.744 | **21.436 ± 0.252** | 29.325 ± 1.345 | 79.853 ± 33.608 |
| lm_batch_likelihood_sentence_4_4d | 11.193 ± 0.413 | 24.943 ± 4.818 | **9.389 ± 0.262** | 14.429 ± 1.531 | 31.052 ± 39.049 |
| nary_matmul_chain_64 | 0.176 ± 0.006 | **0.026 ± 0.000** | 0.042 ± 0.000 | 0.069 ± 0.005 | 0.068 ± 0.045 |
| str_matrix_chain_multiplication_100 | **5.051 ± 0.066** | 5.602 ± 0.067 | 5.848 ± 0.106 | 9.302 ± 1.098 | 8.685 ± 0.924 |
| str_mps_varying_inner_product_200 | 14.520 ± 0.179 | 14.653 ± 0.515 | **13.107 ± 0.219** | 17.493 ± 4.028 | 16.486 ± 4.796 |
| str_nw_mera_closed_120 | 439.714 ± 24.654 | 460.035 ± 15.023 | **431.548 ± 4.796** | 455.593 ± 14.706 | 459.957 ± 34.714 |
| str_nw_mera_open_26 | 390.244 ± 35.711 | 418.299 ± 10.476 | **322.378 ± 9.554** | 418.100 ± 17.097 | 586.176 ± 39.502 |
| tensornetwork_permutation_focus_step409_316 | **138.525 ± 9.101** | 169.878 ± 21.780 | 182.441 ± 2.113 | 681.659 ± 12.136 | 258.752 ± 39.945 |
| tensornetwork_permutation_light_415 | **133.396 ± 12.061** | 138.872 ± 20.339 | 182.959 ± 1.880 | 687.495 ± 33.764 | 239.067 ± 35.911 |

# CPU Einsum Benchmark Results

- Target profile: `zen5-cpu`
- tenferro-rs commit: `8fec1ca70852b930c092e7177c30bb8af7594a57`
- Benchmark commit: `3f31f38e0491084b9414035754a3809ef0f5ae94`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/zen5-cpu/cpu/einsum/20261027_231715/report.md`.


- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261027_231715/run.yaml`
- Timestamp: `20261027_231715`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261027_231715`.

- tenferro-rs commit: `8fec1ca70852b930c092e7177c30bb8af7594a57`

### CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `enabled`

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


- Source table: `data/results/zen5-cpu/cpu/einsum/20261027_231715/einsum_table_t1_20261027_231715.md`

Logs:

- `data/results/zen5-cpu/cpu/einsum/20261027_231715/tenferro_trace_t1_20261027_231715.log`
- `data/results/zen5-cpu/cpu/einsum/20261027_231715/tenferro_eager_t1_20261027_231715.log`
- `data/results/zen5-cpu/cpu/einsum/20261027_231715/pytorch_cpu_t1_20261027_231715.log`
- `data/results/zen5-cpu/cpu/einsum/20261027_231715/jax_cpu_t1_20261027_231715.log`
- `data/results/zen5-cpu/cpu/einsum/20261027_231715/julia_omeinsum_t1_20261027_231715.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.047 ± 0.081 | 2.166 ± 0.085 | 2.396 ± 0.034 | **2.154 ± 0.028** | 6.443 ± 5.395 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.382 ± 0.006 | **0.305 ± 0.005** | 0.331 ± 0.003 | 0.338 ± 0.010 | 0.457 ± 0.736 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.134 ± 0.014 | 0.120 ± 0.007 | **0.074 ± 0.001** | 0.203 ± 0.016 | 0.691 ± 1.113 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.485 ± 0.005 | 0.120 ± 0.003 | **0.074 ± 0.001** | 0.196 ± 0.015 | 3.342 ± 1.811 |
| bin_elementwise_mul_2048x2048 | 1.607 ± 0.040 | **1.527 ± 0.026** | 6.445 ± 0.145 | 9.317 ± 1.050 | 7.190 ± 2.489 |
| bin_elementwise_mul_512x512 | 0.049 ± 0.006 | **0.048 ± 0.008** | 0.050 ± 0.001 | 0.119 ± 0.009 | 0.087 ± 0.244 |
| bin_matmul_1024 | 35.925 ± 0.179 | **29.722 ± 0.126** | 34.676 ± 0.106 | 30.646 ± 0.095 | 49.426 ± 16.034 |
| bin_matmul_256 | 0.573 ± 0.021 | **0.460 ± 0.008** | 0.695 ± 0.014 | 0.530 ± 0.004 | 0.703 ± 0.006 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.005 ± 0.000 | **0.003 ± 0.000** | 0.011 ± 0.000 | 0.064 ± 0.005 | 0.007 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.018 ± 0.000 | **0.016 ± 0.000** | 0.054 ± 0.000 | 0.087 ± 0.009 | 0.037 ± 0.009 |
| bin_omeinsum_matmul_10x10 | 0.010 ± 0.000 | **0.005 ± 0.000** | 0.011 ± 0.000 | 0.045 ± 0.006 | 0.012 ± 0.001 |
| bin_outer_product_4096 | 21.457 ± 0.654 | 21.532 ± 0.576 | **20.847 ± 0.912** | 21.456 ± 3.914 | 154.639 ± 30.207 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 38.449 ± 0.417 | **30.948 ± 0.136** | 34.726 ± 0.068 | 34.763 ± 0.194 | 43.122 ± 11.754 |
| gm_queen5_5_3.wcsp | 4492.272 ± 25.246 | 3761.453 ± 6.039 | 3522.951 ± 63.128 | **2736.080 ± 15.119** | 5007.983 ± 65.698 |
| lm_batch_likelihood_brackets_4_4d | **12.527 ± 0.057** | 18.377 ± 0.966 | 16.241 ± 0.156 | 14.358 ± 0.313 | 17.717 ± 7.574 |
| lm_batch_likelihood_sentence_3_12d | **39.877 ± 0.232** | 74.201 ± 0.350 | 56.789 ± 0.457 | 51.156 ± 0.137 | 84.874 ± 38.273 |
| lm_batch_likelihood_sentence_4_4d | **13.188 ± 0.176** | 21.176 ± 0.333 | 17.606 ± 0.291 | 14.778 ± 0.136 | 24.167 ± 23.864 |
| nary_matmul_chain_64 | 0.028 ± 0.000 | **0.023 ± 0.000** | 0.031 ± 0.000 | 0.073 ± 0.006 | 0.054 ± 0.007 |
| str_matrix_chain_multiplication_100 | **8.009 ± 0.058** | 8.348 ± 0.042 | 8.824 ± 0.045 | 10.304 ± 0.253 | 11.130 ± 0.941 |
| str_mps_varying_inner_product_200 | **10.612 ± 0.107** | 12.654 ± 0.129 | 12.347 ± 0.031 | 17.596 ± 0.178 | 18.326 ± 29.974 |
| str_nw_mera_closed_120 | **948.940 ± 2.771** | 987.014 ± 5.354 | 1056.507 ± 2.733 | 1116.942 ± 3.994 | 1189.776 ± 35.081 |
| str_nw_mera_open_26 | 756.553 ± 2.166 | 808.681 ± 2.418 | **654.409 ± 1.261** | 922.037 ± 2.208 | 1093.886 ± 33.365 |
| tensornetwork_permutation_focus_step409_316 | **259.165 ± 0.987** | 342.096 ± 1.296 | 466.793 ± 0.857 | 417.033 ± 1.571 | 387.125 ± 48.689 |
| tensornetwork_permutation_light_415 | **256.016 ± 1.483** | 266.211 ± 1.241 | 450.252 ± 1.470 | 423.649 ± 1.472 | 390.496 ± 107.997 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.047 ± 0.081 | 2.166 ± 0.085 | 2.396 ± 0.034 | **2.154 ± 0.028** | 6.443 ± 5.395 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.382 ± 0.006 | **0.305 ± 0.005** | 0.331 ± 0.003 | 0.338 ± 0.010 | 0.457 ± 0.736 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.134 ± 0.014 | 0.120 ± 0.007 | **0.074 ± 0.001** | 0.203 ± 0.016 | 0.691 ± 1.113 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.485 ± 0.005 | 0.120 ± 0.003 | **0.074 ± 0.001** | 0.196 ± 0.015 | 3.342 ± 1.811 |
| bin_elementwise_mul_2048x2048 | 1.607 ± 0.040 | **1.527 ± 0.026** | 6.445 ± 0.145 | 9.317 ± 1.050 | 7.190 ± 2.489 |
| bin_elementwise_mul_512x512 | 0.049 ± 0.006 | **0.048 ± 0.008** | 0.050 ± 0.001 | 0.119 ± 0.009 | 0.087 ± 0.244 |
| bin_matmul_1024 | 35.925 ± 0.179 | **29.722 ± 0.126** | 34.676 ± 0.106 | 30.646 ± 0.095 | 49.426 ± 16.034 |
| bin_matmul_256 | 0.573 ± 0.021 | **0.460 ± 0.008** | 0.695 ± 0.014 | 0.530 ± 0.004 | 0.703 ± 0.006 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.005 ± 0.000 | **0.003 ± 0.000** | 0.011 ± 0.000 | 0.064 ± 0.005 | 0.007 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.018 ± 0.000 | **0.016 ± 0.000** | 0.054 ± 0.000 | 0.087 ± 0.009 | 0.037 ± 0.009 |
| bin_omeinsum_matmul_10x10 | 0.010 ± 0.000 | **0.005 ± 0.000** | 0.011 ± 0.000 | 0.045 ± 0.006 | 0.012 ± 0.001 |
| bin_outer_product_4096 | 21.457 ± 0.654 | 21.532 ± 0.576 | **20.847 ± 0.912** | 21.456 ± 3.914 | 154.639 ± 30.207 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 38.449 ± 0.417 | **30.948 ± 0.136** | 34.726 ± 0.068 | 34.763 ± 0.194 | 43.122 ± 11.754 |
| gm_queen5_5_3.wcsp | 1485.442 ± 5.491 | 1584.827 ± 2.815 | 1465.193 ± 6.489 | **1067.073 ± 1.826** | 2009.878 ± 15.527 |
| lm_batch_likelihood_brackets_4_4d | **12.578 ± 0.092** | 18.744 ± 0.322 | 16.768 ± 0.129 | 13.805 ± 0.157 | 28.937 ± 38.364 |
| lm_batch_likelihood_sentence_3_12d | 44.663 ± 0.446 | 81.928 ± 0.763 | **41.633 ± 0.205** | 64.109 ± 0.289 | 91.663 ± 24.404 |
| lm_batch_likelihood_sentence_4_4d | **13.679 ± 0.157** | 23.709 ± 0.758 | 18.121 ± 0.049 | 13.925 ± 0.160 | 27.787 ± 32.991 |
| nary_matmul_chain_64 | 0.028 ± 0.000 | **0.023 ± 0.000** | 0.031 ± 0.000 | 0.073 ± 0.006 | 0.054 ± 0.007 |
| str_matrix_chain_multiplication_100 | **8.219 ± 0.075** | 8.348 ± 0.031 | 8.828 ± 0.036 | 10.342 ± 0.173 | 11.006 ± 0.930 |
| str_mps_varying_inner_product_200 | 11.847 ± 0.052 | 12.047 ± 0.076 | **11.428 ± 0.041** | 16.857 ± 0.307 | 15.188 ± 9.499 |
| str_nw_mera_closed_120 | 888.520 ± 1.115 | 877.532 ± 4.800 | **874.883 ± 2.927** | 926.983 ± 1.513 | 1073.027 ± 21.064 |
| str_nw_mera_open_26 | 751.918 ± 2.763 | 798.729 ± 1.986 | **671.829 ± 2.533** | 932.501 ± 1.794 | 1070.435 ± 32.258 |
| tensornetwork_permutation_focus_step409_316 | **259.165 ± 0.987** | 342.096 ± 1.296 | 466.793 ± 0.857 | 417.033 ± 1.571 | 387.125 ± 48.689 |
| tensornetwork_permutation_light_415 | **256.016 ± 1.483** | 266.211 ± 1.241 | 450.252 ± 1.470 | 423.649 ± 1.472 | 390.496 ± 107.997 |

## Threads: 4

Source report: `data/results/zen5-cpu/cpu/einsum/20261027_235042/report.md`.


- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261027_235042/run.yaml`
- Timestamp: `20261027_235042`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261027_235042`.

- tenferro-rs commit: `8fec1ca70852b930c092e7177c30bb8af7594a57`

### CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `enabled`

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


- Source table: `data/results/zen5-cpu/cpu/einsum/20261027_235042/einsum_table_t4_20261027_235042.md`

Logs:

- `data/results/zen5-cpu/cpu/einsum/20261027_235042/tenferro_trace_t4_20261027_235042.log`
- `data/results/zen5-cpu/cpu/einsum/20261027_235042/tenferro_eager_t4_20261027_235042.log`
- `data/results/zen5-cpu/cpu/einsum/20261027_235042/pytorch_cpu_t4_20261027_235042.log`
- `data/results/zen5-cpu/cpu/einsum/20261027_235042/jax_cpu_t4_20261027_235042.log`
- `data/results/zen5-cpu/cpu/einsum/20261027_235042/julia_omeinsum_t4_20261027_235042.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.246 ± 1.510 | 1.974 ± 0.025 | **0.600 ± 0.007** | 1.115 ± 0.274 | 4.824 ± 3.754 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.645 ± 0.309 | 0.763 ± 0.005 | **0.095 ± 0.001** | 0.434 ± 0.360 | 0.422 ± 0.700 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.741 ± 0.009 | 0.552 ± 0.005 | **0.029 ± 0.000** | 0.159 ± 0.096 | 1.423 ± 0.276 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.319 ± 0.005 | 0.542 ± 0.003 | **0.027 ± 0.001** | 0.211 ± 0.153 | 4.034 ± 3.592 |
| bin_elementwise_mul_2048x2048 | 1.929 ± 0.022 | **1.486 ± 0.484** | 3.150 ± 0.219 | 5.202 ± 0.613 | 7.306 ± 0.793 |
| bin_elementwise_mul_512x512 | 0.702 ± 0.004 | 0.465 ± 0.003 | **0.019 ± 0.001** | 0.222 ± 0.099 | 0.077 ± 0.244 |
| bin_matmul_1024 | 17.878 ± 0.589 | 16.251 ± 6.691 | 14.721 ± 0.170 | **8.802 ± 0.121** | 23.128 ± 12.476 |
| bin_matmul_256 | 0.899 ± 0.004 | **0.140 ± 0.001** | 0.351 ± 0.007 | 0.229 ± 0.176 | 0.338 ± 0.021 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.005 ± 0.000 | **0.003 ± 0.000** | 0.012 ± 0.001 | 0.064 ± 0.006 | 0.006 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.018 ± 0.000 | **0.017 ± 0.000** | 0.055 ± 0.001 | 0.105 ± 0.041 | 0.043 ± 0.012 |
| bin_omeinsum_matmul_10x10 | 0.010 ± 0.000 | **0.006 ± 0.000** | 0.011 ± 0.001 | 0.039 ± 0.092 | 0.012 ± 0.001 |
| bin_outer_product_4096 | 11.105 ± 3.773 | 11.441 ± 3.865 | **7.507 ± 0.442** | 12.945 ± 0.167 | 140.577 ± 4.374 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 17.571 ± 0.485 | 18.258 ± 7.728 | 12.732 ± 0.182 | **12.206 ± 0.389** | 19.839 ± 9.967 |
| gm_queen5_5_3.wcsp | 3647.344 ± 65.549 | 2379.539 ± 69.696 | **1375.702 ± 6.216** | 1736.151 ± 32.565 | 4581.752 ± 21.314 |
| lm_batch_likelihood_brackets_4_4d | 14.756 ± 0.047 | 19.000 ± 0.660 | **5.370 ± 0.050** | 10.050 ± 0.335 | 15.527 ± 5.423 |
| lm_batch_likelihood_sentence_3_12d | 27.434 ± 3.537 | 61.126 ± 48.272 | **16.965 ± 0.068** | 19.949 ± 1.095 | 58.235 ± 22.569 |
| lm_batch_likelihood_sentence_4_4d | 20.623 ± 5.836 | 34.835 ± 0.854 | **5.624 ± 0.065** | 10.030 ± 0.402 | 35.908 ± 19.399 |
| nary_matmul_chain_64 | 0.063 ± 0.001 | **0.016 ± 0.000** | 0.025 ± 0.001 | 0.051 ± 0.008 | 0.047 ± 0.032 |
| str_matrix_chain_multiplication_100 | 6.540 ± 0.020 | **3.382 ± 0.011** | 3.609 ± 0.021 | 5.322 ± 0.408 | 7.381 ± 0.617 |
| str_mps_varying_inner_product_200 | 12.694 ± 0.101 | 9.058 ± 0.106 | **8.135 ± 0.055** | 10.361 ± 0.245 | 18.246 ± 3.301 |
| str_nw_mera_closed_120 | 415.837 ± 3.099 | 440.974 ± 11.282 | 386.730 ± 1.625 | **357.625 ± 6.872** | 547.987 ± 15.014 |
| str_nw_mera_open_26 | 414.597 ± 4.057 | 466.773 ± 102.083 | **236.201 ± 1.339** | 332.863 ± 5.670 | 642.859 ± 13.094 |
| tensornetwork_permutation_focus_step409_316 | 212.276 ± 2.742 | 238.555 ± 20.734 | **147.208 ± 0.417** | 220.242 ± 3.315 | 273.073 ± 26.585 |
| tensornetwork_permutation_light_415 | 206.231 ± 27.970 | 207.306 ± 22.942 | **144.041 ± 0.368** | 227.021 ± 7.230 | 265.425 ± 15.746 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.246 ± 1.510 | 1.974 ± 0.025 | **0.600 ± 0.007** | 1.115 ± 0.274 | 4.824 ± 3.754 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.645 ± 0.309 | 0.763 ± 0.005 | **0.095 ± 0.001** | 0.434 ± 0.360 | 0.422 ± 0.700 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.741 ± 0.009 | 0.552 ± 0.005 | **0.029 ± 0.000** | 0.159 ± 0.096 | 1.423 ± 0.276 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.319 ± 0.005 | 0.542 ± 0.003 | **0.027 ± 0.001** | 0.211 ± 0.153 | 4.034 ± 3.592 |
| bin_elementwise_mul_2048x2048 | 1.929 ± 0.022 | **1.486 ± 0.484** | 3.150 ± 0.219 | 5.202 ± 0.613 | 7.306 ± 0.793 |
| bin_elementwise_mul_512x512 | 0.702 ± 0.004 | 0.465 ± 0.003 | **0.019 ± 0.001** | 0.222 ± 0.099 | 0.077 ± 0.244 |
| bin_matmul_1024 | 17.878 ± 0.589 | 16.251 ± 6.691 | 14.721 ± 0.170 | **8.802 ± 0.121** | 23.128 ± 12.476 |
| bin_matmul_256 | 0.899 ± 0.004 | **0.140 ± 0.001** | 0.351 ± 0.007 | 0.229 ± 0.176 | 0.338 ± 0.021 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.005 ± 0.000 | **0.003 ± 0.000** | 0.012 ± 0.001 | 0.064 ± 0.006 | 0.006 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.018 ± 0.000 | **0.017 ± 0.000** | 0.055 ± 0.001 | 0.105 ± 0.041 | 0.043 ± 0.012 |
| bin_omeinsum_matmul_10x10 | 0.010 ± 0.000 | **0.006 ± 0.000** | 0.011 ± 0.001 | 0.039 ± 0.092 | 0.012 ± 0.001 |
| bin_outer_product_4096 | 11.105 ± 3.773 | 11.441 ± 3.865 | **7.507 ± 0.442** | 12.945 ± 0.167 | 140.577 ± 4.374 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 17.571 ± 0.485 | 18.258 ± 7.728 | 12.732 ± 0.182 | **12.206 ± 0.389** | 19.839 ± 9.967 |
| gm_queen5_5_3.wcsp | 1121.566 ± 7.438 | 948.081 ± 153.408 | **578.710 ± 1.033** | 640.613 ± 8.390 | 1658.170 ± 15.339 |
| lm_batch_likelihood_brackets_4_4d | 18.926 ± 0.093 | 25.079 ± 1.216 | **5.397 ± 0.069** | 10.382 ± 0.433 | 30.606 ± 18.535 |
| lm_batch_likelihood_sentence_3_12d | 27.720 ± 0.321 | 109.431 ± 44.509 | 27.052 ± 0.432 | **25.227 ± 1.251** | 66.763 ± 22.698 |
| lm_batch_likelihood_sentence_4_4d | 12.901 ± 3.499 | 28.737 ± 3.395 | **6.202 ± 0.116** | 10.139 ± 0.423 | 28.104 ± 19.357 |
| nary_matmul_chain_64 | 0.063 ± 0.001 | **0.016 ± 0.000** | 0.025 ± 0.001 | 0.051 ± 0.008 | 0.047 ± 0.032 |
| str_matrix_chain_multiplication_100 | **3.203 ± 0.013** | 3.363 ± 0.025 | 3.608 ± 0.043 | 5.224 ± 0.118 | 7.341 ± 0.650 |
| str_mps_varying_inner_product_200 | 10.260 ± 0.044 | 9.826 ± 0.050 | **8.178 ± 0.021** | 9.884 ± 0.109 | 15.921 ± 5.393 |
| str_nw_mera_closed_120 | 383.280 ± 2.502 | 370.964 ± 6.333 | 293.342 ± 1.175 | **277.434 ± 2.290** | 461.747 ± 27.942 |
| str_nw_mera_open_26 | 411.038 ± 3.382 | 458.514 ± 99.271 | **245.024 ± 1.744** | 337.496 ± 6.789 | 631.238 ± 23.747 |
| tensornetwork_permutation_focus_step409_316 | 212.276 ± 2.742 | 238.555 ± 20.734 | **147.208 ± 0.417** | 220.242 ± 3.315 | 273.073 ± 26.585 |
| tensornetwork_permutation_light_415 | 206.231 ± 27.970 | 207.306 ± 22.942 | **144.041 ± 0.368** | 227.021 ± 7.230 | 265.425 ± 15.746 |

## Threads: 8

Source report: `data/results/zen5-cpu/cpu/einsum/20261028_001612/report.md`.


- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261028_001612/run.yaml`
- Timestamp: `20261028_001612`

Latest run: `./scripts/run_all.sh 8`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261028_001612`.

- tenferro-rs commit: `8fec1ca70852b930c092e7177c30bb8af7594a57`

### CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `enabled`

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

- Julia: version `1.13.3`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `8`

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


- Source table: `data/results/zen5-cpu/cpu/einsum/20261028_001612/einsum_table_t8_20261028_001612.md`

Logs:

- `data/results/zen5-cpu/cpu/einsum/20261028_001612/tenferro_trace_t8_20261028_001612.log`
- `data/results/zen5-cpu/cpu/einsum/20261028_001612/tenferro_eager_t8_20261028_001612.log`
- `data/results/zen5-cpu/cpu/einsum/20261028_001612/pytorch_cpu_t8_20261028_001612.log`
- `data/results/zen5-cpu/cpu/einsum/20261028_001612/jax_cpu_t8_20261028_001612.log`
- `data/results/zen5-cpu/cpu/einsum/20261028_001612/julia_omeinsum_t8_20261028_001612.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=8, OMP_NUM_THREADS=8, RAYON_NUM_THREADS=8.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.345 ± 0.968 | 2.072 ± 0.023 | **0.551 ± 0.015** | 1.064 ± 0.087 | 9.428 ± 10.707 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.897 ± 0.350 | 1.092 ± 0.006 | **0.080 ± 0.002** | 0.763 ± 0.219 | 0.768 ± 0.817 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 1.190 ± 0.416 | 0.791 ± 0.007 | **0.030 ± 0.001** | 0.438 ± 0.247 | 1.553 ± 1.404 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.839 ± 0.008 | 0.405 ± 0.006 | **0.030 ± 0.005** | 0.474 ± 0.268 | 4.301 ± 1.597 |
| bin_elementwise_mul_2048x2048 | **2.097 ± 0.037** | 2.155 ± 0.340 | 2.892 ± 0.277 | 5.322 ± 0.198 | 9.144 ± 0.522 |
| bin_elementwise_mul_512x512 | 2.448 ± 0.140 | 1.085 ± 0.397 | **0.021 ± 0.001** | 0.252 ± 0.201 | 0.116 ± 0.399 |
| bin_matmul_1024 | 18.409 ± 3.397 | 14.013 ± 1.839 | 12.585 ± 0.165 | **8.858 ± 0.394** | 30.558 ± 15.645 |
| bin_matmul_256 | 1.749 ± 0.300 | **0.111 ± 0.002** | 0.281 ± 0.006 | 0.218 ± 0.056 | 0.412 ± 0.006 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.005 ± 0.000** | 0.018 ± 0.001 | 0.103 ± 0.006 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.001 | **0.027 ± 0.001** | 0.088 ± 0.002 | 0.147 ± 0.074 | 0.038 ± 0.026 |
| bin_omeinsum_matmul_10x10 | 0.046 ± 0.001 | **0.009 ± 0.000** | 0.017 ± 0.000 | 0.063 ± 0.029 | 0.018 ± 0.002 |
| bin_outer_product_4096 | 9.392 ± 2.800 | 9.046 ± 0.626 | **7.454 ± 0.282** | 12.498 ± 0.468 | 197.616 ± 6.924 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 23.002 ± 0.384 | 15.224 ± 5.549 | 11.051 ± 0.247 | **9.936 ± 0.297** | 23.495 ± 6.107 |
| gm_queen5_5_3.wcsp | 4009.796 ± 39.209 | 2214.568 ± 49.833 | **1243.864 ± 12.744** | 1563.443 ± 27.959 | 5626.624 ± 34.861 |
| lm_batch_likelihood_brackets_4_4d | 21.416 ± 2.632 | 39.825 ± 8.985 | **6.353 ± 0.124** | 13.519 ± 0.457 | 50.697 ± 25.946 |
| lm_batch_likelihood_sentence_3_12d | 32.823 ± 0.397 | 125.616 ± 12.520 | 27.876 ± 0.155 | **22.639 ± 3.793** | 74.549 ± 27.278 |
| lm_batch_likelihood_sentence_4_4d | 21.564 ± 0.223 | 39.085 ± 3.573 | **6.656 ± 0.040** | 13.259 ± 0.177 | 44.301 ± 22.738 |
| nary_matmul_chain_64 | 0.093 ± 0.001 | 0.077 ± 0.001 | **0.031 ± 0.001** | 0.085 ± 0.010 | 0.080 ± 0.026 |
| str_matrix_chain_multiplication_100 | 7.690 ± 0.055 | 8.091 ± 0.048 | **3.861 ± 0.020** | 5.332 ± 0.152 | 8.560 ± 1.118 |
| str_mps_varying_inner_product_200 | 15.722 ± 6.264 | 13.648 ± 7.724 | **9.944 ± 0.035** | 14.521 ± 0.123 | 22.127 ± 2.937 |
| str_nw_mera_closed_120 | 418.191 ± 4.944 | 454.306 ± 16.615 | 357.773 ± 2.418 | **328.203 ± 16.327** | 565.853 ± 13.959 |
| str_nw_mera_open_26 | 449.497 ± 2.409 | 434.313 ± 27.184 | **222.787 ± 1.270** | 278.737 ± 3.608 | 705.791 ± 16.419 |
| tensornetwork_permutation_focus_step409_316 | 233.983 ± 15.852 | 274.506 ± 12.049 | **128.458 ± 0.885** | 277.987 ± 3.846 | 334.694 ± 38.522 |
| tensornetwork_permutation_light_415 | 218.170 ± 2.015 | 229.150 ± 2.323 | **126.197 ± 0.933** | 286.035 ± 6.460 | 326.374 ± 34.140 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=8, OMP_NUM_THREADS=8, RAYON_NUM_THREADS=8.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 3.345 ± 0.968 | 2.072 ± 0.023 | **0.551 ± 0.015** | 1.064 ± 0.087 | 9.428 ± 10.707 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.897 ± 0.350 | 1.092 ± 0.006 | **0.080 ± 0.002** | 0.763 ± 0.219 | 0.768 ± 0.817 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 1.190 ± 0.416 | 0.791 ± 0.007 | **0.030 ± 0.001** | 0.438 ± 0.247 | 1.553 ± 1.404 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.839 ± 0.008 | 0.405 ± 0.006 | **0.030 ± 0.005** | 0.474 ± 0.268 | 4.301 ± 1.597 |
| bin_elementwise_mul_2048x2048 | **2.097 ± 0.037** | 2.155 ± 0.340 | 2.892 ± 0.277 | 5.322 ± 0.198 | 9.144 ± 0.522 |
| bin_elementwise_mul_512x512 | 2.448 ± 0.140 | 1.085 ± 0.397 | **0.021 ± 0.001** | 0.252 ± 0.201 | 0.116 ± 0.399 |
| bin_matmul_1024 | 18.409 ± 3.397 | 14.013 ± 1.839 | 12.585 ± 0.165 | **8.858 ± 0.394** | 30.558 ± 15.645 |
| bin_matmul_256 | 1.749 ± 0.300 | **0.111 ± 0.002** | 0.281 ± 0.006 | 0.218 ± 0.056 | 0.412 ± 0.006 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.005 ± 0.000** | 0.018 ± 0.001 | 0.103 ± 0.006 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.001 | **0.027 ± 0.001** | 0.088 ± 0.002 | 0.147 ± 0.074 | 0.038 ± 0.026 |
| bin_omeinsum_matmul_10x10 | 0.046 ± 0.001 | **0.009 ± 0.000** | 0.017 ± 0.000 | 0.063 ± 0.029 | 0.018 ± 0.002 |
| bin_outer_product_4096 | 9.392 ± 2.800 | 9.046 ± 0.626 | **7.454 ± 0.282** | 12.498 ± 0.468 | 197.616 ± 6.924 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 23.002 ± 0.384 | 15.224 ± 5.549 | 11.051 ± 0.247 | **9.936 ± 0.297** | 23.495 ± 6.107 |
| gm_queen5_5_3.wcsp | 1227.358 ± 7.209 | 939.443 ± 178.223 | **510.419 ± 3.960** | 575.919 ± 11.914 | 1988.501 ± 81.016 |
| lm_batch_likelihood_brackets_4_4d | 27.041 ± 0.326 | 43.212 ± 0.660 | **5.971 ± 0.214** | 13.143 ± 0.296 | 35.960 ± 19.421 |
| lm_batch_likelihood_sentence_3_12d | 24.891 ± 1.440 | 149.707 ± 61.922 | **15.592 ± 0.142** | 26.671 ± 0.376 | 72.385 ± 25.219 |
| lm_batch_likelihood_sentence_4_4d | 30.016 ± 0.117 | 42.034 ± 12.687 | **6.729 ± 0.059** | 13.175 ± 0.373 | 35.087 ± 22.288 |
| nary_matmul_chain_64 | 0.093 ± 0.001 | 0.077 ± 0.001 | **0.031 ± 0.001** | 0.085 ± 0.010 | 0.080 ± 0.026 |
| str_matrix_chain_multiplication_100 | 8.085 ± 0.118 | 8.069 ± 0.224 | **3.851 ± 0.018** | 5.284 ± 0.161 | 8.224 ± 0.755 |
| str_mps_varying_inner_product_200 | 15.417 ± 5.640 | 17.712 ± 5.773 | **9.840 ± 0.021** | 14.167 ± 0.066 | 19.372 ± 5.262 |
| str_nw_mera_closed_120 | 384.184 ± 2.145 | 359.499 ± 6.435 | 262.954 ± 1.490 | **233.187 ± 8.547** | 468.168 ± 36.532 |
| str_nw_mera_open_26 | 455.975 ± 6.457 | 432.567 ± 40.124 | **224.839 ± 2.295** | 290.107 ± 7.740 | 706.364 ± 46.942 |
| tensornetwork_permutation_focus_step409_316 | 233.983 ± 15.852 | 274.506 ± 12.049 | **128.458 ± 0.885** | 277.987 ± 3.846 | 334.694 ± 38.522 |
| tensornetwork_permutation_light_415 | 218.170 ± 2.015 | 229.150 ± 2.323 | **126.197 ± 0.933** | 286.035 ± 6.460 | 326.374 ± 34.140 |

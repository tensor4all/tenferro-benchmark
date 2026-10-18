# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `c2ddccb14b89f8e9fa8a8fe3957367e32fb4e677`
- Benchmark commit: `e4b601dcd685fe1bafd45e6da97bc134f45c5049`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261018_180413/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261018_180413/run.yaml`
- Timestamp: `20261018_180413`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261018_180413`.

- tenferro-rs commit: `c2ddccb14b89f8e9fa8a8fe3957367e32fb4e677`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
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


- Source table: `data/results/amd-cpu/cpu/einsum/20261018_180413/einsum_table_t1_20261018_180413.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20261018_180413/tenferro_trace_t1_20261018_180413.log`
- `data/results/amd-cpu/cpu/einsum/20261018_180413/tenferro_eager_t1_20261018_180413.log`
- `data/results/amd-cpu/cpu/einsum/20261018_180413/pytorch_cpu_t1_20261018_180413.log`
- `data/results/amd-cpu/cpu/einsum/20261018_180413/jax_cpu_t1_20261018_180413.log`
- `data/results/amd-cpu/cpu/einsum/20261018_180413/julia_omeinsum_t1_20261018_180413.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.242 ± 0.561 | **3.818 ± 0.009** | 4.740 ± 0.112 | 4.262 ± 0.051 | 9.774 ± 5.773 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.820 ± 0.054 | **0.602 ± 0.024** | 0.693 ± 0.017 | 0.803 ± 0.018 | 0.904 ± 0.039 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.202 ± 0.013 | **0.200 ± 0.007** | 0.218 ± 0.052 | 0.574 ± 0.025 | 1.110 ± 1.021 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.410 ± 0.030 | 0.208 ± 0.006 | **0.202 ± 0.019** | 0.577 ± 0.026 | 3.602 ± 5.613 |
| bin_elementwise_mul_2048x2048 | 1.890 ± 0.037 | **1.584 ± 0.046** | 4.589 ± 0.075 | 5.151 ± 0.108 | 5.030 ± 0.134 |
| bin_elementwise_mul_512x512 | **0.123 ± 0.005** | 0.124 ± 0.009 | 0.140 ± 0.014 | 0.350 ± 0.027 | 0.154 ± 0.753 |
| bin_matmul_1024 | 66.176 ± 0.325 | **49.381 ± 0.387** | 76.513 ± 0.566 | 64.036 ± 2.163 | 75.014 ± 23.762 |
| bin_matmul_256 | 1.129 ± 0.020 | **0.835 ± 0.012** | 1.373 ± 0.025 | 1.187 ± 0.031 | 1.270 ± 0.014 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.006 ± 0.000** | 0.036 ± 0.005 | 0.203 ± 0.026 | 0.014 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.049 ± 0.003 | **0.044 ± 0.004** | 0.148 ± 0.011 | 0.232 ± 0.033 | 0.049 ± 0.002 |
| bin_omeinsum_matmul_10x10 | 0.026 ± 0.001 | **0.011 ± 0.000** | 0.034 ± 0.003 | 0.152 ± 0.029 | 0.024 ± 0.001 |
| bin_outer_product_4096 | 9.512 ± 0.165 | **9.270 ± 0.171** | 10.347 ± 0.197 | 9.903 ± 0.164 | 934.605 ± 5.708 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 65.382 ± 0.479 | **52.048 ± 0.312** | 61.191 ± 0.489 | 66.150 ± 4.883 | 71.056 ± 24.979 |
| gm_queen5_5_3.wcsp | 5265.924 ± 457.308 | 4423.209 ± 42.012 | **3740.283 ± 779.978** | 5135.539 ± 150.905 | 5095.780 ± 290.764 |
| lm_batch_likelihood_brackets_4_4d | 21.810 ± 0.318 | 32.257 ± 1.739 | 35.089 ± 0.742 | **14.330 ± 0.361** | 33.280 ± 8.830 |
| lm_batch_likelihood_sentence_3_12d | **52.655 ± 0.366** | 87.818 ± 1.631 | 64.643 ± 1.607 | 65.601 ± 1.497 | 100.991 ± 31.261 |
| lm_batch_likelihood_sentence_4_4d | 24.022 ± 0.175 | 36.878 ± 1.687 | 36.274 ± 0.893 | **16.151 ± 0.161** | 64.999 ± 24.856 |
| nary_matmul_chain_64 | 0.061 ± 0.002 | **0.054 ± 0.009** | 0.080 ± 0.005 | 0.213 ± 0.024 | 0.119 ± 0.022 |
| str_matrix_chain_multiplication_100 | **13.782 ± 0.156** | 14.942 ± 0.167 | 15.344 ± 0.172 | 14.629 ± 0.451 | 15.833 ± 0.492 |
| str_mps_varying_inner_product_200 | **19.426 ± 0.096** | 20.446 ± 0.304 | 23.925 ± 0.236 | 23.506 ± 0.386 | 25.333 ± 5.745 |
| str_nw_mera_closed_120 | **1314.975 ± 10.804** | 1332.467 ± 13.390 | 1386.160 ± 8.242 | 1877.878 ± 4.324 | 1344.705 ± 26.022 |
| str_nw_mera_open_26 | 897.745 ± 5.484 | 1010.786 ± 4.041 | **859.497 ± 1.662** | 1404.597 ± 3.187 | 1134.845 ± 40.469 |
| tensornetwork_permutation_focus_step409_316 | **307.424 ± 1.564** | 342.939 ± 4.275 | 734.804 ± 1.095 | 1544.970 ± 10.180 | 400.527 ± 14.338 |
| tensornetwork_permutation_light_415 | **300.014 ± 1.664** | 310.297 ± 2.327 | 697.217 ± 2.439 | 1551.955 ± 7.860 | 396.898 ± 78.025 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 7.242 ± 0.561 | **3.818 ± 0.009** | 4.740 ± 0.112 | 4.262 ± 0.051 | 9.774 ± 5.773 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.820 ± 0.054 | **0.602 ± 0.024** | 0.693 ± 0.017 | 0.803 ± 0.018 | 0.904 ± 0.039 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.202 ± 0.013 | **0.200 ± 0.007** | 0.218 ± 0.052 | 0.574 ± 0.025 | 1.110 ± 1.021 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.410 ± 0.030 | 0.208 ± 0.006 | **0.202 ± 0.019** | 0.577 ± 0.026 | 3.602 ± 5.613 |
| bin_elementwise_mul_2048x2048 | 1.890 ± 0.037 | **1.584 ± 0.046** | 4.589 ± 0.075 | 5.151 ± 0.108 | 5.030 ± 0.134 |
| bin_elementwise_mul_512x512 | **0.123 ± 0.005** | 0.124 ± 0.009 | 0.140 ± 0.014 | 0.350 ± 0.027 | 0.154 ± 0.753 |
| bin_matmul_1024 | 66.176 ± 0.325 | **49.381 ± 0.387** | 76.513 ± 0.566 | 64.036 ± 2.163 | 75.014 ± 23.762 |
| bin_matmul_256 | 1.129 ± 0.020 | **0.835 ± 0.012** | 1.373 ± 0.025 | 1.187 ± 0.031 | 1.270 ± 0.014 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.006 ± 0.000** | 0.036 ± 0.005 | 0.203 ± 0.026 | 0.014 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.049 ± 0.003 | **0.044 ± 0.004** | 0.148 ± 0.011 | 0.232 ± 0.033 | 0.049 ± 0.002 |
| bin_omeinsum_matmul_10x10 | 0.026 ± 0.001 | **0.011 ± 0.000** | 0.034 ± 0.003 | 0.152 ± 0.029 | 0.024 ± 0.001 |
| bin_outer_product_4096 | 9.512 ± 0.165 | **9.270 ± 0.171** | 10.347 ± 0.197 | 9.903 ± 0.164 | 934.605 ± 5.708 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 65.382 ± 0.479 | **52.048 ± 0.312** | 61.191 ± 0.489 | 66.150 ± 4.883 | 71.056 ± 24.979 |
| gm_queen5_5_3.wcsp | 1730.784 ± 9.209 | 1601.848 ± 15.168 | **1558.615 ± 5.058** | 1806.615 ± 8.054 | 1866.412 ± 50.507 |
| lm_batch_likelihood_brackets_4_4d | 22.234 ± 0.111 | 32.401 ± 0.813 | 35.563 ± 0.307 | **14.281 ± 0.310** | 47.283 ± 32.837 |
| lm_batch_likelihood_sentence_3_12d | **51.719 ± 0.356** | 96.439 ± 3.797 | 69.941 ± 0.999 | 70.941 ± 2.209 | 117.200 ± 36.943 |
| lm_batch_likelihood_sentence_4_4d | 23.144 ± 0.117 | 38.710 ± 1.940 | 37.664 ± 0.226 | **15.504 ± 0.259** | 47.637 ± 41.049 |
| nary_matmul_chain_64 | 0.061 ± 0.002 | **0.054 ± 0.009** | 0.080 ± 0.005 | 0.213 ± 0.024 | 0.119 ± 0.022 |
| str_matrix_chain_multiplication_100 | 14.894 ± 0.093 | 15.054 ± 0.174 | 15.428 ± 0.083 | **14.397 ± 0.294** | 16.488 ± 0.520 |
| str_mps_varying_inner_product_200 | **22.718 ± 0.186** | 22.856 ± 0.352 | 24.927 ± 0.129 | 22.992 ± 0.309 | 25.812 ± 9.831 |
| str_nw_mera_closed_120 | **1188.922 ± 2.204** | 1202.987 ± 10.978 | 1210.338 ± 1.933 | 1709.000 ± 5.571 | 1288.856 ± 39.222 |
| str_nw_mera_open_26 | 911.219 ± 6.006 | 1017.630 ± 4.169 | **883.824 ± 1.624** | 1443.856 ± 2.448 | 1235.182 ± 53.921 |
| tensornetwork_permutation_focus_step409_316 | **307.424 ± 1.564** | 342.939 ± 4.275 | 734.804 ± 1.095 | 1544.970 ± 10.180 | 400.527 ± 14.338 |
| tensornetwork_permutation_light_415 | **300.014 ± 1.664** | 310.297 ± 2.327 | 697.217 ± 2.439 | 1551.955 ± 7.860 | 396.898 ± 78.025 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261018_183622/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261018_183622/run.yaml`
- Timestamp: `20261018_183622`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261018_183622`.

- tenferro-rs commit: `c2ddccb14b89f8e9fa8a8fe3957367e32fb4e677`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
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


- Source table: `data/results/amd-cpu/cpu/einsum/20261018_183622/einsum_table_t4_20261018_183622.md`

Logs:

- `data/results/amd-cpu/cpu/einsum/20261018_183622/tenferro_trace_t4_20261018_183622.log`
- `data/results/amd-cpu/cpu/einsum/20261018_183622/tenferro_eager_t4_20261018_183622.log`
- `data/results/amd-cpu/cpu/einsum/20261018_183622/pytorch_cpu_t4_20261018_183622.log`
- `data/results/amd-cpu/cpu/einsum/20261018_183622/jax_cpu_t4_20261018_183622.log`
- `data/results/amd-cpu/cpu/einsum/20261018_183622/julia_omeinsum_t4_20261018_183622.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.395 ± 0.143 | 2.841 ± 0.018 | **2.387 ± 2.927** | 2.974 ± 0.431 | 11.613 ± 3.852 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.894 ± 0.054 | 0.672 ± 0.026 | **0.379 ± 0.016** | 0.807 ± 0.017 | 0.912 ± 1.159 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.260 ± 0.016 | 0.265 ± 0.022 | **0.184 ± 0.010** | 0.478 ± 0.019 | 1.676 ± 0.429 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.358 ± 0.022 | 0.262 ± 0.024 | **0.175 ± 0.013** | 0.472 ± 0.013 | 3.728 ± 3.662 |
| bin_elementwise_mul_2048x2048 | 1.401 ± 0.533 | **1.116 ± 0.030** | 2.942 ± 0.119 | 4.150 ± 0.148 | 5.253 ± 0.257 |
| bin_elementwise_mul_512x512 | 0.139 ± 0.015 | 0.133 ± 0.023 | **0.100 ± 0.002** | 0.419 ± 0.066 | 0.147 ± 0.801 |
| bin_matmul_1024 | 33.069 ± 0.572 | **31.127 ± 11.888** | 51.950 ± 1.227 | 36.563 ± 0.380 | 50.157 ± 20.179 |
| bin_matmul_256 | 0.684 ± 0.030 | **0.487 ± 0.029** | 0.871 ± 0.014 | 0.889 ± 0.038 | 1.921 ± 1.001 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.020 ± 0.001 | **0.009 ± 0.000** | 0.052 ± 0.004 | 0.203 ± 0.027 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.123 ± 0.012 | 0.089 ± 0.008 | 0.221 ± 0.014 | 0.273 ± 0.021 | **0.049 ± 0.002** |
| bin_omeinsum_matmul_10x10 | 0.074 ± 0.004 | **0.019 ± 0.001** | 0.048 ± 0.004 | 0.150 ± 0.018 | 0.025 ± 0.005 |
| bin_outer_product_4096 | 7.802 ± 0.406 | 12.913 ± 2.151 | **5.662 ± 0.282** | 6.572 ± 0.113 | 929.709 ± 1.799 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 30.826 ± 1.254 | **26.043 ± 0.677** | 32.309 ± 0.347 | 38.962 ± 1.107 | 47.765 ± 9.727 |
| gm_queen5_5_3.wcsp | 4114.625 ± 361.741 | 3632.569 ± 109.835 | **1644.762 ± 9.698** | 3629.114 ± 44.100 | 4400.582 ± 221.233 |
| lm_batch_likelihood_brackets_4_4d | 23.742 ± 2.084 | 42.840 ± 3.688 | **16.848 ± 0.591** | 22.493 ± 0.878 | 42.490 ± 19.704 |
| lm_batch_likelihood_sentence_3_12d | 35.377 ± 7.370 | 82.105 ± 7.257 | **32.036 ± 1.003** | 44.264 ± 2.252 | 98.514 ± 34.290 |
| lm_batch_likelihood_sentence_4_4d | 25.958 ± 2.270 | 44.758 ± 3.045 | **18.056 ± 0.183** | 23.024 ± 0.617 | 77.573 ± 37.442 |
| nary_matmul_chain_64 | 0.086 ± 0.005 | **0.057 ± 0.001** | 0.094 ± 0.003 | 0.223 ± 0.015 | 0.149 ± 0.050 |
| str_matrix_chain_multiplication_100 | **8.784 ± 0.064** | 9.495 ± 0.117 | 11.009 ± 0.179 | 15.719 ± 0.555 | 14.453 ± 0.682 |
| str_mps_varying_inner_product_200 | **23.082 ± 0.524** | 24.452 ± 0.471 | 25.719 ± 0.679 | 32.431 ± 1.243 | 30.545 ± 5.011 |
| str_nw_mera_closed_120 | **671.549 ± 23.820** | 697.064 ± 38.779 | 685.798 ± 7.694 | 916.334 ± 9.123 | 832.738 ± 17.677 |
| str_nw_mera_open_26 | 513.515 ± 13.030 | 563.854 ± 38.188 | **448.332 ± 6.377** | 680.436 ± 7.293 | 851.867 ± 28.527 |
| tensornetwork_permutation_focus_step409_316 | **226.914 ± 26.948** | 241.267 ± 26.123 | 268.423 ± 1.626 | 789.925 ± 5.465 | 423.522 ± 20.969 |
| tensornetwork_permutation_light_415 | **227.407 ± 36.925** | 245.158 ± 29.651 | 262.528 ± 1.835 | 806.911 ± 13.245 | 407.129 ± 43.692 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.395 ± 0.143 | 2.841 ± 0.018 | **2.387 ± 2.927** | 2.974 ± 0.431 | 11.613 ± 3.852 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.894 ± 0.054 | 0.672 ± 0.026 | **0.379 ± 0.016** | 0.807 ± 0.017 | 0.912 ± 1.159 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.260 ± 0.016 | 0.265 ± 0.022 | **0.184 ± 0.010** | 0.478 ± 0.019 | 1.676 ± 0.429 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.358 ± 0.022 | 0.262 ± 0.024 | **0.175 ± 0.013** | 0.472 ± 0.013 | 3.728 ± 3.662 |
| bin_elementwise_mul_2048x2048 | 1.401 ± 0.533 | **1.116 ± 0.030** | 2.942 ± 0.119 | 4.150 ± 0.148 | 5.253 ± 0.257 |
| bin_elementwise_mul_512x512 | 0.139 ± 0.015 | 0.133 ± 0.023 | **0.100 ± 0.002** | 0.419 ± 0.066 | 0.147 ± 0.801 |
| bin_matmul_1024 | 33.069 ± 0.572 | **31.127 ± 11.888** | 51.950 ± 1.227 | 36.563 ± 0.380 | 50.157 ± 20.179 |
| bin_matmul_256 | 0.684 ± 0.030 | **0.487 ± 0.029** | 0.871 ± 0.014 | 0.889 ± 0.038 | 1.921 ± 1.001 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.020 ± 0.001 | **0.009 ± 0.000** | 0.052 ± 0.004 | 0.203 ± 0.027 | 0.013 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.123 ± 0.012 | 0.089 ± 0.008 | 0.221 ± 0.014 | 0.273 ± 0.021 | **0.049 ± 0.002** |
| bin_omeinsum_matmul_10x10 | 0.074 ± 0.004 | **0.019 ± 0.001** | 0.048 ± 0.004 | 0.150 ± 0.018 | 0.025 ± 0.005 |
| bin_outer_product_4096 | 7.802 ± 0.406 | 12.913 ± 2.151 | **5.662 ± 0.282** | 6.572 ± 0.113 | 929.709 ± 1.799 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 30.826 ± 1.254 | **26.043 ± 0.677** | 32.309 ± 0.347 | 38.962 ± 1.107 | 47.765 ± 9.727 |
| gm_queen5_5_3.wcsp | 1190.636 ± 36.761 | 1072.376 ± 63.425 | **738.487 ± 7.792** | 1168.470 ± 16.323 | 1526.200 ± 32.476 |
| lm_batch_likelihood_brackets_4_4d | 22.557 ± 3.436 | 42.255 ± 1.952 | **17.893 ± 0.238** | 22.554 ± 1.529 | 53.701 ± 38.541 |
| lm_batch_likelihood_sentence_3_12d | 33.524 ± 5.774 | 87.087 ± 17.557 | **32.965 ± 0.495** | 49.720 ± 2.019 | 98.623 ± 34.348 |
| lm_batch_likelihood_sentence_4_4d | 22.751 ± 3.596 | 45.270 ± 2.150 | **18.635 ± 0.248** | 23.296 ± 1.673 | 56.024 ± 39.423 |
| nary_matmul_chain_64 | 0.086 ± 0.005 | **0.057 ± 0.001** | 0.094 ± 0.003 | 0.223 ± 0.015 | 0.149 ± 0.050 |
| str_matrix_chain_multiplication_100 | **9.239 ± 0.136** | 9.415 ± 0.191 | 11.041 ± 0.363 | 15.518 ± 0.580 | 14.229 ± 0.577 |
| str_mps_varying_inner_product_200 | 28.178 ± 0.792 | **27.523 ± 1.081** | 29.135 ± 0.386 | 32.955 ± 0.585 | 29.852 ± 6.555 |
| str_nw_mera_closed_120 | 629.665 ± 25.008 | **608.275 ± 17.980** | 623.881 ± 4.368 | 843.752 ± 8.986 | 707.547 ± 58.891 |
| str_nw_mera_open_26 | 519.436 ± 14.106 | 585.457 ± 10.752 | **454.636 ± 4.283** | 704.361 ± 8.942 | 844.664 ± 16.587 |
| tensornetwork_permutation_focus_step409_316 | **226.914 ± 26.948** | 241.267 ± 26.123 | 268.423 ± 1.626 | 789.925 ± 5.465 | 423.522 ± 20.969 |
| tensornetwork_permutation_light_415 | **227.407 ± 36.925** | 245.158 ± 29.651 | 262.528 ± 1.835 | 806.911 ± 13.245 | 407.129 ± 43.692 |

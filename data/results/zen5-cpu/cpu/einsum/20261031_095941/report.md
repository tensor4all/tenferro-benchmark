# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261031_095941/run.yaml`
- Timestamp: `2026-10-31T00:59:41.343309+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261031_095941`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

## CPU Information

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

## Thread Environment

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

## Tenferro CPU BLAS Backend

- tenferro-rs features: `blas-openblas`
- BLAS implementation: `openblas`
- BLAS version: `unknown`
- BLAS root: `/usr`
- BLAS library: `/usr`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Julia / OMEinsum.jl Backend

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

## Threads: 4

Logs:

- `data/results/zen5-cpu/cpu/einsum/20261031_095941/jax_cpu_t4_20261031_095941.log`
- `data/results/zen5-cpu/cpu/einsum/20261031_095941/julia_omeinsum_t4_20261031_095941.log`
- `data/results/zen5-cpu/cpu/einsum/20261031_095941/pytorch_cpu_t4_20261031_095941.log`
- `data/results/zen5-cpu/cpu/einsum/20261031_095941/tenferro_eager_t4_20261031_095941.log`
- `data/results/zen5-cpu/cpu/einsum/20261031_095941/tenferro_trace_t4_20261031_095941.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 13.366 ± 1.897 | 1.627 ± 0.006 | **1.523 ± 3.967** | 2.177 ± 0.570 | 8.707 ± 6.320 |
| bin_batched_matmul_b32_m64_n64_k64 | 2.996 ± 2.362 | 0.547 ± 0.005 | **0.242 ± 0.005** | 1.024 ± 0.022 | 0.953 ± 1.294 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.245 ± 0.005 | 0.242 ± 0.004 | **0.068 ± 0.001** | 0.407 ± 0.134 | 0.779 ± 2.236 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.200 ± 0.002 | 0.242 ± 0.004 | **0.069 ± 0.003** | 0.376 ± 0.148 | 5.180 ± 4.930 |
| bin_elementwise_mul_2048x2048 | 0.797 ± 0.104 | **0.784 ± 0.011** | 3.838 ± 0.472 | 7.821 ± 0.190 | 12.998 ± 0.851 |
| bin_elementwise_mul_512x512 | 0.043 ± 0.001 | **0.042 ± 0.008** | 0.052 ± 0.001 | 0.253 ± 0.037 | 0.122 ± 0.420 |
| bin_matmul_1024 | 33.995 ± 1.031 | 20.332 ± 0.032 | 34.800 ± 0.787 | **20.196 ± 0.171** | 48.768 ± 18.013 |
| bin_matmul_256 | 12.999 ± 0.018 | **0.350 ± 0.006** | 0.908 ± 0.005 | 0.625 ± 0.111 | 1.344 ± 0.506 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.007 ± 0.000** | 0.030 ± 0.002 | 0.166 ± 0.008 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.037 ± 0.001 | **0.034 ± 0.001** | 0.141 ± 0.001 | 0.240 ± 0.069 | 0.051 ± 0.003 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.001 | **0.011 ± 0.000** | 0.029 ± 0.001 | 0.100 ± 0.020 | 0.028 ± 0.002 |
| bin_outer_product_4096 | 13.806 ± 0.337 | 14.731 ± 0.553 | **11.868 ± 0.506** | 22.956 ± 0.282 | 298.306 ± 2.707 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 40.992 ± 8.988 | **23.256 ± 0.053** | 26.008 ± 0.366 | 23.529 ± 2.282 | 44.715 ± 18.378 |
| gm_queen5_5_3.wcsp | 4908.819 ± 28.607 | 4052.216 ± 52.415 | **2220.712 ± 10.670** | 3058.000 ± 53.575 | 7249.728 ± 60.588 |
| lm_batch_likelihood_brackets_4_4d | 133.932 ± 53.959 | 153.845 ± 47.987 | **11.131 ± 0.096** | 20.866 ± 0.500 | 52.036 ± 23.023 |
| lm_batch_likelihood_sentence_3_12d | 133.003 ± 8.941 | 205.894 ± 57.971 | 43.470 ± 0.584 | **35.962 ± 5.038** | 104.312 ± 19.458 |
| lm_batch_likelihood_sentence_4_4d | 148.001 ± 58.938 | 173.719 ± 48.980 | **11.977 ± 0.113** | 21.408 ± 0.460 | 44.116 ± 25.687 |
| nary_matmul_chain_64 | 0.055 ± 0.001 | **0.043 ± 0.000** | 0.061 ± 0.002 | 0.133 ± 0.021 | 0.104 ± 0.018 |
| str_matrix_chain_multiplication_100 | 7.771 ± 0.076 | 7.852 ± 0.067 | **7.498 ± 0.037** | 11.072 ± 0.467 | 12.174 ± 6.596 |
| str_mps_varying_inner_product_200 | **17.327 ± 0.039** | 17.673 ± 0.189 | 17.946 ± 0.137 | 22.382 ± 0.228 | 29.351 ± 19.938 |
| str_nw_mera_closed_120 | 828.930 ± 7.598 | 959.067 ± 19.813 | **706.684 ± 1.645** | 743.015 ± 9.132 | 1043.603 ± 36.296 |
| str_nw_mera_open_26 | 736.217 ± 13.696 | 703.703 ± 50.718 | **440.450 ± 2.119** | 603.403 ± 4.420 | 1022.814 ± 48.186 |
| tensornetwork_permutation_focus_step409_316 | 367.195 ± 2.241 | 417.000 ± 4.947 | **300.225 ± 1.843** | 361.784 ± 10.889 | 472.369 ± 22.357 |
| tensornetwork_permutation_light_415 | 373.907 ± 4.934 | 422.064 ± 85.037 | **290.330 ± 0.990** | 382.253 ± 8.619 | 458.864 ± 39.930 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 13.366 ± 1.897 | 1.627 ± 0.006 | **1.523 ± 3.967** | 2.177 ± 0.570 | 8.707 ± 6.320 |
| bin_batched_matmul_b32_m64_n64_k64 | 2.996 ± 2.362 | 0.547 ± 0.005 | **0.242 ± 0.005** | 1.024 ± 0.022 | 0.953 ± 1.294 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.245 ± 0.005 | 0.242 ± 0.004 | **0.068 ± 0.001** | 0.407 ± 0.134 | 0.779 ± 2.236 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.200 ± 0.002 | 0.242 ± 0.004 | **0.069 ± 0.003** | 0.376 ± 0.148 | 5.180 ± 4.930 |
| bin_elementwise_mul_2048x2048 | 0.797 ± 0.104 | **0.784 ± 0.011** | 3.838 ± 0.472 | 7.821 ± 0.190 | 12.998 ± 0.851 |
| bin_elementwise_mul_512x512 | 0.043 ± 0.001 | **0.042 ± 0.008** | 0.052 ± 0.001 | 0.253 ± 0.037 | 0.122 ± 0.420 |
| bin_matmul_1024 | 33.995 ± 1.031 | 20.332 ± 0.032 | 34.800 ± 0.787 | **20.196 ± 0.171** | 48.768 ± 18.013 |
| bin_matmul_256 | 12.999 ± 0.018 | **0.350 ± 0.006** | 0.908 ± 0.005 | 0.625 ± 0.111 | 1.344 ± 0.506 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.007 ± 0.000** | 0.030 ± 0.002 | 0.166 ± 0.008 | 0.016 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.037 ± 0.001 | **0.034 ± 0.001** | 0.141 ± 0.001 | 0.240 ± 0.069 | 0.051 ± 0.003 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.001 | **0.011 ± 0.000** | 0.029 ± 0.001 | 0.100 ± 0.020 | 0.028 ± 0.002 |
| bin_outer_product_4096 | 13.806 ± 0.337 | 14.731 ± 0.553 | **11.868 ± 0.506** | 22.956 ± 0.282 | 298.306 ± 2.707 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 40.992 ± 8.988 | **23.256 ± 0.053** | 26.008 ± 0.366 | 23.529 ± 2.282 | 44.715 ± 18.378 |
| gm_queen5_5_3.wcsp | 1620.167 ± 9.067 | 1428.317 ± 25.424 | **971.705 ± 3.755** | 1170.944 ± 13.351 | 2645.499 ± 32.526 |
| lm_batch_likelihood_brackets_4_4d | 147.006 ± 84.032 | 209.852 ± 82.744 | **11.887 ± 0.120** | 21.170 ± 0.469 | 61.989 ± 25.447 |
| lm_batch_likelihood_sentence_3_12d | 102.996 ± 6.957 | 202.229 ± 80.095 | **25.257 ± 0.059** | 42.460 ± 4.629 | 97.925 ± 40.697 |
| lm_batch_likelihood_sentence_4_4d | 143.967 ± 11.017 | 160.666 ± 35.511 | **12.765 ± 0.112** | 20.840 ± 0.344 | 42.403 ± 24.503 |
| nary_matmul_chain_64 | 0.055 ± 0.001 | **0.043 ± 0.000** | 0.061 ± 0.002 | 0.133 ± 0.021 | 0.104 ± 0.018 |
| str_matrix_chain_multiplication_100 | 7.918 ± 0.020 | 7.959 ± 0.027 | **7.436 ± 0.038** | 10.864 ± 0.454 | 12.232 ± 0.470 |
| str_mps_varying_inner_product_200 | 18.886 ± 0.138 | 17.961 ± 0.081 | **17.697 ± 0.076** | 21.714 ± 0.200 | 24.745 ± 5.872 |
| str_nw_mera_closed_120 | 818.914 ± 4.029 | 792.854 ± 8.470 | **577.232 ± 3.672** | 594.969 ± 14.222 | 874.685 ± 39.077 |
| str_nw_mera_open_26 | 747.564 ± 5.431 | 632.986 ± 54.506 | **447.861 ± 3.268** | 629.270 ± 4.886 | 1064.046 ± 29.598 |
| tensornetwork_permutation_focus_step409_316 | 367.195 ± 2.241 | 417.000 ± 4.947 | **300.225 ± 1.843** | 361.784 ± 10.889 | 472.369 ± 22.357 |
| tensornetwork_permutation_light_415 | 373.907 ± 4.934 | 422.064 ± 85.037 | **290.330 ± 0.990** | 382.253 ± 8.619 | 458.864 ± 39.930 |

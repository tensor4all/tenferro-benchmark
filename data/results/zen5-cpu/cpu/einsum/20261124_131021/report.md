# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_131021/run.yaml`
- Timestamp: `2026-11-24T04:10:21.867323+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261124_131021`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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
- Python platform: `Linux-7.0.0-43-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `disabled`
- Container: `none`

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

- tenferro-rs features: `native`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
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

- `data/results/zen5-cpu/cpu/einsum/20261124_131021/jax_cpu_t4_20261124_131021.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_131021/julia_omeinsum_t4_20261124_131021.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_131021/pytorch_cpu_t4_20261124_131021.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_131021/tenferro_eager_t4_20261124_131021.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_131021/tenferro_trace_t4_20261124_131021.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.772 ± 0.011 | **1.469 ± 0.004** | 1.664 ± 1.854 | 2.129 ± 0.259 | 11.198 ± 22.393 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.478 ± 0.005 | 0.365 ± 0.004 | **0.270 ± 0.005** | 0.926 ± 0.112 | 0.980 ± 1.257 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.243 ± 0.003 | 0.241 ± 0.004 | **0.070 ± 0.002** | 0.421 ± 0.195 | 2.517 ± 1.372 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.201 ± 0.006 | 0.243 ± 0.005 | **0.071 ± 0.002** | 0.521 ± 0.055 | 5.326 ± 3.707 |
| bin_elementwise_mul_2048x2048 | 0.807 ± 0.111 | **0.805 ± 0.152** | 4.049 ± 0.419 | 7.836 ± 0.464 | 12.971 ± 1.029 |
| bin_elementwise_mul_512x512 | 0.041 ± 0.002 | **0.038 ± 0.001** | 0.052 ± 0.002 | 0.394 ± 0.136 | 0.119 ± 0.530 |
| bin_matmul_1024 | 19.481 ± 0.140 | **17.328 ± 0.050** | 34.446 ± 0.185 | 20.393 ± 0.221 | 51.157 ± 11.884 |
| bin_matmul_256 | 0.369 ± 0.005 | **0.298 ± 0.004** | 0.897 ± 0.005 | 0.620 ± 0.126 | 0.837 ± 0.498 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.013 ± 0.000 | **0.008 ± 0.000** | 0.029 ± 0.001 | 0.168 ± 0.014 | 0.023 ± 0.008 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.001 | **0.029 ± 0.001** | 0.143 ± 0.002 | 0.234 ± 0.056 | 0.077 ± 0.016 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.001 | **0.012 ± 0.000** | 0.029 ± 0.001 | 0.100 ± 0.011 | 0.028 ± 0.003 |
| bin_outer_product_4096 | 14.264 ± 0.567 | 14.449 ± 0.426 | **12.228 ± 0.266** | 23.147 ± 0.347 | 299.938 ± 4.227 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 19.917 ± 0.094 | **18.438 ± 0.062** | 26.034 ± 0.906 | 24.005 ± 0.979 | 44.625 ± 12.821 |
| gm_queen5_5_3.wcsp | 2821.935 ± 25.228 | 2785.921 ± 69.682 | **2283.418 ± 13.351** | 3050.912 ± 38.183 | 7427.256 ± 129.092 |
| lm_batch_likelihood_brackets_4_4d | 12.752 ± 0.177 | 14.812 ± 0.331 | **11.122 ± 0.239** | 21.721 ± 1.005 | 29.208 ± 18.938 |
| lm_batch_likelihood_sentence_3_12d | **21.986 ± 0.306** | 39.014 ± 0.593 | 43.634 ± 0.724 | 36.252 ± 1.686 | 85.459 ± 23.056 |
| lm_batch_likelihood_sentence_4_4d | 13.133 ± 0.306 | 14.789 ± 0.361 | **11.785 ± 0.170** | 21.400 ± 0.484 | 42.687 ± 24.934 |
| nary_matmul_chain_64 | 0.042 ± 0.001 | **0.033 ± 0.001** | 0.062 ± 0.001 | 0.134 ± 0.021 | 0.099 ± 0.036 |
| str_matrix_chain_multiplication_100 | **6.305 ± 0.199** | 6.397 ± 0.171 | 7.567 ± 0.013 | 10.970 ± 0.873 | 12.514 ± 6.547 |
| str_mps_varying_inner_product_200 | 10.908 ± 0.714 | **10.764 ± 0.157** | 18.033 ± 0.116 | 22.906 ± 0.416 | 28.434 ± 25.086 |
| str_nw_mera_closed_120 | **543.669 ± 1.647** | 557.848 ± 3.823 | 723.022 ± 6.224 | 741.762 ± 12.666 | 1049.017 ± 34.205 |
| str_nw_mera_open_26 | 472.548 ± 5.518 | 516.236 ± 7.827 | **447.467 ± 11.046** | 613.193 ± 12.826 | 1070.026 ± 38.566 |
| tensornetwork_permutation_focus_step409_316 | **147.932 ± 2.348** | 175.751 ± 1.507 | 307.124 ± 14.548 | 371.532 ± 11.717 | 481.585 ± 36.299 |
| tensornetwork_permutation_light_415 | 153.585 ± 4.040 | **125.953 ± 1.631** | 295.661 ± 4.940 | 378.906 ± 11.692 | 490.835 ± 32.409 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.772 ± 0.011 | **1.469 ± 0.004** | 1.664 ± 1.854 | 2.129 ± 0.259 | 11.198 ± 22.393 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.478 ± 0.005 | 0.365 ± 0.004 | **0.270 ± 0.005** | 0.926 ± 0.112 | 0.980 ± 1.257 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.243 ± 0.003 | 0.241 ± 0.004 | **0.070 ± 0.002** | 0.421 ± 0.195 | 2.517 ± 1.372 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.201 ± 0.006 | 0.243 ± 0.005 | **0.071 ± 0.002** | 0.521 ± 0.055 | 5.326 ± 3.707 |
| bin_elementwise_mul_2048x2048 | 0.807 ± 0.111 | **0.805 ± 0.152** | 4.049 ± 0.419 | 7.836 ± 0.464 | 12.971 ± 1.029 |
| bin_elementwise_mul_512x512 | 0.041 ± 0.002 | **0.038 ± 0.001** | 0.052 ± 0.002 | 0.394 ± 0.136 | 0.119 ± 0.530 |
| bin_matmul_1024 | 19.481 ± 0.140 | **17.328 ± 0.050** | 34.446 ± 0.185 | 20.393 ± 0.221 | 51.157 ± 11.884 |
| bin_matmul_256 | 0.369 ± 0.005 | **0.298 ± 0.004** | 0.897 ± 0.005 | 0.620 ± 0.126 | 0.837 ± 0.498 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.013 ± 0.000 | **0.008 ± 0.000** | 0.029 ± 0.001 | 0.168 ± 0.014 | 0.023 ± 0.008 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.030 ± 0.001 | **0.029 ± 0.001** | 0.143 ± 0.002 | 0.234 ± 0.056 | 0.077 ± 0.016 |
| bin_omeinsum_matmul_10x10 | 0.023 ± 0.001 | **0.012 ± 0.000** | 0.029 ± 0.001 | 0.100 ± 0.011 | 0.028 ± 0.003 |
| bin_outer_product_4096 | 14.264 ± 0.567 | 14.449 ± 0.426 | **12.228 ± 0.266** | 23.147 ± 0.347 | 299.938 ± 4.227 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 19.917 ± 0.094 | **18.438 ± 0.062** | 26.034 ± 0.906 | 24.005 ± 0.979 | 44.625 ± 12.821 |
| gm_queen5_5_3.wcsp | **727.270 ± 6.620** | 781.110 ± 8.790 | 992.582 ± 4.394 | 1177.189 ± 18.986 | 2705.795 ± 33.229 |
| lm_batch_likelihood_brackets_4_4d | 12.939 ± 0.485 | 14.138 ± 0.323 | **12.022 ± 0.311** | 21.439 ± 0.527 | 52.688 ± 25.679 |
| lm_batch_likelihood_sentence_3_12d | **24.807 ± 0.373** | 39.826 ± 0.322 | 25.863 ± 0.520 | 43.057 ± 1.984 | 101.623 ± 31.290 |
| lm_batch_likelihood_sentence_4_4d | **11.796 ± 0.098** | 14.775 ± 0.152 | 12.874 ± 0.286 | 21.306 ± 0.260 | 45.188 ± 35.212 |
| nary_matmul_chain_64 | 0.042 ± 0.001 | **0.033 ± 0.001** | 0.062 ± 0.001 | 0.134 ± 0.021 | 0.099 ± 0.036 |
| str_matrix_chain_multiplication_100 | 6.528 ± 0.060 | **6.365 ± 0.126** | 7.498 ± 0.065 | 11.229 ± 0.441 | 12.611 ± 1.600 |
| str_mps_varying_inner_product_200 | 13.201 ± 0.110 | **11.179 ± 0.496** | 17.854 ± 0.256 | 22.005 ± 0.295 | 24.537 ± 6.290 |
| str_nw_mera_closed_120 | 568.491 ± 2.306 | **529.873 ± 5.530** | 592.986 ± 4.374 | 592.807 ± 7.552 | 877.491 ± 47.914 |
| str_nw_mera_open_26 | 475.198 ± 3.648 | 515.173 ± 2.620 | **453.967 ± 8.603** | 628.130 ± 5.710 | 1068.171 ± 20.878 |
| tensornetwork_permutation_focus_step409_316 | **147.932 ± 2.348** | 175.751 ± 1.507 | 307.124 ± 14.548 | 371.532 ± 11.717 | 481.585 ± 36.299 |
| tensornetwork_permutation_light_415 | 153.585 ± 4.040 | **125.953 ± 1.631** | 295.661 ± 4.940 | 378.906 ± 11.692 | 490.835 ± 32.409 |

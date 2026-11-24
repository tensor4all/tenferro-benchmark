# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_135202/run.yaml`
- Timestamp: `2026-11-24T04:52:02.812988+00:00`

Latest run: `./scripts/run_all.sh 8`.

This file is generated from one suite run under `data/results/zen5-cpu/cpu/einsum/20261124_135202`.

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

## Tenferro CPU BLAS Backend

- tenferro-rs features: `native`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Julia / OMEinsum.jl Backend

- Julia: version `1.13.1`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `8`

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

## Threads: 8

Logs:

- `data/results/zen5-cpu/cpu/einsum/20261124_135202/jax_cpu_t8_20261124_135202.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_135202/julia_omeinsum_t8_20261124_135202.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_135202/pytorch_cpu_t8_20261124_135202.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_135202/tenferro_eager_t8_20261124_135202.log`
- `data/results/zen5-cpu/cpu/einsum/20261124_135202/tenferro_trace_t8_20261124_135202.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=8, OMP_NUM_THREADS=8, RAYON_NUM_THREADS=8.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.498 ± 0.016 | **1.065 ± 0.014** | 5.997 ± 0.009 | 1.468 ± 0.047 | 11.750 ± 5.936 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.473 ± 0.004 | **0.312 ± 0.008** | 5.999 ± 0.016 | 1.040 ± 0.027 | 1.065 ± 0.070 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.131 ± 0.028 | 0.129 ± 0.013 | **0.049 ± 0.001** | 0.550 ± 0.218 | 0.947 ± 0.377 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.109 ± 0.000 | 0.128 ± 0.001 | **0.049 ± 0.001** | 0.575 ± 0.199 | 3.460 ± 1.273 |
| bin_elementwise_mul_2048x2048 | **1.114 ± 0.269** | 1.215 ± 0.204 | 3.087 ± 0.292 | 6.158 ± 0.374 | 9.863 ± 6.715 |
| bin_elementwise_mul_512x512 | 0.026 ± 0.001 | **0.021 ± 0.001** | 0.037 ± 0.003 | 0.320 ± 0.192 | 0.143 ± 0.021 |
| bin_matmul_1024 | 10.619 ± 0.081 | **8.906 ± 0.017** | 18.822 ± 0.145 | 13.157 ± 1.436 | 43.805 ± 20.307 |
| bin_matmul_256 | 0.243 ± 0.006 | **0.169 ± 0.002** | 0.460 ± 0.005 | 0.429 ± 0.144 | 0.680 ± 0.006 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.001 | **0.008 ± 0.000** | 0.029 ± 0.001 | 0.167 ± 0.010 | 0.015 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.030 ± 0.001** | **0.030 ± 0.001** | 0.143 ± 0.002 | 0.228 ± 0.052 | 0.053 ± 0.002 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.001 | **0.012 ± 0.000** | 0.028 ± 0.001 | 0.100 ± 0.015 | 0.028 ± 0.002 |
| bin_outer_product_4096 | 9.065 ± 0.308 | 9.056 ± 0.333 | **8.491 ± 0.340** | 16.531 ± 0.271 | 298.982 ± 3.450 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 11.698 ± 0.068 | **10.361 ± 0.007** | 14.862 ± 0.189 | 13.548 ± 0.633 | 33.551 ± 16.843 |
| gm_queen5_5_3.wcsp | 2399.922 ± 129.566 | 1749.726 ± 22.292 | **1584.361 ± 8.948** | 2074.625 ± 20.812 | 7512.527 ± 83.772 |
| lm_batch_likelihood_brackets_4_4d | 10.262 ± 0.079 | 12.291 ± 0.136 | **8.805 ± 0.206** | 19.681 ± 0.737 | 31.868 ± 14.230 |
| lm_batch_likelihood_sentence_3_12d | **15.284 ± 0.401** | 20.813 ± 0.300 | 33.369 ± 0.455 | 31.697 ± 6.889 | 92.496 ± 19.234 |
| lm_batch_likelihood_sentence_4_4d | 11.113 ± 0.110 | 12.247 ± 0.114 | **9.252 ± 1.014** | 19.313 ± 0.171 | 42.297 ± 33.111 |
| nary_matmul_chain_64 | 0.040 ± 0.001 | **0.030 ± 0.000** | 0.051 ± 0.002 | 0.135 ± 0.008 | 0.110 ± 0.038 |
| str_matrix_chain_multiplication_100 | 5.777 ± 0.038 | 5.747 ± 0.022 | **5.564 ± 0.076** | 7.659 ± 0.313 | 11.510 ± 5.521 |
| str_mps_varying_inner_product_200 | 10.372 ± 0.144 | **10.302 ± 0.355** | 15.560 ± 0.163 | 22.207 ± 0.174 | 29.292 ± 17.779 |
| str_nw_mera_closed_120 | **337.421 ± 2.897** | 386.004 ± 2.369 | 488.238 ± 9.677 | 453.802 ± 13.712 | 811.968 ± 46.228 |
| str_nw_mera_open_26 | 343.211 ± 1.595 | 374.407 ± 6.475 | **301.721 ± 8.519** | 362.850 ± 7.202 | 890.828 ± 27.734 |
| tensornetwork_permutation_focus_step409_316 | **97.679 ± 0.965** | 189.167 ± 5.304 | 183.995 ± 2.217 | 360.322 ± 9.275 | 462.313 ± 49.129 |
| tensornetwork_permutation_light_415 | **100.743 ± 1.113** | 137.980 ± 1.399 | 181.555 ± 1.428 | 443.780 ± 164.762 | 427.616 ± 51.157 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=8, OMP_NUM_THREADS=8, RAYON_NUM_THREADS=8.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.498 ± 0.016 | **1.065 ± 0.014** | 5.997 ± 0.009 | 1.468 ± 0.047 | 11.750 ± 5.936 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.473 ± 0.004 | **0.312 ± 0.008** | 5.999 ± 0.016 | 1.040 ± 0.027 | 1.065 ± 0.070 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.131 ± 0.028 | 0.129 ± 0.013 | **0.049 ± 0.001** | 0.550 ± 0.218 | 0.947 ± 0.377 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.109 ± 0.000 | 0.128 ± 0.001 | **0.049 ± 0.001** | 0.575 ± 0.199 | 3.460 ± 1.273 |
| bin_elementwise_mul_2048x2048 | **1.114 ± 0.269** | 1.215 ± 0.204 | 3.087 ± 0.292 | 6.158 ± 0.374 | 9.863 ± 6.715 |
| bin_elementwise_mul_512x512 | 0.026 ± 0.001 | **0.021 ± 0.001** | 0.037 ± 0.003 | 0.320 ± 0.192 | 0.143 ± 0.021 |
| bin_matmul_1024 | 10.619 ± 0.081 | **8.906 ± 0.017** | 18.822 ± 0.145 | 13.157 ± 1.436 | 43.805 ± 20.307 |
| bin_matmul_256 | 0.243 ± 0.006 | **0.169 ± 0.002** | 0.460 ± 0.005 | 0.429 ± 0.144 | 0.680 ± 0.006 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.001 | **0.008 ± 0.000** | 0.029 ± 0.001 | 0.167 ± 0.010 | 0.015 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.030 ± 0.001** | **0.030 ± 0.001** | 0.143 ± 0.002 | 0.228 ± 0.052 | 0.053 ± 0.002 |
| bin_omeinsum_matmul_10x10 | 0.024 ± 0.001 | **0.012 ± 0.000** | 0.028 ± 0.001 | 0.100 ± 0.015 | 0.028 ± 0.002 |
| bin_outer_product_4096 | 9.065 ± 0.308 | 9.056 ± 0.333 | **8.491 ± 0.340** | 16.531 ± 0.271 | 298.982 ± 3.450 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 11.698 ± 0.068 | **10.361 ± 0.007** | 14.862 ± 0.189 | 13.548 ± 0.633 | 33.551 ± 16.843 |
| gm_queen5_5_3.wcsp | 577.333 ± 5.209 | **514.001 ± 9.459** | 669.535 ± 4.912 | 805.819 ± 16.083 | 2635.690 ± 56.795 |
| lm_batch_likelihood_brackets_4_4d | 10.565 ± 0.148 | 11.628 ± 0.094 | **8.367 ± 0.434** | 19.261 ± 0.431 | 70.851 ± 42.284 |
| lm_batch_likelihood_sentence_3_12d | **15.575 ± 0.082** | 30.165 ± 0.226 | 18.596 ± 0.081 | 37.776 ± 2.739 | 95.772 ± 40.057 |
| lm_batch_likelihood_sentence_4_4d | 9.751 ± 0.083 | 12.656 ± 1.004 | **9.108 ± 0.647** | 18.974 ± 0.295 | 46.292 ± 34.753 |
| nary_matmul_chain_64 | 0.040 ± 0.001 | **0.030 ± 0.000** | 0.051 ± 0.002 | 0.135 ± 0.008 | 0.110 ± 0.038 |
| str_matrix_chain_multiplication_100 | **5.549 ± 0.401** | 5.692 ± 0.150 | 5.581 ± 0.067 | 7.716 ± 0.151 | 11.603 ± 0.403 |
| str_mps_varying_inner_product_200 | 13.458 ± 0.082 | **11.433 ± 0.199** | 15.039 ± 0.058 | 21.478 ± 0.317 | 25.839 ± 5.707 |
| str_nw_mera_closed_120 | 428.966 ± 2.775 | 380.753 ± 3.957 | 369.825 ± 3.714 | **348.774 ± 7.352** | 644.892 ± 36.744 |
| str_nw_mera_open_26 | 349.525 ± 2.889 | 380.406 ± 3.398 | **296.618 ± 2.823** | 403.211 ± 5.584 | 916.589 ± 50.586 |
| tensornetwork_permutation_focus_step409_316 | **97.679 ± 0.965** | 189.167 ± 5.304 | 183.995 ± 2.217 | 360.322 ± 9.275 | 462.313 ± 49.129 |
| tensornetwork_permutation_light_415 | **100.743 ± 1.113** | 137.980 ± 1.399 | 181.555 ± 1.428 | 443.780 ± 164.762 | 427.616 ± 51.157 |

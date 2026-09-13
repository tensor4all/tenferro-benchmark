# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260913_153332/run.yaml`
- Timestamp: `20260913_153332`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260913_153332`.

- tenferro-rs commit: `a793c2e95693f053722fbff8d0db25722336c21f`

## CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-accelerate`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Julia / OMEinsum.jl Backend

- Julia: version `1.12.5`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.dylib)`, probe threads `4`

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

- Source table: `data/results/mac-cpu/cpu/einsum/20260913_153332/einsum_table_t4_20260913_153332.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260913_153332/tenferro_trace_t4_20260913_153332.log`
- `data/results/mac-cpu/cpu/einsum/20260913_153332/tenferro_eager_t4_20260913_153332.log`
- `data/results/mac-cpu/cpu/einsum/20260913_153332/pytorch_cpu_t4_20260913_153332.log`
- `data/results/mac-cpu/cpu/einsum/20260913_153332/jax_cpu_t4_20260913_153332.log`
- `data/results/mac-cpu/cpu/einsum/20260913_153332/julia_omeinsum_t4_20260913_153332.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.870 ± 0.113 | 0.720 ± 0.204 | **0.390 ± 0.008** | 0.641 ± 0.017 | 1.929 ± 0.857 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.193 ± 0.023 | 0.172 ± 0.015 | **0.086 ± 0.001** | 0.148 ± 0.007 | 0.641 ± 0.218 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.080 ± 0.015 | 0.073 ± 0.023 | **0.055 ± 0.003** | 0.114 ± 0.011 | 0.326 ± 0.040 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.164 ± 0.015 | 0.065 ± 0.012 | **0.054 ± 0.009** | 0.120 ± 0.026 | 0.657 ± 0.520 |
| bin_elementwise_mul_2048x2048 | 0.575 ± 0.011 | 1.314 ± 0.079 | **0.553 ± 0.009** | 0.774 ± 0.041 | 0.929 ± 0.164 |
| bin_matmul_1024 | 5.135 ± 0.226 | 4.879 ± 0.717 | **4.505 ± 0.088** | 9.141 ± 0.093 | 14.641 ± 3.270 |
| bin_matmul_256 | 0.152 ± 0.005 | 0.136 ± 0.005 | **0.088 ± 0.003** | 0.229 ± 0.015 | 0.300 ± 0.005 |
| bin_outer_product_4096 | 0.580 ± 0.009 | **0.569 ± 0.058** | 0.576 ± 0.005 | 0.634 ± 0.030 | 70.114 ± 13.169 |
| gm_queen5_5_3.wcsp | 518.490 ± 46.556 | 571.573 ± 26.963 | **447.460 ± 10.165** | 678.801 ± 17.271 | 1036.301 ± 8.695 |
| lm_batch_likelihood_brackets_4_4d | 7.201 ± 0.108 | 12.752 ± 0.136 | 8.859 ± 0.595 | **6.956 ± 0.181** | 9.666 ± 1.176 |
| lm_batch_likelihood_sentence_3_12d | **9.180 ± 0.235** | 16.622 ± 1.452 | 19.976 ± 1.644 | 11.749 ± 0.142 | 34.736 ± 16.982 |
| lm_batch_likelihood_sentence_4_4d | 7.744 ± 0.185 | 13.709 ± 0.331 | 8.748 ± 0.381 | **7.179 ± 0.159** | 13.299 ± 18.225 |
| nary_matmul_chain_64 | 0.035 ± 0.004 | 0.061 ± 0.009 | **0.013 ± 0.000** | 0.061 ± 0.003 | 0.042 ± 0.002 |
| str_matrix_chain_multiplication_100 | 2.924 ± 0.100 | 5.042 ± 0.049 | **2.114 ± 0.107** | 7.905 ± 0.335 | 5.199 ± 0.410 |
| str_mps_varying_inner_product_200 | 7.012 ± 0.116 | 11.585 ± 0.070 | **3.820 ± 0.065** | 15.819 ± 0.432 | 9.810 ± 3.879 |
| str_nw_mera_closed_120 | 142.181 ± 3.194 | 158.597 ± 3.105 | **140.774 ± 2.227** | 250.619 ± 0.860 | 318.244 ± 36.025 |
| str_nw_mera_open_26 | 147.229 ± 6.789 | 150.782 ± 1.138 | **120.362 ± 3.386** | 184.724 ± 1.959 | 266.692 ± 18.365 |
| tensornetwork_permutation_focus_step409_316 | **62.643 ± 34.106** | 109.797 ± 26.690 | 74.449 ± 5.884 | 89.360 ± 1.426 | 89.691 ± 19.517 |
| tensornetwork_permutation_light_415 | **65.360 ± 2.058** | 83.727 ± 7.039 | 71.610 ± 2.269 | 93.255 ± 1.208 | 89.806 ± 17.059 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.870 ± 0.113 | 0.720 ± 0.204 | **0.390 ± 0.008** | 0.641 ± 0.017 | 1.929 ± 0.857 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.193 ± 0.023 | 0.172 ± 0.015 | **0.086 ± 0.001** | 0.148 ± 0.007 | 0.641 ± 0.218 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.080 ± 0.015 | 0.073 ± 0.023 | **0.055 ± 0.003** | 0.114 ± 0.011 | 0.326 ± 0.040 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.164 ± 0.015 | 0.065 ± 0.012 | **0.054 ± 0.009** | 0.120 ± 0.026 | 0.657 ± 0.520 |
| bin_elementwise_mul_2048x2048 | 0.575 ± 0.011 | 1.314 ± 0.079 | **0.553 ± 0.009** | 0.774 ± 0.041 | 0.929 ± 0.164 |
| bin_matmul_1024 | 5.135 ± 0.226 | 4.879 ± 0.717 | **4.505 ± 0.088** | 9.141 ± 0.093 | 14.641 ± 3.270 |
| bin_matmul_256 | 0.152 ± 0.005 | 0.136 ± 0.005 | **0.088 ± 0.003** | 0.229 ± 0.015 | 0.300 ± 0.005 |
| bin_outer_product_4096 | 0.580 ± 0.009 | **0.569 ± 0.058** | 0.576 ± 0.005 | 0.634 ± 0.030 | 70.114 ± 13.169 |
| gm_queen5_5_3.wcsp | **144.295 ± 15.518** | 204.987 ± 8.352 | 183.590 ± 3.941 | 260.333 ± 2.379 | 435.519 ± 36.045 |
| lm_batch_likelihood_brackets_4_4d | 8.218 ± 0.316 | 13.213 ± 0.163 | 10.049 ± 0.753 | **6.694 ± 0.200** | 15.549 ± 16.887 |
| lm_batch_likelihood_sentence_3_12d | **10.405 ± 0.463** | 16.523 ± 0.666 | 19.558 ± 1.705 | 12.237 ± 0.226 | 39.831 ± 17.282 |
| lm_batch_likelihood_sentence_4_4d | 8.260 ± 0.208 | 13.370 ± 0.096 | 10.053 ± 0.775 | **6.452 ± 0.298** | 15.886 ± 18.426 |
| nary_matmul_chain_64 | 0.035 ± 0.004 | 0.061 ± 0.009 | **0.013 ± 0.000** | 0.061 ± 0.003 | 0.042 ± 0.002 |
| str_matrix_chain_multiplication_100 | 3.099 ± 0.510 | 5.017 ± 0.173 | **2.119 ± 0.046** | 7.563 ± 0.342 | 6.790 ± 0.857 |
| str_mps_varying_inner_product_200 | 8.386 ± 1.478 | 12.059 ± 0.116 | **7.606 ± 0.245** | 16.995 ± 0.376 | 9.424 ± 0.731 |
| str_nw_mera_closed_120 | 131.150 ± 3.285 | 140.747 ± 1.639 | **129.176 ± 2.816** | 249.414 ± 2.146 | 304.349 ± 13.121 |
| str_nw_mera_open_26 | 158.410 ± 15.164 | 147.378 ± 2.007 | **122.639 ± 1.841** | 191.949 ± 2.810 | 266.387 ± 21.043 |
| tensornetwork_permutation_focus_step409_316 | **62.643 ± 34.106** | 109.797 ± 26.690 | 74.449 ± 5.884 | 89.360 ± 1.426 | 89.691 ± 19.517 |
| tensornetwork_permutation_light_415 | **65.360 ± 2.058** | 83.727 ± 7.039 | 71.610 ± 2.269 | 93.255 ± 1.208 | 89.806 ± 17.059 |

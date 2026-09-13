# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260913_183437/run.yaml`
- Timestamp: `20260913_183437`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260913_183437`.

- tenferro-rs commit: `28dfc7e383a653e1461d38b73ced3d563b4efdad`

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

- Source table: `data/results/mac-cpu/cpu/einsum/20260913_183437/einsum_table_t4_20260913_183437.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260913_183437/tenferro_trace_t4_20260913_183437.log`
- `data/results/mac-cpu/cpu/einsum/20260913_183437/tenferro_eager_t4_20260913_183437.log`
- `data/results/mac-cpu/cpu/einsum/20260913_183437/pytorch_cpu_t4_20260913_183437.log`
- `data/results/mac-cpu/cpu/einsum/20260913_183437/jax_cpu_t4_20260913_183437.log`
- `data/results/mac-cpu/cpu/einsum/20260913_183437/julia_omeinsum_t4_20260913_183437.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.827 ± 0.165 | 0.732 ± 0.112 | **0.388 ± 0.002** | 0.646 ± 0.071 | 2.079 ± 3.470 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.211 ± 0.041 | 0.134 ± 0.027 | **0.091 ± 0.001** | 0.155 ± 0.018 | 0.577 ± 0.179 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.077 ± 0.024 | 0.091 ± 0.009 | **0.051 ± 0.004** | 0.117 ± 0.014 | 0.361 ± 0.094 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.167 ± 0.015 | 0.079 ± 0.011 | **0.052 ± 0.002** | 0.121 ± 0.038 | 0.764 ± 0.661 |
| bin_elementwise_mul_2048x2048 | 0.614 ± 0.049 | 1.290 ± 0.073 | **0.587 ± 0.024** | 0.775 ± 0.044 | 0.958 ± 0.103 |
| bin_matmul_1024 | 6.555 ± 0.423 | 5.078 ± 0.314 | **4.918 ± 0.199** | 9.414 ± 0.086 | 15.785 ± 4.354 |
| bin_matmul_256 | 0.159 ± 0.008 | 0.119 ± 0.006 | **0.087 ± 0.001** | 0.233 ± 0.020 | 0.381 ± 0.082 |
| bin_outer_product_4096 | 0.591 ± 0.010 | 0.586 ± 0.033 | **0.584 ± 0.136** | 0.629 ± 0.045 | 72.282 ± 14.632 |
| gm_queen5_5_3.wcsp | 530.502 ± 57.379 | 611.690 ± 93.150 | **474.619 ± 20.509** | 693.427 ± 17.711 | 1061.221 ± 16.175 |
| lm_batch_likelihood_brackets_4_4d | **6.800 ± 0.112** | 12.306 ± 0.182 | 6.845 ± 0.203 | 7.069 ± 0.181 | 9.850 ± 0.806 |
| lm_batch_likelihood_sentence_3_12d | **9.381 ± 0.163** | 16.689 ± 0.329 | 17.997 ± 0.637 | 12.046 ± 0.226 | 24.743 ± 15.001 |
| lm_batch_likelihood_sentence_4_4d | **7.033 ± 0.124** | 13.467 ± 0.130 | 7.203 ± 0.074 | 7.128 ± 0.264 | 14.540 ± 19.518 |
| nary_matmul_chain_64 | 0.034 ± 0.004 | 0.043 ± 0.011 | **0.013 ± 0.001** | 0.062 ± 0.009 | 0.044 ± 0.004 |
| str_matrix_chain_multiplication_100 | **2.091 ± 0.085** | 3.852 ± 0.163 | 2.164 ± 0.024 | 8.071 ± 0.361 | 5.895 ± 1.083 |
| str_mps_varying_inner_product_200 | 5.798 ± 0.086 | 9.072 ± 0.097 | **3.730 ± 0.038** | 16.217 ± 0.341 | 10.168 ± 3.994 |
| str_nw_mera_closed_120 | 145.571 ± 3.014 | 155.345 ± 1.679 | **138.955 ± 0.757** | 252.215 ± 4.985 | 318.463 ± 4.155 |
| str_nw_mera_open_26 | 148.560 ± 4.145 | 150.312 ± 1.266 | **121.233 ± 1.633** | 185.111 ± 4.058 | 274.054 ± 22.061 |
| tensornetwork_permutation_focus_step409_316 | **59.426 ± 1.183** | 122.820 ± 18.645 | 75.797 ± 3.520 | 92.316 ± 1.256 | 87.863 ± 16.387 |
| tensornetwork_permutation_light_415 | **59.785 ± 27.676** | 103.799 ± 24.924 | 71.702 ± 7.196 | 96.131 ± 2.119 | 88.091 ± 3.708 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.827 ± 0.165 | 0.732 ± 0.112 | **0.388 ± 0.002** | 0.646 ± 0.071 | 2.079 ± 3.470 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.211 ± 0.041 | 0.134 ± 0.027 | **0.091 ± 0.001** | 0.155 ± 0.018 | 0.577 ± 0.179 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.077 ± 0.024 | 0.091 ± 0.009 | **0.051 ± 0.004** | 0.117 ± 0.014 | 0.361 ± 0.094 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.167 ± 0.015 | 0.079 ± 0.011 | **0.052 ± 0.002** | 0.121 ± 0.038 | 0.764 ± 0.661 |
| bin_elementwise_mul_2048x2048 | 0.614 ± 0.049 | 1.290 ± 0.073 | **0.587 ± 0.024** | 0.775 ± 0.044 | 0.958 ± 0.103 |
| bin_matmul_1024 | 6.555 ± 0.423 | 5.078 ± 0.314 | **4.918 ± 0.199** | 9.414 ± 0.086 | 15.785 ± 4.354 |
| bin_matmul_256 | 0.159 ± 0.008 | 0.119 ± 0.006 | **0.087 ± 0.001** | 0.233 ± 0.020 | 0.381 ± 0.082 |
| bin_outer_product_4096 | 0.591 ± 0.010 | 0.586 ± 0.033 | **0.584 ± 0.136** | 0.629 ± 0.045 | 72.282 ± 14.632 |
| gm_queen5_5_3.wcsp | **142.759 ± 17.268** | 211.564 ± 10.597 | 189.523 ± 8.486 | 265.197 ± 2.374 | 414.576 ± 13.075 |
| lm_batch_likelihood_brackets_4_4d | 7.481 ± 0.087 | 12.538 ± 0.260 | 9.815 ± 0.800 | **6.667 ± 0.135** | 15.040 ± 15.907 |
| lm_batch_likelihood_sentence_3_12d | **9.687 ± 0.266** | 16.630 ± 0.323 | 20.162 ± 1.123 | 12.315 ± 0.145 | 35.208 ± 17.264 |
| lm_batch_likelihood_sentence_4_4d | 7.558 ± 0.239 | 13.195 ± 0.397 | 9.677 ± 0.749 | **6.450 ± 0.255** | 14.607 ± 16.861 |
| nary_matmul_chain_64 | 0.034 ± 0.004 | 0.043 ± 0.011 | **0.013 ± 0.001** | 0.062 ± 0.009 | 0.044 ± 0.004 |
| str_matrix_chain_multiplication_100 | 2.186 ± 0.073 | 3.781 ± 0.138 | **2.168 ± 0.102** | 7.992 ± 0.447 | 5.635 ± 0.474 |
| str_mps_varying_inner_product_200 | 6.390 ± 0.384 | 9.899 ± 0.132 | **6.133 ± 0.550** | 17.215 ± 0.403 | 8.725 ± 0.609 |
| str_nw_mera_closed_120 | 129.389 ± 1.500 | 138.885 ± 1.164 | **127.876 ± 3.098** | 251.120 ± 1.297 | 303.375 ± 21.981 |
| str_nw_mera_open_26 | 153.154 ± 6.203 | 146.833 ± 1.018 | **122.873 ± 1.387** | 193.525 ± 0.868 | 271.664 ± 21.018 |
| tensornetwork_permutation_focus_step409_316 | **59.426 ± 1.183** | 122.820 ± 18.645 | 75.797 ± 3.520 | 92.316 ± 1.256 | 87.863 ± 16.387 |
| tensornetwork_permutation_light_415 | **59.785 ± 27.676** | 103.799 ± 24.924 | 71.702 ± 7.196 | 96.131 ± 2.119 | 88.091 ± 3.708 |

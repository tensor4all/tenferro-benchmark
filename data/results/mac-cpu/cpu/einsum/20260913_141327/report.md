# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260913_141327/run.yaml`
- Timestamp: `20260913_141327`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260913_141327`.

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`

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

- Julia: version `1.12.5`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.dylib)`, probe threads `1`

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

## Threads: 1

- Source table: `data/results/mac-cpu/cpu/einsum/20260913_141327/einsum_table_t1_20260913_141327.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260913_141327/tenferro_trace_t1_20260913_141327.log`
- `data/results/mac-cpu/cpu/einsum/20260913_141327/tenferro_eager_t1_20260913_141327.log`
- `data/results/mac-cpu/cpu/einsum/20260913_141327/pytorch_cpu_t1_20260913_141327.log`
- `data/results/mac-cpu/cpu/einsum/20260913_141327/jax_cpu_t1_20260913_141327.log`
- `data/results/mac-cpu/cpu/einsum/20260913_141327/julia_omeinsum_t1_20260913_141327.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.124 ± 0.304 | 0.817 ± 0.178 | 0.388 ± 0.017 | **0.384 ± 0.055** | 4.064 ± 4.976 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.219 ± 0.039 | 0.135 ± 0.008 | **0.086 ± 0.002** | 0.190 ± 0.034 | 0.610 ± 0.163 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.116 ± 0.009 | 0.111 ± 0.018 | **0.109 ± 0.010** | 0.121 ± 0.028 | 0.541 ± 0.395 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.451 ± 0.032 | **0.108 ± 0.018** | 0.118 ± 0.003 | 0.119 ± 0.021 | 0.821 ± 0.433 |
| bin_elementwise_mul_2048x2048 | 1.029 ± 0.086 | 2.218 ± 0.203 | **0.840 ± 0.046** | 0.841 ± 0.034 | 0.988 ± 0.697 |
| bin_matmul_1024 | 6.529 ± 0.330 | 6.256 ± 1.236 | 5.574 ± 0.460 | **3.437 ± 0.105** | 41.356 ± 4.882 |
| bin_matmul_256 | 0.147 ± 0.007 | 0.108 ± 0.008 | **0.089 ± 0.002** | 0.217 ± 0.035 | 0.773 ± 0.142 |
| bin_outer_product_4096 | 1.034 ± 0.072 | 1.041 ± 0.043 | 1.530 ± 0.080 | **0.649 ± 0.039** | 68.273 ± 44.457 |
| gm_queen5_5_3.wcsp | 923.940 ± 4.012 | 1116.085 ± 2.857 | 927.551 ± 9.843 | **470.054 ± 5.130** | 1459.815 ± 101.078 |
| lm_batch_likelihood_brackets_4_4d | 8.625 ± 0.264 | 14.584 ± 0.224 | 9.161 ± 0.128 | **4.746 ± 0.177** | 12.516 ± 5.382 |
| lm_batch_likelihood_sentence_3_12d | 11.063 ± 0.403 | 27.536 ± 0.524 | 18.929 ± 0.304 | **6.785 ± 0.113** | 90.409 ± 51.052 |
| lm_batch_likelihood_sentence_4_4d | 8.932 ± 0.395 | 16.656 ± 0.313 | 9.486 ± 0.091 | **5.086 ± 0.078** | 19.639 ± 50.608 |
| nary_matmul_chain_64 | 0.024 ± 0.001 | 0.017 ± 0.000 | **0.013 ± 0.001** | 0.059 ± 0.007 | 0.034 ± 0.007 |
| str_matrix_chain_multiplication_100 | 2.398 ± 0.166 | 2.855 ± 0.145 | **2.220 ± 0.100** | 6.118 ± 0.119 | 9.774 ± 0.263 |
| str_mps_varying_inner_product_200 | 6.020 ± 0.155 | 7.175 ± 0.055 | **3.715 ± 0.114** | 12.595 ± 0.111 | 11.362 ± 5.572 |
| str_nw_mera_closed_120 | 180.704 ± 2.783 | 193.817 ± 0.729 | 163.979 ± 1.250 | **87.690 ± 1.867** | 993.978 ± 1.356 |
| str_nw_mera_open_26 | 188.736 ± 0.665 | 207.305 ± 1.253 | 132.197 ± 0.967 | **68.607 ± 0.862** | 738.306 ± 59.505 |
| tensornetwork_permutation_focus_step409_316 | 125.562 ± 0.844 | 210.410 ± 11.951 | 158.010 ± 0.993 | **70.847 ± 3.641** | 184.149 ± 8.036 |
| tensornetwork_permutation_light_415 | 126.827 ± 0.677 | 158.621 ± 2.625 | 138.329 ± 0.809 | **76.543 ± 5.676** | 185.034 ± 0.966 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.124 ± 0.304 | 0.817 ± 0.178 | 0.388 ± 0.017 | **0.384 ± 0.055** | 4.064 ± 4.976 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.219 ± 0.039 | 0.135 ± 0.008 | **0.086 ± 0.002** | 0.190 ± 0.034 | 0.610 ± 0.163 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.116 ± 0.009 | 0.111 ± 0.018 | **0.109 ± 0.010** | 0.121 ± 0.028 | 0.541 ± 0.395 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.451 ± 0.032 | **0.108 ± 0.018** | 0.118 ± 0.003 | 0.119 ± 0.021 | 0.821 ± 0.433 |
| bin_elementwise_mul_2048x2048 | 1.029 ± 0.086 | 2.218 ± 0.203 | **0.840 ± 0.046** | 0.841 ± 0.034 | 0.988 ± 0.697 |
| bin_matmul_1024 | 6.529 ± 0.330 | 6.256 ± 1.236 | 5.574 ± 0.460 | **3.437 ± 0.105** | 41.356 ± 4.882 |
| bin_matmul_256 | 0.147 ± 0.007 | 0.108 ± 0.008 | **0.089 ± 0.002** | 0.217 ± 0.035 | 0.773 ± 0.142 |
| bin_outer_product_4096 | 1.034 ± 0.072 | 1.041 ± 0.043 | 1.530 ± 0.080 | **0.649 ± 0.039** | 68.273 ± 44.457 |
| gm_queen5_5_3.wcsp | 238.519 ± 0.721 | 411.123 ± 2.086 | 347.067 ± 6.556 | **159.978 ± 22.130** | 801.282 ± 19.650 |
| lm_batch_likelihood_brackets_4_4d | 9.610 ± 0.163 | 15.536 ± 0.298 | 9.340 ± 0.194 | **4.912 ± 0.605** | 20.224 ± 53.217 |
| lm_batch_likelihood_sentence_3_12d | 12.753 ± 0.283 | 29.090 ± 0.251 | 18.445 ± 0.359 | **7.474 ± 0.235** | 45.157 ± 48.753 |
| lm_batch_likelihood_sentence_4_4d | 9.844 ± 0.085 | 17.361 ± 0.145 | 8.941 ± 0.332 | **5.483 ± 0.377** | 19.827 ± 53.089 |
| nary_matmul_chain_64 | 0.024 ± 0.001 | 0.017 ± 0.000 | **0.013 ± 0.001** | 0.059 ± 0.007 | 0.034 ± 0.007 |
| str_matrix_chain_multiplication_100 | 2.412 ± 0.093 | 2.875 ± 0.102 | **2.166 ± 0.048** | 6.376 ± 0.550 | 9.831 ± 1.106 |
| str_mps_varying_inner_product_200 | 7.130 ± 0.242 | 7.734 ± 0.135 | **4.734 ± 0.174** | 13.900 ± 0.222 | 10.314 ± 6.992 |
| str_nw_mera_closed_120 | 154.369 ± 0.580 | 163.501 ± 0.412 | 133.428 ± 2.060 | **85.330 ± 1.444** | 967.621 ± 59.887 |
| str_nw_mera_open_26 | 191.976 ± 2.564 | 205.499 ± 3.792 | 137.430 ± 0.308 | **77.412 ± 1.459** | 696.144 ± 60.096 |
| tensornetwork_permutation_focus_step409_316 | 125.562 ± 0.844 | 210.410 ± 11.951 | 158.010 ± 0.993 | **70.847 ± 3.641** | 184.149 ± 8.036 |
| tensornetwork_permutation_light_415 | 126.827 ± 0.677 | 158.621 ± 2.625 | 138.329 ± 0.809 | **76.543 ± 5.676** | 185.034 ± 0.966 |

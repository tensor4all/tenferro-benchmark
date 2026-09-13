# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260913_152441/run.yaml`
- Timestamp: `20260913_152441`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260913_152441`.

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

- Source table: `data/results/mac-cpu/cpu/einsum/20260913_152441/einsum_table_t1_20260913_152441.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260913_152441/tenferro_trace_t1_20260913_152441.log`
- `data/results/mac-cpu/cpu/einsum/20260913_152441/tenferro_eager_t1_20260913_152441.log`
- `data/results/mac-cpu/cpu/einsum/20260913_152441/pytorch_cpu_t1_20260913_152441.log`
- `data/results/mac-cpu/cpu/einsum/20260913_152441/jax_cpu_t1_20260913_152441.log`
- `data/results/mac-cpu/cpu/einsum/20260913_152441/julia_omeinsum_t1_20260913_152441.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.025 ± 0.153 | 0.745 ± 0.164 | **0.387 ± 0.002** | 2.285 ± 0.068 | 3.744 ± 4.571 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.191 ± 0.013 | 0.130 ± 0.020 | **0.087 ± 0.004** | 0.355 ± 0.022 | 0.682 ± 0.201 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.107 ± 0.009** | 0.113 ± 0.003 | 0.108 ± 0.003 | 0.203 ± 0.018 | 0.333 ± 0.397 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.410 ± 0.035 | **0.091 ± 0.017** | 0.114 ± 0.004 | 0.207 ± 0.007 | 0.710 ± 0.870 |
| bin_elementwise_mul_2048x2048 | 0.950 ± 0.067 | 2.129 ± 0.173 | **0.803 ± 0.022** | 1.516 ± 0.064 | 0.900 ± 0.445 |
| bin_matmul_1024 | 6.314 ± 0.323 | 5.189 ± 0.190 | **4.459 ± 0.048** | 35.844 ± 2.266 | 39.925 ± 4.863 |
| bin_matmul_256 | 0.135 ± 0.001 | 0.101 ± 0.000 | **0.088 ± 0.001** | 0.619 ± 0.020 | 0.714 ± 0.018 |
| bin_outer_product_4096 | **0.966 ± 0.018** | 1.039 ± 0.017 | 1.478 ± 0.061 | 1.376 ± 0.403 | 60.360 ± 39.532 |
| gm_queen5_5_3.wcsp | **893.686 ± 7.603** | 1068.761 ± 3.177 | 922.239 ± 24.665 | 1298.300 ± 42.481 | 1373.606 ± 104.458 |
| lm_batch_likelihood_brackets_4_4d | 7.924 ± 0.080 | 13.113 ± 0.179 | 9.583 ± 1.040 | **7.211 ± 0.097** | 18.528 ± 51.591 |
| lm_batch_likelihood_sentence_3_12d | **10.098 ± 0.131** | 23.712 ± 0.131 | 18.420 ± 0.835 | 28.653 ± 0.312 | 59.158 ± 45.765 |
| lm_batch_likelihood_sentence_4_4d | 8.247 ± 0.069 | 14.987 ± 0.186 | 9.739 ± 0.580 | **8.232 ± 0.057** | 19.693 ± 46.116 |
| nary_matmul_chain_64 | 0.014 ± 0.001 | 0.018 ± 0.000 | **0.013 ± 0.001** | 0.057 ± 0.005 | 0.041 ± 0.010 |
| str_matrix_chain_multiplication_100 | **2.093 ± 0.049** | 2.885 ± 0.035 | 2.193 ± 0.045 | 7.895 ± 0.109 | 9.827 ± 0.219 |
| str_mps_varying_inner_product_200 | 5.521 ± 0.045 | 7.183 ± 0.039 | **3.574 ± 0.064** | 11.955 ± 0.142 | 11.298 ± 6.260 |
| str_nw_mera_closed_120 | 184.910 ± 15.094 | 182.470 ± 1.353 | **156.496 ± 1.361** | 831.175 ± 1.333 | 997.266 ± 2.352 |
| str_nw_mera_open_26 | 189.256 ± 1.174 | 199.500 ± 7.931 | **127.968 ± 1.734** | 617.953 ± 2.703 | 741.180 ± 12.210 |
| tensornetwork_permutation_focus_step409_316 | **123.492 ± 0.660** | 178.831 ± 4.842 | 153.266 ± 5.090 | 203.357 ± 1.037 | 186.321 ± 51.828 |
| tensornetwork_permutation_light_415 | **124.617 ± 0.747** | 151.623 ± 1.136 | 135.958 ± 3.645 | 207.103 ± 0.734 | 186.033 ± 0.598 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.025 ± 0.153 | 0.745 ± 0.164 | **0.387 ± 0.002** | 2.285 ± 0.068 | 3.744 ± 4.571 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.191 ± 0.013 | 0.130 ± 0.020 | **0.087 ± 0.004** | 0.355 ± 0.022 | 0.682 ± 0.201 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.107 ± 0.009** | 0.113 ± 0.003 | 0.108 ± 0.003 | 0.203 ± 0.018 | 0.333 ± 0.397 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.410 ± 0.035 | **0.091 ± 0.017** | 0.114 ± 0.004 | 0.207 ± 0.007 | 0.710 ± 0.870 |
| bin_elementwise_mul_2048x2048 | 0.950 ± 0.067 | 2.129 ± 0.173 | **0.803 ± 0.022** | 1.516 ± 0.064 | 0.900 ± 0.445 |
| bin_matmul_1024 | 6.314 ± 0.323 | 5.189 ± 0.190 | **4.459 ± 0.048** | 35.844 ± 2.266 | 39.925 ± 4.863 |
| bin_matmul_256 | 0.135 ± 0.001 | 0.101 ± 0.000 | **0.088 ± 0.001** | 0.619 ± 0.020 | 0.714 ± 0.018 |
| bin_outer_product_4096 | **0.966 ± 0.018** | 1.039 ± 0.017 | 1.478 ± 0.061 | 1.376 ± 0.403 | 60.360 ± 39.532 |
| gm_queen5_5_3.wcsp | **238.540 ± 11.842** | 397.667 ± 1.670 | 325.552 ± 35.018 | 629.382 ± 2.900 | 802.275 ± 3.827 |
| lm_batch_likelihood_brackets_4_4d | 9.264 ± 0.301 | 14.303 ± 0.153 | 9.126 ± 0.482 | **6.908 ± 0.143** | 20.639 ± 53.207 |
| lm_batch_likelihood_sentence_3_12d | **12.368 ± 0.393** | 25.362 ± 0.220 | 17.934 ± 0.670 | 29.768 ± 0.179 | 45.427 ± 47.532 |
| lm_batch_likelihood_sentence_4_4d | 9.581 ± 0.259 | 15.559 ± 0.168 | 9.303 ± 0.870 | **7.674 ± 0.049** | 19.903 ± 53.174 |
| nary_matmul_chain_64 | 0.014 ± 0.001 | 0.018 ± 0.000 | **0.013 ± 0.001** | 0.057 ± 0.005 | 0.041 ± 0.010 |
| str_matrix_chain_multiplication_100 | 2.177 ± 0.038 | 2.871 ± 0.033 | **2.162 ± 0.048** | 7.892 ± 0.114 | 9.804 ± 0.490 |
| str_mps_varying_inner_product_200 | 6.674 ± 0.124 | 7.760 ± 0.023 | **4.481 ± 0.179** | 12.843 ± 0.229 | 9.852 ± 7.628 |
| str_nw_mera_closed_120 | 149.323 ± 0.635 | 156.636 ± 0.548 | **130.336 ± 0.807** | 806.406 ± 2.260 | 967.409 ± 61.073 |
| str_nw_mera_open_26 | 184.589 ± 1.302 | 195.485 ± 2.644 | **134.698 ± 2.911** | 628.427 ± 2.912 | 699.950 ± 61.296 |
| tensornetwork_permutation_focus_step409_316 | **123.492 ± 0.660** | 178.831 ± 4.842 | 153.266 ± 5.090 | 203.357 ± 1.037 | 186.321 ± 51.828 |
| tensornetwork_permutation_light_415 | **124.617 ± 0.747** | 151.623 ± 1.136 | 135.958 ± 3.645 | 207.103 ± 0.734 | 186.033 ± 0.598 |

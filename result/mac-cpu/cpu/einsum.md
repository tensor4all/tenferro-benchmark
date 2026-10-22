# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `d5ef90538a4c05810fe63bc1b5713d871ac98279`
- Benchmark commit: `1711cc92c7d04a38b85df1170873ddd4cf029b5b`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261022_022731/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261022_022731/run.yaml`
- Timestamp: `20261022_022731`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261022_022731`.

- tenferro-rs commit: `d5ef90538a4c05810fe63bc1b5713d871ac98279`

### CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`

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

- tenferro-rs features: `blas-accelerate`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.2`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.0.3.30.dylib)`, probe threads `1`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20261022_022731/einsum_table_t1_20261022_022731.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20261022_022731/tenferro_trace_t1_20261022_022731.log`
- `data/results/mac-cpu/cpu/einsum/20261022_022731/tenferro_eager_t1_20261022_022731.log`
- `data/results/mac-cpu/cpu/einsum/20261022_022731/pytorch_cpu_t1_20261022_022731.log`
- `data/results/mac-cpu/cpu/einsum/20261022_022731/jax_cpu_t1_20261022_022731.log`
- `data/results/mac-cpu/cpu/einsum/20261022_022731/julia_omeinsum_t1_20261022_022731.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.041 ± 0.310 | **0.616 ± 0.060** | 0.809 ± 0.139 | 3.946 ± 0.048 | 4.118 ± 4.526 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.128 ± 0.001 | **0.085 ± 0.004** | 0.138 ± 0.011 | 0.630 ± 0.017 | 1.067 ± 0.947 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.117 ± 0.007** | 0.132 ± 0.010 | 0.293 ± 0.072 | 0.448 ± 0.059 | 0.553 ± 0.242 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.433 ± 0.071 | **0.140 ± 0.010** | 0.223 ± 0.034 | 0.468 ± 0.054 | 0.892 ± 7.522 |
| bin_elementwise_mul_2048x2048 | 2.755 ± 0.584 | 2.434 ± 0.441 | 2.357 ± 0.300 | 3.326 ± 0.115 | **2.126 ± 0.021** |
| bin_elementwise_mul_512x512 | 0.078 ± 0.010 | 0.081 ± 0.002 | 0.107 ± 0.020 | 0.283 ± 0.016 | **0.072 ± 0.012** |
| bin_matmul_1024 | 29.476 ± 11.727 | **11.838 ± 4.831** | 14.176 ± 1.356 | 56.147 ± 4.706 | 53.192 ± 16.221 |
| bin_matmul_256 | 0.160 ± 0.006 | **0.129 ± 0.006** | **0.129 ± 0.006** | 0.967 ± 0.022 | 0.888 ± 0.021 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.004 ± 0.000** | 0.026 ± 0.022 | 0.086 ± 0.009 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.072 ± 0.061 | **0.025 ± 0.002** | 0.093 ± 0.022 | 0.145 ± 0.009 | 0.055 ± 0.016 |
| bin_omeinsum_matmul_10x10 | 0.042 ± 0.011 | **0.007 ± 0.001** | 0.016 ± 0.001 | 0.134 ± 0.014 | 0.016 ± 0.001 |
| bin_outer_product_4096 | 6.822 ± 1.715 | **2.623 ± 0.523** | 4.136 ± 1.223 | 3.272 ± 0.199 | 461.853 ± 243.125 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 24.007 ± 11.007 | 12.203 ± 1.410 | **11.341 ± 0.511** | 59.100 ± 3.118 | 48.545 ± 4.605 |
| gm_queen5_5_3.wcsp | 5402.347 ± 569.949 | 7615.452 ± 541.768 | **3417.590 ± 992.566** | 6147.743 ± 2759.895 | 3885.735 ± 1590.682 |
| lm_batch_likelihood_brackets_4_4d | 15.607 ± 2.990 | 23.328 ± 9.629 | **15.516 ± 1.957** | 17.805 ± 7.495 | 17.594 ± 1.188 |
| lm_batch_likelihood_sentence_3_12d | **25.850 ± 5.437** | 55.496 ± 11.663 | 34.963 ± 8.379 | 48.487 ± 7.869 | 77.016 ± 13.569 |
| lm_batch_likelihood_sentence_4_4d | 16.662 ± 1.994 | 25.107 ± 9.241 | **15.870 ± 1.996** | 17.147 ± 3.454 | 39.081 ± 16.472 |
| nary_matmul_chain_64 | 0.044 ± 0.016 | **0.010 ± 0.000** | 0.031 ± 0.007 | 0.110 ± 0.025 | 0.086 ± 0.025 |
| str_matrix_chain_multiplication_100 | 6.285 ± 1.029 | 6.146 ± 1.707 | **5.340 ± 0.610** | 14.482 ± 3.589 | 13.069 ± 1.914 |
| str_mps_varying_inner_product_200 | 9.175 ± 1.503 | 10.871 ± 6.422 | **7.779 ± 0.668** | 26.054 ± 5.631 | 17.138 ± 20.444 |
| str_nw_mera_closed_120 | 403.268 ± 39.925 | 473.004 ± 160.290 | **384.942 ± 45.964** | 1613.767 ± 137.365 | 1205.229 ± 88.007 |
| str_nw_mera_open_26 | 319.789 ± 13.293 | 472.229 ± 44.258 | **250.751 ± 7.730** | 1147.690 ± 172.024 | 890.333 ± 71.304 |
| tensornetwork_permutation_focus_step409_316 | **148.335 ± 13.089** | 311.934 ± 15.639 | 278.029 ± 56.149 | 354.774 ± 57.835 | 209.408 ± 7.187 |
| tensornetwork_permutation_light_415 | **139.448 ± 8.368** | 151.092 ± 12.813 | 225.968 ± 11.530 | 327.923 ± 36.568 | 217.505 ± 24.896 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.041 ± 0.310 | **0.616 ± 0.060** | 0.809 ± 0.139 | 3.946 ± 0.048 | 4.118 ± 4.526 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.128 ± 0.001 | **0.085 ± 0.004** | 0.138 ± 0.011 | 0.630 ± 0.017 | 1.067 ± 0.947 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.117 ± 0.007** | 0.132 ± 0.010 | 0.293 ± 0.072 | 0.448 ± 0.059 | 0.553 ± 0.242 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.433 ± 0.071 | **0.140 ± 0.010** | 0.223 ± 0.034 | 0.468 ± 0.054 | 0.892 ± 7.522 |
| bin_elementwise_mul_2048x2048 | 2.755 ± 0.584 | 2.434 ± 0.441 | 2.357 ± 0.300 | 3.326 ± 0.115 | **2.126 ± 0.021** |
| bin_elementwise_mul_512x512 | 0.078 ± 0.010 | 0.081 ± 0.002 | 0.107 ± 0.020 | 0.283 ± 0.016 | **0.072 ± 0.012** |
| bin_matmul_1024 | 29.476 ± 11.727 | **11.838 ± 4.831** | 14.176 ± 1.356 | 56.147 ± 4.706 | 53.192 ± 16.221 |
| bin_matmul_256 | 0.160 ± 0.006 | **0.129 ± 0.006** | **0.129 ± 0.006** | 0.967 ± 0.022 | 0.888 ± 0.021 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.004 ± 0.000** | 0.026 ± 0.022 | 0.086 ± 0.009 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.072 ± 0.061 | **0.025 ± 0.002** | 0.093 ± 0.022 | 0.145 ± 0.009 | 0.055 ± 0.016 |
| bin_omeinsum_matmul_10x10 | 0.042 ± 0.011 | **0.007 ± 0.001** | 0.016 ± 0.001 | 0.134 ± 0.014 | 0.016 ± 0.001 |
| bin_outer_product_4096 | 6.822 ± 1.715 | **2.623 ± 0.523** | 4.136 ± 1.223 | 3.272 ± 0.199 | 461.853 ± 243.125 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 24.007 ± 11.007 | 12.203 ± 1.410 | **11.341 ± 0.511** | 59.100 ± 3.118 | 48.545 ± 4.605 |
| gm_queen5_5_3.wcsp | 1058.636 ± 196.174 | **756.009 ± 83.115** | 914.476 ± 103.169 | 1112.727 ± 92.606 | 899.203 ± 21.349 |
| lm_batch_likelihood_brackets_4_4d | 16.593 ± 1.215 | 20.095 ± 5.947 | 16.622 ± 0.701 | **12.429 ± 0.524** | 53.559 ± 42.408 |
| lm_batch_likelihood_sentence_3_12d | 35.174 ± 5.859 | 38.446 ± 3.761 | **26.169 ± 0.618** | 53.228 ± 5.160 | 74.103 ± 21.949 |
| lm_batch_likelihood_sentence_4_4d | 15.316 ± 1.950 | 23.466 ± 5.560 | 15.227 ± 0.349 | **12.604 ± 0.114** | 27.114 ± 19.363 |
| nary_matmul_chain_64 | 0.044 ± 0.016 | **0.010 ± 0.000** | 0.031 ± 0.007 | 0.110 ± 0.025 | 0.086 ± 0.025 |
| str_matrix_chain_multiplication_100 | 6.446 ± 1.361 | **5.189 ± 0.438** | 5.301 ± 0.202 | 13.029 ± 1.225 | 12.983 ± 6.381 |
| str_mps_varying_inner_product_200 | 15.876 ± 5.357 | 11.492 ± 1.152 | **8.456 ± 0.922** | 22.685 ± 0.219 | 15.299 ± 10.164 |
| str_nw_mera_closed_120 | 453.620 ± 114.449 | 308.184 ± 23.685 | **293.930 ± 10.482** | 1378.841 ± 46.345 | 1143.474 ± 65.574 |
| str_nw_mera_open_26 | 412.291 ± 39.421 | 486.905 ± 102.483 | **254.015 ± 57.040** | 1069.706 ± 45.655 | 910.928 ± 67.591 |
| tensornetwork_permutation_focus_step409_316 | **148.335 ± 13.089** | 311.934 ± 15.639 | 278.029 ± 56.149 | 354.774 ± 57.835 | 209.408 ± 7.187 |
| tensornetwork_permutation_light_415 | **139.448 ± 8.368** | 151.092 ± 12.813 | 225.968 ± 11.530 | 327.923 ± 36.568 | 217.505 ± 24.896 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261022_025208/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261022_025208/run.yaml`
- Timestamp: `20261022_025208`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261022_025208`.

- tenferro-rs commit: `d5ef90538a4c05810fe63bc1b5713d871ac98279`

### CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`

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

- tenferro-rs features: `blas-accelerate`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.2`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.0.3.30.dylib)`, probe threads `4`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20261022_025208/einsum_table_t4_20261022_025208.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20261022_025208/tenferro_trace_t4_20261022_025208.log`
- `data/results/mac-cpu/cpu/einsum/20261022_025208/tenferro_eager_t4_20261022_025208.log`
- `data/results/mac-cpu/cpu/einsum/20261022_025208/pytorch_cpu_t4_20261022_025208.log`
- `data/results/mac-cpu/cpu/einsum/20261022_025208/jax_cpu_t4_20261022_025208.log`
- `data/results/mac-cpu/cpu/einsum/20261022_025208/julia_omeinsum_t4_20261022_025208.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.986 ± 0.029 | 0.653 ± 0.080 | **0.579 ± 0.009** | 1.395 ± 0.036 | 92.156 ± 24.046 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.173 ± 0.123 | **0.085 ± 0.011** | 0.112 ± 0.001 | 0.579 ± 0.014 | 0.557 ± 0.964 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.168 ± 0.017 | **0.147 ± 0.018** | 0.214 ± 0.031 | 0.320 ± 0.006 | 0.589 ± 0.215 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.237 ± 0.053 | **0.133 ± 0.005** | 0.162 ± 0.015 | 0.350 ± 0.008 | 1.128 ± 3.129 |
| bin_elementwise_mul_2048x2048 | 2.142 ± 0.184 | **1.874 ± 0.043** | 1.929 ± 0.015 | 2.699 ± 0.063 | 2.111 ± 0.291 |
| bin_elementwise_mul_512x512 | 0.065 ± 0.004 | **0.053 ± 0.006** | 0.069 ± 0.002 | 0.255 ± 0.017 | 0.078 ± 0.037 |
| bin_matmul_1024 | 9.674 ± 1.444 | **7.961 ± 0.896** | 8.185 ± 0.112 | 17.781 ± 0.397 | 47.743 ± 8.323 |
| bin_matmul_256 | 0.176 ± 0.006 | **0.102 ± 0.000** | 0.109 ± 0.000 | 0.470 ± 0.060 | 6.332 ± 3.631 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.004 ± 0.000** | 0.017 ± 0.001 | 0.082 ± 0.006 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.026 ± 0.002 | **0.024 ± 0.001** | 0.070 ± 0.001 | 0.172 ± 0.015 | 0.078 ± 0.015 |
| bin_omeinsum_matmul_10x10 | 0.014 ± 0.000 | **0.006 ± 0.000** | 0.015 ± 0.001 | 0.081 ± 0.010 | 0.016 ± 0.001 |
| bin_outer_product_4096 | 2.209 ± 0.328 | **2.103 ± 0.032** | 2.247 ± 0.010 | 2.211 ± 0.030 | 386.796 ± 205.659 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 11.137 ± 0.162 | **8.784 ± 0.162** | 9.246 ± 0.099 | 19.320 ± 0.234 | 35.665 ± 11.469 |
| gm_queen5_5_3.wcsp | 2203.227 ± 526.432 | 3519.127 ± 137.562 | **1745.164 ± 719.763** | 3521.311 ± 1082.666 | 9698.738 ± 4887.821 |
| lm_batch_likelihood_brackets_4_4d | **9.958 ± 0.083** | 14.293 ± 4.064 | 14.698 ± 1.985 | 13.001 ± 0.119 | 234.674 ± 92.191 |
| lm_batch_likelihood_sentence_3_12d | **15.257 ± 0.651** | 31.104 ± 1.683 | 24.515 ± 3.206 | 25.424 ± 2.585 | 161.991 ± 43.100 |
| lm_batch_likelihood_sentence_4_4d | **10.179 ± 0.099** | 17.276 ± 0.460 | 11.341 ± 0.119 | 13.768 ± 0.550 | 147.319 ± 68.932 |
| nary_matmul_chain_64 | 0.019 ± 0.001 | **0.010 ± 0.000** | 0.021 ± 0.001 | 0.115 ± 0.009 | 0.216 ± 0.022 |
| str_matrix_chain_multiplication_100 | **4.132 ± 0.022** | 4.481 ± 0.251 | 4.635 ± 0.110 | 15.531 ± 3.685 | 291.571 ± 24.490 |
| str_mps_varying_inner_product_200 | 8.381 ± 0.050 | 8.517 ± 0.113 | **7.109 ± 0.286** | 18.513 ± 1.933 | 34.029 ± 25.957 |
| str_nw_mera_closed_120 | 300.910 ± 4.123 | 298.949 ± 34.536 | **285.332 ± 7.603** | 450.978 ± 25.076 | 1226.919 ± 340.559 |
| str_nw_mera_open_26 | 244.210 ± 3.429 | 301.634 ± 19.426 | **196.517 ± 21.637** | 364.016 ± 13.041 | 1219.526 ± 410.727 |
| tensornetwork_permutation_focus_step409_316 | **107.582 ± 18.392** | 154.689 ± 3.845 | 130.513 ± 18.899 | 205.083 ± 28.629 | 499.463 ± 68.477 |
| tensornetwork_permutation_light_415 | **88.304 ± 0.336** | 97.647 ± 7.476 | 113.821 ± 15.134 | 207.777 ± 8.528 | 499.421 ± 38.254 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.986 ± 0.029 | 0.653 ± 0.080 | **0.579 ± 0.009** | 1.395 ± 0.036 | 92.156 ± 24.046 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.173 ± 0.123 | **0.085 ± 0.011** | 0.112 ± 0.001 | 0.579 ± 0.014 | 0.557 ± 0.964 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.168 ± 0.017 | **0.147 ± 0.018** | 0.214 ± 0.031 | 0.320 ± 0.006 | 0.589 ± 0.215 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.237 ± 0.053 | **0.133 ± 0.005** | 0.162 ± 0.015 | 0.350 ± 0.008 | 1.128 ± 3.129 |
| bin_elementwise_mul_2048x2048 | 2.142 ± 0.184 | **1.874 ± 0.043** | 1.929 ± 0.015 | 2.699 ± 0.063 | 2.111 ± 0.291 |
| bin_elementwise_mul_512x512 | 0.065 ± 0.004 | **0.053 ± 0.006** | 0.069 ± 0.002 | 0.255 ± 0.017 | 0.078 ± 0.037 |
| bin_matmul_1024 | 9.674 ± 1.444 | **7.961 ± 0.896** | 8.185 ± 0.112 | 17.781 ± 0.397 | 47.743 ± 8.323 |
| bin_matmul_256 | 0.176 ± 0.006 | **0.102 ± 0.000** | 0.109 ± 0.000 | 0.470 ± 0.060 | 6.332 ± 3.631 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.004 ± 0.000** | 0.017 ± 0.001 | 0.082 ± 0.006 | 0.009 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.026 ± 0.002 | **0.024 ± 0.001** | 0.070 ± 0.001 | 0.172 ± 0.015 | 0.078 ± 0.015 |
| bin_omeinsum_matmul_10x10 | 0.014 ± 0.000 | **0.006 ± 0.000** | 0.015 ± 0.001 | 0.081 ± 0.010 | 0.016 ± 0.001 |
| bin_outer_product_4096 | 2.209 ± 0.328 | **2.103 ± 0.032** | 2.247 ± 0.010 | 2.211 ± 0.030 | 386.796 ± 205.659 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 11.137 ± 0.162 | **8.784 ± 0.162** | 9.246 ± 0.099 | 19.320 ± 0.234 | 35.665 ± 11.469 |
| gm_queen5_5_3.wcsp | 506.063 ± 21.346 | **455.211 ± 52.008** | 456.489 ± 66.215 | 570.884 ± 22.943 | 2160.118 ± 501.836 |
| lm_batch_likelihood_brackets_4_4d | **11.179 ± 0.258** | 14.889 ± 3.823 | 11.567 ± 0.106 | 13.076 ± 0.270 | 128.655 ± 26.555 |
| lm_batch_likelihood_sentence_3_12d | **17.976 ± 0.249** | 28.835 ± 2.864 | 22.106 ± 0.155 | 30.936 ± 2.905 | 175.660 ± 54.590 |
| lm_batch_likelihood_sentence_4_4d | **8.748 ± 0.394** | 19.138 ± 0.797 | 11.686 ± 0.121 | 13.507 ± 0.209 | 143.907 ± 133.167 |
| nary_matmul_chain_64 | 0.019 ± 0.001 | **0.010 ± 0.000** | 0.021 ± 0.001 | 0.115 ± 0.009 | 0.216 ± 0.022 |
| str_matrix_chain_multiplication_100 | 4.581 ± 0.242 | 4.887 ± 0.217 | **4.507 ± 0.064** | 9.536 ± 0.435 | 293.903 ± 49.532 |
| str_mps_varying_inner_product_200 | 10.165 ± 0.452 | 10.752 ± 0.499 | **9.405 ± 0.267** | 19.414 ± 0.239 | 61.977 ± 36.129 |
| str_nw_mera_closed_120 | 257.645 ± 7.132 | 243.782 ± 9.436 | **233.791 ± 1.640** | 424.338 ± 11.718 | 980.804 ± 253.878 |
| str_nw_mera_open_26 | 247.993 ± 7.188 | 312.073 ± 7.794 | **198.722 ± 21.606** | 377.181 ± 22.522 | 1314.217 ± 229.384 |
| tensornetwork_permutation_focus_step409_316 | **107.582 ± 18.392** | 154.689 ± 3.845 | 130.513 ± 18.899 | 205.083 ± 28.629 | 499.463 ± 68.477 |
| tensornetwork_permutation_light_415 | **88.304 ± 0.336** | 97.647 ± 7.476 | 113.821 ± 15.134 | 207.777 ± 8.528 | 499.421 ± 38.254 |

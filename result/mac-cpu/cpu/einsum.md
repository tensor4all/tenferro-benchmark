# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `74d138135bc227c5a1ff41db8863f853b647d863`
- Benchmark commit: `cc2153b9548e1542f955c011dccfb77a98196fed`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261023_172845/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261023_172845/run.yaml`
- Timestamp: `20261023_172845`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261023_172845`.

- tenferro-rs commit: `74d138135bc227c5a1ff41db8863f853b647d863`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20261023_172845/einsum_table_t1_20261023_172845.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20261023_172845/tenferro_trace_t1_20261023_172845.log`
- `data/results/mac-cpu/cpu/einsum/20261023_172845/tenferro_eager_t1_20261023_172845.log`
- `data/results/mac-cpu/cpu/einsum/20261023_172845/pytorch_cpu_t1_20261023_172845.log`
- `data/results/mac-cpu/cpu/einsum/20261023_172845/jax_cpu_t1_20261023_172845.log`
- `data/results/mac-cpu/cpu/einsum/20261023_172845/julia_omeinsum_t1_20261023_172845.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.165 ± 0.081 | 0.745 ± 0.100 | **0.580 ± 0.012** | 7.339 ± 6.417 | 6.675 ± 6.756 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.140 ± 0.011 | **0.089 ± 0.005** | 0.115 ± 0.001 | 1.752 ± 1.214 | 2.279 ± 3.586 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.113 ± 0.001** | 0.229 ± 0.128 | 0.198 ± 0.023 | 0.636 ± 0.572 | 0.835 ± 0.463 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.390 ± 0.042 | **0.165 ± 0.020** | 0.219 ± 0.017 | 0.664 ± 0.279 | 2.713 ± 1.436 |
| bin_elementwise_mul_2048x2048 | **1.944 ± 0.162** | 2.489 ± 0.117 | 1.965 ± 0.035 | 6.500 ± 1.997 | 5.470 ± 1.669 |
| bin_elementwise_mul_512x512 | 0.073 ± 0.000 | **0.071 ± 0.001** | 0.076 ± 0.020 | 0.908 ± 0.482 | 0.096 ± 0.068 |
| bin_matmul_1024 | 10.319 ± 1.739 | 10.731 ± 0.555 | **9.914 ± 1.105** | 89.403 ± 23.909 | 85.680 ± 30.404 |
| bin_matmul_256 | 0.191 ± 0.048 | **0.121 ± 0.010** | **0.121 ± 0.009** | 1.467 ± 0.963 | 2.193 ± 1.173 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.002 | **0.004 ± 0.000** | 0.019 ± 0.001 | 0.125 ± 0.067 | 0.012 ± 0.008 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.028 ± 0.002 | **0.026 ± 0.011** | 0.079 ± 0.002 | 0.221 ± 0.069 | 0.306 ± 0.485 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.006 ± 0.000** | 0.017 ± 0.001 | 0.219 ± 0.033 | 0.018 ± 0.001 |
| bin_outer_product_4096 | 2.451 ± 0.193 | **2.135 ± 0.205** | 2.738 ± 0.280 | 3.744 ± 0.899 | 249.795 ± 21.005 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 16.203 ± 2.975 | **9.121 ± 2.488** | 9.699 ± 0.964 | 76.960 ± 14.947 | 69.534 ± 19.286 |
| gm_queen5_5_3.wcsp | 4473.764 ± 687.732 | 3900.797 ± 403.781 | **3835.662 ± 536.682** | 6402.187 ± 1459.168 | 8820.800 ± 4848.960 |
| lm_batch_likelihood_brackets_4_4d | 15.876 ± 1.675 | 19.977 ± 2.381 | **15.445 ± 1.002** | 15.925 ± 2.329 | 38.833 ± 29.771 |
| lm_batch_likelihood_sentence_3_12d | **24.403 ± 2.485** | 43.131 ± 4.428 | 28.495 ± 1.126 | 59.746 ± 2.704 | 138.381 ± 65.645 |
| lm_batch_likelihood_sentence_4_4d | **14.693 ± 1.354** | 20.812 ± 1.369 | 15.846 ± 1.079 | 17.338 ± 1.284 | 59.970 ± 33.998 |
| nary_matmul_chain_64 | 0.020 ± 0.018 | **0.010 ± 0.000** | 0.023 ± 0.002 | 0.219 ± 0.038 | 0.211 ± 0.178 |
| str_matrix_chain_multiplication_100 | **4.309 ± 0.880** | 4.778 ± 0.147 | 5.474 ± 0.297 | 19.856 ± 5.125 | 21.765 ± 11.222 |
| str_mps_varying_inner_product_200 | 9.977 ± 1.265 | 10.023 ± 0.875 | **6.638 ± 0.669** | 24.684 ± 10.056 | 31.227 ± 24.400 |
| str_nw_mera_closed_120 | 419.832 ± 38.343 | 383.351 ± 17.393 | **371.282 ± 49.009** | 1958.344 ± 255.094 | 1895.186 ± 546.958 |
| str_nw_mera_open_26 | 320.850 ± 35.233 | 425.117 ± 25.411 | **227.335 ± 14.848** | 1316.739 ± 238.138 | 1081.182 ± 174.392 |
| tensornetwork_permutation_focus_step409_316 | **149.711 ± 12.378** | 283.168 ± 5.418 | 253.194 ± 23.264 | 422.916 ± 91.183 | 257.280 ± 31.455 |
| tensornetwork_permutation_light_415 | **133.816 ± 5.636** | 140.818 ± 7.939 | 211.458 ± 19.428 | 388.312 ± 77.597 | 223.906 ± 61.615 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.165 ± 0.081 | 0.745 ± 0.100 | **0.580 ± 0.012** | 7.339 ± 6.417 | 6.675 ± 6.756 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.140 ± 0.011 | **0.089 ± 0.005** | 0.115 ± 0.001 | 1.752 ± 1.214 | 2.279 ± 3.586 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.113 ± 0.001** | 0.229 ± 0.128 | 0.198 ± 0.023 | 0.636 ± 0.572 | 0.835 ± 0.463 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.390 ± 0.042 | **0.165 ± 0.020** | 0.219 ± 0.017 | 0.664 ± 0.279 | 2.713 ± 1.436 |
| bin_elementwise_mul_2048x2048 | **1.944 ± 0.162** | 2.489 ± 0.117 | 1.965 ± 0.035 | 6.500 ± 1.997 | 5.470 ± 1.669 |
| bin_elementwise_mul_512x512 | 0.073 ± 0.000 | **0.071 ± 0.001** | 0.076 ± 0.020 | 0.908 ± 0.482 | 0.096 ± 0.068 |
| bin_matmul_1024 | 10.319 ± 1.739 | 10.731 ± 0.555 | **9.914 ± 1.105** | 89.403 ± 23.909 | 85.680 ± 30.404 |
| bin_matmul_256 | 0.191 ± 0.048 | **0.121 ± 0.010** | **0.121 ± 0.009** | 1.467 ± 0.963 | 2.193 ± 1.173 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.002 | **0.004 ± 0.000** | 0.019 ± 0.001 | 0.125 ± 0.067 | 0.012 ± 0.008 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.028 ± 0.002 | **0.026 ± 0.011** | 0.079 ± 0.002 | 0.221 ± 0.069 | 0.306 ± 0.485 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.006 ± 0.000** | 0.017 ± 0.001 | 0.219 ± 0.033 | 0.018 ± 0.001 |
| bin_outer_product_4096 | 2.451 ± 0.193 | **2.135 ± 0.205** | 2.738 ± 0.280 | 3.744 ± 0.899 | 249.795 ± 21.005 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 16.203 ± 2.975 | **9.121 ± 2.488** | 9.699 ± 0.964 | 76.960 ± 14.947 | 69.534 ± 19.286 |
| gm_queen5_5_3.wcsp | 841.613 ± 45.421 | **718.684 ± 34.852** | 871.706 ± 230.406 | 1637.707 ± 882.507 | 2130.036 ± 317.220 |
| lm_batch_likelihood_brackets_4_4d | **15.061 ± 0.577** | 19.966 ± 12.253 | 19.021 ± 4.860 | 20.658 ± 3.667 | 35.575 ± 34.423 |
| lm_batch_likelihood_sentence_3_12d | **28.766 ± 3.341** | 39.349 ± 1.373 | 31.658 ± 1.840 | 82.165 ± 17.247 | 82.237 ± 21.902 |
| lm_batch_likelihood_sentence_4_4d | **12.363 ± 0.638** | 21.301 ± 0.860 | 16.909 ± 2.337 | 16.877 ± 1.366 | 37.372 ± 28.578 |
| nary_matmul_chain_64 | 0.020 ± 0.018 | **0.010 ± 0.000** | 0.023 ± 0.002 | 0.219 ± 0.038 | 0.211 ± 0.178 |
| str_matrix_chain_multiplication_100 | 5.112 ± 0.287 | 5.221 ± 0.322 | **4.976 ± 0.432** | 15.927 ± 3.715 | 17.198 ± 9.604 |
| str_mps_varying_inner_product_200 | 11.504 ± 0.500 | 10.129 ± 0.787 | **8.430 ± 0.859** | 32.757 ± 9.786 | 16.857 ± 8.669 |
| str_nw_mera_closed_120 | 342.503 ± 40.758 | **321.456 ± 24.700** | 450.100 ± 112.908 | 1602.082 ± 173.414 | 1358.438 ± 197.258 |
| str_nw_mera_open_26 | 304.102 ± 33.953 | 426.216 ± 11.663 | **300.089 ± 54.765** | 1278.031 ± 215.882 | 1054.401 ± 93.740 |
| tensornetwork_permutation_focus_step409_316 | **149.711 ± 12.378** | 283.168 ± 5.418 | 253.194 ± 23.264 | 422.916 ± 91.183 | 257.280 ± 31.455 |
| tensornetwork_permutation_light_415 | **133.816 ± 5.636** | 140.818 ± 7.939 | 211.458 ± 19.428 | 388.312 ± 77.597 | 223.906 ± 61.615 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261023_175432/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261023_175432/run.yaml`
- Timestamp: `20261023_175432`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261023_175432`.

- tenferro-rs commit: `74d138135bc227c5a1ff41db8863f853b647d863`

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


- Source table: `data/results/mac-cpu/cpu/einsum/20261023_175432/einsum_table_t4_20261023_175432.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20261023_175432/tenferro_trace_t4_20261023_175432.log`
- `data/results/mac-cpu/cpu/einsum/20261023_175432/tenferro_eager_t4_20261023_175432.log`
- `data/results/mac-cpu/cpu/einsum/20261023_175432/pytorch_cpu_t4_20261023_175432.log`
- `data/results/mac-cpu/cpu/einsum/20261023_175432/jax_cpu_t4_20261023_175432.log`
- `data/results/mac-cpu/cpu/einsum/20261023_175432/julia_omeinsum_t4_20261023_175432.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.397 ± 0.208 | **0.647 ± 0.019** | 0.731 ± 0.025 | 2.790 ± 0.130 | 95.179 ± 44.401 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.325 ± 0.224 | 0.338 ± 0.257 | **0.138 ± 0.184** | 0.708 ± 0.024 | 0.763 ± 2.316 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.218 ± 0.067 | **0.200 ± 0.132** | 0.226 ± 0.010 | 0.500 ± 0.021 | 0.678 ± 0.548 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.372 ± 0.053 | **0.177 ± 0.032** | 0.228 ± 0.020 | 0.529 ± 0.025 | 1.518 ± 4.347 |
| bin_elementwise_mul_2048x2048 | 2.708 ± 0.265 | **2.641 ± 0.337** | 2.998 ± 0.106 | 3.304 ± 0.188 | 3.254 ± 0.462 |
| bin_elementwise_mul_512x512 | 0.123 ± 0.057 | 0.110 ± 0.037 | 0.414 ± 0.249 | 0.376 ± 0.034 | **0.078 ± 0.030** |
| bin_matmul_1024 | 14.398 ± 0.707 | **10.501 ± 0.380** | 14.194 ± 2.380 | 27.968 ± 2.840 | 48.682 ± 26.721 |
| bin_matmul_256 | 0.774 ± 0.557 | 0.120 ± 0.002 | **0.119 ± 0.001** | 0.880 ± 0.037 | 3.768 ± 1.912 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | **0.008 ± 0.006** | 0.011 ± 0.001 | 0.020 ± 0.004 | 0.087 ± 0.005 | 0.010 ± 0.007 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.002 | **0.025 ± 0.002** | 0.082 ± 0.010 | 0.188 ± 0.032 | 0.082 ± 0.014 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.002 | **0.007 ± 0.001** | 0.017 ± 0.001 | 0.056 ± 0.133 | 0.030 ± 0.009 |
| bin_outer_product_4096 | 3.149 ± 1.014 | **2.969 ± 0.237** | 3.333 ± 0.710 | 3.147 ± 0.037 | 264.967 ± 46.005 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 15.393 ± 0.342 | 11.617 ± 0.526 | **10.688 ± 0.307** | 32.269 ± 5.219 | 70.066 ± 22.834 |
| gm_queen5_5_3.wcsp | 4263.372 ± 591.222 | 6372.606 ± 465.721 | **2844.605 ± 169.721** | 4921.062 ± 609.329 | 10821.689 ± 1758.149 |
| lm_batch_likelihood_brackets_4_4d | **15.555 ± 2.308** | 26.424 ± 9.327 | 27.339 ± 6.641 | 29.718 ± 6.503 | 86.437 ± 45.802 |
| lm_batch_likelihood_sentence_3_12d | **23.400 ± 1.662** | 44.550 ± 11.195 | 39.406 ± 12.043 | 41.060 ± 4.268 | 120.480 ± 34.212 |
| lm_batch_likelihood_sentence_4_4d | **14.754 ± 0.792** | 25.538 ± 11.701 | 18.054 ± 6.471 | 22.326 ± 2.543 | 91.495 ± 28.676 |
| nary_matmul_chain_64 | 0.020 ± 0.002 | **0.010 ± 0.001** | 0.024 ± 0.001 | 0.222 ± 0.026 | 0.098 ± 0.012 |
| str_matrix_chain_multiplication_100 | **4.875 ± 0.161** | 6.998 ± 2.652 | 5.728 ± 0.770 | 24.538 ± 7.084 | 291.356 ± 100.274 |
| str_mps_varying_inner_product_200 | 15.582 ± 4.512 | 14.977 ± 3.281 | **9.126 ± 1.258** | 33.312 ± 4.143 | 27.380 ± 29.710 |
| str_nw_mera_closed_120 | 367.533 ± 6.795 | 370.763 ± 32.051 | **356.890 ± 57.580** | 648.899 ± 25.583 | 1451.161 ± 553.030 |
| str_nw_mera_open_26 | 299.887 ± 42.773 | 384.030 ± 90.354 | **236.959 ± 5.710** | 516.847 ± 87.206 | 880.478 ± 259.908 |
| tensornetwork_permutation_focus_step409_316 | **121.030 ± 3.241** | 187.205 ± 8.994 | 198.380 ± 9.434 | 294.794 ± 63.547 | 442.267 ± 96.584 |
| tensornetwork_permutation_light_415 | **110.610 ± 4.541** | 113.067 ± 6.482 | 178.751 ± 32.121 | 271.686 ± 28.680 | 428.250 ± 124.016 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 1.397 ± 0.208 | **0.647 ± 0.019** | 0.731 ± 0.025 | 2.790 ± 0.130 | 95.179 ± 44.401 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.325 ± 0.224 | 0.338 ± 0.257 | **0.138 ± 0.184** | 0.708 ± 0.024 | 0.763 ± 2.316 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.218 ± 0.067 | **0.200 ± 0.132** | 0.226 ± 0.010 | 0.500 ± 0.021 | 0.678 ± 0.548 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.372 ± 0.053 | **0.177 ± 0.032** | 0.228 ± 0.020 | 0.529 ± 0.025 | 1.518 ± 4.347 |
| bin_elementwise_mul_2048x2048 | 2.708 ± 0.265 | **2.641 ± 0.337** | 2.998 ± 0.106 | 3.304 ± 0.188 | 3.254 ± 0.462 |
| bin_elementwise_mul_512x512 | 0.123 ± 0.057 | 0.110 ± 0.037 | 0.414 ± 0.249 | 0.376 ± 0.034 | **0.078 ± 0.030** |
| bin_matmul_1024 | 14.398 ± 0.707 | **10.501 ± 0.380** | 14.194 ± 2.380 | 27.968 ± 2.840 | 48.682 ± 26.721 |
| bin_matmul_256 | 0.774 ± 0.557 | 0.120 ± 0.002 | **0.119 ± 0.001** | 0.880 ± 0.037 | 3.768 ± 1.912 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | **0.008 ± 0.006** | 0.011 ± 0.001 | 0.020 ± 0.004 | 0.087 ± 0.005 | 0.010 ± 0.007 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.029 ± 0.002 | **0.025 ± 0.002** | 0.082 ± 0.010 | 0.188 ± 0.032 | 0.082 ± 0.014 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.002 | **0.007 ± 0.001** | 0.017 ± 0.001 | 0.056 ± 0.133 | 0.030 ± 0.009 |
| bin_outer_product_4096 | 3.149 ± 1.014 | **2.969 ± 0.237** | 3.333 ± 0.710 | 3.147 ± 0.037 | 264.967 ± 46.005 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 15.393 ± 0.342 | 11.617 ± 0.526 | **10.688 ± 0.307** | 32.269 ± 5.219 | 70.066 ± 22.834 |
| gm_queen5_5_3.wcsp | 636.381 ± 21.399 | **586.948 ± 25.040** | 588.119 ± 20.478 | 809.706 ± 73.856 | 1045.933 ± 552.709 |
| lm_batch_likelihood_brackets_4_4d | **14.255 ± 0.525** | 25.582 ± 12.651 | 26.465 ± 2.301 | 20.745 ± 1.288 | 111.132 ± 22.909 |
| lm_batch_likelihood_sentence_3_12d | **25.555 ± 1.003** | 41.444 ± 2.717 | 27.409 ± 12.292 | 45.477 ± 5.357 | 137.638 ± 32.215 |
| lm_batch_likelihood_sentence_4_4d | **12.226 ± 0.823** | 29.571 ± 4.156 | 17.612 ± 1.235 | 29.769 ± 4.736 | 119.997 ± 37.924 |
| nary_matmul_chain_64 | 0.020 ± 0.002 | **0.010 ± 0.001** | 0.024 ± 0.001 | 0.222 ± 0.026 | 0.098 ± 0.012 |
| str_matrix_chain_multiplication_100 | **4.817 ± 0.177** | 5.444 ± 0.807 | 5.226 ± 0.095 | 19.727 ± 3.239 | 303.921 ± 43.355 |
| str_mps_varying_inner_product_200 | 16.713 ± 0.770 | 17.407 ± 2.164 | **11.954 ± 0.542** | 29.304 ± 6.293 | 25.995 ± 17.642 |
| str_nw_mera_closed_120 | 319.138 ± 53.519 | 336.084 ± 46.667 | **284.205 ± 8.660** | 641.385 ± 42.493 | 1026.616 ± 286.373 |
| str_nw_mera_open_26 | 278.498 ± 23.862 | 383.978 ± 70.793 | **242.997 ± 7.265** | 559.781 ± 40.679 | 903.164 ± 233.778 |
| tensornetwork_permutation_focus_step409_316 | **121.030 ± 3.241** | 187.205 ± 8.994 | 198.380 ± 9.434 | 294.794 ± 63.547 | 442.267 ± 96.584 |
| tensornetwork_permutation_light_415 | **110.610 ± 4.541** | 113.067 ± 6.482 | 178.751 ± 32.121 | 271.686 ± 28.680 | 428.250 ± 124.016 |

# CPU Einsum Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Benchmark commit: `9185e8ca26c3fc600458361d0f477495a717daf3`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261124_172410/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261124_172410/run.yaml`
- Timestamp: `2026-11-24T17:24:10.690570+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261124_172410`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

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

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.4`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.0.3.30.dylib)`, probe threads `1`

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


Logs:

- `data/results/mac-cpu/cpu/einsum/20261124_172410/jax_cpu_t1_20261124_172410.log`
- `data/results/mac-cpu/cpu/einsum/20261124_172410/julia_omeinsum_t1_20261124_172410.log`
- `data/results/mac-cpu/cpu/einsum/20261124_172410/pytorch_cpu_t1_20261124_172410.log`
- `data/results/mac-cpu/cpu/einsum/20261124_172410/tenferro_eager_t1_20261124_172410.log`
- `data/results/mac-cpu/cpu/einsum/20261124_172410/tenferro_trace_t1_20261124_172410.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.709 ± 0.668 | 3.521 ± 0.206 | **0.711 ± 0.034** | 4.798 ± 1.220 | 7.142 ± 6.526 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.233 ± 3.608 | 0.504 ± 0.048 | **0.119 ± 0.003** | 0.885 ± 0.177 | 1.397 ± 2.812 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.257 ± 0.017 | 0.255 ± 0.040 | **0.186 ± 0.004** | 0.518 ± 0.087 | 0.685 ± 0.382 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.103 ± 0.359 | 0.241 ± 0.006 | **0.186 ± 0.001** | 0.590 ± 0.076 | 1.573 ± 6.359 |
| bin_elementwise_mul_2048x2048 | 3.572 ± 1.774 | **2.242 ± 0.204** | 2.374 ± 0.271 | 3.305 ± 0.055 | 2.718 ± 0.585 |
| bin_elementwise_mul_512x512 | 0.094 ± 0.004 | 0.091 ± 0.019 | 0.110 ± 0.010 | 0.417 ± 0.095 | **0.064 ± 0.023** |
| bin_matmul_1024 | 63.308 ± 5.376 | 52.704 ± 1.350 | **12.280 ± 1.096** | 61.861 ± 4.277 | 66.524 ± 14.587 |
| bin_matmul_256 | 0.914 ± 0.026 | 0.857 ± 0.015 | **0.131 ± 0.011** | 1.094 ± 0.056 | 1.027 ± 0.050 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.005 ± 0.001** | 0.035 ± 0.011 | 0.104 ± 0.008 | 0.009 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.031 ± 0.007 | **0.023 ± 0.002** | 0.094 ± 0.033 | 0.140 ± 0.045 | 0.095 ± 0.016 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.008 ± 0.001** | 0.017 ± 0.007 | 0.072 ± 0.012 | 0.029 ± 0.009 |
| bin_outer_product_4096 | 5.287 ± 0.303 | 5.496 ± 0.543 | 3.918 ± 0.171 | **2.896 ± 0.379** | 257.664 ± 32.894 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 113.998 ± 15.949 | 99.518 ± 14.117 | **11.894 ± 0.470** | 65.371 ± 1.222 | 61.318 ± 11.491 |
| gm_queen5_5_3.wcsp | 5631.472 ± 393.383 | 5071.390 ± 429.848 | **4035.672 ± 192.899** | 5721.293 ± 342.640 | 7662.013 ± 576.593 |
| lm_batch_likelihood_brackets_4_4d | 27.793 ± 2.507 | 25.520 ± 0.576 | 14.075 ± 0.588 | **12.344 ± 0.452** | 22.167 ± 29.034 |
| lm_batch_likelihood_sentence_3_12d | 92.390 ± 16.622 | 89.171 ± 2.615 | **27.239 ± 2.456** | 59.387 ± 6.269 | 74.063 ± 18.855 |
| lm_batch_likelihood_sentence_4_4d | 28.006 ± 3.189 | 27.611 ± 0.912 | **14.935 ± 0.262** | 16.550 ± 1.574 | 26.952 ± 6.777 |
| nary_matmul_chain_64 | 0.040 ± 0.007 | 0.034 ± 0.002 | **0.023 ± 0.002** | 0.166 ± 0.068 | 0.055 ± 0.068 |
| str_matrix_chain_multiplication_100 | 15.279 ± 1.705 | 13.163 ± 0.161 | **4.647 ± 0.116** | 15.371 ± 0.810 | 17.217 ± 6.992 |
| str_mps_varying_inner_product_200 | 21.136 ± 3.272 | 23.766 ± 8.876 | **6.516 ± 0.053** | 16.890 ± 0.130 | 19.251 ± 2.862 |
| str_nw_mera_closed_120 | 2474.344 ± 158.107 | 2114.753 ± 43.685 | **363.459 ± 32.427** | 2083.583 ± 395.894 | 1328.441 ± 81.410 |
| str_nw_mera_open_26 | 1586.074 ± 118.841 | 1484.820 ± 54.470 | **239.824 ± 16.701** | 1326.355 ± 174.356 | 1236.475 ± 367.428 |
| tensornetwork_permutation_focus_step409_316 | 280.241 ± 18.644 | 321.145 ± 28.849 | 262.668 ± 12.180 | 492.188 ± 253.639 | **246.321 ± 56.335** |
| tensornetwork_permutation_light_415 | 282.371 ± 14.048 | 264.565 ± 12.037 | **220.250 ± 14.480** | 566.599 ± 120.784 | 255.119 ± 46.247 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 4.709 ± 0.668 | 3.521 ± 0.206 | **0.711 ± 0.034** | 4.798 ± 1.220 | 7.142 ± 6.526 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.233 ± 3.608 | 0.504 ± 0.048 | **0.119 ± 0.003** | 0.885 ± 0.177 | 1.397 ± 2.812 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.257 ± 0.017 | 0.255 ± 0.040 | **0.186 ± 0.004** | 0.518 ± 0.087 | 0.685 ± 0.382 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 1.103 ± 0.359 | 0.241 ± 0.006 | **0.186 ± 0.001** | 0.590 ± 0.076 | 1.573 ± 6.359 |
| bin_elementwise_mul_2048x2048 | 3.572 ± 1.774 | **2.242 ± 0.204** | 2.374 ± 0.271 | 3.305 ± 0.055 | 2.718 ± 0.585 |
| bin_elementwise_mul_512x512 | 0.094 ± 0.004 | 0.091 ± 0.019 | 0.110 ± 0.010 | 0.417 ± 0.095 | **0.064 ± 0.023** |
| bin_matmul_1024 | 63.308 ± 5.376 | 52.704 ± 1.350 | **12.280 ± 1.096** | 61.861 ± 4.277 | 66.524 ± 14.587 |
| bin_matmul_256 | 0.914 ± 0.026 | 0.857 ± 0.015 | **0.131 ± 0.011** | 1.094 ± 0.056 | 1.027 ± 0.050 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.007 ± 0.000 | **0.005 ± 0.001** | 0.035 ± 0.011 | 0.104 ± 0.008 | 0.009 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.031 ± 0.007 | **0.023 ± 0.002** | 0.094 ± 0.033 | 0.140 ± 0.045 | 0.095 ± 0.016 |
| bin_omeinsum_matmul_10x10 | 0.015 ± 0.001 | **0.008 ± 0.001** | 0.017 ± 0.007 | 0.072 ± 0.012 | 0.029 ± 0.009 |
| bin_outer_product_4096 | 5.287 ± 0.303 | 5.496 ± 0.543 | 3.918 ± 0.171 | **2.896 ± 0.379** | 257.664 ± 32.894 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 113.998 ± 15.949 | 99.518 ± 14.117 | **11.894 ± 0.470** | 65.371 ± 1.222 | 61.318 ± 11.491 |
| gm_queen5_5_3.wcsp | 1796.213 ± 214.161 | 1676.314 ± 207.914 | **731.267 ± 67.775** | 1477.285 ± 178.070 | 2472.129 ± 126.551 |
| lm_batch_likelihood_brackets_4_4d | 24.383 ± 2.637 | 23.808 ± 0.350 | **15.195 ± 0.891** | 15.827 ± 1.257 | 28.630 ± 11.418 |
| lm_batch_likelihood_sentence_3_12d | 94.473 ± 11.721 | 94.061 ± 3.428 | **26.282 ± 2.628** | 55.161 ± 6.411 | 90.909 ± 39.736 |
| lm_batch_likelihood_sentence_4_4d | 23.019 ± 2.467 | 23.659 ± 0.305 | 14.065 ± 0.451 | **13.153 ± 0.574** | 25.382 ± 35.015 |
| nary_matmul_chain_64 | 0.040 ± 0.007 | 0.034 ± 0.002 | **0.023 ± 0.002** | 0.166 ± 0.068 | 0.055 ± 0.068 |
| str_matrix_chain_multiplication_100 | 13.633 ± 0.444 | 13.188 ± 0.244 | **5.431 ± 0.680** | 14.273 ± 1.334 | 23.326 ± 8.359 |
| str_mps_varying_inner_product_200 | 24.068 ± 2.142 | 21.234 ± 0.772 | **9.191 ± 0.522** | 24.603 ± 2.023 | 18.456 ± 2.934 |
| str_nw_mera_closed_120 | 2060.419 ± 80.152 | 2039.366 ± 52.516 | **286.015 ± 11.833** | 1990.959 ± 271.418 | 1617.338 ± 329.241 |
| str_nw_mera_open_26 | 1576.724 ± 260.934 | 1550.841 ± 131.902 | **244.795 ± 20.581** | 1333.395 ± 139.596 | 1170.380 ± 95.058 |
| tensornetwork_permutation_focus_step409_316 | 280.241 ± 18.644 | 321.145 ± 28.849 | 262.668 ± 12.180 | 492.188 ± 253.639 | **246.321 ± 56.335** |
| tensornetwork_permutation_light_415 | 282.371 ± 14.048 | 264.565 ± 12.037 | **220.250 ± 14.480** | 566.599 ± 120.784 | 255.119 ± 46.247 |

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261124_175807/report.md`.


- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261124_175807/run.yaml`
- Timestamp: `2026-11-24T17:58:07.132443+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20261124_175807`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

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

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.4`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.0.3.30.dylib)`, probe threads `4`

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


Logs:

- `data/results/mac-cpu/cpu/einsum/20261124_175807/jax_cpu_t4_20261124_175807.log`
- `data/results/mac-cpu/cpu/einsum/20261124_175807/julia_omeinsum_t4_20261124_175807.log`
- `data/results/mac-cpu/cpu/einsum/20261124_175807/pytorch_cpu_t4_20261124_175807.log`
- `data/results/mac-cpu/cpu/einsum/20261124_175807/tenferro_eager_t4_20261124_175807.log`
- `data/results/mac-cpu/cpu/einsum/20261124_175807/tenferro_trace_t4_20261124_175807.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.329 ± 1.097 | 4.657 ± 0.454 | **0.931 ± 0.210** | 3.111 ± 0.811 | 195.484 ± 139.019 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.836 ± 0.150 | 0.647 ± 0.055 | **0.135 ± 0.004** | 0.825 ± 0.107 | 2.271 ± 4.473 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.214 ± 0.050 | **0.176 ± 0.039** | 0.283 ± 0.082 | 0.509 ± 0.088 | 0.685 ± 0.576 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.324 ± 0.027 | **0.210 ± 0.081** | 0.251 ± 0.063 | 0.707 ± 0.098 | 2.001 ± 3.880 |
| bin_elementwise_mul_2048x2048 | 3.136 ± 0.356 | 2.972 ± 0.460 | 3.141 ± 0.464 | 3.715 ± 0.571 | **2.925 ± 0.712** |
| bin_elementwise_mul_512x512 | 0.117 ± 0.038 | 0.167 ± 0.125 | 0.197 ± 0.082 | 0.439 ± 0.062 | **0.081 ± 0.014** |
| bin_matmul_1024 | 27.329 ± 2.890 | 27.214 ± 3.587 | **14.504 ± 1.605** | 29.432 ± 4.064 | 48.525 ± 11.858 |
| bin_matmul_256 | 0.898 ± 0.091 | 0.758 ± 0.205 | **0.127 ± 0.002** | 1.025 ± 0.142 | 3.513 ± 1.823 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.001 | **0.005 ± 0.000** | 0.018 ± 0.001 | 0.086 ± 0.004 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.025 ± 0.004** | **0.025 ± 0.001** | 0.075 ± 0.003 | 0.198 ± 0.025 | 0.100 ± 0.010 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.002 | **0.008 ± 0.000** | 0.016 ± 0.001 | 0.153 ± 0.009 | 0.017 ± 0.005 |
| bin_outer_product_4096 | **3.316 ± 0.310** | 3.468 ± 0.314 | 3.939 ± 0.838 | 3.441 ± 0.772 | 228.068 ± 56.384 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 49.783 ± 8.947 | 51.030 ± 4.100 | **12.617 ± 1.763** | 31.585 ± 5.216 | 43.200 ± 14.425 |
| gm_queen5_5_3.wcsp | **3258.157 ± 572.196** | 4887.441 ± 314.441 | 3928.081 ± 168.715 | 6178.398 ± 510.758 | 10156.091 ± 479.016 |
| lm_batch_likelihood_brackets_4_4d | 21.751 ± 1.393 | 28.360 ± 5.995 | **19.304 ± 5.139** | 22.768 ± 1.904 | 178.421 ± 64.084 |
| lm_batch_likelihood_sentence_3_12d | 52.746 ± 2.058 | 67.438 ± 11.081 | **30.515 ± 2.974** | 48.426 ± 4.367 | 104.645 ± 22.431 |
| lm_batch_likelihood_sentence_4_4d | 23.850 ± 0.613 | 28.531 ± 3.859 | **20.072 ± 2.140** | 23.619 ± 3.104 | 88.546 ± 9.486 |
| nary_matmul_chain_64 | 0.101 ± 0.036 | 0.081 ± 0.009 | **0.030 ± 0.007** | 0.174 ± 0.108 | 0.109 ± 0.016 |
| str_matrix_chain_multiplication_100 | 14.842 ± 0.536 | 17.112 ± 1.209 | **6.382 ± 1.537** | 19.437 ± 1.165 | 318.471 ± 76.860 |
| str_mps_varying_inner_product_200 | 36.189 ± 2.284 | 40.726 ± 2.836 | **10.730 ± 1.396** | 35.775 ± 10.962 | 65.626 ± 22.372 |
| str_nw_mera_closed_120 | 1278.087 ± 149.291 | 1359.999 ± 137.392 | **391.662 ± 22.715** | 709.279 ± 261.058 | 1090.119 ± 190.934 |
| str_nw_mera_open_26 | 801.061 ± 47.724 | 1012.070 ± 216.410 | **280.519 ± 23.083** | 555.390 ± 125.079 | 874.652 ± 103.278 |
| tensornetwork_permutation_focus_step409_316 | **169.195 ± 53.184** | 200.531 ± 21.946 | 198.537 ± 21.341 | 291.312 ± 49.358 | 448.590 ± 60.600 |
| tensornetwork_permutation_light_415 | **150.293 ± 15.780** | 173.481 ± 40.544 | 167.124 ± 16.432 | 299.619 ± 12.143 | 443.709 ± 57.582 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 5.329 ± 1.097 | 4.657 ± 0.454 | **0.931 ± 0.210** | 3.111 ± 0.811 | 195.484 ± 139.019 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.836 ± 0.150 | 0.647 ± 0.055 | **0.135 ± 0.004** | 0.825 ± 0.107 | 2.271 ± 4.473 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.214 ± 0.050 | **0.176 ± 0.039** | 0.283 ± 0.082 | 0.509 ± 0.088 | 0.685 ± 0.576 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.324 ± 0.027 | **0.210 ± 0.081** | 0.251 ± 0.063 | 0.707 ± 0.098 | 2.001 ± 3.880 |
| bin_elementwise_mul_2048x2048 | 3.136 ± 0.356 | 2.972 ± 0.460 | 3.141 ± 0.464 | 3.715 ± 0.571 | **2.925 ± 0.712** |
| bin_elementwise_mul_512x512 | 0.117 ± 0.038 | 0.167 ± 0.125 | 0.197 ± 0.082 | 0.439 ± 0.062 | **0.081 ± 0.014** |
| bin_matmul_1024 | 27.329 ± 2.890 | 27.214 ± 3.587 | **14.504 ± 1.605** | 29.432 ± 4.064 | 48.525 ± 11.858 |
| bin_matmul_256 | 0.898 ± 0.091 | 0.758 ± 0.205 | **0.127 ± 0.002** | 1.025 ± 0.142 | 3.513 ± 1.823 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.001 | **0.005 ± 0.000** | 0.018 ± 0.001 | 0.086 ± 0.004 | 0.010 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | **0.025 ± 0.004** | **0.025 ± 0.001** | 0.075 ± 0.003 | 0.198 ± 0.025 | 0.100 ± 0.010 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.002 | **0.008 ± 0.000** | 0.016 ± 0.001 | 0.153 ± 0.009 | 0.017 ± 0.005 |
| bin_outer_product_4096 | **3.316 ± 0.310** | 3.468 ± 0.314 | 3.939 ± 0.838 | 3.441 ± 0.772 | 228.068 ± 56.384 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 49.783 ± 8.947 | 51.030 ± 4.100 | **12.617 ± 1.763** | 31.585 ± 5.216 | 43.200 ± 14.425 |
| gm_queen5_5_3.wcsp | 1086.316 ± 67.057 | 1254.605 ± 308.949 | **672.993 ± 79.207** | 942.156 ± 184.920 | 2207.772 ± 524.951 |
| lm_batch_likelihood_brackets_4_4d | 28.047 ± 8.112 | 26.976 ± 10.572 | **18.348 ± 1.143** | 26.334 ± 10.446 | 182.399 ± 93.872 |
| lm_batch_likelihood_sentence_3_12d | 52.792 ± 2.858 | 89.253 ± 14.967 | **29.858 ± 4.310** | 45.290 ± 2.108 | 128.185 ± 41.100 |
| lm_batch_likelihood_sentence_4_4d | 20.781 ± 0.872 | 28.958 ± 4.651 | **20.025 ± 1.895** | 24.102 ± 3.980 | 113.053 ± 30.566 |
| nary_matmul_chain_64 | 0.101 ± 0.036 | 0.081 ± 0.009 | **0.030 ± 0.007** | 0.174 ± 0.108 | 0.109 ± 0.016 |
| str_matrix_chain_multiplication_100 | 15.556 ± 1.470 | 17.716 ± 1.698 | **6.195 ± 1.106** | 18.178 ± 1.847 | 270.157 ± 43.420 |
| str_mps_varying_inner_product_200 | 44.232 ± 0.915 | 51.585 ± 3.573 | **15.156 ± 2.762** | 35.471 ± 2.465 | 62.665 ± 28.279 |
| str_nw_mera_closed_120 | 1261.009 ± 171.895 | 1373.998 ± 88.485 | **370.860 ± 14.681** | 628.984 ± 12.078 | 861.731 ± 81.811 |
| str_nw_mera_open_26 | 814.072 ± 48.451 | 976.972 ± 75.160 | **277.404 ± 11.990** | 561.712 ± 62.783 | 964.004 ± 178.410 |
| tensornetwork_permutation_focus_step409_316 | **169.195 ± 53.184** | 200.531 ± 21.946 | 198.537 ± 21.341 | 291.312 ± 49.358 | 448.590 ± 60.600 |
| tensornetwork_permutation_light_415 | **150.293 ± 15.780** | 173.481 ± 40.544 | 167.124 ± 16.432 | 299.619 ± 12.143 | 443.709 ± 57.582 |

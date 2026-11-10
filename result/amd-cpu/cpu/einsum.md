# CPU Einsum Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `15cf8f674203aeca079999fdb339cf4d35801b88`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261110_180539/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261110_180539/run.yaml`
- Timestamp: `2026-11-10T18:05:39.033513+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261110_180539`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

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

- tenferro-rs features: `blas-mkl`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.2`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.3`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `1`

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

- `data/results/amd-cpu/cpu/einsum/20261110_180539/jax_cpu_t1_20261110_180539.log`
- `data/results/amd-cpu/cpu/einsum/20261110_180539/julia_omeinsum_t1_20261110_180539.log`
- `data/results/amd-cpu/cpu/einsum/20261110_180539/pytorch_cpu_t1_20261110_180539.log`
- `data/results/amd-cpu/cpu/einsum/20261110_180539/tenferro_eager_t1_20261110_180539.log`
- `data/results/amd-cpu/cpu/einsum/20261110_180539/tenferro_trace_t1_20261110_180539.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.964 ± 0.148 | 3.514 ± 0.083 | **3.512 ± 0.018** | 3.793 ± 0.133 | 10.743 ± 6.791 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.626 ± 0.007 | **0.504 ± 0.014** | 0.548 ± 0.013 | 0.627 ± 0.026 | 0.629 ± 1.473 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.642 ± 0.015 | 0.624 ± 0.009 | **0.120 ± 0.003** | 0.480 ± 0.040 | 1.705 ± 2.477 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.945 ± 0.004 | 0.621 ± 0.009 | **0.126 ± 0.006** | 0.455 ± 0.021 | 5.195 ± 2.864 |
| bin_elementwise_mul_2048x2048 | 2.072 ± 0.094 | **2.033 ± 0.122** | 13.711 ± 0.480 | 15.652 ± 0.168 | 17.173 ± 2.315 |
| bin_elementwise_mul_512x512 | 0.073 ± 0.001 | **0.071 ± 0.001** | 0.081 ± 0.005 | 0.266 ± 0.022 | 0.325 ± 0.222 |
| bin_matmul_1024 | 50.883 ± 0.381 | **43.293 ± 0.513** | 51.585 ± 0.184 | 47.333 ± 0.694 | 60.291 ± 23.084 |
| bin_matmul_256 | 0.930 ± 0.038 | **0.691 ± 0.009** | 1.230 ± 0.006 | 0.859 ± 0.011 | 1.090 ± 0.009 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.005 ± 0.002** | 0.021 ± 0.011 | 0.133 ± 0.018 | 0.011 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.032 ± 0.001 | **0.029 ± 0.002** | 0.095 ± 0.008 | 0.203 ± 0.028 | 0.065 ± 0.027 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.000 | **0.013 ± 0.001** | 0.019 ± 0.001 | 0.102 ± 0.014 | 0.019 ± 0.001 |
| bin_outer_product_4096 | 52.234 ± 0.278 | 52.534 ± 0.268 | **45.597 ± 1.109** | 47.341 ± 2.080 | 272.164 ± 4.663 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 48.700 ± 0.625 | **44.717 ± 0.886** | 53.944 ± 1.500 | 49.837 ± 0.422 | 54.440 ± 14.345 |
| gm_queen5_5_3.wcsp | 6059.983 ± 117.590 | 6178.337 ± 51.433 | **5581.898 ± 74.158** | 5992.468 ± 72.671 | 8271.279 ± 84.624 |
| lm_batch_likelihood_brackets_4_4d | 22.812 ± 0.384 | 26.943 ± 0.919 | 26.788 ± 1.332 | **13.160 ± 3.631** | 52.238 ± 39.083 |
| lm_batch_likelihood_sentence_3_12d | **52.373 ± 3.608** | 110.311 ± 25.024 | 60.017 ± 1.719 | 86.163 ± 2.657 | 109.131 ± 24.131 |
| lm_batch_likelihood_sentence_4_4d | 26.099 ± 1.386 | 30.267 ± 1.481 | 31.273 ± 1.678 | **14.513 ± 0.691** | 29.776 ± 25.555 |
| nary_matmul_chain_64 | 0.045 ± 0.001 | **0.037 ± 0.000** | 0.050 ± 0.003 | 0.116 ± 0.011 | 0.089 ± 0.057 |
| str_matrix_chain_multiplication_100 | **13.102 ± 0.414** | 14.739 ± 0.130 | 16.816 ± 0.751 | 13.195 ± 0.121 | 15.338 ± 0.619 |
| str_mps_varying_inner_product_200 | **17.627 ± 0.225** | 20.558 ± 0.205 | 21.472 ± 1.769 | 21.878 ± 6.123 | 24.496 ± 7.242 |
| str_nw_mera_closed_120 | **1473.663 ± 18.809** | 1605.726 ± 25.069 | 1736.907 ± 10.268 | 1892.513 ± 16.160 | 1737.683 ± 43.581 |
| str_nw_mera_open_26 | 1123.036 ± 19.244 | 1341.317 ± 14.352 | **1074.500 ± 21.712** | 1555.708 ± 33.797 | 1553.614 ± 52.892 |
| tensornetwork_permutation_focus_step409_316 | **405.803 ± 4.234** | 501.212 ± 4.370 | 720.984 ± 3.102 | 759.330 ± 60.054 | 555.558 ± 30.072 |
| tensornetwork_permutation_light_415 | **399.627 ± 1.106** | 439.386 ± 3.364 | 689.501 ± 5.334 | 734.356 ± 117.671 | 552.434 ± 54.857 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 6.964 ± 0.148 | 3.514 ± 0.083 | **3.512 ± 0.018** | 3.793 ± 0.133 | 10.743 ± 6.791 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.626 ± 0.007 | **0.504 ± 0.014** | 0.548 ± 0.013 | 0.627 ± 0.026 | 0.629 ± 1.473 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.642 ± 0.015 | 0.624 ± 0.009 | **0.120 ± 0.003** | 0.480 ± 0.040 | 1.705 ± 2.477 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.945 ± 0.004 | 0.621 ± 0.009 | **0.126 ± 0.006** | 0.455 ± 0.021 | 5.195 ± 2.864 |
| bin_elementwise_mul_2048x2048 | 2.072 ± 0.094 | **2.033 ± 0.122** | 13.711 ± 0.480 | 15.652 ± 0.168 | 17.173 ± 2.315 |
| bin_elementwise_mul_512x512 | 0.073 ± 0.001 | **0.071 ± 0.001** | 0.081 ± 0.005 | 0.266 ± 0.022 | 0.325 ± 0.222 |
| bin_matmul_1024 | 50.883 ± 0.381 | **43.293 ± 0.513** | 51.585 ± 0.184 | 47.333 ± 0.694 | 60.291 ± 23.084 |
| bin_matmul_256 | 0.930 ± 0.038 | **0.691 ± 0.009** | 1.230 ± 0.006 | 0.859 ± 0.011 | 1.090 ± 0.009 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.008 ± 0.000 | **0.005 ± 0.002** | 0.021 ± 0.011 | 0.133 ± 0.018 | 0.011 ± 0.001 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.032 ± 0.001 | **0.029 ± 0.002** | 0.095 ± 0.008 | 0.203 ± 0.028 | 0.065 ± 0.027 |
| bin_omeinsum_matmul_10x10 | 0.016 ± 0.000 | **0.013 ± 0.001** | 0.019 ± 0.001 | 0.102 ± 0.014 | 0.019 ± 0.001 |
| bin_outer_product_4096 | 52.234 ± 0.278 | 52.534 ± 0.268 | **45.597 ± 1.109** | 47.341 ± 2.080 | 272.164 ± 4.663 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 48.700 ± 0.625 | **44.717 ± 0.886** | 53.944 ± 1.500 | 49.837 ± 0.422 | 54.440 ± 14.345 |
| gm_queen5_5_3.wcsp | **2050.619 ± 43.527** | 2398.303 ± 24.564 | 2473.578 ± 115.898 | 2245.659 ± 10.762 | 3018.382 ± 85.580 |
| lm_batch_likelihood_brackets_4_4d | 23.955 ± 0.321 | 27.141 ± 0.866 | 28.604 ± 0.421 | **14.550 ± 1.034** | 34.571 ± 29.242 |
| lm_batch_likelihood_sentence_3_12d | **54.080 ± 3.023** | 142.575 ± 0.875 | 66.049 ± 1.199 | 109.459 ± 3.222 | 104.299 ± 24.081 |
| lm_batch_likelihood_sentence_4_4d | 25.851 ± 0.885 | 32.412 ± 1.306 | 29.448 ± 0.232 | **14.196 ± 0.292** | 33.041 ± 38.021 |
| nary_matmul_chain_64 | 0.045 ± 0.001 | **0.037 ± 0.000** | 0.050 ± 0.003 | 0.116 ± 0.011 | 0.089 ± 0.057 |
| str_matrix_chain_multiplication_100 | 14.449 ± 0.094 | 14.750 ± 0.126 | 15.038 ± 0.151 | **13.659 ± 0.196** | 15.117 ± 0.660 |
| str_mps_varying_inner_product_200 | 20.651 ± 0.154 | 20.872 ± 0.134 | **20.105 ± 0.182** | 24.386 ± 0.155 | 21.577 ± 1.413 |
| str_nw_mera_closed_120 | **1325.675 ± 18.695** | 1334.369 ± 15.390 | 1380.947 ± 14.537 | 1481.211 ± 19.955 | 1447.626 ± 70.411 |
| str_nw_mera_open_26 | 1133.143 ± 10.529 | 1307.510 ± 13.566 | **1100.110 ± 21.847** | 1605.735 ± 23.095 | 1547.614 ± 64.341 |
| tensornetwork_permutation_focus_step409_316 | **405.803 ± 4.234** | 501.212 ± 4.370 | 720.984 ± 3.102 | 759.330 ± 60.054 | 555.558 ± 30.072 |
| tensornetwork_permutation_light_415 | **399.627 ± 1.106** | 439.386 ± 3.364 | 689.501 ± 5.334 | 734.356 ± 117.671 | 552.434 ± 54.857 |

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261110_184045/report.md`.


- Suite: `cpu/einsum`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261110_184045/run.yaml`
- Timestamp: `2026-11-10T18:40:45.384615+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/amd-cpu/cpu/einsum/20261110_184045`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

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

- tenferro-rs features: `blas-mkl`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.2`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

### Julia / OMEinsum.jl Backend

- Julia: version `1.13.3`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.so)`, probe threads `4`

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

- `data/results/amd-cpu/cpu/einsum/20261110_184045/jax_cpu_t4_20261110_184045.log`
- `data/results/amd-cpu/cpu/einsum/20261110_184045/julia_omeinsum_t4_20261110_184045.log`
- `data/results/amd-cpu/cpu/einsum/20261110_184045/pytorch_cpu_t4_20261110_184045.log`
- `data/results/amd-cpu/cpu/einsum/20261110_184045/tenferro_eager_t4_20261110_184045.log`
- `data/results/amd-cpu/cpu/einsum/20261110_184045/tenferro_trace_t4_20261110_184045.log`

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 8.901 ± 0.055 | 4.580 ± 0.048 | **2.321 ± 2.820** | 2.401 ± 0.632 | 10.082 ± 13.792 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.629 ± 0.008 | 1.176 ± 0.018 | **0.295 ± 0.012** | 0.576 ± 0.014 | 0.610 ± 0.339 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.241 ± 0.010 | 0.239 ± 0.012 | **0.079 ± 0.010** | 0.467 ± 0.013 | 2.236 ± 2.333 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.242 ± 0.007 | 0.237 ± 0.015 | **0.077 ± 0.002** | 0.342 ± 0.016 | 5.635 ± 5.217 |
| bin_elementwise_mul_2048x2048 | **0.822 ± 0.073** | 0.860 ± 0.101 | 5.495 ± 0.155 | 8.276 ± 0.354 | 16.201 ± 0.550 |
| bin_elementwise_mul_512x512 | 0.048 ± 0.008 | **0.044 ± 0.000** | 0.065 ± 0.021 | 0.230 ± 0.019 | 0.124 ± 0.794 |
| bin_matmul_1024 | 55.222 ± 0.625 | 45.501 ± 0.908 | 30.901 ± 0.437 | **26.381 ± 0.546** | 42.946 ± 30.173 |
| bin_matmul_256 | 1.152 ± 0.020 | 0.797 ± 0.163 | 0.825 ± 0.017 | **0.615 ± 0.050** | 0.716 ± 0.016 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.007 ± 0.000** | 0.033 ± 0.008 | 0.125 ± 0.015 | 0.010 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.046 ± 0.001 | 0.044 ± 0.002 | 0.136 ± 0.015 | 0.176 ± 0.023 | **0.036 ± 0.002** |
| bin_omeinsum_matmul_10x10 | 0.047 ± 0.002 | 0.035 ± 0.002 | 0.029 ± 0.001 | 0.106 ± 0.022 | **0.019 ± 0.001** |
| bin_outer_product_4096 | 17.503 ± 0.116 | 17.511 ± 0.131 | **15.603 ± 0.923** | 24.288 ± 2.727 | 255.472 ± 2.454 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 53.184 ± 0.563 | 47.991 ± 0.989 | **25.879 ± 0.499** | 28.389 ± 1.460 | 40.555 ± 19.317 |
| gm_queen5_5_3.wcsp | 5375.566 ± 42.671 | 4894.099 ± 79.738 | **2307.025 ± 22.754** | 3778.099 ± 70.318 | 7491.218 ± 96.477 |
| lm_batch_likelihood_brackets_4_4d | 26.292 ± 0.633 | 40.514 ± 2.423 | **13.582 ± 0.524** | 18.358 ± 1.114 | 40.429 ± 43.278 |
| lm_batch_likelihood_sentence_3_12d | 57.525 ± 0.375 | 163.775 ± 6.023 | **31.700 ± 1.096** | 44.177 ± 5.025 | 105.154 ± 18.922 |
| lm_batch_likelihood_sentence_4_4d | 28.925 ± 0.360 | 44.729 ± 1.448 | **14.418 ± 0.343** | 19.024 ± 0.850 | 39.951 ± 31.280 |
| nary_matmul_chain_64 | 0.112 ± 0.002 | 0.100 ± 0.009 | **0.063 ± 0.002** | 0.114 ± 0.012 | 0.116 ± 0.041 |
| str_matrix_chain_multiplication_100 | 15.011 ± 0.129 | 17.194 ± 0.070 | **9.666 ± 0.104** | 13.294 ± 0.695 | 12.470 ± 1.390 |
| str_mps_varying_inner_product_200 | 22.636 ± 0.118 | 25.055 ± 0.124 | **22.052 ± 0.168** | 25.247 ± 0.413 | 25.414 ± 18.298 |
| str_nw_mera_closed_120 | 1501.059 ± 4.410 | 1650.638 ± 42.404 | **834.589 ± 10.863** | 895.963 ± 26.457 | 1125.258 ± 43.619 |
| str_nw_mera_open_26 | 1142.622 ± 31.899 | 1199.423 ± 12.799 | **510.494 ± 5.664** | 759.835 ± 7.796 | 1173.636 ± 49.782 |
| tensornetwork_permutation_focus_step409_316 | 409.541 ± 2.897 | 445.575 ± 5.764 | **294.300 ± 1.352** | 387.180 ± 14.914 | 502.159 ± 54.703 |
| tensornetwork_permutation_light_415 | 405.274 ± 3.883 | 406.617 ± 1.103 | **287.406 ± 2.962** | 394.744 ± 38.200 | 479.409 ± 60.250 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 8.901 ± 0.055 | 4.580 ± 0.048 | **2.321 ± 2.820** | 2.401 ± 0.632 | 10.082 ± 13.792 |
| bin_batched_matmul_b32_m64_n64_k64 | 1.629 ± 0.008 | 1.176 ± 0.018 | **0.295 ± 0.012** | 0.576 ± 0.014 | 0.610 ± 0.339 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.241 ± 0.010 | 0.239 ± 0.012 | **0.079 ± 0.010** | 0.467 ± 0.013 | 2.236 ± 2.333 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.242 ± 0.007 | 0.237 ± 0.015 | **0.077 ± 0.002** | 0.342 ± 0.016 | 5.635 ± 5.217 |
| bin_elementwise_mul_2048x2048 | **0.822 ± 0.073** | 0.860 ± 0.101 | 5.495 ± 0.155 | 8.276 ± 0.354 | 16.201 ± 0.550 |
| bin_elementwise_mul_512x512 | 0.048 ± 0.008 | **0.044 ± 0.000** | 0.065 ± 0.021 | 0.230 ± 0.019 | 0.124 ± 0.794 |
| bin_matmul_1024 | 55.222 ± 0.625 | 45.501 ± 0.908 | 30.901 ± 0.437 | **26.381 ± 0.546** | 42.946 ± 30.173 |
| bin_matmul_256 | 1.152 ± 0.020 | 0.797 ± 0.163 | 0.825 ± 0.017 | **0.615 ± 0.050** | 0.716 ± 0.016 |
| bin_omeinsum_batched_matmul_8x8_batch_4 | 0.012 ± 0.000 | **0.007 ± 0.000** | 0.033 ± 0.008 | 0.125 ± 0.015 | 0.010 ± 0.002 |
| bin_omeinsum_high_d_12x12_contract_4_batch_4 | 0.046 ± 0.001 | 0.044 ± 0.002 | 0.136 ± 0.015 | 0.176 ± 0.023 | **0.036 ± 0.002** |
| bin_omeinsum_matmul_10x10 | 0.047 ± 0.002 | 0.035 ± 0.002 | 0.029 ± 0.001 | 0.106 ± 0.022 | **0.019 ± 0.001** |
| bin_outer_product_4096 | 17.503 ± 0.116 | 17.511 ± 0.131 | **15.603 ± 0.923** | 24.288 ± 2.727 | 255.472 ± 2.454 |
| bin_permuted_r4_abcd_dbef_acef_d32 | 53.184 ± 0.563 | 47.991 ± 0.989 | **25.879 ± 0.499** | 28.389 ± 1.460 | 40.555 ± 19.317 |
| gm_queen5_5_3.wcsp | 1994.570 ± 15.475 | 1840.358 ± 21.949 | **1034.557 ± 11.845** | 1492.789 ± 18.281 | 2700.789 ± 30.527 |
| lm_batch_likelihood_brackets_4_4d | 26.465 ± 0.491 | 41.613 ± 1.089 | **13.656 ± 0.243** | 19.067 ± 1.522 | 43.263 ± 37.258 |
| lm_batch_likelihood_sentence_3_12d | 54.621 ± 0.591 | 186.662 ± 1.976 | 51.477 ± 1.813 | **42.142 ± 2.108** | 106.439 ± 24.008 |
| lm_batch_likelihood_sentence_4_4d | 28.777 ± 0.751 | 47.757 ± 2.482 | **14.747 ± 0.398** | 19.721 ± 0.872 | 43.269 ± 37.483 |
| nary_matmul_chain_64 | 0.112 ± 0.002 | 0.100 ± 0.009 | **0.063 ± 0.002** | 0.114 ± 0.012 | 0.116 ± 0.041 |
| str_matrix_chain_multiplication_100 | 16.637 ± 0.177 | 17.505 ± 0.236 | **9.697 ± 0.083** | 12.938 ± 0.459 | 11.731 ± 0.430 |
| str_mps_varying_inner_product_200 | 25.838 ± 0.166 | 26.463 ± 0.206 | 24.893 ± 0.469 | 27.769 ± 0.611 | **21.918 ± 6.365** |
| str_nw_mera_closed_120 | 1357.894 ± 8.536 | 1343.332 ± 17.602 | **668.095 ± 29.200** | 736.885 ± 24.492 | 888.690 ± 56.048 |
| str_nw_mera_open_26 | 1137.923 ± 23.135 | 1165.478 ± 3.544 | **519.600 ± 4.167** | 789.396 ± 17.402 | 1102.656 ± 55.233 |
| tensornetwork_permutation_focus_step409_316 | 409.541 ± 2.897 | 445.575 ± 5.764 | **294.300 ± 1.352** | 387.180 ± 14.914 | 502.159 ± 54.703 |
| tensornetwork_permutation_light_415 | 405.274 ± 3.883 | 406.617 ± 1.103 | **287.406 ± 2.962** | 394.744 ± 38.200 | 479.409 ± 60.250 |

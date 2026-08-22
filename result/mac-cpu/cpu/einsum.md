# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260822_090304/run.yaml`
- Timestamp: `20260822_090304`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260822_090304`.

- tenferro-rs commit: `a21a4c602fc6700b9bc0c3f1b14ebd19b9d7ec45`

## CPU Information

- Model: `Apple M4`
- Vendor: `Apple`
- Logical CPUs: `10`
- Physical CPUs: `10`
- Sockets: `1`
- Cores per socket: `10`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Performance: 4 physical / 4 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 4 CPUs/L2); Efficiency: 6 physical / 6 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.2-arm64-arm-64bit`

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

- Julia: version `1.12.7`, OMEinsum.jl `0.9.4`, BLAS provider `LBTConfig([ILP64] libopenblas64_.dylib)`, probe threads `4`

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

- Source table: `data/results/mac-cpu/cpu/einsum/20260822_090304/einsum_table_t4_20260822_090304.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260822_090304/tenferro_trace_t4_20260822_090304.log`
- `data/results/mac-cpu/cpu/einsum/20260822_090304/tenferro_eager_t4_20260822_090304.log`
- `data/results/mac-cpu/cpu/einsum/20260822_090304/pytorch_cpu_t4_20260822_090304.log`
- `data/results/mac-cpu/cpu/einsum/20260822_090304/jax_cpu_t4_20260822_090304.log`
- `data/results/mac-cpu/cpu/einsum/20260822_090304/julia_omeinsum_t4_20260822_090304.log`

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.822 ± 5.608 | 0.673 ± 0.060 | **0.408 ± 0.004** | 0.634 ± 0.052 | 4.394 ± 5.479 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.420 ± 0.704 | 0.129 ± 0.027 | **0.104 ± 0.003** | 0.192 ± 0.024 | 0.812 ± 0.193 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.269 ± 1.346 | 0.114 ± 0.039 | **0.103 ± 0.008** | 0.138 ± 0.028 | 0.491 ± 0.131 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.008 ± 4.167 | **0.097 ± 0.030** | 0.101 ± 0.005 | 0.144 ± 0.017 | 1.154 ± 4.829 |
| bin_elementwise_mul_2048x2048 | 4.666 ± 4.983 | 2.768 ± 0.049 | **1.105 ± 0.046** | 1.534 ± 0.033 | 1.272 ± 0.092 |
| bin_matmul_1024 | 23.853 ± 10.119 | 5.860 ± 0.244 | **4.327 ± 0.021** | 7.668 ± 0.330 | 27.441 ± 8.755 |
| bin_matmul_256 | 1.035 ± 2.848 | 0.173 ± 0.029 | **0.086 ± 0.000** | 0.270 ± 0.040 | 0.504 ± 0.134 |
| bin_outer_product_4096 | 4.200 ± 4.472 | **1.752 ± 0.203** | 1.795 ± 0.012 | 1.892 ± 0.024 | 84.954 ± 21.285 |
| gm_queen5_5_3.wcsp | 813.688 ± 158.705 | 1001.175 ± 347.570 | **617.687 ± 28.775** | 1169.508 ± 511.497 | 1974.296 ± 270.637 |
| lm_batch_likelihood_brackets_4_4d | 122.808 ± 80.571 | 21.640 ± 0.672 | **9.095 ± 0.670** | 20.240 ± 5.410 | 18.670 ± 3.482 |
| lm_batch_likelihood_sentence_3_12d | 46.549 ± 23.938 | 39.752 ± 1.238 | **25.102 ± 0.316** | 37.390 ± 4.444 | 95.566 ± 106.756 |
| lm_batch_likelihood_sentence_4_4d | 21.144 ± 2.990 | 22.563 ± 1.169 | **8.439 ± 0.779** | 9.950 ± 0.437 | 111.709 ± 36.274 |
| nary_matmul_chain_64 | 0.086 ± 0.011 | 0.094 ± 0.022 | **0.031 ± 0.001** | 0.111 ± 0.004 | 0.121 ± 0.008 |
| str_matrix_chain_multiplication_100 | 6.409 ± 0.461 | 7.764 ± 0.441 | **3.143 ± 0.056** | 7.916 ± 0.139 | 42.839 ± 34.116 |
| str_mps_varying_inner_product_200 | 16.245 ± 0.963 | 19.286 ± 0.648 | **6.318 ± 0.205** | 37.598 ± 3.179 | 27.913 ± 13.404 |
| str_nw_mera_closed_120 | 187.471 ± 7.711 | 229.109 ± 5.173 | **142.625 ± 0.896** | 352.498 ± 19.854 | 680.215 ± 95.811 |
| str_nw_mera_open_26 | 192.189 ± 25.986 | 235.878 ± 11.497 | **117.441 ± 0.843** | 261.095 ± 20.594 | 557.772 ± 38.414 |
| tensornetwork_permutation_focus_step409_316 | 117.404 ± 1.772 | 156.390 ± 24.165 | **79.592 ± 2.239** | 202.711 ± 15.095 | 165.115 ± 32.260 |
| tensornetwork_permutation_light_415 | 120.251 ± 3.945 | 160.572 ± 5.761 | **83.848 ± 3.572** | 214.503 ± 10.831 | 180.118 ± 34.023 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 2.822 ± 5.608 | 0.673 ± 0.060 | **0.408 ± 0.004** | 0.634 ± 0.052 | 4.394 ± 5.479 |
| bin_batched_matmul_b32_m64_n64_k64 | 0.420 ± 0.704 | 0.129 ± 0.027 | **0.104 ± 0.003** | 0.192 ± 0.024 | 0.812 ± 0.193 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.269 ± 1.346 | 0.114 ± 0.039 | **0.103 ± 0.008** | 0.138 ± 0.028 | 0.491 ± 0.131 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 2.008 ± 4.167 | **0.097 ± 0.030** | 0.101 ± 0.005 | 0.144 ± 0.017 | 1.154 ± 4.829 |
| bin_elementwise_mul_2048x2048 | 4.666 ± 4.983 | 2.768 ± 0.049 | **1.105 ± 0.046** | 1.534 ± 0.033 | 1.272 ± 0.092 |
| bin_matmul_1024 | 23.853 ± 10.119 | 5.860 ± 0.244 | **4.327 ± 0.021** | 7.668 ± 0.330 | 27.441 ± 8.755 |
| bin_matmul_256 | 1.035 ± 2.848 | 0.173 ± 0.029 | **0.086 ± 0.000** | 0.270 ± 0.040 | 0.504 ± 0.134 |
| bin_outer_product_4096 | 4.200 ± 4.472 | **1.752 ± 0.203** | 1.795 ± 0.012 | 1.892 ± 0.024 | 84.954 ± 21.285 |
| gm_queen5_5_3.wcsp | 234.455 ± 21.186 | 370.995 ± 18.441 | **227.823 ± 5.083** | 534.930 ± 54.772 | 658.479 ± 55.845 |
| lm_batch_likelihood_brackets_4_4d | 13.827 ± 0.561 | 19.287 ± 0.765 | **8.767 ± 0.559** | 18.316 ± 2.671 | 20.033 ± 12.396 |
| lm_batch_likelihood_sentence_3_12d | **21.782 ± 1.044** | 34.116 ± 2.333 | 23.897 ± 0.394 | 26.614 ± 2.578 | 59.195 ± 27.274 |
| lm_batch_likelihood_sentence_4_4d | 12.457 ± 0.859 | 18.937 ± 0.834 | **8.701 ± 0.243** | 10.681 ± 0.704 | 23.002 ± 25.544 |
| nary_matmul_chain_64 | 0.086 ± 0.011 | 0.094 ± 0.022 | **0.031 ± 0.001** | 0.111 ± 0.004 | 0.121 ± 0.008 |
| str_matrix_chain_multiplication_100 | 3.814 ± 0.123 | 5.940 ± 0.208 | **3.198 ± 0.111** | 11.537 ± 0.643 | 8.340 ± 0.586 |
| str_mps_varying_inner_product_200 | 9.731 ± 0.633 | 15.463 ± 0.240 | **7.505 ± 0.182** | 18.229 ± 0.284 | 13.448 ± 1.296 |
| str_nw_mera_closed_120 | 193.518 ± 234.606 | 154.550 ± 3.244 | **125.480 ± 0.673** | 348.201 ± 38.158 | 530.509 ± 14.779 |
| str_nw_mera_open_26 | 197.924 ± 9.800 | 208.219 ± 2.597 | **118.250 ± 1.093** | 429.307 ± 36.320 | 448.109 ± 74.862 |
| tensornetwork_permutation_focus_step409_316 | 117.404 ± 1.772 | 156.390 ± 24.165 | **79.592 ± 2.239** | 202.711 ± 15.095 | 165.115 ± 32.260 |
| tensornetwork_permutation_light_415 | 120.251 ± 3.945 | 160.572 ± 5.761 | **83.848 ± 3.572** | 214.503 ± 10.831 | 180.118 ± 34.023 |

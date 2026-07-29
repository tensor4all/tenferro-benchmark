# Einsum Benchmark Results

- Suite: `cpu/einsum`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20260729_195802/run.yaml`
- Timestamp: `20260729_195802`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one suite run under `data/results/mac-cpu/cpu/einsum/20260729_195802`.

- tenferro-rs commit: `57c41c218356e871532e4dea4465a6bcf2d6c8fb`

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
- Python platform: `macOS-26.5.1-arm64-arm-64bit`

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

## Threads: 4

- Source table: `data/results/mac-cpu/cpu/einsum/20260729_195802/einsum_table_t4_20260729_195802.md`

Logs:

- `data/results/mac-cpu/cpu/einsum/20260729_195802/tenferro_trace_t4_20260729_195802.log`
- `data/results/mac-cpu/cpu/einsum/20260729_195802/tenferro_eager_t4_20260729_195802.log`
- `data/results/mac-cpu/cpu/einsum/20260729_195802/pytorch_cpu_t4_20260729_195802.log`
- `data/results/mac-cpu/cpu/einsum/20260729_195802/jax_cpu_t4_20260729_195802.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.783 ± 0.162 | 0.550 ± 0.181 | 0.378 ± 0.016 | **0.361 ± 0.042** |
| bin_batched_matmul_b32_m64_n64_k64 | 0.174 ± 0.014 | 0.109 ± 0.034 | **0.095 ± 0.005** | 0.180 ± 0.039 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.065 ± 0.010** | 0.075 ± 0.021 | 0.066 ± 0.007 | 0.128 ± 0.017 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.169 ± 0.024 | 0.070 ± 0.014 | **0.067 ± 0.012** | 0.131 ± 0.022 |
| bin_elementwise_mul_2048x2048 | 0.642 ± 0.022 | 0.631 ± 0.025 | **0.610 ± 0.088** | 0.809 ± 0.085 |
| bin_matmul_1024 | 6.711 ± 2.170 | 4.580 ± 0.257 | 4.806 ± 1.997 | **3.516 ± 0.508** |
| bin_matmul_256 | 0.135 ± 0.008 | 0.099 ± 0.007 | **0.096 ± 0.020** | 0.232 ± 0.026 |
| bin_outer_product_4096 | 0.597 ± 0.056 | **0.578 ± 0.014** | 0.621 ± 0.042 | 0.653 ± 0.046 |
| gm_queen5_5_3.wcsp | 604.274 ± 40.110 | 475.788 ± 14.135 | 485.998 ± 8.673 | **461.929 ± 15.771** |
| lm_batch_likelihood_brackets_4_4d | 7.681 ± 0.123 | 10.908 ± 0.190 | 7.951 ± 0.783 | **5.830 ± 0.098** |
| lm_batch_likelihood_sentence_3_12d | 9.958 ± 0.314 | 14.244 ± 0.482 | 17.528 ± 1.044 | **7.184 ± 0.307** |
| lm_batch_likelihood_sentence_4_4d | 8.323 ± 0.132 | 11.787 ± 0.128 | 7.936 ± 0.138 | **6.293 ± 0.482** |
| nary_matmul_chain_64 | 0.032 ± 0.006 | 0.037 ± 0.004 | **0.024 ± 0.001** | 0.079 ± 0.017 |
| str_matrix_chain_multiplication_100 | **2.716 ± 0.042** | 3.759 ± 0.082 | 2.758 ± 0.081 | 6.232 ± 0.220 |
| str_mps_varying_inner_product_200 | 6.772 ± 2.650 | 12.481 ± 0.402 | **5.386 ± 0.076** | 14.115 ± 0.258 |
| str_nw_mera_closed_120 | 156.636 ± 8.560 | 148.609 ± 2.919 | 143.748 ± 4.026 | **89.506 ± 1.189** |
| str_nw_mera_open_26 | 146.946 ± 5.437 | 139.655 ± 2.537 | 121.895 ± 2.316 | **70.466 ± 1.803** |
| tensornetwork_permutation_focus_step409_316 | 87.685 ± 3.929 | 89.455 ± 4.516 | 77.679 ± 8.134 | **76.565 ± 5.664** |
| tensornetwork_permutation_light_415 | 91.583 ± 3.641 | 82.989 ± 7.323 | **78.814 ± 0.580** | 83.032 ± 10.076 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m128_n128_k128 | 0.783 ± 0.162 | 0.550 ± 0.181 | 0.378 ± 0.016 | **0.361 ± 0.042** |
| bin_batched_matmul_b32_m64_n64_k64 | 0.174 ± 0.014 | 0.109 ± 0.034 | **0.095 ± 0.005** | 0.180 ± 0.039 |
| bin_batched_outer_product_compact_j16_k16_o64_t64 | **0.065 ± 0.010** | 0.075 ± 0.021 | 0.066 ± 0.007 | 0.128 ± 0.017 |
| bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.169 ± 0.024 | 0.070 ± 0.014 | **0.067 ± 0.012** | 0.131 ± 0.022 |
| bin_elementwise_mul_2048x2048 | 0.642 ± 0.022 | 0.631 ± 0.025 | **0.610 ± 0.088** | 0.809 ± 0.085 |
| bin_matmul_1024 | 6.711 ± 2.170 | 4.580 ± 0.257 | 4.806 ± 1.997 | **3.516 ± 0.508** |
| bin_matmul_256 | 0.135 ± 0.008 | 0.099 ± 0.007 | **0.096 ± 0.020** | 0.232 ± 0.026 |
| bin_outer_product_4096 | 0.597 ± 0.056 | **0.578 ± 0.014** | 0.621 ± 0.042 | 0.653 ± 0.046 |
| gm_queen5_5_3.wcsp | **154.207 ± 6.269** | 190.452 ± 19.633 | 193.958 ± 7.615 | 158.229 ± 7.714 |
| lm_batch_likelihood_brackets_4_4d | 8.328 ± 0.089 | 11.446 ± 0.230 | 11.052 ± 0.700 | **5.846 ± 0.240** |
| lm_batch_likelihood_sentence_3_12d | 10.366 ± 0.226 | 15.133 ± 0.978 | 18.998 ± 1.328 | **7.251 ± 0.249** |
| lm_batch_likelihood_sentence_4_4d | 8.520 ± 0.106 | 12.018 ± 1.187 | 10.824 ± 0.531 | **5.422 ± 0.365** |
| nary_matmul_chain_64 | 0.032 ± 0.006 | 0.037 ± 0.004 | **0.024 ± 0.001** | 0.079 ± 0.017 |
| str_matrix_chain_multiplication_100 | 2.901 ± 0.243 | 4.007 ± 0.336 | **2.780 ± 0.131** | 6.300 ± 0.162 |
| str_mps_varying_inner_product_200 | **7.591 ± 0.161** | 12.805 ± 0.293 | 11.704 ± 1.013 | 15.529 ± 0.387 |
| str_nw_mera_closed_120 | 137.657 ± 1.564 | 133.219 ± 1.380 | 130.531 ± 2.883 | **86.709 ± 2.207** |
| str_nw_mera_open_26 | 142.044 ± 4.872 | 140.807 ± 3.645 | 124.481 ± 2.352 | **76.569 ± 1.850** |
| tensornetwork_permutation_focus_step409_316 | 87.685 ± 3.929 | 89.455 ± 4.516 | 77.679 ± 8.134 | **76.565 ± 5.664** |
| tensornetwork_permutation_light_415 | 91.583 ± 3.641 | 82.989 ± 7.323 | **78.814 ± 0.580** | 83.032 ± 10.076 |

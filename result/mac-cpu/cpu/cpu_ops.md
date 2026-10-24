# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`
- Benchmark commit: `50a7dc9b3ea08f5dba14db07df1e09314cac56f0`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261024_172655/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20261024_172655`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261024_172655`.

- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261024_172655/cpu_ops_t1_20261024_172655.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20261024_172655/cpu_ops_t1_20261024_172655.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.013 ± 0.007 | 0.011 ± 0.001 | 0.009 ± 0.004 | 0.018 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.030 ± 0.009 | 0.025 ± 0.007 | 0.023 ± 0.003 | 0.042 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.023 ± 0.001 | 0.051 ± 0.018 | 0.028 ± 0.002 | 0.049 ± 0.020 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.205 ± 0.046 | 0.096 ± 0.006 | 0.092 ± 0.015 | 0.122 ± 0.023 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.009 ± 0.001 | 0.125 ± 0.153 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.006 ± 0.001 | 0.006 ± 0.001 | 0.010 ± 0.001 | 0.102 ± 0.014 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.055 ± 0.007 | 0.056 ± 0.033 | 0.069 ± 0.049 | 0.114 ± 0.092 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.009 ± 0.002 | 0.200 ± 0.142 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.006 ± 0.003 | 0.007 ± 0.001 | 0.011 ± 0.001 | 0.096 ± 0.022 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.001 | 0.008 ± 0.001 | 0.007 ± 0.002 | 0.014 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.030 ± 0.006 | 0.024 ± 0.011 | 0.021 ± 0.011 | 0.032 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.011 ± 0.001 | 0.013 ± 0.001 | 0.013 ± 0.002 | 0.026 ± 0.007 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.046 ± 0.035 | 0.045 ± 0.004 | 0.053 ± 0.029 | 0.082 ± 0.043 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.008 ± 0.006 | 0.012 ± 0.001 | 0.011 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.013 ± 0.004 | 0.013 ± 0.003 | 0.018 ± 0.003 | 0.013 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.004 | 0.011 ± 0.006 | 0.014 ± 0.004 | 0.011 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.037 ± 0.017 | 0.018 ± 0.000 | 0.023 ± 0.002 | 0.023 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.013 ± 0.001 | 0.015 ± 0.002 | 0.018 ± 0.004 | 0.051 ± 0.038 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.045 ± 0.071 | 0.069 ± 0.019 | 0.057 ± 0.027 | 0.148 ± 0.017 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.045 ± 0.003 | 0.054 ± 0.010 | 0.055 ± 0.012 | 0.096 ± 0.030 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.249 ± 0.053 | 0.209 ± 0.017 | 0.196 ± 0.039 | 0.297 ± 0.047 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.008 | 0.053 ± 0.008 | 0.011 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.001 | 0.071 ± 0.010 | 0.007 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.001 | 0.083 ± 0.019 | 0.012 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.077 ± 0.005 | 0.042 ± 0.036 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.003 | 0.064 ± 0.022 | 0.022 ± 0.020 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.033 ± 0.004 | 0.067 ± 0.007 | 0.018 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.028 ± 0.005 | 0.059 ± 0.005 | 0.010 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.045 ± 0.012 | 0.072 ± 0.018 | 0.031 ± 0.025 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.362 ± 0.058 | 0.301 ± 0.011 | 0.367 ± 0.089 | 0.751 ± 0.425 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 4.807 ± 0.314 | 5.343 ± 0.725 | 7.517 ± 2.509 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 33.565 ± 12.031 | 36.606 ± 11.451 | 51.909 ± 18.163 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 4.927 ± 0.268 | 4.946 ± 0.117 | 7.005 ± 0.255 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 33.334 ± 5.175 | 35.708 ± 4.609 | 60.939 ± 18.449 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 0.995 ± 0.033 | 1.914 ± 0.038 | 3.802 ± 0.922 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 8.200 ± 1.946 | 24.376 ± 20.491 | 35.051 ± 10.814 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 1.075 ± 0.044 | 1.289 ± 0.300 | 3.796 ± 1.515 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 8.175 ± 1.337 | 10.589 ± 2.059 | 31.258 ± 15.452 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.016 ± 0.002 | 0.008 ± 0.000 | 0.011 ± 0.007 | 0.054 ± 0.040 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.002 | 0.053 ± 0.028 | 0.070 ± 0.006 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 2.429 ± 0.320 | 2.926 ± 0.663 | 5.327 ± 0.727 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 13.877 ± 0.770 | 17.945 ± 1.000 | 40.722 ± 12.038 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 2.212 ± 0.047 | 2.766 ± 0.528 | 6.897 ± 2.459 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 20.412 ± 2.561 | 13.315 ± 0.260 | 42.093 ± 17.108 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.004 | 0.083 ± 0.022 | 0.057 ± 0.039 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.266 ± 0.012 | 0.416 ± 0.055 | 0.396 ± 0.396 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 1.655 ± 0.189 | 3.936 ± 1.767 | 1.952 ± 1.375 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.310 ± 0.032 | 0.526 ± 0.097 | 0.318 ± 0.026 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 2.022 ± 0.283 | 3.960 ± 1.148 | 2.117 ± 1.724 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.474 ± 0.014 | 0.477 ± 0.078 | 0.523 ± 0.023 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 8.946 ± 0.492 | 10.149 ± 1.658 | 19.100 ± 9.923 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 72.511 ± 9.293 | 68.084 ± 14.200 | 79.313 ± 5.835 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 8.669 ± 0.098 | 9.303 ± 0.279 | 14.267 ± 3.689 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 65.672 ± 4.578 | 58.899 ± 1.562 | 78.765 ± 24.212 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.079 ± 0.037 | 0.020 ± 0.001 | 0.028 ± 0.007 | 0.252 ± 0.125 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.192 ± 0.140 | 0.128 ± 0.008 | 0.286 ± 0.099 | 1.581 ± 1.157 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.519 ± 0.017 | 0.519 ± 0.030 | 0.700 ± 0.555 | 3.768 ± 1.199 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.071 ± 0.009 | 0.065 ± 0.001 | 0.082 ± 0.019 | 0.116 ± 0.066 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.022 ± 0.002 | 0.021 ± 0.002 | 0.040 ± 0.017 | 0.039 ± 0.014 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.023 ± 0.003 | 0.025 ± 0.001 | 0.046 ± 0.002 | 0.066 ± 0.040 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.029 ± 0.002 | 0.032 ± 0.001 | 0.107 ± 0.021 | 0.296 ± 0.107 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.675 ± 0.125 | 0.391 ± 0.009 | 0.506 ± 0.153 | 0.617 ± 0.230 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.004 ± 0.001 | 0.004 ± 0.001 | 0.006 ± 0.005 | 0.011 ± 0.005 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.006 ± 0.003 | 0.005 ± 0.000 | 0.007 ± 0.002 | 0.017 ± 0.014 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.014 ± 0.014 | 0.009 ± 0.000 | 0.009 ± 0.001 | 0.023 ± 0.032 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.009 ± 0.004 | 0.013 ± 0.003 | 0.087 ± 0.023 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.003 ± 0.001 | 0.006 ± 0.001 | 0.012 ± 0.003 | 0.102 ± 0.026 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.013 ± 0.006 | 0.177 ± 0.075 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | - | 0.016 ± 0.001 | 0.069 ± 0.003 | 0.013 ± 0.008 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | - | 0.022 ± 0.011 | 0.066 ± 0.004 | 0.025 ± 0.020 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | - | 0.023 ± 0.001 | 0.093 ± 0.035 | 0.018 ± 0.010 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | - | 0.017 ± 0.006 | 0.060 ± 0.006 | 0.016 ± 0.005 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | - | 0.018 ± 0.006 | 0.062 ± 0.018 | 0.015 ± 0.013 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | - | 0.025 ± 0.002 | 0.176 ± 0.108 | 0.022 ± 0.005 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | - | 0.020 ± 0.010 | 0.107 ± 0.014 | 0.011 ± 0.013 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | - | 0.018 ± 0.001 | 0.096 ± 0.003 | 0.016 ± 0.018 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | - | 0.018 ± 0.004 | 0.113 ± 0.006 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | - | 0.016 ± 0.003 | 0.090 ± 0.014 | 0.024 ± 0.019 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | - | 0.018 ± 0.002 | 0.078 ± 0.004 | 0.025 ± 0.016 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | - | 0.019 ± 0.001 | 0.096 ± 0.017 | 0.018 ± 0.003 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.008 ± 0.003 | 0.026 ± 0.002 | 0.009 ± 0.006 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.008 ± 0.000 | 0.031 ± 0.015 | 0.012 ± 0.010 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.032 ± 0.004 | 0.011 ± 0.009 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | - | 0.025 ± 0.001 | 0.077 ± 0.009 | 0.012 ± 0.014 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | - | 0.027 ± 0.004 | 0.079 ± 0.007 | 0.031 ± 0.004 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | - | 0.028 ± 0.001 | 0.095 ± 0.024 | 0.012 ± 0.009 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | - | 0.028 ± 0.012 | 0.080 ± 0.015 | 0.015 ± 0.009 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | - | 0.027 ± 0.001 | 0.074 ± 0.003 | 0.013 ± 0.018 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | - | 0.031 ± 0.003 | 0.102 ± 0.016 | 0.017 ± 0.010 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.001 | 0.038 ± 0.005 | 0.017 ± 0.011 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.001 | 0.041 ± 0.011 | 0.016 ± 0.012 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.004 | 0.042 ± 0.004 | 0.023 ± 0.021 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.014 ± 0.006 | 0.178 ± 0.067 | 0.012 ± 0.007 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.012 ± 0.000 | 0.153 ± 0.014 | 0.015 ± 0.013 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.012 ± 0.002 | 0.176 ± 0.022 | 0.009 ± 0.009 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.013 ± 0.005 | 0.083 ± 0.036 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.015 ± 0.005 | 0.084 ± 0.007 | 0.017 ± 0.012 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.014 ± 0.003 | 0.089 ± 0.018 | 0.012 ± 0.002 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.001 | 0.029 ± 0.001 | 0.012 ± 0.007 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.002 | 0.031 ± 0.007 | 0.018 ± 0.009 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.003 | 0.040 ± 0.004 | 0.034 ± 0.015 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | - | 0.011 ± 0.007 | 0.099 ± 0.014 | 0.013 ± 0.011 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | - | 0.013 ± 0.001 | 0.094 ± 0.011 | 0.038 ± 0.027 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | - | 0.026 ± 0.001 | 0.140 ± 0.023 | 0.034 ± 0.018 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | - | 0.011 ± 0.001 | 0.065 ± 0.007 | 0.018 ± 0.006 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | - | 0.014 ± 0.001 | 0.064 ± 0.008 | 0.019 ± 0.019 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | - | 0.021 ± 0.001 | 0.082 ± 0.018 | 0.036 ± 0.018 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.004 ± 0.004 | 0.001 ± 0.001 | 0.008 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.004 ± 0.000 | 0.005 ± 0.004 | 0.002 ± 0.001 | 0.013 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.003 ± 0.002 | 0.003 ± 0.001 | 0.001 ± 0.000 | 0.010 ± 0.004 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.007 ± 0.001 | 0.007 ± 0.001 | 0.008 ± 0.002 | 0.027 ± 0.017 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.001 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.018 ± 0.009 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.004 ± 0.001 | 0.005 ± 0.002 | 0.004 ± 0.001 | 0.008 ± 0.007 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.004 ± 0.004 | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.015 ± 0.007 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.006 ± 0.001 | 0.005 ± 0.001 | 0.006 ± 0.002 | 0.015 ± 0.019 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.004 ± 0.001 | 0.005 ± 0.002 | 0.015 ± 0.005 | 0.007 ± 0.011 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.011 ± 0.007 | 0.012 ± 0.010 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.005 ± 0.004 | 0.005 ± 0.000 | 0.010 ± 0.002 | 0.009 ± 0.008 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.004 ± 0.002 | 0.005 ± 0.000 | 0.010 ± 0.002 | 0.012 ± 0.010 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.009 ± 0.002 | 0.005 ± 0.000 | 0.012 ± 0.002 | 0.023 ± 0.019 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.006 ± 0.002 | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.008 ± 0.015 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.005 | 0.019 ± 0.006 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.008 ± 0.002 | 0.007 ± 0.000 | 0.010 ± 0.002 | 0.018 ± 0.017 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.019 ± 0.002 | 0.016 ± 0.002 | 0.027 ± 0.035 | 0.037 ± 0.030 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 47.2x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 30.9x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 18.6x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 21.1x faster than the slowest successful cell (`jax-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 71.1x faster than the slowest successful cell (`jax-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 48.1x faster than the slowest successful cell (`jax-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 15.3x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=1, shape=`128x128`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.252 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=1, shape=`256x256`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`jax-cpu`, 1.581 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=1, shape=`64x64,rhs=64`): `tenferro-eager` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.296 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 38.2x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 38.8x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 63.7x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 30.8x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 22.4x faster than the slowest successful cell (`pytorch-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261024_175546/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20261024_175546`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261024_175546`.

- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261024_175546/cpu_ops_t4_20261024_175546.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20261024_175546/cpu_ops_t4_20261024_175546.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.003 | 0.013 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.017 ± 0.001 | 0.018 ± 0.000 | 0.017 ± 0.000 | 0.049 ± 0.056 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.022 ± 0.001 | 0.023 ± 0.000 | 0.021 ± 0.000 | 0.027 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.076 ± 0.003 | 0.078 ± 0.006 | 0.074 ± 0.001 | 0.105 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.062 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.079 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.046 ± 0.002 | 0.047 ± 0.001 | 0.059 ± 0.002 | 0.170 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.101 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.004 | 0.081 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.000 | 0.010 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.015 ± 0.001 | 0.015 ± 0.000 | 0.033 ± 0.000 | 0.023 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.027 ± 0.001 | 0.015 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.032 ± 0.002 | 0.034 ± 0.001 | 0.052 ± 0.002 | 0.048 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.028 ± 0.006 | 0.007 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.044 ± 0.002 | 0.012 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.028 ± 0.006 | 0.009 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.053 ± 0.009 | 0.018 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.035 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.037 ± 0.001 | 0.039 ± 0.001 | 0.037 ± 0.000 | 0.116 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.042 ± 0.001 | 0.043 ± 0.001 | 0.040 ± 0.000 | 0.084 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.148 ± 0.004 | 0.154 ± 0.004 | 0.169 ± 0.066 | 0.233 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.000 | 0.058 ± 0.008 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.001 | 0.050 ± 0.001 | 0.006 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.057 ± 0.012 | 0.011 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.003 | 0.060 ± 0.012 | 0.043 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.001 | 0.064 ± 0.028 | 0.014 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.001 | 0.089 ± 0.014 | 0.016 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.000 | 0.084 ± 0.006 | 0.009 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.034 ± 0.000 | 0.099 ± 0.006 | 0.033 ± 0.004 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.282 ± 0.000 | 0.327 ± 0.059 | 0.329 ± 0.029 | 0.290 ± 0.011 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 4.732 ± 0.240 | 4.633 ± 0.161 | 4.860 ± 0.054 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 23.539 ± 0.155 | 25.609 ± 0.328 | 26.903 ± 0.153 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 4.707 ± 0.306 | 4.269 ± 0.059 | 5.664 ± 0.778 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 25.019 ± 0.069 | 23.269 ± 0.237 | 27.006 ± 0.185 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 1.046 ± 0.110 | 2.022 ± 0.066 | 1.727 ± 0.064 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 6.258 ± 0.115 | 12.306 ± 1.450 | 8.589 ± 0.533 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 1.097 ± 0.092 | 1.498 ± 0.134 | 1.702 ± 0.101 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 6.764 ± 0.211 | 7.604 ± 0.625 | 13.488 ± 3.408 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.015 ± 0.002 | 0.009 ± 0.001 | 0.011 ± 0.001 | 0.030 ± 0.000 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.003 | 0.045 ± 0.002 | 0.066 ± 0.001 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 2.184 ± 0.189 | 2.756 ± 0.042 | 2.647 ± 0.055 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 11.290 ± 0.028 | 11.736 ± 0.293 | 13.707 ± 0.131 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 2.309 ± 0.164 | 2.357 ± 0.350 | 2.575 ± 0.038 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 10.559 ± 0.206 | 11.123 ± 0.105 | 14.098 ± 0.436 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.041 ± 0.001 | 0.067 ± 0.004 | 0.046 ± 0.011 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.298 ± 0.007 | 0.434 ± 0.033 | 0.309 ± 0.015 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.340 ± 0.033 | 1.362 ± 0.045 | 1.315 ± 0.031 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.296 ± 0.008 | 0.628 ± 0.057 | 0.332 ± 0.032 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.320 ± 0.006 | 2.262 ± 0.012 | 1.219 ± 0.019 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.459 ± 0.088 | 0.432 ± 0.021 | 0.436 ± 0.004 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 8.447 ± 0.176 | 10.154 ± 0.327 | 8.574 ± 0.058 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 46.901 ± 11.610 | 47.561 ± 0.306 | 64.596 ± 7.163 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 8.804 ± 0.101 | 8.971 ± 0.068 | 8.538 ± 0.059 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 46.731 ± 0.477 | 43.461 ± 0.251 | 46.734 ± 0.230 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.019 ± 0.000 | 0.019 ± 0.002 | 0.025 ± 0.003 | 0.095 ± 0.006 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.113 ± 0.002 | 0.121 ± 0.019 | 0.168 ± 0.018 | 0.512 ± 0.165 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.418 ± 0.003 | 0.454 ± 0.039 | 0.608 ± 0.101 | 1.390 ± 0.095 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.062 ± 0.001 | 0.068 ± 0.002 | 0.107 ± 0.004 | 0.072 ± 0.004 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.018 ± 0.000 | 0.022 ± 0.004 | 0.034 ± 0.003 | 0.042 ± 0.015 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.020 ± 0.001 | 0.025 ± 0.001 | 0.040 ± 0.004 | 0.043 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.025 ± 0.000 | 0.031 ± 0.000 | 0.045 ± 0.004 | 0.097 ± 0.009 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.376 ± 0.007 | 0.466 ± 0.091 | 0.448 ± 0.061 | 0.409 ± 0.013 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.008 ± 0.001 | 0.008 ± 0.000 | 0.008 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.079 ± 0.016 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.063 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.080 ± 0.005 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | - | 0.015 ± 0.001 | 0.066 ± 0.005 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | - | 0.017 ± 0.000 | 0.067 ± 0.007 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | - | 0.030 ± 0.005 | 0.076 ± 0.005 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | - | 0.015 ± 0.000 | 0.078 ± 0.008 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | - | 0.016 ± 0.000 | 0.058 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | - | 0.022 ± 0.006 | 0.068 ± 0.003 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | - | 0.016 ± 0.003 | 0.433 ± 0.081 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | - | 0.016 ± 0.001 | 0.277 ± 0.026 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | - | 0.022 ± 0.003 | 0.252 ± 0.019 | 0.005 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | - | 0.014 ± 0.000 | 0.439 ± 0.157 | 0.007 ± 0.002 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | - | 0.015 ± 0.000 | 0.257 ± 0.046 | 0.008 ± 0.004 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | - | 0.023 ± 0.004 | 0.189 ± 0.011 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.008 ± 0.000 | 0.024 ± 0.002 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.000 | 0.024 ± 0.001 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.000 | 0.025 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | - | 0.028 ± 0.001 | 0.131 ± 0.012 | 0.007 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | - | 0.025 ± 0.000 | 0.180 ± 0.034 | 0.009 ± 0.005 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | - | 0.033 ± 0.001 | 0.131 ± 0.048 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | - | 0.024 ± 0.000 | 0.155 ± 0.171 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | - | 0.025 ± 0.001 | 0.162 ± 0.072 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | - | 0.036 ± 0.009 | 0.162 ± 0.033 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.037 ± 0.004 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.001 | 0.052 ± 0.022 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.002 | 0.038 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.010 ± 0.000 | 0.205 ± 0.030 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.011 ± 0.000 | 0.176 ± 0.033 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.017 ± 0.005 | 0.158 ± 0.014 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.011 ± 0.001 | 0.080 ± 0.014 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.011 ± 0.000 | 0.082 ± 0.008 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.015 ± 0.002 | 0.074 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.011 ± 0.001 | 0.030 ± 0.004 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.040 ± 0.006 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.008 | 0.038 ± 0.002 | 0.017 ± 0.003 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | - | 0.011 ± 0.000 | 0.091 ± 0.013 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | - | 0.012 ± 0.000 | 0.092 ± 0.012 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | - | 0.026 ± 0.004 | 0.103 ± 0.002 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | - | 0.011 ± 0.001 | 0.060 ± 0.004 | 0.012 ± 0.005 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | - | 0.012 ± 0.001 | 0.062 ± 0.003 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | - | 0.023 ± 0.003 | 0.069 ± 0.001 | 0.017 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.004 ± 0.002 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.008 ± 0.003 | 0.023 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.019 | 0.006 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.030 ± 0.005 | 0.006 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.029 ± 0.012 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.008 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.004 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.013 ± 0.002 | 0.013 ± 0.002 | 0.018 ± 0.006 | 0.018 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 25.9x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 18.2x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 21.6x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 16.3x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 36.8x faster than the slowest successful cell (`jax-cpu`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 27.7x faster than the slowest successful cell (`jax-cpu`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 39.5x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 15.3x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 31.0x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 37.4x faster than the slowest successful cell (`jax-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 58.7x faster than the slowest successful cell (`pytorch-cpu`, 0.433 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 27.0x faster than the slowest successful cell (`pytorch-cpu`, 0.433 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 42.1x faster than the slowest successful cell (`pytorch-cpu`, 0.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 17.2x faster than the slowest successful cell (`pytorch-cpu`, 0.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 51.8x faster than the slowest successful cell (`pytorch-cpu`, 0.252 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.252 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 64.5x faster than the slowest successful cell (`pytorch-cpu`, 0.439 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 30.8x faster than the slowest successful cell (`pytorch-cpu`, 0.439 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 33.5x faster than the slowest successful cell (`pytorch-cpu`, 0.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 39.3x faster than the slowest successful cell (`pytorch-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 19.7x faster than the slowest successful cell (`pytorch-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 25.2x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 25.9x faster than the slowest successful cell (`pytorch-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 18.1x faster than the slowest successful cell (`pytorch-cpu`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 32.0x faster than the slowest successful cell (`pytorch-cpu`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 26.7x faster than the slowest successful cell (`pytorch-cpu`, 0.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 0.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 30.3x faster than the slowest successful cell (`pytorch-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 26.1x faster than the slowest successful cell (`pytorch-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

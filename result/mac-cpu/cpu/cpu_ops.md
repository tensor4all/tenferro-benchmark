# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `ec30d812749ef2bd6dc72245a56150aa77cb2f9c`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261108_172311/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261108_172311/run.yaml`
- Timestamp: `2026-11-08T17:23:11.781207+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261108_172311`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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

- tenferro-rs features: `blas-accelerate`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20261108_172311/cpu_ops_t1_20261108_172311.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.006 ± 0.001 | 0.008 ± 0.001 | 0.019 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.020 ± 0.000 | 0.018 ± 0.000 | 0.023 ± 0.001 | 0.044 ± 0.014 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.024 ± 0.000 | 0.023 ± 0.000 | 0.030 ± 0.002 | 0.037 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.083 ± 0.004 | 0.085 ± 0.004 | 0.090 ± 0.003 | 0.120 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.001 | 0.089 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.105 ± 0.021 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.049 ± 0.002 | 0.048 ± 0.001 | 0.044 ± 0.008 | 0.120 ± 0.021 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.017 ± 0.008 | 0.111 ± 0.019 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.092 ± 0.031 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.017 ± 0.001 | 0.015 ± 0.000 | 0.021 ± 0.003 | 0.048 ± 0.013 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.002 | 0.020 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.037 ± 0.001 | 0.035 ± 0.001 | 0.041 ± 0.001 | 0.049 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.012 | 0.010 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.001 | 0.010 ± 0.000 | 0.019 ± 0.005 | 0.016 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.012 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.015 ± 0.004 | 0.014 ± 0.000 | 0.024 ± 0.003 | 0.024 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.014 ± 0.001 | 0.013 ± 0.001 | 0.017 ± 0.003 | 0.052 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.039 ± 0.002 | 0.040 ± 0.016 | 0.048 ± 0.005 | 0.160 ± 0.018 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.047 ± 0.002 | 0.044 ± 0.000 | 0.069 ± 0.024 | 0.083 ± 0.009 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.169 ± 0.008 | 0.173 ± 0.006 | 0.173 ± 0.005 | 0.247 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.001 | 0.131 ± 0.150 | 0.009 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.082 ± 0.026 | 0.010 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.000 | 0.057 ± 0.012 | 0.012 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.001 | 0.075 ± 0.006 | 0.032 ± 0.012 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.038 ± 0.009 | 0.107 ± 0.204 | 0.017 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.046 ± 0.002 | 0.069 ± 0.018 | 0.022 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.031 ± 0.002 | 0.048 ± 0.005 | 0.013 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.062 ± 0.004 | 0.069 ± 0.007 | 0.027 ± 0.004 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.305 ± 0.005 | 0.370 ± 0.126 | 0.361 ± 0.031 | 0.377 ± 0.043 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 11.575 ± 0.436 | 8.943 ± 0.371 | 6.999 ± 1.464 | 6.571 ± 0.024 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 65.957 ± 19.614 | 55.351 ± 1.703 | 32.695 ± 0.303 | 47.452 ± 2.986 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 5.962 ± 0.440 | 11.409 ± 2.858 | 6.491 ± 1.190 | 8.550 ± 0.850 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 34.950 ± 3.771 | 55.443 ± 2.142 | 29.925 ± 1.378 | 43.838 ± 2.475 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 3.180 ± 0.418 | 1.575 ± 0.671 | 3.316 ± 2.359 | 4.726 ± 2.363 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 22.594 ± 5.839 | 9.695 ± 0.055 | 18.853 ± 0.795 | 19.753 ± 0.262 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 2.794 ± 0.141 | 1.547 ± 0.239 | 1.749 ± 0.234 | 4.523 ± 1.926 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 18.996 ± 0.452 | 10.255 ± 0.385 | 10.300 ± 0.662 | 19.113 ± 0.063 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.016 ± 0.002 | 0.013 ± 0.014 | 0.010 ± 0.001 | 0.038 ± 0.005 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.010 | 0.049 ± 0.007 | 0.078 ± 0.012 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 6.498 ± 0.501 | 3.832 ± 0.052 | 3.703 ± 1.227 | 4.692 ± 0.541 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 36.225 ± 1.404 | 25.668 ± 2.326 | 17.715 ± 0.505 | 35.425 ± 2.203 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 4.212 ± 0.091 | 4.044 ± 1.130 | 3.785 ± 1.058 | 4.821 ± 0.964 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 26.429 ± 2.006 | 24.941 ± 0.888 | 17.209 ± 0.401 | 40.581 ± 7.553 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.123 ± 0.026 | 0.103 ± 0.028 | 0.047 ± 0.012 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 0.723 ± 0.106 | 0.582 ± 0.083 | 0.593 ± 0.074 | 0.672 ± 0.485 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 4.131 ± 1.353 | 3.205 ± 0.129 | 2.446 ± 0.086 | 1.902 ± 0.032 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.457 ± 0.021 | 0.582 ± 0.416 | 0.715 ± 0.058 | 0.662 ± 1.146 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 1.915 ± 0.070 | 3.185 ± 0.235 | 3.589 ± 0.138 | 1.617 ± 0.034 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.605 ± 0.073 | 0.565 ± 0.043 | 0.544 ± 0.030 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 19.311 ± 1.006 | 16.085 ± 4.684 | 14.139 ± 2.198 | 10.990 ± 0.361 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 121.868 ± 5.263 | 107.147 ± 12.823 | 66.707 ± 13.478 | 92.956 ± 13.532 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 9.809 ± 0.411 | 16.361 ± 2.812 | 11.297 ± 3.242 | 11.113 ± 0.201 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 62.236 ± 1.228 | 99.958 ± 4.873 | 56.795 ± 1.392 | 75.259 ± 2.987 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.025 ± 0.002 | 0.020 ± 0.001 | 0.030 ± 0.003 | 0.147 ± 0.029 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.161 ± 0.007 | 0.130 ± 0.011 | 0.186 ± 0.022 | 0.861 ± 0.048 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.514 ± 0.013 | 0.563 ± 0.309 | 0.721 ± 0.057 | 4.162 ± 2.263 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.066 ± 0.002 | 0.070 ± 0.022 | 0.081 ± 0.009 | 0.091 ± 0.039 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.020 ± 0.000 | 0.022 ± 0.021 | 0.033 ± 0.004 | 0.044 ± 0.011 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.022 ± 0.000 | 0.026 ± 0.014 | 0.047 ± 0.004 | 0.062 ± 0.004 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.027 ± 0.000 | 0.032 ± 0.001 | 0.096 ± 0.073 | 0.222 ± 0.014 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.434 ± 0.094 | 0.535 ± 0.142 | 0.484 ± 0.056 | 0.691 ± 0.449 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.005 ± 0.002 | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.005 ± 0.001 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.009 ± 0.000 | 0.008 ± 0.001 | 0.009 ± 0.000 | 0.016 ± 0.006 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.001 | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.084 ± 0.005 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.006 ± 0.005 | 0.008 ± 0.001 | 0.084 ± 0.015 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.003 ± 0.001 | 0.006 ± 0.001 | 0.008 ± 0.001 | 0.194 ± 0.062 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.123 ± 0.028 | 0.017 ± 0.000 | 0.095 ± 0.032 | 0.012 ± 0.007 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.149 ± 0.088 | 0.022 ± 0.005 | 0.087 ± 0.012 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.138 ± 0.017 | 0.030 ± 0.000 | 0.095 ± 0.014 | 0.012 ± 0.004 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.038 ± 0.004 | 0.017 ± 0.000 | 0.093 ± 0.028 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.051 ± 0.013 | 0.021 ± 0.002 | 0.075 ± 0.007 | 0.011 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.048 ± 0.004 | 0.026 ± 0.001 | 0.080 ± 0.014 | 0.012 ± 0.002 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.488 ± 0.095 | 0.020 ± 0.002 | 0.134 ± 0.035 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.503 ± 0.161 | 0.024 ± 0.010 | 0.117 ± 0.014 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.414 ± 0.035 | 0.021 ± 0.005 | 0.125 ± 0.012 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.084 ± 0.008 | 0.026 ± 0.004 | 0.086 ± 0.030 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.157 ± 0.050 | 0.020 ± 0.009 | 0.099 ± 0.011 | 0.012 ± 0.010 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.094 ± 0.003 | 0.030 ± 0.009 | 0.097 ± 0.004 | 0.008 ± 0.003 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.008 ± 0.000 | 0.025 ± 0.002 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.000 | 0.030 ± 0.004 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.001 | 0.030 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.635 ± 0.103 | 0.029 ± 0.001 | 0.084 ± 0.033 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.757 ± 0.250 | 0.031 ± 0.001 | 0.095 ± 0.004 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.695 ± 0.105 | 0.035 ± 0.003 | 0.092 ± 0.002 | 0.011 ± 0.008 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.081 ± 0.012 | 0.031 ± 0.005 | 0.082 ± 0.037 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.112 ± 0.064 | 0.030 ± 0.007 | 0.096 ± 0.005 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.091 ± 0.005 | 0.038 ± 0.000 | 0.106 ± 0.018 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.003 | 0.044 ± 0.003 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.001 | 0.051 ± 0.011 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.002 | 0.044 ± 0.011 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.096 ± 0.039 | 0.014 ± 0.001 | 0.192 ± 0.079 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.089 ± 0.006 | 0.016 ± 0.001 | 0.210 ± 0.047 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.076 ± 0.010 | 0.017 ± 0.008 | 0.196 ± 0.033 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.045 ± 0.016 | 0.015 ± 0.000 | 0.108 ± 0.037 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.045 ± 0.007 | 0.016 ± 0.002 | 0.098 ± 0.010 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.039 ± 0.001 | 0.016 ± 0.013 | 0.099 ± 0.022 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.032 ± 0.002 | 0.014 ± 0.010 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.001 | 0.039 ± 0.002 | 0.015 ± 0.006 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.001 | 0.041 ± 0.001 | 0.022 ± 0.003 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.141 ± 0.064 | 0.011 ± 0.001 | 0.107 ± 0.014 | 0.012 ± 0.004 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.151 ± 0.167 | 0.017 ± 0.001 | 0.163 ± 0.275 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.137 ± 0.009 | 0.031 ± 0.001 | 0.132 ± 0.023 | 0.019 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.036 ± 0.001 | 0.012 ± 0.000 | 0.077 ± 0.021 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.044 ± 0.048 | 0.016 ± 0.001 | 0.076 ± 0.030 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.052 ± 0.005 | 0.029 ± 0.005 | 0.090 ± 0.007 | 0.020 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.001 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.004 ± 0.001 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.008 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.009 ± 0.001 | 0.006 ± 0.001 | 0.008 ± 0.001 | 0.024 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.007 ± 0.002 | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.004 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.005 ± 0.001 | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.018 ± 0.010 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.004 ± 0.002 | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.007 ± 0.002 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.007 ± 0.001 | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.009 ± 0.006 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.010 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.002 | 0.009 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.016 ± 0.009 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.002 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.003 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.017 ± 0.006 | 0.014 ± 0.001 | 0.019 ± 0.002 | 0.033 ± 0.011 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 34.1x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 30.0x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 24.6x faster than the slowest successful cell (`jax-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 24.8x faster than the slowest successful cell (`jax-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 36.1x faster than the slowest successful cell (`jax-cpu`, 0.111 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 37.3x faster than the slowest successful cell (`jax-cpu`, 0.111 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 16.5x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 28.8x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 33.6x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 23.2x faster than the slowest successful cell (`jax-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 67.9x faster than the slowest successful cell (`jax-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 34.9x faster than the slowest successful cell (`jax-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-eager`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.7x faster than the slowest successful cell (`tenferro-eager`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 11.8x faster than the slowest successful cell (`tenferro-eager`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 52.2x faster than the slowest successful cell (`tenferro-eager`, 0.488 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 24.8x faster than the slowest successful cell (`tenferro-eager`, 0.488 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 55.3x faster than the slowest successful cell (`tenferro-eager`, 0.503 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 20.9x faster than the slowest successful cell (`tenferro-eager`, 0.503 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 41.7x faster than the slowest successful cell (`tenferro-eager`, 0.414 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 19.3x faster than the slowest successful cell (`tenferro-eager`, 0.414 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.2x faster than the slowest successful cell (`tenferro-eager`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 65.2x faster than the slowest successful cell (`tenferro-eager`, 0.635 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 21.7x faster than the slowest successful cell (`tenferro-eager`, 0.635 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 80.8x faster than the slowest successful cell (`tenferro-eager`, 0.757 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 24.1x faster than the slowest successful cell (`tenferro-eager`, 0.757 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 65.2x faster than the slowest successful cell (`tenferro-eager`, 0.695 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 20.1x faster than the slowest successful cell (`tenferro-eager`, 0.695 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.5x faster than the slowest successful cell (`tenferro-eager`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 22.1x faster than the slowest successful cell (`pytorch-cpu`, 0.192 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.192 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 23.3x faster than the slowest successful cell (`pytorch-cpu`, 0.210 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 0.210 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 20.6x faster than the slowest successful cell (`pytorch-cpu`, 0.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.4x faster than the slowest successful cell (`tenferro-eager`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`tenferro-eager`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261108_175931/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261108_175931/run.yaml`
- Timestamp: `2026-11-08T17:59:31.505952+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261108_175931`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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

- tenferro-rs features: `blas-accelerate`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20261108_175931/cpu_ops_t4_20261108_175931.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.001 | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.021 ± 0.002 | 0.020 ± 0.000 | 0.047 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.024 ± 0.000 | 0.023 ± 0.000 | 0.024 ± 0.001 | 0.033 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.086 ± 0.001 | 0.078 ± 0.001 | 0.084 ± 0.006 | 0.112 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.006 ± 0.005 | 0.008 ± 0.000 | 0.068 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.093 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.056 ± 0.015 | 0.062 ± 0.010 | 0.071 ± 0.012 | 0.179 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.010 ± 0.002 | 0.088 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.098 ± 0.029 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.037 ± 0.004 | 0.012 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.016 ± 0.000 | 0.016 ± 0.001 | 0.045 ± 0.004 | 0.024 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.011 ± 0.002 | 0.039 ± 0.006 | 0.016 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.035 ± 0.001 | 0.034 ± 0.000 | 0.072 ± 0.003 | 0.053 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.071 ± 0.013 | 0.008 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.042 ± 0.015 | 0.012 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.009 ± 0.001 | 0.044 ± 0.004 | 0.012 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.001 | 0.015 ± 0.000 | 0.059 ± 0.009 | 0.020 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.020 ± 0.006 | 0.014 ± 0.000 | 0.039 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.039 ± 0.001 | 0.041 ± 0.008 | 0.041 ± 0.002 | 0.130 ± 0.010 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.046 ± 0.003 | 0.044 ± 0.003 | 0.047 ± 0.008 | 0.068 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.161 ± 0.001 | 0.159 ± 0.003 | 0.166 ± 0.016 | 0.279 ± 0.079 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.011 ± 0.001 | 0.087 ± 0.042 | 0.007 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.062 ± 0.008 | 0.007 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.001 | 0.057 ± 0.006 | 0.018 ± 0.010 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.001 | 0.061 ± 0.001 | 0.050 ± 0.037 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.041 ± 0.000 | 0.109 ± 0.017 | 0.012 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.055 ± 0.002 | 0.106 ± 0.002 | 0.016 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.045 ± 0.001 | 0.099 ± 0.002 | 0.014 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.071 ± 0.001 | 0.110 ± 0.003 | 0.055 ± 0.044 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.296 ± 0.005 | 0.331 ± 0.057 | 0.307 ± 0.003 | 0.310 ± 0.010 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 11.471 ± 0.598 | 8.152 ± 0.226 | 5.216 ± 0.395 | 5.157 ± 0.272 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 60.866 ± 6.093 | 66.699 ± 15.401 | 35.450 ± 6.422 | 36.233 ± 16.826 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 6.712 ± 0.849 | 8.216 ± 1.210 | 5.010 ± 0.190 | 5.625 ± 0.030 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 31.013 ± 1.635 | 56.482 ± 8.268 | 28.182 ± 0.634 | 33.470 ± 1.952 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 2.823 ± 0.106 | 1.269 ± 0.012 | 2.450 ± 0.107 | 1.544 ± 0.070 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 18.405 ± 1.472 | 10.415 ± 2.684 | 14.995 ± 0.116 | 10.951 ± 0.995 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 2.387 ± 0.037 | 1.349 ± 0.092 | 1.654 ± 0.048 | 1.485 ± 0.014 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 17.059 ± 0.654 | 11.231 ± 1.024 | 8.701 ± 0.260 | 10.830 ± 1.169 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.016 ± 0.002 | 0.009 ± 0.002 | 0.010 ± 0.000 | 0.034 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.031 ± 0.005 | 0.051 ± 0.007 | 0.070 ± 0.001 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 5.349 ± 0.073 | 3.798 ± 0.232 | 2.883 ± 0.058 | 2.783 ± 0.053 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 32.431 ± 1.872 | 23.087 ± 1.156 | 15.525 ± 0.321 | 15.611 ± 4.970 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 3.390 ± 0.018 | 4.881 ± 1.622 | 2.806 ± 0.239 | 2.734 ± 0.160 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 22.862 ± 1.245 | 24.062 ± 1.137 | 14.751 ± 0.706 | 16.254 ± 1.161 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.104 ± 0.008 | 0.072 ± 0.002 | 0.053 ± 0.017 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.695 ± 0.058 | 0.559 ± 0.025 | 0.528 ± 0.071 | 0.341 ± 0.014 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 3.841 ± 0.635 | 3.680 ± 0.379 | 1.679 ± 0.088 | 1.837 ± 0.191 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.467 ± 0.028 | 0.534 ± 0.018 | 0.667 ± 0.057 | 0.318 ± 0.012 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 2.195 ± 0.455 | 3.652 ± 0.452 | 3.081 ± 0.121 | 1.724 ± 0.157 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.657 ± 0.032 | 0.611 ± 0.106 | 0.495 ± 0.025 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 18.197 ± 0.166 | 14.627 ± 0.578 | 10.463 ± 0.525 | 9.094 ± 0.167 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 111.463 ± 0.447 | 83.691 ± 2.978 | 54.358 ± 0.818 | 56.710 ± 5.113 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 9.045 ± 0.134 | 15.518 ± 1.052 | 9.324 ± 0.152 | 9.020 ± 0.055 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 61.325 ± 6.947 | 94.740 ± 7.760 | 54.889 ± 1.919 | 54.669 ± 3.675 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.023 ± 0.001 | 0.020 ± 0.001 | 0.026 ± 0.002 | 0.106 ± 0.004 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.140 ± 0.009 | 0.127 ± 0.004 | 0.174 ± 0.006 | 0.422 ± 0.142 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.482 ± 0.008 | 0.535 ± 0.010 | 0.657 ± 0.052 | 1.569 ± 0.564 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.068 ± 0.003 | 0.075 ± 0.010 | 0.104 ± 0.004 | 0.080 ± 0.003 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.021 ± 0.000 | 0.027 ± 0.008 | 0.035 ± 0.007 | 0.046 ± 0.012 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.023 ± 0.000 | 0.029 ± 0.000 | 0.064 ± 0.023 | 0.050 ± 0.003 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.029 ± 0.000 | 0.035 ± 0.001 | 0.085 ± 0.032 | 0.114 ± 0.017 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.401 ± 0.008 | 0.429 ± 0.011 | 0.404 ± 0.010 | 0.447 ± 0.007 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.005 ± 0.001 | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.007 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.006 ± 0.001 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.002 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.009 ± 0.001 | 0.008 ± 0.000 | 0.009 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.067 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.007 ± 0.001 | 0.086 ± 0.027 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.087 ± 0.012 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.132 ± 0.003 | 0.023 ± 0.004 | 0.067 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.134 ± 0.009 | 0.024 ± 0.001 | 0.069 ± 0.005 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.150 ± 0.007 | 0.050 ± 0.010 | 0.079 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.053 ± 0.022 | 0.025 ± 0.002 | 0.061 ± 0.005 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.059 ± 0.009 | 0.024 ± 0.000 | 0.082 ± 0.052 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.061 ± 0.001 | 0.035 ± 0.010 | 0.071 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.377 ± 0.013 | 0.027 ± 0.000 | 0.301 ± 0.011 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.680 ± 0.199 | 0.026 ± 0.000 | 0.278 ± 0.042 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.483 ± 0.072 | 0.033 ± 0.004 | 0.295 ± 0.014 | 0.005 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.104 ± 0.005 | 0.028 ± 0.003 | 0.244 ± 0.034 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.140 ± 0.041 | 0.024 ± 0.000 | 0.219 ± 0.069 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.118 ± 0.005 | 0.032 ± 0.004 | 0.229 ± 0.013 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.000 | 0.027 ± 0.003 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.000 | 0.024 ± 0.002 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.004 | 0.033 ± 0.011 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.573 ± 0.043 | 0.041 ± 0.004 | 0.147 ± 0.025 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.566 ± 0.046 | 0.039 ± 0.002 | 0.163 ± 0.007 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.638 ± 0.049 | 0.061 ± 0.029 | 0.153 ± 0.006 | 0.008 ± 0.003 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.107 ± 0.004 | 0.046 ± 0.012 | 0.180 ± 0.013 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.112 ± 0.020 | 0.038 ± 0.000 | 0.171 ± 0.010 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.127 ± 0.012 | 0.059 ± 0.011 | 0.164 ± 0.009 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.031 ± 0.000 | 0.041 ± 0.005 | 0.010 ± 0.004 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.031 ± 0.001 | 0.038 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.035 ± 0.018 | 0.044 ± 0.003 | 0.012 ± 0.003 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.082 ± 0.022 | 0.022 ± 0.002 | 0.198 ± 0.042 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.105 ± 0.011 | 0.020 ± 0.000 | 0.214 ± 0.058 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.082 ± 0.003 | 0.025 ± 0.009 | 0.158 ± 0.017 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.043 ± 0.004 | 0.023 ± 0.003 | 0.084 ± 0.008 | 0.007 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.058 ± 0.007 | 0.021 ± 0.001 | 0.088 ± 0.006 | 0.008 ± 0.004 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.045 ± 0.002 | 0.024 ± 0.001 | 0.073 ± 0.001 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.034 ± 0.006 | 0.011 ± 0.002 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.000 | 0.032 ± 0.002 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.034 ± 0.003 | 0.043 ± 0.003 | 0.026 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.120 ± 0.005 | 0.016 ± 0.000 | 0.107 ± 0.024 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.117 ± 0.047 | 0.020 ± 0.000 | 0.092 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.207 ± 0.047 | 0.041 ± 0.004 | 0.123 ± 0.011 | 0.017 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.047 ± 0.003 | 0.015 ± 0.000 | 0.063 ± 0.005 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.047 ± 0.007 | 0.019 ± 0.006 | 0.062 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.070 ± 0.011 | 0.036 ± 0.007 | 0.073 ± 0.005 | 0.028 ± 0.008 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.003 ± 0.001 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.007 ± 0.006 | 0.004 ± 0.001 | 0.002 ± 0.000 | 0.006 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.003 ± 0.001 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.011 ± 0.004 | 0.006 ± 0.000 | 0.008 ± 0.002 | 0.028 ± 0.004 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.003 ± 0.001 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.028 ± 0.004 | 0.008 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.032 ± 0.001 | 0.008 ± 0.003 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.035 ± 0.015 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.005 ± 0.001 | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.005 ± 0.001 | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.007 ± 0.003 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.011 ± 0.003 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.003 | 0.006 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.006 ± 0.001 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.011 ± 0.001 | 0.007 ± 0.000 | 0.008 ± 0.001 | 0.015 ± 0.004 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.017 ± 0.003 | 0.013 ± 0.001 | 0.016 ± 0.002 | 0.019 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 21.3x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 20.6x faster than the slowest successful cell (`jax-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 18.9x faster than the slowest successful cell (`jax-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 24.5x faster than the slowest successful cell (`jax-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 24.3x faster than the slowest successful cell (`jax-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 17.5x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `tenferro-eager` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 19.8x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 25.0x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.3x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 23.3x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 16.2x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.6x faster than the slowest successful cell (`tenferro-eager`, 0.132 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.5x faster than the slowest successful cell (`tenferro-eager`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 17.7x faster than the slowest successful cell (`tenferro-eager`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 51.0x faster than the slowest successful cell (`tenferro-eager`, 0.377 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`tenferro-eager`, 0.377 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 89.2x faster than the slowest successful cell (`tenferro-eager`, 0.680 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 26.0x faster than the slowest successful cell (`tenferro-eager`, 0.680 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 90.7x faster than the slowest successful cell (`tenferro-eager`, 0.483 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`tenferro-eager`, 0.483 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 32.8x faster than the slowest successful cell (`pytorch-cpu`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 22.9x faster than the slowest successful cell (`pytorch-cpu`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 40.7x faster than the slowest successful cell (`pytorch-cpu`, 0.229 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 78.8x faster than the slowest successful cell (`tenferro-eager`, 0.573 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.1x faster than the slowest successful cell (`tenferro-eager`, 0.573 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 64.7x faster than the slowest successful cell (`tenferro-eager`, 0.566 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`tenferro-eager`, 0.566 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 77.6x faster than the slowest successful cell (`tenferro-eager`, 0.638 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`tenferro-eager`, 0.638 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 25.0x faster than the slowest successful cell (`pytorch-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 20.3x faster than the slowest successful cell (`pytorch-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 30.0x faster than the slowest successful cell (`pytorch-cpu`, 0.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 32.7x faster than the slowest successful cell (`pytorch-cpu`, 0.214 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.214 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 20.8x faster than the slowest successful cell (`pytorch-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 11.6x faster than the slowest successful cell (`tenferro-eager`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 12.0x faster than the slowest successful cell (`tenferro-eager`, 0.207 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

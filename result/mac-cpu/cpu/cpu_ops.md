# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `d6655b567e09aa29c3201655e1de418716934876`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261107_172300/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261107_172300/run.yaml`
- Timestamp: `2026-11-07T17:23:00.301852+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261107_172300`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261107_172300/cpu_ops_t1_20261107_172300.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.001 | 0.008 ± 0.002 | 0.008 ± 0.002 | 0.021 ± 0.009 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.022 ± 0.015 | 0.020 ± 0.002 | 0.046 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.028 ± 0.015 | 0.028 ± 0.006 | 0.028 ± 0.010 | 0.031 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.084 ± 0.002 | 0.094 ± 0.008 | 0.098 ± 0.010 | 0.114 ± 0.061 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.011 ± 0.003 | 0.076 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.009 ± 0.002 | 0.087 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.062 ± 0.015 | 0.057 ± 0.008 | 0.044 ± 0.018 | 0.107 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.001 | 0.004 ± 0.003 | 0.010 ± 0.001 | 0.222 ± 0.086 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.091 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.001 | 0.008 ± 0.001 | 0.007 ± 0.003 | 0.013 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.016 ± 0.000 | 0.018 ± 0.002 | 0.020 ± 0.002 | 0.033 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.014 ± 0.001 | 0.015 ± 0.008 | 0.016 ± 0.003 | 0.021 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.037 ± 0.001 | 0.042 ± 0.003 | 0.042 ± 0.007 | 0.051 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.001 | 0.007 ± 0.001 | 0.012 ± 0.005 | 0.009 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.012 ± 0.007 | 0.015 ± 0.001 | 0.015 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.012 ± 0.002 | 0.013 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.002 | 0.022 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.015 ± 0.001 | 0.015 ± 0.002 | 0.016 ± 0.001 | 0.045 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.041 ± 0.000 | 0.050 ± 0.008 | 0.053 ± 0.023 | 0.162 ± 0.024 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.049 ± 0.002 | 0.059 ± 0.018 | 0.074 ± 0.034 | 0.097 ± 0.016 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.168 ± 0.013 | 0.192 ± 0.032 | 0.162 ± 0.003 | 0.237 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.002 | 0.093 ± 0.024 | 0.007 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.016 | 0.065 ± 0.024 | 0.009 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.011 ± 0.004 | 0.067 ± 0.006 | 0.015 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.064 ± 0.005 | 0.029 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.037 ± 0.015 | 0.064 ± 0.047 | 0.021 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.057 ± 0.002 | 0.072 ± 0.021 | 0.019 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.016 | 0.056 ± 0.006 | 0.009 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.073 ± 0.009 | 0.060 ± 0.002 | 0.028 ± 0.012 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.319 ± 0.097 | 0.285 ± 0.031 | 0.387 ± 0.125 | 0.371 ± 0.061 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 11.816 ± 0.933 | 7.932 ± 0.652 | 6.640 ± 0.761 | 7.426 ± 0.928 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 53.720 ± 3.919 | 62.124 ± 6.953 | 39.918 ± 7.902 | 38.458 ± 3.814 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 5.955 ± 0.950 | 8.091 ± 0.085 | 7.623 ± 2.475 | 8.378 ± 0.974 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 28.087 ± 0.583 | 61.567 ± 10.186 | 30.441 ± 0.673 | 56.484 ± 16.759 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 2.736 ± 0.335 | 1.081 ± 0.034 | 2.650 ± 0.439 | 2.961 ± 0.108 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 16.852 ± 2.938 | 11.982 ± 3.135 | 16.962 ± 3.136 | 19.892 ± 0.849 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 2.245 ± 0.429 | 1.251 ± 0.003 | 1.810 ± 0.697 | 2.528 ± 0.109 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 14.708 ± 0.111 | 11.531 ± 3.570 | 8.796 ± 0.273 | 18.653 ± 1.183 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.015 ± 0.002 | 0.007 ± 0.001 | 0.012 ± 0.003 | 0.029 ± 0.004 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.001 | 0.051 ± 0.040 | 0.071 ± 0.011 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 6.621 ± 4.035 | 3.586 ± 0.068 | 3.792 ± 1.038 | 5.726 ± 1.549 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 29.400 ± 1.813 | 23.858 ± 0.787 | 17.966 ± 0.468 | 36.710 ± 5.331 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 4.389 ± 2.136 | 3.626 ± 0.066 | 3.217 ± 0.669 | 5.980 ± 0.464 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 21.472 ± 1.123 | 24.975 ± 0.541 | 17.372 ± 0.521 | 29.828 ± 3.763 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.083 ± 0.001 | 0.092 ± 0.058 | 0.049 ± 0.009 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 0.635 ± 0.021 | 0.479 ± 0.006 | 0.488 ± 0.038 | 0.348 ± 0.032 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 3.686 ± 3.664 | 4.103 ± 0.994 | 2.199 ± 0.464 | 1.648 ± 0.129 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.358 ± 0.046 | 0.495 ± 0.005 | 0.614 ± 0.080 | 0.289 ± 0.006 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 2.560 ± 0.282 | 3.661 ± 1.153 | 3.112 ± 0.292 | 1.638 ± 0.222 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.643 ± 0.033 | 0.573 ± 0.265 | 0.467 ± 0.013 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 25.830 ± 2.918 | 15.132 ± 0.700 | 11.363 ± 1.389 | 13.079 ± 0.941 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 111.932 ± 1.811 | 109.606 ± 6.664 | 66.883 ± 9.207 | 88.139 ± 11.366 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 10.780 ± 2.281 | 14.743 ± 0.117 | 10.954 ± 1.049 | 12.958 ± 0.903 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 59.722 ± 16.077 | 107.391 ± 2.334 | 60.570 ± 4.484 | 69.177 ± 20.880 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.026 ± 0.002 | 0.019 ± 0.003 | 0.031 ± 0.079 | 0.128 ± 0.037 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.188 ± 0.071 | 0.122 ± 0.003 | 0.238 ± 0.054 | 0.931 ± 0.139 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.600 ± 0.048 | 0.484 ± 0.042 | 0.777 ± 1.049 | 3.332 ± 0.367 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.070 ± 0.033 | 0.063 ± 0.001 | 0.072 ± 0.001 | 0.085 ± 0.017 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.020 ± 0.001 | 0.021 ± 0.000 | 0.035 ± 0.012 | 0.035 ± 0.029 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.022 ± 0.002 | 0.025 ± 0.000 | 0.046 ± 0.006 | 0.054 ± 0.007 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.027 ± 0.000 | 0.032 ± 0.007 | 0.062 ± 0.008 | 0.096 ± 0.004 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.433 ± 0.315 | 0.395 ± 0.018 | 0.479 ± 0.084 | 0.413 ± 0.016 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.007 ± 0.003 | 0.002 ± 0.000 | 0.005 ± 0.002 | 0.009 ± 0.002 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.006 | 0.012 ± 0.005 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.008 ± 0.004 | 0.007 ± 0.000 | 0.012 ± 0.006 | 0.014 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.125 ± 0.072 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.003 ± 0.001 | 0.005 ± 0.001 | 0.015 ± 0.007 | 0.101 ± 0.013 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.006 ± 0.004 | 0.005 ± 0.000 | 0.012 ± 0.002 | 0.106 ± 0.015 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.148 ± 0.052 | 0.015 ± 0.000 | 0.094 ± 0.078 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.132 ± 0.021 | 0.017 ± 0.001 | 0.119 ± 0.059 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.216 ± 0.328 | 0.028 ± 0.000 | 0.139 ± 0.160 | 0.011 ± 0.003 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.053 ± 0.038 | 0.014 ± 0.000 | 0.091 ± 0.009 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.065 ± 0.009 | 0.018 ± 0.000 | 0.115 ± 0.113 | 0.013 ± 0.004 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.142 ± 0.059 | 0.025 ± 0.001 | 0.079 ± 0.010 | 0.009 ± 0.004 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.510 ± 0.151 | 0.017 ± 0.000 | 0.214 ± 0.076 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.453 ± 0.251 | 0.019 ± 0.000 | 0.215 ± 0.061 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.644 ± 0.496 | 0.020 ± 0.001 | 0.150 ± 0.044 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.101 ± 0.016 | 0.016 ± 0.001 | 0.125 ± 0.220 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.133 ± 0.076 | 0.016 ± 0.001 | 0.148 ± 0.038 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.119 ± 0.145 | 0.022 ± 0.001 | 0.102 ± 0.093 | 0.006 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.000 | 0.032 ± 0.019 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.001 | 0.042 ± 0.014 | 0.007 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.000 | 0.041 ± 0.011 | 0.011 ± 0.009 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.809 ± 0.095 | 0.026 ± 0.002 | 0.112 ± 0.031 | 0.015 ± 0.011 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.658 ± 0.104 | 0.028 ± 0.002 | 0.124 ± 0.060 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 2.381 ± 0.395 | 0.033 ± 0.009 | 0.186 ± 0.098 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.108 ± 0.062 | 0.027 ± 0.002 | 0.116 ± 0.264 | 0.011 ± 0.007 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.094 ± 0.048 | 0.028 ± 0.002 | 0.108 ± 0.011 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.091 ± 0.189 | 0.036 ± 0.003 | 0.099 ± 0.168 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.001 | 0.146 ± 0.087 | 0.012 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.001 | 0.053 ± 0.038 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.005 | 0.048 ± 0.006 | 0.013 ± 0.005 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.089 ± 0.021 | 0.013 ± 0.000 | 0.417 ± 0.318 | 0.011 ± 0.004 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.154 ± 0.083 | 0.013 ± 0.000 | 0.314 ± 0.134 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.085 ± 0.037 | 0.016 ± 0.000 | 0.172 ± 0.158 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.048 ± 0.019 | 0.013 ± 0.000 | 0.172 ± 0.061 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.045 ± 0.053 | 0.013 ± 0.000 | 0.104 ± 0.044 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.038 ± 0.010 | 0.016 ± 0.001 | 0.088 ± 0.013 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.076 ± 0.070 | 0.011 ± 0.012 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.040 ± 0.014 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.002 | 0.051 ± 0.004 | 0.021 ± 0.009 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.152 ± 0.056 | 0.011 ± 0.000 | 0.417 ± 0.306 | 0.012 ± 0.006 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.112 ± 0.028 | 0.015 ± 0.000 | 0.120 ± 0.070 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.154 ± 0.029 | 0.030 ± 0.001 | 0.142 ± 0.033 | 0.021 ± 0.003 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.046 ± 0.006 | 0.010 ± 0.000 | 0.096 ± 0.028 | 0.014 ± 0.018 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.041 ± 0.005 | 0.015 ± 0.001 | 0.080 ± 0.044 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.182 ± 0.082 | 0.026 ± 0.002 | 0.112 ± 0.061 | 0.020 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.001 | 0.007 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.005 ± 0.001 | 0.003 ± 0.000 | 0.003 ± 0.001 | 0.005 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.002 ± 0.001 | 0.006 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.009 ± 0.003 | 0.005 ± 0.000 | 0.012 ± 0.010 | 0.024 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.001 | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.007 ± 0.002 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.010 ± 0.007 | 0.002 ± 0.000 | 0.004 ± 0.002 | 0.009 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.004 ± 0.001 | 0.003 ± 0.000 | 0.004 ± 0.003 | 0.010 ± 0.004 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.005 ± 0.003 | 0.004 ± 0.001 | 0.005 ± 0.003 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.005 ± 0.002 | 0.004 ± 0.000 | 0.010 ± 0.002 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.006 ± 0.001 | 0.004 ± 0.000 | 0.011 ± 0.003 | 0.008 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.005 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.004 ± 0.002 | 0.004 ± 0.000 | 0.012 ± 0.003 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.005 ± 0.001 | 0.005 ± 0.001 | 0.010 ± 0.006 | 0.007 ± 0.003 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.004 ± 0.004 | 0.005 ± 0.000 | 0.011 ± 0.011 | 0.011 ± 0.004 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.006 ± 0.001 | 0.003 ± 0.000 | 0.008 ± 0.002 | 0.015 ± 0.002 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.008 ± 0.001 | 0.006 ± 0.001 | 0.022 ± 0.017 | 0.019 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.021 ± 0.008 | 0.013 ± 0.000 | 0.021 ± 0.002 | 0.021 ± 0.004 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 26.6x faster than the slowest successful cell (`jax-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`jax-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 21.3x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 21.3x faster than the slowest successful cell (`jax-cpu`, 0.222 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 69.9x faster than the slowest successful cell (`jax-cpu`, 0.222 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 53.3x faster than the slowest successful cell (`jax-cpu`, 0.222 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 18.2x faster than the slowest successful cell (`jax-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 36.7x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 28.2x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 34.2x faster than the slowest successful cell (`jax-cpu`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 20.2x faster than the slowest successful cell (`jax-cpu`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 16.7x faster than the slowest successful cell (`jax-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 22.5x faster than the slowest successful cell (`jax-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.1x faster than the slowest successful cell (`tenferro-eager`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.9x faster than the slowest successful cell (`tenferro-eager`, 0.132 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 20.6x faster than the slowest successful cell (`tenferro-eager`, 0.216 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 61.7x faster than the slowest successful cell (`tenferro-eager`, 0.510 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 29.5x faster than the slowest successful cell (`tenferro-eager`, 0.510 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 46.6x faster than the slowest successful cell (`tenferro-eager`, 0.453 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 24.5x faster than the slowest successful cell (`tenferro-eager`, 0.453 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 77.8x faster than the slowest successful cell (`tenferro-eager`, 0.644 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 32.0x faster than the slowest successful cell (`tenferro-eager`, 0.644 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 19.1x faster than the slowest successful cell (`tenferro-eager`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 55.4x faster than the slowest successful cell (`tenferro-eager`, 0.809 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 31.6x faster than the slowest successful cell (`tenferro-eager`, 0.809 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 94.5x faster than the slowest successful cell (`tenferro-eager`, 0.658 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 23.6x faster than the slowest successful cell (`tenferro-eager`, 0.658 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 228.5x faster than the slowest successful cell (`tenferro-eager`, 2.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 2.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 72.6x faster than the slowest successful cell (`tenferro-eager`, 2.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 38.8x faster than the slowest successful cell (`pytorch-cpu`, 0.417 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 31.5x faster than the slowest successful cell (`pytorch-cpu`, 0.417 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 36.6x faster than the slowest successful cell (`pytorch-cpu`, 0.314 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 23.7x faster than the slowest successful cell (`pytorch-cpu`, 0.314 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 20.3x faster than the slowest successful cell (`pytorch-cpu`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 24.2x faster than the slowest successful cell (`pytorch-cpu`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 36.2x faster than the slowest successful cell (`pytorch-cpu`, 0.417 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 38.9x faster than the slowest successful cell (`pytorch-cpu`, 0.417 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261107_175904/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261107_175904/run.yaml`
- Timestamp: `2026-11-07T17:59:04.170477+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261107_175904`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261107_175904/cpu_ops_t4_20261107_175904.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.010 ± 0.003 | 0.010 ± 0.007 | 0.016 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.026 ± 0.008 | 0.024 ± 0.004 | 0.072 ± 0.033 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.024 ± 0.001 | 0.028 ± 0.015 | 0.031 ± 0.008 | 0.038 ± 0.012 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.117 ± 0.090 | 0.111 ± 0.047 | 0.101 ± 0.034 | 0.208 ± 0.106 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.003 | 0.092 ± 0.130 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.002 | 0.009 ± 0.003 | 0.086 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.051 ± 0.005 | 0.049 ± 0.001 | 0.109 ± 0.042 | 0.334 ± 0.091 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.011 ± 0.003 | 0.083 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.006 ± 0.001 | 0.008 ± 0.005 | 0.013 ± 0.006 | 0.098 ± 0.021 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.050 ± 0.009 | 0.018 ± 0.008 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.017 ± 0.001 | 0.017 ± 0.001 | 0.085 ± 0.043 | 0.028 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.012 ± 0.002 | 0.058 ± 0.010 | 0.020 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.036 ± 0.003 | 0.043 ± 0.006 | 0.134 ± 0.041 | 0.100 ± 0.030 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.010 ± 0.003 | 0.049 ± 0.023 | 0.009 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.015 ± 0.007 | 0.083 ± 0.041 | 0.014 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.001 | 0.011 ± 0.002 | 0.067 ± 0.047 | 0.012 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.001 | 0.017 ± 0.004 | 0.085 ± 0.032 | 0.046 ± 0.014 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.014 ± 0.001 | 0.016 ± 0.002 | 0.016 ± 0.003 | 0.058 ± 0.019 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.043 ± 0.001 | 0.050 ± 0.020 | 0.049 ± 0.029 | 0.152 ± 0.031 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.047 ± 0.001 | 0.051 ± 0.003 | 0.058 ± 0.020 | 0.075 ± 0.024 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.166 ± 0.057 | 0.203 ± 0.143 | 0.209 ± 0.047 | 0.264 ± 0.155 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.002 | 0.131 ± 0.143 | 0.008 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.002 | 0.070 ± 0.028 | 0.007 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.011 | 0.063 ± 0.017 | 0.013 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.031 ± 0.004 | 0.082 ± 0.016 | 0.081 ± 0.020 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.046 ± 0.021 | 0.157 ± 0.050 | 0.014 ± 0.011 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.069 ± 0.019 | 0.186 ± 0.046 | 0.018 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.046 ± 0.003 | 0.122 ± 0.120 | 0.009 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.129 ± 0.012 | 0.230 ± 0.109 | 0.042 ± 0.028 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.415 ± 0.093 | 0.554 ± 0.036 | 0.303 ± 0.031 | 0.324 ± 0.012 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 10.226 ± 0.224 | 8.275 ± 1.093 | 6.097 ± 1.415 | 5.400 ± 0.186 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 69.410 ± 6.901 | 66.258 ± 7.289 | 33.034 ± 4.625 | 40.267 ± 8.820 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 5.656 ± 1.011 | 9.069 ± 2.272 | 4.927 ± 0.100 | 5.372 ± 0.178 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 28.482 ± 2.239 | 55.977 ± 15.813 | 32.950 ± 3.343 | 42.506 ± 14.394 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 2.521 ± 0.599 | 2.448 ± 0.565 | 2.547 ± 0.151 | 1.448 ± 0.033 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 16.039 ± 0.404 | 18.834 ± 8.391 | 21.615 ± 2.715 | 16.350 ± 3.547 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 2.613 ± 0.499 | 2.215 ± 1.262 | 1.662 ± 0.199 | 1.445 ± 0.032 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 14.885 ± 0.045 | 15.180 ± 2.948 | 14.922 ± 8.414 | 17.675 ± 3.760 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.019 ± 0.002 | 0.009 ± 0.007 | 0.011 ± 0.004 | 0.043 ± 0.026 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.033 ± 0.007 | 0.082 ± 0.031 | 0.072 ± 0.006 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 4.983 ± 0.731 | 4.170 ± 0.307 | 2.724 ± 0.026 | 3.403 ± 0.698 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 43.446 ± 15.007 | 31.208 ± 8.520 | 15.917 ± 0.405 | 25.176 ± 11.990 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 3.575 ± 0.371 | 3.791 ± 0.213 | 2.818 ± 2.946 | 3.093 ± 0.422 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 35.359 ± 10.791 | 35.832 ± 1.996 | 15.792 ± 5.099 | 19.946 ± 4.452 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.181 ± 0.018 | 0.071 ± 0.009 | 0.053 ± 0.010 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.829 ± 0.061 | 0.682 ± 0.349 | 0.499 ± 0.045 | 0.315 ± 0.006 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 3.123 ± 0.391 | 4.317 ± 1.629 | 1.915 ± 0.231 | 3.824 ± 2.655 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.479 ± 0.058 | 0.793 ± 0.336 | 0.647 ± 0.046 | 0.307 ± 0.025 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 1.996 ± 0.324 | 3.166 ± 0.210 | 4.820 ± 3.134 | 2.581 ± 1.862 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.859 ± 0.614 | 0.733 ± 0.277 | 0.479 ± 0.009 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 20.318 ± 4.740 | 20.662 ± 3.732 | 10.790 ± 2.127 | 10.319 ± 0.441 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 114.094 ± 14.113 | 121.687 ± 20.610 | 75.409 ± 22.046 | 60.180 ± 18.985 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 9.160 ± 0.339 | 14.880 ± 0.716 | 8.968 ± 0.236 | 10.163 ± 0.109 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 59.314 ± 7.751 | 133.798 ± 26.384 | 57.674 ± 9.997 | 59.587 ± 7.870 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.030 ± 0.050 | 0.021 ± 0.002 | 0.031 ± 0.003 | 0.194 ± 0.069 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.173 ± 0.023 | 0.127 ± 0.006 | 0.173 ± 0.020 | 0.779 ± 0.089 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.529 ± 0.076 | 1.236 ± 1.084 | 0.631 ± 0.053 | 1.445 ± 1.191 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.107 ± 0.037 | 0.115 ± 0.070 | 0.101 ± 0.023 | 0.081 ± 0.007 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.025 ± 0.001 | 0.024 ± 0.025 | 0.036 ± 0.011 | 0.039 ± 0.011 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.023 ± 0.003 | 0.028 ± 0.000 | 0.051 ± 0.005 | 0.050 ± 0.011 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.028 ± 0.064 | 0.035 ± 0.002 | 0.074 ± 0.050 | 0.112 ± 0.010 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.484 ± 0.026 | 0.576 ± 0.200 | 0.382 ± 0.121 | 0.555 ± 0.046 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.005 ± 0.001 | 0.004 ± 0.001 | 0.005 ± 0.002 | 0.009 ± 0.007 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.014 ± 0.010 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.008 ± 0.001 | 0.014 ± 0.013 | 0.009 ± 0.002 | 0.017 ± 0.006 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.004 | 0.121 ± 0.093 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.002 | 0.009 ± 0.002 | 0.010 ± 0.002 | 0.083 ± 0.032 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.003 ± 0.002 | 0.008 ± 0.002 | 0.010 ± 0.002 | 0.159 ± 0.075 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.162 ± 0.047 | 0.027 ± 0.003 | 0.110 ± 0.040 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.136 ± 0.003 | 0.030 ± 0.001 | 0.081 ± 0.031 | 0.011 ± 0.004 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.153 ± 0.103 | 0.051 ± 0.023 | 0.097 ± 0.018 | 0.026 ± 0.008 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.047 ± 0.004 | 0.029 ± 0.005 | 0.088 ± 0.060 | 0.011 ± 0.011 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.044 ± 0.005 | 0.033 ± 0.008 | 0.059 ± 0.005 | 0.013 ± 0.003 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.058 ± 0.002 | 0.049 ± 0.029 | 0.091 ± 0.032 | 0.043 ± 0.022 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.439 ± 0.014 | 0.031 ± 0.002 | 0.314 ± 0.038 | 0.011 ± 0.004 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.473 ± 0.063 | 0.034 ± 0.002 | 0.243 ± 0.048 | 0.017 ± 0.004 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.424 ± 0.006 | 0.039 ± 0.032 | 0.286 ± 0.076 | 0.019 ± 0.018 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.114 ± 0.008 | 0.031 ± 0.007 | 0.273 ± 0.118 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.125 ± 0.008 | 0.040 ± 0.009 | 0.198 ± 0.015 | 0.018 ± 0.005 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.126 ± 0.009 | 0.043 ± 0.003 | 0.217 ± 0.055 | 0.017 ± 0.012 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.001 | 0.024 ± 0.007 | 0.009 ± 0.004 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.003 | 0.025 ± 0.001 | 0.023 ± 0.010 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.005 | 0.030 ± 0.001 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.892 ± 0.240 | 0.051 ± 0.029 | 0.157 ± 0.012 | 0.010 ± 0.007 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.526 ± 0.019 | 0.050 ± 0.002 | 0.164 ± 0.037 | 0.013 ± 0.006 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.560 ± 0.059 | 0.079 ± 0.038 | 0.168 ± 0.010 | 0.023 ± 0.005 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.232 ± 0.031 | 0.041 ± 0.011 | 0.235 ± 0.220 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.114 ± 0.019 | 0.049 ± 0.002 | 0.148 ± 0.020 | 0.016 ± 0.007 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.127 ± 0.008 | 0.080 ± 0.031 | 0.172 ± 0.010 | 0.023 ± 0.009 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.034 ± 0.003 | 0.043 ± 0.007 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.035 ± 0.003 | 0.041 ± 0.002 | 0.012 ± 0.010 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.066 ± 0.035 | 0.041 ± 0.001 | 0.014 ± 0.009 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.083 ± 0.004 | 0.025 ± 0.005 | 0.191 ± 0.108 | 0.012 ± 0.006 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.096 ± 0.006 | 0.041 ± 0.023 | 0.158 ± 0.030 | 0.011 ± 0.007 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.124 ± 0.030 | 0.028 ± 0.010 | 0.205 ± 0.032 | 0.013 ± 0.005 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.048 ± 0.003 | 0.025 ± 0.004 | 0.088 ± 0.014 | 0.014 ± 0.010 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.047 ± 0.014 | 0.027 ± 0.004 | 0.083 ± 0.048 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.055 ± 0.006 | 0.029 ± 0.026 | 0.098 ± 0.010 | 0.029 ± 0.010 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.001 | 0.032 ± 0.003 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.004 | 0.035 ± 0.002 | 0.017 ± 0.005 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.052 ± 0.027 | 0.042 ± 0.006 | 0.020 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.164 ± 0.016 | 0.019 ± 0.002 | 0.099 ± 0.007 | 0.013 ± 0.005 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.115 ± 0.026 | 0.022 ± 0.001 | 0.100 ± 0.008 | 0.017 ± 0.044 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.132 ± 0.019 | 0.048 ± 0.019 | 0.118 ± 0.004 | 0.054 ± 0.011 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.056 ± 0.001 | 0.018 ± 0.005 | 0.059 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.054 ± 0.015 | 0.023 ± 0.001 | 0.064 ± 0.001 | 0.025 ± 0.006 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.051 ± 0.004 | 0.052 ± 0.013 | 0.074 ± 0.002 | 0.020 ± 0.011 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.003 ± 0.001 | 0.003 ± 0.001 | 0.001 ± 0.000 | 0.010 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.002 ± 0.003 | 0.012 ± 0.005 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.010 ± 0.005 | 0.005 ± 0.000 | 0.009 ± 0.003 | 0.045 ± 0.029 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.004 ± 0.001 | 0.001 ± 0.000 | 0.007 ± 0.003 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.006 ± 0.003 | 0.003 ± 0.002 | 0.047 ± 0.010 | 0.010 ± 0.011 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.051 ± 0.022 | 0.009 ± 0.003 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.008 ± 0.003 | 0.008 ± 0.003 | 0.034 ± 0.017 | 0.013 ± 0.005 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.005 ± 0.001 | 0.006 ± 0.001 | 0.009 ± 0.001 | 0.010 ± 0.009 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.006 ± 0.001 | 0.007 ± 0.002 | 0.010 ± 0.004 | 0.007 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.004 ± 0.001 | 0.009 ± 0.005 | 0.009 ± 0.001 | 0.012 ± 0.006 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.004 ± 0.000 | 0.006 ± 0.002 | 0.011 ± 0.002 | 0.013 ± 0.012 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.009 ± 0.001 | 0.008 ± 0.005 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.005 ± 0.001 | 0.009 ± 0.004 | 0.009 ± 0.000 | 0.007 ± 0.004 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.006 ± 0.000 | 0.005 ± 0.005 | 0.006 ± 0.002 | 0.013 ± 0.004 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.007 ± 0.001 | 0.007 ± 0.001 | 0.010 ± 0.002 | 0.022 ± 0.008 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.014 ± 0.001 | 0.029 ± 0.028 | 0.015 ± 0.004 | 0.035 ± 0.028 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 28.2x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 24.4x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 18.5x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 21.8x faster than the slowest successful cell (`jax-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 19.9x faster than the slowest successful cell (`jax-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 16.8x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 37.9x faster than the slowest successful cell (`jax-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`jax-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 29.0x faster than the slowest successful cell (`jax-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 16.5x faster than the slowest successful cell (`jax-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 52.3x faster than the slowest successful cell (`jax-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 19.6x faster than the slowest successful cell (`jax-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 20.3x faster than the slowest successful cell (`tenferro-eager`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.5x faster than the slowest successful cell (`tenferro-eager`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 41.6x faster than the slowest successful cell (`tenferro-eager`, 0.439 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`tenferro-eager`, 0.439 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 28.5x faster than the slowest successful cell (`tenferro-eager`, 0.473 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.0x faster than the slowest successful cell (`tenferro-eager`, 0.473 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 22.6x faster than the slowest successful cell (`tenferro-eager`, 0.424 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`tenferro-eager`, 0.424 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 35.3x faster than the slowest successful cell (`pytorch-cpu`, 0.273 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 87.6x faster than the slowest successful cell (`tenferro-eager`, 0.892 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`tenferro-eager`, 0.892 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 41.1x faster than the slowest successful cell (`tenferro-eager`, 0.526 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`tenferro-eager`, 0.526 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 24.1x faster than the slowest successful cell (`tenferro-eager`, 0.560 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

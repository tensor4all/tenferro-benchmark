# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `ad9aa8a2fc06c36f8ce1d1dfb1ce9fb1cd9a9c46`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261109_172454/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261109_172454/run.yaml`
- Timestamp: `2026-11-09T17:24:54.736713+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261109_172454`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261109_172454/cpu_ops_t1_20261109_172454.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.001 | 0.010 ± 0.005 | 0.009 ± 0.002 | 0.025 ± 0.011 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.023 ± 0.002 | 0.032 ± 0.008 | 0.020 ± 0.001 | 0.101 ± 0.055 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.028 ± 0.025 | 0.032 ± 0.002 | 0.025 ± 0.000 | 0.062 ± 0.026 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.085 ± 0.015 | 0.087 ± 0.009 | 0.092 ± 0.008 | 0.119 ± 0.023 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.004 ± 0.002 | 0.011 ± 0.002 | 0.099 ± 0.070 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.005 | 0.006 ± 0.003 | 0.009 ± 0.001 | 0.126 ± 0.130 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.064 ± 0.010 | 0.062 ± 0.027 | 0.042 ± 0.019 | 0.135 ± 0.034 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.004 ± 0.001 | 0.004 ± 0.001 | 0.009 ± 0.001 | 0.183 ± 0.118 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.006 ± 0.002 | 0.006 ± 0.002 | 0.011 ± 0.001 | 0.096 ± 0.013 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.001 | 0.011 ± 0.005 | 0.007 ± 0.000 | 0.018 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.020 ± 0.006 | 0.022 ± 0.009 | 0.019 ± 0.006 | 0.042 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.015 ± 0.001 | 0.013 ± 0.003 | 0.012 ± 0.001 | 0.029 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.044 ± 0.004 | 0.039 ± 0.002 | 0.043 ± 0.002 | 0.049 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.010 ± 0.003 | 0.014 ± 0.005 | 0.013 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.013 ± 0.007 | 0.018 ± 0.006 | 0.018 ± 0.006 | 0.020 ± 0.013 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.002 | 0.011 ± 0.007 | 0.013 ± 0.002 | 0.015 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.002 | 0.014 ± 0.003 | 0.026 ± 0.006 | 0.022 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.015 ± 0.001 | 0.015 ± 0.001 | 0.015 ± 0.001 | 0.054 ± 0.018 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.069 ± 0.018 | 0.055 ± 0.017 | 0.058 ± 0.031 | 0.192 ± 0.036 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.053 ± 0.008 | 0.074 ± 0.020 | 0.046 ± 0.001 | 0.108 ± 0.072 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.210 ± 0.023 | 0.195 ± 0.018 | 0.185 ± 0.028 | 0.259 ± 0.022 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.008 | 0.078 ± 0.012 | 0.009 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.058 ± 0.014 | 0.009 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.012 | 0.067 ± 0.006 | 0.020 ± 0.012 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.001 | 0.073 ± 0.051 | 0.030 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.043 ± 0.023 | 0.056 ± 0.008 | 0.026 ± 0.029 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.059 ± 0.004 | 0.055 ± 0.004 | 0.022 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.046 ± 0.020 | 0.054 ± 0.005 | 0.025 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.068 ± 0.005 | 0.070 ± 0.007 | 0.033 ± 0.006 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.388 ± 0.073 | 0.329 ± 0.035 | 0.314 ± 0.008 | 0.338 ± 0.052 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 12.982 ± 2.296 | 8.334 ± 0.546 | 5.514 ± 0.311 | 7.176 ± 0.602 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 72.339 ± 2.928 | 48.127 ± 7.627 | 33.598 ± 1.574 | 66.609 ± 22.296 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 6.880 ± 0.669 | 8.052 ± 0.198 | 5.374 ± 0.254 | 7.044 ± 0.764 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 34.480 ± 7.485 | 57.444 ± 0.641 | 29.813 ± 1.951 | 53.011 ± 5.390 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 3.095 ± 0.181 | 1.290 ± 0.055 | 2.148 ± 0.093 | 3.999 ± 1.453 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 18.616 ± 1.717 | 9.642 ± 0.757 | 16.246 ± 0.680 | 24.832 ± 0.866 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 2.964 ± 1.362 | 1.306 ± 0.027 | 1.337 ± 0.233 | 2.988 ± 0.380 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 16.741 ± 4.316 | 11.052 ± 4.757 | 8.537 ± 0.169 | 25.766 ± 2.131 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.023 ± 0.004 | 0.007 ± 0.000 | 0.013 ± 0.009 | 0.037 ± 0.011 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.000 | 0.050 ± 0.006 | 0.072 ± 0.010 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 11.088 ± 2.727 | 3.811 ± 0.153 | 2.843 ± 0.248 | 5.517 ± 1.022 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 31.207 ± 4.834 | 22.104 ± 3.651 | 17.563 ± 0.332 | 41.720 ± 6.274 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 4.317 ± 1.156 | 3.900 ± 0.094 | 2.960 ± 0.357 | 4.782 ± 0.270 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 29.250 ± 14.078 | 20.281 ± 0.789 | 17.015 ± 0.822 | 34.395 ± 2.210 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.080 ± 0.003 | 0.073 ± 0.012 | 0.052 ± 0.019 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 0.787 ± 0.172 | 0.637 ± 0.074 | 0.471 ± 0.055 | 0.367 ± 0.070 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 3.704 ± 0.568 | 4.624 ± 1.268 | 2.057 ± 0.093 | 3.464 ± 1.219 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.455 ± 0.114 | 0.547 ± 0.026 | 0.638 ± 0.097 | 0.388 ± 0.040 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 3.605 ± 1.220 | 3.493 ± 1.054 | 3.206 ± 0.370 | 2.506 ± 0.644 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.579 ± 0.037 | 0.533 ± 0.043 | 0.593 ± 0.038 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 29.484 ± 3.290 | 13.871 ± 0.712 | 10.587 ± 0.391 | 12.164 ± 1.152 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 110.877 ± 4.949 | 89.552 ± 5.473 | 64.984 ± 3.012 | 94.908 ± 7.111 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 14.803 ± 10.388 | 14.999 ± 2.856 | 9.686 ± 0.080 | 11.032 ± 0.255 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 55.349 ± 3.779 | 103.588 ± 22.155 | 56.117 ± 1.629 | 85.119 ± 5.348 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.026 ± 0.042 | 0.019 ± 0.000 | 0.030 ± 0.006 | 0.139 ± 0.053 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.206 ± 0.082 | 0.121 ± 0.003 | 0.196 ± 0.036 | 0.835 ± 0.012 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.488 ± 0.015 | 0.514 ± 0.028 | 0.669 ± 0.025 | 3.567 ± 0.118 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.081 ± 0.024 | 0.064 ± 0.001 | 0.072 ± 0.002 | 0.076 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.030 ± 0.006 | 0.019 ± 0.001 | 0.038 ± 0.007 | 0.036 ± 0.004 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.034 ± 0.050 | 0.027 ± 0.001 | 0.072 ± 0.044 | 0.056 ± 0.005 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.040 ± 0.019 | 0.032 ± 0.001 | 0.164 ± 0.110 | 0.119 ± 0.046 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.672 ± 0.093 | 0.411 ± 0.026 | 0.411 ± 0.009 | 0.451 ± 0.026 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.007 | 0.008 ± 0.004 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.002 | 0.012 ± 0.004 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.008 ± 0.003 | 0.008 ± 0.001 | 0.020 ± 0.011 | 0.016 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.001 | 0.005 ± 0.000 | 0.009 ± 0.002 | 0.103 ± 0.030 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.003 ± 0.004 | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.080 ± 0.108 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.006 ± 0.001 | 0.017 ± 0.004 | 0.105 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.113 ± 0.005 | 0.015 ± 0.001 | 0.065 ± 0.004 | 0.007 ± 0.004 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.102 ± 0.002 | 0.022 ± 0.003 | 0.070 ± 0.004 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.130 ± 0.010 | 0.029 ± 0.004 | 0.121 ± 0.043 | 0.016 ± 0.004 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.039 ± 0.003 | 0.016 ± 0.001 | 0.103 ± 0.022 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.034 ± 0.001 | 0.021 ± 0.002 | 0.087 ± 0.016 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.056 ± 0.033 | 0.025 ± 0.004 | 0.097 ± 0.016 | 0.017 ± 0.011 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.517 ± 0.145 | 0.018 ± 0.000 | 0.131 ± 0.016 | 0.010 ± 0.004 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.344 ± 0.007 | 0.021 ± 0.000 | 0.156 ± 0.019 | 0.008 ± 0.003 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.442 ± 0.115 | 0.019 ± 0.002 | 0.142 ± 0.018 | 0.015 ± 0.009 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.094 ± 0.008 | 0.017 ± 0.003 | 0.117 ± 0.010 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.077 ± 0.003 | 0.019 ± 0.000 | 0.109 ± 0.054 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.167 ± 0.073 | 0.023 ± 0.001 | 0.121 ± 0.015 | 0.019 ± 0.009 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.002 | 0.073 ± 0.071 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.008 ± 0.000 | 0.033 ± 0.002 | 0.016 ± 0.011 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.001 | 0.040 ± 0.005 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.602 ± 0.029 | 0.028 ± 0.002 | 0.096 ± 0.006 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.476 ± 0.024 | 0.033 ± 0.002 | 0.112 ± 0.026 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.513 ± 0.012 | 0.035 ± 0.003 | 0.120 ± 0.019 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.072 ± 0.003 | 0.028 ± 0.001 | 0.111 ± 0.028 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.069 ± 0.003 | 0.032 ± 0.005 | 0.134 ± 0.034 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.081 ± 0.003 | 0.041 ± 0.001 | 0.108 ± 0.005 | 0.022 ± 0.015 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.001 | 0.052 ± 0.061 | 0.011 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.002 | 0.054 ± 0.001 | 0.014 ± 0.009 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.005 | 0.047 ± 0.014 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.086 ± 0.006 | 0.014 ± 0.002 | 0.204 ± 0.028 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.069 ± 0.004 | 0.017 ± 0.003 | 0.232 ± 0.048 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.079 ± 0.006 | 0.018 ± 0.001 | 0.234 ± 0.061 | 0.010 ± 0.006 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.042 ± 0.002 | 0.014 ± 0.000 | 0.096 ± 0.007 | 0.009 ± 0.008 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.034 ± 0.002 | 0.015 ± 0.001 | 0.105 ± 0.014 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.043 ± 0.012 | 0.017 ± 0.002 | 0.109 ± 0.026 | 0.011 ± 0.006 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.011 ± 0.000 | 0.045 ± 0.051 | 0.013 ± 0.008 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.000 | 0.032 ± 0.011 | 0.014 ± 0.003 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.001 | 0.059 ± 0.011 | 0.022 ± 0.015 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.115 ± 0.005 | 0.012 ± 0.000 | 0.173 ± 0.102 | 0.012 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.097 ± 0.005 | 0.017 ± 0.000 | 0.136 ± 0.033 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.115 ± 0.005 | 0.031 ± 0.001 | 0.144 ± 0.036 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.034 ± 0.004 | 0.011 ± 0.000 | 0.104 ± 0.028 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.036 ± 0.002 | 0.017 ± 0.001 | 0.088 ± 0.013 | 0.014 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.041 ± 0.002 | 0.029 ± 0.003 | 0.114 ± 0.018 | 0.021 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.003 ± 0.001 | 0.001 ± 0.000 | 0.006 ± 0.003 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.001 | 0.010 ± 0.011 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.003 ± 0.001 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.004 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.008 ± 0.001 | 0.006 ± 0.001 | 0.007 ± 0.001 | 0.047 ± 0.022 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.002 ± 0.001 | 0.001 ± 0.000 | 0.007 ± 0.002 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.008 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.004 ± 0.003 | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.010 ± 0.002 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.003 | 0.011 ± 0.003 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.014 ± 0.002 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.015 ± 0.008 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.002 | 0.009 ± 0.010 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.004 | 0.009 ± 0.003 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.004 | 0.009 ± 0.003 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.004 | 0.007 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.005 ± 0.002 | 0.004 ± 0.000 | 0.011 ± 0.007 | 0.017 ± 0.008 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.008 ± 0.003 | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.003 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.013 ± 0.001 | 0.015 ± 0.004 | 0.025 ± 0.012 | 0.032 ± 0.007 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 31.6x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 25.2x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 27.3x faster than the slowest successful cell (`jax-cpu`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 22.3x faster than the slowest successful cell (`jax-cpu`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 19.3x faster than the slowest successful cell (`jax-cpu`, 0.183 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 52.2x faster than the slowest successful cell (`jax-cpu`, 0.183 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 45.3x faster than the slowest successful cell (`jax-cpu`, 0.183 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 15.7x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 36.4x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 21.2x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 31.6x faster than the slowest successful cell (`jax-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`jax-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 43.2x faster than the slowest successful cell (`jax-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 18.4x faster than the slowest successful cell (`jax-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.8x faster than the slowest successful cell (`tenferro-eager`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.8x faster than the slowest successful cell (`tenferro-eager`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 50.6x faster than the slowest successful cell (`tenferro-eager`, 0.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 28.7x faster than the slowest successful cell (`tenferro-eager`, 0.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 40.6x faster than the slowest successful cell (`tenferro-eager`, 0.344 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 16.5x faster than the slowest successful cell (`tenferro-eager`, 0.344 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 29.7x faster than the slowest successful cell (`tenferro-eager`, 0.442 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 23.0x faster than the slowest successful cell (`tenferro-eager`, 0.442 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 66.8x faster than the slowest successful cell (`tenferro-eager`, 0.602 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 21.4x faster than the slowest successful cell (`tenferro-eager`, 0.602 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 47.7x faster than the slowest successful cell (`tenferro-eager`, 0.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`tenferro-eager`, 0.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 59.7x faster than the slowest successful cell (`tenferro-eager`, 0.513 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`tenferro-eager`, 0.513 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.111 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 20.0x faster than the slowest successful cell (`pytorch-cpu`, 0.204 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.204 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 25.8x faster than the slowest successful cell (`pytorch-cpu`, 0.232 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.232 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 22.8x faster than the slowest successful cell (`pytorch-cpu`, 0.234 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 0.234 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261109_180327/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261109_180327/run.yaml`
- Timestamp: `2026-11-09T18:03:27.099105+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261109_180327`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261109_180327/cpu_ops_t4_20261109_180327.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.014 | 0.007 ± 0.000 | 0.008 ± 0.002 | 0.020 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.020 ± 0.001 | 0.030 ± 0.027 | 0.020 ± 0.002 | 0.047 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.023 ± 0.000 | 0.023 ± 0.001 | 0.023 ± 0.001 | 0.031 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.092 ± 0.008 | 0.094 ± 0.004 | 0.098 ± 0.027 | 0.113 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.004 | 0.004 ± 0.001 | 0.009 ± 0.002 | 0.067 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.008 ± 0.003 | 0.087 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.086 ± 0.046 | 0.064 ± 0.005 | 0.113 ± 0.026 | 0.183 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.088 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.002 | 0.081 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.002 | 0.008 ± 0.001 | 0.047 ± 0.002 | 0.012 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.018 ± 0.001 | 0.021 ± 0.002 | 0.080 ± 0.009 | 0.029 ± 0.010 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.013 ± 0.001 | 0.011 ± 0.000 | 0.041 ± 0.007 | 0.016 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.039 ± 0.010 | 0.037 ± 0.005 | 0.059 ± 0.009 | 0.053 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.003 | 0.007 ± 0.000 | 0.066 ± 0.016 | 0.008 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.012 ± 0.000 | 0.014 ± 0.001 | 0.088 ± 0.027 | 0.012 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.001 | 0.008 ± 0.000 | 0.039 ± 0.033 | 0.010 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.019 ± 0.007 | 0.017 ± 0.001 | 0.059 ± 0.008 | 0.024 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.026 ± 0.004 | 0.022 ± 0.010 | 0.015 ± 0.001 | 0.039 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.041 ± 0.002 | 0.039 ± 0.001 | 0.041 ± 0.002 | 0.122 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.054 ± 0.004 | 0.043 ± 0.001 | 0.052 ± 0.007 | 0.068 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.178 ± 0.022 | 0.164 ± 0.008 | 0.185 ± 0.046 | 0.244 ± 0.012 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.001 | 0.078 ± 0.025 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.001 | 0.087 ± 0.041 | 0.007 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.011 ± 0.000 | 0.067 ± 0.020 | 0.011 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.000 | 0.062 ± 0.013 | 0.051 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.037 ± 0.001 | 0.144 ± 0.027 | 0.012 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.058 ± 0.000 | 0.173 ± 0.035 | 0.017 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.040 ± 0.000 | 0.092 ± 0.005 | 0.009 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.072 ± 0.001 | 0.152 ± 0.046 | 0.033 ± 0.003 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.312 ± 0.018 | 0.314 ± 0.016 | 0.296 ± 0.041 | 0.350 ± 0.027 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 12.108 ± 1.404 | 10.162 ± 2.745 | 4.919 ± 0.054 | 5.297 ± 0.049 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 62.771 ± 57.417 | 63.180 ± 9.878 | 27.829 ± 2.735 | 28.352 ± 0.336 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 5.844 ± 0.792 | 10.545 ± 2.332 | 5.113 ± 0.178 | 5.354 ± 0.051 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 31.163 ± 1.770 | 66.406 ± 14.136 | 25.898 ± 2.617 | 28.525 ± 0.133 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 3.262 ± 0.212 | 1.388 ± 1.128 | 2.250 ± 0.479 | 1.476 ± 0.046 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 18.164 ± 5.191 | 11.494 ± 2.289 | 17.941 ± 12.105 | 8.962 ± 0.366 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 2.889 ± 0.262 | 1.590 ± 0.800 | 1.526 ± 0.091 | 1.423 ± 0.034 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 19.329 ± 0.867 | 14.992 ± 1.610 | 8.157 ± 2.508 | 8.918 ± 0.367 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.022 ± 0.003 | 0.009 ± 0.002 | 0.011 ± 0.002 | 0.033 ± 0.003 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.031 ± 0.001 | 0.053 ± 0.009 | 0.075 ± 0.005 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 7.466 ± 0.958 | 3.835 ± 0.142 | 3.617 ± 0.169 | 2.892 ± 0.075 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 32.844 ± 3.642 | 34.171 ± 9.727 | 20.780 ± 2.812 | 15.161 ± 0.278 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 5.115 ± 0.618 | 5.244 ± 4.545 | 3.530 ± 0.360 | 2.934 ± 0.124 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 24.535 ± 2.810 | 21.388 ± 9.973 | 23.658 ± 14.888 | 15.780 ± 0.433 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.116 ± 0.047 | 0.081 ± 0.019 | 0.043 ± 0.007 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.931 ± 0.227 | 0.578 ± 0.089 | 0.533 ± 0.031 | 0.352 ± 0.010 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 3.707 ± 1.198 | 4.306 ± 1.353 | 1.881 ± 0.716 | 1.712 ± 0.212 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.340 ± 0.056 | 0.717 ± 0.096 | 0.839 ± 0.102 | 0.344 ± 0.006 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 2.719 ± 0.707 | 3.906 ± 0.895 | 2.715 ± 0.186 | 1.440 ± 0.053 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.620 ± 0.034 | 0.495 ± 0.056 | 0.479 ± 0.016 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 26.071 ± 2.665 | 15.719 ± 1.646 | 9.641 ± 0.480 | 10.342 ± 0.202 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 116.366 ± 21.807 | 109.638 ± 13.460 | 54.710 ± 3.213 | 52.184 ± 2.700 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 14.732 ± 1.798 | 17.687 ± 1.083 | 10.574 ± 2.592 | 10.315 ± 0.230 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 70.834 ± 16.299 | 90.015 ± 6.905 | 50.786 ± 5.449 | 53.870 ± 1.717 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.024 ± 0.002 | 0.019 ± 0.000 | 0.031 ± 0.009 | 0.113 ± 0.034 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.153 ± 0.007 | 0.126 ± 0.007 | 0.210 ± 0.024 | 0.420 ± 0.034 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.466 ± 0.009 | 0.456 ± 0.004 | 0.685 ± 0.140 | 1.534 ± 0.217 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.068 ± 0.004 | 0.073 ± 0.005 | 0.108 ± 0.007 | 0.080 ± 0.004 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.025 ± 0.002 | 0.023 ± 0.001 | 0.054 ± 0.030 | 0.039 ± 0.010 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.025 ± 0.003 | 0.028 ± 0.000 | 0.052 ± 0.029 | 0.062 ± 0.003 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.035 ± 0.010 | 0.034 ± 0.001 | 0.117 ± 0.128 | 0.106 ± 0.011 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.406 ± 0.013 | 0.478 ± 0.057 | 0.393 ± 0.025 | 0.455 ± 0.014 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.008 ± 0.001 | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.009 ± 0.002 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.006 ± 0.002 | 0.009 ± 0.002 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.009 ± 0.001 | 0.011 ± 0.011 | 0.011 ± 0.004 | 0.012 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.004 ± 0.002 | 0.007 ± 0.005 | 0.011 ± 0.004 | 0.080 ± 0.029 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.009 ± 0.002 | 0.008 ± 0.001 | 0.085 ± 0.012 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.008 ± 0.002 | 0.009 ± 0.000 | 0.098 ± 0.007 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.134 ± 0.014 | 0.025 ± 0.003 | 0.082 ± 0.015 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.114 ± 0.006 | 0.026 ± 0.001 | 0.066 ± 0.004 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.133 ± 0.013 | 0.033 ± 0.001 | 0.093 ± 0.010 | 0.010 ± 0.007 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.043 ± 0.001 | 0.023 ± 0.001 | 0.129 ± 0.060 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.047 ± 0.008 | 0.034 ± 0.020 | 0.058 ± 0.002 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.056 ± 0.003 | 0.030 ± 0.004 | 0.076 ± 0.006 | 0.012 ± 0.006 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.417 ± 0.033 | 0.028 ± 0.001 | 0.402 ± 0.121 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.443 ± 0.054 | 0.029 ± 0.001 | 0.282 ± 0.011 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.385 ± 0.009 | 0.029 ± 0.006 | 0.309 ± 0.030 | 0.009 ± 0.008 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.134 ± 0.014 | 0.026 ± 0.003 | 0.281 ± 0.214 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.108 ± 0.008 | 0.025 ± 0.002 | 0.230 ± 0.040 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.119 ± 0.006 | 0.035 ± 0.003 | 0.259 ± 0.007 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.001 | 0.028 ± 0.004 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.035 ± 0.007 | 0.010 ± 0.005 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.001 | 0.038 ± 0.003 | 0.016 ± 0.013 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.562 ± 0.310 | 0.042 ± 0.002 | 0.150 ± 0.017 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.592 ± 0.303 | 0.043 ± 0.002 | 0.218 ± 0.019 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.525 ± 0.023 | 0.050 ± 0.039 | 0.193 ± 0.059 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.097 ± 0.036 | 0.043 ± 0.002 | 0.179 ± 0.029 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.102 ± 0.029 | 0.042 ± 0.002 | 0.226 ± 0.025 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.120 ± 0.010 | 0.048 ± 0.004 | 0.197 ± 0.029 | 0.018 ± 0.006 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.034 ± 0.006 | 0.037 ± 0.001 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.032 ± 0.001 | 0.053 ± 0.031 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.032 ± 0.000 | 0.048 ± 0.017 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.114 ± 0.045 | 0.022 ± 0.002 | 0.183 ± 0.129 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.087 ± 0.013 | 0.021 ± 0.001 | 0.176 ± 0.028 | 0.007 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.096 ± 0.022 | 0.028 ± 0.010 | 0.190 ± 0.025 | 0.007 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.063 ± 0.011 | 0.023 ± 0.007 | 0.095 ± 0.037 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.049 ± 0.005 | 0.021 ± 0.001 | 0.105 ± 0.023 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.045 ± 0.007 | 0.022 ± 0.001 | 0.087 ± 0.005 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.001 | 0.031 ± 0.002 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.003 | 0.045 ± 0.009 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.031 ± 0.001 | 0.042 ± 0.005 | 0.031 ± 0.032 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.178 ± 0.152 | 0.018 ± 0.001 | 0.093 ± 0.005 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.120 ± 0.017 | 0.021 ± 0.002 | 0.116 ± 0.022 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.125 ± 0.028 | 0.037 ± 0.003 | 0.134 ± 0.018 | 0.035 ± 0.026 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.069 ± 0.026 | 0.016 ± 0.002 | 0.064 ± 0.013 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.041 ± 0.001 | 0.022 ± 0.001 | 0.086 ± 0.021 | 0.013 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.051 ± 0.004 | 0.033 ± 0.005 | 0.092 ± 0.010 | 0.023 ± 0.006 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.003 ± 0.003 | 0.003 ± 0.000 | 0.001 ± 0.001 | 0.008 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.014 ± 0.013 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.004 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.009 ± 0.006 | 0.006 ± 0.002 | 0.007 ± 0.004 | 0.030 ± 0.031 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.001 | 0.005 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.005 ± 0.005 | 0.004 ± 0.000 | 0.031 ± 0.010 | 0.009 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.005 ± 0.001 | 0.005 ± 0.002 | 0.052 ± 0.032 | 0.007 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.006 ± 0.004 | 0.053 ± 0.008 | 0.009 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.006 ± 0.004 | 0.006 ± 0.000 | 0.008 ± 0.004 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.005 ± 0.006 | 0.006 ± 0.001 | 0.008 ± 0.001 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.004 ± 0.000 | 0.007 ± 0.002 | 0.010 ± 0.003 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.010 ± 0.002 | 0.006 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.008 ± 0.004 | 0.011 ± 0.008 | 0.006 ± 0.003 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.004 | 0.011 ± 0.001 | 0.007 ± 0.002 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.009 ± 0.005 | 0.005 ± 0.000 | 0.009 ± 0.003 | 0.015 ± 0.003 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.009 ± 0.004 | 0.013 ± 0.004 | 0.008 ± 0.001 | 0.015 ± 0.002 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.015 ± 0.002 | 0.019 ± 0.030 | 0.016 ± 0.002 | 0.020 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 19.1x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 18.8x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 23.7x faster than the slowest successful cell (`jax-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 23.7x faster than the slowest successful cell (`jax-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 21.8x faster than the slowest successful cell (`jax-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 21.3x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 30.1x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 21.3x faster than the slowest successful cell (`tenferro-eager`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.6x faster than the slowest successful cell (`tenferro-eager`, 0.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 13.8x faster than the slowest successful cell (`tenferro-eager`, 0.133 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.7x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 63.9x faster than the slowest successful cell (`tenferro-eager`, 0.417 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`tenferro-eager`, 0.417 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 62.4x faster than the slowest successful cell (`tenferro-eager`, 0.443 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`tenferro-eager`, 0.443 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 42.5x faster than the slowest successful cell (`tenferro-eager`, 0.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`tenferro-eager`, 0.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 42.0x faster than the slowest successful cell (`pytorch-cpu`, 0.281 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.281 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 31.0x faster than the slowest successful cell (`pytorch-cpu`, 0.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 45.9x faster than the slowest successful cell (`pytorch-cpu`, 0.259 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 67.9x faster than the slowest successful cell (`tenferro-eager`, 0.562 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`tenferro-eager`, 0.562 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 64.5x faster than the slowest successful cell (`tenferro-eager`, 0.592 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`tenferro-eager`, 0.592 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 103.6x faster than the slowest successful cell (`tenferro-eager`, 0.525 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`tenferro-eager`, 0.525 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 25.4x faster than the slowest successful cell (`pytorch-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 25.9x faster than the slowest successful cell (`pytorch-cpu`, 0.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 26.8x faster than the slowest successful cell (`pytorch-cpu`, 0.183 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 25.4x faster than the slowest successful cell (`pytorch-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 27.5x faster than the slowest successful cell (`pytorch-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 18.7x faster than the slowest successful cell (`tenferro-eager`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

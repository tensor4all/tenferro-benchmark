# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `4edf89377c13d5c8a287dd562b529de6bb85f4dc`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261112_172325/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261112_172325/run.yaml`
- Timestamp: `2026-11-12T17:23:25.841634+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261112_172325`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261112_172325/cpu_ops_t1_20261112_172325.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.005 | 0.016 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.018 ± 0.001 | 0.019 ± 0.006 | 0.021 ± 0.003 | 0.042 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.023 ± 0.001 | 0.022 ± 0.000 | 0.025 ± 0.002 | 0.037 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.080 ± 0.002 | 0.080 ± 0.032 | 0.091 ± 0.005 | 0.113 ± 0.030 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.001 | 0.079 ± 0.024 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.086 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.048 ± 0.002 | 0.046 ± 0.002 | 0.039 ± 0.009 | 0.108 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.002 | 0.092 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.010 ± 0.001 | 0.098 ± 0.017 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.006 | 0.013 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.002 | 0.027 ± 0.009 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | 0.016 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.035 ± 0.001 | 0.033 ± 0.004 | 0.042 ± 0.001 | 0.049 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.008 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.001 | 0.010 ± 0.010 | 0.015 ± 0.003 | 0.014 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.003 | 0.012 ± 0.001 | 0.011 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.014 ± 0.000 | 0.014 ± 0.003 | 0.021 ± 0.004 | 0.020 ± 0.034 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.002 | 0.042 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.039 ± 0.001 | 0.039 ± 0.000 | 0.040 ± 0.006 | 0.126 ± 0.044 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.044 ± 0.001 | 0.049 ± 0.002 | 0.043 ± 0.001 | 0.072 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.162 ± 0.004 | 0.205 ± 0.052 | 0.162 ± 0.021 | 0.276 ± 0.013 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.008 ± 0.000 | 0.061 ± 0.009 | 0.008 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.001 | 0.055 ± 0.005 | 0.011 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.003 | 0.060 ± 0.006 | 0.013 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.001 | 0.072 ± 0.011 | 0.030 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.008 | 0.044 ± 0.008 | 0.013 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.048 ± 0.013 | 0.051 ± 0.002 | 0.025 ± 0.014 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.032 ± 0.009 | 0.051 ± 0.007 | 0.009 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.061 ± 0.010 | 0.072 ± 0.029 | 0.027 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.330 ± 0.020 | 0.324 ± 0.044 | 0.318 ± 0.004 | 0.338 ± 0.054 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 11.230 ± 1.026 | 8.206 ± 0.287 | 5.907 ± 2.732 | 6.859 ± 0.076 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 55.807 ± 0.972 | 45.645 ± 1.465 | 31.191 ± 1.268 | 45.815 ± 12.957 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 5.877 ± 0.852 | 8.119 ± 0.434 | 5.705 ± 0.338 | 7.047 ± 3.590 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 29.589 ± 0.990 | 44.321 ± 0.531 | 29.741 ± 5.527 | 46.658 ± 11.449 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 2.984 ± 0.487 | 1.263 ± 0.210 | 2.328 ± 0.764 | 3.255 ± 1.594 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 16.492 ± 0.493 | 7.500 ± 0.309 | 18.449 ± 3.239 | 20.307 ± 2.299 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 2.620 ± 1.410 | 1.329 ± 0.074 | 1.886 ± 1.830 | 2.579 ± 4.864 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 14.974 ± 0.228 | 7.949 ± 0.191 | 8.076 ± 2.214 | 23.284 ± 25.756 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.017 ± 0.007 | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.029 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.001 | 0.071 ± 0.052 | 0.074 ± 0.017 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 5.666 ± 1.654 | 3.698 ± 0.041 | 3.362 ± 0.869 | 5.131 ± 0.505 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 29.483 ± 3.215 | 19.816 ± 0.776 | 17.616 ± 6.010 | 26.095 ± 2.536 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 3.586 ± 0.202 | 3.552 ± 0.064 | 3.514 ± 0.907 | 5.719 ± 2.358 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 21.859 ± 1.648 | 19.870 ± 0.619 | 16.347 ± 5.808 | 31.911 ± 3.592 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.074 ± 0.001 | 0.133 ± 0.042 | 0.037 ± 0.009 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 0.695 ± 0.143 | 0.582 ± 0.187 | 0.530 ± 0.113 | 0.351 ± 0.296 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 3.153 ± 0.164 | 2.474 ± 0.029 | 2.149 ± 2.251 | 2.893 ± 3.352 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.371 ± 0.012 | 0.481 ± 0.024 | 0.810 ± 0.393 | 0.315 ± 0.036 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 1.657 ± 0.112 | 2.386 ± 0.057 | 3.045 ± 0.074 | 2.634 ± 1.653 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.555 ± 0.008 | 0.645 ± 0.129 | 0.569 ± 0.080 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 19.888 ± 3.333 | 13.722 ± 0.739 | 11.896 ± 2.239 | 11.223 ± 0.786 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 121.425 ± 14.129 | 89.426 ± 3.868 | 62.705 ± 3.398 | 64.096 ± 9.845 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 9.556 ± 0.130 | 13.817 ± 0.220 | 10.800 ± 2.689 | 10.778 ± 0.674 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 58.620 ± 1.384 | 80.038 ± 2.170 | 58.338 ± 3.612 | 59.276 ± 5.632 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.027 ± 0.002 | 0.018 ± 0.001 | 0.032 ± 0.006 | 0.168 ± 0.140 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.175 ± 0.013 | 0.116 ± 0.004 | 0.198 ± 0.029 | 0.907 ± 0.095 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.524 ± 0.033 | 0.451 ± 0.024 | 0.747 ± 0.338 | 3.506 ± 6.344 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.071 ± 0.009 | 0.061 ± 0.004 | 0.074 ± 0.003 | 0.078 ± 0.005 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.023 ± 0.013 | 0.020 ± 0.001 | 0.035 ± 0.008 | 0.029 ± 0.010 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.022 ± 0.000 | 0.025 ± 0.000 | 0.042 ± 0.006 | 0.061 ± 0.008 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.027 ± 0.000 | 0.032 ± 0.000 | 0.061 ± 0.012 | 0.132 ± 0.164 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.510 ± 0.090 | 0.397 ± 0.007 | 0.425 ± 0.028 | 0.505 ± 0.060 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.003 ± 0.001 | 0.005 ± 0.001 | 0.004 ± 0.000 | 0.010 ± 0.002 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.004 ± 0.001 | 0.004 ± 0.002 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.008 ± 0.002 | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.018 ± 0.007 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.001 | 0.006 ± 0.001 | 0.008 ± 0.001 | 0.094 ± 0.020 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.003 ± 0.001 | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.091 ± 0.008 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.141 ± 0.079 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.148 ± 0.124 | 0.017 ± 0.001 | 0.088 ± 0.016 | 0.010 ± 0.004 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.140 ± 0.089 | 0.017 ± 0.000 | 0.070 ± 0.005 | 0.010 ± 0.006 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.122 ± 0.005 | 0.026 ± 0.002 | 0.079 ± 0.009 | 0.011 ± 0.005 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.034 ± 0.067 | 0.018 ± 0.001 | 0.079 ± 0.053 | 0.009 ± 0.005 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.035 ± 0.002 | 0.017 ± 0.000 | 0.065 ± 0.017 | 0.017 ± 0.009 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.044 ± 0.003 | 0.023 ± 0.001 | 0.120 ± 0.135 | 0.014 ± 0.003 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.744 ± 0.191 | 0.020 ± 0.001 | 0.122 ± 0.012 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.359 ± 0.017 | 0.018 ± 0.000 | 0.105 ± 0.027 | 0.010 ± 0.006 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.372 ± 0.050 | 0.018 ± 0.000 | 0.128 ± 0.103 | 0.007 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.080 ± 0.009 | 0.020 ± 0.003 | 0.087 ± 0.039 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.111 ± 0.046 | 0.016 ± 0.000 | 0.081 ± 0.003 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.087 ± 0.004 | 0.021 ± 0.000 | 0.101 ± 0.052 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.003 | 0.025 ± 0.001 | 0.013 ± 0.010 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.000 | 0.025 ± 0.003 | 0.018 ± 0.014 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.001 | 0.039 ± 0.009 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.877 ± 0.497 | 0.031 ± 0.005 | 0.080 ± 0.010 | 0.010 ± 0.005 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.490 ± 0.054 | 0.027 ± 0.001 | 0.081 ± 0.004 | 0.021 ± 0.006 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.529 ± 0.060 | 0.029 ± 0.008 | 0.098 ± 0.109 | 0.017 ± 0.014 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.160 ± 0.026 | 0.038 ± 0.031 | 0.098 ± 0.052 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.077 ± 0.036 | 0.027 ± 0.003 | 0.084 ± 0.042 | 0.014 ± 0.008 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.080 ± 0.040 | 0.036 ± 0.005 | 0.108 ± 0.033 | 0.011 ± 0.006 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.001 | 0.041 ± 0.015 | 0.013 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.000 | 0.043 ± 0.023 | 0.012 ± 0.013 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.024 | 0.042 ± 0.003 | 0.011 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.084 ± 0.059 | 0.015 ± 0.001 | 0.193 ± 0.076 | 0.010 ± 0.007 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.078 ± 0.018 | 0.014 ± 0.001 | 0.164 ± 0.017 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.073 ± 0.004 | 0.014 ± 0.000 | 0.235 ± 0.191 | 0.010 ± 0.007 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.041 ± 0.014 | 0.016 ± 0.001 | 0.154 ± 0.083 | 0.011 ± 0.003 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.040 ± 0.017 | 0.014 ± 0.002 | 0.078 ± 0.085 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.036 ± 0.002 | 0.014 ± 0.000 | 0.096 ± 0.159 | 0.012 ± 0.007 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.011 ± 0.001 | 0.029 ± 0.001 | 0.013 ± 0.002 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.036 ± 0.013 | 0.036 ± 0.028 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.001 | 0.044 ± 0.003 | 0.019 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.106 ± 0.129 | 0.019 ± 0.005 | 0.092 ± 0.007 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.184 ± 0.167 | 0.014 ± 0.000 | 0.095 ± 0.002 | 0.028 ± 0.024 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.126 ± 0.017 | 0.029 ± 0.001 | 0.157 ± 0.058 | 0.019 ± 0.005 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.043 ± 0.027 | 0.012 ± 0.008 | 0.077 ± 0.042 | 0.012 ± 0.004 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.041 ± 0.028 | 0.013 ± 0.001 | 0.067 ± 0.008 | 0.023 ± 0.009 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.044 ± 0.002 | 0.025 ± 0.001 | 0.111 ± 0.112 | 0.018 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.001 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.005 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.004 ± 0.003 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.003 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.004 ± 0.002 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.024 ± 0.005 | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.026 ± 0.014 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.008 ± 0.004 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.003 | 0.003 ± 0.001 | 0.011 ± 0.008 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.005 ± 0.001 | 0.003 ± 0.001 | 0.003 ± 0.001 | 0.009 ± 0.004 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.004 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.006 ± 0.004 | 0.007 ± 0.003 | 0.008 ± 0.001 | 0.010 ± 0.007 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.004 ± 0.003 | 0.004 ± 0.001 | 0.009 ± 0.004 | 0.007 ± 0.003 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.006 ± 0.003 | 0.004 ± 0.001 | 0.009 ± 0.005 | 0.009 ± 0.003 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.005 ± 0.002 | 0.004 ± 0.000 | 0.009 ± 0.002 | 0.009 ± 0.005 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.008 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.005 ± 0.005 | 0.004 ± 0.001 | 0.006 ± 0.001 | 0.015 ± 0.003 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.007 ± 0.001 | 0.006 ± 0.002 | 0.011 ± 0.003 | 0.016 ± 0.002 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.013 ± 0.001 | 0.013 ± 0.002 | 0.016 ± 0.002 | 0.025 ± 0.008 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 31.5x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 30.8x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 21.6x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 20.9x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 32.7x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 27.5x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `pytorch-cpu` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 19.4x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 19.6x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 34.5x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 16.5x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 26.5x faster than the slowest successful cell (`jax-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 16.3x faster than the slowest successful cell (`jax-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 57.9x faster than the slowest successful cell (`jax-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 28.4x faster than the slowest successful cell (`jax-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.1x faster than the slowest successful cell (`tenferro-eager`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.7x faster than the slowest successful cell (`tenferro-eager`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 11.1x faster than the slowest successful cell (`tenferro-eager`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 90.1x faster than the slowest successful cell (`tenferro-eager`, 0.744 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 36.9x faster than the slowest successful cell (`tenferro-eager`, 0.744 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 36.5x faster than the slowest successful cell (`tenferro-eager`, 0.359 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 20.2x faster than the slowest successful cell (`tenferro-eager`, 0.359 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 51.7x faster than the slowest successful cell (`tenferro-eager`, 0.372 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 20.1x faster than the slowest successful cell (`tenferro-eager`, 0.372 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.1x faster than the slowest successful cell (`tenferro-eager`, 0.111 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 90.8x faster than the slowest successful cell (`tenferro-eager`, 0.877 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`tenferro-eager`, 0.877 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 28.3x faster than the slowest successful cell (`tenferro-eager`, 0.877 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 23.2x faster than the slowest successful cell (`tenferro-eager`, 0.490 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 18.2x faster than the slowest successful cell (`tenferro-eager`, 0.490 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 32.0x faster than the slowest successful cell (`tenferro-eager`, 0.529 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 18.2x faster than the slowest successful cell (`tenferro-eager`, 0.529 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 18.0x faster than the slowest successful cell (`tenferro-eager`, 0.160 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 24.1x faster than the slowest successful cell (`pytorch-cpu`, 0.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`tenferro-eager`, 0.184 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261112_175944/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261112_175944/run.yaml`
- Timestamp: `2026-11-12T17:59:44.081935+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261112_175944`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261112_175944/cpu_ops_t4_20261112_175944.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.004 | 0.016 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.020 ± 0.001 | 0.021 ± 0.007 | 0.020 ± 0.002 | 0.067 ± 0.013 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.025 ± 0.011 | 0.030 ± 0.007 | 0.027 ± 0.001 | 0.033 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.095 ± 0.007 | 0.160 ± 0.111 | 0.095 ± 0.031 | 0.125 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.004 ± 0.001 | 0.003 ± 0.000 | 0.010 ± 0.003 | 0.084 ± 0.014 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.002 | 0.010 ± 0.003 | 0.008 ± 0.001 | 0.095 ± 0.022 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.055 ± 0.013 | 0.060 ± 0.011 | 0.078 ± 0.035 | 0.216 ± 0.022 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.009 ± 0.001 | 0.096 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.002 | 0.012 ± 0.006 | 0.079 ± 0.007 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.033 ± 0.008 | 0.011 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.017 ± 0.002 | 0.028 ± 0.010 | 0.055 ± 0.014 | 0.028 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.018 ± 0.005 | 0.044 ± 0.005 | 0.018 ± 0.009 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.038 ± 0.008 | 0.045 ± 0.054 | 0.097 ± 0.046 | 0.076 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.049 ± 0.007 | 0.010 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.021 ± 0.005 | 0.066 ± 0.024 | 0.034 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.005 | 0.013 ± 0.003 | 0.043 ± 0.010 | 0.011 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.018 ± 0.005 | 0.017 ± 0.003 | 0.062 ± 0.022 | 0.047 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.022 ± 0.009 | 0.013 ± 0.000 | 0.021 ± 0.011 | 0.042 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.042 ± 0.003 | 0.055 ± 0.013 | 0.048 ± 0.006 | 0.130 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.061 ± 0.030 | 0.071 ± 0.026 | 0.050 ± 0.008 | 0.073 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.180 ± 0.012 | 0.219 ± 0.067 | 0.328 ± 0.234 | 0.276 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.002 | 0.078 ± 0.027 | 0.008 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.008 | 0.059 ± 0.004 | 0.010 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.003 | 0.068 ± 0.017 | 0.030 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.034 ± 0.003 | 0.065 ± 0.003 | 0.069 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.082 ± 0.035 | 0.109 ± 0.026 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.097 ± 0.030 | 0.115 ± 0.067 | 0.045 ± 0.017 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.058 ± 0.065 | 0.101 ± 0.015 | 0.028 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.082 ± 0.033 | 0.109 ± 0.020 | 0.059 ± 0.007 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.305 ± 0.010 | 0.385 ± 0.112 | 0.306 ± 0.018 | 0.357 ± 0.013 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 10.513 ± 0.566 | 10.205 ± 1.500 | 6.207 ± 0.353 | 6.342 ± 0.123 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 65.440 ± 5.947 | 56.045 ± 6.331 | 33.850 ± 3.121 | 35.953 ± 0.825 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 5.574 ± 0.192 | 10.492 ± 0.760 | 5.132 ± 0.089 | 7.223 ± 1.269 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 36.763 ± 3.035 | 60.552 ± 4.967 | 31.693 ± 1.919 | 36.267 ± 1.953 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 3.378 ± 0.190 | 1.373 ± 0.503 | 3.404 ± 0.361 | 2.535 ± 0.078 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 19.338 ± 0.765 | 9.310 ± 1.691 | 17.700 ± 4.374 | 14.204 ± 0.291 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 2.479 ± 0.094 | 1.307 ± 0.090 | 1.798 ± 0.550 | 2.396 ± 0.704 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 18.017 ± 3.101 | 10.187 ± 2.140 | 8.434 ± 1.116 | 14.698 ± 8.774 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.015 ± 0.000 | 0.009 ± 0.001 | 0.012 ± 0.002 | 0.037 ± 0.003 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.001 | 0.059 ± 0.028 | 0.104 ± 0.016 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 5.837 ± 0.408 | 3.982 ± 1.147 | 3.475 ± 1.179 | 6.077 ± 4.039 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 28.479 ± 4.686 | 18.373 ± 0.471 | 14.859 ± 3.057 | 22.994 ± 1.447 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 3.391 ± 0.251 | 3.670 ± 0.569 | 6.614 ± 2.884 | 4.065 ± 0.170 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 24.332 ± 0.688 | 19.333 ± 4.811 | 18.480 ± 1.279 | 23.025 ± 0.658 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.100 ± 0.020 | 0.071 ± 0.009 | 0.066 ± 0.018 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.670 ± 0.102 | 0.527 ± 0.011 | 0.565 ± 0.119 | 0.347 ± 0.005 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 3.776 ± 0.352 | 3.312 ± 0.660 | 2.079 ± 0.697 | 3.640 ± 0.797 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.377 ± 0.019 | 0.553 ± 0.079 | 0.663 ± 0.088 | 0.391 ± 0.028 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 2.104 ± 0.127 | 3.653 ± 0.403 | 2.874 ± 0.530 | 1.701 ± 0.012 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.956 ± 0.193 | 0.578 ± 0.081 | 0.515 ± 0.012 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 18.409 ± 1.342 | 15.046 ± 3.474 | 12.862 ± 4.638 | 13.110 ± 4.461 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 102.358 ± 2.581 | 98.124 ± 4.660 | 59.568 ± 18.021 | 62.838 ± 2.814 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 10.552 ± 2.555 | 13.269 ± 4.305 | 12.273 ± 3.700 | 14.328 ± 4.621 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 60.920 ± 13.917 | 110.261 ± 7.983 | 54.991 ± 2.645 | 63.133 ± 2.049 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.028 ± 0.003 | 0.020 ± 0.000 | 0.024 ± 0.003 | 0.186 ± 0.041 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.164 ± 0.018 | 0.126 ± 0.002 | 0.183 ± 0.023 | 0.924 ± 0.352 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.499 ± 0.019 | 0.476 ± 0.107 | 0.698 ± 1.176 | 2.726 ± 1.443 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.069 ± 0.003 | 0.078 ± 0.012 | 0.105 ± 0.007 | 0.129 ± 0.095 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.020 ± 0.001 | 0.025 ± 0.016 | 0.035 ± 0.002 | 0.077 ± 0.016 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.021 ± 0.000 | 0.028 ± 0.001 | 0.043 ± 0.013 | 0.122 ± 0.033 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.028 ± 0.001 | 0.035 ± 0.000 | 0.049 ± 0.019 | 0.246 ± 0.064 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.408 ± 0.041 | 0.430 ± 0.006 | 0.390 ± 0.019 | 0.466 ± 0.014 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.007 ± 0.002 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.006 ± 0.006 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.001 | 0.006 ± 0.000 | 0.014 ± 0.009 | 0.093 ± 0.019 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.001 | 0.012 ± 0.004 | 0.008 ± 0.001 | 0.066 ± 0.014 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.003 ± 0.001 | 0.007 ± 0.002 | 0.009 ± 0.004 | 0.084 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.145 ± 0.076 | 0.039 ± 0.013 | 0.074 ± 0.007 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.129 ± 0.065 | 0.031 ± 0.006 | 0.064 ± 0.004 | 0.016 ± 0.007 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.139 ± 0.007 | 0.035 ± 0.000 | 0.092 ± 0.021 | 0.022 ± 0.008 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.042 ± 0.001 | 0.033 ± 0.011 | 0.058 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.047 ± 0.023 | 0.040 ± 0.005 | 0.057 ± 0.005 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.062 ± 0.022 | 0.030 ± 0.001 | 0.066 ± 0.002 | 0.026 ± 0.010 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.378 ± 0.006 | 0.031 ± 0.011 | 0.293 ± 0.018 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.515 ± 0.149 | 0.033 ± 0.009 | 0.271 ± 0.028 | 0.010 ± 0.004 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.387 ± 0.009 | 0.026 ± 0.002 | 0.264 ± 0.037 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.108 ± 0.024 | 0.037 ± 0.016 | 0.220 ± 0.011 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.109 ± 0.009 | 0.027 ± 0.012 | 0.226 ± 0.045 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.118 ± 0.002 | 0.027 ± 0.002 | 0.221 ± 0.025 | 0.013 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.000 | 0.025 ± 0.009 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.027 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.011 ± 0.000 | 0.027 ± 0.001 | 0.009 ± 0.006 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.553 ± 0.044 | 0.043 ± 0.004 | 0.158 ± 0.016 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.685 ± 0.311 | 0.052 ± 0.021 | 0.222 ± 0.068 | 0.016 ± 0.010 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.522 ± 0.047 | 0.043 ± 0.002 | 0.171 ± 0.022 | 0.014 ± 0.003 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.100 ± 0.113 | 0.046 ± 0.036 | 0.169 ± 0.020 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.106 ± 0.012 | 0.060 ± 0.010 | 0.203 ± 0.024 | 0.014 ± 0.023 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.116 ± 0.036 | 0.050 ± 0.001 | 0.202 ± 0.031 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.001 | 0.046 ± 0.014 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.034 ± 0.011 | 0.036 ± 0.002 | 0.013 ± 0.012 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.033 ± 0.003 | 0.043 ± 0.002 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.078 ± 0.003 | 0.029 ± 0.002 | 0.151 ± 0.006 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.082 ± 0.001 | 0.028 ± 0.011 | 0.144 ± 0.001 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.104 ± 0.083 | 0.023 ± 0.001 | 0.149 ± 0.006 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.044 ± 0.009 | 0.034 ± 0.007 | 0.076 ± 0.003 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.046 ± 0.002 | 0.023 ± 0.000 | 0.081 ± 0.004 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.050 ± 0.002 | 0.025 ± 0.001 | 0.072 ± 0.002 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.001 | 0.032 ± 0.002 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.016 | 0.033 ± 0.006 | 0.014 ± 0.008 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.031 ± 0.007 | 0.044 ± 0.004 | 0.046 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.091 ± 0.020 | 0.016 ± 0.000 | 0.114 ± 0.028 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.093 ± 0.003 | 0.022 ± 0.003 | 0.102 ± 0.016 | 0.020 ± 0.016 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.123 ± 0.003 | 0.037 ± 0.003 | 0.117 ± 0.008 | 0.045 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.038 ± 0.001 | 0.016 ± 0.001 | 0.067 ± 0.007 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.053 ± 0.009 | 0.022 ± 0.003 | 0.060 ± 0.001 | 0.014 ± 0.007 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.080 ± 0.037 | 0.032 ± 0.002 | 0.077 ± 0.002 | 0.046 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.003 ± 0.001 | 0.004 ± 0.000 | 0.002 ± 0.001 | 0.007 ± 0.008 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.007 ± 0.002 | 0.004 ± 0.002 | 0.002 ± 0.001 | 0.014 ± 0.004 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.014 ± 0.003 | 0.010 ± 0.001 | 0.007 ± 0.001 | 0.058 ± 0.008 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.001 | 0.008 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.006 ± 0.001 | 0.004 ± 0.000 | 0.029 ± 0.010 | 0.007 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.006 ± 0.003 | 0.031 ± 0.020 | 0.007 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.005 ± 0.005 | 0.028 ± 0.004 | 0.009 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.004 ± 0.001 | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.004 ± 0.000 | 0.009 ± 0.003 | 0.009 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.004 ± 0.000 | 0.008 ± 0.002 | 0.008 ± 0.001 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.005 ± 0.001 | 0.007 ± 0.001 | 0.009 ± 0.000 | 0.007 ± 0.003 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.008 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.006 | 0.013 ± 0.003 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.007 ± 0.002 | 0.007 ± 0.001 | 0.008 ± 0.001 | 0.017 ± 0.004 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.016 ± 0.002 | 0.018 ± 0.002 | 0.017 ± 0.004 | 0.021 ± 0.002 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 21.9x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 25.8x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 17.5x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 26.9x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 23.9x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 30.7x faster than the slowest successful cell (`jax-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`jax-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 24.3x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 26.2x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 20.8x faster than the slowest successful cell (`tenferro-eager`, 0.145 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 52.8x faster than the slowest successful cell (`tenferro-eager`, 0.378 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`tenferro-eager`, 0.378 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 49.1x faster than the slowest successful cell (`tenferro-eager`, 0.515 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 0.515 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 24.8x faster than the slowest successful cell (`tenferro-eager`, 0.387 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`tenferro-eager`, 0.387 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 32.1x faster than the slowest successful cell (`pytorch-cpu`, 0.220 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 27.2x faster than the slowest successful cell (`pytorch-cpu`, 0.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.221 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 74.0x faster than the slowest successful cell (`tenferro-eager`, 0.553 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`tenferro-eager`, 0.553 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 43.2x faster than the slowest successful cell (`tenferro-eager`, 0.685 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`tenferro-eager`, 0.685 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 36.9x faster than the slowest successful cell (`tenferro-eager`, 0.522 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 0.522 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 23.0x faster than the slowest successful cell (`pytorch-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.203 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 21.6x faster than the slowest successful cell (`pytorch-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 20.9x faster than the slowest successful cell (`pytorch-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 20.3x faster than the slowest successful cell (`pytorch-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

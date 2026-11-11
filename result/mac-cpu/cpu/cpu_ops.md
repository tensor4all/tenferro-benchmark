# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `35979dcd7110a29d0728b2e128c72914f4c3f67b`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261111_172341/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261111_172341/run.yaml`
- Timestamp: `2026-11-11T17:23:41.760374+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261111_172341`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261111_172341/cpu_ops_t1_20261111_172341.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.015 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.020 ± 0.001 | 0.017 ± 0.000 | 0.019 ± 0.000 | 0.043 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.021 ± 0.000 | 0.022 ± 0.000 | 0.024 ± 0.001 | 0.031 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.086 ± 0.016 | 0.075 ± 0.000 | 0.081 ± 0.002 | 0.109 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.072 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.084 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.046 ± 0.000 | 0.051 ± 0.001 | 0.035 ± 0.003 | 0.108 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.001 | 0.085 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.103 ± 0.007 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.016 ± 0.001 | 0.015 ± 0.000 | 0.017 ± 0.002 | 0.025 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.001 | 0.017 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.036 ± 0.002 | 0.032 ± 0.000 | 0.038 ± 0.001 | 0.048 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.001 | 0.009 ± 0.000 | 0.015 ± 0.002 | 0.013 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.011 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.002 | 0.012 ± 0.001 | 0.019 ± 0.003 | 0.021 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.001 | 0.041 ± 0.015 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.038 ± 0.002 | 0.038 ± 0.001 | 0.041 ± 0.000 | 0.121 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.041 ± 0.001 | 0.042 ± 0.001 | 0.049 ± 0.002 | 0.069 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.177 ± 0.017 | 0.153 ± 0.003 | 0.162 ± 0.002 | 0.280 ± 0.012 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.008 ± 0.000 | 0.053 ± 0.004 | 0.008 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.056 ± 0.000 | 0.008 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.000 | 0.052 ± 0.001 | 0.009 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.061 ± 0.005 | 0.029 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.028 ± 0.002 | 0.045 ± 0.001 | 0.012 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.042 ± 0.000 | 0.054 ± 0.003 | 0.020 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.043 ± 0.000 | 0.010 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.055 ± 0.000 | 0.062 ± 0.002 | 0.026 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.468 ± 0.147 | 0.283 ± 0.004 | 0.324 ± 0.016 | 0.335 ± 0.007 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 12.605 ± 1.991 | 7.298 ± 0.057 | 5.009 ± 0.391 | 6.885 ± 0.419 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 54.256 ± 1.426 | 48.794 ± 0.942 | 29.343 ± 0.284 | 37.186 ± 0.996 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 5.193 ± 0.781 | 7.238 ± 0.070 | 4.801 ± 0.256 | 6.575 ± 0.112 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 28.028 ± 0.284 | 48.591 ± 1.766 | 26.929 ± 0.494 | 36.268 ± 1.584 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 3.021 ± 0.456 | 1.054 ± 0.005 | 1.981 ± 0.084 | 2.826 ± 0.173 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 15.953 ± 0.594 | 8.120 ± 0.737 | 14.644 ± 0.100 | 19.342 ± 1.500 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 3.035 ± 0.442 | 1.347 ± 0.020 | 1.216 ± 0.033 | 3.019 ± 0.389 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 13.858 ± 1.084 | 8.437 ± 0.047 | 8.168 ± 0.073 | 19.829 ± 1.196 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.027 ± 0.014 | 0.007 ± 0.000 | 0.010 ± 0.004 | 0.029 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.000 | 0.044 ± 0.005 | 0.066 ± 0.003 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 5.588 ± 0.268 | 3.207 ± 0.013 | 2.567 ± 0.208 | 4.696 ± 0.261 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 37.092 ± 4.434 | 20.804 ± 0.750 | 14.698 ± 0.340 | 28.258 ± 0.738 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 3.741 ± 0.215 | 3.226 ± 0.015 | 2.386 ± 0.037 | 4.344 ± 0.150 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 32.110 ± 4.344 | 20.243 ± 0.432 | 13.961 ± 0.150 | 28.742 ± 0.272 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.072 ± 0.000 | 0.072 ± 0.003 | 0.035 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 0.673 ± 0.089 | 0.523 ± 0.070 | 0.441 ± 0.037 | 0.430 ± 0.152 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 2.879 ± 0.141 | 2.721 ± 0.175 | 1.763 ± 0.146 | 1.892 ± 0.059 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.386 ± 0.084 | 0.539 ± 0.013 | 0.537 ± 0.011 | 0.425 ± 0.087 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 1.475 ± 0.046 | 2.601 ± 0.035 | 2.821 ± 0.098 | 1.802 ± 0.147 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.525 ± 0.007 | 0.471 ± 0.087 | 0.497 ± 0.018 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 29.401 ± 1.474 | 13.031 ± 0.052 | 9.527 ± 0.132 | 10.872 ± 0.175 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 120.055 ± 6.667 | 88.228 ± 3.853 | 50.846 ± 1.213 | 64.457 ± 12.811 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 12.883 ± 1.562 | 13.071 ± 0.069 | 8.697 ± 0.215 | 11.029 ± 0.310 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 71.922 ± 12.036 | 80.685 ± 0.917 | 45.994 ± 0.196 | 61.595 ± 1.034 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.023 ± 0.002 | 0.018 ± 0.000 | 0.025 ± 0.002 | 0.122 ± 0.012 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.164 ± 0.018 | 0.122 ± 0.008 | 0.166 ± 0.023 | 0.868 ± 0.048 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.482 ± 0.014 | 0.459 ± 0.026 | 0.640 ± 0.009 | 4.067 ± 1.197 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.140 ± 0.017 | 0.062 ± 0.002 | 0.068 ± 0.008 | 0.078 ± 0.011 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.023 ± 0.003 | 0.019 ± 0.000 | 0.034 ± 0.001 | 0.028 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.052 ± 0.002 | 0.024 ± 0.001 | 0.036 ± 0.003 | 0.048 ± 0.011 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.090 ± 0.024 | 0.030 ± 0.001 | 0.049 ± 0.007 | 0.108 ± 0.013 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.538 ± 0.035 | 0.389 ± 0.005 | 0.381 ± 0.003 | 0.513 ± 0.064 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.007 ± 0.000 | 0.007 ± 0.004 | 0.009 ± 0.002 | 0.023 ± 0.010 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.003 | 0.067 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.005 ± 0.001 | 0.007 ± 0.000 | 0.086 ± 0.015 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.095 ± 0.021 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.096 ± 0.001 | 0.015 ± 0.001 | 0.061 ± 0.002 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.095 ± 0.001 | 0.017 ± 0.002 | 0.075 ± 0.004 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.167 ± 0.024 | 0.025 ± 0.000 | 0.076 ± 0.004 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.031 ± 0.001 | 0.016 ± 0.000 | 0.053 ± 0.001 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.033 ± 0.003 | 0.018 ± 0.001 | 0.066 ± 0.007 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.044 ± 0.017 | 0.022 ± 0.000 | 0.066 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.335 ± 0.014 | 0.018 ± 0.001 | 0.095 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.339 ± 0.004 | 0.018 ± 0.001 | 0.112 ± 0.016 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.388 ± 0.017 | 0.018 ± 0.000 | 0.098 ± 0.003 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.075 ± 0.001 | 0.017 ± 0.001 | 0.077 ± 0.004 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.072 ± 0.004 | 0.017 ± 0.000 | 0.091 ± 0.031 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.091 ± 0.017 | 0.020 ± 0.001 | 0.081 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.000 | 0.027 ± 0.004 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.006 ± 0.000 | 0.022 ± 0.000 | 0.011 ± 0.003 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.008 ± 0.001 | 0.027 ± 0.001 | 0.007 ± 0.003 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.466 ± 0.014 | 0.027 ± 0.000 | 0.071 ± 0.003 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.435 ± 0.002 | 0.025 ± 0.000 | 0.084 ± 0.005 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.480 ± 0.038 | 0.029 ± 0.000 | 0.092 ± 0.029 | 0.012 ± 0.016 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.066 ± 0.001 | 0.027 ± 0.000 | 0.070 ± 0.004 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.063 ± 0.000 | 0.025 ± 0.000 | 0.080 ± 0.007 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.091 ± 0.018 | 0.032 ± 0.000 | 0.083 ± 0.012 | 0.018 ± 0.008 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.003 | 0.052 ± 0.018 | 0.012 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.039 ± 0.003 | 0.011 ± 0.005 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.000 | 0.037 ± 0.001 | 0.011 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.069 ± 0.001 | 0.014 ± 0.000 | 0.146 ± 0.027 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.065 ± 0.005 | 0.014 ± 0.001 | 0.169 ± 0.040 | 0.011 ± 0.004 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.076 ± 0.022 | 0.014 ± 0.000 | 0.144 ± 0.006 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.033 ± 0.001 | 0.014 ± 0.001 | 0.071 ± 0.002 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.031 ± 0.001 | 0.014 ± 0.000 | 0.085 ± 0.032 | 0.011 ± 0.012 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.062 ± 0.011 | 0.014 ± 0.000 | 0.129 ± 0.006 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.011 ± 0.001 | 0.043 ± 0.008 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.001 | 0.030 ± 0.000 | 0.016 ± 0.005 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.001 | 0.038 ± 0.000 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.085 ± 0.007 | 0.011 ± 0.001 | 0.124 ± 0.025 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.084 ± 0.000 | 0.014 ± 0.000 | 0.104 ± 0.004 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.107 ± 0.002 | 0.027 ± 0.000 | 0.102 ± 0.019 | 0.020 ± 0.011 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.031 ± 0.001 | 0.011 ± 0.001 | 0.058 ± 0.002 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.031 ± 0.001 | 0.013 ± 0.000 | 0.073 ± 0.006 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.040 ± 0.002 | 0.024 ± 0.000 | 0.080 ± 0.012 | 0.020 ± 0.003 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.008 ± 0.001 | 0.006 ± 0.001 | 0.007 ± 0.001 | 0.025 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.002 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.004 ± 0.000 | 0.012 ± 0.004 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.003 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.002 | 0.007 ± 0.007 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.007 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.002 | 0.012 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.001 | 0.007 ± 0.000 | 0.016 ± 0.002 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.012 ± 0.000 | 0.013 ± 0.002 | 0.015 ± 0.002 | 0.021 ± 0.002 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 30.5x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 26.5x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 22.3x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 20.0x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 32.7x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 26.9x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 19.9x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 21.6x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 30.0x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 39.6x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 18.4x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 42.0x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 20.3x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.3x faster than the slowest successful cell (`tenferro-eager`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.2x faster than the slowest successful cell (`tenferro-eager`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 17.0x faster than the slowest successful cell (`tenferro-eager`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 48.8x faster than the slowest successful cell (`tenferro-eager`, 0.335 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 18.5x faster than the slowest successful cell (`tenferro-eager`, 0.335 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 39.1x faster than the slowest successful cell (`tenferro-eager`, 0.339 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 18.8x faster than the slowest successful cell (`tenferro-eager`, 0.339 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 72.3x faster than the slowest successful cell (`tenferro-eager`, 0.388 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 21.4x faster than the slowest successful cell (`tenferro-eager`, 0.388 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 13.3x faster than the slowest successful cell (`tenferro-eager`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 64.4x faster than the slowest successful cell (`tenferro-eager`, 0.466 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 17.2x faster than the slowest successful cell (`tenferro-eager`, 0.466 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 49.5x faster than the slowest successful cell (`tenferro-eager`, 0.435 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`tenferro-eager`, 0.435 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 38.5x faster than the slowest successful cell (`tenferro-eager`, 0.480 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 16.6x faster than the slowest successful cell (`tenferro-eager`, 0.480 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 21.9x faster than the slowest successful cell (`pytorch-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 17.2x faster than the slowest successful cell (`pytorch-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261111_175728/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261111_175728/run.yaml`
- Timestamp: `2026-11-11T17:57:28.429255+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261111_175728`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261111_175728/cpu_ops_t4_20261111_175728.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.002 | 0.015 ± 0.006 | 0.009 ± 0.001 | 0.023 ± 0.026 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.026 ± 0.005 | 0.020 ± 0.002 | 0.022 ± 0.003 | 0.099 ± 0.012 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.026 ± 0.001 | 0.026 ± 0.002 | 0.027 ± 0.003 | 0.033 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.089 ± 0.014 | 0.087 ± 0.003 | 0.097 ± 0.055 | 0.126 ± 0.099 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.005 ± 0.002 | 0.008 ± 0.001 | 0.219 ± 0.138 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.006 ± 0.001 | 0.008 ± 0.001 | 0.008 ± 0.001 | 0.100 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.055 ± 0.007 | 0.059 ± 0.006 | 0.087 ± 0.013 | 0.307 ± 0.126 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.009 ± 0.002 | 0.159 ± 0.181 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.009 ± 0.001 | 0.008 ± 0.003 | 0.012 ± 0.005 | 0.103 ± 0.018 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.010 ± 0.012 | 0.040 ± 0.007 | 0.013 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.019 ± 0.002 | 0.028 ± 0.013 | 0.056 ± 0.015 | 0.046 ± 0.030 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.014 ± 0.001 | 0.012 ± 0.001 | 0.042 ± 0.001 | 0.017 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.045 ± 0.002 | 0.038 ± 0.035 | 0.103 ± 0.031 | 0.095 ± 0.016 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.001 | 0.010 ± 0.001 | 0.054 ± 0.005 | 0.010 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.011 ± 0.002 | 0.015 ± 0.004 | 0.050 ± 0.011 | 0.021 ± 0.015 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.001 | 0.009 ± 0.001 | 0.045 ± 0.002 | 0.012 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.019 ± 0.001 | 0.020 ± 0.015 | 0.079 ± 0.042 | 0.026 ± 0.016 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.017 ± 0.005 | 0.015 ± 0.002 | 0.050 ± 0.019 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.041 ± 0.008 | 0.054 ± 0.002 | 0.054 ± 0.008 | 0.160 ± 0.113 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.057 ± 0.007 | 0.049 ± 0.007 | 0.049 ± 0.007 | 0.082 ± 0.040 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.207 ± 0.021 | 0.179 ± 0.007 | 0.206 ± 0.051 | 0.323 ± 0.040 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.002 | 0.069 ± 0.017 | 0.008 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.008 | 0.062 ± 0.009 | 0.012 ± 0.011 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.002 | 0.061 ± 0.013 | 0.017 ± 0.016 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.009 | 0.078 ± 0.012 | 0.056 ± 0.025 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.059 ± 0.014 | 0.106 ± 0.026 | 0.025 ± 0.015 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.085 ± 0.024 | 0.174 ± 0.084 | 0.036 ± 0.023 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.054 ± 0.003 | 0.091 ± 0.015 | 0.015 ± 0.018 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.100 ± 0.037 | 0.158 ± 0.082 | 0.066 ± 0.040 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.326 ± 0.013 | 0.305 ± 0.013 | 0.390 ± 0.093 | 0.348 ± 0.045 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 10.217 ± 0.371 | 8.204 ± 2.493 | 5.917 ± 1.227 | 8.263 ± 0.898 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 69.528 ± 10.145 | 66.952 ± 12.116 | 37.774 ± 8.718 | 42.727 ± 8.784 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 5.241 ± 0.046 | 7.589 ± 0.495 | 5.446 ± 1.033 | 9.776 ± 4.289 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 29.725 ± 2.192 | 60.806 ± 6.247 | 33.270 ± 2.450 | 38.838 ± 2.494 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 2.871 ± 0.156 | 1.054 ± 0.003 | 2.878 ± 0.318 | 2.463 ± 1.128 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 20.343 ± 2.073 | 12.201 ± 3.666 | 17.283 ± 1.672 | 14.711 ± 3.995 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 2.415 ± 0.211 | 1.155 ± 0.069 | 1.774 ± 0.611 | 2.580 ± 0.960 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 16.107 ± 2.422 | 9.671 ± 0.230 | 9.851 ± 1.812 | 20.153 ± 4.618 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.016 ± 0.001 | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.037 ± 0.004 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.028 ± 0.000 | 0.048 ± 0.009 | 0.083 ± 0.019 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 5.933 ± 0.806 | 3.901 ± 0.636 | 3.483 ± 1.165 | 4.495 ± 1.468 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 56.756 ± 12.536 | 37.245 ± 13.258 | 14.488 ± 1.757 | 26.365 ± 14.969 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 3.515 ± 0.040 | 3.290 ± 0.064 | 3.049 ± 1.182 | 4.292 ± 1.619 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 27.411 ± 0.925 | 35.892 ± 2.218 | 17.554 ± 1.343 | 32.778 ± 6.916 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.089 ± 0.004 | 0.075 ± 0.005 | 0.067 ± 0.009 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 1.198 ± 0.571 | 0.455 ± 0.009 | 0.637 ± 0.145 | 0.340 ± 0.016 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 4.129 ± 0.586 | 3.944 ± 0.531 | 2.256 ± 0.648 | 2.431 ± 0.783 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.420 ± 0.034 | 0.466 ± 0.015 | 0.825 ± 0.157 | 0.848 ± 0.315 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 2.978 ± 1.708 | 4.105 ± 0.958 | 3.355 ± 0.909 | 2.975 ± 3.034 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.652 ± 0.064 | 0.530 ± 0.132 | 0.532 ± 0.072 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 18.681 ± 1.645 | 13.554 ± 1.221 | 11.135 ± 2.361 | 11.450 ± 0.610 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 117.618 ± 14.527 | 86.610 ± 7.660 | 67.920 ± 8.212 | 71.221 ± 6.581 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 9.493 ± 0.668 | 13.234 ± 0.062 | 9.794 ± 1.560 | 11.110 ± 0.413 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 61.883 ± 3.275 | 95.785 ± 29.428 | 55.392 ± 5.740 | 70.783 ± 8.852 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.034 ± 0.009 | 0.019 ± 0.000 | 0.028 ± 0.014 | 0.208 ± 0.069 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.181 ± 0.030 | 0.120 ± 0.001 | 0.193 ± 0.012 | 0.562 ± 0.196 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.551 ± 0.084 | 0.469 ± 0.011 | 0.670 ± 0.271 | 2.396 ± 2.826 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.070 ± 0.008 | 0.068 ± 0.013 | 0.160 ± 0.077 | 0.113 ± 0.021 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.021 ± 0.001 | 0.023 ± 0.000 | 0.036 ± 0.025 | 0.049 ± 0.016 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.023 ± 0.000 | 0.027 ± 0.000 | 0.060 ± 0.031 | 0.071 ± 0.055 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.028 ± 0.000 | 0.034 ± 0.001 | 0.122 ± 0.039 | 0.304 ± 0.233 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.398 ± 0.067 | 0.399 ± 0.006 | 0.473 ± 0.039 | 0.922 ± 0.459 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.004 ± 0.001 | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.008 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.002 | 0.096 ± 0.024 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.001 | 0.005 ± 0.000 | 0.009 ± 0.002 | 0.083 ± 0.013 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.102 ± 0.005 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.126 ± 0.006 | 0.023 ± 0.003 | 0.071 ± 0.004 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.159 ± 0.020 | 0.025 ± 0.000 | 0.106 ± 0.029 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.143 ± 0.010 | 0.035 ± 0.001 | 0.086 ± 0.017 | 0.023 ± 0.007 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.047 ± 0.002 | 0.023 ± 0.003 | 0.062 ± 0.004 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.055 ± 0.008 | 0.027 ± 0.005 | 0.101 ± 0.038 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.061 ± 0.005 | 0.030 ± 0.001 | 0.083 ± 0.011 | 0.023 ± 0.004 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.450 ± 0.019 | 0.027 ± 0.002 | 0.328 ± 0.076 | 0.012 ± 0.003 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.472 ± 0.042 | 0.027 ± 0.000 | 0.200 ± 0.098 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.463 ± 0.056 | 0.027 ± 0.000 | 0.326 ± 0.038 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.110 ± 0.002 | 0.023 ± 0.000 | 0.233 ± 0.074 | 0.010 ± 0.004 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.125 ± 0.009 | 0.032 ± 0.002 | 0.253 ± 0.273 | 0.008 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.125 ± 0.029 | 0.031 ± 0.001 | 0.293 ± 0.059 | 0.017 ± 0.007 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.001 | 0.025 ± 0.001 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.000 | 0.032 ± 0.005 | 0.010 ± 0.005 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.004 | 0.030 ± 0.002 | 0.009 ± 0.005 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.564 ± 0.020 | 0.039 ± 0.001 | 0.167 ± 0.022 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.633 ± 0.021 | 0.041 ± 0.002 | 0.182 ± 0.160 | 0.012 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.615 ± 0.033 | 0.051 ± 0.011 | 0.211 ± 0.106 | 0.014 ± 0.004 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.102 ± 0.014 | 0.041 ± 0.004 | 0.168 ± 0.024 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.128 ± 0.005 | 0.040 ± 0.003 | 0.242 ± 0.070 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.119 ± 0.011 | 0.054 ± 0.004 | 0.180 ± 0.019 | 0.017 ± 0.015 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.032 ± 0.001 | 0.039 ± 0.002 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.032 ± 0.003 | 0.078 ± 0.008 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.032 ± 0.002 | 0.046 ± 0.005 | 0.012 ± 0.012 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.084 ± 0.006 | 0.022 ± 0.003 | 0.174 ± 0.029 | 0.011 ± 0.020 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.095 ± 0.006 | 0.024 ± 0.004 | 0.253 ± 0.132 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.092 ± 0.006 | 0.022 ± 0.001 | 0.217 ± 0.058 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.061 ± 0.010 | 0.021 ± 0.001 | 0.080 ± 0.004 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.057 ± 0.015 | 0.021 ± 0.002 | 0.102 ± 0.066 | 0.007 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.044 ± 0.008 | 0.024 ± 0.003 | 0.084 ± 0.019 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.001 | 0.032 ± 0.006 | 0.011 ± 0.002 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.000 | 0.044 ± 0.006 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.000 | 0.045 ± 0.003 | 0.043 ± 0.020 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.108 ± 0.008 | 0.017 ± 0.000 | 0.096 ± 0.007 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.116 ± 0.007 | 0.020 ± 0.003 | 0.215 ± 0.144 | 0.025 ± 0.016 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.135 ± 0.005 | 0.036 ± 0.008 | 0.129 ± 0.025 | 0.026 ± 0.019 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.044 ± 0.002 | 0.015 ± 0.000 | 0.069 ± 0.009 | 0.020 ± 0.012 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.051 ± 0.002 | 0.019 ± 0.001 | 0.092 ± 0.050 | 0.018 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.058 ± 0.003 | 0.030 ± 0.001 | 0.080 ± 0.006 | 0.025 ± 0.004 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.005 ± 0.001 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.019 ± 0.012 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.003 ± 0.001 | 0.003 ± 0.000 | 0.001 ± 0.001 | 0.006 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.002 | 0.060 ± 0.033 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.003 ± 0.001 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.005 ± 0.002 | 0.003 ± 0.001 | 0.033 ± 0.006 | 0.010 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.035 ± 0.004 | 0.010 ± 0.003 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.006 ± 0.000 | 0.005 ± 0.001 | 0.036 ± 0.015 | 0.009 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.008 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.005 ± 0.002 | 0.007 ± 0.001 | 0.012 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.011 ± 0.002 | 0.006 ± 0.003 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.005 ± 0.005 | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.010 ± 0.004 | 0.009 ± 0.000 | 0.008 ± 0.007 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.012 ± 0.005 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.007 ± 0.001 | 0.007 ± 0.001 | 0.008 ± 0.000 | 0.015 ± 0.004 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.017 ± 0.004 | 0.014 ± 0.001 | 0.015 ± 0.000 | 0.024 ± 0.002 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 27.2x faster than the slowest successful cell (`jax-cpu`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 70.5x faster than the slowest successful cell (`jax-cpu`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 48.4x faster than the slowest successful cell (`jax-cpu`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 18.0x faster than the slowest successful cell (`jax-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 17.7x faster than the slowest successful cell (`jax-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 40.5x faster than the slowest successful cell (`jax-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 43.7x faster than the slowest successful cell (`jax-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`128x128`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.208 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`64x64,rhs=64`): `tenferro-eager` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.304 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 28.8x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 28.8x faster than the slowest successful cell (`jax-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 16.3x faster than the slowest successful cell (`jax-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 29.1x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 19.1x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 17.1x faster than the slowest successful cell (`tenferro-eager`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 15.2x faster than the slowest successful cell (`tenferro-eager`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 39.1x faster than the slowest successful cell (`tenferro-eager`, 0.450 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 0.450 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 50.4x faster than the slowest successful cell (`tenferro-eager`, 0.472 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 17.3x faster than the slowest successful cell (`tenferro-eager`, 0.472 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 49.8x faster than the slowest successful cell (`tenferro-eager`, 0.463 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`tenferro-eager`, 0.463 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 23.9x faster than the slowest successful cell (`pytorch-cpu`, 0.233 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.233 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 30.1x faster than the slowest successful cell (`pytorch-cpu`, 0.253 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.293 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 66.4x faster than the slowest successful cell (`tenferro-eager`, 0.564 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`tenferro-eager`, 0.564 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 54.9x faster than the slowest successful cell (`tenferro-eager`, 0.633 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 0.633 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 45.5x faster than the slowest successful cell (`tenferro-eager`, 0.615 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 0.615 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 22.4x faster than the slowest successful cell (`pytorch-cpu`, 0.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 30.4x faster than the slowest successful cell (`pytorch-cpu`, 0.253 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.253 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 26.0x faster than the slowest successful cell (`pytorch-cpu`, 0.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.215 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`64x64`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

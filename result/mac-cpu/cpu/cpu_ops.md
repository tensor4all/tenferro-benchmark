# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `74d138135bc227c5a1ff41db8863f853b647d863`
- Benchmark commit: `cc2153b9548e1542f955c011dccfb77a98196fed`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261023_172845/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20261023_172845`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261023_172845`.

- tenferro-rs commit: `74d138135bc227c5a1ff41db8863f853b647d863`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261023_172845/cpu_ops_t1_20261023_172845.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20261023_172845/cpu_ops_t1_20261023_172845.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.003 | 0.010 ± 0.001 | 0.008 ± 0.001 | 0.016 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.033 ± 0.044 | 0.024 ± 0.012 | 0.021 ± 0.004 | 0.051 ± 0.011 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.029 ± 0.007 | 0.028 ± 0.004 | 0.025 ± 0.002 | 0.032 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.087 ± 0.022 | 0.102 ± 0.010 | 0.081 ± 0.005 | 0.112 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.001 | 0.005 ± 0.002 | 0.008 ± 0.000 | 0.073 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.006 ± 0.004 | 0.007 ± 0.003 | 0.009 ± 0.000 | 0.086 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.051 ± 0.012 | 0.052 ± 0.003 | 0.039 ± 0.004 | 0.104 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.001 | 0.004 ± 0.001 | 0.009 ± 0.000 | 0.089 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.011 ± 0.004 | 0.010 ± 0.000 | 0.094 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.003 | 0.008 ± 0.003 | 0.008 ± 0.002 | 0.012 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.019 ± 0.004 | 0.024 ± 0.003 | 0.018 ± 0.003 | 0.026 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.013 ± 0.006 | 0.014 ± 0.002 | 0.012 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.036 ± 0.000 | 0.038 ± 0.002 | 0.037 ± 0.017 | 0.051 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.003 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.012 ± 0.006 | 0.011 ± 0.011 | 0.016 ± 0.001 | 0.028 ± 0.019 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.010 ± 0.002 | 0.012 ± 0.001 | 0.011 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.024 ± 0.006 | 0.015 ± 0.001 | 0.020 ± 0.001 | 0.020 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.016 ± 0.004 | 0.022 ± 0.004 | 0.015 ± 0.000 | 0.041 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.075 ± 0.024 | 0.048 ± 0.017 | 0.042 ± 0.004 | 0.131 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.046 ± 0.001 | 0.049 ± 0.010 | 0.046 ± 0.001 | 0.071 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.222 ± 0.079 | 0.198 ± 0.046 | 0.162 ± 0.008 | 0.261 ± 0.012 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.001 | 0.060 ± 0.004 | 0.007 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.003 | 0.063 ± 0.015 | 0.009 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.011 | 0.061 ± 0.004 | 0.014 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.064 ± 0.009 | 0.028 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.004 | 0.047 ± 0.003 | 0.013 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.050 ± 0.033 | 0.053 ± 0.001 | 0.022 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.004 | 0.047 ± 0.001 | 0.015 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.035 ± 0.001 | 0.072 ± 0.014 | 0.026 ± 0.006 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.390 ± 0.064 | 0.316 ± 0.005 | 0.297 ± 0.022 | 0.339 ± 0.013 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 5.086 ± 0.202 | 5.635 ± 0.398 | 8.242 ± 1.671 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 31.257 ± 6.275 | 29.048 ± 3.498 | 41.052 ± 0.527 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 5.630 ± 1.807 | 4.426 ± 0.049 | 7.449 ± 2.113 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 30.050 ± 1.592 | 27.937 ± 0.949 | 40.756 ± 1.094 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 1.098 ± 0.168 | 1.743 ± 0.022 | 3.407 ± 1.567 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 7.730 ± 0.092 | 14.935 ± 1.590 | 20.210 ± 0.429 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 1.106 ± 0.076 | 1.061 ± 0.015 | 2.644 ± 0.124 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 8.115 ± 0.244 | 7.611 ± 0.132 | 20.319 ± 2.853 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.018 ± 0.003 | 0.008 ± 0.011 | 0.010 ± 0.001 | 0.034 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.047 ± 0.009 | 0.070 ± 0.001 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 2.405 ± 0.129 | 2.788 ± 0.189 | 5.081 ± 2.403 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 14.045 ± 0.368 | 15.647 ± 1.253 | 31.754 ± 2.389 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 2.406 ± 0.221 | 2.666 ± 0.243 | 4.850 ± 3.048 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 13.886 ± 2.280 | 13.595 ± 3.433 | 31.104 ± 3.932 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.041 ± 0.000 | 0.071 ± 0.002 | 0.048 ± 0.010 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.286 ± 0.243 | 0.478 ± 0.077 | 0.321 ± 0.006 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 1.855 ± 2.819 | 1.828 ± 0.105 | 2.196 ± 2.998 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.285 ± 0.006 | 0.570 ± 0.013 | 0.342 ± 0.016 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 1.561 ± 0.064 | 2.615 ± 0.191 | 1.859 ± 0.232 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.415 ± 0.007 | 0.436 ± 0.013 | 0.621 ± 0.360 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 9.147 ± 0.036 | 10.290 ± 0.467 | 10.856 ± 0.140 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 57.114 ± 4.927 | 59.378 ± 12.848 | 67.015 ± 1.660 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 9.147 ± 0.178 | 9.358 ± 0.389 | 11.783 ± 3.825 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 56.743 ± 5.610 | 52.934 ± 4.382 | 67.577 ± 2.266 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.028 ± 0.013 | 0.020 ± 0.000 | 0.026 ± 0.003 | 0.129 ± 0.095 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.185 ± 0.028 | 0.123 ± 0.003 | 0.170 ± 0.028 | 0.876 ± 0.196 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.504 ± 0.069 | 0.497 ± 0.019 | 0.607 ± 0.050 | 3.585 ± 2.134 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.067 ± 0.001 | 0.069 ± 0.002 | 0.069 ± 0.004 | 0.081 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.020 ± 0.009 | 0.022 ± 0.000 | 0.034 ± 0.004 | 0.047 ± 0.016 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.038 ± 0.005 | 0.027 ± 0.001 | 0.033 ± 0.004 | 0.062 ± 0.007 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.028 ± 0.001 | 0.032 ± 0.001 | 0.045 ± 0.001 | 0.138 ± 0.051 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.468 ± 0.096 | 0.409 ± 0.003 | 0.413 ± 0.056 | 0.556 ± 0.212 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.005 ± 0.001 | 0.005 ± 0.003 | 0.004 ± 0.000 | 0.008 ± 0.004 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.006 ± 0.002 | 0.007 ± 0.002 | 0.005 ± 0.000 | 0.010 ± 0.003 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.013 ± 0.004 | 0.009 ± 0.000 | 0.013 ± 0.010 | 0.013 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.001 | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.074 ± 0.021 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.002 ± 0.002 | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.074 ± 0.009 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.002 ± 0.004 | 0.006 ± 0.000 | 0.009 ± 0.002 | 0.089 ± 0.006 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | - | 0.015 ± 0.004 | 0.067 ± 0.002 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | - | 0.017 ± 0.002 | 0.070 ± 0.004 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | - | 0.025 ± 0.000 | 0.081 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | - | 0.018 ± 0.006 | 0.058 ± 0.001 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | - | 0.017 ± 0.000 | 0.059 ± 0.003 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | - | 0.023 ± 0.010 | 0.073 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | - | 0.018 ± 0.013 | 0.100 ± 0.004 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | - | 0.018 ± 0.001 | 0.104 ± 0.004 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | - | 0.017 ± 0.000 | 0.123 ± 0.017 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | - | 0.020 ± 0.003 | 0.079 ± 0.011 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | - | 0.016 ± 0.000 | 0.079 ± 0.009 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | - | 0.020 ± 0.000 | 0.089 ± 0.003 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.002 | 0.030 ± 0.005 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.008 | 0.024 ± 0.001 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.035 ± 0.008 | 0.009 ± 0.004 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | - | 0.033 ± 0.008 | 0.078 ± 0.024 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | - | 0.029 ± 0.006 | 0.076 ± 0.003 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | - | 0.029 ± 0.036 | 0.087 ± 0.003 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | - | 0.032 ± 0.007 | 0.075 ± 0.001 | 0.008 ± 0.007 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | - | 0.035 ± 0.012 | 0.076 ± 0.007 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | - | 0.033 ± 0.003 | 0.085 ± 0.001 | 0.011 ± 0.010 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.006 | 0.047 ± 0.038 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.002 | 0.039 ± 0.001 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.039 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.016 ± 0.010 | 0.163 ± 0.027 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.011 ± 0.000 | 0.157 ± 0.010 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.012 ± 0.003 | 0.155 ± 0.004 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.012 ± 0.001 | 0.076 ± 0.001 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.012 ± 0.000 | 0.076 ± 0.002 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.012 ± 0.001 | 0.077 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.011 | 0.033 ± 0.042 | 0.013 ± 0.005 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.003 | 0.033 ± 0.031 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.001 | 0.043 ± 0.005 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | - | 0.012 ± 0.013 | 0.108 ± 0.030 | 0.013 ± 0.004 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | - | 0.014 ± 0.002 | 0.097 ± 0.009 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | - | 0.026 ± 0.011 | 0.114 ± 0.008 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | - | 0.013 ± 0.002 | 0.063 ± 0.002 | 0.014 ± 0.006 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | - | 0.013 ± 0.003 | 0.064 ± 0.005 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | - | 0.020 ± 0.001 | 0.075 ± 0.007 | 0.018 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.003 ± 0.001 | 0.001 ± 0.000 | 0.005 ± 0.009 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.004 ± 0.000 | 0.006 ± 0.003 | 0.002 ± 0.000 | 0.006 ± 0.006 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.001 | 0.004 ± 0.001 | 0.001 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.007 ± 0.001 | 0.007 ± 0.004 | 0.006 ± 0.001 | 0.025 ± 0.006 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.002 ± 0.002 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.006 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.005 ± 0.002 | 0.005 ± 0.002 | 0.003 ± 0.001 | 0.008 ± 0.002 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.005 ± 0.004 | 0.005 ± 0.000 | 0.007 ± 0.003 | 0.009 ± 0.002 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.003 | 0.008 ± 0.000 | 0.007 ± 0.003 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.007 ± 0.003 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.004 ± 0.004 | 0.008 ± 0.002 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.004 ± 0.002 | 0.007 ± 0.003 | 0.009 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.005 | 0.007 ± 0.003 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.006 ± 0.006 | 0.005 ± 0.000 | 0.014 ± 0.010 | 0.007 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.007 ± 0.006 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.008 ± 0.002 | 0.008 ± 0.003 | 0.008 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.016 ± 0.002 | 0.014 ± 0.001 | 0.018 ± 0.013 | 0.020 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 21.7x faster than the slowest successful cell (`jax-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 29.5x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 22.1x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 17.5x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 23.2x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 31.2x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 35.9x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 22.5x faster than the slowest successful cell (`pytorch-cpu`, 0.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 19.3x faster than the slowest successful cell (`pytorch-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261023_175432/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20261023_175432`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261023_175432`.

- tenferro-rs commit: `74d138135bc227c5a1ff41db8863f853b647d863`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261023_175432/cpu_ops_t4_20261023_175432.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20261023_175432/cpu_ops_t4_20261023_175432.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.009 ± 0.002 | 0.009 ± 0.002 | 0.016 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.001 | 0.019 ± 0.001 | 0.021 ± 0.003 | 0.067 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.022 ± 0.001 | 0.024 ± 0.001 | 0.026 ± 0.004 | 0.033 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.079 ± 0.005 | 0.096 ± 0.001 | 0.080 ± 0.013 | 0.120 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.012 ± 0.003 | 0.078 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.089 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.044 ± 0.004 | 0.050 ± 0.001 | 0.089 ± 0.040 | 0.251 ± 0.078 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.009 ± 0.000 | 0.084 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.003 | 0.077 ± 0.018 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.036 ± 0.007 | 0.015 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.057 ± 0.019 | 0.029 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.010 ± 0.001 | 0.012 ± 0.001 | 0.043 ± 0.011 | 0.018 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.032 ± 0.007 | 0.044 ± 0.002 | 0.067 ± 0.010 | 0.093 ± 0.022 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.037 ± 0.003 | 0.009 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.013 ± 0.001 | 0.010 ± 0.000 | 0.050 ± 0.028 | 0.035 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.001 | 0.007 ± 0.000 | 0.041 ± 0.029 | 0.011 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.052 ± 0.007 | 0.051 ± 0.009 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.013 ± 0.001 | 0.014 ± 0.001 | 0.033 ± 0.027 | 0.043 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.049 ± 0.005 | 0.041 ± 0.001 | 0.045 ± 0.009 | 0.144 ± 0.013 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.040 ± 0.000 | 0.044 ± 0.001 | 0.048 ± 0.004 | 0.075 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.186 ± 0.065 | 0.154 ± 0.004 | 0.182 ± 0.089 | 0.318 ± 0.089 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.001 | 0.053 ± 0.003 | 0.008 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.059 ± 0.009 | 0.011 ± 0.011 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.064 ± 0.018 | 0.031 ± 0.016 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.003 | 0.059 ± 0.007 | 0.082 ± 0.025 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.003 | 0.087 ± 0.013 | 0.014 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.118 ± 0.029 | 0.047 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.001 | 0.093 ± 0.005 | 0.025 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.036 ± 0.003 | 0.102 ± 0.060 | 0.071 ± 0.022 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.303 ± 0.020 | 0.341 ± 0.004 | 0.342 ± 0.065 | 0.372 ± 0.018 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 4.926 ± 0.491 | 6.023 ± 0.562 | 7.485 ± 3.072 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 28.857 ± 0.600 | 36.245 ± 16.585 | 34.556 ± 3.478 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 6.070 ± 0.461 | 5.403 ± 0.153 | 7.023 ± 1.090 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 29.014 ± 1.274 | 30.686 ± 4.866 | 34.637 ± 2.313 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 1.682 ± 0.295 | 3.297 ± 0.348 | 2.998 ± 1.491 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 9.587 ± 1.285 | 19.020 ± 4.205 | 10.956 ± 3.369 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 1.261 ± 0.243 | 2.364 ± 0.816 | 2.656 ± 1.611 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 7.940 ± 0.319 | 20.914 ± 9.351 | 11.082 ± 3.736 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.014 ± 0.000 | 0.008 ± 0.000 | 0.019 ± 0.016 | 0.045 ± 0.021 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.005 | 0.053 ± 0.009 | 0.084 ± 0.050 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 4.106 ± 1.868 | 3.812 ± 0.650 | 3.728 ± 1.858 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 15.296 ± 0.949 | 16.797 ± 1.347 | 24.850 ± 7.190 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 2.876 ± 0.298 | 4.071 ± 4.410 | 4.625 ± 1.596 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 17.475 ± 2.718 | 16.151 ± 5.793 | 25.101 ± 3.284 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.002 | 0.101 ± 0.036 | 0.064 ± 0.016 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.328 ± 0.033 | 0.720 ± 0.223 | 0.354 ± 0.021 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.384 ± 0.015 | 2.348 ± 0.556 | 2.884 ± 4.100 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.508 ± 0.040 | 0.741 ± 0.124 | 0.524 ± 0.194 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.563 ± 0.281 | 4.933 ± 2.351 | 2.274 ± 1.532 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.479 ± 0.173 | 0.560 ± 0.143 | 0.563 ± 0.137 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 9.548 ± 0.552 | 11.924 ± 0.828 | 10.498 ± 0.582 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 62.686 ± 10.217 | 65.097 ± 13.239 | 71.775 ± 7.913 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 9.663 ± 0.285 | 9.634 ± 0.613 | 10.925 ± 1.046 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 71.659 ± 2.771 | 55.223 ± 6.573 | 77.494 ± 24.121 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.022 ± 0.000 | 0.019 ± 0.001 | 0.030 ± 0.011 | 0.150 ± 0.032 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.134 ± 0.014 | 0.115 ± 0.006 | 0.288 ± 0.170 | 0.468 ± 0.243 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.422 ± 0.035 | 0.425 ± 0.051 | 0.807 ± 1.942 | 1.729 ± 0.736 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.065 ± 0.003 | 0.064 ± 0.002 | 0.172 ± 0.065 | 0.084 ± 0.004 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.017 ± 0.000 | 0.024 ± 0.002 | 0.045 ± 0.022 | 0.056 ± 0.018 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.019 ± 0.000 | 0.027 ± 0.001 | 0.051 ± 0.001 | 0.129 ± 0.063 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.026 ± 0.001 | 0.034 ± 0.001 | 0.142 ± 0.077 | 0.259 ± 0.160 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.411 ± 0.028 | 0.415 ± 0.043 | 0.619 ± 0.326 | 0.526 ± 0.028 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.002 | 0.006 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.004 ± 0.002 | 0.005 ± 0.000 | 0.008 ± 0.004 | 0.011 ± 0.003 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.008 ± 0.001 | 0.010 ± 0.003 | 0.011 ± 0.003 | 0.014 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.102 ± 0.028 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.097 ± 0.030 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.005 | 0.098 ± 0.005 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | - | 0.015 ± 0.001 | 0.121 ± 0.065 | 0.008 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | - | 0.016 ± 0.001 | 0.103 ± 0.087 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | - | 0.022 ± 0.001 | 0.242 ± 0.072 | 0.010 ± 0.008 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | - | 0.014 ± 0.000 | 0.071 ± 0.021 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | - | 0.015 ± 0.001 | 0.113 ± 0.028 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | - | 0.029 ± 0.012 | 0.235 ± 0.194 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | - | 0.017 ± 0.002 | 0.521 ± 0.229 | 0.010 ± 0.005 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | - | 0.015 ± 0.000 | 0.409 ± 0.085 | 0.018 ± 0.011 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | - | 0.016 ± 0.003 | 0.484 ± 0.069 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | - | 0.015 ± 0.000 | 0.457 ± 0.169 | 0.007 ± 0.014 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | - | 0.014 ± 0.001 | 0.332 ± 0.220 | 0.011 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | - | 0.019 ± 0.002 | 0.421 ± 0.176 | 0.019 ± 0.012 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.001 | 0.038 ± 0.011 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.000 | 0.027 ± 0.002 | 0.014 ± 0.014 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.001 | 0.053 ± 0.047 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | - | 0.030 ± 0.004 | 0.243 ± 0.028 | 0.010 ± 0.006 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | - | 0.024 ± 0.004 | 0.294 ± 0.241 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | - | 0.034 ± 0.033 | 0.293 ± 0.288 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | - | 0.026 ± 0.001 | 0.263 ± 0.037 | 0.008 ± 0.006 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | - | 0.025 ± 0.002 | 0.257 ± 0.086 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | - | 0.030 ± 0.002 | 0.311 ± 0.175 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.001 | 0.056 ± 0.006 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.001 | 0.063 ± 0.037 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.002 | 0.066 ± 0.032 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.012 ± 0.001 | 0.212 ± 0.248 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.010 ± 0.000 | 0.192 ± 0.032 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.021 ± 0.004 | 0.201 ± 0.739 | 0.013 ± 0.006 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.011 ± 0.000 | 0.092 ± 0.092 | 0.008 ± 0.007 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.011 ± 0.000 | 0.102 ± 0.017 | 0.009 ± 0.004 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.013 ± 0.000 | 0.087 ± 0.086 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.002 | 0.040 ± 0.007 | 0.012 ± 0.008 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.011 ± 0.000 | 0.042 ± 0.018 | 0.015 ± 0.006 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.000 | 0.052 ± 0.048 | 0.021 ± 0.003 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | - | 0.010 ± 0.001 | 0.211 ± 0.077 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | - | 0.012 ± 0.001 | 0.247 ± 0.117 | 0.013 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | - | 0.023 ± 0.003 | 0.184 ± 0.074 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | - | 0.011 ± 0.000 | 0.072 ± 0.019 | 0.013 ± 0.003 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | - | 0.012 ± 0.000 | 0.091 ± 0.021 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | - | 0.027 ± 0.011 | 0.133 ± 0.085 | 0.018 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.002 ± 0.001 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.003 ± 0.001 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.017 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.001 | 0.009 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.028 ± 0.042 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.008 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.027 ± 0.008 | 0.007 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.004 ± 0.001 | 0.006 ± 0.001 | 0.044 ± 0.010 | 0.008 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.005 ± 0.002 | 0.050 ± 0.004 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.006 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.016 ± 0.008 | 0.006 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.009 ± 0.005 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.008 | 0.007 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.005 | 0.008 ± 0.005 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.004 ± 0.002 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.006 ± 0.001 | 0.007 ± 0.001 | 0.009 ± 0.000 | 0.015 ± 0.003 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.013 ± 0.001 | 0.013 ± 0.002 | 0.015 ± 0.001 | 0.023 ± 0.004 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 31.6x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 22.9x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 24.1x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 31.3x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 38.2x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 18.9x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 42.2x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 44.7x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 19.0x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 23.8x faster than the slowest successful cell (`pytorch-cpu`, 0.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 27.1x faster than the slowest successful cell (`pytorch-cpu`, 0.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 49.8x faster than the slowest successful cell (`pytorch-cpu`, 0.521 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 30.9x faster than the slowest successful cell (`pytorch-cpu`, 0.521 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 22.1x faster than the slowest successful cell (`pytorch-cpu`, 0.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 27.0x faster than the slowest successful cell (`pytorch-cpu`, 0.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 85.6x faster than the slowest successful cell (`pytorch-cpu`, 0.484 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 29.6x faster than the slowest successful cell (`pytorch-cpu`, 0.484 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 61.9x faster than the slowest successful cell (`pytorch-cpu`, 0.457 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 30.6x faster than the slowest successful cell (`pytorch-cpu`, 0.457 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 30.4x faster than the slowest successful cell (`pytorch-cpu`, 0.332 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 23.0x faster than the slowest successful cell (`pytorch-cpu`, 0.332 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 22.5x faster than the slowest successful cell (`pytorch-cpu`, 0.421 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 21.6x faster than the slowest successful cell (`pytorch-cpu`, 0.421 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 23.4x faster than the slowest successful cell (`pytorch-cpu`, 0.243 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 31.6x faster than the slowest successful cell (`pytorch-cpu`, 0.294 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.294 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 52.6x faster than the slowest successful cell (`pytorch-cpu`, 0.293 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 31.7x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 29.6x faster than the slowest successful cell (`pytorch-cpu`, 0.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 59.6x faster than the slowest successful cell (`pytorch-cpu`, 0.311 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.311 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 27.6x faster than the slowest successful cell (`pytorch-cpu`, 0.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 18.0x faster than the slowest successful cell (`pytorch-cpu`, 0.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 22.5x faster than the slowest successful cell (`pytorch-cpu`, 0.192 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 0.192 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.201 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 0.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 20.4x faster than the slowest successful cell (`pytorch-cpu`, 0.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 0.247 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.247 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.184 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

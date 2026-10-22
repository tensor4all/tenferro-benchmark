# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`
- Benchmark commit: `12e0b50f983ca3253ff46841eb15912448ae3e20`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261022_172907/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20261022_172907`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261022_172907`.

- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261022_172907/cpu_ops_t1_20261022_172907.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20261022_172907/cpu_ops_t1_20261022_172907.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.001 | 0.009 ± 0.003 | 0.008 ± 0.000 | 0.029 ± 0.014 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.030 ± 0.038 | 0.021 ± 0.003 | 0.021 ± 0.002 | 0.053 ± 0.014 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.037 ± 0.009 | 0.026 ± 0.000 | 0.028 ± 0.007 | 0.044 ± 0.027 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.137 ± 0.119 | 0.108 ± 0.004 | 0.108 ± 0.032 | 0.122 ± 0.026 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.002 | 0.005 ± 0.003 | 0.010 ± 0.001 | 0.321 ± 0.274 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.003 | 0.006 ± 0.001 | 0.008 ± 0.001 | 0.176 ± 0.126 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.121 ± 0.009 | 0.080 ± 0.027 | 0.043 ± 0.021 | 0.272 ± 0.363 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.006 ± 0.005 | 0.004 ± 0.000 | 0.013 ± 0.024 | 0.141 ± 0.078 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.013 ± 0.001 | 0.006 ± 0.000 | 0.012 ± 0.006 | 0.105 ± 0.019 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.004 | 0.008 ± 0.000 | 0.007 ± 0.001 | 0.016 ± 0.018 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.008 | 0.028 ± 0.007 | 0.018 ± 0.005 | 0.041 ± 0.028 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.013 ± 0.002 | 0.014 ± 0.000 | 0.012 ± 0.002 | 0.024 ± 0.009 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.048 ± 0.004 | 0.040 ± 0.008 | 0.042 ± 0.005 | 0.051 ± 0.016 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.004 | 0.010 ± 0.009 | 0.012 ± 0.001 | 0.010 ± 0.014 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.022 ± 0.004 | 0.014 ± 0.003 | 0.019 ± 0.005 | 0.021 ± 0.016 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.014 ± 0.002 | 0.008 ± 0.001 | 0.013 ± 0.005 | 0.015 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.023 ± 0.013 | 0.019 ± 0.004 | 0.031 ± 0.014 | 0.022 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.024 ± 0.013 | 0.016 ± 0.006 | 0.016 ± 0.001 | 0.080 ± 0.067 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.082 ± 0.035 | 0.059 ± 0.010 | 0.042 ± 0.002 | 0.227 ± 0.097 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.072 ± 0.008 | 0.054 ± 0.014 | 0.060 ± 0.025 | 0.106 ± 0.049 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.309 ± 0.211 | 0.169 ± 0.010 | 0.255 ± 0.150 | 0.266 ± 0.062 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.001 | 0.084 ± 0.031 | 0.010 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.002 | 0.111 ± 0.113 | 0.011 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.002 | 0.073 ± 0.041 | 0.023 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.003 | 0.080 ± 0.005 | 0.033 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.003 | 0.077 ± 0.026 | 0.029 ± 0.017 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.001 | 0.067 ± 0.045 | 0.021 ± 0.018 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.003 | 0.046 ± 0.002 | 0.025 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.050 ± 0.004 | 0.078 ± 0.034 | 0.029 ± 0.002 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.374 ± 0.015 | 0.501 ± 0.183 | 0.301 ± 0.016 | 0.351 ± 0.011 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 6.952 ± 1.671 | 5.125 ± 0.163 | 7.793 ± 0.839 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 35.534 ± 1.741 | 31.135 ± 0.564 | 47.590 ± 4.674 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 6.376 ± 2.432 | 4.646 ± 0.052 | 7.311 ± 0.780 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 33.753 ± 5.131 | 28.590 ± 1.189 | 47.115 ± 1.831 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 1.329 ± 0.823 | 1.967 ± 0.073 | 3.090 ± 1.115 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 8.995 ± 5.001 | 19.545 ± 6.710 | 25.033 ± 0.928 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 2.222 ± 1.279 | 1.187 ± 0.045 | 3.915 ± 1.104 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 9.105 ± 1.276 | 9.509 ± 0.343 | 24.691 ± 4.952 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.041 ± 0.023 | 0.009 ± 0.001 | 0.010 ± 0.003 | 0.038 ± 0.009 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.042 ± 0.019 | 0.044 ± 0.001 | 0.086 ± 0.077 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 4.166 ± 3.103 | 2.533 ± 0.025 | 5.904 ± 0.330 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 22.111 ± 1.716 | 16.211 ± 0.202 | 47.472 ± 10.732 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 3.179 ± 0.459 | 2.362 ± 0.131 | 4.842 ± 0.245 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 20.213 ± 6.780 | 15.459 ± 0.309 | 38.095 ± 7.792 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.069 ± 0.024 | 0.080 ± 0.006 | 0.051 ± 0.009 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.377 ± 0.329 | 0.429 ± 0.048 | 0.372 ± 0.063 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 1.665 ± 0.146 | 2.101 ± 0.320 | 2.585 ± 2.490 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.348 ± 0.125 | 0.885 ± 0.285 | 0.392 ± 0.288 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 1.627 ± 0.436 | 3.133 ± 0.185 | 3.360 ± 2.711 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.598 ± 0.095 | 0.450 ± 0.053 | 0.640 ± 0.142 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 12.159 ± 2.005 | 9.511 ± 0.135 | 12.429 ± 0.966 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 73.868 ± 29.011 | 61.947 ± 4.444 | 95.544 ± 14.805 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 11.879 ± 2.180 | 8.773 ± 0.565 | 14.423 ± 2.880 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 78.424 ± 8.373 | 54.050 ± 2.694 | 86.763 ± 7.660 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.037 ± 0.014 | 0.020 ± 0.001 | 0.030 ± 0.017 | 0.150 ± 0.032 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.414 ± 0.358 | 0.210 ± 0.441 | 0.201 ± 0.042 | 0.875 ± 0.042 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.539 ± 0.120 | 2.003 ± 1.582 | 0.665 ± 0.050 | 3.539 ± 0.249 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.095 ± 0.023 | 0.114 ± 0.049 | 0.072 ± 0.003 | 0.079 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.051 ± 0.003 | 0.027 ± 0.007 | 0.032 ± 0.005 | 0.030 ± 0.006 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.029 ± 0.005 | 0.027 ± 0.000 | 0.035 ± 0.003 | 0.083 ± 0.052 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.027 ± 0.000 | 0.033 ± 0.001 | 0.050 ± 0.013 | 0.218 ± 0.056 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.527 ± 0.172 | 0.630 ± 0.207 | 0.419 ± 0.015 | 0.429 ± 0.029 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.004 ± 0.001 | 0.005 ± 0.001 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.006 ± 0.003 | 0.010 ± 0.002 | 0.006 ± 0.002 | 0.010 ± 0.002 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.013 ± 0.008 | 0.011 ± 0.003 | 0.011 ± 0.003 | 0.014 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.012 ± 0.003 | 0.008 ± 0.000 | 0.096 ± 0.009 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.003 ± 0.001 | 0.009 ± 0.003 | 0.009 ± 0.011 | 0.140 ± 0.019 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.004 ± 0.001 | 0.008 ± 0.001 | 0.009 ± 0.002 | 0.129 ± 0.022 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | - | 0.033 ± 0.014 | 0.076 ± 0.012 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | - | 0.026 ± 0.018 | 0.136 ± 0.241 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | - | 0.027 ± 0.014 | 0.101 ± 0.011 | 0.012 ± 0.004 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | - | 0.033 ± 0.013 | 0.062 ± 0.013 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | - | 0.028 ± 0.005 | 0.139 ± 0.087 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | - | 0.035 ± 0.016 | 0.088 ± 0.016 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | - | 0.033 ± 0.016 | 0.118 ± 0.059 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | - | 0.026 ± 0.015 | 0.102 ± 0.016 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | - | 0.028 ± 0.015 | 0.176 ± 0.040 | 0.007 ± 0.002 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | - | 0.023 ± 0.015 | 0.096 ± 0.033 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | - | 0.021 ± 0.045 | 0.093 ± 0.015 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | - | 0.022 ± 0.009 | 0.100 ± 0.094 | 0.005 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.009 | 0.030 ± 0.009 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.010 | 0.037 ± 0.021 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.004 | 0.036 ± 0.007 | 0.007 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | - | 0.031 ± 0.029 | 0.095 ± 0.022 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | - | 0.078 ± 0.008 | 0.098 ± 0.031 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | - | 0.042 ± 0.041 | 0.112 ± 0.034 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | - | 0.045 ± 0.034 | 0.088 ± 0.005 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | - | 0.043 ± 0.020 | 0.129 ± 0.214 | 0.011 ± 0.003 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | - | 0.061 ± 0.025 | 0.112 ± 0.034 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.015 | 0.046 ± 0.007 | 0.011 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.031 ± 0.018 | 0.045 ± 0.009 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.032 ± 0.019 | 0.046 ± 0.033 | 0.010 ± 0.022 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.027 ± 0.015 | 0.172 ± 0.034 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.015 ± 0.008 | 0.607 ± 0.362 | 0.011 ± 0.004 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.019 ± 0.009 | 0.307 ± 0.082 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.025 ± 0.019 | 0.078 ± 0.016 | 0.011 ± 0.005 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.024 ± 0.014 | 0.158 ± 0.084 | 0.011 ± 0.003 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.031 ± 0.019 | 0.146 ± 0.065 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.009 | 0.034 ± 0.008 | 0.014 ± 0.003 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.008 | 0.045 ± 0.039 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.032 ± 0.004 | 0.073 ± 0.039 | 0.020 ± 0.003 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | - | 0.022 ± 0.010 | 0.128 ± 0.027 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | - | 0.034 ± 0.013 | 0.110 ± 0.016 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | - | 0.039 ± 0.024 | 0.108 ± 0.003 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | - | 0.021 ± 0.015 | 0.081 ± 0.022 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | - | 0.032 ± 0.003 | 0.076 ± 0.013 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | - | 0.038 ± 0.014 | 0.082 ± 0.070 | 0.020 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.001 | 0.009 ± 0.002 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.005 ± 0.001 | 0.007 ± 0.004 | 0.002 ± 0.000 | 0.008 ± 0.004 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.001 | 0.005 ± 0.001 | 0.001 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.019 ± 0.008 | 0.010 ± 0.007 | 0.008 ± 0.013 | 0.024 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.001 | 0.005 ± 0.002 | 0.001 ± 0.000 | 0.008 ± 0.003 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.002 | 0.003 ± 0.001 | 0.008 ± 0.002 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.008 ± 0.003 | 0.003 ± 0.000 | 0.010 ± 0.002 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.008 ± 0.002 | 0.007 ± 0.002 | 0.004 ± 0.001 | 0.011 ± 0.004 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.004 ± 0.001 | 0.014 ± 0.007 | 0.009 ± 0.001 | 0.007 ± 0.002 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.004 ± 0.002 | 0.007 ± 0.004 | 0.009 ± 0.002 | 0.008 ± 0.002 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.005 ± 0.002 | 0.006 ± 0.006 | 0.011 ± 0.002 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.004 ± 0.000 | 0.009 ± 0.009 | 0.010 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.006 ± 0.001 | 0.010 ± 0.004 | 0.009 ± 0.001 | 0.007 ± 0.002 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.006 ± 0.003 | 0.007 ± 0.003 | 0.010 ± 0.002 | 0.006 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.006 ± 0.002 | 0.011 ± 0.006 | 0.006 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.009 ± 0.001 | 0.015 ± 0.011 | 0.009 ± 0.011 | 0.021 ± 0.006 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.018 ± 0.001 | 0.025 ± 0.011 | 0.016 ± 0.016 | 0.022 ± 0.006 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 31.6x faster than the slowest successful cell (`jax-cpu`, 0.321 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 58.1x faster than the slowest successful cell (`jax-cpu`, 0.321 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 64.4x faster than the slowest successful cell (`jax-cpu`, 0.321 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 20.7x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 38.2x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 28.8x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 24.4x faster than the slowest successful cell (`jax-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 32.2x faster than the slowest successful cell (`jax-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 17.5x faster than the slowest successful cell (`jax-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 33.8x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 16.4x faster than the slowest successful cell (`jax-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 53.2x faster than the slowest successful cell (`jax-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 35.6x faster than the slowest successful cell (`jax-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`jax-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 0.139 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 24.7x faster than the slowest successful cell (`pytorch-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 20.0x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 20.6x faster than the slowest successful cell (`pytorch-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 20.6x faster than the slowest successful cell (`pytorch-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 20.5x faster than the slowest successful cell (`pytorch-cpu`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 57.5x faster than the slowest successful cell (`pytorch-cpu`, 0.607 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 39.3x faster than the slowest successful cell (`pytorch-cpu`, 0.607 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 40.3x faster than the slowest successful cell (`pytorch-cpu`, 0.307 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.307 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261022_180005/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20261022_180005`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261022_180005`.

- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261022_180005/cpu_ops_t4_20261022_180005.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20261022_180005/cpu_ops_t4_20261022_180005.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.002 | 0.011 ± 0.003 | 0.011 ± 0.002 | 0.023 ± 0.019 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.028 ± 0.018 | 0.038 ± 0.010 | 0.019 ± 0.002 | 0.075 ± 0.011 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.037 ± 0.008 | 0.035 ± 0.013 | 0.024 ± 0.001 | 0.039 ± 0.012 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.128 ± 0.036 | 0.148 ± 0.018 | 0.101 ± 0.019 | 0.126 ± 0.019 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.004 ± 0.001 | 0.005 ± 0.001 | 0.008 ± 0.000 | 0.103 ± 0.024 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.001 | 0.010 ± 0.002 | 0.008 ± 0.000 | 0.144 ± 0.109 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.062 ± 0.029 | 0.057 ± 0.003 | 0.069 ± 0.003 | 0.290 ± 0.193 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.005 ± 0.002 | 0.004 ± 0.001 | 0.009 ± 0.000 | 0.087 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.006 ± 0.001 | 0.010 ± 0.004 | 0.011 ± 0.002 | 0.094 ± 0.056 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.003 | 0.009 ± 0.001 | 0.036 ± 0.002 | 0.016 ± 0.017 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.020 ± 0.019 | 0.026 ± 0.003 | 0.050 ± 0.001 | 0.029 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.015 ± 0.012 | 0.017 ± 0.006 | 0.040 ± 0.002 | 0.018 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.057 ± 0.020 | 0.073 ± 0.019 | 0.116 ± 0.023 | 0.075 ± 0.065 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.012 ± 0.004 | 0.010 ± 0.000 | 0.046 ± 0.018 | 0.012 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.002 | 0.022 ± 0.005 | 0.051 ± 0.002 | 0.028 ± 0.015 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.013 ± 0.006 | 0.011 ± 0.001 | 0.047 ± 0.005 | 0.013 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.023 ± 0.007 | 0.026 ± 0.005 | 0.058 ± 0.019 | 0.025 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.020 ± 0.009 | 0.022 ± 0.008 | 0.015 ± 0.001 | 0.048 ± 0.009 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.071 ± 0.028 | 0.080 ± 0.022 | 0.045 ± 0.004 | 0.159 ± 0.010 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.061 ± 0.021 | 0.047 ± 0.014 | 0.047 ± 0.004 | 0.074 ± 0.020 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.222 ± 0.054 | 0.236 ± 0.091 | 0.232 ± 0.088 | 0.283 ± 0.030 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.003 | 0.065 ± 0.004 | 0.014 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.028 ± 0.008 | 0.059 ± 0.027 | 0.010 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.002 | 0.060 ± 0.005 | 0.025 ± 0.010 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.032 ± 0.009 | 0.076 ± 0.031 | 0.067 ± 0.016 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.007 | 0.088 ± 0.019 | 0.017 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.057 ± 0.031 | 0.106 ± 0.027 | 0.032 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.005 | 0.098 ± 0.005 | 0.029 ± 0.019 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.075 ± 0.017 | 0.144 ± 0.084 | 0.059 ± 0.014 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.457 ± 0.087 | 0.581 ± 0.235 | 0.353 ± 0.059 | 0.361 ± 0.138 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 6.688 ± 0.737 | 5.228 ± 0.260 | 6.296 ± 0.407 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 48.278 ± 15.955 | 32.531 ± 2.187 | 43.258 ± 4.707 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 6.116 ± 0.715 | 5.058 ± 0.199 | 7.147 ± 1.307 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 39.861 ± 8.019 | 30.043 ± 3.329 | 40.748 ± 2.086 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 2.334 ± 0.408 | 2.639 ± 0.322 | 3.382 ± 1.431 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 9.299 ± 2.156 | 14.459 ± 0.581 | 18.185 ± 7.641 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 2.562 ± 0.243 | 1.621 ± 0.350 | 2.686 ± 0.107 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 9.464 ± 5.154 | 8.340 ± 0.066 | 15.593 ± 1.240 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.022 ± 0.004 | 0.011 ± 0.003 | 0.011 ± 0.001 | 0.036 ± 0.005 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.039 ± 0.014 | 0.049 ± 0.007 | 0.080 ± 0.089 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 4.202 ± 0.848 | 2.772 ± 0.199 | 3.566 ± 0.353 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 20.838 ± 9.148 | 13.973 ± 0.349 | 22.497 ± 3.800 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 4.285 ± 1.426 | 2.877 ± 0.109 | 3.446 ± 1.094 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 22.780 ± 5.210 | 13.267 ± 0.204 | 23.351 ± 1.812 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.058 ± 0.009 | 0.066 ± 0.004 | 0.081 ± 0.033 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.532 ± 0.086 | 0.484 ± 0.039 | 0.454 ± 0.094 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 2.909 ± 0.919 | 1.616 ± 0.318 | 2.267 ± 0.598 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.532 ± 0.028 | 0.704 ± 0.051 | 0.518 ± 0.044 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.920 ± 0.426 | 2.420 ± 0.148 | 1.824 ± 0.188 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.759 ± 0.110 | 0.449 ± 0.065 | 0.724 ± 0.364 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 11.367 ± 1.074 | 9.919 ± 0.247 | 10.419 ± 1.017 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 78.919 ± 4.919 | 51.415 ± 0.802 | 74.219 ± 15.440 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 12.193 ± 1.402 | 8.749 ± 0.067 | 10.496 ± 0.879 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 73.897 ± 21.659 | 49.751 ± 0.276 | 72.146 ± 7.666 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.027 ± 0.002 | 0.023 ± 0.002 | 0.031 ± 0.010 | 0.129 ± 0.027 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.171 ± 0.105 | 0.188 ± 0.053 | 0.185 ± 0.029 | 0.404 ± 0.012 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.514 ± 0.025 | 0.957 ± 0.380 | 0.727 ± 0.384 | 1.933 ± 1.050 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.070 ± 0.034 | 0.093 ± 0.027 | 0.159 ± 0.053 | 0.082 ± 0.013 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.024 ± 0.010 | 0.030 ± 0.009 | 0.033 ± 0.006 | 0.048 ± 0.022 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.023 ± 0.010 | 0.027 ± 0.006 | 0.048 ± 0.005 | 0.101 ± 0.017 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.027 ± 0.001 | 0.034 ± 0.001 | 0.069 ± 0.030 | 0.219 ± 0.054 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.533 ± 0.156 | 0.823 ± 0.361 | 0.535 ± 0.158 | 0.501 ± 0.016 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.008 ± 0.003 | 0.005 ± 0.003 | 0.008 ± 0.002 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.006 ± 0.001 | 0.006 ± 0.001 | 0.006 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.012 ± 0.002 | 0.013 ± 0.004 | 0.011 ± 0.006 | 0.016 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.009 ± 0.003 | 0.012 ± 0.004 | 0.072 ± 0.024 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.007 ± 0.001 | 0.008 ± 0.002 | 0.097 ± 0.023 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.003 ± 0.002 | 0.007 ± 0.002 | 0.009 ± 0.002 | 0.097 ± 0.007 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | - | 0.019 ± 0.005 | 0.070 ± 0.029 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | - | 0.023 ± 0.004 | 0.076 ± 0.043 | 0.012 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | - | 0.038 ± 0.039 | 0.189 ± 0.076 | 0.020 ± 0.004 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | - | 0.023 ± 0.002 | 0.070 ± 0.021 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | - | 0.024 ± 0.020 | 0.067 ± 0.013 | 0.011 ± 0.004 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | - | 0.030 ± 0.003 | 0.152 ± 0.073 | 0.013 ± 0.008 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | - | 0.025 ± 0.003 | 0.375 ± 0.096 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | - | 0.020 ± 0.002 | 0.303 ± 0.063 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | - | 0.030 ± 0.004 | 0.268 ± 0.260 | 0.010 ± 0.006 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | - | 0.026 ± 0.006 | 0.318 ± 0.098 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | - | 0.036 ± 0.018 | 0.243 ± 0.091 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | - | 0.028 ± 0.017 | 0.208 ± 0.052 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.004 | 0.046 ± 0.030 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.003 | 0.039 ± 0.019 | 0.011 ± 0.004 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.007 | 0.039 ± 0.008 | 0.010 ± 0.008 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | - | 0.047 ± 0.019 | 0.161 ± 0.011 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | - | 0.044 ± 0.012 | 0.182 ± 0.092 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | - | 0.060 ± 0.016 | 0.437 ± 0.207 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | - | 0.042 ± 0.004 | 0.169 ± 0.027 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | - | 0.034 ± 0.030 | 0.176 ± 0.027 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | - | 0.045 ± 0.019 | 0.183 ± 0.032 | 0.012 ± 0.007 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.017 | 0.047 ± 0.003 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.021 | 0.053 ± 0.022 | 0.011 ± 0.005 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.004 | 0.066 ± 0.028 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.020 ± 0.006 | 0.257 ± 0.064 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.016 ± 0.010 | 0.169 ± 0.014 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.016 ± 0.006 | 0.333 ± 0.136 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.018 ± 0.004 | 0.097 ± 0.022 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.016 ± 0.012 | 0.084 ± 0.010 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.030 ± 0.014 | 0.079 ± 0.005 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.008 | 0.057 ± 0.015 | 0.012 ± 0.002 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.003 | 0.057 ± 0.014 | 0.019 ± 0.029 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.038 ± 0.009 | 0.055 ± 0.050 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | - | 0.021 ± 0.015 | 0.099 ± 0.012 | 0.013 ± 0.003 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | - | 0.018 ± 0.006 | 0.111 ± 0.049 | 0.015 ± 0.005 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | - | 0.041 ± 0.009 | 0.222 ± 0.110 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | - | 0.023 ± 0.003 | 0.067 ± 0.004 | 0.011 ± 0.009 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | - | 0.044 ± 0.036 | 0.079 ± 0.020 | 0.017 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | - | 0.029 ± 0.012 | 0.123 ± 0.024 | 0.018 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.003 ± 0.002 | 0.004 ± 0.002 | 0.001 ± 0.000 | 0.006 ± 0.004 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.003 ± 0.000 | 0.006 ± 0.010 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.003 ± 0.001 | 0.005 ± 0.002 | 0.001 ± 0.001 | 0.006 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.008 ± 0.001 | 0.008 ± 0.004 | 0.008 ± 0.003 | 0.043 ± 0.022 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.003 ± 0.001 | 0.004 ± 0.002 | 0.001 ± 0.001 | 0.007 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.005 ± 0.001 | 0.007 ± 0.003 | 0.032 ± 0.014 | 0.008 ± 0.003 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.004 ± 0.004 | 0.006 ± 0.004 | 0.032 ± 0.018 | 0.009 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.007 ± 0.002 | 0.007 ± 0.001 | 0.038 ± 0.003 | 0.008 ± 0.003 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.004 ± 0.000 | 0.009 ± 0.006 | 0.011 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.004 ± 0.002 | 0.008 ± 0.004 | 0.011 ± 0.011 | 0.006 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.005 ± 0.002 | 0.007 ± 0.005 | 0.010 ± 0.001 | 0.008 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.004 ± 0.001 | 0.006 ± 0.001 | 0.011 ± 0.002 | 0.007 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.007 ± 0.002 | 0.008 ± 0.004 | 0.011 ± 0.004 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.007 ± 0.001 | 0.007 ± 0.002 | 0.011 ± 0.004 | 0.006 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.006 ± 0.002 | 0.007 ± 0.002 | 0.010 ± 0.006 | 0.015 ± 0.004 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.010 ± 0.005 | 0.013 ± 0.003 | 0.008 ± 0.003 | 0.018 ± 0.007 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.025 ± 0.012 | 0.022 ± 0.009 | 0.020 ± 0.012 | 0.022 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 24.0x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 22.5x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 17.5x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 26.2x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 19.9x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 25.6x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 31.6x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 34.1x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 54.9x faster than the slowest successful cell (`pytorch-cpu`, 0.375 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.375 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 40.3x faster than the slowest successful cell (`pytorch-cpu`, 0.303 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.303 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 28.0x faster than the slowest successful cell (`pytorch-cpu`, 0.268 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 43.2x faster than the slowest successful cell (`pytorch-cpu`, 0.318 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.318 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 25.8x faster than the slowest successful cell (`pytorch-cpu`, 0.243 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 21.9x faster than the slowest successful cell (`pytorch-cpu`, 0.208 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 0.182 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 87.4x faster than the slowest successful cell (`pytorch-cpu`, 0.437 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 18.0x faster than the slowest successful cell (`pytorch-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.183 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 38.2x faster than the slowest successful cell (`pytorch-cpu`, 0.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 18.0x faster than the slowest successful cell (`pytorch-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 41.7x faster than the slowest successful cell (`pytorch-cpu`, 0.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 20.9x faster than the slowest successful cell (`pytorch-cpu`, 0.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.222 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

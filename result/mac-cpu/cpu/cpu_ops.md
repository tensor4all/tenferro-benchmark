# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `15cf8f674203aeca079999fdb339cf4d35801b88`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261110_172255/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261110_172255/run.yaml`
- Timestamp: `2026-11-10T17:22:55.246162+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261110_172255`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Apple M2 Pro (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `5`
- Physical CPUs: `5`
- Sockets: `1`
- Cores per socket: `5`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 5 physical / 5 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 5 CPUs/L2)`
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


- CSV: `data/results/mac-cpu/cpu/einsum/20261110_172255/cpu_ops_t1_20261110_172255.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.015 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.016 ± 0.002 | 0.018 ± 0.001 | 0.020 ± 0.001 | 0.043 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.024 ± 0.001 | 0.021 ± 0.001 | 0.026 ± 0.004 | 0.033 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.086 ± 0.007 | 0.073 ± 0.001 | 0.077 ± 0.013 | 0.108 ± 0.019 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.078 ± 0.020 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | 0.075 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.042 ± 0.002 | 0.050 ± 0.000 | 0.041 ± 0.014 | 0.105 ± 0.020 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.001 | 0.075 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.003 | 0.094 ± 0.026 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.011 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.014 ± 0.001 | 0.016 ± 0.000 | 0.019 ± 0.002 | 0.023 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.012 ± 0.001 | 0.011 ± 0.001 | 0.013 ± 0.001 | 0.017 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.041 ± 0.001 | 0.036 ± 0.001 | 0.042 ± 0.001 | 0.042 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.011 ± 0.001 | 0.008 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.001 | 0.011 ± 0.000 | 0.017 ± 0.002 | 0.013 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.009 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.013 ± 0.001 | 0.014 ± 0.000 | 0.018 ± 0.002 | 0.021 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.003 | 0.045 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.039 ± 0.004 | 0.039 ± 0.003 | 0.044 ± 0.009 | 0.133 ± 0.025 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.042 ± 0.002 | 0.040 ± 0.001 | 0.046 ± 0.003 | 0.068 ± 0.014 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.178 ± 0.004 | 0.161 ± 0.007 | 0.173 ± 0.009 | 0.235 ± 0.063 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.008 ± 0.001 | 0.059 ± 0.004 | 0.007 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.052 ± 0.004 | 0.007 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.057 ± 0.006 | 0.010 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.000 | 0.064 ± 0.006 | 0.026 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.028 ± 0.001 | 0.045 ± 0.004 | 0.011 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.000 | 0.054 ± 0.010 | 0.021 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.031 ± 0.001 | 0.046 ± 0.003 | 0.011 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.059 ± 0.004 | 0.056 ± 0.009 | 0.026 ± 0.003 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.260 ± 0.007 | 0.254 ± 0.007 | 0.295 ± 0.031 | 0.300 ± 0.003 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 9.201 ± 0.533 | 7.808 ± 0.206 | 5.233 ± 0.211 | 6.337 ± 0.441 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 45.386 ± 2.892 | 38.826 ± 4.464 | 27.548 ± 3.633 | 32.064 ± 3.514 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 4.617 ± 0.107 | 7.430 ± 0.289 | 5.279 ± 1.367 | 6.931 ± 0.627 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 23.222 ± 2.112 | 42.177 ± 1.944 | 22.544 ± 3.671 | 34.707 ± 5.990 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 2.232 ± 0.058 | 0.994 ± 0.032 | 1.792 ± 0.190 | 2.633 ± 0.452 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 13.226 ± 0.296 | 6.130 ± 0.079 | 11.965 ± 1.432 | 15.712 ± 1.221 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 1.693 ± 0.135 | 1.168 ± 0.048 | 1.075 ± 0.070 | 2.437 ± 0.219 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 11.669 ± 0.494 | 7.155 ± 0.226 | 7.569 ± 0.838 | 14.605 ± 0.371 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.013 ± 0.003 | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.029 ± 0.003 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.000 | 0.046 ± 0.004 | 0.071 ± 0.006 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 4.457 ± 0.053 | 3.072 ± 0.123 | 2.600 ± 0.895 | 4.667 ± 0.275 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 22.669 ± 0.506 | 16.886 ± 1.150 | 14.637 ± 0.443 | 26.318 ± 1.543 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 2.787 ± 0.111 | 3.072 ± 0.142 | 2.781 ± 0.405 | 4.650 ± 0.121 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 18.851 ± 0.424 | 17.942 ± 0.812 | 13.181 ± 0.600 | 26.846 ± 1.593 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.078 ± 0.003 | 0.071 ± 0.012 | 0.038 ± 0.003 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 0.462 ± 0.008 | 0.478 ± 0.015 | 0.471 ± 0.052 | 0.275 ± 0.009 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 2.738 ± 0.267 | 2.048 ± 0.053 | 1.654 ± 0.083 | 1.345 ± 0.243 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.259 ± 0.012 | 0.493 ± 0.016 | 0.558 ± 0.117 | 0.282 ± 0.028 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 1.137 ± 0.048 | 2.297 ± 0.171 | 2.503 ± 0.035 | 1.152 ± 0.151 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.558 ± 0.027 | 0.406 ± 0.067 | 0.483 ± 0.016 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 14.755 ± 0.389 | 11.114 ± 0.362 | 9.519 ± 0.904 | 10.159 ± 0.662 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 71.647 ± 3.067 | 55.893 ± 3.957 | 43.659 ± 6.418 | 49.700 ± 2.909 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 7.200 ± 0.399 | 11.334 ± 0.189 | 8.137 ± 0.256 | 9.854 ± 0.458 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 38.668 ± 1.132 | 58.083 ± 0.629 | 38.835 ± 4.044 | 50.709 ± 0.970 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.018 ± 0.002 | 0.017 ± 0.001 | 0.032 ± 0.009 | 0.114 ± 0.010 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.112 ± 0.007 | 0.108 ± 0.003 | 0.158 ± 0.011 | 0.814 ± 0.105 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.421 ± 0.097 | 0.445 ± 0.022 | 0.585 ± 0.027 | 3.485 ± 0.555 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.058 ± 0.005 | 0.055 ± 0.003 | 0.068 ± 0.001 | 0.072 ± 0.006 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.018 ± 0.001 | 0.020 ± 0.002 | 0.027 ± 0.003 | 0.031 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.019 ± 0.002 | 0.024 ± 0.001 | 0.032 ± 0.002 | 0.046 ± 0.009 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.026 ± 0.001 | 0.032 ± 0.001 | 0.038 ± 0.010 | 0.124 ± 0.097 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.400 ± 0.014 | 0.396 ± 0.006 | 0.434 ± 0.010 | 0.416 ± 0.022 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.008 ± 0.001 | 0.011 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.058 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.060 ± 0.005 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.107 ± 0.034 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.089 ± 0.007 | 0.014 ± 0.001 | 0.062 ± 0.004 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.092 ± 0.002 | 0.017 ± 0.000 | 0.068 ± 0.011 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.117 ± 0.006 | 0.023 ± 0.001 | 0.082 ± 0.006 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.029 ± 0.003 | 0.015 ± 0.004 | 0.049 ± 0.004 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.025 ± 0.001 | 0.017 ± 0.000 | 0.052 ± 0.005 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.034 ± 0.000 | 0.019 ± 0.000 | 0.070 ± 0.006 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.335 ± 0.025 | 0.018 ± 0.001 | 0.108 ± 0.005 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.280 ± 0.004 | 0.016 ± 0.002 | 0.103 ± 0.008 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.325 ± 0.030 | 0.016 ± 0.001 | 0.123 ± 0.011 | 0.007 ± 0.002 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.074 ± 0.003 | 0.017 ± 0.000 | 0.082 ± 0.004 | 0.005 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.060 ± 0.000 | 0.014 ± 0.001 | 0.080 ± 0.008 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.077 ± 0.007 | 0.020 ± 0.000 | 0.088 ± 0.007 | 0.008 ± 0.005 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.001 | 0.020 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.000 | 0.025 ± 0.003 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.000 | 0.028 ± 0.002 | 0.006 ± 0.003 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.485 ± 0.005 | 0.023 ± 0.001 | 0.084 ± 0.030 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.657 ± 0.171 | 0.024 ± 0.001 | 0.080 ± 0.015 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.458 ± 0.015 | 0.027 ± 0.003 | 0.087 ± 0.006 | 0.007 ± 0.003 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.060 ± 0.005 | 0.023 ± 0.001 | 0.063 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.059 ± 0.006 | 0.022 ± 0.002 | 0.064 ± 0.002 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.073 ± 0.003 | 0.031 ± 0.000 | 0.085 ± 0.009 | 0.007 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.001 | 0.036 ± 0.004 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.000 | 0.040 ± 0.003 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.001 | 0.041 ± 0.002 | 0.007 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.065 ± 0.003 | 0.013 ± 0.001 | 0.161 ± 0.020 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.056 ± 0.012 | 0.011 ± 0.000 | 0.155 ± 0.021 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.069 ± 0.006 | 0.014 ± 0.000 | 0.143 ± 0.022 | 0.008 ± 0.004 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.029 ± 0.002 | 0.013 ± 0.000 | 0.074 ± 0.017 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.025 ± 0.000 | 0.011 ± 0.000 | 0.064 ± 0.005 | 0.005 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.030 ± 0.002 | 0.013 ± 0.001 | 0.067 ± 0.007 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.024 ± 0.003 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.026 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.001 | 0.040 ± 0.002 | 0.019 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.084 ± 0.008 | 0.010 ± 0.002 | 0.095 ± 0.008 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.092 ± 0.026 | 0.014 ± 0.001 | 0.103 ± 0.021 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.109 ± 0.009 | 0.027 ± 0.000 | 0.108 ± 0.005 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.027 ± 0.001 | 0.009 ± 0.001 | 0.081 ± 0.023 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.034 ± 0.002 | 0.014 ± 0.001 | 0.053 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.036 ± 0.002 | 0.021 ± 0.001 | 0.069 ± 0.003 | 0.018 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.004 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.025 ± 0.004 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.002 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.005 ± 0.002 | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.008 ± 0.002 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.006 ± 0.002 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.003 ± 0.001 | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.005 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.012 ± 0.002 | 0.011 ± 0.001 | 0.011 ± 0.001 | 0.020 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 36.3x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 28.2x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 17.5x faster than the slowest successful cell (`jax-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 26.4x faster than the slowest successful cell (`jax-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 24.1x faster than the slowest successful cell (`jax-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 18.6x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 20.6x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 27.7x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 28.1x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 52.8x faster than the slowest successful cell (`jax-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 21.8x faster than the slowest successful cell (`jax-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 12.5x faster than the slowest successful cell (`tenferro-eager`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 64.7x faster than the slowest successful cell (`tenferro-eager`, 0.335 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 18.6x faster than the slowest successful cell (`tenferro-eager`, 0.335 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 50.4x faster than the slowest successful cell (`tenferro-eager`, 0.280 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 18.0x faster than the slowest successful cell (`tenferro-eager`, 0.280 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 44.6x faster than the slowest successful cell (`tenferro-eager`, 0.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 20.3x faster than the slowest successful cell (`tenferro-eager`, 0.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 75.9x faster than the slowest successful cell (`tenferro-eager`, 0.485 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 20.9x faster than the slowest successful cell (`tenferro-eager`, 0.485 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 97.2x faster than the slowest successful cell (`tenferro-eager`, 0.657 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 27.4x faster than the slowest successful cell (`tenferro-eager`, 0.657 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 67.7x faster than the slowest successful cell (`tenferro-eager`, 0.458 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`tenferro-eager`, 0.458 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 24.7x faster than the slowest successful cell (`pytorch-cpu`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 24.1x faster than the slowest successful cell (`pytorch-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261110_173954/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261110_173954/run.yaml`
- Timestamp: `2026-11-10T17:39:54.268002+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261110_173954`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Apple M2 Pro (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `5`
- Physical CPUs: `5`
- Sockets: `1`
- Cores per socket: `5`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 5 physical / 5 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 5 CPUs/L2)`
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


- CSV: `data/results/mac-cpu/cpu/einsum/20261110_173954/cpu_ops_t4_20261110_173954.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.011 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.019 ± 0.001 | 0.015 ± 0.000 | 0.017 ± 0.003 | 0.042 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.024 ± 0.003 | 0.023 ± 0.001 | 0.020 ± 0.001 | 0.027 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.079 ± 0.010 | 0.081 ± 0.005 | 0.071 ± 0.004 | 0.106 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.001 | 0.072 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.072 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.048 ± 0.004 | 0.056 ± 0.005 | 0.062 ± 0.005 | 0.168 ± 0.015 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.001 | 0.077 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.084 ± 0.009 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.001 | 0.006 ± 0.000 | 0.035 ± 0.006 | 0.009 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.017 ± 0.003 | 0.013 ± 0.001 | 0.067 ± 0.003 | 0.021 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.011 ± 0.002 | 0.011 ± 0.001 | 0.041 ± 0.011 | 0.014 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.036 ± 0.002 | 0.030 ± 0.000 | 0.074 ± 0.006 | 0.052 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.045 ± 0.008 | 0.006 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.012 ± 0.000 | 0.010 ± 0.001 | 0.048 ± 0.002 | 0.012 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.038 ± 0.002 | 0.009 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.015 ± 0.001 | 0.012 ± 0.000 | 0.051 ± 0.007 | 0.023 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.013 ± 0.001 | 0.012 ± 0.001 | 0.012 ± 0.001 | 0.032 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.038 ± 0.002 | 0.034 ± 0.000 | 0.039 ± 0.006 | 0.107 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.040 ± 0.001 | 0.042 ± 0.002 | 0.040 ± 0.005 | 0.056 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.167 ± 0.010 | 0.139 ± 0.002 | 0.144 ± 0.037 | 0.237 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.011 ± 0.000 | 0.065 ± 0.005 | 0.005 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.062 ± 0.006 | 0.008 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.001 | 0.081 ± 0.003 | 0.012 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.075 ± 0.005 | 0.049 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.036 ± 0.001 | 0.096 ± 0.009 | 0.011 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.051 ± 0.001 | 0.088 ± 0.001 | 0.017 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.036 ± 0.004 | 0.113 ± 0.008 | 0.008 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.056 ± 0.000 | 0.103 ± 0.022 | 0.032 ± 0.003 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.242 ± 0.010 | 0.260 ± 0.020 | 0.275 ± 0.061 | 0.259 ± 0.040 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 10.543 ± 0.386 | 6.790 ± 0.094 | 4.987 ± 0.604 | 5.374 ± 1.059 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 47.854 ± 2.325 | 46.729 ± 2.183 | 24.518 ± 2.626 | 23.298 ± 3.389 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 5.536 ± 0.121 | 7.706 ± 0.345 | 4.856 ± 0.107 | 5.343 ± 0.678 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 24.163 ± 1.990 | 47.459 ± 2.463 | 21.463 ± 1.734 | 24.526 ± 3.343 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 2.455 ± 0.279 | 0.903 ± 0.046 | 2.380 ± 0.163 | 1.672 ± 0.293 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 12.373 ± 0.650 | 6.297 ± 0.289 | 11.909 ± 1.070 | 6.808 ± 1.166 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 2.236 ± 0.059 | 1.146 ± 0.127 | 1.603 ± 0.076 | 1.579 ± 0.453 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 11.382 ± 0.316 | 7.665 ± 0.502 | 6.614 ± 0.694 | 7.912 ± 2.435 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.015 ± 0.001 | 0.008 ± 0.000 | 0.009 ± 0.001 | 0.033 ± 0.004 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.001 | 0.040 ± 0.001 | 0.071 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 4.477 ± 0.372 | 2.874 ± 0.190 | 3.125 ± 0.479 | 2.519 ± 0.706 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 25.077 ± 0.883 | 16.075 ± 0.244 | 12.425 ± 1.211 | 12.562 ± 0.971 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 2.904 ± 0.313 | 2.865 ± 0.093 | 3.028 ± 0.275 | 3.171 ± 0.332 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 17.182 ± 0.942 | 15.956 ± 0.156 | 10.884 ± 0.424 | 11.722 ± 0.414 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.080 ± 0.001 | 0.070 ± 0.013 | 0.039 ± 0.010 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.665 ± 0.037 | 0.422 ± 0.038 | 0.511 ± 0.031 | 0.297 ± 0.003 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 2.341 ± 0.104 | 2.158 ± 0.070 | 1.579 ± 0.139 | 1.414 ± 0.204 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.408 ± 0.014 | 0.476 ± 0.038 | 0.669 ± 0.042 | 0.308 ± 0.025 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 1.464 ± 0.074 | 2.499 ± 0.675 | 2.092 ± 0.119 | 1.449 ± 0.095 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.568 ± 0.013 | 0.381 ± 0.024 | 0.464 ± 0.046 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 15.611 ± 0.306 | 10.283 ± 0.058 | 9.752 ± 0.176 | 7.543 ± 1.866 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 76.308 ± 12.169 | 57.864 ± 1.753 | 42.010 ± 1.816 | 36.473 ± 5.109 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 7.055 ± 0.222 | 11.398 ± 1.430 | 8.302 ± 0.406 | 7.287 ± 0.091 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 42.263 ± 5.032 | 56.322 ± 2.841 | 38.965 ± 1.233 | 39.991 ± 8.329 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.024 ± 0.001 | 0.016 ± 0.000 | 0.026 ± 0.003 | 0.092 ± 0.021 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.120 ± 0.002 | 0.097 ± 0.001 | 0.140 ± 0.031 | 0.333 ± 0.062 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.441 ± 0.027 | 0.386 ± 0.010 | 0.569 ± 0.039 | 1.286 ± 0.206 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.060 ± 0.003 | 0.053 ± 0.005 | 0.106 ± 0.017 | 0.072 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.018 ± 0.000 | 0.024 ± 0.004 | 0.028 ± 0.004 | 0.036 ± 0.008 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.021 ± 0.001 | 0.027 ± 0.001 | 0.033 ± 0.003 | 0.054 ± 0.011 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.025 ± 0.001 | 0.033 ± 0.001 | 0.032 ± 0.007 | 0.078 ± 0.008 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.434 ± 0.012 | 0.391 ± 0.039 | 0.371 ± 0.042 | 0.370 ± 0.019 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.005 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.006 ± 0.002 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.007 ± 0.001 | 0.007 ± 0.001 | 0.007 ± 0.001 | 0.009 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.055 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.056 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.079 ± 0.011 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.085 ± 0.001 | 0.021 ± 0.002 | 0.061 ± 0.011 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.101 ± 0.006 | 0.021 ± 0.001 | 0.058 ± 0.001 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.102 ± 0.001 | 0.028 ± 0.001 | 0.076 ± 0.010 | 0.011 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.030 ± 0.000 | 0.018 ± 0.002 | 0.047 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.035 ± 0.001 | 0.021 ± 0.001 | 0.047 ± 0.001 | 0.007 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.042 ± 0.000 | 0.022 ± 0.000 | 0.067 ± 0.004 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.293 ± 0.001 | 0.023 ± 0.002 | 0.284 ± 0.046 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.327 ± 0.008 | 0.024 ± 0.002 | 0.269 ± 0.015 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.333 ± 0.043 | 0.021 ± 0.000 | 0.374 ± 0.032 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.079 ± 0.001 | 0.021 ± 0.001 | 0.200 ± 0.031 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.104 ± 0.002 | 0.021 ± 0.002 | 0.195 ± 0.008 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.092 ± 0.013 | 0.024 ± 0.000 | 0.284 ± 0.016 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.008 ± 0.001 | 0.021 ± 0.001 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.000 | 0.020 ± 0.000 | 0.006 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.011 ± 0.001 | 0.025 ± 0.001 | 0.005 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.485 ± 0.015 | 0.032 ± 0.003 | 0.115 ± 0.007 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.448 ± 0.038 | 0.030 ± 0.001 | 0.201 ± 0.017 | 0.007 ± 0.003 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.538 ± 0.053 | 0.032 ± 0.001 | 0.124 ± 0.008 | 0.005 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.070 ± 0.000 | 0.034 ± 0.002 | 0.172 ± 0.075 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.081 ± 0.005 | 0.035 ± 0.002 | 0.213 ± 0.021 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.095 ± 0.009 | 0.045 ± 0.001 | 0.159 ± 0.078 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.006 | 0.032 ± 0.002 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.001 | 0.032 ± 0.003 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.002 | 0.032 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.071 ± 0.008 | 0.018 ± 0.001 | 0.129 ± 0.010 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.075 ± 0.002 | 0.018 ± 0.002 | 0.126 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.062 ± 0.007 | 0.016 ± 0.000 | 0.161 ± 0.008 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.036 ± 0.003 | 0.017 ± 0.000 | 0.061 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.031 ± 0.000 | 0.016 ± 0.001 | 0.060 ± 0.000 | 0.005 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.036 ± 0.001 | 0.016 ± 0.000 | 0.075 ± 0.004 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.001 | 0.025 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.002 | 0.026 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.001 | 0.031 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.094 ± 0.008 | 0.013 ± 0.001 | 0.108 ± 0.007 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.082 ± 0.011 | 0.016 ± 0.000 | 0.107 ± 0.008 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.094 ± 0.001 | 0.032 ± 0.001 | 0.098 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.035 ± 0.002 | 0.012 ± 0.001 | 0.065 ± 0.008 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.031 ± 0.005 | 0.016 ± 0.002 | 0.065 ± 0.005 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.044 ± 0.010 | 0.026 ± 0.002 | 0.057 ± 0.002 | 0.016 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.002 ± 0.001 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.001 | 0.024 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.029 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.036 ± 0.007 | 0.005 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.004 ± 0.001 | 0.003 ± 0.000 | 0.030 ± 0.002 | 0.006 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.003 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.005 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.002 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.007 ± 0.001 | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.011 ± 0.002 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | 0.018 ± 0.007 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 23.5x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 23.2x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 15.7x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 17.8x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 25.7x faster than the slowest successful cell (`jax-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 22.3x faster than the slowest successful cell (`jax-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 24.4x faster than the slowest successful cell (`jax-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 19.7x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 29.2x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 13.7x faster than the slowest successful cell (`tenferro-eager`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 13.6x faster than the slowest successful cell (`tenferro-eager`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 60.1x faster than the slowest successful cell (`tenferro-eager`, 0.293 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 0.293 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 40.9x faster than the slowest successful cell (`tenferro-eager`, 0.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`tenferro-eager`, 0.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 48.1x faster than the slowest successful cell (`pytorch-cpu`, 0.374 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 17.4x faster than the slowest successful cell (`pytorch-cpu`, 0.374 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 40.6x faster than the slowest successful cell (`pytorch-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 32.3x faster than the slowest successful cell (`pytorch-cpu`, 0.195 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 49.5x faster than the slowest successful cell (`pytorch-cpu`, 0.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 70.5x faster than the slowest successful cell (`tenferro-eager`, 0.485 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`tenferro-eager`, 0.485 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 62.6x faster than the slowest successful cell (`tenferro-eager`, 0.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`tenferro-eager`, 0.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 106.4x faster than the slowest successful cell (`tenferro-eager`, 0.538 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 0.538 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 24.2x faster than the slowest successful cell (`pytorch-cpu`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 33.7x faster than the slowest successful cell (`pytorch-cpu`, 0.213 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 28.5x faster than the slowest successful cell (`pytorch-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 26.6x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 25.7x faster than the slowest successful cell (`pytorch-cpu`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 28.9x faster than the slowest successful cell (`pytorch-cpu`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

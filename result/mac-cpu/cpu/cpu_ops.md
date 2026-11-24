# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Benchmark commit: `9185e8ca26c3fc600458361d0f477495a717daf3`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261124_172410/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261124_172410/run.yaml`
- Timestamp: `2026-11-24T17:24:10.690570+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261124_172410`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20261124_172410/cpu_ops_t1_20261124_172410.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.020 ± 0.000 | 0.020 ± 0.001 | 0.020 ± 0.003 | 0.043 ± 0.013 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.023 ± 0.001 | 0.022 ± 0.001 | 0.025 ± 0.001 | 0.034 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.091 ± 0.019 | 0.085 ± 0.008 | 0.093 ± 0.066 | 0.113 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.078 ± 0.022 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.004 | 0.092 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.041 ± 0.002 | 0.040 ± 0.002 | 0.040 ± 0.007 | 0.105 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.001 | 0.095 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.094 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.023 ± 0.015 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.015 ± 0.001 | 0.014 ± 0.001 | 0.019 ± 0.006 | 0.027 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.018 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.028 ± 0.001 | 0.027 ± 0.001 | 0.039 ± 0.008 | 0.051 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.005 ± 0.002 | 0.011 ± 0.000 | 0.010 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.006 | 0.017 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.013 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.003 | 0.020 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.019 ± 0.010 | 0.055 ± 0.031 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.037 ± 0.000 | 0.037 ± 0.001 | 0.045 ± 0.011 | 0.140 ± 0.025 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.037 ± 0.001 | 0.037 ± 0.001 | 0.047 ± 0.002 | 0.074 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.143 ± 0.001 | 0.142 ± 0.012 | 0.170 ± 0.010 | 0.254 ± 0.023 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.035 ± 0.001 | 0.009 ± 0.000 | 0.056 ± 0.022 | 0.008 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.042 ± 0.006 | 0.015 ± 0.003 | 0.089 ± 0.063 | 0.010 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.041 ± 0.004 | 0.010 ± 0.000 | 0.076 ± 0.026 | 0.015 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.045 ± 0.001 | 0.017 ± 0.000 | 0.067 ± 0.009 | 0.032 ± 0.021 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.046 ± 0.001 | 0.029 ± 0.001 | 0.048 ± 0.002 | 0.015 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.051 ± 0.001 | 0.047 ± 0.002 | 0.062 ± 0.013 | 0.022 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.048 ± 0.001 | 0.032 ± 0.000 | 0.084 ± 0.035 | 0.023 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.058 ± 0.000 | 0.062 ± 0.002 | 0.062 ± 0.012 | 0.028 ± 0.004 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.275 ± 0.003 | 0.265 ± 0.005 | 0.403 ± 0.130 | 0.354 ± 0.064 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 15.392 ± 0.401 | 11.764 ± 0.307 | 5.640 ± 0.427 | 6.737 ± 0.388 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 87.331 ± 4.772 | 74.686 ± 10.132 | 39.605 ± 4.585 | 42.608 ± 2.021 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 8.778 ± 0.257 | 13.187 ± 3.405 | 5.193 ± 0.462 | 7.066 ± 0.559 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 61.626 ± 16.545 | 63.795 ± 1.898 | 32.961 ± 3.555 | 40.389 ± 2.090 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 10.580 ± 0.742 | 5.409 ± 0.555 | 2.629 ± 0.913 | 2.837 ± 0.221 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 92.454 ± 18.485 | 36.482 ± 1.753 | 19.761 ± 5.723 | 20.739 ± 1.578 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 9.659 ± 0.511 | 5.550 ± 2.464 | 2.201 ± 0.965 | 2.815 ± 1.074 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 76.742 ± 19.231 | 36.626 ± 0.563 | 13.545 ± 5.166 | 20.108 ± 0.858 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.026 ± 0.001 | 0.019 ± 0.006 | 0.013 ± 0.002 | 0.034 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.091 ± 0.009 | 0.055 ± 0.003 | 0.052 ± 0.010 | 0.072 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 15.512 ± 2.046 | 8.322 ± 0.261 | 3.183 ± 1.457 | 4.721 ± 0.204 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 99.599 ± 1.673 | 66.065 ± 5.862 | 19.475 ± 7.620 | 31.960 ± 0.568 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 11.841 ± 5.430 | 8.173 ± 0.300 | 2.625 ± 0.285 | 4.661 ± 0.166 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 81.907 ± 1.528 | 62.676 ± 1.663 | 19.365 ± 2.674 | 31.259 ± 0.862 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.066 ± 0.023 | 0.081 ± 0.001 | 0.070 ± 0.005 | 0.061 ± 0.027 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.251 ± 0.116 | 1.117 ± 0.129 | 0.492 ± 0.039 | 0.300 ± 0.015 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 7.421 ± 0.463 | 6.892 ± 0.419 | 1.999 ± 0.273 | 1.802 ± 0.127 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.643 ± 0.012 | 1.022 ± 0.289 | 0.564 ± 0.065 | 0.357 ± 0.105 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 4.025 ± 1.729 | 6.558 ± 0.109 | 3.291 ± 0.394 | 1.562 ± 0.164 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.634 ± 0.006 | 0.866 ± 0.855 | 0.461 ± 0.024 | 0.700 ± 0.401 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 31.117 ± 8.124 | 21.940 ± 0.513 | 10.398 ± 0.495 | 11.135 ± 0.451 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 172.388 ± 3.481 | 145.910 ± 33.336 | 62.474 ± 1.411 | 68.352 ± 1.244 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 15.558 ± 0.532 | 22.200 ± 0.119 | 9.400 ± 0.509 | 11.188 ± 0.549 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 91.825 ± 0.698 | 147.098 ± 23.061 | 54.971 ± 2.893 | 67.593 ± 0.866 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.125 ± 0.009 | 0.115 ± 0.003 | 0.028 ± 0.006 | 0.132 ± 0.008 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.855 ± 0.050 | 0.847 ± 0.032 | 0.201 ± 0.017 | 0.891 ± 0.368 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 3.253 ± 0.176 | 3.424 ± 0.081 | 0.678 ± 0.100 | 3.539 ± 0.181 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.091 ± 0.001 | 0.091 ± 0.002 | 0.086 ± 0.045 | 0.084 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.022 ± 0.001 | 0.021 ± 0.003 | 0.036 ± 0.007 | 0.043 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.028 ± 0.000 | 0.026 ± 0.000 | 0.046 ± 0.008 | 0.065 ± 0.012 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.056 ± 0.004 | 0.042 ± 0.000 | 0.065 ± 0.028 | 0.102 ± 0.018 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.570 ± 0.035 | 0.576 ± 0.021 | 0.476 ± 0.061 | 0.454 ± 0.033 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.001 | 0.004 ± 0.000 | 0.009 ± 0.005 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.005 ± 0.002 | 0.003 ± 0.000 | 0.005 ± 0.004 | 0.010 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.006 ± 0.003 | 0.008 ± 0.001 | 0.072 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.004 ± 0.002 | 0.005 ± 0.000 | 0.008 ± 0.003 | 0.072 ± 0.007 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.095 ± 0.006 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.186 ± 0.155 | 0.017 ± 0.001 | 0.071 ± 0.018 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.136 ± 0.052 | 0.019 ± 0.001 | 0.074 ± 0.029 | 0.011 ± 0.006 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.136 ± 0.153 | 0.025 ± 0.001 | 0.085 ± 0.030 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.062 ± 0.019 | 0.016 ± 0.000 | 0.061 ± 0.007 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.066 ± 0.046 | 0.017 ± 0.002 | 0.060 ± 0.004 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.039 ± 0.004 | 0.025 ± 0.002 | 0.075 ± 0.005 | 0.013 ± 0.003 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.483 ± 0.572 | 0.019 ± 0.001 | 0.123 ± 0.013 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.398 ± 0.241 | 0.020 ± 0.008 | 0.104 ± 0.009 | 0.011 ± 0.004 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.355 ± 0.007 | 0.023 ± 0.001 | 0.119 ± 0.020 | 0.015 ± 0.012 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.088 ± 0.022 | 0.017 ± 0.000 | 0.080 ± 0.003 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.075 ± 0.002 | 0.018 ± 0.003 | 0.080 ± 0.004 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.078 ± 0.004 | 0.020 ± 0.000 | 0.117 ± 0.067 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.034 ± 0.010 | 0.007 ± 0.000 | 0.024 ± 0.004 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.103 ± 0.007 | 0.007 ± 0.000 | 0.034 ± 0.021 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.032 ± 0.007 | 0.007 ± 0.001 | 0.034 ± 0.003 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.568 ± 0.365 | 0.026 ± 0.000 | 0.078 ± 0.004 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.904 ± 0.591 | 0.032 ± 0.004 | 0.090 ± 0.030 | 0.010 ± 0.007 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.496 ± 0.010 | 0.028 ± 0.001 | 0.087 ± 0.024 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.108 ± 0.018 | 0.026 ± 0.002 | 0.077 ± 0.003 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.162 ± 0.061 | 0.027 ± 0.001 | 0.087 ± 0.067 | 0.011 ± 0.007 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.074 ± 0.029 | 0.028 ± 0.000 | 0.094 ± 0.032 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.042 ± 0.001 | 0.021 ± 0.001 | 0.039 ± 0.003 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.122 ± 0.004 | 0.021 ± 0.001 | 0.066 ± 0.028 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.046 ± 0.007 | 0.021 ± 0.004 | 0.046 ± 0.005 | 0.011 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.102 ± 0.036 | 0.014 ± 0.000 | 0.163 ± 0.016 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.075 ± 0.002 | 0.015 ± 0.000 | 0.204 ± 0.144 | 0.012 ± 0.008 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.073 ± 0.008 | 0.016 ± 0.002 | 0.174 ± 0.022 | 0.008 ± 0.003 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.052 ± 0.038 | 0.014 ± 0.001 | 0.079 ± 0.003 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.035 ± 0.004 | 0.015 ± 0.000 | 0.079 ± 0.015 | 0.012 ± 0.008 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.036 ± 0.002 | 0.017 ± 0.001 | 0.078 ± 0.004 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.037 ± 0.003 | 0.011 ± 0.001 | 0.030 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.112 ± 0.005 | 0.014 ± 0.000 | 0.077 ± 0.047 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.044 ± 0.001 | 0.022 ± 0.000 | 0.042 ± 0.002 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.089 ± 0.002 | 0.011 ± 0.000 | 0.102 ± 0.018 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.101 ± 0.090 | 0.014 ± 0.000 | 0.183 ± 0.084 | 0.020 ± 0.006 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.103 ± 0.005 | 0.023 ± 0.001 | 0.113 ± 0.008 | 0.018 ± 0.004 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.032 ± 0.004 | 0.011 ± 0.000 | 0.063 ± 0.008 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.041 ± 0.016 | 0.015 ± 0.002 | 0.074 ± 0.068 | 0.016 ± 0.010 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.041 ± 0.001 | 0.022 ± 0.001 | 0.075 ± 0.006 | 0.018 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.003 ± 0.001 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.021 ± 0.001 | 0.019 ± 0.001 | 0.009 ± 0.002 | 0.025 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.007 ± 0.005 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.013 ± 0.008 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.004 ± 0.002 | 0.002 ± 0.000 | 0.004 ± 0.002 | 0.008 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.003 ± 0.001 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.008 ± 0.002 | 0.008 ± 0.004 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.011 ± 0.007 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.004 ± 0.003 | 0.004 ± 0.000 | 0.013 ± 0.005 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.004 ± 0.004 | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.003 | 0.007 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.003 | 0.006 ± 0.000 | 0.017 ± 0.011 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.008 ± 0.002 | 0.005 ± 0.000 | 0.008 ± 0.004 | 0.015 ± 0.004 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.001 | 0.021 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 28.2x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 27.0x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 20.2x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 32.5x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 33.1x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 21.1x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 20.4x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 26.8x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 17.7x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 39.5x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 20.1x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 25.7x faster than the slowest successful cell (`tenferro-eager`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`tenferro-eager`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.2x faster than the slowest successful cell (`tenferro-eager`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 62.8x faster than the slowest successful cell (`tenferro-eager`, 0.483 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 25.9x faster than the slowest successful cell (`tenferro-eager`, 0.483 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 36.1x faster than the slowest successful cell (`tenferro-eager`, 0.398 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 19.7x faster than the slowest successful cell (`tenferro-eager`, 0.398 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 23.5x faster than the slowest successful cell (`tenferro-eager`, 0.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`tenferro-eager`, 0.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.5x faster than the slowest successful cell (`tenferro-eager`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.0x faster than the slowest successful cell (`tenferro-eager`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 69.6x faster than the slowest successful cell (`tenferro-eager`, 0.568 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 22.0x faster than the slowest successful cell (`tenferro-eager`, 0.568 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 92.4x faster than the slowest successful cell (`tenferro-eager`, 0.904 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 0.904 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 28.1x faster than the slowest successful cell (`tenferro-eager`, 0.904 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 46.8x faster than the slowest successful cell (`tenferro-eager`, 0.496 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 17.7x faster than the slowest successful cell (`tenferro-eager`, 0.496 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.6x faster than the slowest successful cell (`tenferro-eager`, 0.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 14.9x faster than the slowest successful cell (`tenferro-eager`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 12.2x faster than the slowest successful cell (`tenferro-eager`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 21.2x faster than the slowest successful cell (`pytorch-cpu`, 0.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 17.5x faster than the slowest successful cell (`pytorch-cpu`, 0.204 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 0.204 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 21.0x faster than the slowest successful cell (`pytorch-cpu`, 0.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.183 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261124_175807/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261124_175807/run.yaml`
- Timestamp: `2026-11-24T17:58:07.132443+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261124_175807`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20261124_175807/cpu_ops_t4_20261124_175807.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.003 | 0.016 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.021 ± 0.001 | 0.022 ± 0.005 | 0.020 ± 0.018 | 0.079 ± 0.020 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.025 ± 0.001 | 0.023 ± 0.000 | 0.027 ± 0.014 | 0.034 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.090 ± 0.002 | 0.131 ± 0.043 | 0.095 ± 0.014 | 0.125 ± 0.072 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.004 ± 0.003 | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.074 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.005 | 0.009 ± 0.004 | 0.095 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.042 ± 0.003 | 0.042 ± 0.002 | 0.100 ± 0.009 | 0.265 ± 0.088 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.097 ± 0.017 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.010 ± 0.001 | 0.086 ± 0.019 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.003 | 0.006 ± 0.000 | 0.043 ± 0.032 | 0.013 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.015 ± 0.000 | 0.018 ± 0.002 | 0.061 ± 0.035 | 0.028 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.049 ± 0.020 | 0.018 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.029 ± 0.001 | 0.029 ± 0.006 | 0.079 ± 0.020 | 0.082 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.040 ± 0.020 | 0.009 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.011 ± 0.003 | 0.016 ± 0.008 | 0.050 ± 0.020 | 0.033 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.059 ± 0.033 | 0.012 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.015 ± 0.001 | 0.015 ± 0.002 | 0.058 ± 0.050 | 0.053 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.015 ± 0.007 | 0.012 ± 0.001 | 0.015 ± 0.001 | 0.042 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.039 ± 0.000 | 0.042 ± 0.011 | 0.044 ± 0.006 | 0.139 ± 0.021 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.038 ± 0.001 | 0.038 ± 0.001 | 0.050 ± 0.013 | 0.075 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.146 ± 0.006 | 0.162 ± 0.020 | 0.169 ± 0.008 | 0.282 ± 0.035 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.044 ± 0.009 | 0.011 ± 0.001 | 0.059 ± 0.006 | 0.014 ± 0.013 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.048 ± 0.003 | 0.018 ± 0.010 | 0.061 ± 0.002 | 0.014 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.043 ± 0.002 | 0.012 ± 0.000 | 0.065 ± 0.041 | 0.030 ± 0.011 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.052 ± 0.002 | 0.023 ± 0.014 | 0.069 ± 0.008 | 0.094 ± 0.021 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.058 ± 0.004 | 0.040 ± 0.001 | 0.092 ± 0.041 | 0.020 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.063 ± 0.009 | 0.062 ± 0.039 | 0.103 ± 0.048 | 0.048 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.058 ± 0.001 | 0.043 ± 0.002 | 0.129 ± 0.048 | 0.027 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.068 ± 0.004 | 0.075 ± 0.018 | 0.216 ± 0.053 | 0.065 ± 0.014 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.267 ± 0.006 | 0.267 ± 0.012 | 0.315 ± 0.017 | 0.357 ± 0.024 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 17.746 ± 1.750 | 11.479 ± 0.215 | 5.800 ± 0.257 | 6.563 ± 0.391 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 83.438 ± 1.888 | 62.405 ± 1.056 | 33.783 ± 1.842 | 37.916 ± 2.696 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 9.138 ± 0.244 | 11.423 ± 0.165 | 5.413 ± 0.145 | 6.751 ± 3.226 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 46.830 ± 1.688 | 63.816 ± 2.099 | 30.266 ± 1.411 | 43.236 ± 11.168 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 10.304 ± 0.316 | 5.868 ± 0.225 | 3.481 ± 0.281 | 3.523 ± 1.396 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 69.559 ± 12.824 | 33.569 ± 4.396 | 18.489 ± 0.996 | 17.594 ± 5.130 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 9.405 ± 1.420 | 6.046 ± 0.428 | 2.143 ± 0.434 | 4.245 ± 2.460 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 60.698 ± 11.616 | 31.070 ± 1.451 | 10.759 ± 0.597 | 17.032 ± 3.917 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.054 ± 0.020 | 0.038 ± 0.007 | 0.011 ± 0.002 | 0.043 ± 0.015 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.296 ± 0.087 | 0.092 ± 0.030 | 0.054 ± 0.015 | 0.130 ± 0.048 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 13.335 ± 0.257 | 8.681 ± 3.064 | 5.000 ± 2.151 | 4.901 ± 1.951 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 101.932 ± 17.410 | 55.497 ± 21.920 | 16.935 ± 0.998 | 26.568 ± 4.195 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 10.372 ± 0.175 | 8.338 ± 0.313 | 7.511 ± 3.444 | 5.057 ± 1.148 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 64.881 ± 2.031 | 49.879 ± 1.354 | 16.299 ± 1.178 | 24.397 ± 4.869 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.078 ± 0.000 | 0.213 ± 0.056 | 0.074 ± 0.009 | 0.075 ± 0.018 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 2.122 ± 0.053 | 1.879 ± 0.123 | 0.607 ± 0.102 | 0.515 ± 0.284 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 9.586 ± 1.210 | 8.340 ± 1.817 | 2.328 ± 0.653 | 2.460 ± 0.712 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 1.492 ± 0.409 | 1.982 ± 0.140 | 0.758 ± 0.075 | 0.988 ± 0.259 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 4.849 ± 0.078 | 8.565 ± 0.507 | 3.927 ± 0.351 | 1.879 ± 0.198 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.865 ± 0.064 | 1.024 ± 0.310 | 0.491 ± 0.026 | 0.543 ± 0.033 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 29.460 ± 1.133 | 22.107 ± 1.097 | 11.314 ± 0.634 | 11.386 ± 1.653 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 180.388 ± 29.579 | 128.032 ± 11.317 | 62.603 ± 1.387 | 81.435 ± 9.488 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 15.375 ± 0.556 | 21.699 ± 0.546 | 10.051 ± 5.028 | 13.232 ± 1.677 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 92.907 ± 15.236 | 128.281 ± 3.072 | 57.318 ± 1.926 | 66.222 ± 5.725 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.162 ± 0.007 | 0.161 ± 0.007 | 0.030 ± 0.004 | 0.192 ± 0.013 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.731 ± 0.022 | 0.779 ± 0.104 | 0.241 ± 0.058 | 0.721 ± 0.239 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 2.525 ± 0.041 | 2.690 ± 0.159 | 0.794 ± 0.249 | 2.535 ± 0.646 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.096 ± 0.006 | 0.104 ± 0.006 | 0.102 ± 0.019 | 0.106 ± 0.007 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.023 ± 0.000 | 0.023 ± 0.001 | 0.036 ± 0.003 | 0.061 ± 0.005 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.029 ± 0.001 | 0.028 ± 0.000 | 0.044 ± 0.008 | 0.105 ± 0.030 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.060 ± 0.001 | 0.045 ± 0.000 | 0.077 ± 0.056 | 0.154 ± 0.095 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.583 ± 0.005 | 0.559 ± 0.114 | 0.423 ± 0.024 | 0.466 ± 0.028 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.004 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.008 ± 0.000 | 0.008 ± 0.004 | 0.009 ± 0.004 | 0.014 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.075 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.003 | 0.074 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.007 ± 0.005 | 0.009 ± 0.001 | 0.090 ± 0.010 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.114 ± 0.002 | 0.024 ± 0.001 | 0.072 ± 0.007 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.117 ± 0.007 | 0.040 ± 0.018 | 0.070 ± 0.002 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.136 ± 0.029 | 0.032 ± 0.001 | 0.087 ± 0.040 | 0.026 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.043 ± 0.001 | 0.023 ± 0.001 | 0.065 ± 0.017 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.044 ± 0.000 | 0.047 ± 0.017 | 0.060 ± 0.004 | 0.011 ± 0.005 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.050 ± 0.012 | 0.032 ± 0.001 | 0.091 ± 0.027 | 0.025 ± 0.004 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.382 ± 0.019 | 0.028 ± 0.001 | 0.363 ± 0.067 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.408 ± 0.037 | 0.058 ± 0.017 | 0.352 ± 0.030 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.387 ± 0.034 | 0.030 ± 0.001 | 0.400 ± 0.291 | 0.013 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.100 ± 0.004 | 0.025 ± 0.000 | 0.244 ± 0.064 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.103 ± 0.002 | 0.037 ± 0.010 | 0.261 ± 0.029 | 0.009 ± 0.004 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.104 ± 0.003 | 0.028 ± 0.002 | 0.291 ± 0.084 | 0.013 ± 0.004 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.048 ± 0.010 | 0.009 ± 0.000 | 0.025 ± 0.011 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.039 ± 0.001 | 0.009 ± 0.000 | 0.025 ± 0.001 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.040 ± 0.001 | 0.013 ± 0.009 | 0.039 ± 0.029 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.497 ± 0.021 | 0.038 ± 0.001 | 0.228 ± 0.117 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.493 ± 0.045 | 0.045 ± 0.030 | 0.134 ± 0.041 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.518 ± 0.072 | 0.041 ± 0.001 | 0.316 ± 0.200 | 0.017 ± 0.003 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.094 ± 0.001 | 0.037 ± 0.001 | 0.180 ± 0.131 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.095 ± 0.001 | 0.067 ± 0.028 | 0.147 ± 0.040 | 0.012 ± 0.010 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.119 ± 0.052 | 0.041 ± 0.000 | 0.228 ± 0.067 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.053 ± 0.001 | 0.032 ± 0.001 | 0.041 ± 0.003 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.052 ± 0.001 | 0.039 ± 0.018 | 0.041 ± 0.003 | 0.011 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.052 ± 0.004 | 0.033 ± 0.008 | 0.054 ± 0.019 | 0.012 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.080 ± 0.002 | 0.022 ± 0.000 | 0.164 ± 0.015 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.081 ± 0.005 | 0.023 ± 0.003 | 0.163 ± 0.011 | 0.013 ± 0.009 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.085 ± 0.001 | 0.023 ± 0.000 | 0.166 ± 0.023 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.047 ± 0.002 | 0.022 ± 0.001 | 0.083 ± 0.074 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.045 ± 0.002 | 0.023 ± 0.001 | 0.076 ± 0.008 | 0.010 ± 0.005 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.047 ± 0.001 | 0.023 ± 0.000 | 0.079 ± 0.009 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.045 ± 0.001 | 0.016 ± 0.000 | 0.031 ± 0.002 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.047 ± 0.000 | 0.018 ± 0.002 | 0.034 ± 0.002 | 0.015 ± 0.003 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.052 ± 0.001 | 0.039 ± 0.016 | 0.041 ± 0.003 | 0.045 ± 0.020 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.095 ± 0.002 | 0.017 ± 0.001 | 0.127 ± 0.097 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.098 ± 0.001 | 0.035 ± 0.009 | 0.095 ± 0.006 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.113 ± 0.003 | 0.029 ± 0.003 | 0.189 ± 0.053 | 0.046 ± 0.003 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.039 ± 0.001 | 0.016 ± 0.000 | 0.066 ± 0.020 | 0.011 ± 0.005 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.042 ± 0.002 | 0.023 ± 0.009 | 0.064 ± 0.003 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.066 ± 0.070 | 0.028 ± 0.008 | 0.082 ± 0.054 | 0.023 ± 0.017 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.007 ± 0.001 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.016 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.034 ± 0.003 | 0.025 ± 0.002 | 0.009 ± 0.002 | 0.060 ± 0.007 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.005 ± 0.003 | 0.001 ± 0.000 | 0.009 ± 0.005 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.005 ± 0.005 | 0.003 ± 0.000 | 0.043 ± 0.005 | 0.008 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.004 ± 0.001 | 0.003 ± 0.000 | 0.047 ± 0.006 | 0.009 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.004 ± 0.001 | 0.048 ± 0.005 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.009 ± 0.002 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.004 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.005 ± 0.000 | 0.010 ± 0.002 | 0.012 ± 0.001 | 0.007 ± 0.004 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.004 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.002 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.012 ± 0.000 | 0.011 ± 0.004 | 0.015 ± 0.001 | 0.022 ± 0.002 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 21.0x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 20.8x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 17.5x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 26.7x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 26.1x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 22.2x faster than the slowest successful cell (`jax-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 24.4x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 28.1x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 0.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.6x faster than the slowest successful cell (`tenferro-eager`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 48.5x faster than the slowest successful cell (`tenferro-eager`, 0.382 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`tenferro-eager`, 0.382 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 45.2x faster than the slowest successful cell (`tenferro-eager`, 0.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 30.0x faster than the slowest successful cell (`pytorch-cpu`, 0.400 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.400 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 31.9x faster than the slowest successful cell (`pytorch-cpu`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 28.6x faster than the slowest successful cell (`pytorch-cpu`, 0.261 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 21.7x faster than the slowest successful cell (`pytorch-cpu`, 0.291 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.291 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 60.7x faster than the slowest successful cell (`tenferro-eager`, 0.497 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`tenferro-eager`, 0.497 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 52.8x faster than the slowest successful cell (`tenferro-eager`, 0.493 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`tenferro-eager`, 0.493 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 31.0x faster than the slowest successful cell (`tenferro-eager`, 0.518 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 0.518 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 22.2x faster than the slowest successful cell (`pytorch-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.228 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 21.7x faster than the slowest successful cell (`pytorch-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 20.5x faster than the slowest successful cell (`pytorch-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

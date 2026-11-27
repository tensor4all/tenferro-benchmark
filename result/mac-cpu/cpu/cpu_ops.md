# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `7da876a214acb65eb9ea099adef67b41ef8efc98`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261127_172420/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261127_172420/run.yaml`
- Timestamp: `2026-11-27T17:24:20.827597+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261127_172420`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261127_172420/cpu_ops_t1_20261127_172420.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.010 ± 0.009 | 0.007 ± 0.000 | 0.028 ± 0.020 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.025 ± 0.002 | 0.025 ± 0.003 | 0.023 ± 0.004 | 0.043 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.027 ± 0.000 | 0.029 ± 0.009 | 0.026 ± 0.009 | 0.035 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.096 ± 0.007 | 0.105 ± 0.012 | 0.084 ± 0.010 | 0.109 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.002 ± 0.001 | 0.004 ± 0.002 | 0.008 ± 0.001 | 0.074 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.001 | 0.004 ± 0.002 | 0.009 ± 0.000 | 0.121 ± 0.112 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.040 ± 0.002 | 0.048 ± 0.031 | 0.037 ± 0.005 | 0.109 ± 0.021 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.001 | 0.003 ± 0.002 | 0.009 ± 0.001 | 0.095 ± 0.065 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.008 ± 0.004 | 0.010 ± 0.000 | 0.119 ± 0.154 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.010 ± 0.002 | 0.006 ± 0.001 | 0.032 ± 0.025 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.015 ± 0.001 | 0.014 ± 0.003 | 0.021 ± 0.007 | 0.028 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.010 ± 0.001 | 0.014 ± 0.001 | 0.011 ± 0.002 | 0.019 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.031 ± 0.003 | 0.040 ± 0.006 | 0.036 ± 0.001 | 0.052 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.001 | 0.009 ± 0.002 | 0.010 ± 0.004 | 0.010 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.012 ± 0.001 | 0.009 ± 0.001 | 0.017 ± 0.004 | 0.018 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.011 ± 0.005 | 0.011 ± 0.001 | 0.012 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.015 ± 0.000 | 0.015 ± 0.001 | 0.020 ± 0.001 | 0.020 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.017 ± 0.008 | 0.016 ± 0.018 | 0.014 ± 0.001 | 0.044 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.043 ± 0.002 | 0.053 ± 0.014 | 0.049 ± 0.024 | 0.181 ± 0.062 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.042 ± 0.007 | 0.042 ± 0.006 | 0.045 ± 0.001 | 0.077 ± 0.044 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.169 ± 0.034 | 0.206 ± 0.038 | 0.164 ± 0.012 | 0.311 ± 0.108 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.040 ± 0.005 | 0.013 ± 0.005 | 0.066 ± 0.015 | 0.009 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.046 ± 0.002 | 0.015 ± 0.003 | 0.058 ± 0.001 | 0.007 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.039 ± 0.005 | 0.014 ± 0.003 | 0.059 ± 0.006 | 0.015 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.050 ± 0.006 | 0.020 ± 0.000 | 0.063 ± 0.005 | 0.028 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.051 ± 0.006 | 0.041 ± 0.045 | 0.061 ± 0.014 | 0.018 ± 0.010 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.060 ± 0.004 | 0.051 ± 0.019 | 0.053 ± 0.002 | 0.020 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.059 ± 0.007 | 0.036 ± 0.004 | 0.049 ± 0.003 | 0.018 ± 0.012 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.071 ± 0.009 | 0.071 ± 0.014 | 0.063 ± 0.008 | 0.029 ± 0.004 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.400 ± 0.075 | 0.295 ± 0.128 | 0.324 ± 0.009 | 0.354 ± 0.006 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 17.521 ± 0.636 | 12.975 ± 1.910 | 6.051 ± 1.865 | 6.885 ± 0.269 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 83.653 ± 0.437 | 86.892 ± 6.576 | 26.823 ± 3.029 | 43.397 ± 1.072 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 8.801 ± 0.429 | 12.391 ± 4.770 | 6.599 ± 1.996 | 7.052 ± 0.179 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 47.669 ± 1.238 | 99.560 ± 27.620 | 26.650 ± 1.856 | 40.476 ± 3.874 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 10.920 ± 1.364 | 7.213 ± 2.147 | 2.459 ± 0.342 | 3.005 ± 0.181 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 73.664 ± 2.749 | 57.193 ± 9.592 | 15.371 ± 2.489 | 22.470 ± 2.241 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 9.762 ± 1.268 | 6.914 ± 1.960 | 1.191 ± 0.162 | 2.866 ± 0.225 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 67.272 ± 3.803 | 55.343 ± 3.812 | 9.104 ± 0.687 | 20.056 ± 1.548 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.028 ± 0.014 | 0.021 ± 0.009 | 0.013 ± 0.012 | 0.034 ± 0.005 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.100 ± 0.078 | 0.067 ± 0.026 | 0.054 ± 0.031 | 0.073 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 17.429 ± 3.681 | 11.299 ± 4.178 | 2.890 ± 0.553 | 4.813 ± 0.276 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 115.987 ± 29.366 | 83.371 ± 8.270 | 14.558 ± 2.547 | 33.950 ± 3.652 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 14.141 ± 1.798 | 10.219 ± 3.054 | 2.154 ± 0.157 | 4.785 ± 0.072 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 80.514 ± 1.677 | 77.974 ± 12.019 | 12.994 ± 1.623 | 33.230 ± 2.806 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.073 ± 0.016 | 0.088 ± 0.031 | 0.088 ± 0.024 | 0.056 ± 0.053 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.274 ± 0.293 | 1.457 ± 0.915 | 0.465 ± 0.081 | 0.328 ± 0.020 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 6.875 ± 0.255 | 8.206 ± 3.975 | 1.903 ± 0.277 | 2.333 ± 0.238 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.611 ± 0.029 | 1.199 ± 0.335 | 0.513 ± 0.050 | 0.364 ± 0.027 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 3.652 ± 0.185 | 10.070 ± 3.268 | 2.716 ± 0.623 | 1.732 ± 0.130 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.613 ± 0.420 | 0.868 ± 0.125 | 0.619 ± 0.278 | 0.539 ± 0.154 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 27.603 ± 0.923 | 22.182 ± 0.715 | 10.822 ± 1.702 | 13.227 ± 3.046 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 173.900 ± 50.585 | 141.394 ± 11.704 | 56.611 ± 6.873 | 69.978 ± 7.356 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 16.087 ± 0.511 | 26.622 ± 3.193 | 9.141 ± 0.414 | 13.983 ± 2.197 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 95.458 ± 7.905 | 144.072 ± 9.832 | 54.434 ± 4.777 | 85.483 ± 16.884 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.154 ± 0.053 | 0.179 ± 0.124 | 0.028 ± 0.002 | 0.135 ± 0.004 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.954 ± 0.047 | 0.985 ± 0.933 | 0.181 ± 0.015 | 1.019 ± 0.623 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 4.857 ± 1.374 | 5.395 ± 1.534 | 0.640 ± 0.018 | 3.516 ± 0.387 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.145 ± 0.064 | 0.111 ± 0.026 | 0.074 ± 0.004 | 0.084 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.032 ± 0.009 | 0.023 ± 0.004 | 0.036 ± 0.001 | 0.042 ± 0.007 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.030 ± 0.005 | 0.027 ± 0.001 | 0.049 ± 0.016 | 0.071 ± 0.006 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.061 ± 0.010 | 0.044 ± 0.000 | 0.071 ± 0.019 | 0.202 ± 0.076 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.582 ± 0.419 | 0.471 ± 0.075 | 0.427 ± 0.081 | 0.468 ± 0.022 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.008 ± 0.003 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.009 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.013 ± 0.004 | 0.007 ± 0.000 | 0.010 ± 0.003 | 0.014 ± 0.005 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.001 | 0.005 ± 0.000 | 0.009 ± 0.006 | 0.078 ± 0.009 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.004 ± 0.002 | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.074 ± 0.019 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.009 ± 0.002 | 0.141 ± 0.033 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.116 ± 0.041 | 0.016 ± 0.000 | 0.074 ± 0.010 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.121 ± 0.029 | 0.019 ± 0.001 | 0.097 ± 0.090 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.244 ± 0.074 | 0.026 ± 0.001 | 0.084 ± 0.008 | 0.013 ± 0.004 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.051 ± 0.033 | 0.016 ± 0.000 | 0.070 ± 0.013 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.040 ± 0.015 | 0.020 ± 0.005 | 0.070 ± 0.068 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.081 ± 0.028 | 0.027 ± 0.002 | 0.077 ± 0.004 | 0.011 ± 0.002 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.584 ± 0.118 | 0.018 ± 0.000 | 0.103 ± 0.017 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.435 ± 0.175 | 0.022 ± 0.003 | 0.112 ± 0.005 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.573 ± 0.180 | 0.022 ± 0.004 | 0.117 ± 0.007 | 0.010 ± 0.004 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.104 ± 0.036 | 0.016 ± 0.002 | 0.100 ± 0.017 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.085 ± 0.007 | 0.018 ± 0.001 | 0.103 ± 0.024 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.127 ± 0.077 | 0.020 ± 0.003 | 0.091 ± 0.011 | 0.008 ± 0.008 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.042 ± 0.011 | 0.006 ± 0.000 | 0.031 ± 0.031 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.038 ± 0.001 | 0.008 ± 0.001 | 0.027 ± 0.004 | 0.009 ± 0.010 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.036 ± 0.006 | 0.007 ± 0.002 | 0.042 ± 0.029 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.494 ± 0.032 | 0.025 ± 0.000 | 0.084 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.431 ± 0.024 | 0.027 ± 0.001 | 0.088 ± 0.015 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.798 ± 0.264 | 0.032 ± 0.004 | 0.112 ± 0.040 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.076 ± 0.002 | 0.025 ± 0.001 | 0.087 ± 0.010 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.073 ± 0.002 | 0.029 ± 0.002 | 0.082 ± 0.024 | 0.011 ± 0.009 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.099 ± 0.119 | 0.032 ± 0.005 | 0.100 ± 0.012 | 0.010 ± 0.006 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.053 ± 0.002 | 0.021 ± 0.000 | 0.086 ± 0.039 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.046 ± 0.005 | 0.023 ± 0.001 | 0.044 ± 0.011 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.046 ± 0.009 | 0.023 ± 0.002 | 0.084 ± 0.055 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.080 ± 0.023 | 0.015 ± 0.002 | 0.169 ± 0.008 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.080 ± 0.054 | 0.015 ± 0.001 | 0.167 ± 0.197 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.122 ± 0.070 | 0.020 ± 0.008 | 0.160 ± 0.013 | 0.008 ± 0.005 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.058 ± 0.045 | 0.014 ± 0.000 | 0.114 ± 0.153 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.058 ± 0.030 | 0.016 ± 0.000 | 0.087 ± 0.010 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.045 ± 0.026 | 0.026 ± 0.007 | 0.078 ± 0.017 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.230 ± 0.049 | 0.011 ± 0.000 | 0.041 ± 0.023 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.043 ± 0.004 | 0.014 ± 0.001 | 0.036 ± 0.007 | 0.014 ± 0.010 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.045 ± 0.008 | 0.025 ± 0.003 | 0.056 ± 0.029 | 0.020 ± 0.003 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.106 ± 0.005 | 0.012 ± 0.000 | 0.124 ± 0.100 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.112 ± 0.043 | 0.016 ± 0.002 | 0.114 ± 0.019 | 0.011 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.317 ± 0.086 | 0.027 ± 0.006 | 0.166 ± 0.066 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.041 ± 0.007 | 0.010 ± 0.000 | 0.116 ± 0.365 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.047 ± 0.026 | 0.016 ± 0.001 | 0.072 ± 0.112 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.072 ± 0.051 | 0.025 ± 0.003 | 0.114 ± 0.098 | 0.017 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.002 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.002 | 0.010 ± 0.004 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.001 | 0.003 ± 0.000 | 0.001 ± 0.001 | 0.005 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.023 ± 0.004 | 0.019 ± 0.003 | 0.007 ± 0.001 | 0.024 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.002 ± 0.001 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.012 ± 0.005 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.004 ± 0.002 | 0.002 ± 0.000 | 0.004 ± 0.003 | 0.009 ± 0.002 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.004 ± 0.004 | 0.003 ± 0.000 | 0.003 ± 0.001 | 0.008 ± 0.003 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.008 ± 0.004 | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.013 ± 0.005 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.003 | 0.009 ± 0.008 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.004 ± 0.002 | 0.005 ± 0.001 | 0.012 ± 0.005 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.007 ± 0.004 | 0.005 ± 0.001 | 0.010 ± 0.001 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.005 ± 0.003 | 0.005 ± 0.002 | 0.010 ± 0.002 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.009 ± 0.008 | 0.005 ± 0.000 | 0.010 ± 0.002 | 0.007 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.010 | 0.013 ± 0.008 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.008 ± 0.003 | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.014 ± 0.002 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.023 ± 0.010 | 0.010 ± 0.001 | 0.017 ± 0.004 | 0.022 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 30.1x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 19.0x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 14.1x faster than the slowest successful cell (`jax-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 29.5x faster than the slowest successful cell (`jax-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 29.3x faster than the slowest successful cell (`jax-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 30.9x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 30.5x faster than the slowest successful cell (`jax-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 23.0x faster than the slowest successful cell (`jax-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`jax-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 26.9x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 19.7x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 15.4x faster than the slowest successful cell (`jax-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 29.1x faster than the slowest successful cell (`jax-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 28.0x faster than the slowest successful cell (`jax-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.1x faster than the slowest successful cell (`tenferro-eager`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.9x faster than the slowest successful cell (`tenferro-eager`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 19.5x faster than the slowest successful cell (`tenferro-eager`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 77.1x faster than the slowest successful cell (`tenferro-eager`, 0.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 33.0x faster than the slowest successful cell (`tenferro-eager`, 0.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 52.0x faster than the slowest successful cell (`tenferro-eager`, 0.435 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 20.2x faster than the slowest successful cell (`tenferro-eager`, 0.435 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 57.9x faster than the slowest successful cell (`tenferro-eager`, 0.573 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 26.0x faster than the slowest successful cell (`tenferro-eager`, 0.573 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.8x faster than the slowest successful cell (`tenferro-eager`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 15.8x faster than the slowest successful cell (`tenferro-eager`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 61.7x faster than the slowest successful cell (`tenferro-eager`, 0.494 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 19.4x faster than the slowest successful cell (`tenferro-eager`, 0.494 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 50.1x faster than the slowest successful cell (`tenferro-eager`, 0.431 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`tenferro-eager`, 0.431 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 79.0x faster than the slowest successful cell (`tenferro-eager`, 0.798 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 24.8x faster than the slowest successful cell (`tenferro-eager`, 0.798 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 21.6x faster than the slowest successful cell (`pytorch-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 22.6x faster than the slowest successful cell (`pytorch-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.160 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`2x2`): `jax-cpu` is 21.7x faster than the slowest successful cell (`tenferro-eager`, 0.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 20.9x faster than the slowest successful cell (`tenferro-eager`, 0.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 17.8x faster than the slowest successful cell (`tenferro-eager`, 0.317 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`tenferro-eager`, 0.317 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261127_180034/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261127_180034/run.yaml`
- Timestamp: `2026-11-27T18:00:34.399034+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261127_180034`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261127_180034/cpu_ops_t4_20261127_180034.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.001 | 0.008 ± 0.001 | 0.015 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.001 | 0.021 ± 0.001 | 0.019 ± 0.001 | 0.064 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.026 ± 0.002 | 0.023 ± 0.002 | 0.027 ± 0.010 | 0.031 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.090 ± 0.003 | 0.088 ± 0.001 | 0.106 ± 0.032 | 0.122 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.069 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.001 | 0.004 ± 0.002 | 0.008 ± 0.000 | 0.087 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.039 ± 0.004 | 0.040 ± 0.001 | 0.089 ± 0.013 | 0.188 ± 0.036 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.001 | 0.008 ± 0.000 | 0.089 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.088 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.031 ± 0.002 | 0.012 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.013 ± 0.001 | 0.013 ± 0.000 | 0.039 ± 0.004 | 0.026 ± 0.008 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.038 ± 0.009 | 0.016 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.028 ± 0.001 | 0.026 ± 0.001 | 0.069 ± 0.013 | 0.055 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.041 ± 0.006 | 0.011 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.055 ± 0.004 | 0.030 ± 0.022 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.042 ± 0.007 | 0.010 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.054 ± 0.009 | 0.018 ± 0.010 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.001 | 0.039 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.036 ± 0.002 | 0.037 ± 0.003 | 0.041 ± 0.001 | 0.128 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.041 ± 0.002 | 0.038 ± 0.003 | 0.053 ± 0.015 | 0.069 ± 0.022 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.137 ± 0.012 | 0.142 ± 0.003 | 0.161 ± 0.013 | 0.276 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.036 ± 0.000 | 0.008 ± 0.001 | 0.055 ± 0.001 | 0.008 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.040 ± 0.002 | 0.013 ± 0.000 | 0.060 ± 0.002 | 0.008 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.037 ± 0.001 | 0.010 ± 0.000 | 0.056 ± 0.000 | 0.014 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.050 ± 0.011 | 0.017 ± 0.001 | 0.060 ± 0.004 | 0.049 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.048 ± 0.000 | 0.029 ± 0.001 | 0.096 ± 0.029 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.053 ± 0.000 | 0.044 ± 0.006 | 0.111 ± 0.009 | 0.019 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.050 ± 0.001 | 0.056 ± 0.035 | 0.084 ± 0.008 | 0.014 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.060 ± 0.004 | 0.059 ± 0.004 | 0.112 ± 0.012 | 0.039 ± 0.008 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.263 ± 0.005 | 0.297 ± 0.009 | 0.299 ± 0.019 | 0.359 ± 0.012 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 15.049 ± 0.425 | 11.396 ± 0.235 | 5.893 ± 0.143 | 5.725 ± 0.091 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 86.868 ± 7.589 | 60.612 ± 1.384 | 29.631 ± 0.916 | 31.236 ± 2.281 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 8.306 ± 1.185 | 11.038 ± 0.199 | 5.221 ± 0.082 | 5.824 ± 0.082 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 43.357 ± 1.165 | 61.307 ± 0.619 | 26.724 ± 1.433 | 31.907 ± 4.881 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 9.046 ± 1.072 | 4.765 ± 0.106 | 3.273 ± 0.085 | 1.836 ± 0.029 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 53.986 ± 3.676 | 25.200 ± 2.646 | 17.729 ± 0.776 | 14.513 ± 2.204 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 8.822 ± 0.255 | 4.861 ± 0.333 | 2.293 ± 0.508 | 1.839 ± 0.034 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 49.953 ± 4.815 | 24.689 ± 1.506 | 10.353 ± 1.345 | 10.778 ± 0.270 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.055 ± 0.020 | 0.049 ± 0.013 | 0.011 ± 0.001 | 0.034 ± 0.003 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.176 ± 0.081 | 0.134 ± 0.051 | 0.047 ± 0.002 | 0.076 ± 0.001 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 13.532 ± 1.481 | 7.657 ± 0.266 | 3.596 ± 1.693 | 4.653 ± 0.323 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 63.414 ± 9.091 | 40.636 ± 3.578 | 16.011 ± 0.133 | 17.456 ± 0.088 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 10.007 ± 1.321 | 7.520 ± 0.284 | 5.433 ± 1.423 | 3.248 ± 0.022 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 58.267 ± 0.787 | 40.624 ± 5.152 | 15.763 ± 0.346 | 18.033 ± 0.512 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.068 ± 0.007 | 0.102 ± 0.037 | 0.132 ± 0.122 | 0.050 ± 0.019 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 1.990 ± 0.339 | 1.637 ± 0.143 | 0.532 ± 0.109 | 0.324 ± 0.003 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 7.338 ± 0.449 | 7.038 ± 0.330 | 1.718 ± 0.042 | 1.590 ± 0.032 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 1.134 ± 0.381 | 1.641 ± 0.065 | 0.829 ± 1.432 | 0.321 ± 0.017 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 4.635 ± 0.099 | 6.957 ± 0.158 | 2.977 ± 0.228 | 1.417 ± 0.013 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.673 ± 0.034 | 0.898 ± 0.192 | 0.474 ± 0.024 | 0.506 ± 0.024 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 27.379 ± 3.088 | 21.909 ± 2.242 | 16.394 ± 5.981 | 13.020 ± 2.807 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 180.443 ± 12.554 | 121.227 ± 3.295 | 57.476 ± 7.315 | 63.936 ± 1.163 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 13.787 ± 2.353 | 20.075 ± 2.596 | 10.539 ± 2.969 | 11.417 ± 0.640 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 83.418 ± 12.693 | 117.067 ± 3.329 | 52.689 ± 6.509 | 55.607 ± 2.814 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.157 ± 0.007 | 0.110 ± 0.070 | 0.029 ± 0.014 | 0.161 ± 0.020 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.669 ± 0.157 | 0.829 ± 0.265 | 0.182 ± 0.041 | 0.698 ± 0.327 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 2.348 ± 0.328 | 2.231 ± 0.348 | 0.661 ± 0.094 | 1.892 ± 0.065 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.092 ± 0.007 | 0.102 ± 0.024 | 0.104 ± 0.029 | 0.085 ± 0.037 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.021 ± 0.001 | 0.032 ± 0.005 | 0.033 ± 0.006 | 0.064 ± 0.026 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.028 ± 0.001 | 0.028 ± 0.000 | 0.038 ± 0.002 | 0.068 ± 0.033 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.056 ± 0.000 | 0.048 ± 0.008 | 0.057 ± 0.035 | 0.141 ± 0.042 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.459 ± 0.008 | 0.516 ± 0.012 | 0.429 ± 0.303 | 0.443 ± 0.049 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.068 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.078 ± 0.018 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.087 ± 0.004 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.124 ± 0.012 | 0.017 ± 0.001 | 0.120 ± 0.047 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.106 ± 0.002 | 0.020 ± 0.000 | 0.072 ± 0.003 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.111 ± 0.004 | 0.025 ± 0.001 | 0.079 ± 0.004 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.035 ± 0.016 | 0.017 ± 0.000 | 0.057 ± 0.034 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.034 ± 0.001 | 0.020 ± 0.001 | 0.062 ± 0.002 | 0.012 ± 0.004 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.043 ± 0.004 | 0.026 ± 0.001 | 0.068 ± 0.007 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.515 ± 0.053 | 0.019 ± 0.000 | 0.392 ± 0.052 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.353 ± 0.007 | 0.021 ± 0.001 | 0.298 ± 0.082 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.373 ± 0.078 | 0.028 ± 0.007 | 0.356 ± 0.027 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.085 ± 0.004 | 0.018 ± 0.002 | 0.211 ± 0.013 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.081 ± 0.004 | 0.019 ± 0.000 | 0.215 ± 0.018 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.084 ± 0.008 | 0.022 ± 0.008 | 0.284 ± 0.112 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.035 ± 0.005 | 0.007 ± 0.000 | 0.029 ± 0.013 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.036 ± 0.003 | 0.007 ± 0.000 | 0.025 ± 0.003 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.033 ± 0.001 | 0.008 ± 0.000 | 0.029 ± 0.000 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.391 ± 0.038 | 0.026 ± 0.000 | 0.280 ± 0.058 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.415 ± 0.021 | 0.029 ± 0.001 | 0.155 ± 0.008 | 0.015 ± 0.006 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.422 ± 0.028 | 0.029 ± 0.000 | 0.148 ± 0.020 | 0.015 ± 0.002 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.113 ± 0.069 | 0.028 ± 0.002 | 0.273 ± 0.095 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.068 ± 0.002 | 0.029 ± 0.001 | 0.145 ± 0.010 | 0.012 ± 0.006 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.076 ± 0.005 | 0.029 ± 0.000 | 0.144 ± 0.047 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.045 ± 0.004 | 0.022 ± 0.000 | 0.038 ± 0.001 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.043 ± 0.000 | 0.022 ± 0.000 | 0.040 ± 0.005 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.044 ± 0.005 | 0.022 ± 0.001 | 0.041 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.078 ± 0.003 | 0.015 ± 0.000 | 0.158 ± 0.007 | 0.007 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.079 ± 0.004 | 0.016 ± 0.000 | 0.154 ± 0.018 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.076 ± 0.003 | 0.020 ± 0.003 | 0.165 ± 0.016 | 0.014 ± 0.010 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.047 ± 0.006 | 0.015 ± 0.000 | 0.078 ± 0.008 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.036 ± 0.003 | 0.016 ± 0.001 | 0.074 ± 0.003 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.038 ± 0.001 | 0.020 ± 0.004 | 0.077 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.040 ± 0.005 | 0.011 ± 0.000 | 0.036 ± 0.047 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.039 ± 0.001 | 0.015 ± 0.000 | 0.032 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.045 ± 0.006 | 0.024 ± 0.002 | 0.043 ± 0.003 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.091 ± 0.002 | 0.012 ± 0.000 | 0.096 ± 0.013 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.096 ± 0.035 | 0.015 ± 0.000 | 0.092 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.103 ± 0.002 | 0.024 ± 0.000 | 0.113 ± 0.001 | 0.046 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.033 ± 0.003 | 0.011 ± 0.001 | 0.074 ± 0.024 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.033 ± 0.001 | 0.015 ± 0.000 | 0.063 ± 0.001 | 0.014 ± 0.003 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.041 ± 0.023 | 0.023 ± 0.002 | 0.075 ± 0.006 | 0.047 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.027 ± 0.001 | 0.017 ± 0.007 | 0.007 ± 0.001 | 0.027 ± 0.030 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.005 ± 0.003 | 0.003 ± 0.000 | 0.029 ± 0.004 | 0.008 ± 0.007 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.030 ± 0.015 | 0.008 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.029 ± 0.017 | 0.008 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.006 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.015 ± 0.008 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.011 ± 0.000 | 0.010 ± 0.001 | 0.014 ± 0.001 | 0.020 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 28.2x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 25.9x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 22.6x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 23.3x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 32.0x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 29.7x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 20.3x faster than the slowest successful cell (`jax-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 19.7x faster than the slowest successful cell (`jax-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 24.4x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 28.5x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 35.4x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 18.0x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 17.4x faster than the slowest successful cell (`tenferro-eager`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 74.8x faster than the slowest successful cell (`tenferro-eager`, 0.515 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 27.0x faster than the slowest successful cell (`tenferro-eager`, 0.515 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 35.1x faster than the slowest successful cell (`tenferro-eager`, 0.353 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`tenferro-eager`, 0.353 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 44.6x faster than the slowest successful cell (`tenferro-eager`, 0.373 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`tenferro-eager`, 0.373 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 30.0x faster than the slowest successful cell (`pytorch-cpu`, 0.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 22.4x faster than the slowest successful cell (`pytorch-cpu`, 0.215 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.215 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 33.6x faster than the slowest successful cell (`pytorch-cpu`, 0.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 50.3x faster than the slowest successful cell (`tenferro-eager`, 0.391 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`tenferro-eager`, 0.391 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 26.8x faster than the slowest successful cell (`tenferro-eager`, 0.415 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`tenferro-eager`, 0.415 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 28.0x faster than the slowest successful cell (`tenferro-eager`, 0.422 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`tenferro-eager`, 0.422 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 35.5x faster than the slowest successful cell (`pytorch-cpu`, 0.273 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.145 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 21.3x faster than the slowest successful cell (`pytorch-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.165 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

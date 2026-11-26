# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `9b03dd718025b3350d93eaab13a8b93dcc66cab2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261126_172406/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261126_172406/run.yaml`
- Timestamp: `2026-11-26T17:24:06.847382+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261126_172406`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261126_172406/cpu_ops_t1_20261126_172406.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.013 ± 0.006 | 0.011 ± 0.007 | 0.017 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.015 | 0.022 ± 0.010 | 0.022 ± 0.002 | 0.039 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.024 ± 0.001 | 0.031 ± 0.000 | 0.024 ± 0.001 | 0.039 ± 0.012 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.088 ± 0.001 | 0.094 ± 0.092 | 0.098 ± 0.018 | 0.143 ± 0.065 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.002 ± 0.000 | 0.003 ± 0.002 | 0.008 ± 0.003 | 0.079 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.001 | 0.090 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.057 ± 0.014 | 0.083 ± 0.065 | 0.036 ± 0.004 | 0.110 ± 0.016 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.005 ± 0.004 | 0.009 ± 0.000 | 0.090 ± 0.012 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.005 ± 0.003 | 0.004 ± 0.003 | 0.012 ± 0.001 | 0.094 ± 0.027 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.002 | 0.007 ± 0.001 | 0.013 ± 0.009 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.013 ± 0.005 | 0.012 ± 0.000 | 0.021 ± 0.001 | 0.024 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.013 ± 0.006 | 0.011 ± 0.000 | 0.021 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.027 ± 0.001 | 0.039 ± 0.009 | 0.041 ± 0.002 | 0.051 ± 0.027 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.011 ± 0.002 | 0.009 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.001 | 0.016 ± 0.004 | 0.020 ± 0.009 | 0.014 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.012 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.013 ± 0.000 | 0.013 ± 0.002 | 0.024 ± 0.002 | 0.023 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.013 ± 0.005 | 0.013 ± 0.005 | 0.014 ± 0.001 | 0.053 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.040 ± 0.023 | 0.046 ± 0.038 | 0.039 ± 0.008 | 0.123 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.041 ± 0.018 | 0.037 ± 0.003 | 0.044 ± 0.001 | 0.075 ± 0.011 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.177 ± 0.074 | 0.150 ± 0.045 | 0.175 ± 0.016 | 0.253 ± 0.052 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.034 ± 0.001 | 0.016 ± 0.000 | 0.066 ± 0.006 | 0.008 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.040 ± 0.005 | 0.016 ± 0.017 | 0.068 ± 0.003 | 0.010 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.036 ± 0.002 | 0.009 ± 0.000 | 0.062 ± 0.004 | 0.009 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.043 ± 0.001 | 0.016 ± 0.000 | 0.076 ± 0.009 | 0.032 ± 0.012 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.047 ± 0.001 | 0.037 ± 0.046 | 0.046 ± 0.009 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.057 ± 0.027 | 0.040 ± 0.030 | 0.073 ± 0.060 | 0.020 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.048 ± 0.004 | 0.037 ± 0.020 | 0.046 ± 0.002 | 0.010 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.059 ± 0.006 | 0.056 ± 0.008 | 0.075 ± 0.010 | 0.028 ± 0.003 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.268 ± 0.003 | 0.268 ± 0.012 | 0.307 ± 0.012 | 0.383 ± 0.058 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 27.307 ± 5.121 | 12.305 ± 0.885 | 5.609 ± 1.658 | 5.956 ± 1.346 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 96.894 ± 17.455 | 62.235 ± 0.294 | 30.627 ± 10.087 | 48.320 ± 9.972 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 20.430 ± 7.404 | 11.143 ± 5.149 | 5.058 ± 0.489 | 6.888 ± 1.379 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 49.690 ± 13.521 | 62.573 ± 1.468 | 27.822 ± 1.780 | 48.777 ± 12.728 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 11.493 ± 2.815 | 5.088 ± 0.420 | 1.990 ± 0.178 | 2.538 ± 0.223 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 70.400 ± 9.291 | 44.997 ± 12.854 | 16.765 ± 4.435 | 17.610 ± 5.677 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 29.118 ± 2.058 | 5.671 ± 0.354 | 1.408 ± 0.590 | 2.297 ± 0.244 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 74.317 ± 10.373 | 46.129 ± 8.596 | 9.278 ± 0.963 | 18.844 ± 1.386 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.026 ± 0.000 | 0.019 ± 0.000 | 0.011 ± 0.001 | 0.031 ± 0.005 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.089 ± 0.004 | 0.054 ± 0.003 | 0.052 ± 0.015 | 0.062 ± 0.010 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 15.142 ± 3.663 | 8.187 ± 0.331 | 2.903 ± 0.139 | 4.876 ± 1.089 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 129.338 ± 24.908 | 60.685 ± 5.498 | 15.612 ± 0.535 | 31.021 ± 4.234 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 16.358 ± 6.006 | 8.202 ± 0.110 | 2.748 ± 0.292 | 4.582 ± 0.428 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 100.509 ± 16.571 | 55.294 ± 8.743 | 14.018 ± 0.654 | 37.059 ± 21.430 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.068 ± 0.002 | 0.078 ± 0.006 | 0.073 ± 0.024 | 0.038 ± 0.004 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 4.313 ± 3.180 | 1.091 ± 0.101 | 0.450 ± 0.052 | 0.252 ± 0.003 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 6.688 ± 0.574 | 8.056 ± 6.622 | 2.397 ± 0.147 | 1.445 ± 0.119 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 2.399 ± 0.278 | 1.099 ± 0.184 | 0.597 ± 0.110 | 0.362 ± 0.056 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 3.453 ± 0.343 | 7.487 ± 6.938 | 3.225 ± 0.254 | 1.709 ± 0.088 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.529 ± 0.004 | 0.748 ± 0.008 | 0.567 ± 0.089 | 0.490 ± 0.050 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 38.232 ± 12.210 | 19.896 ± 0.548 | 10.099 ± 0.296 | 11.683 ± 2.185 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 209.729 ± 36.355 | 154.245 ± 3.420 | 59.690 ± 0.571 | 58.435 ± 5.096 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 16.871 ± 5.487 | 20.512 ± 0.719 | 9.335 ± 0.686 | 11.169 ± 4.184 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 108.716 ± 8.896 | 124.226 ± 5.812 | 55.737 ± 4.589 | 62.038 ± 21.561 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.117 ± 0.002 | 0.112 ± 0.006 | 0.025 ± 0.002 | 0.129 ± 0.029 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.874 ± 0.069 | 0.847 ± 0.055 | 0.161 ± 0.008 | 0.922 ± 0.084 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 3.300 ± 0.052 | 3.382 ± 0.979 | 0.592 ± 0.023 | 3.685 ± 0.305 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.094 ± 0.003 | 0.123 ± 0.047 | 0.070 ± 0.003 | 0.077 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.021 ± 0.001 | 0.021 ± 0.001 | 0.036 ± 0.007 | 0.030 ± 0.006 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.030 ± 0.001 | 0.026 ± 0.000 | 0.053 ± 0.003 | 0.056 ± 0.009 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.057 ± 0.007 | 0.042 ± 0.000 | 0.096 ± 0.034 | 0.283 ± 0.157 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.466 ± 0.035 | 0.469 ± 0.040 | 0.416 ± 0.032 | 0.519 ± 0.180 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.009 ± 0.002 | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.002 | 0.012 ± 0.003 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.007 ± 0.010 | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.009 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.008 ± 0.001 | 0.005 ± 0.002 | 0.008 ± 0.000 | 0.105 ± 0.023 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.003 ± 0.002 | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.089 ± 0.025 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.003 | 0.102 ± 0.022 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.149 ± 0.068 | 0.026 ± 0.028 | 0.067 ± 0.005 | 0.008 ± 0.018 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.114 ± 0.006 | 0.019 ± 0.000 | 0.087 ± 0.026 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.117 ± 0.007 | 0.025 ± 0.001 | 0.119 ± 0.039 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.033 ± 0.012 | 0.027 ± 0.012 | 0.055 ± 0.001 | 0.011 ± 0.009 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.034 ± 0.001 | 0.018 ± 0.001 | 0.079 ± 0.052 | 0.011 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.041 ± 0.003 | 0.024 ± 0.001 | 0.092 ± 0.018 | 0.013 ± 0.005 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.408 ± 0.200 | 0.019 ± 0.009 | 0.101 ± 0.010 | 0.013 ± 0.005 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.360 ± 0.012 | 0.019 ± 0.001 | 0.129 ± 0.074 | 0.010 ± 0.004 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.366 ± 0.014 | 0.021 ± 0.000 | 0.139 ± 0.149 | 0.007 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.103 ± 0.032 | 0.018 ± 0.000 | 0.082 ± 0.008 | 0.022 ± 0.017 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.075 ± 0.011 | 0.017 ± 0.001 | 0.099 ± 0.017 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.081 ± 0.008 | 0.018 ± 0.000 | 0.168 ± 0.077 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.131 ± 0.067 | 0.010 ± 0.002 | 0.026 ± 0.004 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.033 ± 0.001 | 0.007 ± 0.000 | 0.030 ± 0.006 | 0.014 ± 0.006 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.032 ± 0.001 | 0.007 ± 0.000 | 0.041 ± 0.016 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.709 ± 0.082 | 0.046 ± 0.016 | 0.097 ± 0.010 | 0.010 ± 0.005 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.550 ± 0.578 | 0.026 ± 0.003 | 0.093 ± 0.015 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.430 ± 0.017 | 0.028 ± 0.000 | 0.154 ± 0.196 | 0.006 ± 0.004 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.127 ± 0.069 | 0.054 ± 0.022 | 0.082 ± 0.021 | 0.017 ± 0.011 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.090 ± 0.054 | 0.026 ± 0.001 | 0.088 ± 0.017 | 0.011 ± 0.002 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.073 ± 0.039 | 0.028 ± 0.008 | 0.123 ± 0.078 | 0.007 ± 0.005 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.142 ± 0.014 | 0.038 ± 0.011 | 0.054 ± 0.020 | 0.016 ± 0.006 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.044 ± 0.001 | 0.021 ± 0.001 | 0.050 ± 0.026 | 0.012 ± 0.006 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.043 ± 0.002 | 0.022 ± 0.001 | 0.046 ± 0.024 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.163 ± 0.139 | 0.016 ± 0.009 | 0.157 ± 0.012 | 0.014 ± 0.005 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.095 ± 0.016 | 0.014 ± 0.001 | 0.187 ± 0.026 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.074 ± 0.003 | 0.015 ± 0.001 | 0.193 ± 0.160 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.063 ± 0.033 | 0.016 ± 0.009 | 0.076 ± 0.004 | 0.010 ± 0.005 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.035 ± 0.001 | 0.015 ± 0.001 | 0.130 ± 0.146 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.036 ± 0.001 | 0.016 ± 0.002 | 0.086 ± 0.037 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.039 ± 0.002 | 0.017 ± 0.003 | 0.046 ± 0.024 | 0.012 ± 0.002 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.039 ± 0.001 | 0.014 ± 0.000 | 0.033 ± 0.004 | 0.022 ± 0.009 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.044 ± 0.001 | 0.022 ± 0.000 | 0.048 ± 0.007 | 0.021 ± 0.003 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.266 ± 0.116 | 0.024 ± 0.012 | 0.114 ± 0.065 | 0.016 ± 0.010 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.092 ± 0.091 | 0.014 ± 0.000 | 0.100 ± 0.004 | 0.014 ± 0.009 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.101 ± 0.001 | 0.023 ± 0.000 | 0.144 ± 0.089 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.052 ± 0.066 | 0.023 ± 0.016 | 0.105 ± 0.016 | 0.013 ± 0.009 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.073 ± 0.062 | 0.014 ± 0.000 | 0.064 ± 0.002 | 0.013 ± 0.004 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.040 ± 0.001 | 0.022 ± 0.003 | 0.079 ± 0.020 | 0.020 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.004 ± 0.003 | 0.002 ± 0.001 | 0.001 ± 0.000 | 0.008 ± 0.005 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.016 ± 0.003 | 0.005 ± 0.000 | 0.002 ± 0.001 | 0.010 ± 0.005 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.001 | 0.002 ± 0.002 | 0.001 ± 0.000 | 0.007 ± 0.006 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.066 ± 0.005 | 0.022 ± 0.007 | 0.007 ± 0.001 | 0.029 ± 0.012 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.007 ± 0.009 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.012 ± 0.003 | 0.003 ± 0.002 | 0.003 ± 0.002 | 0.009 ± 0.004 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.004 ± 0.003 | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.011 ± 0.010 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.012 ± 0.002 | 0.005 ± 0.002 | 0.009 ± 0.002 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.013 ± 0.003 | 0.006 ± 0.005 | 0.010 ± 0.005 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.005 ± 0.003 | 0.004 ± 0.000 | 0.010 ± 0.002 | 0.009 ± 0.005 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.009 ± 0.007 | 0.004 ± 0.000 | 0.010 ± 0.004 | 0.008 ± 0.004 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.005 ± 0.002 | 0.004 ± 0.000 | 0.009 ± 0.002 | 0.010 ± 0.005 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.010 ± 0.002 | 0.011 ± 0.007 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.008 ± 0.008 | 0.005 ± 0.002 | 0.007 ± 0.004 | 0.020 ± 0.014 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.006 ± 0.003 | 0.006 ± 0.002 | 0.008 ± 0.001 | 0.018 ± 0.008 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.019 ± 0.004 | 0.032 ± 0.024 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 32.7x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 27.1x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 26.7x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 26.6x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 32.6x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 16.5x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 20.0x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=1, shape=`256x256`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 29.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=1, shape=`256x256`): `pytorch-cpu` is 20.7x faster than the slowest successful cell (`tenferro-eager`, 29.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=1, shape=`256x256,rhs=1`): `jax-cpu` is 17.1x faster than the slowest successful cell (`tenferro-eager`, 4.313 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 19.3x faster than the slowest successful cell (`jax-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 30.7x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 43.3x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 18.6x faster than the slowest successful cell (`tenferro-eager`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 0.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 32.1x faster than the slowest successful cell (`tenferro-eager`, 0.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 21.9x faster than the slowest successful cell (`tenferro-eager`, 0.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 37.5x faster than the slowest successful cell (`tenferro-eager`, 0.360 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 19.4x faster than the slowest successful cell (`tenferro-eager`, 0.360 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 51.1x faster than the slowest successful cell (`tenferro-eager`, 0.366 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 17.7x faster than the slowest successful cell (`tenferro-eager`, 0.366 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 27.5x faster than the slowest successful cell (`pytorch-cpu`, 0.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.2x faster than the slowest successful cell (`tenferro-eager`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 71.9x faster than the slowest successful cell (`tenferro-eager`, 0.709 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 0.709 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 56.8x faster than the slowest successful cell (`tenferro-eager`, 0.550 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`tenferro-eager`, 0.550 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 71.9x faster than the slowest successful cell (`tenferro-eager`, 0.430 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 0.430 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 11.4x faster than the slowest successful cell (`tenferro-eager`, 0.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`tenferro-eager`, 0.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 23.8x faster than the slowest successful cell (`pytorch-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 24.0x faster than the slowest successful cell (`pytorch-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 0.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`tenferro-eager`, 0.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261126_180245/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261126_180245/run.yaml`
- Timestamp: `2026-11-26T18:02:45.132485+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261126_180245`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261126_180245/cpu_ops_t4_20261126_180245.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.015 ± 0.008 | 0.008 ± 0.001 | 0.017 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.000 | 0.026 ± 0.003 | 0.022 ± 0.011 | 0.049 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.026 ± 0.001 | 0.028 ± 0.009 | 0.034 ± 0.020 | 0.033 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.083 ± 0.001 | 0.111 ± 0.019 | 0.088 ± 0.012 | 0.120 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.004 ± 0.003 | 0.008 ± 0.001 | 0.066 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.002 | 0.089 ± 0.067 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.040 ± 0.004 | 0.048 ± 0.004 | 0.066 ± 0.085 | 0.232 ± 0.075 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.001 | 0.010 ± 0.005 | 0.089 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.002 | 0.009 ± 0.001 | 0.085 ± 0.039 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.038 ± 0.021 | 0.012 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.014 ± 0.000 | 0.015 ± 0.002 | 0.070 ± 0.018 | 0.026 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.010 ± 0.002 | 0.041 ± 0.015 | 0.017 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.025 ± 0.001 | 0.038 ± 0.009 | 0.064 ± 0.006 | 0.054 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.058 ± 0.020 | 0.009 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.013 ± 0.001 | 0.048 ± 0.008 | 0.013 ± 0.025 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.008 ± 0.001 | 0.044 ± 0.041 | 0.010 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.012 ± 0.000 | 0.015 ± 0.001 | 0.059 ± 0.025 | 0.026 ± 0.020 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.013 ± 0.000 | 0.013 ± 0.002 | 0.013 ± 0.001 | 0.043 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.041 ± 0.002 | 0.043 ± 0.003 | 0.042 ± 0.010 | 0.132 ± 0.026 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.044 ± 0.002 | 0.042 ± 0.003 | 0.053 ± 0.032 | 0.077 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.136 ± 0.000 | 0.183 ± 0.056 | 0.161 ± 0.012 | 0.281 ± 0.021 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.035 ± 0.001 | 0.010 ± 0.002 | 0.060 ± 0.008 | 0.010 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.045 ± 0.007 | 0.018 ± 0.002 | 0.056 ± 0.002 | 0.010 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.042 ± 0.003 | 0.013 ± 0.002 | 0.075 ± 0.155 | 0.011 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.043 ± 0.003 | 0.024 ± 0.002 | 0.079 ± 0.008 | 0.050 ± 0.018 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.050 ± 0.008 | 0.036 ± 0.003 | 0.083 ± 0.017 | 0.014 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.063 ± 0.006 | 0.069 ± 0.029 | 0.103 ± 0.003 | 0.018 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.059 ± 0.004 | 0.035 ± 0.026 | 0.173 ± 0.090 | 0.025 ± 0.018 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.058 ± 0.001 | 0.073 ± 0.046 | 0.119 ± 0.021 | 0.035 ± 0.003 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.267 ± 0.007 | 0.288 ± 0.007 | 0.314 ± 0.018 | 0.329 ± 0.017 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 13.407 ± 0.921 | 12.031 ± 0.329 | 5.566 ± 0.414 | 6.076 ± 0.745 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 74.503 ± 0.914 | 91.805 ± 9.905 | 30.529 ± 6.331 | 27.607 ± 0.709 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 7.408 ± 0.834 | 11.890 ± 0.813 | 6.091 ± 2.007 | 6.194 ± 1.179 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 40.466 ± 1.168 | 71.447 ± 6.211 | 29.211 ± 3.144 | 36.755 ± 9.509 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 6.434 ± 0.080 | 5.502 ± 0.234 | 3.715 ± 0.721 | 1.763 ± 0.284 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 38.331 ± 0.334 | 34.180 ± 3.571 | 18.651 ± 3.950 | 8.488 ± 0.673 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 6.493 ± 1.478 | 5.465 ± 0.282 | 1.612 ± 0.284 | 1.957 ± 0.218 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 35.979 ± 0.830 | 37.170 ± 9.481 | 11.702 ± 3.169 | 13.754 ± 1.485 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.027 ± 0.006 | 0.038 ± 0.007 | 0.011 ± 0.002 | 0.032 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.156 ± 0.006 | 0.082 ± 0.007 | 0.052 ± 0.035 | 0.080 ± 0.007 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 12.615 ± 0.635 | 8.742 ± 1.210 | 5.644 ± 3.421 | 3.069 ± 0.562 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 54.418 ± 3.410 | 47.530 ± 5.064 | 12.722 ± 0.996 | 14.266 ± 0.453 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 9.414 ± 0.802 | 8.135 ± 0.303 | 2.911 ± 2.139 | 2.569 ± 0.201 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 50.219 ± 4.966 | 54.208 ± 9.395 | 16.291 ± 7.140 | 14.020 ± 0.966 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.065 ± 0.012 | 0.089 ± 0.006 | 0.082 ± 0.092 | 0.039 ± 0.009 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 1.095 ± 0.044 | 1.672 ± 0.166 | 0.472 ± 0.052 | 0.334 ± 0.059 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 5.664 ± 0.110 | 10.058 ± 1.947 | 1.936 ± 0.402 | 1.966 ± 0.211 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.570 ± 0.020 | 1.718 ± 0.117 | 0.972 ± 0.250 | 0.344 ± 0.042 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 2.838 ± 0.065 | 11.775 ± 1.197 | 2.612 ± 0.104 | 1.827 ± 0.191 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.553 ± 0.011 | 0.895 ± 0.049 | 0.458 ± 0.115 | 0.540 ± 0.104 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 23.265 ± 0.326 | 21.273 ± 1.400 | 11.437 ± 3.065 | 10.536 ± 1.700 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 150.174 ± 5.876 | 136.202 ± 9.157 | 55.352 ± 5.821 | 54.590 ± 5.601 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 12.125 ± 0.071 | 20.173 ± 0.472 | 12.349 ± 3.203 | 9.215 ± 0.985 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 76.708 ± 5.586 | 128.658 ± 1.600 | 58.068 ± 13.747 | 52.353 ± 3.697 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.083 ± 0.031 | 0.155 ± 0.012 | 0.026 ± 0.009 | 0.129 ± 0.021 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.468 ± 0.022 | 0.649 ± 0.094 | 0.168 ± 0.047 | 0.533 ± 0.186 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 1.639 ± 0.231 | 2.915 ± 1.548 | 0.632 ± 0.209 | 1.649 ± 0.357 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.089 ± 0.011 | 0.104 ± 0.002 | 0.106 ± 0.006 | 0.094 ± 0.029 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.019 ± 0.004 | 0.023 ± 0.001 | 0.033 ± 0.006 | 0.041 ± 0.003 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.027 ± 0.000 | 0.027 ± 0.009 | 0.042 ± 0.008 | 0.093 ± 0.027 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.054 ± 0.000 | 0.045 ± 0.016 | 0.054 ± 0.006 | 0.175 ± 0.027 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.435 ± 0.015 | 0.506 ± 0.017 | 0.436 ± 0.011 | 0.444 ± 0.062 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.001 | 0.004 ± 0.001 | 0.008 ± 0.002 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.005 ± 0.004 | 0.009 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.008 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.003 | 0.015 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.007 | 0.067 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.005 | 0.067 ± 0.008 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | 0.127 ± 0.067 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.147 ± 0.030 | 0.019 ± 0.000 | 0.174 ± 0.091 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.101 ± 0.002 | 0.019 ± 0.001 | 0.078 ± 0.011 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.135 ± 0.007 | 0.027 ± 0.001 | 0.077 ± 0.009 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.042 ± 0.022 | 0.018 ± 0.001 | 0.166 ± 0.067 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.032 ± 0.000 | 0.018 ± 0.001 | 0.058 ± 0.002 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.044 ± 0.004 | 0.028 ± 0.001 | 0.070 ± 0.005 | 0.011 ± 0.011 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.507 ± 0.147 | 0.021 ± 0.000 | 0.470 ± 0.401 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.328 ± 0.013 | 0.022 ± 0.000 | 0.364 ± 0.136 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.397 ± 0.056 | 0.022 ± 0.003 | 0.321 ± 0.125 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.123 ± 0.039 | 0.019 ± 0.000 | 0.442 ± 0.212 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.074 ± 0.003 | 0.019 ± 0.000 | 0.313 ± 0.211 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.087 ± 0.001 | 0.022 ± 0.002 | 0.264 ± 0.071 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.038 ± 0.005 | 0.007 ± 0.000 | 0.034 ± 0.015 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.033 ± 0.001 | 0.007 ± 0.000 | 0.031 ± 0.012 | 0.008 ± 0.009 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.032 ± 0.001 | 0.008 ± 0.000 | 0.029 ± 0.003 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.497 ± 0.146 | 0.030 ± 0.001 | 0.361 ± 0.292 | 0.010 ± 0.003 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.400 ± 0.016 | 0.027 ± 0.001 | 0.167 ± 0.014 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.397 ± 0.015 | 0.035 ± 0.002 | 0.198 ± 0.082 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.138 ± 0.074 | 0.030 ± 0.000 | 0.383 ± 0.146 | 0.010 ± 0.005 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.067 ± 0.013 | 0.028 ± 0.000 | 0.230 ± 0.019 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.088 ± 0.007 | 0.033 ± 0.002 | 0.161 ± 0.013 | 0.006 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.131 ± 0.056 | 0.023 ± 0.001 | 0.053 ± 0.045 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.043 ± 0.006 | 0.021 ± 0.002 | 0.068 ± 0.032 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.043 ± 0.001 | 0.025 ± 0.001 | 0.042 ± 0.003 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.114 ± 0.032 | 0.016 ± 0.005 | 0.176 ± 0.006 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.067 ± 0.001 | 0.018 ± 0.003 | 0.158 ± 0.010 | 0.008 ± 0.003 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.081 ± 0.010 | 0.017 ± 0.001 | 0.201 ± 0.030 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.046 ± 0.017 | 0.016 ± 0.001 | 0.085 ± 0.006 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.034 ± 0.002 | 0.018 ± 0.000 | 0.082 ± 0.018 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.036 ± 0.001 | 0.018 ± 0.001 | 0.101 ± 0.025 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.040 ± 0.026 | 0.012 ± 0.000 | 0.060 ± 0.014 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.038 ± 0.002 | 0.015 ± 0.000 | 0.040 ± 0.012 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.043 ± 0.002 | 0.026 ± 0.001 | 0.043 ± 0.011 | 0.020 ± 0.019 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.125 ± 0.017 | 0.013 ± 0.000 | 0.116 ± 0.018 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.089 ± 0.003 | 0.015 ± 0.000 | 0.099 ± 0.010 | 0.011 ± 0.004 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.097 ± 0.004 | 0.028 ± 0.003 | 0.120 ± 0.010 | 0.018 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.044 ± 0.027 | 0.013 ± 0.000 | 0.124 ± 0.103 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.034 ± 0.002 | 0.015 ± 0.000 | 0.071 ± 0.021 | 0.012 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.038 ± 0.000 | 0.025 ± 0.002 | 0.078 ± 0.064 | 0.018 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.001 | 0.008 ± 0.006 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.002 ± 0.001 | 0.002 ± 0.000 | 0.001 ± 0.001 | 0.005 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.026 ± 0.001 | 0.014 ± 0.000 | 0.008 ± 0.009 | 0.051 ± 0.026 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.004 ± 0.001 | 0.002 ± 0.000 | 0.051 ± 0.008 | 0.007 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.051 ± 0.006 | 0.009 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.051 ± 0.012 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.006 ± 0.001 | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.007 ± 0.004 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.006 ± 0.002 | 0.005 ± 0.000 | 0.010 ± 0.007 | 0.006 ± 0.004 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.003 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.007 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.008 | 0.006 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.009 ± 0.001 | 0.007 ± 0.003 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.004 ± 0.001 | 0.003 ± 0.001 | 0.007 ± 0.006 | 0.011 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.007 | 0.014 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.014 ± 0.003 | 0.011 ± 0.000 | 0.017 ± 0.006 | 0.027 ± 0.007 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 24.7x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 24.6x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 21.9x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 31.4x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 28.2x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 18.6x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `tenferro-eager` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 28.8x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 25.7x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 50.3x faster than the slowest successful cell (`jax-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 23.4x faster than the slowest successful cell (`jax-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 23.9x faster than the slowest successful cell (`pytorch-cpu`, 0.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.3x faster than the slowest successful cell (`tenferro-eager`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 14.9x faster than the slowest successful cell (`tenferro-eager`, 0.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 21.9x faster than the slowest successful cell (`pytorch-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 69.1x faster than the slowest successful cell (`tenferro-eager`, 0.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 24.1x faster than the slowest successful cell (`tenferro-eager`, 0.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 45.3x faster than the slowest successful cell (`pytorch-cpu`, 0.364 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 0.364 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 49.1x faster than the slowest successful cell (`tenferro-eager`, 0.397 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 17.9x faster than the slowest successful cell (`tenferro-eager`, 0.397 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 60.4x faster than the slowest successful cell (`pytorch-cpu`, 0.442 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 22.9x faster than the slowest successful cell (`pytorch-cpu`, 0.442 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 40.1x faster than the slowest successful cell (`pytorch-cpu`, 0.313 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 0.313 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 29.2x faster than the slowest successful cell (`pytorch-cpu`, 0.264 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.264 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 47.6x faster than the slowest successful cell (`tenferro-eager`, 0.497 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 0.497 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 44.8x faster than the slowest successful cell (`tenferro-eager`, 0.400 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`tenferro-eager`, 0.400 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 60.0x faster than the slowest successful cell (`tenferro-eager`, 0.397 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`tenferro-eager`, 0.397 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 38.3x faster than the slowest successful cell (`pytorch-cpu`, 0.383 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.383 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 24.6x faster than the slowest successful cell (`pytorch-cpu`, 0.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 27.8x faster than the slowest successful cell (`pytorch-cpu`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 15.8x faster than the slowest successful cell (`tenferro-eager`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 24.8x faster than the slowest successful cell (`pytorch-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 23.4x faster than the slowest successful cell (`pytorch-cpu`, 0.201 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.201 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 13.0x faster than the slowest successful cell (`tenferro-eager`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 17.5x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

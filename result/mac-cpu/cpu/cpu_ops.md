# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `74775e4a2ad3480fb9ee9f859e0037b8cac731f2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261125_172313/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261125_172313/run.yaml`
- Timestamp: `2026-11-25T17:23:13.361566+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261125_172313`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261125_172313/cpu_ops_t1_20261125_172313.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.001 | 0.011 ± 0.007 | 0.011 ± 0.002 | 0.017 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.025 ± 0.001 | 0.049 ± 0.036 | 0.020 ± 0.001 | 0.044 ± 0.026 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.027 ± 0.001 | 0.049 ± 0.012 | 0.024 ± 0.002 | 0.038 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.104 ± 0.006 | 0.169 ± 0.020 | 0.107 ± 0.021 | 0.150 ± 0.054 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.004 ± 0.003 | 0.008 ± 0.002 | 0.135 ± 0.126 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.010 ± 0.006 | 0.011 ± 0.003 | 0.158 ± 0.098 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.042 ± 0.007 | 0.050 ± 0.010 | 0.040 ± 0.004 | 0.219 ± 0.139 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.001 | 0.006 ± 0.004 | 0.010 ± 0.004 | 0.111 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.010 ± 0.005 | 0.014 ± 0.008 | 0.120 ± 0.103 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.005 | 0.009 ± 0.002 | 0.016 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.015 ± 0.001 | 0.036 ± 0.008 | 0.021 ± 0.003 | 0.031 ± 0.010 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.010 ± 0.001 | 0.013 ± 0.003 | 0.011 ± 0.001 | 0.023 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.031 ± 0.001 | 0.051 ± 0.049 | 0.042 ± 0.040 | 0.051 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.002 | 0.013 ± 0.003 | 0.010 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.012 ± 0.000 | 0.029 ± 0.011 | 0.020 ± 0.004 | 0.017 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.013 ± 0.004 | 0.013 ± 0.001 | 0.012 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.000 | 0.034 ± 0.005 | 0.023 ± 0.002 | 0.024 ± 0.018 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.016 ± 0.005 | 0.016 ± 0.002 | 0.059 ± 0.012 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.048 ± 0.002 | 0.099 ± 0.018 | 0.043 ± 0.057 | 0.193 ± 0.070 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.044 ± 0.004 | 0.056 ± 0.017 | 0.050 ± 0.011 | 0.097 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.161 ± 0.001 | 0.264 ± 0.155 | 0.265 ± 0.181 | 0.268 ± 0.262 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.036 ± 0.001 | 0.023 ± 0.015 | 0.098 ± 0.043 | 0.010 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.039 ± 0.002 | 0.046 ± 0.005 | 0.071 ± 0.005 | 0.013 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.036 ± 0.001 | 0.018 ± 0.006 | 0.077 ± 0.011 | 0.021 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.050 ± 0.005 | 0.028 ± 0.008 | 0.085 ± 0.010 | 0.033 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.049 ± 0.006 | 0.039 ± 0.133 | 0.054 ± 0.008 | 0.020 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.053 ± 0.003 | 0.073 ± 0.038 | 0.068 ± 0.011 | 0.027 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.049 ± 0.002 | 0.055 ± 0.010 | 0.081 ± 0.022 | 0.037 ± 0.023 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.061 ± 0.007 | 0.151 ± 0.091 | 0.080 ± 0.010 | 0.058 ± 0.036 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.284 ± 0.052 | 0.261 ± 0.006 | 0.310 ± 0.023 | 0.396 ± 0.068 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 18.056 ± 6.633 | 12.751 ± 0.402 | 5.724 ± 0.099 | 8.175 ± 1.276 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 89.593 ± 14.236 | 64.242 ± 5.327 | 36.277 ± 1.387 | 44.390 ± 2.072 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 10.937 ± 2.472 | 12.757 ± 1.919 | 5.274 ± 0.164 | 10.532 ± 5.492 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 61.972 ± 9.740 | 63.201 ± 1.174 | 35.166 ± 5.723 | 45.318 ± 9.481 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 11.906 ± 3.735 | 6.780 ± 0.323 | 2.228 ± 0.076 | 5.220 ± 2.240 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 82.919 ± 13.885 | 35.963 ± 1.065 | 27.903 ± 6.868 | 24.012 ± 0.825 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 10.247 ± 0.962 | 6.593 ± 0.285 | 1.312 ± 0.209 | 9.236 ± 2.452 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 85.995 ± 11.596 | 41.381 ± 2.421 | 21.506 ± 7.935 | 23.582 ± 2.844 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.026 ± 0.010 | 0.020 ± 0.002 | 0.014 ± 0.010 | 0.037 ± 0.007 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.087 ± 0.006 | 0.055 ± 0.022 | 0.067 ± 0.029 | 0.076 ± 0.010 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 14.982 ± 0.943 | 10.123 ± 0.427 | 3.320 ± 0.522 | 6.204 ± 2.497 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 115.625 ± 16.650 | 60.691 ± 6.124 | 18.368 ± 5.011 | 33.576 ± 1.839 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 11.893 ± 3.855 | 9.015 ± 0.450 | 3.075 ± 0.118 | 5.610 ± 3.880 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 87.730 ± 1.621 | 56.019 ± 0.639 | 19.113 ± 0.841 | 34.190 ± 1.153 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.065 ± 0.001 | 0.081 ± 0.004 | 0.080 ± 0.011 | 0.054 ± 0.043 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.536 ± 0.217 | 1.284 ± 0.086 | 0.910 ± 0.812 | 0.743 ± 0.033 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 8.490 ± 0.706 | 12.064 ± 5.679 | 4.605 ± 2.268 | 2.413 ± 0.244 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 1.309 ± 0.539 | 1.402 ± 0.170 | 0.646 ± 0.081 | 0.657 ± 0.518 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 4.358 ± 0.096 | 9.891 ± 6.722 | 6.896 ± 2.775 | 2.277 ± 0.184 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.524 ± 0.007 | 0.798 ± 0.081 | 1.317 ± 1.130 | 0.694 ± 0.206 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 26.851 ± 1.573 | 23.088 ± 0.769 | 12.052 ± 3.866 | 16.209 ± 2.592 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 251.873 ± 35.166 | 151.403 ± 9.623 | 60.929 ± 10.133 | 81.103 ± 5.019 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 14.237 ± 1.304 | 22.629 ± 0.437 | 10.765 ± 4.506 | 16.168 ± 3.099 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 114.685 ± 19.432 | 147.581 ± 7.865 | 55.434 ± 0.618 | 74.309 ± 3.365 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.113 ± 0.011 | 0.107 ± 0.000 | 0.029 ± 0.028 | 0.164 ± 0.023 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.860 ± 0.017 | 0.842 ± 0.042 | 0.201 ± 0.053 | 1.100 ± 0.125 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 3.140 ± 0.022 | 3.267 ± 0.054 | 0.688 ± 0.082 | 3.922 ± 0.365 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.097 ± 0.013 | 0.090 ± 0.002 | 0.082 ± 0.010 | 0.085 ± 0.016 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.021 ± 0.008 | 0.021 ± 0.002 | 0.042 ± 0.035 | 0.045 ± 0.010 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.028 ± 0.001 | 0.026 ± 0.000 | 0.053 ± 0.040 | 0.080 ± 0.037 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.055 ± 0.000 | 0.043 ± 0.000 | 0.157 ± 0.135 | 0.323 ± 0.106 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.471 ± 0.011 | 0.465 ± 0.016 | 0.460 ± 0.034 | 0.805 ± 0.344 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.004 ± 0.001 | 0.003 ± 0.000 | 0.005 ± 0.003 | 0.008 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.005 ± 0.002 | 0.004 ± 0.002 | 0.006 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.016 ± 0.006 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.094 ± 0.018 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.003 ± 0.001 | 0.006 ± 0.001 | 0.008 ± 0.001 | 0.085 ± 0.006 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.001 | 0.013 ± 0.006 | 0.102 ± 0.015 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.133 ± 0.013 | 0.026 ± 0.014 | 0.103 ± 0.024 | 0.009 ± 0.011 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.121 ± 0.031 | 0.028 ± 0.025 | 0.070 ± 0.006 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.116 ± 0.003 | 0.024 ± 0.000 | 0.103 ± 0.020 | 0.023 ± 0.006 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.046 ± 0.040 | 0.019 ± 0.017 | 0.085 ± 0.041 | 0.012 ± 0.009 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.037 ± 0.002 | 0.024 ± 0.009 | 0.061 ± 0.004 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.042 ± 0.005 | 0.024 ± 0.000 | 0.112 ± 0.015 | 0.014 ± 0.005 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.667 ± 0.146 | 0.028 ± 0.008 | 0.129 ± 0.015 | 0.014 ± 0.005 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.378 ± 0.025 | 0.024 ± 0.006 | 0.100 ± 0.005 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.401 ± 0.039 | 0.023 ± 0.012 | 0.152 ± 0.010 | 0.018 ± 0.015 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.123 ± 0.035 | 0.025 ± 0.006 | 0.108 ± 0.016 | 0.011 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.074 ± 0.011 | 0.022 ± 0.013 | 0.085 ± 0.041 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.079 ± 0.013 | 0.019 ± 0.000 | 0.096 ± 0.061 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.039 ± 0.016 | 0.007 ± 0.000 | 0.091 ± 0.023 | 0.011 ± 0.003 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.033 ± 0.007 | 0.009 ± 0.002 | 0.025 ± 0.001 | 0.011 ± 0.003 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.034 ± 0.001 | 0.007 ± 0.001 | 0.034 ± 0.001 | 0.011 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.568 ± 0.052 | 0.030 ± 0.004 | 0.121 ± 0.078 | 0.012 ± 0.008 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.461 ± 0.146 | 0.038 ± 0.007 | 0.077 ± 0.002 | 0.010 ± 0.010 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.423 ± 0.033 | 0.029 ± 0.001 | 0.109 ± 0.014 | 0.010 ± 0.004 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.090 ± 0.008 | 0.029 ± 0.001 | 0.086 ± 0.012 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.074 ± 0.004 | 0.036 ± 0.018 | 0.078 ± 0.021 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.072 ± 0.004 | 0.028 ± 0.001 | 0.159 ± 0.081 | 0.008 ± 0.007 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.046 ± 0.005 | 0.024 ± 0.002 | 0.099 ± 0.082 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.042 ± 0.009 | 0.021 ± 0.004 | 0.039 ± 0.003 | 0.013 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.044 ± 0.001 | 0.022 ± 0.000 | 0.055 ± 0.002 | 0.012 ± 0.005 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.097 ± 0.020 | 0.026 ± 0.014 | 0.227 ± 0.027 | 0.012 ± 0.009 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.072 ± 0.001 | 0.015 ± 0.000 | 0.166 ± 0.039 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.071 ± 0.008 | 0.015 ± 0.001 | 0.721 ± 0.880 | 0.007 ± 0.003 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.043 ± 0.003 | 0.030 ± 0.007 | 0.092 ± 0.013 | 0.015 ± 0.007 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.036 ± 0.001 | 0.015 ± 0.003 | 0.087 ± 0.008 | 0.009 ± 0.004 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.037 ± 0.000 | 0.015 ± 0.000 | 0.137 ± 0.099 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.132 ± 0.030 | 0.013 ± 0.004 | 0.044 ± 0.069 | 0.014 ± 0.002 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.041 ± 0.005 | 0.019 ± 0.002 | 0.033 ± 0.003 | 0.019 ± 0.007 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.044 ± 0.001 | 0.022 ± 0.003 | 0.050 ± 0.005 | 0.022 ± 0.003 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.122 ± 0.129 | 0.013 ± 0.000 | 0.142 ± 0.172 | 0.012 ± 0.009 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.102 ± 0.033 | 0.015 ± 0.004 | 0.097 ± 0.007 | 0.019 ± 0.009 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.110 ± 0.008 | 0.024 ± 0.013 | 0.141 ± 0.052 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.042 ± 0.001 | 0.012 ± 0.000 | 0.072 ± 0.003 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.050 ± 0.061 | 0.015 ± 0.004 | 0.069 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.044 ± 0.003 | 0.022 ± 0.000 | 0.092 ± 0.022 | 0.022 ± 0.006 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.001 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.005 ± 0.009 | 0.005 ± 0.000 | 0.003 ± 0.001 | 0.011 ± 0.016 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.003 ± 0.001 | 0.001 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.031 ± 0.027 | 0.021 ± 0.001 | 0.010 ± 0.003 | 0.028 ± 0.003 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.002 ± 0.001 | 0.002 ± 0.001 | 0.001 ± 0.001 | 0.007 ± 0.002 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.007 ± 0.003 | 0.003 ± 0.000 | 0.004 ± 0.002 | 0.010 ± 0.005 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.004 ± 0.001 | 0.003 ± 0.001 | 0.003 ± 0.000 | 0.010 ± 0.002 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.002 | 0.012 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.006 ± 0.002 | 0.004 ± 0.000 | 0.013 ± 0.008 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.016 ± 0.014 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.004 ± 0.001 | 0.004 ± 0.002 | 0.009 ± 0.003 | 0.007 ± 0.002 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.002 | 0.011 ± 0.006 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.002 | 0.008 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.004 | 0.013 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.010 ± 0.005 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.011 ± 0.011 | 0.010 ± 0.001 | 0.019 ± 0.013 | 0.026 ± 0.004 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 16.1x faster than the slowest successful cell (`jax-cpu`, 0.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 48.9x faster than the slowest successful cell (`jax-cpu`, 0.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 30.9x faster than the slowest successful cell (`jax-cpu`, 0.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 39.6x faster than the slowest successful cell (`jax-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 16.6x faster than the slowest successful cell (`jax-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.111 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 31.7x faster than the slowest successful cell (`jax-cpu`, 0.111 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 19.2x faster than the slowest successful cell (`jax-cpu`, 0.111 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 23.1x faster than the slowest successful cell (`jax-cpu`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 35.4x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 33.5x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 41.6x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 21.8x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.0x faster than the slowest successful cell (`tenferro-eager`, 0.133 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 47.1x faster than the slowest successful cell (`tenferro-eager`, 0.667 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 23.4x faster than the slowest successful cell (`tenferro-eager`, 0.667 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 46.7x faster than the slowest successful cell (`tenferro-eager`, 0.378 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`tenferro-eager`, 0.378 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 22.7x faster than the slowest successful cell (`tenferro-eager`, 0.401 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`tenferro-eager`, 0.401 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.7x faster than the slowest successful cell (`tenferro-eager`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 45.8x faster than the slowest successful cell (`tenferro-eager`, 0.568 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 18.8x faster than the slowest successful cell (`tenferro-eager`, 0.568 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 47.6x faster than the slowest successful cell (`tenferro-eager`, 0.461 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`tenferro-eager`, 0.461 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 42.5x faster than the slowest successful cell (`tenferro-eager`, 0.423 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`tenferro-eager`, 0.423 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 0.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 104.4x faster than the slowest successful cell (`pytorch-cpu`, 0.721 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-eager` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.721 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 48.7x faster than the slowest successful cell (`pytorch-cpu`, 0.721 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.137 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 0.132 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261125_180002/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261125_180002/run.yaml`
- Timestamp: `2026-11-25T18:00:02.301847+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261125_180002`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261125_180002/cpu_ops_t4_20261125_180002.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.003 | 0.009 ± 0.001 | 0.008 ± 0.002 | 0.016 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.025 ± 0.016 | 0.026 ± 0.001 | 0.029 ± 0.025 | 0.070 ± 0.021 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.042 ± 0.016 | 0.028 ± 0.001 | 0.028 ± 0.001 | 0.037 ± 0.013 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.103 ± 0.021 | 0.103 ± 0.007 | 0.094 ± 0.004 | 0.123 ± 0.057 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.004 ± 0.001 | 0.003 ± 0.000 | 0.011 ± 0.005 | 0.100 ± 0.020 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.012 ± 0.007 | 0.084 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.100 ± 0.014 | 0.037 ± 0.000 | 0.102 ± 0.040 | 0.258 ± 0.041 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.002 | 0.003 ± 0.000 | 0.012 ± 0.001 | 0.134 ± 0.072 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | 0.093 ± 0.015 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.003 | 0.005 ± 0.000 | 0.064 ± 0.028 | 0.015 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.025 ± 0.015 | 0.015 ± 0.001 | 0.055 ± 0.009 | 0.026 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.013 ± 0.007 | 0.010 ± 0.001 | 0.057 ± 0.008 | 0.028 ± 0.012 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.027 ± 0.010 | 0.031 ± 0.002 | 0.093 ± 0.019 | 0.081 ± 0.032 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.003 | 0.007 ± 0.000 | 0.062 ± 0.016 | 0.009 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.011 ± 0.004 | 0.010 ± 0.001 | 0.084 ± 0.048 | 0.027 ± 0.010 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.010 | 0.007 ± 0.000 | 0.058 ± 0.015 | 0.013 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.022 ± 0.009 | 0.015 ± 0.002 | 0.084 ± 0.008 | 0.046 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.015 ± 0.005 | 0.012 ± 0.001 | 0.015 ± 0.002 | 0.054 ± 0.020 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.056 ± 0.024 | 0.036 ± 0.007 | 0.059 ± 0.015 | 0.131 ± 0.020 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.062 ± 0.020 | 0.040 ± 0.003 | 0.053 ± 0.003 | 0.093 ± 0.069 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.169 ± 0.088 | 0.158 ± 0.004 | 0.170 ± 0.074 | 0.284 ± 0.064 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.070 ± 0.093 | 0.010 ± 0.001 | 0.083 ± 0.091 | 0.009 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.041 ± 0.003 | 0.014 ± 0.001 | 0.069 ± 0.013 | 0.012 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.151 ± 0.016 | 0.011 ± 0.002 | 0.093 ± 0.022 | 0.030 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.144 ± 0.195 | 0.019 ± 0.000 | 0.081 ± 0.007 | 0.070 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.049 ± 0.001 | 0.032 ± 0.001 | 0.159 ± 0.092 | 0.014 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.054 ± 0.001 | 0.049 ± 0.002 | 0.126 ± 0.058 | 0.043 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.054 ± 0.001 | 0.036 ± 0.002 | 0.139 ± 0.043 | 0.024 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.070 ± 0.002 | 0.075 ± 0.011 | 0.142 ± 0.019 | 0.063 ± 0.009 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.305 ± 0.015 | 0.327 ± 0.126 | 0.318 ± 0.206 | 0.385 ± 0.252 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 16.804 ± 1.643 | 12.245 ± 0.777 | 5.665 ± 0.142 | 6.620 ± 0.317 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 91.042 ± 3.049 | 71.507 ± 17.997 | 35.591 ± 1.669 | 37.689 ± 2.347 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 9.662 ± 1.047 | 12.215 ± 0.605 | 5.030 ± 0.191 | 7.245 ± 0.530 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 62.727 ± 9.025 | 68.165 ± 19.798 | 32.167 ± 2.234 | 35.575 ± 1.113 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 14.144 ± 7.947 | 5.970 ± 2.906 | 3.527 ± 0.905 | 2.784 ± 0.203 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 63.794 ± 8.114 | 32.298 ± 2.641 | 18.504 ± 0.671 | 16.816 ± 8.098 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 9.840 ± 4.356 | 6.188 ± 3.854 | 1.997 ± 0.140 | 2.576 ± 0.198 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 64.804 ± 13.086 | 30.845 ± 0.946 | 11.392 ± 1.687 | 17.876 ± 8.677 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.054 ± 0.005 | 0.037 ± 0.008 | 0.014 ± 0.019 | 0.042 ± 0.010 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.263 ± 0.162 | 0.087 ± 0.002 | 0.080 ± 0.051 | 0.098 ± 0.016 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 13.601 ± 1.662 | 9.449 ± 0.827 | 3.198 ± 0.111 | 5.274 ± 2.483 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 87.789 ± 11.859 | 67.362 ± 9.002 | 26.014 ± 16.528 | 23.012 ± 2.538 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 8.268 ± 1.205 | 9.140 ± 0.933 | 3.055 ± 0.283 | 5.430 ± 1.265 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 61.242 ± 4.026 | 60.598 ± 12.409 | 17.433 ± 1.045 | 25.401 ± 5.900 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.068 ± 0.001 | 0.084 ± 0.007 | 0.073 ± 0.016 | 0.068 ± 0.024 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 2.753 ± 0.301 | 1.921 ± 0.500 | 0.525 ± 0.347 | 0.389 ± 0.080 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 12.932 ± 5.541 | 8.806 ± 0.200 | 2.770 ± 0.532 | 3.343 ± 1.492 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 1.023 ± 0.053 | 1.964 ± 0.764 | 0.694 ± 0.019 | 0.902 ± 0.359 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 5.695 ± 3.252 | 8.925 ± 0.152 | 3.917 ± 0.259 | 1.862 ± 0.225 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 1.760 ± 1.165 | 1.011 ± 0.054 | 0.454 ± 0.007 | 0.506 ± 0.209 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 29.990 ± 6.095 | 21.992 ± 0.963 | 10.691 ± 0.050 | 11.393 ± 1.163 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 165.301 ± 6.530 | 128.953 ± 16.119 | 85.012 ± 11.511 | 83.365 ± 9.413 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 14.601 ± 3.233 | 23.276 ± 8.639 | 9.972 ± 3.744 | 11.253 ± 3.256 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 92.573 ± 2.157 | 134.196 ± 25.976 | 70.085 ± 7.879 | 67.067 ± 5.758 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.170 ± 0.022 | 0.159 ± 0.018 | 0.029 ± 0.002 | 0.199 ± 0.014 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.762 ± 0.189 | 0.669 ± 0.034 | 0.193 ± 0.016 | 0.677 ± 0.056 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 2.205 ± 0.318 | 3.162 ± 0.835 | 0.750 ± 2.047 | 2.363 ± 0.048 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.151 ± 0.041 | 0.133 ± 0.026 | 0.127 ± 0.162 | 0.173 ± 0.077 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.061 ± 0.004 | 0.025 ± 0.001 | 0.035 ± 0.002 | 0.068 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.028 ± 0.012 | 0.028 ± 0.021 | 0.051 ± 0.011 | 0.084 ± 0.047 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.057 ± 0.000 | 0.044 ± 0.007 | 0.069 ± 0.047 | 0.204 ± 0.037 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.525 ± 0.083 | 0.592 ± 0.158 | 0.407 ± 0.006 | 0.447 ± 0.128 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.003 ± 0.001 | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.005 ± 0.001 | 0.004 ± 0.000 | 0.006 ± 0.004 | 0.010 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.009 ± 0.006 | 0.008 ± 0.001 | 0.009 ± 0.000 | 0.018 ± 0.007 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.005 ± 0.003 | 0.012 ± 0.003 | 0.071 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.006 ± 0.002 | 0.016 ± 0.008 | 0.077 ± 0.010 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.003 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.106 ± 0.022 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.122 ± 0.022 | 0.022 ± 0.013 | 0.077 ± 0.029 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.183 ± 0.046 | 0.024 ± 0.002 | 0.077 ± 0.016 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.115 ± 0.008 | 0.030 ± 0.003 | 0.097 ± 0.009 | 0.021 ± 0.004 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.037 ± 0.002 | 0.022 ± 0.020 | 0.076 ± 0.147 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.049 ± 0.014 | 0.023 ± 0.006 | 0.074 ± 0.078 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.044 ± 0.002 | 0.028 ± 0.001 | 0.093 ± 0.014 | 0.024 ± 0.005 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.436 ± 0.035 | 0.022 ± 0.001 | 0.557 ± 0.397 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.407 ± 0.150 | 0.021 ± 0.005 | 0.713 ± 0.415 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.407 ± 0.040 | 0.024 ± 0.001 | 0.732 ± 0.069 | 0.013 ± 0.005 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.095 ± 0.016 | 0.020 ± 0.001 | 0.431 ± 0.177 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.146 ± 0.044 | 0.022 ± 0.001 | 0.604 ± 0.215 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.099 ± 0.017 | 0.023 ± 0.002 | 0.476 ± 0.190 | 0.020 ± 0.006 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.043 ± 0.012 | 0.009 ± 0.005 | 0.031 ± 0.010 | 0.009 ± 0.004 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.040 ± 0.037 | 0.009 ± 0.003 | 0.027 ± 0.008 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.117 ± 0.021 | 0.009 ± 0.001 | 0.032 ± 0.002 | 0.013 ± 0.009 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.459 ± 0.028 | 0.041 ± 0.008 | 0.216 ± 0.096 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.652 ± 0.132 | 0.033 ± 0.001 | 0.157 ± 0.034 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.518 ± 0.121 | 0.031 ± 0.003 | 0.194 ± 0.025 | 0.016 ± 0.002 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.080 ± 0.003 | 0.032 ± 0.008 | 0.203 ± 0.061 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.115 ± 0.059 | 0.032 ± 0.002 | 0.175 ± 0.027 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.099 ± 0.014 | 0.034 ± 0.003 | 0.179 ± 0.009 | 0.016 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.051 ± 0.011 | 0.024 ± 0.008 | 0.040 ± 0.007 | 0.015 ± 0.005 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.056 ± 0.086 | 0.026 ± 0.002 | 0.043 ± 0.004 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.050 ± 0.007 | 0.026 ± 0.001 | 0.044 ± 0.003 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.097 ± 0.020 | 0.020 ± 0.003 | 0.311 ± 0.159 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.135 ± 0.051 | 0.018 ± 0.001 | 0.222 ± 0.155 | 0.008 ± 0.004 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.082 ± 0.003 | 0.018 ± 0.001 | 0.200 ± 0.014 | 0.026 ± 0.016 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.042 ± 0.002 | 0.017 ± 0.000 | 0.100 ± 0.046 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.065 ± 0.027 | 0.018 ± 0.001 | 0.096 ± 0.051 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.040 ± 0.001 | 0.018 ± 0.001 | 0.089 ± 0.010 | 0.009 ± 0.008 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.048 ± 0.045 | 0.013 ± 0.002 | 0.037 ± 0.009 | 0.015 ± 0.005 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.058 ± 0.014 | 0.017 ± 0.004 | 0.034 ± 0.001 | 0.017 ± 0.003 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.075 ± 0.039 | 0.030 ± 0.008 | 0.043 ± 0.011 | 0.047 ± 0.015 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.107 ± 0.010 | 0.013 ± 0.002 | 0.098 ± 0.055 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.149 ± 0.051 | 0.020 ± 0.002 | 0.111 ± 0.026 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.128 ± 0.028 | 0.027 ± 0.005 | 0.178 ± 0.116 | 0.040 ± 0.023 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.038 ± 0.003 | 0.013 ± 0.001 | 0.149 ± 0.064 | 0.012 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.067 ± 0.012 | 0.017 ± 0.002 | 0.066 ± 0.004 | 0.014 ± 0.003 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.062 ± 0.028 | 0.027 ± 0.009 | 0.090 ± 0.004 | 0.043 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.003 ± 0.002 | 0.002 ± 0.001 | 0.005 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.006 ± 0.001 | 0.008 ± 0.003 | 0.003 ± 0.000 | 0.015 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.002 ± 0.004 | 0.006 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.030 ± 0.002 | 0.041 ± 0.013 | 0.008 ± 0.001 | 0.059 ± 0.013 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.003 ± 0.001 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.004 ± 0.002 | 0.004 ± 0.003 | 0.046 ± 0.012 | 0.007 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.003 ± 0.001 | 0.049 ± 0.007 | 0.015 ± 0.010 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.005 ± 0.001 | 0.004 ± 0.000 | 0.043 ± 0.033 | 0.016 ± 0.011 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.003 | 0.009 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.004 ± 0.001 | 0.005 ± 0.002 | 0.009 ± 0.000 | 0.007 ± 0.003 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.004 ± 0.001 | 0.005 ± 0.001 | 0.012 ± 0.005 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.005 ± 0.002 | 0.005 ± 0.000 | 0.014 ± 0.003 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.006 ± 0.001 | 0.005 ± 0.000 | 0.009 ± 0.005 | 0.007 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.008 ± 0.002 | 0.005 ± 0.000 | 0.011 ± 0.006 | 0.006 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.001 | 0.006 ± 0.002 | 0.012 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.006 ± 0.001 | 0.006 ± 0.003 | 0.009 ± 0.002 | 0.017 ± 0.007 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.017 ± 0.003 | 0.014 ± 0.001 | 0.015 ± 0.001 | 0.030 ± 0.007 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 26.9x faster than the slowest successful cell (`jax-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 35.7x faster than the slowest successful cell (`jax-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 20.8x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 22.1x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 38.6x faster than the slowest successful cell (`jax-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 40.3x faster than the slowest successful cell (`jax-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 18.3x faster than the slowest successful cell (`jax-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 18.8x faster than the slowest successful cell (`jax-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 14.1x faster than the slowest successful cell (`tenferro-eager`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 25.7x faster than the slowest successful cell (`jax-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 32.3x faster than the slowest successful cell (`jax-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 30.5x faster than the slowest successful cell (`jax-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 20.0x faster than the slowest successful cell (`jax-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 17.6x faster than the slowest successful cell (`tenferro-eager`, 0.183 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 69.8x faster than the slowest successful cell (`pytorch-cpu`, 0.557 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 25.1x faster than the slowest successful cell (`pytorch-cpu`, 0.557 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 80.7x faster than the slowest successful cell (`pytorch-cpu`, 0.713 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 34.5x faster than the slowest successful cell (`pytorch-cpu`, 0.713 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 57.1x faster than the slowest successful cell (`pytorch-cpu`, 0.732 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 30.0x faster than the slowest successful cell (`pytorch-cpu`, 0.732 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 54.3x faster than the slowest successful cell (`pytorch-cpu`, 0.431 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 21.0x faster than the slowest successful cell (`pytorch-cpu`, 0.431 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 63.4x faster than the slowest successful cell (`pytorch-cpu`, 0.604 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 27.9x faster than the slowest successful cell (`pytorch-cpu`, 0.604 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 24.0x faster than the slowest successful cell (`pytorch-cpu`, 0.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`pytorch-cpu`, 0.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 45.2x faster than the slowest successful cell (`tenferro-eager`, 0.459 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`tenferro-eager`, 0.459 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 65.2x faster than the slowest successful cell (`tenferro-eager`, 0.652 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 20.0x faster than the slowest successful cell (`tenferro-eager`, 0.652 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 32.5x faster than the slowest successful cell (`tenferro-eager`, 0.518 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 0.518 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 22.7x faster than the slowest successful cell (`pytorch-cpu`, 0.203 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 17.2x faster than the slowest successful cell (`pytorch-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 41.5x faster than the slowest successful cell (`pytorch-cpu`, 0.311 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.311 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 27.3x faster than the slowest successful cell (`pytorch-cpu`, 0.222 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.222 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-eager`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

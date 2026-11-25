# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `74775e4a2ad3480fb9ee9f859e0037b8cac731f2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261125_175248/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261125_175248/run.yaml`
- Timestamp: `2026-11-25T17:52:48.228690+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261125_175248`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

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

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/amd-cpu/cpu/einsum/20261125_175248/cpu_ops_t1_20261125_175248.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.018 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.034 ± 0.001 | 0.034 ± 0.000 | 0.041 ± 0.001 | 0.032 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.035 ± 0.000 | 0.034 ± 0.000 | 0.039 ± 0.001 | 0.043 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.131 ± 0.000 | 0.129 ± 0.001 | 0.133 ± 0.001 | 0.146 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.000 | 0.179 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.016 ± 0.000 | 0.190 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.225 ± 0.001 | 0.227 ± 0.001 | 0.176 ± 0.003 | 0.204 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.190 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.025 ± 0.001 | 0.189 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.017 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.018 ± 0.000 | 0.020 ± 0.000 | 0.030 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.025 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.046 ± 0.000 | 0.045 ± 0.000 | 0.040 ± 0.001 | 0.060 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.019 ± 0.000 | 0.015 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.024 ± 0.001 | 0.022 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.001 | 0.018 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.024 ± 0.000 | 0.024 ± 0.000 | 0.027 ± 0.000 | 0.049 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.020 ± 0.000 | 0.019 ± 0.000 | 0.028 ± 0.001 | 0.035 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.066 ± 0.001 | 0.065 ± 0.001 | 0.080 ± 0.001 | 0.095 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.055 ± 0.000 | 0.054 ± 0.001 | 0.076 ± 0.002 | 0.077 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.207 ± 0.001 | 0.205 ± 0.001 | 0.262 ± 0.002 | 0.247 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.088 ± 0.002 | 0.022 ± 0.000 | 0.141 ± 0.001 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.121 ± 0.006 | 0.055 ± 0.000 | 0.144 ± 0.001 | 0.015 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.089 ± 0.001 | 0.022 ± 0.000 | 0.148 ± 0.006 | 0.015 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.126 ± 0.003 | 0.056 ± 0.000 | 0.174 ± 0.009 | 0.052 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.131 ± 0.001 | 0.054 ± 0.000 | 0.132 ± 0.002 | 0.025 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.152 ± 0.001 | 0.093 ± 0.000 | 0.145 ± 0.003 | 0.040 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.134 ± 0.001 | 0.061 ± 0.001 | 0.141 ± 0.003 | 0.024 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.163 ± 0.004 | 0.115 ± 0.001 | 0.152 ± 0.004 | 0.060 ± 0.013 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.354 ± 0.001 | 0.352 ± 0.000 | 0.325 ± 0.006 | 0.384 ± 0.007 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 16.347 ± 0.090 | 12.097 ± 0.109 | 10.420 ± 0.181 | 8.889 ± 0.061 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 89.850 ± 0.693 | 66.374 ± 0.191 | 66.581 ± 0.504 | 58.320 ± 0.458 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 8.966 ± 0.060 | 12.005 ± 0.075 | 8.538 ± 0.108 | 8.993 ± 0.032 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 55.056 ± 0.819 | 65.933 ± 0.628 | 52.279 ± 0.440 | 58.115 ± 0.221 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 9.720 ± 0.086 | 4.784 ± 0.022 | 12.431 ± 0.315 | 4.345 ± 0.090 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 71.865 ± 1.130 | 34.295 ± 0.202 | 79.900 ± 0.646 | 31.439 ± 1.514 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 8.935 ± 0.192 | 5.056 ± 0.057 | 5.437 ± 0.099 | 4.402 ± 0.049 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 68.942 ± 0.390 | 39.059 ± 0.484 | 42.656 ± 0.701 | 32.865 ± 2.201 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.028 ± 0.002 | 0.019 ± 0.000 | 0.033 ± 0.001 | 0.034 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.148 ± 0.017 | 0.054 ± 0.001 | 0.151 ± 0.003 | 0.084 ± 0.003 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 14.605 ± 0.660 | 8.227 ± 0.092 | 7.425 ± 0.071 | 7.172 ± 0.087 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 106.820 ± 1.519 | 60.035 ± 0.355 | 52.373 ± 0.800 | 45.843 ± 2.429 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 11.202 ± 0.112 | 8.235 ± 0.026 | 7.389 ± 0.207 | 7.185 ± 0.044 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 92.257 ± 2.636 | 59.761 ± 1.799 | 52.075 ± 0.523 | 50.766 ± 3.406 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.161 ± 0.008 | 0.126 ± 0.001 | 0.174 ± 0.006 | 0.073 ± 0.015 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.324 ± 0.020 | 1.096 ± 0.593 | 1.235 ± 0.020 | 0.677 ± 0.019 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 6.635 ± 0.107 | 6.411 ± 0.074 | 5.529 ± 0.093 | 4.165 ± 0.057 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.725 ± 0.012 | 1.121 ± 0.005 | 1.721 ± 0.031 | 0.658 ± 0.004 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 3.557 ± 0.692 | 6.378 ± 0.047 | 9.634 ± 0.697 | 4.050 ± 0.029 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.722 ± 0.020 | 0.853 ± 0.013 | 0.781 ± 0.008 | 0.653 ± 0.003 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 27.396 ± 0.237 | 20.393 ± 0.100 | 21.055 ± 0.269 | 15.639 ± 0.126 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 179.224 ± 0.459 | 132.675 ± 1.156 | 134.890 ± 0.163 | 110.251 ± 0.642 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 14.666 ± 0.318 | 20.287 ± 0.267 | 16.049 ± 0.098 | 15.603 ± 0.055 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 94.141 ± 0.397 | 130.002 ± 0.649 | 107.698 ± 0.623 | 116.704 ± 1.164 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.097 ± 0.001 | 0.097 ± 0.000 | 0.201 ± 0.004 | 0.194 ± 0.068 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.733 ± 0.189 | 0.716 ± 0.007 | 1.076 ± 0.021 | 1.486 ± 0.027 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 3.010 ± 0.020 | 3.027 ± 0.048 | 3.651 ± 0.125 | 4.445 ± 0.096 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.111 ± 0.000 | 0.110 ± 0.000 | 0.116 ± 0.002 | 0.109 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.027 ± 0.000 | 0.034 ± 0.000 | 0.053 ± 0.001 | 0.045 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.036 ± 0.001 | 0.039 ± 0.003 | 0.069 ± 0.003 | 0.074 ± 0.007 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.070 ± 0.000 | 0.052 ± 0.000 | 0.121 ± 0.004 | 0.157 ± 0.007 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.561 ± 0.009 | 0.556 ± 0.002 | 0.635 ± 0.015 | 0.588 ± 0.006 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.171 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.177 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.188 ± 0.032 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.249 ± 0.003 | 0.033 ± 0.000 | 0.195 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.261 ± 0.003 | 0.039 ± 0.000 | 0.198 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.270 ± 0.002 | 0.048 ± 0.000 | 0.206 ± 0.008 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.095 ± 0.002 | 0.034 ± 0.000 | 0.192 ± 0.004 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.101 ± 0.002 | 0.040 ± 0.000 | 0.198 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.108 ± 0.002 | 0.049 ± 0.001 | 0.205 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.685 ± 0.008 | 0.033 ± 0.000 | 0.274 ± 0.006 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.696 ± 0.009 | 0.034 ± 0.000 | 0.271 ± 0.007 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.692 ± 0.005 | 0.037 ± 0.001 | 0.276 ± 0.006 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.178 ± 0.006 | 0.031 ± 0.000 | 0.255 ± 0.007 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.178 ± 0.004 | 0.032 ± 0.002 | 0.258 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.181 ± 0.004 | 0.035 ± 0.000 | 0.259 ± 0.006 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.076 ± 0.003 | 0.012 ± 0.000 | 0.070 ± 0.001 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.075 ± 0.001 | 0.012 ± 0.000 | 0.073 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.077 ± 0.004 | 0.012 ± 0.000 | 0.073 ± 0.001 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.797 ± 0.006 | 0.052 ± 0.001 | 0.216 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.815 ± 0.007 | 0.053 ± 0.001 | 0.216 ± 0.006 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.818 ± 0.012 | 0.058 ± 0.011 | 0.221 ± 0.003 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.171 ± 0.002 | 0.056 ± 0.000 | 0.246 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.171 ± 0.003 | 0.057 ± 0.000 | 0.245 ± 0.007 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.177 ± 0.003 | 0.063 ± 0.001 | 0.247 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.120 ± 0.002 | 0.040 ± 0.001 | 0.118 ± 0.002 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.120 ± 0.001 | 0.040 ± 0.000 | 0.121 ± 0.001 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.121 ± 0.001 | 0.043 ± 0.000 | 0.120 ± 0.000 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.187 ± 0.002 | 0.028 ± 0.000 | 0.408 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.190 ± 0.001 | 0.028 ± 0.000 | 0.414 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.193 ± 0.002 | 0.030 ± 0.000 | 0.410 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.100 ± 0.002 | 0.028 ± 0.000 | 0.244 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.101 ± 0.002 | 0.029 ± 0.000 | 0.244 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.105 ± 0.001 | 0.030 ± 0.000 | 0.245 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.102 ± 0.003 | 0.020 ± 0.000 | 0.105 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.106 ± 0.010 | 0.024 ± 0.000 | 0.113 ± 0.001 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.113 ± 0.002 | 0.036 ± 0.000 | 0.124 ± 0.001 | 0.021 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.222 ± 0.004 | 0.021 ± 0.000 | 0.261 ± 0.009 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.228 ± 0.002 | 0.025 ± 0.000 | 0.268 ± 0.001 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.244 ± 0.003 | 0.036 ± 0.000 | 0.282 ± 0.005 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.087 ± 0.001 | 0.020 ± 0.000 | 0.212 ± 0.009 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.091 ± 0.003 | 0.024 ± 0.001 | 0.218 ± 0.003 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.100 ± 0.002 | 0.036 ± 0.000 | 0.228 ± 0.005 | 0.019 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.017 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.041 ± 0.001 | 0.039 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.024 ± 0.000 | 0.027 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 26.7x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 24.9x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 28.1x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 26.7x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 47.8x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 23.1x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 49.8x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 23.8x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 14.1x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 51.1x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 24.8x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 21.6x faster than the slowest successful cell (`tenferro-eager`, 0.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 19.3x faster than the slowest successful cell (`tenferro-eager`, 0.261 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 20.9x faster than the slowest successful cell (`tenferro-eager`, 0.270 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.192 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 53.7x faster than the slowest successful cell (`tenferro-eager`, 0.685 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 20.7x faster than the slowest successful cell (`tenferro-eager`, 0.685 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 53.1x faster than the slowest successful cell (`tenferro-eager`, 0.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 20.7x faster than the slowest successful cell (`tenferro-eager`, 0.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 47.6x faster than the slowest successful cell (`tenferro-eager`, 0.692 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 18.9x faster than the slowest successful cell (`tenferro-eager`, 0.692 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 0.258 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.259 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 61.9x faster than the slowest successful cell (`tenferro-eager`, 0.797 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`tenferro-eager`, 0.797 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 59.6x faster than the slowest successful cell (`tenferro-eager`, 0.815 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 0.815 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 48.5x faster than the slowest successful cell (`tenferro-eager`, 0.818 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 14.0x faster than the slowest successful cell (`tenferro-eager`, 0.818 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.246 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 0.245 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.247 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 32.2x faster than the slowest successful cell (`pytorch-cpu`, 0.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 32.8x faster than the slowest successful cell (`pytorch-cpu`, 0.414 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.414 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 29.0x faster than the slowest successful cell (`pytorch-cpu`, 0.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 16.6x faster than the slowest successful cell (`pytorch-cpu`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.245 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.4x faster than the slowest successful cell (`pytorch-cpu`, 0.261 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.261 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 0.268 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.268 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.282 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.218 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.228 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261125_183303/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261125_183303/run.yaml`
- Timestamp: `2026-11-25T18:33:03.871566+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261125_183303`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

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

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/amd-cpu/cpu/einsum/20261125_183303/cpu_ops_t4_20261125_183303.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.000 | 0.019 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.036 ± 0.000 | 0.036 ± 0.000 | 0.040 ± 0.001 | 0.046 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.036 ± 0.000 | 0.035 ± 0.000 | 0.040 ± 0.001 | 0.044 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.132 ± 0.000 | 0.130 ± 0.000 | 0.132 ± 0.001 | 0.139 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.176 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.016 ± 0.000 | 0.188 ± 0.040 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.228 ± 0.001 | 0.227 ± 0.000 | 0.122 ± 0.002 | 0.272 ± 0.058 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.194 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.025 ± 0.000 | 0.235 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.016 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.019 ± 0.000 | 0.032 ± 0.000 | 0.030 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.023 ± 0.001 | 0.025 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.047 ± 0.000 | 0.046 ± 0.000 | 0.059 ± 0.001 | 0.065 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.030 ± 0.001 | 0.016 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.036 ± 0.001 | 0.032 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.050 ± 0.005 | 0.018 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.025 ± 0.000 | 0.026 ± 0.000 | 0.039 ± 0.001 | 0.042 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.021 ± 0.000 | 0.021 ± 0.000 | 0.028 ± 0.001 | 0.036 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.067 ± 0.001 | 0.066 ± 0.000 | 0.076 ± 0.001 | 0.094 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.056 ± 0.000 | 0.055 ± 0.001 | 0.075 ± 0.001 | 0.077 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.207 ± 0.000 | 0.208 ± 0.002 | 0.260 ± 0.005 | 0.255 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.094 ± 0.001 | 0.025 ± 0.000 | 0.142 ± 0.000 | 0.013 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.126 ± 0.002 | 0.058 ± 0.001 | 0.145 ± 0.003 | 0.017 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.095 ± 0.004 | 0.025 ± 0.000 | 0.148 ± 0.002 | 0.022 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.132 ± 0.003 | 0.058 ± 0.000 | 0.173 ± 0.011 | 0.060 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.139 ± 0.001 | 0.067 ± 0.000 | 0.176 ± 0.002 | 0.025 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.160 ± 0.009 | 0.105 ± 0.000 | 0.186 ± 0.001 | 0.055 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.141 ± 0.002 | 0.073 ± 0.000 | 0.183 ± 0.002 | 0.030 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.170 ± 0.001 | 0.130 ± 0.000 | 0.199 ± 0.003 | 0.094 ± 0.004 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.358 ± 0.007 | 0.352 ± 0.001 | 0.346 ± 0.006 | 0.428 ± 0.003 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 18.777 ± 0.170 | 14.716 ± 0.267 | 7.158 ± 5.914 | 10.984 ± 1.008 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 89.712 ± 0.481 | 64.909 ± 0.632 | 41.898 ± 1.089 | 49.952 ± 2.975 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 10.990 ± 0.107 | 14.799 ± 0.036 | 6.059 ± 0.280 | 10.994 ± 1.414 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 49.525 ± 0.968 | 64.863 ± 0.928 | 30.238 ± 1.429 | 57.418 ± 4.442 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 7.511 ± 0.063 | 3.921 ± 0.026 | 7.370 ± 0.191 | 5.721 ± 0.812 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 40.193 ± 0.133 | 21.322 ± 0.041 | 47.786 ± 0.621 | 31.654 ± 4.598 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 6.097 ± 0.069 | 3.972 ± 0.011 | 3.580 ± 0.039 | 6.187 ± 0.501 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 36.908 ± 0.066 | 23.524 ± 0.383 | 24.505 ± 0.238 | 36.024 ± 3.821 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.054 ± 0.001 | 0.038 ± 0.000 | 0.088 ± 0.304 | 0.038 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.215 ± 0.021 | 0.102 ± 0.002 | 0.176 ± 0.007 | 0.081 ± 0.006 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 13.371 ± 0.069 | 9.341 ± 0.176 | 4.832 ± 0.255 | 8.988 ± 1.928 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 58.522 ± 1.138 | 33.434 ± 0.398 | 33.477 ± 0.430 | 47.860 ± 2.041 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 9.067 ± 0.060 | 9.317 ± 0.243 | 5.388 ± 1.865 | 8.967 ± 1.217 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 48.256 ± 1.060 | 33.787 ± 0.219 | 32.802 ± 0.322 | 55.978 ± 7.049 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.217 ± 0.022 | 0.136 ± 0.001 | 0.178 ± 0.005 | 0.066 ± 0.004 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 1.481 ± 0.009 | 1.251 ± 0.034 | 2.932 ± 3.892 | 0.902 ± 0.146 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 5.388 ± 0.099 | 5.003 ± 0.107 | 3.207 ± 0.570 | 3.793 ± 0.085 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.870 ± 0.024 | 1.256 ± 0.024 | 1.242 ± 0.023 | 0.987 ± 0.058 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 2.954 ± 0.093 | 4.980 ± 0.007 | 4.784 ± 0.060 | 3.732 ± 0.038 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 1.019 ± 0.004 | 1.211 ± 0.009 | 0.993 ± 0.026 | 0.907 ± 0.050 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 29.287 ± 0.216 | 22.302 ± 0.114 | 16.444 ± 0.085 | 17.360 ± 0.922 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 175.841 ± 0.811 | 126.839 ± 0.580 | 94.623 ± 1.097 | 107.797 ± 10.801 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 16.338 ± 0.096 | 22.198 ± 0.096 | 14.849 ± 0.300 | 17.647 ± 0.626 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 90.115 ± 0.977 | 126.274 ± 1.515 | 83.882 ± 0.483 | 105.779 ± 10.787 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.073 ± 0.001 | 0.073 ± 0.001 | 0.155 ± 0.006 | 0.161 ± 0.042 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.396 ± 0.002 | 0.397 ± 0.004 | 0.665 ± 0.021 | 0.755 ± 0.081 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 1.584 ± 0.021 | 1.641 ± 3.381 | 2.606 ± 0.175 | 3.183 ± 0.185 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.112 ± 0.000 | 0.111 ± 0.003 | 0.177 ± 0.005 | 0.115 ± 0.005 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.028 ± 0.000 | 0.036 ± 0.001 | 0.051 ± 0.001 | 0.058 ± 0.001 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.049 ± 0.002 | 0.052 ± 0.003 | 0.099 ± 0.013 | 0.154 ± 0.133 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.081 ± 0.008 | 0.066 ± 0.001 | 0.134 ± 0.005 | 0.195 ± 0.006 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.566 ± 0.003 | 0.559 ± 0.010 | 0.827 ± 0.005 | 0.870 ± 0.025 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.174 ± 0.007 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.177 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.191 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.258 ± 0.004 | 0.043 ± 0.000 | 0.197 ± 0.024 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.271 ± 0.029 | 0.048 ± 0.000 | 0.196 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.284 ± 0.001 | 0.058 ± 0.001 | 0.204 ± 0.005 | 0.018 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.103 ± 0.001 | 0.043 ± 0.000 | 0.189 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.113 ± 0.002 | 0.048 ± 0.001 | 0.196 ± 0.033 | 0.016 ± 0.004 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.114 ± 0.001 | 0.058 ± 0.001 | 0.200 ± 0.003 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.716 ± 0.003 | 0.044 ± 0.001 | 0.371 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.746 ± 0.020 | 0.045 ± 0.001 | 0.371 ± 0.010 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.733 ± 0.011 | 0.047 ± 0.000 | 0.374 ± 0.004 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.203 ± 0.052 | 0.041 ± 0.000 | 0.328 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.202 ± 0.002 | 0.042 ± 0.000 | 0.327 ± 0.008 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.206 ± 0.001 | 0.045 ± 0.001 | 0.330 ± 0.010 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.081 ± 0.006 | 0.015 ± 0.000 | 0.072 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.080 ± 0.001 | 0.015 ± 0.000 | 0.074 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.081 ± 0.001 | 0.015 ± 0.000 | 0.075 ± 0.002 | 0.015 ± 0.005 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.838 ± 0.009 | 0.067 ± 0.001 | 0.281 ± 0.006 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.851 ± 0.006 | 0.068 ± 0.000 | 0.279 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.863 ± 0.005 | 0.074 ± 0.001 | 0.283 ± 0.004 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.196 ± 0.001 | 0.070 ± 0.000 | 0.310 ± 0.003 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.197 ± 0.002 | 0.071 ± 0.001 | 0.310 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.205 ± 0.003 | 0.076 ± 0.001 | 0.315 ± 0.010 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.125 ± 0.003 | 0.052 ± 0.000 | 0.118 ± 0.000 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.128 ± 0.005 | 0.052 ± 0.000 | 0.123 ± 0.003 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.129 ± 0.007 | 0.055 ± 0.000 | 0.122 ± 0.020 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.201 ± 0.006 | 0.036 ± 0.000 | 0.398 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.198 ± 0.002 | 0.036 ± 0.000 | 0.403 ± 0.009 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.201 ± 0.001 | 0.039 ± 0.001 | 0.400 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.108 ± 0.003 | 0.036 ± 0.000 | 0.238 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.107 ± 0.000 | 0.037 ± 0.000 | 0.239 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.109 ± 0.000 | 0.039 ± 0.000 | 0.242 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.107 ± 0.007 | 0.027 ± 0.001 | 0.106 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.110 ± 0.004 | 0.031 ± 0.000 | 0.114 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.119 ± 0.003 | 0.042 ± 0.000 | 0.126 ± 0.001 | 0.031 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.223 ± 0.001 | 0.027 ± 0.000 | 0.258 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.232 ± 0.004 | 0.032 ± 0.000 | 0.263 ± 0.006 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.249 ± 0.002 | 0.044 ± 0.006 | 0.277 ± 0.005 | 0.032 ± 0.007 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.092 ± 0.001 | 0.026 ± 0.000 | 0.208 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.097 ± 0.003 | 0.031 ± 0.000 | 0.215 ± 0.002 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.105 ± 0.001 | 0.042 ± 0.000 | 0.226 ± 0.006 | 0.032 ± 0.006 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.031 ± 0.000 | 0.031 ± 0.000 | 0.035 ± 0.001 | 0.038 ± 0.006 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.012 ± 0.004 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.006 ± 0.002 | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.024 ± 0.000 | 0.028 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 23.5x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 22.5x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 24.5x faster than the slowest successful cell (`jax-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 24.9x faster than the slowest successful cell (`jax-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 13.0x faster than the slowest successful cell (`jax-cpu`, 0.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 40.5x faster than the slowest successful cell (`jax-cpu`, 0.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 20.8x faster than the slowest successful cell (`jax-cpu`, 0.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 40.1x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 21.0x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 41.0x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 22.9x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 22.6x faster than the slowest successful cell (`tenferro-eager`, 0.258 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 18.3x faster than the slowest successful cell (`tenferro-eager`, 0.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 0.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 55.4x faster than the slowest successful cell (`tenferro-eager`, 0.716 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 16.3x faster than the slowest successful cell (`tenferro-eager`, 0.716 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 51.6x faster than the slowest successful cell (`tenferro-eager`, 0.746 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`tenferro-eager`, 0.746 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 36.0x faster than the slowest successful cell (`tenferro-eager`, 0.733 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 0.733 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 25.2x faster than the slowest successful cell (`pytorch-cpu`, 0.328 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 21.8x faster than the slowest successful cell (`pytorch-cpu`, 0.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 0.330 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 64.6x faster than the slowest successful cell (`tenferro-eager`, 0.838 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`tenferro-eager`, 0.838 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 54.7x faster than the slowest successful cell (`tenferro-eager`, 0.851 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`tenferro-eager`, 0.851 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 40.7x faster than the slowest successful cell (`tenferro-eager`, 0.863 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`tenferro-eager`, 0.863 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 24.4x faster than the slowest successful cell (`pytorch-cpu`, 0.310 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 20.5x faster than the slowest successful cell (`pytorch-cpu`, 0.310 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 17.2x faster than the slowest successful cell (`pytorch-cpu`, 0.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 31.6x faster than the slowest successful cell (`pytorch-cpu`, 0.398 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.398 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 31.2x faster than the slowest successful cell (`pytorch-cpu`, 0.403 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.403 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 29.3x faster than the slowest successful cell (`pytorch-cpu`, 0.400 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.400 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 0.238 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.239 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 0.258 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.208 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.215 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `7da876a214acb65eb9ea099adef67b41ef8efc98`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261127_175648/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261127_175648/run.yaml`
- Timestamp: `2026-11-27T17:56:48.092254+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261127_175648`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

### CPU Information

- Model: `INTEL(R) XEON(R) PLATINUM 8573C`
- Vendor: `GenuineIntel`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261127_175648/cpu_ops_t1_20261127_175648.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.002 | 0.010 ± 0.001 | 0.015 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.031 ± 0.000 | 0.030 ± 0.000 | 0.037 ± 0.000 | 0.040 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.035 ± 0.000 | 0.034 ± 0.000 | 0.036 ± 0.000 | 0.040 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.129 ± 0.001 | 0.127 ± 0.000 | 0.121 ± 0.001 | 0.131 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.130 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.015 ± 0.000 | 0.143 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.261 ± 0.002 | 0.265 ± 0.025 | 0.153 ± 0.006 | 0.196 ± 0.042 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.000 | 0.142 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.022 ± 0.000 | 0.148 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.027 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.000 | 0.023 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.049 ± 0.001 | 0.048 ± 0.000 | 0.050 ± 0.001 | 0.056 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.000 | 0.014 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.020 ± 0.000 | 0.026 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.020 ± 0.000 | 0.019 ± 0.000 | 0.022 ± 0.000 | 0.039 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.019 ± 0.000 | 0.019 ± 0.000 | 0.026 ± 0.001 | 0.030 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.062 ± 0.001 | 0.061 ± 0.003 | 0.076 ± 0.000 | 0.084 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.058 ± 0.001 | 0.056 ± 0.000 | 0.069 ± 0.000 | 0.068 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.217 ± 0.001 | 0.216 ± 0.001 | 0.249 ± 0.001 | 0.227 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.067 ± 0.016 | 0.024 ± 0.000 | 0.088 ± 0.002 | 0.012 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.097 ± 0.003 | 0.060 ± 0.003 | 0.093 ± 0.002 | 0.027 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.061 ± 0.002 | 0.024 ± 0.000 | 0.092 ± 0.001 | 0.023 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.107 ± 0.009 | 0.062 ± 0.000 | 0.119 ± 0.007 | 0.044 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.071 ± 0.017 | 0.042 ± 0.000 | 0.067 ± 0.000 | 0.023 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.090 ± 0.004 | 0.075 ± 0.001 | 0.079 ± 0.002 | 0.052 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.076 ± 0.014 | 0.047 ± 0.000 | 0.070 ± 0.001 | 0.033 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.099 ± 0.007 | 0.092 ± 0.001 | 0.085 ± 0.001 | 0.067 ± 0.000 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.305 ± 0.001 | 0.302 ± 0.002 | 0.221 ± 0.002 | 0.318 ± 0.004 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 13.328 ± 0.235 | 10.307 ± 0.090 | 5.786 ± 0.070 | 7.268 ± 0.053 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 68.632 ± 0.344 | 52.063 ± 0.166 | 37.022 ± 3.295 | 44.792 ± 0.274 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 7.032 ± 0.046 | 10.294 ± 0.015 | 4.930 ± 0.076 | 7.230 ± 0.058 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 42.625 ± 0.282 | 51.695 ± 0.342 | 27.118 ± 0.350 | 44.412 ± 0.462 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 7.007 ± 0.061 | 3.501 ± 0.021 | 4.972 ± 0.045 | 4.995 ± 0.050 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 47.692 ± 0.528 | 22.528 ± 2.583 | 42.331 ± 0.270 | 33.758 ± 0.227 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 6.136 ± 0.013 | 3.586 ± 0.025 | 2.951 ± 0.013 | 4.976 ± 0.044 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 45.953 ± 0.231 | 26.791 ± 1.199 | 22.424 ± 0.393 | 34.431 ± 0.171 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.022 ± 0.000 | 0.013 ± 0.001 | 0.020 ± 0.001 | 0.047 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.089 ± 0.001 | 0.037 ± 0.001 | 0.079 ± 0.001 | 0.080 ± 0.001 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 10.130 ± 0.014 | 5.683 ± 0.049 | 4.069 ± 0.044 | 6.471 ± 0.037 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 74.439 ± 0.516 | 41.072 ± 0.613 | 28.382 ± 2.724 | 39.582 ± 0.607 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 7.387 ± 0.055 | 5.778 ± 0.215 | 3.951 ± 0.022 | 6.327 ± 0.064 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 64.183 ± 0.354 | 40.947 ± 0.862 | 26.099 ± 0.415 | 44.668 ± 0.421 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.094 ± 0.011 | 0.093 ± 0.001 | 0.103 ± 0.001 | 0.058 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.128 ± 0.033 | 0.820 ± 0.003 | 0.876 ± 0.113 | 0.597 ± 0.007 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 5.106 ± 0.052 | 4.578 ± 0.032 | 3.143 ± 0.054 | 4.570 ± 0.516 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.612 ± 0.011 | 0.821 ± 0.011 | 1.065 ± 0.035 | 0.617 ± 0.024 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 2.777 ± 0.047 | 4.576 ± 0.033 | 5.009 ± 0.044 | 4.164 ± 0.304 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.711 ± 0.009 | 0.897 ± 0.009 | 0.557 ± 0.004 | 0.528 ± 0.018 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 25.588 ± 0.260 | 19.365 ± 0.177 | 10.695 ± 0.233 | 12.178 ± 0.213 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 153.485 ± 0.403 | 114.920 ± 0.507 | 73.862 ± 1.133 | 77.801 ± 0.723 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 12.993 ± 0.222 | 19.261 ± 0.471 | 8.636 ± 0.088 | 12.053 ± 0.104 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 78.373 ± 0.204 | 112.214 ± 3.676 | 52.602 ± 0.281 | 85.904 ± 1.419 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.063 ± 0.004 | 0.055 ± 0.003 | 0.108 ± 0.001 | 0.193 ± 0.002 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.417 ± 0.011 | 0.402 ± 0.004 | 0.623 ± 0.007 | 1.430 ± 0.013 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 1.579 ± 0.011 | 1.586 ± 0.055 | 1.946 ± 0.017 | 4.952 ± 0.071 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.104 ± 0.001 | 0.104 ± 0.002 | 0.066 ± 0.000 | 0.094 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.023 ± 0.001 | 0.023 ± 0.000 | 0.040 ± 0.000 | 0.042 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.029 ± 0.002 | 0.026 ± 0.001 | 0.047 ± 0.003 | 0.058 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.053 ± 0.004 | 0.039 ± 0.001 | 0.063 ± 0.002 | 0.132 ± 0.005 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.591 ± 0.001 | 0.589 ± 0.006 | 0.456 ± 0.005 | 0.489 ± 0.005 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.001 | 0.017 ± 0.005 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.124 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.127 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.140 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.181 ± 0.008 | 0.020 ± 0.000 | 0.109 ± 0.007 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.189 ± 0.006 | 0.023 ± 0.000 | 0.113 ± 0.009 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.204 ± 0.002 | 0.031 ± 0.000 | 0.117 ± 0.002 | 0.022 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.044 ± 0.001 | 0.020 ± 0.001 | 0.100 ± 0.003 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.045 ± 0.001 | 0.022 ± 0.000 | 0.103 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.052 ± 0.001 | 0.030 ± 0.000 | 0.107 ± 0.003 | 0.022 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.716 ± 0.011 | 0.027 ± 0.002 | 0.170 ± 0.004 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.729 ± 0.056 | 0.028 ± 0.000 | 0.166 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.731 ± 0.007 | 0.029 ± 0.000 | 0.170 ± 0.004 | 0.026 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.119 ± 0.003 | 0.025 ± 0.000 | 0.140 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.119 ± 0.002 | 0.025 ± 0.000 | 0.142 ± 0.007 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.121 ± 0.002 | 0.027 ± 0.000 | 0.144 ± 0.001 | 0.025 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.053 ± 0.001 | 0.012 ± 0.000 | 0.035 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.050 ± 0.008 | 0.014 ± 0.000 | 0.037 ± 0.001 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.054 ± 0.002 | 0.013 ± 0.000 | 0.039 ± 0.003 | 0.025 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.832 ± 0.008 | 0.034 ± 0.000 | 0.126 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.874 ± 0.011 | 0.036 ± 0.000 | 0.123 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.881 ± 0.003 | 0.037 ± 0.000 | 0.128 ± 0.003 | 0.024 ± 0.002 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.107 ± 0.001 | 0.033 ± 0.000 | 0.133 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.108 ± 0.001 | 0.035 ± 0.000 | 0.135 ± 0.006 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.109 ± 0.003 | 0.038 ± 0.000 | 0.136 ± 0.002 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.062 ± 0.003 | 0.029 ± 0.000 | 0.056 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.059 ± 0.002 | 0.031 ± 0.000 | 0.058 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.063 ± 0.003 | 0.032 ± 0.000 | 0.059 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.119 ± 0.000 | 0.029 ± 0.001 | 0.268 ± 0.005 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.119 ± 0.001 | 0.021 ± 0.000 | 0.272 ± 0.005 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.124 ± 0.002 | 0.022 ± 0.000 | 0.270 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.049 ± 0.000 | 0.021 ± 0.000 | 0.128 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.049 ± 0.002 | 0.021 ± 0.000 | 0.127 ± 0.017 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.050 ± 0.001 | 0.022 ± 0.000 | 0.128 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.052 ± 0.012 | 0.015 ± 0.000 | 0.043 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.052 ± 0.004 | 0.020 ± 0.000 | 0.050 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.064 ± 0.006 | 0.031 ± 0.000 | 0.058 ± 0.001 | 0.028 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.154 ± 0.002 | 0.016 ± 0.000 | 0.167 ± 0.021 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.162 ± 0.003 | 0.020 ± 0.000 | 0.162 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.187 ± 0.005 | 0.032 ± 0.000 | 0.174 ± 0.004 | 0.027 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.043 ± 0.001 | 0.015 ± 0.000 | 0.120 ± 0.010 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.046 ± 0.002 | 0.020 ± 0.000 | 0.114 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.058 ± 0.001 | 0.032 ± 0.000 | 0.122 ± 0.006 | 0.028 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.013 ± 0.002 | 0.012 ± 0.000 | 0.024 ± 0.000 | 0.043 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.000 | 0.023 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 16.2x faster than the slowest successful cell (`jax-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`jax-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 16.3x faster than the slowest successful cell (`jax-cpu`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`jax-cpu`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 26.0x faster than the slowest successful cell (`jax-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`jax-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 28.5x faster than the slowest successful cell (`jax-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 29.5x faster than the slowest successful cell (`jax-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 16.4x faster than the slowest successful cell (`jax-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 18.9x faster than the slowest successful cell (`tenferro-eager`, 0.181 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 17.1x faster than the slowest successful cell (`tenferro-eager`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 64.6x faster than the slowest successful cell (`tenferro-eager`, 0.716 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 26.2x faster than the slowest successful cell (`tenferro-eager`, 0.716 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 63.1x faster than the slowest successful cell (`tenferro-eager`, 0.729 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 26.3x faster than the slowest successful cell (`tenferro-eager`, 0.729 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 27.6x faster than the slowest successful cell (`tenferro-eager`, 0.731 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 25.5x faster than the slowest successful cell (`tenferro-eager`, 0.731 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 77.0x faster than the slowest successful cell (`tenferro-eager`, 0.832 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 24.6x faster than the slowest successful cell (`tenferro-eager`, 0.832 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 76.5x faster than the slowest successful cell (`tenferro-eager`, 0.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 24.5x faster than the slowest successful cell (`tenferro-eager`, 0.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 37.3x faster than the slowest successful cell (`tenferro-eager`, 0.881 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 23.9x faster than the slowest successful cell (`tenferro-eager`, 0.881 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.133 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 24.9x faster than the slowest successful cell (`pytorch-cpu`, 0.268 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 24.4x faster than the slowest successful cell (`pytorch-cpu`, 0.272 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.272 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 23.1x faster than the slowest successful cell (`pytorch-cpu`, 0.270 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.270 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.4x faster than the slowest successful cell (`tenferro-eager`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261127_183055/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261127_183055/run.yaml`
- Timestamp: `2026-11-27T18:30:55.648913+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261127_183055`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

### CPU Information

- Model: `INTEL(R) XEON(R) PLATINUM 8573C`
- Vendor: `GenuineIntel`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261127_183055/cpu_ops_t4_20261127_183055.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.031 ± 0.001 | 0.030 ± 0.000 | 0.036 ± 0.000 | 0.063 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.035 ± 0.000 | 0.034 ± 0.000 | 0.036 ± 0.000 | 0.040 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.129 ± 0.000 | 0.128 ± 0.001 | 0.120 ± 0.014 | 0.158 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.129 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.021 ± 0.000 | 0.021 ± 0.001 | 0.015 ± 0.000 | 0.141 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.100 ± 0.000 | 0.098 ± 0.003 | 0.104 ± 0.013 | 0.247 ± 0.096 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.000 | 0.144 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.021 ± 0.000 | 0.021 ± 0.000 | 0.022 ± 0.000 | 0.147 ± 0.021 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.001 | 0.021 ± 0.000 | 0.027 ± 0.000 | 0.028 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.024 ± 0.001 | 0.023 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.050 ± 0.000 | 0.049 ± 0.000 | 0.065 ± 0.001 | 0.086 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.025 ± 0.001 | 0.014 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.030 ± 0.000 | 0.030 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.026 ± 0.001 | 0.017 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.034 ± 0.001 | 0.044 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.020 ± 0.000 | 0.019 ± 0.000 | 0.025 ± 0.001 | 0.030 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.063 ± 0.000 | 0.061 ± 0.000 | 0.073 ± 0.001 | 0.083 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.058 ± 0.000 | 0.057 ± 0.001 | 0.068 ± 0.001 | 0.069 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.218 ± 0.002 | 0.216 ± 0.001 | 0.245 ± 0.001 | 0.265 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.063 ± 0.003 | 0.026 ± 0.001 | 0.088 ± 0.001 | 0.012 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.101 ± 0.013 | 0.062 ± 0.001 | 0.092 ± 0.002 | 0.017 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.067 ± 0.001 | 0.027 ± 0.000 | 0.096 ± 0.004 | 0.025 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.106 ± 0.012 | 0.063 ± 0.001 | 0.120 ± 0.001 | 0.076 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.075 ± 0.005 | 0.049 ± 0.001 | 0.093 ± 0.001 | 0.024 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.097 ± 0.016 | 0.083 ± 0.000 | 0.105 ± 0.001 | 0.066 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.080 ± 0.004 | 0.054 ± 0.001 | 0.098 ± 0.002 | 0.032 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.103 ± 0.004 | 0.099 ± 0.000 | 0.116 ± 0.003 | 0.064 ± 0.001 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.306 ± 0.004 | 0.303 ± 0.002 | 0.223 ± 0.004 | 0.328 ± 0.001 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 15.620 ± 0.055 | 11.638 ± 0.057 | 4.706 ± 0.188 | 9.961 ± 0.511 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 46.035 ± 1.770 | 40.114 ± 0.243 | 23.831 ± 0.472 | 43.135 ± 4.679 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 8.081 ± 0.074 | 11.556 ± 0.065 | 4.054 ± 0.101 | 9.759 ± 1.174 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 26.386 ± 0.162 | 40.007 ± 0.195 | 17.050 ± 0.593 | 48.024 ± 2.490 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 6.461 ± 0.084 | 3.239 ± 0.018 | 4.357 ± 1.008 | 5.978 ± 2.096 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 30.778 ± 0.129 | 16.479 ± 0.086 | 24.498 ± 0.358 | 34.902 ± 6.269 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 4.922 ± 0.074 | 3.265 ± 0.010 | 2.660 ± 0.130 | 4.893 ± 3.025 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 27.637 ± 0.262 | 17.796 ± 0.172 | 13.421 ± 0.137 | 35.640 ± 3.600 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.033 ± 0.001 | 0.024 ± 0.001 | 0.023 ± 0.001 | 0.044 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.130 ± 0.006 | 0.065 ± 0.001 | 0.093 ± 0.002 | 0.090 ± 0.005 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 8.542 ± 0.798 | 4.878 ± 0.021 | 2.847 ± 0.058 | 8.682 ± 1.441 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 45.596 ± 0.563 | 26.930 ± 0.078 | 15.395 ± 0.264 | 43.126 ± 8.850 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 5.914 ± 0.034 | 4.859 ± 0.073 | 2.841 ± 0.099 | 9.628 ± 1.079 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 36.654 ± 0.258 | 25.776 ± 0.388 | 15.018 ± 0.141 | 50.992 ± 9.993 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.133 ± 0.013 | 0.161 ± 0.001 | 0.104 ± 0.003 | 0.063 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 1.388 ± 0.019 | 1.041 ± 0.007 | 0.882 ± 0.024 | 0.935 ± 0.129 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 4.666 ± 0.067 | 4.102 ± 0.120 | 2.159 ± 0.025 | 3.582 ± 0.078 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.760 ± 0.002 | 1.046 ± 0.007 | 1.033 ± 0.037 | 0.992 ± 0.026 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 2.539 ± 0.097 | 4.182 ± 0.119 | 3.721 ± 0.801 | 3.421 ± 0.375 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.840 ± 0.033 | 1.015 ± 0.012 | 0.737 ± 0.429 | 0.859 ± 0.017 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 27.068 ± 0.135 | 19.763 ± 0.152 | 9.899 ± 0.163 | 15.090 ± 1.088 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 111.018 ± 2.129 | 88.594 ± 0.532 | 53.509 ± 1.113 | 90.762 ± 5.051 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 13.755 ± 0.039 | 19.824 ± 0.063 | 8.751 ± 0.170 | 15.359 ± 0.717 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 57.120 ± 0.943 | 87.111 ± 0.209 | 40.222 ± 0.668 | 86.664 ± 1.455 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.058 ± 0.001 | 0.055 ± 0.009 | 0.076 ± 0.005 | 0.170 ± 0.019 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.278 ± 0.006 | 0.266 ± 0.005 | 0.329 ± 0.002 | 0.729 ± 0.077 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.968 ± 0.016 | 0.974 ± 0.013 | 1.108 ± 0.045 | 3.086 ± 0.339 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.110 ± 0.001 | 0.105 ± 0.001 | 0.092 ± 0.002 | 0.104 ± 0.019 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.040 ± 0.001 | 0.042 ± 0.000 | 0.040 ± 0.000 | 0.054 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.049 ± 0.003 | 0.048 ± 0.002 | 0.067 ± 0.004 | 0.144 ± 0.626 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.092 ± 0.003 | 0.076 ± 0.006 | 0.085 ± 0.003 | 0.212 ± 0.035 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.590 ± 0.001 | 0.589 ± 0.003 | 0.520 ± 0.002 | 0.795 ± 0.045 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.123 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.129 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.143 ± 0.007 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.191 ± 0.007 | 0.025 ± 0.000 | 0.110 ± 0.003 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.197 ± 0.005 | 0.028 ± 0.000 | 0.111 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.216 ± 0.010 | 0.036 ± 0.000 | 0.118 ± 0.006 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.053 ± 0.017 | 0.024 ± 0.000 | 0.100 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.054 ± 0.001 | 0.027 ± 0.000 | 0.104 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.059 ± 0.010 | 0.035 ± 0.001 | 0.108 ± 0.013 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.736 ± 0.041 | 0.033 ± 0.000 | 0.235 ± 0.013 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.757 ± 0.009 | 0.035 ± 0.003 | 0.232 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.766 ± 0.014 | 0.036 ± 0.000 | 0.233 ± 0.005 | 0.021 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.139 ± 0.006 | 0.029 ± 0.000 | 0.180 ± 0.004 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.141 ± 0.001 | 0.031 ± 0.000 | 0.181 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.192 ± 0.009 | 0.032 ± 0.000 | 0.185 ± 0.011 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.058 ± 0.004 | 0.015 ± 0.001 | 0.035 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.058 ± 0.003 | 0.015 ± 0.000 | 0.037 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.059 ± 0.000 | 0.014 ± 0.000 | 0.037 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.884 ± 0.022 | 0.044 ± 0.000 | 0.166 ± 0.006 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.895 ± 0.014 | 0.044 ± 0.000 | 0.162 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.924 ± 0.014 | 0.046 ± 0.001 | 0.167 ± 0.004 | 0.022 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.124 ± 0.000 | 0.041 ± 0.001 | 0.166 ± 0.006 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.132 ± 0.004 | 0.043 ± 0.001 | 0.169 ± 0.004 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.125 ± 0.000 | 0.046 ± 0.002 | 0.174 ± 0.015 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.067 ± 0.011 | 0.037 ± 0.000 | 0.055 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.066 ± 0.001 | 0.038 ± 0.000 | 0.058 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.070 ± 0.002 | 0.039 ± 0.000 | 0.058 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.125 ± 0.007 | 0.025 ± 0.000 | 0.276 ± 0.008 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.126 ± 0.004 | 0.026 ± 0.000 | 0.277 ± 0.005 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.130 ± 0.000 | 0.026 ± 0.000 | 0.300 ± 0.058 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.059 ± 0.004 | 0.025 ± 0.000 | 0.127 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.055 ± 0.000 | 0.025 ± 0.000 | 0.129 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.057 ± 0.001 | 0.026 ± 0.000 | 0.134 ± 0.043 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.054 ± 0.002 | 0.018 ± 0.000 | 0.043 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.058 ± 0.004 | 0.023 ± 0.001 | 0.049 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.074 ± 0.021 | 0.035 ± 0.000 | 0.057 ± 0.001 | 0.027 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.158 ± 0.002 | 0.019 ± 0.000 | 0.155 ± 0.005 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.166 ± 0.004 | 0.024 ± 0.000 | 0.162 ± 0.006 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.189 ± 0.008 | 0.036 ± 0.000 | 0.174 ± 0.003 | 0.025 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.048 ± 0.002 | 0.018 ± 0.000 | 0.110 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.052 ± 0.001 | 0.023 ± 0.000 | 0.112 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.064 ± 0.001 | 0.035 ± 0.000 | 0.122 ± 0.001 | 0.025 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.002 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.011 ± 0.000 | 0.010 ± 0.001 | 0.005 ± 0.000 | 0.015 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.022 ± 0.001 | 0.021 ± 0.000 | 0.027 ± 0.002 | 0.037 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.002 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.017 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.000 | 0.023 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 16.1x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 16.2x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 25.2x faster than the slowest successful cell (`jax-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 26.8x faster than the slowest successful cell (`jax-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`jax-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 26.7x faster than the slowest successful cell (`jax-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`jax-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 20.1x faster than the slowest successful cell (`tenferro-eager`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 16.9x faster than the slowest successful cell (`tenferro-eager`, 0.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 11.0x faster than the slowest successful cell (`tenferro-eager`, 0.216 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 65.9x faster than the slowest successful cell (`tenferro-eager`, 0.736 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 22.5x faster than the slowest successful cell (`tenferro-eager`, 0.736 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 62.7x faster than the slowest successful cell (`tenferro-eager`, 0.757 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 21.8x faster than the slowest successful cell (`tenferro-eager`, 0.757 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 36.4x faster than the slowest successful cell (`tenferro-eager`, 0.766 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`tenferro-eager`, 0.766 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 16.6x faster than the slowest successful cell (`pytorch-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.181 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 82.5x faster than the slowest successful cell (`tenferro-eager`, 0.884 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 20.3x faster than the slowest successful cell (`tenferro-eager`, 0.884 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 72.3x faster than the slowest successful cell (`tenferro-eager`, 0.895 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 20.2x faster than the slowest successful cell (`tenferro-eager`, 0.895 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 41.8x faster than the slowest successful cell (`tenferro-eager`, 0.924 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 20.3x faster than the slowest successful cell (`tenferro-eager`, 0.924 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 25.1x faster than the slowest successful cell (`pytorch-cpu`, 0.276 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.276 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 24.7x faster than the slowest successful cell (`pytorch-cpu`, 0.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 25.3x faster than the slowest successful cell (`pytorch-cpu`, 0.300 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.300 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.8x faster than the slowest successful cell (`tenferro-eager`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.0x faster than the slowest successful cell (`tenferro-eager`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

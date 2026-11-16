# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `5c1d5a382dd88d20901598318730d850d3568ac8`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261116_180206/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261116_180206/run.yaml`
- Timestamp: `2026-11-16T18:02:06.211309+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261116_180206`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Intel(R) Xeon(R) 6975P-C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1016-aws-x86_64-with-glibc2.39`
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

- tenferro-rs features: `blas-mkl`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.2`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/amd-cpu/cpu/einsum/20261116_180206/cpu_ops_t1_20261116_180206.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.029 ± 0.000 | 0.027 ± 0.000 | 0.027 ± 0.001 | 0.024 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.028 ± 0.000 | 0.027 ± 0.000 | 0.027 ± 0.000 | 0.028 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.100 ± 0.001 | 0.097 ± 0.001 | 0.092 ± 0.001 | 0.091 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.099 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.106 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.088 ± 0.001 | 0.085 ± 0.001 | 0.116 ± 0.003 | 0.120 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.000 | 0.107 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.106 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.015 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.020 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.027 ± 0.000 | 0.025 ± 0.000 | 0.039 ± 0.001 | 0.042 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.012 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.000 | 0.026 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.018 ± 0.000 | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.022 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.059 ± 0.000 | 0.057 ± 0.000 | 0.055 ± 0.001 | 0.062 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.058 ± 0.001 | 0.056 ± 0.000 | 0.052 ± 0.000 | 0.051 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.215 ± 0.002 | 0.211 ± 0.002 | 0.189 ± 0.002 | 0.175 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.039 ± 0.001 | 0.011 ± 0.000 | 0.063 ± 0.001 | 0.011 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.049 ± 0.011 | 0.021 ± 0.000 | 0.068 ± 0.001 | 0.020 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.041 ± 0.001 | 0.013 ± 0.000 | 0.068 ± 0.001 | 0.018 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.057 ± 0.008 | 0.026 ± 0.000 | 0.090 ± 0.001 | 0.028 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.050 ± 0.005 | 0.028 ± 0.000 | 0.050 ± 0.001 | 0.019 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.058 ± 0.004 | 0.046 ± 0.000 | 0.058 ± 0.001 | 0.034 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.052 ± 0.001 | 0.031 ± 0.000 | 0.054 ± 0.000 | 0.021 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.064 ± 0.002 | 0.055 ± 0.000 | 0.065 ± 0.000 | 0.046 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.184 ± 0.002 | 0.180 ± 0.001 | 0.190 ± 0.002 | 0.259 ± 0.003 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 8.415 ± 0.073 | 5.859 ± 0.048 | 4.941 ± 0.036 | 6.141 ± 0.069 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 48.488 ± 0.345 | 33.362 ± 0.580 | 29.130 ± 0.112 | 38.017 ± 0.216 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 4.607 ± 0.120 | 5.944 ± 0.086 | 4.146 ± 0.051 | 6.099 ± 0.043 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 27.248 ± 0.256 | 34.819 ± 0.087 | 22.843 ± 0.369 | 37.459 ± 0.334 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 5.257 ± 0.075 | 2.299 ± 0.010 | 4.204 ± 0.124 | 4.175 ± 0.057 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 34.042 ± 0.379 | 15.540 ± 0.118 | 36.366 ± 0.582 | 28.535 ± 0.237 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 4.712 ± 0.028 | 2.352 ± 0.030 | 2.533 ± 0.056 | 4.180 ± 0.047 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 31.973 ± 0.653 | 15.895 ± 0.051 | 19.386 ± 0.236 | 29.854 ± 0.532 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.018 ± 0.001 | 0.010 ± 0.000 | 0.020 ± 0.001 | 0.038 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.068 ± 0.005 | 0.032 ± 0.002 | 0.064 ± 0.003 | 0.077 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 8.048 ± 0.102 | 4.205 ± 0.054 | 3.575 ± 0.095 | 5.583 ± 0.074 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 50.190 ± 0.430 | 29.398 ± 0.186 | 24.916 ± 0.250 | 34.468 ± 0.848 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 6.198 ± 0.052 | 4.387 ± 0.051 | 3.502 ± 0.023 | 5.545 ± 0.115 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 40.961 ± 0.339 | 29.815 ± 0.115 | 23.448 ± 0.219 | 39.549 ± 0.705 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.072 ± 0.005 | 0.082 ± 0.002 | 0.084 ± 0.002 | 0.047 ± 0.000 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.100 ± 0.018 | 0.640 ± 0.006 | 0.750 ± 0.025 | 0.545 ± 0.021 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 4.192 ± 0.094 | 3.813 ± 0.070 | 2.987 ± 0.003 | 3.272 ± 0.089 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.493 ± 0.009 | 0.612 ± 0.008 | 0.918 ± 0.037 | 0.550 ± 0.011 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 2.356 ± 0.027 | 3.759 ± 0.009 | 4.708 ± 0.027 | 3.052 ± 0.122 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.481 ± 0.019 | 0.518 ± 0.005 | 0.449 ± 0.011 | 0.460 ± 0.008 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 15.007 ± 0.297 | 10.352 ± 0.089 | 9.228 ± 0.449 | 10.197 ± 0.153 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 97.115 ± 1.426 | 68.486 ± 0.495 | 61.576 ± 0.942 | 69.446 ± 0.377 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 7.434 ± 0.045 | 10.340 ± 0.089 | 7.576 ± 0.342 | 10.109 ± 0.157 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 50.999 ± 1.153 | 70.111 ± 1.007 | 43.585 ± 0.454 | 72.429 ± 1.599 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.052 ± 0.002 | 0.043 ± 0.006 | 0.098 ± 0.004 | 0.166 ± 0.002 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.369 ± 0.015 | 0.366 ± 0.006 | 0.547 ± 0.011 | 1.174 ± 0.019 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 1.468 ± 0.065 | 1.476 ± 0.039 | 1.719 ± 0.079 | 4.162 ± 0.061 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.044 ± 0.001 | 0.042 ± 0.001 | 0.057 ± 0.001 | 0.078 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.023 ± 0.000 | 0.022 ± 0.001 | 0.035 ± 0.001 | 0.035 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.023 ± 0.001 | 0.025 ± 0.002 | 0.042 ± 0.006 | 0.059 ± 0.042 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.030 ± 0.000 | 0.033 ± 0.000 | 0.063 ± 0.015 | 0.165 ± 0.161 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.358 ± 0.006 | 0.348 ± 0.005 | 0.369 ± 0.006 | 0.406 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.098 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.098 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.104 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.142 ± 0.001 | 0.016 ± 0.000 | 0.084 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.152 ± 0.002 | 0.019 ± 0.000 | 0.089 ± 0.003 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.161 ± 0.002 | 0.024 ± 0.000 | 0.094 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.034 ± 0.001 | 0.015 ± 0.000 | 0.079 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.038 ± 0.001 | 0.018 ± 0.000 | 0.081 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.042 ± 0.001 | 0.023 ± 0.000 | 0.086 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.547 ± 0.008 | 0.020 ± 0.000 | 0.129 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.569 ± 0.007 | 0.021 ± 0.000 | 0.131 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.584 ± 0.010 | 0.021 ± 0.000 | 0.133 ± 0.002 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.090 ± 0.001 | 0.018 ± 0.000 | 0.111 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.093 ± 0.005 | 0.019 ± 0.000 | 0.113 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.096 ± 0.002 | 0.019 ± 0.000 | 0.115 ± 0.001 | 0.018 ± 0.004 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.034 ± 0.001 | 0.008 ± 0.000 | 0.027 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.036 ± 0.001 | 0.008 ± 0.000 | 0.031 ± 0.002 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.034 ± 0.001 | 0.008 ± 0.000 | 0.029 ± 0.001 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.781 ± 0.006 | 0.027 ± 0.000 | 0.097 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.814 ± 0.004 | 0.029 ± 0.001 | 0.099 ± 0.006 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.822 ± 0.015 | 0.030 ± 0.000 | 0.101 ± 0.002 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.081 ± 0.001 | 0.027 ± 0.000 | 0.105 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.084 ± 0.000 | 0.028 ± 0.000 | 0.108 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.086 ± 0.001 | 0.030 ± 0.001 | 0.109 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.045 ± 0.001 | 0.022 ± 0.000 | 0.042 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.048 ± 0.003 | 0.022 ± 0.001 | 0.044 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.046 ± 0.001 | 0.023 ± 0.000 | 0.045 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.093 ± 0.001 | 0.015 ± 0.000 | 0.206 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.095 ± 0.003 | 0.016 ± 0.000 | 0.211 ± 0.003 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.098 ± 0.003 | 0.017 ± 0.000 | 0.213 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.037 ± 0.001 | 0.015 ± 0.000 | 0.099 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.037 ± 0.001 | 0.015 ± 0.001 | 0.102 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.039 ± 0.002 | 0.016 ± 0.001 | 0.103 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.037 ± 0.002 | 0.011 ± 0.000 | 0.033 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.047 ± 0.016 | 0.016 ± 0.001 | 0.038 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.051 ± 0.002 | 0.025 ± 0.000 | 0.046 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.124 ± 0.001 | 0.013 ± 0.000 | 0.120 ± 0.003 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.133 ± 0.002 | 0.017 ± 0.000 | 0.126 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.152 ± 0.008 | 0.027 ± 0.000 | 0.137 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.034 ± 0.002 | 0.011 ± 0.000 | 0.085 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.038 ± 0.001 | 0.015 ± 0.000 | 0.090 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.047 ± 0.001 | 0.026 ± 0.000 | 0.100 ± 0.012 | 0.015 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.009 ± 0.001 | 0.008 ± 0.000 | 0.022 ± 0.001 | 0.035 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.002 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 32.1x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 29.1x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 17.7x faster than the slowest successful cell (`jax-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`jax-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 25.2x faster than the slowest successful cell (`jax-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 27.2x faster than the slowest successful cell (`jax-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 40.2x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 19.3x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 40.8x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 19.3x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 41.2x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 18.9x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 18.8x faster than the slowest successful cell (`tenferro-eager`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 17.3x faster than the slowest successful cell (`tenferro-eager`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-eager`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 62.7x faster than the slowest successful cell (`tenferro-eager`, 0.547 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 27.2x faster than the slowest successful cell (`tenferro-eager`, 0.547 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 64.0x faster than the slowest successful cell (`tenferro-eager`, 0.569 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 27.2x faster than the slowest successful cell (`tenferro-eager`, 0.569 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 32.0x faster than the slowest successful cell (`tenferro-eager`, 0.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 27.7x faster than the slowest successful cell (`tenferro-eager`, 0.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.111 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 88.6x faster than the slowest successful cell (`tenferro-eager`, 0.781 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 28.7x faster than the slowest successful cell (`tenferro-eager`, 0.781 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 86.5x faster than the slowest successful cell (`tenferro-eager`, 0.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 28.4x faster than the slowest successful cell (`tenferro-eager`, 0.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 47.8x faster than the slowest successful cell (`tenferro-eager`, 0.822 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 27.1x faster than the slowest successful cell (`tenferro-eager`, 0.822 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 23.6x faster than the slowest successful cell (`pytorch-cpu`, 0.206 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.206 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 0.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 23.0x faster than the slowest successful cell (`pytorch-cpu`, 0.213 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.213 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.0x faster than the slowest successful cell (`tenferro-eager`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.5x faster than the slowest successful cell (`tenferro-eager`, 0.133 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261116_183058/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261116_183058/run.yaml`
- Timestamp: `2026-11-16T18:30:58.322744+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261116_183058`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Intel(R) Xeon(R) 6975P-C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1016-aws-x86_64-with-glibc2.39`
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

- tenferro-rs features: `blas-mkl`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.2`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/amd-cpu/cpu/einsum/20261116_183058/cpu_ops_t4_20261116_183058.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.028 ± 0.000 | 0.026 ± 0.000 | 0.026 ± 0.001 | 0.030 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.027 ± 0.000 | 0.026 ± 0.000 | 0.027 ± 0.001 | 0.030 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.098 ± 0.001 | 0.096 ± 0.000 | 0.091 ± 0.001 | 0.094 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.099 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.105 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.088 ± 0.001 | 0.085 ± 0.001 | 0.071 ± 0.008 | 0.174 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.000 | 0.112 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.116 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.011 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.022 ± 0.000 | 0.020 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.019 ± 0.000 | 0.018 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.027 ± 0.000 | 0.026 ± 0.000 | 0.053 ± 0.002 | 0.046 ± 0.010 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.018 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.001 | 0.022 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.020 ± 0.000 | 0.014 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.026 ± 0.001 | 0.028 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.019 ± 0.001 | 0.023 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.055 ± 0.000 | 0.053 ± 0.000 | 0.052 ± 0.001 | 0.063 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.056 ± 0.000 | 0.055 ± 0.000 | 0.052 ± 0.001 | 0.052 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.209 ± 0.002 | 0.210 ± 0.001 | 0.185 ± 0.005 | 0.191 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.037 ± 0.001 | 0.012 ± 0.000 | 0.065 ± 0.001 | 0.010 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.050 ± 0.016 | 0.022 ± 0.000 | 0.068 ± 0.000 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.040 ± 0.001 | 0.014 ± 0.000 | 0.069 ± 0.000 | 0.018 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.056 ± 0.002 | 0.027 ± 0.000 | 0.088 ± 0.003 | 0.049 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.050 ± 0.016 | 0.033 ± 0.000 | 0.067 ± 0.001 | 0.019 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.059 ± 0.002 | 0.051 ± 0.000 | 0.077 ± 0.002 | 0.046 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.050 ± 0.002 | 0.035 ± 0.000 | 0.069 ± 0.001 | 0.023 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.063 ± 0.002 | 0.059 ± 0.000 | 0.084 ± 0.004 | 0.058 ± 0.001 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.250 ± 0.001 | 0.237 ± 0.002 | 0.193 ± 0.003 | 0.283 ± 0.004 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 14.571 ± 0.115 | 10.704 ± 0.187 | 4.141 ± 0.083 | 6.683 ± 1.621 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 64.653 ± 0.526 | 48.554 ± 0.534 | 21.066 ± 0.335 | 33.811 ± 4.892 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 7.660 ± 0.145 | 10.871 ± 0.071 | 3.618 ± 0.059 | 7.973 ± 1.335 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 35.395 ± 0.270 | 48.625 ± 0.309 | 15.519 ± 0.534 | 38.614 ± 0.372 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 437.083 ± 60.434 | 401.935 ± 20.005 | 2.905 ± 0.043 | 4.975 ± 2.003 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 1084.687 ± 19.481 | 1054.399 ± 36.409 | 20.978 ± 0.649 | 27.637 ± 3.430 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 225.680 ± 27.063 | 431.148 ± 12.151 | 2.117 ± 0.039 | 4.991 ± 1.480 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 547.382 ± 29.971 | 1011.971 ± 56.951 | 11.275 ± 0.279 | 30.954 ± 3.676 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.025 ± 0.001 | 0.016 ± 0.001 | 0.022 ± 0.001 | 0.036 ± 0.005 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.096 ± 0.003 | 0.048 ± 0.001 | 0.081 ± 0.001 | 0.076 ± 0.009 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 10.194 ± 0.084 | 5.677 ± 0.042 | 2.532 ± 0.119 | 9.968 ± 2.004 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 942.750 ± 26.785 | 893.078 ± 105.707 | 13.721 ± 0.201 | 41.109 ± 11.940 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 7.355 ± 0.037 | 5.741 ± 0.038 | 2.529 ± 0.062 | 7.959 ± 3.692 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 511.777 ± 26.474 | 911.702 ± 55.166 | 13.028 ± 0.313 | 44.362 ± 0.267 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.083 ± 0.028 | 0.085 ± 0.001 | 0.086 ± 0.002 | 0.057 ± 0.006 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 439.345 ± 21.225 | 431.899 ± 17.883 | 1.680 ± 0.754 | 0.705 ± 0.006 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 1015.794 ± 28.866 | 1017.884 ± 48.129 | 2.041 ± 0.086 | 3.038 ± 0.198 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 229.189 ± 6.128 | 417.012 ± 24.154 | 0.957 ± 0.052 | 0.714 ± 0.018 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 485.035 ± 39.221 | 1043.052 ± 21.047 | 2.929 ± 0.038 | 2.957 ± 0.102 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.573 ± 0.040 | 0.570 ± 0.004 | 0.527 ± 0.007 | 0.589 ± 0.075 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 17.767 ± 0.185 | 11.416 ± 0.038 | 8.211 ± 0.672 | 24.971 ± 4.844 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 114.241 ± 1.034 | 80.280 ± 0.397 | 47.185 ± 1.012 | 75.977 ± 1.997 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 9.042 ± 0.108 | 12.006 ± 0.136 | 7.199 ± 0.150 | 13.954 ± 3.673 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 59.295 ± 1.460 | 81.001 ± 0.407 | 34.960 ± 0.603 | 74.522 ± 7.938 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.066 ± 0.003 | 0.058 ± 0.002 | 0.068 ± 0.003 | 0.123 ± 0.022 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.434 ± 0.006 | 0.396 ± 0.002 | 0.282 ± 0.010 | 0.625 ± 0.048 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 1.789 ± 0.058 | 1.685 ± 0.039 | 0.973 ± 0.036 | 2.381 ± 0.147 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.069 ± 0.001 | 0.063 ± 0.001 | 0.082 ± 0.003 | 0.091 ± 0.004 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.023 ± 0.000 | 0.022 ± 0.000 | 0.034 ± 0.002 | 0.047 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.034 ± 0.001 | 0.035 ± 0.002 | 0.054 ± 0.013 | 0.099 ± 0.012 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.046 ± 0.001 | 0.045 ± 0.003 | 0.069 ± 0.008 | 0.147 ± 0.006 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.410 ± 0.005 | 0.395 ± 0.008 | 0.439 ± 0.024 | 0.550 ± 0.011 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.097 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.097 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.109 ± 0.007 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.147 ± 0.001 | 0.018 ± 0.000 | 0.085 ± 0.003 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.154 ± 0.004 | 0.021 ± 0.000 | 0.086 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.161 ± 0.006 | 0.026 ± 0.000 | 0.092 ± 0.002 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.040 ± 0.002 | 0.018 ± 0.000 | 0.078 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.043 ± 0.001 | 0.020 ± 0.000 | 0.079 ± 0.008 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.047 ± 0.001 | 0.025 ± 0.000 | 0.081 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.565 ± 0.006 | 0.023 ± 0.000 | 35.978 ± 35.823 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.571 ± 0.003 | 0.024 ± 0.000 | 0.168 ± 0.004 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.600 ± 0.018 | 0.024 ± 0.000 | 0.170 ± 0.003 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.105 ± 0.002 | 0.021 ± 0.000 | 0.134 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.109 ± 0.004 | 0.021 ± 0.000 | 0.135 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.109 ± 0.001 | 0.022 ± 0.000 | 0.137 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.036 ± 0.001 | 0.008 ± 0.000 | 0.027 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.039 ± 0.003 | 0.009 ± 0.000 | 0.029 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.035 ± 0.014 | 0.009 ± 0.000 | 0.028 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.797 ± 0.002 | 0.032 ± 0.000 | 0.122 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.833 ± 0.014 | 0.033 ± 0.000 | 0.120 ± 0.016 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.824 ± 0.005 | 0.035 ± 0.000 | 0.124 ± 0.002 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.096 ± 0.002 | 0.031 ± 0.001 | 0.124 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.097 ± 0.003 | 0.032 ± 0.000 | 0.123 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.099 ± 0.002 | 0.034 ± 0.000 | 0.128 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.049 ± 0.002 | 0.025 ± 0.000 | 0.041 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.056 ± 0.023 | 0.026 ± 0.000 | 0.043 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.049 ± 0.003 | 0.028 ± 0.000 | 0.044 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.099 ± 0.003 | 0.018 ± 0.000 | 0.207 ± 0.003 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.099 ± 0.002 | 0.018 ± 0.000 | 0.207 ± 0.003 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.099 ± 0.001 | 0.019 ± 0.000 | 0.208 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.042 ± 0.001 | 0.017 ± 0.000 | 0.099 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.042 ± 0.001 | 0.017 ± 0.000 | 0.098 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.042 ± 0.002 | 0.018 ± 0.000 | 0.098 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.042 ± 0.004 | 0.012 ± 0.000 | 0.033 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.048 ± 0.003 | 0.016 ± 0.000 | 0.037 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.052 ± 0.004 | 0.026 ± 0.000 | 0.044 ± 0.000 | 0.017 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.125 ± 0.001 | 0.014 ± 0.000 | 0.121 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.136 ± 0.002 | 0.018 ± 0.000 | 0.126 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.154 ± 0.002 | 0.027 ± 0.000 | 0.134 ± 0.002 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.037 ± 0.001 | 0.012 ± 0.000 | 0.085 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.041 ± 0.005 | 0.016 ± 0.000 | 0.087 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.049 ± 0.001 | 0.026 ± 0.000 | 0.094 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.016 ± 0.001 | 0.014 ± 0.001 | 0.023 ± 0.001 | 0.034 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 28.2x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 27.0x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 16.5x faster than the slowest successful cell (`jax-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 26.2x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 26.6x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`jax-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_jvp` (f64, threads=4, shape=`256x256`): `jax-cpu` is 87.9x faster than the slowest successful cell (`tenferro-eager`, 437.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_jvp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 150.5x faster than the slowest successful cell (`tenferro-eager`, 437.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_jvp` (f64, threads=4, shape=`512x512`): `jax-cpu` is 39.2x faster than the slowest successful cell (`tenferro-eager`, 1084.687 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_jvp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 51.7x faster than the slowest successful cell (`tenferro-eager`, 1084.687 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=4, shape=`256x256`): `jax-cpu` is 86.4x faster than the slowest successful cell (`tenferro-trace`, 431.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 203.6x faster than the slowest successful cell (`tenferro-trace`, 431.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=4, shape=`512x512`): `jax-cpu` is 32.7x faster than the slowest successful cell (`tenferro-trace`, 1011.971 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 89.8x faster than the slowest successful cell (`tenferro-trace`, 1011.971 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`512x512`): `jax-cpu` is 22.9x faster than the slowest successful cell (`tenferro-eager`, 942.750 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 68.7x faster than the slowest successful cell (`tenferro-eager`, 942.750 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_vjp` (f64, threads=4, shape=`512x512`): `jax-cpu` is 20.6x faster than the slowest successful cell (`tenferro-trace`, 911.702 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_vjp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 70.0x faster than the slowest successful cell (`tenferro-trace`, 911.702 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`256x256,rhs=1`): `jax-cpu` is 622.9x faster than the slowest successful cell (`tenferro-eager`, 439.345 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`256x256,rhs=1`): `pytorch-cpu` is 261.6x faster than the slowest successful cell (`tenferro-eager`, 439.345 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`512x512,rhs=1`): `jax-cpu` is 335.0x faster than the slowest successful cell (`tenferro-trace`, 1017.884 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`512x512,rhs=1`): `pytorch-cpu` is 498.8x faster than the slowest successful cell (`tenferro-trace`, 1017.884 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`256x256,rhs=1`): `jax-cpu` is 584.0x faster than the slowest successful cell (`tenferro-trace`, 417.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`256x256,rhs=1`): `pytorch-cpu` is 435.9x faster than the slowest successful cell (`tenferro-trace`, 417.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`512x512,rhs=1`): `jax-cpu` is 352.7x faster than the slowest successful cell (`tenferro-trace`, 1043.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`512x512,rhs=1`): `pytorch-cpu` is 356.2x faster than the slowest successful cell (`tenferro-trace`, 1043.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 37.5x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 19.1x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 35.0x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`jax-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 36.0x faster than the slowest successful cell (`jax-cpu`, 0.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 20.7x faster than the slowest successful cell (`jax-cpu`, 0.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.5x faster than the slowest successful cell (`tenferro-eager`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 11.3x faster than the slowest successful cell (`tenferro-eager`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 4146.3x faster than the slowest successful cell (`pytorch-cpu`, 35.978 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 63.7x faster than the slowest successful cell (`pytorch-cpu`, 35.978 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 1548.0x faster than the slowest successful cell (`pytorch-cpu`, 35.978 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 60.6x faster than the slowest successful cell (`tenferro-eager`, 0.571 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 23.9x faster than the slowest successful cell (`tenferro-eager`, 0.571 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 37.4x faster than the slowest successful cell (`tenferro-eager`, 0.600 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 24.5x faster than the slowest successful cell (`tenferro-eager`, 0.600 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 93.6x faster than the slowest successful cell (`tenferro-eager`, 0.797 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 24.8x faster than the slowest successful cell (`tenferro-eager`, 0.797 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 85.7x faster than the slowest successful cell (`tenferro-eager`, 0.833 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 25.4x faster than the slowest successful cell (`tenferro-eager`, 0.833 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 48.4x faster than the slowest successful cell (`tenferro-eager`, 0.824 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 23.8x faster than the slowest successful cell (`tenferro-eager`, 0.824 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 24.2x faster than the slowest successful cell (`pytorch-cpu`, 0.207 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.207 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 23.7x faster than the slowest successful cell (`pytorch-cpu`, 0.207 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.207 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 22.4x faster than the slowest successful cell (`pytorch-cpu`, 0.208 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.208 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.9x faster than the slowest successful cell (`tenferro-eager`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

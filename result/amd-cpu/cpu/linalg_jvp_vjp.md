# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `f1d4d8d961ff6b937768111888ad2a493314892a`
- Benchmark commit: `aecee8b998098c968f4a0f59d6228ac81d3e7b09`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261021_180521/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261021_180521`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261021_180521`.

- tenferro-rs commit: `f1d4d8d961ff6b937768111888ad2a493314892a`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261021_180521/cpu_ops_t1_20261021_180521.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261021_180521/linalg_jvp_vjp_t1_20261021_180521.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 8.491 ± 0.794 | 10.274 ± 0.069 | 9.117 ± 0.018 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 56.130 ± 0.235 | 69.146 ± 0.213 | 61.631 ± 0.523 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 8.443 ± 0.063 | 8.487 ± 0.108 | 9.147 ± 0.057 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 56.302 ± 0.309 | 53.789 ± 0.347 | 61.076 ± 0.329 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 5.011 ± 0.020 | 12.573 ± 0.230 | 4.465 ± 0.024 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 34.626 ± 0.202 | 84.166 ± 0.964 | 33.900 ± 0.607 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 5.353 ± 0.043 | 5.389 ± 0.040 | 4.698 ± 0.086 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 35.645 ± 0.229 | 44.399 ± 0.252 | 34.690 ± 1.350 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 7.332 ± 0.343 | 7.323 ± 0.043 | 7.246 ± 0.063 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 54.270 ± 0.310 | 58.555 ± 0.824 | 50.721 ± 0.246 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 7.155 ± 0.146 | 7.370 ± 0.074 | 7.593 ± 0.194 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 51.128 ± 0.593 | 57.079 ± 0.492 | 56.344 ± 0.578 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.744 ± 0.007 | 1.257 ± 0.009 | 0.670 ± 0.002 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 4.185 ± 0.040 | 5.805 ± 0.020 | 4.167 ± 0.032 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.745 ± 0.012 | 1.750 ± 0.018 | 0.680 ± 0.011 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.146 ± 0.047 | 9.836 ± 0.020 | 4.045 ± 0.035 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 14.781 ± 0.090 | 21.236 ± 0.111 | 15.598 ± 0.066 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 104.889 ± 0.197 | 146.257 ± 0.881 | 114.916 ± 0.331 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 14.801 ± 0.100 | 16.088 ± 0.051 | 15.744 ± 0.059 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 104.056 ± 1.787 | 121.195 ± 0.428 | 117.491 ± 1.372 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.038 ± 0.000 | 0.198 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.044 ± 0.001 | 0.201 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.049 ± 0.001 | 0.208 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.038 ± 0.000 | 0.193 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.045 ± 0.000 | 0.199 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.049 ± 0.000 | 0.203 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.034 ± 0.000 | 0.279 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.034 ± 0.000 | 0.273 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.036 ± 0.000 | 0.276 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.034 ± 0.000 | 0.261 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.034 ± 0.001 | 0.260 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.037 ± 0.000 | 0.261 ± 0.001 | 0.016 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.064 ± 0.000 | 0.220 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.066 ± 0.001 | 0.218 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.068 ± 0.001 | 0.221 ± 0.007 | 0.017 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.066 ± 0.001 | 0.248 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.067 ± 0.000 | 0.252 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.070 ± 0.000 | 0.253 ± 0.007 | 0.014 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.023 ± 0.000 | 0.416 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.023 ± 0.000 | 0.427 ± 0.021 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.024 ± 0.000 | 0.420 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.023 ± 0.011 | 0.244 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.024 ± 0.000 | 0.248 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.025 ± 0.000 | 0.248 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.023 ± 0.000 | 0.268 ± 0.005 | 0.013 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.028 ± 0.000 | 0.271 ± 0.004 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.040 ± 0.000 | 0.284 ± 0.004 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.026 ± 0.000 | 0.214 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.032 ± 0.000 | 0.219 ± 0.004 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.043 ± 0.000 | 0.227 ± 0.005 | 0.019 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261021_183748/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261021_183748`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261021_183748`.

- tenferro-rs commit: `f1d4d8d961ff6b937768111888ad2a493314892a`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261021_183748/cpu_ops_t4_20261021_183748.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261021_183748/linalg_jvp_vjp_t4_20261021_183748.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 6.721 ± 0.073 | 7.231 ± 0.156 | 10.435 ± 0.594 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 33.986 ± 0.383 | 42.234 ± 0.806 | 55.924 ± 5.132 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 6.616 ± 0.057 | 5.989 ± 0.062 | 10.902 ± 0.486 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 34.134 ± 0.287 | 32.157 ± 0.627 | 61.361 ± 8.743 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 3.319 ± 0.044 | 7.757 ± 0.217 | 5.619 ± 1.108 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 19.552 ± 0.192 | 46.842 ± 0.521 | 29.995 ± 7.969 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 3.843 ± 0.066 | 3.711 ± 0.039 | 4.380 ± 3.505 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 20.430 ± 0.075 | 24.355 ± 0.300 | 37.458 ± 6.032 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 5.394 ± 0.047 | 5.091 ± 0.070 | 10.082 ± 0.728 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 31.240 ± 0.164 | 33.490 ± 0.347 | 58.586 ± 5.405 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 5.303 ± 0.076 | 5.473 ± 0.088 | 10.486 ± 0.722 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 28.440 ± 2.971 | 32.962 ± 0.414 | 50.511 ± 5.380 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.582 ± 0.007 | 4.887 ± 17.879 | 0.912 ± 0.134 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 2.360 ± 0.040 | 3.119 ± 0.150 | 3.814 ± 0.386 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.603 ± 0.005 | 1.271 ± 0.011 | 0.909 ± 0.050 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 2.421 ± 0.061 | 4.960 ± 0.132 | 3.733 ± 0.050 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 13.068 ± 0.135 | 16.400 ± 0.102 | 17.583 ± 1.431 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 80.475 ± 0.400 | 91.003 ± 1.022 | 110.285 ± 5.109 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 13.137 ± 0.069 | 14.742 ± 0.091 | 17.137 ± 0.692 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 80.063 ± 0.483 | 80.849 ± 0.760 | 111.744 ± 3.953 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.039 ± 0.000 | 0.197 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.045 ± 0.000 | 0.201 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.050 ± 0.001 | 0.209 ± 0.005 | 0.017 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.039 ± 0.001 | 0.193 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.046 ± 0.004 | 0.201 ± 0.004 | 0.017 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.061 ± 0.028 | 0.203 ± 0.004 | 0.017 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.036 ± 0.000 | 0.385 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.036 ± 0.000 | 0.379 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.037 ± 0.000 | 0.381 ± 0.003 | 0.021 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.037 ± 0.000 | 0.336 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.038 ± 0.000 | 0.332 ± 0.009 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.039 ± 0.000 | 0.337 ± 0.012 | 0.021 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.067 ± 0.001 | 0.289 ± 0.005 | 0.013 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.068 ± 0.000 | 0.286 ± 0.006 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.070 ± 0.001 | 0.288 ± 0.007 | 0.022 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.068 ± 0.000 | 0.315 ± 0.009 | 0.013 ± 0.009 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.070 ± 0.001 | 0.316 ± 0.007 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.079 ± 0.026 | 0.317 ± 0.009 | 0.019 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.023 ± 0.000 | 0.410 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.023 ± 0.000 | 0.406 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.024 ± 0.000 | 0.412 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.024 ± 0.001 | 0.243 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.023 ± 0.000 | 0.246 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.024 ± 0.001 | 0.248 ± 0.005 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.024 ± 0.000 | 0.263 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.029 ± 0.000 | 0.270 ± 0.004 | 0.019 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.041 ± 0.000 | 0.284 ± 0.005 | 0.038 ± 0.006 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.027 ± 0.000 | 0.212 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.033 ± 0.000 | 0.219 ± 0.008 | 0.020 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.044 ± 0.000 | 0.231 ± 0.005 | 0.026 ± 0.011 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

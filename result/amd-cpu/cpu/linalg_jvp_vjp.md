# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a24fccea2dc7cee09df442104201b7c52f9b943b`
- Benchmark commit: `cc2153b9548e1542f955c011dccfb77a98196fed`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261023_175936/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261023_175936`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261023_175936`.

- tenferro-rs commit: `a24fccea2dc7cee09df442104201b7c52f9b943b`

### CPU Information

- Model: `AMD EPYC 9V45 96-Core Processor`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261023_175936/cpu_ops_t1_20261023_175936.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261023_175936/linalg_jvp_vjp_t1_20261023_175936.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 4.927 ± 0.022 | 6.177 ± 0.030 | 4.633 ± 0.009 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 31.778 ± 0.263 | 41.852 ± 1.329 | 30.643 ± 0.257 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 4.965 ± 0.028 | 5.032 ± 0.039 | 4.662 ± 0.002 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 31.889 ± 0.707 | 30.994 ± 0.132 | 30.278 ± 0.183 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 3.007 ± 0.012 | 7.812 ± 0.093 | 2.600 ± 0.037 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 21.967 ± 0.341 | 57.573 ± 4.843 | 20.305 ± 1.236 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 3.046 ± 0.011 | 3.825 ± 0.020 | 2.515 ± 0.004 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 21.713 ± 0.138 | 27.944 ± 0.869 | 19.297 ± 0.924 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 4.189 ± 0.028 | 4.635 ± 0.072 | 3.569 ± 0.036 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 32.299 ± 0.290 | 35.616 ± 2.184 | 26.135 ± 0.633 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 4.114 ± 0.005 | 5.409 ± 0.170 | 3.622 ± 0.080 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 30.335 ± 1.081 | 37.594 ± 0.948 | 27.790 ± 0.133 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.395 ± 0.004 | 0.610 ± 0.016 | 0.352 ± 0.001 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 2.383 ± 0.007 | 3.106 ± 0.020 | 2.305 ± 0.019 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.403 ± 0.003 | 1.040 ± 0.010 | 0.353 ± 0.002 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 2.376 ± 0.003 | 5.509 ± 0.082 | 2.266 ± 0.010 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 8.819 ± 0.027 | 12.959 ± 0.003 | 8.165 ± 0.019 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 62.640 ± 0.963 | 90.728 ± 1.202 | 61.589 ± 0.880 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 8.833 ± 0.041 | 9.493 ± 0.089 | 8.829 ± 0.160 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 61.634 ± 1.413 | 66.582 ± 0.255 | 63.108 ± 2.855 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.011 ± 0.000 | 0.057 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.013 ± 0.000 | 0.060 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.015 ± 0.000 | 0.061 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.011 ± 0.000 | 0.049 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.013 ± 0.000 | 0.049 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.014 ± 0.000 | 0.051 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.014 ± 0.000 | 0.097 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.015 ± 0.000 | 0.096 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.015 ± 0.000 | 0.097 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.013 ± 0.000 | 0.075 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.014 ± 0.000 | 0.076 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.014 ± 0.000 | 0.075 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.020 ± 0.000 | 0.072 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.021 ± 0.000 | 0.067 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.021 ± 0.000 | 0.067 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.020 ± 0.000 | 0.070 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.020 ± 0.001 | 0.069 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.021 ± 0.000 | 0.069 ± 0.000 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.009 ± 0.000 | 0.152 ± 0.003 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.010 ± 0.000 | 0.154 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.010 ± 0.000 | 0.147 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.009 ± 0.000 | 0.064 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.009 ± 0.000 | 0.067 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.009 ± 0.000 | 0.064 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.008 ± 0.000 | 0.095 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.011 ± 0.000 | 0.092 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.017 ± 0.000 | 0.097 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.008 ± 0.000 | 0.056 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.010 ± 0.000 | 0.057 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.016 ± 0.000 | 0.063 ± 0.001 | 0.013 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261023_182242/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261023_182242`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261023_182242`.

- tenferro-rs commit: `a24fccea2dc7cee09df442104201b7c52f9b943b`

### CPU Information

- Model: `AMD EPYC 9V45 96-Core Processor`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261023_182242/cpu_ops_t4_20261023_182242.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261023_182242/linalg_jvp_vjp_t4_20261023_182242.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 3.965 ± 0.065 | 7.256 ± 3.013 | 6.532 ± 1.239 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 19.165 ± 0.100 | 25.799 ± 0.567 | 30.908 ± 1.799 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 3.911 ± 0.040 | 3.606 ± 0.132 | 6.106 ± 1.506 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 19.176 ± 0.114 | 18.576 ± 0.590 | 31.077 ± 1.602 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 2.024 ± 0.018 | 4.033 ± 0.086 | 3.573 ± 0.960 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 12.070 ± 0.240 | 31.328 ± 0.242 | 19.787 ± 3.149 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 2.250 ± 0.018 | 2.351 ± 0.025 | 2.995 ± 1.320 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 12.230 ± 0.226 | 15.657 ± 0.109 | 16.893 ± 2.855 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 3.091 ± 0.057 | 3.055 ± 0.075 | 5.033 ± 2.312 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 18.840 ± 0.303 | 21.979 ± 1.249 | 30.527 ± 1.846 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 3.185 ± 0.051 | 3.204 ± 0.101 | 5.314 ± 1.246 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 17.663 ± 0.193 | 21.423 ± 0.232 | 29.604 ± 1.902 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.302 ± 0.003 | 0.956 ± 1.724 | 0.444 ± 0.028 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.362 ± 0.009 | 1.962 ± 0.051 | 1.690 ± 0.051 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.315 ± 0.008 | 0.629 ± 0.011 | 0.483 ± 0.006 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.371 ± 0.009 | 3.123 ± 0.059 | 1.828 ± 0.015 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 8.017 ± 0.018 | 11.693 ± 0.108 | 9.195 ± 1.021 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 46.488 ± 1.420 | 58.134 ± 1.408 | 64.047 ± 3.811 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 8.138 ± 0.136 | 8.992 ± 0.228 | 10.403 ± 0.203 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 47.013 ± 0.386 | 46.983 ± 0.180 | 64.667 ± 9.013 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.012 ± 0.000 | 0.060 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.013 ± 0.000 | 0.061 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.016 ± 0.000 | 0.064 ± 0.000 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.012 ± 0.000 | 0.051 ± 0.000 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.013 ± 0.000 | 0.051 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.016 ± 0.000 | 0.054 ± 0.000 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.016 ± 0.000 | 0.151 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.016 ± 0.000 | 0.152 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.016 ± 0.000 | 0.151 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.014 ± 0.000 | 0.113 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.014 ± 0.000 | 0.114 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.015 ± 0.000 | 0.115 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.022 ± 0.000 | 0.103 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.022 ± 0.000 | 0.103 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.023 ± 0.000 | 0.105 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.021 ± 0.000 | 0.104 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.022 ± 0.000 | 0.106 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.023 ± 0.000 | 0.107 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.010 ± 0.000 | 0.155 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.010 ± 0.001 | 0.156 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.010 ± 0.000 | 0.156 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.010 ± 0.000 | 0.066 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.010 ± 0.000 | 0.067 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.010 ± 0.000 | 0.068 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.009 ± 0.000 | 0.092 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.011 ± 0.000 | 0.094 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.018 ± 0.000 | 0.103 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.009 ± 0.000 | 0.056 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.011 ± 0.000 | 0.058 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.017 ± 0.000 | 0.065 ± 0.000 | 0.014 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

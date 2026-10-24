# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`
- Benchmark commit: `50a7dc9b3ea08f5dba14db07df1e09314cac56f0`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261024_180212/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261024_180212`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261024_180212`.

- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261024_180212/cpu_ops_t1_20261024_180212.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261024_180212/linalg_jvp_vjp_t1_20261024_180212.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 7.136 ± 0.019 | 8.675 ± 0.027 | 7.051 ± 0.024 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 47.103 ± 0.264 | 58.375 ± 0.553 | 46.866 ± 0.064 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 7.155 ± 0.020 | 7.106 ± 0.078 | 7.041 ± 0.046 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 46.880 ± 0.187 | 45.178 ± 0.272 | 45.937 ± 0.251 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 4.116 ± 0.019 | 10.568 ± 0.317 | 4.135 ± 0.041 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 28.718 ± 0.207 | 71.888 ± 1.122 | 29.913 ± 0.135 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 4.376 ± 0.016 | 5.225 ± 0.061 | 3.988 ± 0.020 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 29.571 ± 0.140 | 38.447 ± 0.946 | 29.787 ± 0.271 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 6.037 ± 0.042 | 6.263 ± 0.081 | 5.682 ± 0.083 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 44.935 ± 0.073 | 49.160 ± 0.759 | 38.814 ± 0.054 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 5.792 ± 0.037 | 6.071 ± 0.034 | 5.612 ± 0.081 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 41.529 ± 0.047 | 46.701 ± 1.113 | 43.649 ± 0.201 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.576 ± 0.003 | 1.012 ± 0.018 | 0.572 ± 0.002 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 3.401 ± 0.068 | 4.917 ± 0.097 | 3.533 ± 0.052 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.569 ± 0.005 | 1.756 ± 0.015 | 0.579 ± 0.006 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 3.374 ± 0.014 | 8.648 ± 0.651 | 3.588 ± 0.035 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 12.224 ± 0.046 | 17.897 ± 0.240 | 12.327 ± 0.020 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 86.896 ± 0.423 | 122.352 ± 1.345 | 90.384 ± 0.462 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 12.281 ± 0.094 | 13.357 ± 0.086 | 12.290 ± 0.091 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 86.261 ± 0.967 | 96.722 ± 0.720 | 91.726 ± 0.874 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.020 ± 0.000 | 0.119 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.022 ± 0.000 | 0.125 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.025 ± 0.000 | 0.131 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.020 ± 0.000 | 0.115 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.022 ± 0.000 | 0.121 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.025 ± 0.000 | 0.126 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.023 ± 0.000 | 0.182 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.024 ± 0.000 | 0.181 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.025 ± 0.000 | 0.184 ± 0.005 | 0.013 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.022 ± 0.000 | 0.171 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.022 ± 0.000 | 0.168 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.023 ± 0.000 | 0.169 ± 0.004 | 0.014 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.035 ± 0.000 | 0.139 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.036 ± 0.000 | 0.138 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.038 ± 0.000 | 0.143 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.035 ± 0.000 | 0.157 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.035 ± 0.000 | 0.160 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.037 ± 0.000 | 0.163 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.015 ± 0.000 | 0.275 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.015 ± 0.000 | 0.273 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.016 ± 0.000 | 0.276 ± 0.007 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.015 ± 0.000 | 0.154 ± 0.028 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.015 ± 0.000 | 0.153 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.015 ± 0.000 | 0.156 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.014 ± 0.000 | 0.179 ± 0.004 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.017 ± 0.000 | 0.186 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.025 ± 0.000 | 0.196 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.013 ± 0.000 | 0.135 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.016 ± 0.000 | 0.141 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.025 ± 0.000 | 0.151 ± 0.002 | 0.016 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261024_183020/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261024_183020`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261024_183020`.

- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261024_183020/cpu_ops_t4_20261024_183020.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261024_183020/linalg_jvp_vjp_t4_20261024_183020.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 5.301 ± 0.074 | 6.065 ± 0.250 | 8.806 ± 0.456 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 27.297 ± 0.254 | 33.735 ± 1.192 | 47.770 ± 2.996 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 5.272 ± 0.066 | 5.001 ± 0.172 | 8.695 ± 0.850 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 26.409 ± 0.779 | 25.999 ± 0.709 | 45.554 ± 3.038 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 2.861 ± 0.034 | 6.957 ± 0.430 | 5.346 ± 0.801 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 17.034 ± 0.111 | 38.401 ± 1.018 | 33.418 ± 10.104 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 3.533 ± 0.162 | 3.051 ± 0.046 | 5.409 ± 1.139 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 17.210 ± 0.024 | 19.469 ± 0.322 | 31.829 ± 7.501 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 4.231 ± 0.025 | 4.394 ± 0.944 | 7.547 ± 0.989 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 26.785 ± 0.115 | 26.750 ± 1.045 | 47.240 ± 8.855 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 4.296 ± 0.055 | 5.039 ± 0.199 | 8.096 ± 1.964 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 24.549 ± 0.797 | 28.723 ± 1.546 | 48.078 ± 4.274 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.471 ± 0.003 | 0.906 ± 1.810 | 0.735 ± 1.159 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.966 ± 0.012 | 2.463 ± 0.063 | 2.963 ± 0.051 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.468 ± 0.012 | 1.019 ± 0.021 | 0.787 ± 0.093 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.939 ± 0.020 | 3.960 ± 0.041 | 3.189 ± 0.681 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 11.070 ± 0.111 | 15.954 ± 0.266 | 14.164 ± 0.802 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 71.241 ± 0.643 | 79.136 ± 0.302 | 88.464 ± 2.702 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 11.149 ± 0.107 | 13.029 ± 0.221 | 14.034 ± 0.949 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 71.628 ± 0.397 | 69.942 ± 0.777 | 87.295 ± 4.117 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.020 ± 0.000 | 0.119 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.022 ± 0.000 | 0.124 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.025 ± 0.000 | 0.130 ± 0.002 | 0.013 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.020 ± 0.000 | 0.113 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.022 ± 0.000 | 0.120 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.025 ± 0.000 | 0.124 ± 0.001 | 0.013 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.024 ± 0.000 | 0.247 ± 0.005 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.024 ± 0.000 | 0.247 ± 0.004 | 0.010 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.025 ± 0.000 | 0.250 ± 0.004 | 0.016 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.022 ± 0.000 | 0.209 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.022 ± 0.000 | 0.209 ± 0.004 | 0.010 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.024 ± 0.000 | 0.211 ± 0.004 | 0.015 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.036 ± 0.000 | 0.180 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.036 ± 0.000 | 0.178 ± 0.004 | 0.010 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.038 ± 0.000 | 0.181 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.036 ± 0.000 | 0.195 ± 0.004 | 0.009 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.037 ± 0.000 | 0.198 ± 0.004 | 0.010 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.039 ± 0.000 | 0.196 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.015 ± 0.000 | 0.273 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.015 ± 0.000 | 0.274 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.016 ± 0.000 | 0.273 ± 0.005 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.015 ± 0.000 | 0.153 ± 0.000 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.015 ± 0.000 | 0.154 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.015 ± 0.000 | 0.154 ± 0.004 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.014 ± 0.000 | 0.176 ± 0.005 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.017 ± 0.000 | 0.184 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.025 ± 0.000 | 0.195 ± 0.004 | 0.018 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.013 ± 0.000 | 0.133 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.016 ± 0.000 | 0.139 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.025 ± 0.000 | 0.149 ± 0.004 | 0.019 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

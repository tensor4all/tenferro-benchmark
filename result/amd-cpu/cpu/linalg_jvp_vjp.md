# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`
- Benchmark commit: `320964928fdd481aaaf81ceae9606c0ea8df559f`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261006_172037/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261006_172037`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261006_172037`.

- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`

### CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`

### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/t4a-2010-bench-migrate/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/amd-cpu/cpu/einsum/20261006_172037/cpu_ops_t1_20261006_172037.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261006_172037/linalg_jvp_vjp_t1_20261006_172037.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 7.537 ± 0.014 | 8.831 ± 0.011 | 8.864 ± 0.755 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 47.268 ± 0.449 | 57.227 ± 0.833 | 47.633 ± 2.472 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 7.491 ± 0.084 | 7.290 ± 0.019 | 8.705 ± 0.032 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 46.895 ± 0.079 | 44.833 ± 0.345 | 45.792 ± 1.836 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 5.255 ± 0.398 | 10.846 ± 0.047 | 4.023 ± 0.023 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 29.789 ± 0.520 | 71.088 ± 0.637 | 24.310 ± 0.559 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 5.194 ± 0.011 | 4.649 ± 0.035 | 4.220 ± 0.026 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 29.331 ± 0.237 | 37.136 ± 0.473 | 28.105 ± 4.177 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 6.471 ± 0.006 | 6.279 ± 0.013 | 6.732 ± 0.035 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 46.085 ± 0.505 | 44.571 ± 0.998 | 43.010 ± 4.969 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 6.398 ± 0.065 | 6.281 ± 0.012 | 6.852 ± 0.032 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 43.404 ± 0.282 | 44.288 ± 0.363 | 44.435 ± 1.604 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.657 ± 0.002 | 1.072 ± 0.011 | 1.422 ± 0.048 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 3.627 ± 0.026 | 4.805 ± 0.069 | 4.280 ± 0.022 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.640 ± 0.002 | 1.471 ± 0.001 | 1.416 ± 0.016 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 3.551 ± 0.117 | 8.151 ± 0.020 | 4.235 ± 0.028 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 13.043 ± 0.547 | 18.086 ± 0.047 | 13.146 ± 0.674 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 89.121 ± 4.801 | 118.288 ± 1.182 | 90.660 ± 1.320 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 12.978 ± 0.426 | 13.594 ± 0.113 | 12.946 ± 0.938 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 87.677 ± 2.333 | 93.180 ± 0.806 | 97.408 ± 3.399 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.053 ± 0.000 | 0.167 ± 0.003 | 0.011 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.059 ± 0.000 | 0.171 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.062 ± 0.000 | 0.178 ± 0.016 | 0.046 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.055 ± 0.002 | 0.167 ± 0.008 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.061 ± 0.005 | 0.170 ± 0.003 | 0.016 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.065 ± 0.002 | 0.210 ± 0.004 | 0.049 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.053 ± 0.010 | 0.234 ± 0.009 | 0.013 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.054 ± 0.006 | 0.233 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.055 ± 0.000 | 0.247 ± 0.046 | 0.049 ± 0.007 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.055 ± 0.000 | 0.221 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.057 ± 0.000 | 0.219 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.057 ± 0.000 | 0.221 ± 0.005 | 0.052 ± 0.006 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.093 ± 0.002 | 0.186 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.094 ± 0.000 | 0.186 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.095 ± 0.000 | 0.188 ± 0.003 | 0.048 ± 0.004 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.094 ± 0.000 | 0.209 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.097 ± 0.014 | 0.211 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.099 ± 0.001 | 0.214 ± 0.003 | 0.042 ± 0.003 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.030 ± 0.000 | 0.347 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.031 ± 0.000 | 0.348 ± 0.001 | 0.013 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.031 ± 0.000 | 0.350 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.031 ± 0.000 | 0.209 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.032 ± 0.000 | 0.211 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.032 ± 0.000 | 0.211 ± 0.003 | 0.016 ± 0.003 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.031 ± 0.000 | 0.228 ± 0.008 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.036 ± 0.000 | 0.243 ± 0.039 | 0.016 ± 0.003 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.046 ± 0.000 | 0.242 ± 0.002 | 0.045 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.032 ± 0.000 | 0.191 ± 0.032 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.037 ± 0.000 | 0.186 ± 0.002 | 0.017 ± 0.003 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.046 ± 0.000 | 0.196 ± 0.003 | 0.053 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261006_175113/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261006_175113`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261006_175113`.

- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`

### CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4`

### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/t4a-2010-bench-migrate/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/amd-cpu/cpu/einsum/20261006_175113/cpu_ops_t4_20261006_175113.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261006_175113/linalg_jvp_vjp_t4_20261006_175113.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 6.588 ± 0.099 | 6.623 ± 0.148 | 13.844 ± 0.071 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 26.779 ± 2.387 | 32.868 ± 0.345 | 47.635 ± 11.162 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 6.632 ± 0.020 | 5.690 ± 0.130 | 13.812 ± 0.154 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 26.513 ± 0.592 | 24.135 ± 0.977 | 48.849 ± 4.550 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 3.243 ± 0.290 | 7.269 ± 0.924 | 3.156 ± 0.259 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 16.173 ± 1.167 | 35.631 ± 0.763 | 17.851 ± 3.369 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 4.061 ± 0.122 | 3.148 ± 0.034 | 3.619 ± 0.319 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 15.981 ± 0.531 | 17.956 ± 0.406 | 17.787 ± 6.863 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 5.463 ± 0.184 | 4.527 ± 0.241 | 7.920 ± 1.884 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 26.640 ± 0.352 | 26.938 ± 0.512 | 34.606 ± 1.584 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 5.047 ± 0.034 | 5.212 ± 0.182 | 7.518 ± 2.060 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 23.537 ± 0.140 | 25.770 ± 0.251 | 48.184 ± 2.333 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.745 ± 0.007 | 0.983 ± 0.019 | 1.299 ± 0.053 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 2.323 ± 0.039 | 2.501 ± 0.012 | 4.628 ± 0.258 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.707 ± 0.005 | 1.335 ± 0.060 | 1.260 ± 0.055 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 2.201 ± 0.112 | 4.250 ± 0.095 | 5.492 ± 0.227 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 13.817 ± 0.954 | 15.821 ± 0.251 | 19.151 ± 9.673 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 70.102 ± 1.438 | 87.144 ± 0.596 | 107.385 ± 10.565 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 13.496 ± 0.239 | 14.766 ± 0.111 | 18.080 ± 6.120 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 72.041 ± 5.018 | 79.937 ± 0.899 | 104.071 ± 7.663 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.063 ± 0.000 | 0.198 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.066 ± 0.005 | 0.204 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.074 ± 0.002 | 0.211 ± 0.004 | 0.032 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.066 ± 0.000 | 0.198 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.071 ± 0.000 | 0.202 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.075 ± 0.000 | 0.205 ± 0.003 | 0.045 ± 0.016 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.064 ± 0.000 | 0.301 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.065 ± 0.001 | 0.296 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.066 ± 0.001 | 0.301 ± 0.005 | 0.057 ± 0.015 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.066 ± 0.000 | 0.278 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.068 ± 0.001 | 0.276 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.069 ± 0.000 | 0.278 ± 0.003 | 0.054 ± 0.016 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.110 ± 0.000 | 0.227 ± 0.017 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.111 ± 0.000 | 0.229 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.103 ± 0.003 | 0.227 ± 0.004 | 0.052 ± 0.014 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.111 ± 0.000 | 0.259 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.114 ± 0.001 | 0.264 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.116 ± 0.001 | 0.260 ± 0.009 | 0.041 ± 0.010 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.036 ± 0.000 | 0.415 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.036 ± 0.001 | 0.421 ± 0.008 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.037 ± 0.000 | 0.415 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.037 ± 0.001 | 0.249 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.038 ± 0.000 | 0.250 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.038 ± 0.000 | 0.250 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.038 ± 0.000 | 0.265 ± 0.009 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.043 ± 0.005 | 0.271 ± 0.004 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.055 ± 0.004 | 0.269 ± 0.031 | 0.041 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.038 ± 0.000 | 0.217 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.044 ± 0.000 | 0.221 ± 0.003 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.050 ± 0.004 | 0.195 ± 0.003 | 0.044 ± 0.004 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

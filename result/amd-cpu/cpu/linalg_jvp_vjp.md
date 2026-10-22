# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`
- Benchmark commit: `12e0b50f983ca3253ff46841eb15912448ae3e20`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261022_180452/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261022_180452`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261022_180452`.

- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261022_180452/cpu_ops_t1_20261022_180452.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261022_180452/linalg_jvp_vjp_t1_20261022_180452.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 8.638 ± 0.116 | 10.346 ± 0.121 | 9.105 ± 0.092 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 57.375 ± 0.367 | 68.072 ± 0.751 | 62.438 ± 0.538 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 8.670 ± 0.064 | 8.529 ± 0.059 | 9.252 ± 0.073 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 57.077 ± 0.313 | 52.294 ± 0.271 | 61.906 ± 0.530 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 5.141 ± 0.133 | 12.537 ± 0.148 | 4.537 ± 0.165 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 35.593 ± 0.190 | 84.013 ± 1.024 | 35.023 ± 1.462 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 5.309 ± 0.051 | 5.576 ± 0.048 | 4.719 ± 0.101 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 36.060 ± 1.155 | 45.148 ± 0.809 | 34.025 ± 1.497 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 7.249 ± 0.028 | 7.497 ± 0.130 | 7.382 ± 0.072 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 53.557 ± 1.324 | 55.509 ± 0.518 | 51.482 ± 0.353 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 7.017 ± 0.055 | 7.646 ± 0.125 | 7.661 ± 0.044 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 49.705 ± 0.525 | 55.060 ± 0.122 | 57.425 ± 0.825 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.735 ± 0.008 | 1.299 ± 0.038 | 0.697 ± 0.007 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 4.243 ± 0.050 | 5.872 ± 0.050 | 4.200 ± 0.103 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.736 ± 0.004 | 1.758 ± 0.018 | 0.697 ± 0.093 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.252 ± 0.088 | 9.895 ± 0.061 | 4.082 ± 0.038 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 14.924 ± 0.250 | 19.342 ± 0.329 | 15.789 ± 0.076 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 107.195 ± 1.448 | 142.937 ± 1.178 | 119.441 ± 1.147 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 14.832 ± 0.080 | 16.159 ± 0.130 | 15.874 ± 0.178 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 108.199 ± 0.975 | 115.042 ± 1.208 | 121.601 ± 1.497 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.034 ± 0.001 | 0.198 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.040 ± 0.002 | 0.203 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.044 ± 0.001 | 0.209 ± 0.006 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.036 ± 0.001 | 0.200 ± 0.005 | 0.012 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.041 ± 0.000 | 0.204 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.045 ± 0.000 | 0.212 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.034 ± 0.000 | 0.282 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.034 ± 0.000 | 0.278 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.035 ± 0.000 | 0.280 ± 0.002 | 0.015 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.033 ± 0.000 | 0.262 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.034 ± 0.000 | 0.262 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.036 ± 0.000 | 0.263 ± 0.004 | 0.015 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.059 ± 0.001 | 0.220 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.061 ± 0.001 | 0.224 ± 0.009 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.064 ± 0.001 | 0.223 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.063 ± 0.001 | 0.250 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.063 ± 0.000 | 0.257 ± 0.006 | 0.015 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.067 ± 0.000 | 0.254 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.021 ± 0.000 | 0.416 ± 0.010 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.022 ± 0.000 | 0.414 ± 0.008 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.022 ± 0.000 | 0.416 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.021 ± 0.000 | 0.251 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.022 ± 0.000 | 0.250 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.022 ± 0.000 | 0.254 ± 0.007 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.022 ± 0.000 | 0.266 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.027 ± 0.000 | 0.274 ± 0.007 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.038 ± 0.000 | 0.287 ± 0.003 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.022 ± 0.000 | 0.220 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.027 ± 0.000 | 0.223 ± 0.006 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.039 ± 0.000 | 0.237 ± 0.002 | 0.019 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261022_183841/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261022_183841`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261022_183841`.

- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261022_183841/cpu_ops_t4_20261022_183841.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261022_183841/linalg_jvp_vjp_t4_20261022_183841.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 6.530 ± 0.053 | 7.062 ± 0.159 | 9.424 ± 0.439 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 33.252 ± 0.311 | 40.440 ± 0.901 | 61.656 ± 1.658 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 6.567 ± 0.023 | 5.943 ± 0.150 | 10.872 ± 0.289 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 33.590 ± 0.512 | 29.738 ± 0.941 | 61.262 ± 0.893 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 3.339 ± 0.038 | 7.172 ± 0.116 | 6.291 ± 1.436 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 19.282 ± 0.232 | 46.676 ± 0.474 | 37.524 ± 2.703 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 3.907 ± 0.009 | 3.589 ± 0.074 | 5.417 ± 2.202 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 20.590 ± 0.479 | 23.922 ± 0.297 | 35.946 ± 7.520 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 5.182 ± 0.114 | 4.822 ± 0.046 | 10.270 ± 1.077 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 31.408 ± 0.363 | 33.339 ± 0.758 | 53.461 ± 8.430 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 5.227 ± 0.053 | 5.251 ± 0.051 | 9.716 ± 1.482 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 27.822 ± 0.386 | 32.037 ± 0.696 | 49.938 ± 4.370 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.601 ± 0.006 | 1.816 ± 0.709 | 0.841 ± 0.006 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 2.395 ± 0.031 | 3.052 ± 0.016 | 3.804 ± 0.030 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.599 ± 0.003 | 1.234 ± 0.028 | 1.183 ± 0.974 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 2.445 ± 0.023 | 4.853 ± 0.084 | 3.476 ± 0.274 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 13.259 ± 0.078 | 18.257 ± 0.277 | 17.321 ± 1.546 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 80.952 ± 0.382 | 89.326 ± 0.952 | 102.846 ± 9.515 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 13.342 ± 0.077 | 14.584 ± 0.133 | 16.828 ± 0.874 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 80.363 ± 0.844 | 80.061 ± 0.596 | 108.356 ± 3.845 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.034 ± 0.000 | 0.197 ± 0.008 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.041 ± 0.000 | 0.200 ± 0.008 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.046 ± 0.000 | 0.206 ± 0.004 | 0.016 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.036 ± 0.000 | 0.192 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.043 ± 0.000 | 0.197 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.046 ± 0.001 | 0.205 ± 0.001 | 0.017 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.034 ± 0.000 | 0.381 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.034 ± 0.000 | 0.377 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.036 ± 0.000 | 0.378 ± 0.001 | 0.022 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.033 ± 0.000 | 0.332 ± 0.011 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.034 ± 0.000 | 0.331 ± 0.010 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.035 ± 0.000 | 0.333 ± 0.011 | 0.021 ± 0.003 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.061 ± 0.001 | 0.287 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.062 ± 0.000 | 0.284 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.065 ± 0.001 | 0.288 ± 0.005 | 0.021 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.064 ± 0.001 | 0.315 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.066 ± 0.001 | 0.317 ± 0.007 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.069 ± 0.001 | 0.318 ± 0.009 | 0.019 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.021 ± 0.000 | 0.409 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.022 ± 0.000 | 0.409 ± 0.004 | 0.013 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.022 ± 0.000 | 0.410 ± 0.003 | 0.014 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.022 ± 0.000 | 0.245 ± 0.007 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.022 ± 0.000 | 0.244 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.023 ± 0.000 | 0.244 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.022 ± 0.000 | 0.262 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.027 ± 0.001 | 0.268 ± 0.004 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.040 ± 0.001 | 0.283 ± 0.003 | 0.026 ± 0.012 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.022 ± 0.000 | 0.213 ± 0.007 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.027 ± 0.000 | 0.219 ± 0.004 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.040 ± 0.000 | 0.230 ± 0.004 | 0.026 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

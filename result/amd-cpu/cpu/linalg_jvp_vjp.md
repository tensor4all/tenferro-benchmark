# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `14bc1ab933e3c7a9c966e08355670331ab9d993c`
- Benchmark commit: `ce89bccec28bc7c863b30c80764d3570747f6df6`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261019_175954/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261019_175954`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261019_175954`.

- tenferro-rs commit: `14bc1ab933e3c7a9c966e08355670331ab9d993c`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261019_175954/cpu_ops_t1_20261019_175954.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261019_175954/linalg_jvp_vjp_t1_20261019_175954.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 5.408 ± 0.060 | 6.724 ± 0.058 | 5.088 ± 0.030 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 35.429 ± 0.241 | 46.609 ± 0.401 | 33.723 ± 0.395 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 5.534 ± 0.068 | 5.780 ± 0.657 | 5.130 ± 0.066 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 35.328 ± 0.063 | 34.903 ± 0.369 | 33.113 ± 0.668 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 3.305 ± 0.084 | 9.204 ± 0.569 | 2.815 ± 0.039 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 24.712 ± 0.108 | 64.878 ± 1.578 | 22.447 ± 0.461 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 3.258 ± 0.016 | 4.221 ± 0.052 | 2.805 ± 0.026 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 23.670 ± 0.362 | 31.096 ± 0.397 | 21.705 ± 0.269 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 4.473 ± 0.005 | 5.129 ± 0.125 | 3.890 ± 0.052 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 37.158 ± 0.246 | 44.451 ± 1.456 | 28.302 ± 0.271 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 4.515 ± 0.064 | 5.139 ± 0.048 | 3.991 ± 0.044 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 35.306 ± 0.257 | 43.816 ± 0.892 | 30.918 ± 0.176 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.430 ± 0.000 | 0.650 ± 0.022 | 0.367 ± 0.001 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 2.587 ± 0.008 | 4.002 ± 0.076 | 2.498 ± 0.029 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.422 ± 0.003 | 1.125 ± 0.181 | 0.381 ± 0.033 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 2.549 ± 0.005 | 6.137 ± 0.082 | 2.534 ± 0.027 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 9.577 ± 0.044 | 14.537 ± 0.326 | 8.855 ± 0.040 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 66.527 ± 0.609 | 99.559 ± 2.664 | 65.897 ± 1.534 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 9.558 ± 0.030 | 10.275 ± 0.143 | 8.947 ± 0.050 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 67.064 ± 0.835 | 83.559 ± 1.385 | 68.633 ± 2.993 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.019 ± 0.001 | 0.061 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.021 ± 0.000 | 0.061 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.023 ± 0.000 | 0.064 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.019 ± 0.001 | 0.051 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.021 ± 0.000 | 0.051 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.023 ± 0.000 | 0.054 ± 0.000 | 0.010 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.022 ± 0.000 | 0.102 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.023 ± 0.000 | 0.098 ± 0.001 | 0.007 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.024 ± 0.000 | 0.099 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.022 ± 0.000 | 0.076 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.023 ± 0.000 | 0.077 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.024 ± 0.000 | 0.078 ± 0.000 | 0.012 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.034 ± 0.002 | 0.070 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.035 ± 0.000 | 0.068 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.036 ± 0.000 | 0.070 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.033 ± 0.000 | 0.070 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.035 ± 0.000 | 0.070 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.036 ± 0.000 | 0.070 ± 0.002 | 0.009 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.013 ± 0.000 | 0.157 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.014 ± 0.000 | 0.157 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.014 ± 0.000 | 0.157 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.012 ± 0.000 | 0.067 ± 0.000 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.013 ± 0.000 | 0.068 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.013 ± 0.002 | 0.068 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.012 ± 0.000 | 0.094 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.014 ± 0.000 | 0.095 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.021 ± 0.000 | 0.105 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.012 ± 0.000 | 0.056 ± 0.000 | 0.007 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.014 ± 0.000 | 0.059 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.021 ± 0.000 | 0.065 ± 0.001 | 0.014 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261019_182327/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261019_182327`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261019_182327`.

- tenferro-rs commit: `14bc1ab933e3c7a9c966e08355670331ab9d993c`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261019_182327/cpu_ops_t4_20261019_182327.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261019_182327/linalg_jvp_vjp_t4_20261019_182327.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 16.881 ± 9.516 | 4.834 ± 13.758 | 6.303 ± 0.315 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 19.859 ± 0.421 | 26.791 ± 0.571 | 29.414 ± 4.032 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 15.573 ± 0.237 | 3.642 ± 0.077 | 6.463 ± 0.124 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 20.919 ± 0.433 | 19.828 ± 0.742 | 33.821 ± 4.649 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 3.340 ± 0.086 | 5.139 ± 0.099 | 3.642 ± 0.427 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 12.656 ± 0.168 | 32.686 ± 0.545 | 21.306 ± 0.952 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 3.296 ± 0.109 | 2.493 ± 0.082 | 3.286 ± 0.602 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 12.903 ± 1.232 | 16.559 ± 0.259 | 21.248 ± 2.835 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 5.470 ± 0.060 | 3.154 ± 0.051 | 5.241 ± 1.321 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 20.473 ± 0.245 | 23.068 ± 0.468 | 31.940 ± 1.720 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 5.265 ± 0.144 | 3.166 ± 0.132 | 5.148 ± 0.281 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 19.627 ± 0.136 | 22.348 ± 0.407 | 32.800 ± 2.017 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.620 ± 0.010 | 2.982 ± 1.671 | 0.471 ± 0.227 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.445 ± 0.038 | 3.070 ± 4.038 | 1.942 ± 0.086 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.586 ± 0.021 | 0.769 ± 0.018 | 0.497 ± 0.040 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.379 ± 0.038 | 2.886 ± 0.056 | 2.064 ± 0.116 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 19.022 ± 0.224 | 12.224 ± 0.453 | 9.712 ± 0.580 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 46.865 ± 1.301 | 59.170 ± 0.557 | 66.659 ± 4.127 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 14.035 ± 0.297 | 9.480 ± 0.392 | 10.409 ± 1.108 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 44.943 ± 0.879 | 52.007 ± 0.891 | 63.097 ± 3.030 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.139 ± 0.004 | 0.057 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.144 ± 0.022 | 0.059 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.128 ± 0.022 | 0.064 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.110 ± 0.008 | 0.048 ± 0.000 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.131 ± 0.011 | 0.050 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.140 ± 0.004 | 0.052 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.101 ± 0.012 | 0.144 ± 0.003 | 0.006 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.107 ± 0.008 | 0.143 ± 0.001 | 0.008 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.117 ± 0.001 | 0.146 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.093 ± 0.018 | 0.107 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.111 ± 0.002 | 0.113 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.110 ± 0.004 | 0.112 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.232 ± 0.038 | 0.098 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.222 ± 0.002 | 0.103 ± 0.004 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.227 ± 0.003 | 0.102 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.186 ± 0.008 | 0.099 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.186 ± 0.047 | 0.101 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.218 ± 0.020 | 0.104 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.074 ± 0.003 | 0.153 ± 0.005 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.072 ± 0.013 | 0.149 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.073 ± 0.011 | 0.151 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.068 ± 0.004 | 0.064 ± 0.000 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.058 ± 0.010 | 0.065 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.068 ± 0.001 | 0.067 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.052 ± 0.001 | 0.086 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.056 ± 0.001 | 0.089 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.075 ± 0.002 | 0.101 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.051 ± 0.004 | 0.053 ± 0.000 | 0.007 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.049 ± 0.007 | 0.056 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.066 ± 0.002 | 0.063 ± 0.001 | 0.014 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

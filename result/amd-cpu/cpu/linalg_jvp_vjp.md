# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`
- Benchmark commit: `ab8e941ebccb8606e3781e9d8f6bddeeeab3d2a2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261101_180030/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261101_180030/run.yaml`
- Timestamp: `2026-11-01T18:00:30.835405+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261101_180030`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

### CPU Information

- Model: `Intel(R) Xeon(R) 6973P-C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261101_180030/cpu_ops_t1_20261101_180030.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 4.542 ± 0.059 | 5.063 ± 0.041 | 6.167 ± 0.065 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 25.200 ± 0.263 | 30.473 ± 3.628 | 42.410 ± 4.305 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 4.527 ± 0.050 | 4.277 ± 0.061 | 6.105 ± 0.050 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 25.199 ± 0.491 | 27.598 ± 2.425 | 38.443 ± 5.209 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 2.115 ± 0.011 | 4.436 ± 0.048 | 4.294 ± 0.046 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 14.606 ± 0.066 | 35.715 ± 0.328 | 27.335 ± 0.344 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 2.218 ± 0.014 | 2.603 ± 0.005 | 4.291 ± 0.102 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 14.919 ± 0.129 | 19.091 ± 0.438 | 29.325 ± 0.324 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 3.172 ± 0.051 | 3.581 ± 0.024 | 5.687 ± 0.085 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 21.540 ± 0.406 | 23.563 ± 0.311 | 34.053 ± 4.117 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 3.417 ± 0.063 | 3.507 ± 0.135 | 5.525 ± 0.155 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 22.021 ± 0.168 | 22.868 ± 0.715 | 45.540 ± 4.597 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.359 ± 0.013 | 0.764 ± 0.026 | 0.565 ± 0.005 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 2.101 ± 0.018 | 3.069 ± 0.088 | 3.674 ± 0.022 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.378 ± 0.022 | 0.974 ± 0.017 | 0.581 ± 0.034 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 2.112 ± 0.023 | 4.831 ± 0.109 | 3.328 ± 0.092 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 7.591 ± 0.130 | 9.321 ± 0.724 | 10.347 ± 0.139 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 49.477 ± 0.755 | 70.566 ± 9.547 | 70.895 ± 0.305 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 7.657 ± 0.036 | 7.293 ± 0.394 | 10.262 ± 0.016 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 50.762 ± 0.561 | 49.431 ± 2.213 | 75.247 ± 3.473 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.015 ± 0.001 | 0.082 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.017 ± 0.000 | 0.087 ± 0.005 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.020 ± 0.000 | 0.107 ± 0.005 | 0.014 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.015 ± 0.000 | 0.078 ± 0.000 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.018 ± 0.000 | 0.082 ± 0.015 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.019 ± 0.001 | 0.095 ± 0.004 | 0.021 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.018 ± 0.000 | 0.129 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.022 ± 0.002 | 0.129 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.019 ± 0.000 | 0.145 ± 0.019 | 0.024 ± 0.015 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.017 ± 0.000 | 0.108 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.019 ± 0.000 | 0.111 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.018 ± 0.000 | 0.112 ± 0.011 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.026 ± 0.000 | 0.096 ± 0.002 | 0.008 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.026 ± 0.001 | 0.098 ± 0.007 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.027 ± 0.000 | 0.096 ± 0.004 | 0.014 ± 0.007 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.025 ± 0.000 | 0.103 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.025 ± 0.000 | 0.106 ± 0.005 | 0.009 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.027 ± 0.000 | 0.106 ± 0.002 | 0.012 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.012 ± 0.000 | 0.216 ± 0.031 | 0.008 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.013 ± 0.000 | 0.213 ± 0.008 | 0.009 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.012 ± 0.000 | 0.217 ± 0.013 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.012 ± 0.000 | 0.104 ± 0.011 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.013 ± 0.000 | 0.100 ± 0.001 | 0.014 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.012 ± 0.000 | 0.100 ± 0.001 | 0.013 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.011 ± 0.000 | 0.121 ± 0.003 | 0.011 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.014 ± 0.000 | 0.125 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.022 ± 0.000 | 0.138 ± 0.019 | 0.022 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.011 ± 0.000 | 0.085 ± 0.001 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.016 ± 0.001 | 0.090 ± 0.003 | 0.011 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.021 ± 0.001 | 0.096 ± 0.001 | 0.023 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261101_183239/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261101_183239/run.yaml`
- Timestamp: `2026-11-01T18:32:39.800639+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261101_183239`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

### CPU Information

- Model: `Intel(R) Xeon(R) 6973P-C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261101_183239/cpu_ops_t4_20261101_183239.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 49.995 ± 2.004 | 4.288 ± 0.361 | 7.533 ± 0.251 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 57.160 ± 2.996 | 20.458 ± 0.861 | 39.962 ± 7.008 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 49.998 ± 3.991 | 3.530 ± 0.073 | 7.021 ± 0.461 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 24.213 ± 2.870 | 15.072 ± 2.172 | 35.963 ± 4.421 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 53.001 ± 3.992 | 2.918 ± 0.048 | 4.476 ± 0.408 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 27.372 ± 22.330 | 21.399 ± 3.970 | 26.340 ± 6.367 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 42.058 ± 1.243 | 1.886 ± 0.096 | 4.130 ± 1.180 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 39.860 ± 23.810 | 11.723 ± 0.157 | 28.557 ± 3.930 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 89.996 ± 7.000 | 3.540 ± 0.742 | 7.126 ± 0.145 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 28.097 ± 62.769 | 14.026 ± 1.235 | 39.381 ± 6.982 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 87.005 ± 3.002 | 3.028 ± 0.009 | 7.360 ± 0.990 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 94.988 ± 2.761 | 13.482 ± 0.922 | 40.156 ± 7.186 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 13.997 ± 0.006 | 2.591 ± 1.937 | 0.725 ± 0.171 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 9.993 ± 6.941 | 2.055 ± 3.859 | 3.345 ± 0.515 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 15.942 ± 1.353 | 0.813 ± 0.047 | 0.759 ± 0.064 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 14.902 ± 1.726 | 2.916 ± 0.084 | 3.135 ± 0.067 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 16.078 ± 0.634 | 8.511 ± 0.308 | 12.981 ± 0.725 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 47.985 ± 2.862 | 47.232 ± 5.623 | 77.968 ± 9.952 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 22.999 ± 1.292 | 8.557 ± 0.940 | 12.389 ± 0.480 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 58.595 ± 3.877 | 40.208 ± 7.051 | 73.949 ± 6.772 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.021 ± 0.001 | 0.087 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.022 ± 0.001 | 0.089 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.026 ± 0.001 | 0.094 ± 0.002 | 0.017 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.020 ± 0.000 | 0.079 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.021 ± 0.000 | 0.082 ± 0.001 | 0.010 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.025 ± 0.000 | 0.084 ± 0.001 | 0.018 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.025 ± 0.000 | 0.185 ± 0.023 | 0.008 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.025 ± 0.001 | 0.170 ± 0.004 | 0.010 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.026 ± 0.000 | 0.172 ± 0.004 | 0.016 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.023 ± 0.000 | 0.157 ± 0.026 | 0.008 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.023 ± 0.001 | 0.139 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.024 ± 0.001 | 0.138 ± 0.001 | 0.014 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.035 ± 0.000 | 0.122 ± 0.004 | 0.008 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.036 ± 0.001 | 0.123 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.037 ± 0.000 | 0.139 ± 0.002 | 0.018 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.034 ± 0.000 | 0.125 ± 0.000 | 0.008 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.035 ± 0.000 | 0.127 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.037 ± 0.001 | 0.128 ± 0.017 | 0.017 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.016 ± 0.000 | 0.244 ± 0.050 | 0.008 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.017 ± 0.000 | 0.215 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.017 ± 0.000 | 0.216 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.016 ± 0.000 | 0.150 ± 0.076 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.016 ± 0.000 | 0.101 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.017 ± 0.000 | 0.103 ± 0.002 | 0.011 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.014 ± 0.000 | 0.137 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.017 ± 0.000 | 0.129 ± 0.002 | 0.012 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.025 ± 0.000 | 0.157 ± 0.007 | 0.020 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.013 ± 0.000 | 0.088 ± 0.002 | 0.009 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.016 ± 0.000 | 0.089 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.023 ± 0.001 | 0.107 ± 0.013 | 0.020 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`
- Benchmark commit: `41db23777fb8781bbb7be5c457ac47b9216e2e3e`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261031_220732/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261031_220732/run.yaml`
- Timestamp: `2026-10-31T22:07:32.514790+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261031_220732`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
- Vendor: `AuthenticAMD`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261031_220732/cpu_ops_t1_20261031_220732.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 7.292 ± 0.016 | 8.627 ± 0.011 | 6.973 ± 0.021 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 47.332 ± 0.917 | 57.086 ± 0.310 | 44.564 ± 0.167 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 7.395 ± 0.043 | 7.090 ± 0.047 | 6.950 ± 0.049 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 47.116 ± 0.039 | 44.412 ± 0.327 | 44.237 ± 0.182 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 4.115 ± 0.027 | 10.125 ± 0.014 | 4.076 ± 0.016 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 28.847 ± 0.196 | 69.205 ± 0.883 | 26.745 ± 0.184 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 4.385 ± 0.031 | 4.535 ± 0.019 | 3.960 ± 0.029 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 29.757 ± 0.041 | 35.972 ± 0.129 | 26.498 ± 0.131 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 6.100 ± 0.042 | 6.217 ± 0.037 | 5.521 ± 0.040 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 44.919 ± 0.064 | 45.941 ± 0.232 | 36.598 ± 0.310 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 5.952 ± 0.034 | 6.101 ± 0.071 | 5.481 ± 0.015 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 41.893 ± 0.643 | 44.515 ± 0.117 | 40.369 ± 0.132 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.576 ± 0.002 | 1.007 ± 0.019 | 0.564 ± 0.001 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 3.408 ± 0.027 | 4.659 ± 0.035 | 3.495 ± 0.021 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.577 ± 0.003 | 1.411 ± 0.021 | 0.571 ± 0.005 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 3.368 ± 0.032 | 7.980 ± 0.043 | 3.580 ± 0.023 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 12.478 ± 0.094 | 17.625 ± 0.091 | 12.177 ± 0.041 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 86.265 ± 0.348 | 117.730 ± 0.685 | 87.788 ± 0.083 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 12.357 ± 0.118 | 13.301 ± 0.021 | 12.154 ± 0.034 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 86.385 ± 0.357 | 93.839 ± 0.778 | 92.238 ± 0.329 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.019 ± 0.000 | 0.118 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.022 ± 0.000 | 0.123 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.025 ± 0.000 | 0.129 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.019 ± 0.000 | 0.113 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.022 ± 0.000 | 0.118 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.025 ± 0.000 | 0.124 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.023 ± 0.000 | 0.178 ± 0.007 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.024 ± 0.001 | 0.178 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.024 ± 0.000 | 0.180 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.022 ± 0.000 | 0.165 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.022 ± 0.000 | 0.165 ± 0.007 | 0.010 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.023 ± 0.000 | 0.168 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.034 ± 0.000 | 0.134 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.035 ± 0.000 | 0.135 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.036 ± 0.000 | 0.139 ± 0.002 | 0.015 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.034 ± 0.000 | 0.154 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.035 ± 0.000 | 0.159 ± 0.004 | 0.010 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.038 ± 0.000 | 0.161 ± 0.002 | 0.011 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.015 ± 0.000 | 0.273 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.015 ± 0.000 | 0.272 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.016 ± 0.000 | 0.274 ± 0.004 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.015 ± 0.000 | 0.151 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.015 ± 0.000 | 0.151 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.015 ± 0.000 | 0.151 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.014 ± 0.000 | 0.176 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.017 ± 0.000 | 0.182 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.025 ± 0.000 | 0.194 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.013 ± 0.000 | 0.132 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.016 ± 0.000 | 0.137 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.024 ± 0.000 | 0.147 ± 0.001 | 0.015 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261031_224329/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261031_224329/run.yaml`
- Timestamp: `2026-10-31T22:43:29.944486+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261031_224329`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
- Vendor: `AuthenticAMD`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261031_224329/cpu_ops_t4_20261031_224329.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 16.006 ± 11.530 | 6.408 ± 6.126 | 8.219 ± 0.522 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 39.667 ± 33.700 | 35.370 ± 0.994 | 45.195 ± 1.785 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 16.132 ± 10.539 | 5.357 ± 0.203 | 8.683 ± 0.443 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 39.955 ± 3.146 | 25.292 ± 0.971 | 45.919 ± 5.964 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 20.994 ± 6.451 | 8.343 ± 0.274 | 5.350 ± 0.893 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 43.360 ± 17.926 | 39.802 ± 0.260 | 28.963 ± 6.505 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 26.800 ± 4.351 | 3.159 ± 0.122 | 4.415 ± 0.857 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 36.370 ± 8.605 | 20.701 ± 0.468 | 28.954 ± 3.992 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 25.995 ± 47.761 | 4.107 ± 0.103 | 8.013 ± 1.548 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 110.998 ± 49.852 | 28.846 ± 0.136 | 48.673 ± 3.354 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 28.318 ± 32.219 | 4.834 ± 0.228 | 7.642 ± 0.671 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 65.169 ± 51.003 | 27.591 ± 0.488 | 45.746 ± 10.920 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 14.997 ± 8.526 | 1.505 ± 3.759 | 0.884 ± 0.284 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 4.637 ± 8.976 | 2.493 ± 1.972 | 3.010 ± 0.072 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 17.997 ± 0.016 | 1.034 ± 0.027 | 0.888 ± 0.428 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 3.999 ± 0.048 | 4.018 ± 0.033 | 3.163 ± 0.065 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 19.001 ± 0.383 | 14.219 ± 0.205 | 13.080 ± 1.438 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 79.198 ± 1.250 | 80.424 ± 0.524 | 86.962 ± 4.009 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 27.908 ± 4.363 | 12.922 ± 0.111 | 15.022 ± 0.997 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 90.834 ± 0.172 | 71.723 ± 0.468 | 88.259 ± 7.356 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.034 ± 0.000 | 0.120 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.037 ± 0.000 | 0.125 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.040 ± 0.001 | 0.130 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.033 ± 0.000 | 0.116 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.036 ± 0.000 | 0.121 ± 0.001 | 0.011 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.039 ± 0.000 | 0.126 ± 0.001 | 0.013 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.042 ± 0.000 | 0.248 ± 0.006 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.042 ± 0.000 | 0.244 ± 0.004 | 0.010 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.043 ± 0.000 | 0.250 ± 0.005 | 0.017 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.038 ± 0.000 | 0.210 ± 0.005 | 0.009 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.038 ± 0.000 | 0.210 ± 0.005 | 0.010 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.039 ± 0.000 | 0.211 ± 0.008 | 0.015 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.061 ± 0.000 | 0.180 ± 0.004 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.061 ± 0.000 | 0.180 ± 0.004 | 0.011 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.063 ± 0.000 | 0.183 ± 0.005 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.061 ± 0.000 | 0.195 ± 0.004 | 0.008 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.061 ± 0.000 | 0.200 ± 0.032 | 0.010 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.063 ± 0.000 | 0.198 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.027 ± 0.000 | 0.272 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.027 ± 0.000 | 0.272 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.027 ± 0.000 | 0.277 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.026 ± 0.000 | 0.154 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.026 ± 0.000 | 0.154 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.027 ± 0.000 | 0.156 ± 0.005 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.022 ± 0.000 | 0.179 ± 0.004 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.025 ± 0.000 | 0.185 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.033 ± 0.000 | 0.198 ± 0.002 | 0.018 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.020 ± 0.000 | 0.134 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.023 ± 0.000 | 0.140 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.031 ± 0.000 | 0.150 ± 0.002 | 0.019 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

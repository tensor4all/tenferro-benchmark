# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`
- Benchmark commit: `4cc17b1c7061855cfd9ddc2d549a09f50199de3c`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261101_012518/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261101_012518/run.yaml`
- Timestamp: `2026-11-01T01:25:18.859596+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261101_012518`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261101_012518/cpu_ops_t1_20261101_012518.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 8.533 ± 0.075 | 10.284 ± 0.126 | 9.047 ± 0.021 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 55.960 ± 0.486 | 69.611 ± 0.072 | 56.722 ± 0.187 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 8.562 ± 0.077 | 8.493 ± 0.207 | 9.059 ± 0.089 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 54.542 ± 1.532 | 54.987 ± 0.207 | 58.355 ± 0.597 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 5.016 ± 0.065 | 12.048 ± 0.085 | 4.319 ± 0.048 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 33.742 ± 0.125 | 83.877 ± 1.070 | 32.452 ± 1.281 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 5.266 ± 0.020 | 5.375 ± 0.091 | 4.297 ± 0.046 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 35.327 ± 0.625 | 42.467 ± 0.338 | 33.271 ± 1.287 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 7.150 ± 0.093 | 7.462 ± 0.030 | 7.098 ± 0.055 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 51.731 ± 0.510 | 54.928 ± 0.386 | 49.395 ± 2.430 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 7.098 ± 0.209 | 7.447 ± 0.032 | 7.204 ± 0.039 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 48.009 ± 0.162 | 54.337 ± 0.406 | 53.311 ± 3.156 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.731 ± 0.003 | 1.247 ± 0.021 | 0.703 ± 0.011 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 4.118 ± 0.106 | 5.703 ± 0.057 | 4.078 ± 0.032 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.735 ± 0.004 | 1.728 ± 0.016 | 0.673 ± 0.004 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.132 ± 0.059 | 9.819 ± 0.173 | 4.003 ± 0.027 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 14.609 ± 0.064 | 18.735 ± 0.119 | 15.637 ± 0.211 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 101.813 ± 0.754 | 144.616 ± 0.906 | 107.506 ± 0.521 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 14.736 ± 0.175 | 15.896 ± 0.071 | 17.245 ± 0.164 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 101.446 ± 0.323 | 120.673 ± 0.806 | 117.131 ± 1.587 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.034 ± 0.001 | 0.197 ± 0.006 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.040 ± 0.000 | 0.202 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.044 ± 0.000 | 0.207 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.036 ± 0.000 | 0.194 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.042 ± 0.001 | 0.198 ± 0.005 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.046 ± 0.000 | 0.205 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.031 ± 0.001 | 0.277 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.032 ± 0.000 | 0.274 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.033 ± 0.000 | 0.279 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.030 ± 0.000 | 0.255 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.031 ± 0.001 | 0.256 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.033 ± 0.001 | 0.262 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.057 ± 0.000 | 0.221 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.059 ± 0.000 | 0.220 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.062 ± 0.000 | 0.225 ± 0.009 | 0.018 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.061 ± 0.001 | 0.246 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.063 ± 0.000 | 0.248 ± 0.009 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.066 ± 0.000 | 0.251 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.021 ± 0.000 | 0.411 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.021 ± 0.000 | 0.416 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.022 ± 0.000 | 0.417 ± 0.008 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.022 ± 0.000 | 0.246 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.021 ± 0.000 | 0.248 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.023 ± 0.000 | 0.248 ± 0.002 | 0.016 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.021 ± 0.000 | 0.263 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.027 ± 0.000 | 0.271 ± 0.003 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.038 ± 0.000 | 0.285 ± 0.004 | 0.019 ± 0.017 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.022 ± 0.001 | 0.213 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.027 ± 0.000 | 0.219 ± 0.002 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.039 ± 0.000 | 0.230 ± 0.003 | 0.036 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261101_020627/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261101_020627/run.yaml`
- Timestamp: `2026-11-01T02:06:27.548364+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261101_020627`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261101_020627/cpu_ops_t4_20261101_020627.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 34.999 ± 0.998 | 7.194 ± 0.160 | 11.395 ± 0.333 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 47.661 ± 0.823 | 42.654 ± 0.855 | 49.418 ± 2.325 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 25.993 ± 0.022 | 5.982 ± 0.172 | 11.437 ± 0.172 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 46.892 ± 0.700 | 31.114 ± 1.008 | 48.980 ± 1.653 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 45.969 ± 6.001 | 7.567 ± 0.134 | 6.078 ± 0.873 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 50.150 ± 15.182 | 47.712 ± 0.554 | 32.943 ± 4.462 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 37.021 ± 3.769 | 3.733 ± 0.082 | 4.499 ± 4.076 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 40.422 ± 25.912 | 24.319 ± 0.228 | 35.428 ± 4.078 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 35.973 ± 16.589 | 5.107 ± 0.292 | 8.989 ± 1.836 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 70.582 ± 23.069 | 33.982 ± 0.169 | 48.321 ± 11.157 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 57.009 ± 8.231 | 5.338 ± 0.031 | 10.278 ± 1.919 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 57.028 ± 52.437 | 32.681 ± 0.249 | 54.103 ± 9.679 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 2.997 ± 0.027 | 2.993 ± 1.917 | 0.921 ± 0.040 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 4.011 ± 0.894 | 3.012 ± 0.040 | 3.796 ± 0.047 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 2.999 ± 0.004 | 1.286 ± 0.052 | 0.961 ± 0.054 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 4.081 ± 4.919 | 4.832 ± 0.075 | 3.749 ± 0.023 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 21.689 ± 0.179 | 16.548 ± 0.081 | 16.439 ± 1.233 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 90.717 ± 11.655 | 92.877 ± 1.104 | 101.943 ± 7.093 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 32.038 ± 0.169 | 14.783 ± 0.163 | 17.681 ± 0.845 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 103.968 ± 6.761 | 82.403 ± 0.513 | 105.902 ± 13.885 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.050 ± 0.001 | 0.196 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.056 ± 0.000 | 0.201 ± 0.006 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.061 ± 0.001 | 0.211 ± 0.006 | 0.019 ± 0.003 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.051 ± 0.001 | 0.194 ± 0.005 | 0.012 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.057 ± 0.000 | 0.200 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.062 ± 0.000 | 0.208 ± 0.003 | 0.020 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.052 ± 0.002 | 0.383 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.052 ± 0.001 | 0.381 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.054 ± 0.001 | 0.385 ± 0.005 | 0.020 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.049 ± 0.000 | 0.336 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.050 ± 0.001 | 0.336 ± 0.011 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.051 ± 0.000 | 0.342 ± 0.008 | 0.018 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.085 ± 0.001 | 0.284 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.087 ± 0.000 | 0.286 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.090 ± 0.000 | 0.294 ± 0.004 | 0.022 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.087 ± 0.002 | 0.317 ± 0.009 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.088 ± 0.001 | 0.319 ± 0.009 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.091 ± 0.002 | 0.325 ± 0.009 | 0.019 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.033 ± 0.000 | 0.409 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.033 ± 0.000 | 0.411 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.034 ± 0.001 | 0.417 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.037 ± 0.003 | 0.246 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.034 ± 0.000 | 0.246 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.035 ± 0.000 | 0.250 ± 0.006 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.031 ± 0.000 | 0.263 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.036 ± 0.001 | 0.270 ± 0.003 | 0.019 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.048 ± 0.000 | 0.289 ± 0.013 | 0.032 ± 0.004 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.030 ± 0.000 | 0.216 ± 0.006 | 0.015 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.036 ± 0.001 | 0.222 ± 0.004 | 0.020 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.048 ± 0.001 | 0.236 ± 0.002 | 0.026 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

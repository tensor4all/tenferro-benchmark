# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `ec30d812749ef2bd6dc72245a56150aa77cb2f9c`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261108_180750/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261108_180750/run.yaml`
- Timestamp: `2026-11-08T18:07:50.799046+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261108_180750`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261108_180750/cpu_ops_t1_20261108_180750.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 17.221 ± 0.427 | 11.600 ± 0.088 | 10.275 ± 0.156 | 9.026 ± 0.068 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 106.670 ± 0.429 | 75.544 ± 0.780 | 64.966 ± 0.257 | 56.404 ± 0.571 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 9.429 ± 0.073 | 11.658 ± 0.883 | 8.430 ± 0.146 | 8.976 ± 0.074 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 61.763 ± 1.504 | 75.229 ± 0.313 | 51.246 ± 0.686 | 56.157 ± 0.345 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 10.998 ± 0.054 | 5.665 ± 0.053 | 12.327 ± 0.057 | 4.345 ± 0.031 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 74.472 ± 1.254 | 37.958 ± 0.326 | 78.660 ± 1.229 | 30.921 ± 0.900 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 10.207 ± 0.079 | 5.924 ± 0.019 | 5.346 ± 0.097 | 4.422 ± 0.131 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 71.380 ± 3.044 | 39.491 ± 0.933 | 41.355 ± 0.314 | 31.895 ± 0.232 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 17.024 ± 0.110 | 10.109 ± 0.061 | 7.177 ± 0.031 | 7.042 ± 0.015 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 115.310 ± 1.097 | 73.023 ± 1.376 | 51.008 ± 0.705 | 50.796 ± 1.116 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 12.655 ± 0.030 | 9.893 ± 0.123 | 7.244 ± 0.139 | 7.202 ± 0.064 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 88.939 ± 0.547 | 69.390 ± 0.664 | 50.483 ± 0.436 | 50.966 ± 2.243 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.962 ± 0.041 | 1.401 ± 0.007 | 1.238 ± 0.016 | 0.687 ± 0.007 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 9.057 ± 0.181 | 8.061 ± 0.107 | 5.403 ± 0.084 | 4.122 ± 0.059 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.951 ± 0.015 | 1.384 ± 0.002 | 1.705 ± 0.006 | 0.661 ± 0.004 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.918 ± 0.110 | 7.975 ± 0.053 | 9.261 ± 0.055 | 3.999 ± 0.017 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 28.455 ± 0.101 | 19.813 ± 0.094 | 21.088 ± 0.374 | 15.679 ± 0.194 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 190.451 ± 0.317 | 138.263 ± 0.340 | 132.364 ± 0.834 | 108.975 ± 0.807 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 14.786 ± 0.101 | 19.842 ± 0.162 | 15.834 ± 0.074 | 15.657 ± 0.065 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 100.299 ± 0.942 | 139.270 ± 1.498 | 106.184 ± 1.200 | 115.309 ± 2.168 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.249 ± 0.006 | 0.036 ± 0.000 | 0.192 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.263 ± 0.002 | 0.043 ± 0.000 | 0.196 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.272 ± 0.003 | 0.049 ± 0.000 | 0.201 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.098 ± 0.001 | 0.037 ± 0.000 | 0.192 ± 0.004 | 0.012 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.106 ± 0.000 | 0.045 ± 0.000 | 0.196 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.113 ± 0.001 | 0.051 ± 0.000 | 0.200 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.691 ± 0.007 | 0.035 ± 0.000 | 0.270 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.696 ± 0.008 | 0.036 ± 0.001 | 0.269 ± 0.008 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.697 ± 0.003 | 0.037 ± 0.000 | 0.269 ± 0.006 | 0.014 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.187 ± 0.001 | 0.033 ± 0.000 | 0.255 ± 0.006 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.189 ± 0.001 | 0.034 ± 0.000 | 0.254 ± 0.005 | 0.015 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.192 ± 0.001 | 0.036 ± 0.000 | 0.253 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.945 ± 0.006 | 0.059 ± 0.001 | 0.218 ± 0.006 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.952 ± 0.004 | 0.061 ± 0.001 | 0.212 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.958 ± 0.001 | 0.065 ± 0.001 | 0.216 ± 0.002 | 0.017 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.178 ± 0.005 | 0.064 ± 0.001 | 0.242 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.182 ± 0.002 | 0.065 ± 0.001 | 0.244 ± 0.005 | 0.015 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.187 ± 0.002 | 0.069 ± 0.000 | 0.245 ± 0.004 | 0.015 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.184 ± 0.004 | 0.025 ± 0.001 | 0.402 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.184 ± 0.004 | 0.026 ± 0.000 | 0.404 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.190 ± 0.002 | 0.027 ± 0.000 | 0.402 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.094 ± 0.000 | 0.025 ± 0.000 | 0.240 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.095 ± 0.002 | 0.025 ± 0.000 | 0.243 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.096 ± 0.001 | 0.027 ± 0.001 | 0.241 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.223 ± 0.002 | 0.023 ± 0.000 | 0.257 ± 0.006 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.236 ± 0.001 | 0.032 ± 0.000 | 0.268 ± 0.005 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.264 ± 0.002 | 0.046 ± 0.000 | 0.277 ± 0.003 | 0.019 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.091 ± 0.001 | 0.023 ± 0.000 | 0.208 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.098 ± 0.000 | 0.031 ± 0.000 | 0.216 ± 0.005 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.111 ± 0.001 | 0.046 ± 0.000 | 0.224 ± 0.003 | 0.019 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261108_184953/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261108_184953/run.yaml`
- Timestamp: `2026-11-08T18:49:53.361000+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261108_184953`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261108_184953/cpu_ops_t4_20261108_184953.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 30.499 ± 0.245 | 26.922 ± 0.315 | 7.035 ± 0.122 | 11.118 ± 0.428 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 154.271 ± 4.871 | 119.162 ± 3.354 | 42.890 ± 0.824 | 54.171 ± 5.036 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 16.685 ± 0.600 | 26.630 ± 1.029 | 5.939 ± 0.075 | 10.038 ± 0.498 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 84.774 ± 0.811 | 117.593 ± 0.054 | 31.087 ± 0.724 | 54.069 ± 8.380 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 16.301 ± 0.132 | 9.451 ± 0.444 | 8.610 ± 0.161 | 5.181 ± 0.828 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 385.737 ± 28.375 | 329.402 ± 24.013 | 46.970 ± 0.661 | 35.952 ± 8.148 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 14.448 ± 9.017 | 9.837 ± 0.020 | 3.681 ± 0.048 | 5.071 ± 0.136 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 213.073 ± 20.969 | 333.809 ± 25.111 | 24.150 ± 0.269 | 28.601 ± 2.557 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 202.149 ± 76.717 | 15.431 ± 23.411 | 4.910 ± 0.068 | 10.010 ± 1.294 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 443.387 ± 10.346 | 386.699 ± 12.163 | 33.386 ± 0.599 | 55.299 ± 7.760 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 17.524 ± 23.517 | 15.780 ± 0.291 | 5.287 ± 0.066 | 9.276 ± 2.420 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 262.381 ± 7.609 | 379.478 ± 13.787 | 32.210 ± 0.954 | 45.658 ± 6.761 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 3.228 ± 0.209 | 2.683 ± 0.083 | 1.072 ± 0.057 | 0.894 ± 0.061 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 314.237 ± 20.656 | 332.031 ± 3.060 | 2.976 ± 0.073 | 3.815 ± 0.068 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.556 ± 0.023 | 2.673 ± 0.163 | 1.265 ± 0.035 | 0.986 ± 0.019 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 159.676 ± 23.687 | 325.869 ± 8.087 | 4.776 ± 0.033 | 3.740 ± 0.156 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 35.467 ± 0.498 | 24.576 ± 0.177 | 16.340 ± 0.170 | 16.838 ± 0.847 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 238.401 ± 2.011 | 178.191 ± 1.028 | 92.072 ± 0.395 | 109.888 ± 3.726 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 20.199 ± 0.096 | 26.132 ± 0.219 | 14.654 ± 0.294 | 19.023 ± 1.264 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 126.608 ± 2.274 | 178.555 ± 1.008 | 81.643 ± 1.347 | 104.918 ± 8.163 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.262 ± 0.004 | 0.046 ± 0.001 | 0.192 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.274 ± 0.004 | 0.054 ± 0.000 | 0.199 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.282 ± 0.003 | 0.059 ± 0.001 | 0.206 ± 0.005 | 0.018 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.110 ± 0.003 | 0.047 ± 0.000 | 0.191 ± 0.002 | 0.012 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.119 ± 0.003 | 0.055 ± 0.000 | 0.198 ± 0.007 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.124 ± 0.000 | 0.061 ± 0.000 | 0.197 ± 0.004 | 0.019 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.717 ± 0.022 | 0.045 ± 0.001 | 0.377 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.725 ± 0.006 | 0.046 ± 0.000 | 0.374 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.722 ± 0.004 | 0.047 ± 0.000 | 0.376 ± 0.004 | 0.020 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.218 ± 0.005 | 0.043 ± 0.000 | 0.328 ± 0.008 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.220 ± 0.004 | 0.044 ± 0.000 | 0.326 ± 0.010 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.224 ± 0.003 | 0.045 ± 0.000 | 0.330 ± 0.012 | 0.018 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.960 ± 0.021 | 0.074 ± 0.001 | 0.282 ± 0.006 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.961 ± 0.013 | 0.077 ± 0.001 | 0.281 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.977 ± 0.009 | 0.079 ± 0.002 | 0.283 ± 0.006 | 0.022 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.211 ± 0.004 | 0.079 ± 0.000 | 0.308 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.212 ± 0.002 | 0.082 ± 0.006 | 0.311 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.216 ± 0.002 | 0.084 ± 0.001 | 0.312 ± 0.007 | 0.019 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.197 ± 0.009 | 0.034 ± 0.000 | 0.403 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.199 ± 0.002 | 0.035 ± 0.000 | 0.403 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.200 ± 0.003 | 0.036 ± 0.001 | 0.403 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.106 ± 0.002 | 0.034 ± 0.001 | 0.238 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.107 ± 0.002 | 0.035 ± 0.000 | 0.239 ± 0.003 | 0.015 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.108 ± 0.001 | 0.036 ± 0.000 | 0.241 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.232 ± 0.002 | 0.029 ± 0.000 | 0.260 ± 0.007 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.247 ± 0.001 | 0.038 ± 0.001 | 0.267 ± 0.003 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.269 ± 0.001 | 0.051 ± 0.001 | 0.281 ± 0.006 | 0.026 ± 0.006 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.100 ± 0.003 | 0.027 ± 0.000 | 0.206 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.109 ± 0.002 | 0.037 ± 0.000 | 0.217 ± 0.004 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.121 ± 0.003 | 0.051 ± 0.000 | 0.225 ± 0.003 | 0.031 ± 0.006 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

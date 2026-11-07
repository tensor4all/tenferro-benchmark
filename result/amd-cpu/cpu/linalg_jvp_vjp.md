# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `d6655b567e09aa29c3201655e1de418716934876`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261107_180711/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261107_180711/run.yaml`
- Timestamp: `2026-11-07T18:07:11.532995+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261107_180711`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261107_180711/cpu_ops_t1_20261107_180711.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 16.750 ± 0.070 | 11.487 ± 0.045 | 10.209 ± 0.055 | 8.909 ± 0.033 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 106.060 ± 0.529 | 74.613 ± 0.267 | 65.218 ± 0.110 | 57.853 ± 0.312 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 9.297 ± 0.092 | 11.623 ± 0.050 | 8.451 ± 0.075 | 8.881 ± 0.038 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 60.321 ± 0.391 | 74.750 ± 0.576 | 51.159 ± 0.240 | 57.019 ± 0.170 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 10.942 ± 0.056 | 5.631 ± 0.048 | 12.388 ± 0.057 | 4.349 ± 0.059 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 74.338 ± 1.201 | 37.988 ± 0.304 | 78.165 ± 1.136 | 30.928 ± 0.184 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 10.052 ± 0.032 | 5.889 ± 0.027 | 5.401 ± 0.039 | 4.298 ± 0.017 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 71.749 ± 0.928 | 39.254 ± 0.681 | 41.697 ± 0.671 | 31.692 ± 0.112 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 16.481 ± 0.099 | 10.069 ± 1.653 | 7.274 ± 0.040 | 7.042 ± 0.028 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 114.530 ± 1.001 | 72.332 ± 0.550 | 50.679 ± 0.424 | 45.580 ± 2.127 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 12.514 ± 0.009 | 9.880 ± 0.064 | 7.299 ± 0.065 | 7.196 ± 0.035 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 90.044 ± 0.905 | 68.570 ± 0.540 | 50.366 ± 0.329 | 51.047 ± 2.264 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.822 ± 0.019 | 1.391 ± 0.007 | 1.240 ± 0.021 | 0.668 ± 0.004 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 8.994 ± 0.116 | 8.018 ± 0.015 | 5.609 ± 0.098 | 4.140 ± 0.075 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.865 ± 0.035 | 1.389 ± 0.012 | 2.115 ± 0.032 | 0.659 ± 0.004 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.427 ± 0.068 | 7.996 ± 0.044 | 9.597 ± 0.064 | 4.055 ± 0.051 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 28.586 ± 0.388 | 19.612 ± 0.050 | 21.114 ± 0.113 | 15.379 ± 0.084 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 191.774 ± 3.339 | 139.758 ± 0.500 | 135.345 ± 0.744 | 109.819 ± 0.161 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 14.833 ± 0.043 | 19.655 ± 0.104 | 15.832 ± 0.035 | 15.371 ± 0.047 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 101.067 ± 0.239 | 140.888 ± 0.961 | 106.974 ± 0.448 | 115.285 ± 0.365 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.251 ± 0.001 | 0.037 ± 0.001 | 0.196 ± 0.005 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.266 ± 0.002 | 0.044 ± 0.001 | 0.199 ± 0.005 | 0.014 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.275 ± 0.003 | 0.051 ± 0.001 | 0.207 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.101 ± 0.001 | 0.039 ± 0.000 | 0.194 ± 0.006 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.110 ± 0.003 | 0.047 ± 0.000 | 0.199 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.115 ± 0.002 | 0.053 ± 0.000 | 0.206 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.692 ± 0.009 | 0.035 ± 0.000 | 0.277 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.702 ± 0.002 | 0.036 ± 0.000 | 0.275 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.717 ± 0.017 | 0.037 ± 0.000 | 0.276 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.192 ± 0.001 | 0.034 ± 0.000 | 0.256 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.194 ± 0.001 | 0.035 ± 0.000 | 0.258 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.198 ± 0.001 | 0.036 ± 0.000 | 0.258 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.977 ± 0.025 | 0.061 ± 0.000 | 0.218 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.952 ± 0.003 | 0.064 ± 0.000 | 0.217 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.971 ± 0.004 | 0.066 ± 0.000 | 0.219 ± 0.004 | 0.017 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.183 ± 0.000 | 0.065 ± 0.001 | 0.249 ± 0.003 | 0.013 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.185 ± 0.002 | 0.067 ± 0.000 | 0.247 ± 0.006 | 0.016 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.191 ± 0.002 | 0.070 ± 0.000 | 0.249 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.190 ± 0.001 | 0.026 ± 0.000 | 0.414 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.189 ± 0.002 | 0.027 ± 0.001 | 0.412 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.194 ± 0.003 | 0.028 ± 0.000 | 0.416 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.098 ± 0.001 | 0.026 ± 0.000 | 0.247 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.098 ± 0.000 | 0.027 ± 0.000 | 0.247 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.100 ± 0.002 | 0.028 ± 0.000 | 0.246 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.226 ± 0.002 | 0.024 ± 0.000 | 0.262 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.242 ± 0.002 | 0.033 ± 0.000 | 0.269 ± 0.006 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.264 ± 0.001 | 0.046 ± 0.001 | 0.283 ± 0.004 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.094 ± 0.004 | 0.024 ± 0.000 | 0.214 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.102 ± 0.001 | 0.034 ± 0.000 | 0.219 ± 0.004 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.113 ± 0.001 | 0.048 ± 0.001 | 0.230 ± 0.003 | 0.019 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261107_184743/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261107_184743/run.yaml`
- Timestamp: `2026-11-07T18:47:43.753266+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261107_184743`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261107_184743/cpu_ops_t4_20261107_184743.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 29.795 ± 0.130 | 26.382 ± 0.021 | 7.036 ± 0.278 | 11.207 ± 0.362 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 153.264 ± 0.180 | 117.613 ± 0.393 | 42.313 ± 0.975 | 49.628 ± 4.471 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 16.071 ± 0.066 | 26.326 ± 0.027 | 5.950 ± 0.022 | 11.230 ± 1.847 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 83.118 ± 0.543 | 116.655 ± 0.262 | 30.619 ± 1.416 | 56.790 ± 2.777 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 16.026 ± 0.067 | 9.281 ± 0.013 | 8.924 ± 0.123 | 5.989 ± 1.044 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 308.599 ± 50.791 | 301.506 ± 25.584 | 48.845 ± 1.114 | 34.737 ± 2.270 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 14.154 ± 0.041 | 9.793 ± 0.034 | 3.597 ± 0.021 | 4.211 ± 1.728 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 212.760 ± 4.408 | 303.113 ± 11.622 | 23.987 ± 0.295 | 28.543 ± 1.964 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 169.000 ± 11.984 | 15.471 ± 11.404 | 4.785 ± 0.066 | 8.311 ± 1.995 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 440.201 ± 11.231 | 392.102 ± 9.753 | 33.408 ± 0.153 | 45.500 ± 10.112 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 17.175 ± 0.060 | 15.254 ± 11.360 | 5.304 ± 0.092 | 9.942 ± 1.182 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 256.507 ± 11.967 | 390.795 ± 11.919 | 33.507 ± 0.484 | 54.352 ± 10.593 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 3.051 ± 0.029 | 2.612 ± 0.028 | 2.463 ± 1.455 | 0.882 ± 0.060 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 290.403 ± 18.028 | 278.160 ± 26.905 | 2.959 ± 0.081 | 3.819 ± 0.120 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.488 ± 0.017 | 2.600 ± 0.029 | 1.401 ± 0.033 | 0.952 ± 0.119 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 136.578 ± 5.699 | 290.004 ± 15.973 | 4.762 ± 0.049 | 3.676 ± 0.045 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 35.025 ± 0.562 | 24.657 ± 0.081 | 16.322 ± 0.030 | 19.113 ± 0.984 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 234.185 ± 2.850 | 176.835 ± 0.349 | 93.543 ± 1.681 | 107.864 ± 4.411 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 20.009 ± 0.138 | 26.228 ± 0.086 | 14.719 ± 0.035 | 16.447 ± 1.436 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 124.764 ± 0.936 | 177.059 ± 0.170 | 82.934 ± 0.419 | 111.894 ± 7.860 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.266 ± 0.002 | 0.073 ± 0.002 | 0.198 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.281 ± 0.003 | 0.055 ± 0.001 | 0.202 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.291 ± 0.010 | 0.062 ± 0.001 | 0.207 ± 0.003 | 0.017 ± 0.003 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.113 ± 0.001 | 0.074 ± 0.001 | 0.193 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.122 ± 0.001 | 0.057 ± 0.000 | 0.200 ± 0.006 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.129 ± 0.002 | 0.065 ± 0.001 | 0.205 ± 0.006 | 0.020 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.722 ± 0.014 | 0.076 ± 0.000 | 0.385 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.733 ± 0.005 | 0.049 ± 0.000 | 0.381 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.731 ± 0.003 | 0.049 ± 0.000 | 0.384 ± 0.005 | 0.020 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.227 ± 0.003 | 0.072 ± 0.001 | 0.336 ± 0.008 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.231 ± 0.004 | 0.047 ± 0.001 | 0.333 ± 0.006 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.235 ± 0.004 | 0.048 ± 0.000 | 0.335 ± 0.009 | 0.018 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.988 ± 0.006 | 0.120 ± 0.001 | 0.289 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.994 ± 0.016 | 0.080 ± 0.001 | 0.285 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.994 ± 0.005 | 0.083 ± 0.001 | 0.290 ± 0.005 | 0.022 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.215 ± 0.001 | 0.123 ± 0.001 | 0.322 ± 0.057 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.219 ± 0.005 | 0.083 ± 0.000 | 0.315 ± 0.008 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.223 ± 0.002 | 0.087 ± 0.001 | 0.320 ± 0.007 | 0.019 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.199 ± 0.001 | 0.055 ± 0.000 | 0.405 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.201 ± 0.001 | 0.035 ± 0.000 | 0.404 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.204 ± 0.002 | 0.037 ± 0.000 | 0.413 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.109 ± 0.001 | 0.056 ± 0.001 | 0.244 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.109 ± 0.001 | 0.035 ± 0.000 | 0.245 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.112 ± 0.002 | 0.036 ± 0.000 | 0.249 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.234 ± 0.030 | 0.049 ± 0.000 | 0.267 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.249 ± 0.001 | 0.040 ± 0.000 | 0.273 ± 0.004 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.274 ± 0.002 | 0.054 ± 0.000 | 0.286 ± 0.004 | 0.026 ± 0.006 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.103 ± 0.001 | 0.047 ± 0.000 | 0.214 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.111 ± 0.002 | 0.039 ± 0.000 | 0.221 ± 0.003 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.124 ± 0.003 | 0.054 ± 0.000 | 0.233 ± 0.002 | 0.032 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `89c207f1ca97cde3ae0f31586747a039ca4e5cc6`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261114_180711/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261114_180711/run.yaml`
- Timestamp: `2026-11-14T18:07:11.056991+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261114_180711`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261114_180711/cpu_ops_t1_20261114_180711.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

Both tenferro arms return the same requested outputs -- the primal and the
declared derivative -- with fixtures and wrapping outside the clock. They are
not the same execution: the eager sample runs the forward pass in a borrowed
session and then the derivative call, which carries the eager API's own AD
preparation on every call (its VJP reuses a prepared derivative when the cache
holds one, its JVP re-prepares), because that API exposes no separate prepare
step. The trace arm compiles and prepares the program before sampling. So the
trace number is a prepared-program steady state and the eager number is the
per-call cost of the public API, not a like-for-like difference in one
operation.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager per call, incl. AD prepare (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 17.127 ± 0.787 | 11.845 ± 2.921 | 10.356 ± 0.151 | 8.987 ± 0.047 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 107.199 ± 1.055 | 76.344 ± 0.602 | 66.242 ± 1.346 | 57.631 ± 0.734 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 9.466 ± 0.129 | 11.940 ± 0.077 | 8.579 ± 0.111 | 9.101 ± 0.135 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 61.246 ± 0.233 | 76.535 ± 0.375 | 52.186 ± 0.140 | 57.071 ± 0.221 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 11.186 ± 0.221 | 5.999 ± 0.465 | 13.256 ± 1.884 | 4.426 ± 0.054 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 74.153 ± 1.003 | 40.502 ± 2.674 | 79.867 ± 2.835 | 31.483 ± 0.355 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 10.873 ± 0.173 | 6.191 ± 0.286 | 5.435 ± 0.034 | 4.280 ± 0.074 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 71.148 ± 0.775 | 39.421 ± 0.932 | 42.685 ± 0.398 | 33.251 ± 1.636 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 16.689 ± 0.405 | 13.636 ± 1.032 | 7.343 ± 0.093 | 7.202 ± 0.035 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 116.768 ± 1.190 | 74.113 ± 0.360 | 54.127 ± 2.587 | 45.625 ± 0.456 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 13.818 ± 0.162 | 12.064 ± 3.020 | 7.579 ± 0.172 | 7.221 ± 0.022 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 91.411 ± 1.162 | 70.326 ± 1.149 | 51.235 ± 0.523 | 51.551 ± 0.597 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.977 ± 0.031 | 1.406 ± 0.012 | 1.263 ± 0.025 | 0.681 ± 0.003 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 8.679 ± 0.058 | 8.099 ± 0.050 | 5.689 ± 0.162 | 4.133 ± 0.021 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.877 ± 0.026 | 1.401 ± 0.013 | 1.760 ± 0.022 | 0.687 ± 0.013 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.395 ± 0.046 | 8.118 ± 0.057 | 9.765 ± 0.275 | 4.053 ± 0.047 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 28.817 ± 0.393 | 22.021 ± 3.863 | 18.802 ± 0.249 | 15.606 ± 0.039 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 194.846 ± 2.068 | 140.351 ± 1.201 | 134.492 ± 0.703 | 109.251 ± 0.888 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 14.973 ± 0.037 | 25.664 ± 0.073 | 15.881 ± 0.208 | 16.079 ± 0.375 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 104.364 ± 1.115 | 140.735 ± 0.925 | 108.083 ± 1.363 | 115.752 ± 0.909 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.251 ± 0.002 | 0.036 ± 0.000 | 0.193 ± 0.003 | 0.012 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.264 ± 0.002 | 0.044 ± 0.001 | 0.201 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.276 ± 0.004 | 0.054 ± 0.029 | 0.206 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.097 ± 0.001 | 0.037 ± 0.000 | 0.193 ± 0.003 | 0.013 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.106 ± 0.001 | 0.045 ± 0.000 | 0.199 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.113 ± 0.001 | 0.051 ± 0.001 | 0.205 ± 0.006 | 0.014 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.704 ± 0.014 | 0.036 ± 0.000 | 0.275 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.705 ± 0.019 | 0.036 ± 0.000 | 0.273 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.715 ± 0.005 | 0.038 ± 0.001 | 0.275 ± 0.003 | 0.015 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.188 ± 0.003 | 0.034 ± 0.000 | 0.257 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.190 ± 0.003 | 0.034 ± 0.000 | 0.259 ± 0.006 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.193 ± 0.003 | 0.036 ± 0.001 | 0.259 ± 0.002 | 0.016 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.951 ± 0.005 | 0.060 ± 0.000 | 0.219 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.963 ± 0.013 | 0.062 ± 0.001 | 0.217 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.974 ± 0.011 | 0.065 ± 0.000 | 0.220 ± 0.005 | 0.018 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.180 ± 0.003 | 0.063 ± 0.000 | 0.247 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.184 ± 0.004 | 0.065 ± 0.001 | 0.247 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.189 ± 0.003 | 0.069 ± 0.001 | 0.250 ± 0.002 | 0.015 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.190 ± 0.001 | 0.027 ± 0.000 | 0.408 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.187 ± 0.001 | 0.027 ± 0.000 | 0.413 ± 0.009 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.191 ± 0.002 | 0.029 ± 0.000 | 0.413 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.095 ± 0.001 | 0.027 ± 0.001 | 0.244 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.095 ± 0.003 | 0.027 ± 0.000 | 0.247 ± 0.006 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.098 ± 0.001 | 0.029 ± 0.000 | 0.247 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.226 ± 0.005 | 0.024 ± 0.000 | 0.264 ± 0.009 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.241 ± 0.004 | 0.032 ± 0.000 | 0.269 ± 0.003 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.263 ± 0.006 | 0.047 ± 0.001 | 0.288 ± 0.008 | 0.035 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.091 ± 0.001 | 0.023 ± 0.000 | 0.214 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.101 ± 0.001 | 0.031 ± 0.000 | 0.217 ± 0.005 | 0.018 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.111 ± 0.001 | 0.046 ± 0.001 | 0.233 ± 0.007 | 0.032 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261114_184911/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261114_184911/run.yaml`
- Timestamp: `2026-11-14T18:49:11.517432+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261114_184911`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261114_184911/cpu_ops_t4_20261114_184911.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

Both tenferro arms return the same requested outputs -- the primal and the
declared derivative -- with fixtures and wrapping outside the clock. They are
not the same execution: the eager sample runs the forward pass in a borrowed
session and then the derivative call, which carries the eager API's own AD
preparation on every call (its VJP reuses a prepared derivative when the cache
holds one, its JVP re-prepares), because that API exposes no separate prepare
step. The trace arm compiles and prepares the program before sampling. So the
trace number is a prepared-program steady state and the eager number is the
per-call cost of the public API, not a like-for-like difference in one
operation.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager per call, incl. AD prepare (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 31.773 ± 0.241 | 26.821 ± 0.294 | 7.112 ± 0.216 | 11.535 ± 0.683 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 155.601 ± 2.818 | 119.053 ± 2.946 | 42.710 ± 0.758 | 53.710 ± 4.838 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 18.531 ± 0.303 | 26.595 ± 0.312 | 6.046 ± 0.196 | 11.815 ± 1.006 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 85.923 ± 1.379 | 123.316 ± 2.349 | 30.830 ± 1.080 | 53.533 ± 6.312 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 17.589 ± 0.643 | 9.494 ± 0.149 | 9.114 ± 0.262 | 5.674 ± 0.918 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 345.799 ± 39.972 | 333.645 ± 29.557 | 49.844 ± 2.089 | 31.731 ± 4.779 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 14.504 ± 0.466 | 9.868 ± 0.098 | 3.598 ± 0.024 | 4.112 ± 1.766 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 204.153 ± 22.216 | 326.375 ± 20.548 | 24.531 ± 0.683 | 35.428 ± 1.932 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 132.335 ± 70.473 | 27.024 ± 47.567 | 5.002 ± 0.428 | 10.144 ± 1.503 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 444.790 ± 19.087 | 397.233 ± 15.369 | 33.800 ± 0.736 | 45.655 ± 10.144 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 18.586 ± 83.538 | 16.634 ± 11.614 | 5.255 ± 0.065 | 9.230 ± 2.167 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 257.969 ± 13.261 | 392.101 ± 13.048 | 33.315 ± 1.053 | 48.493 ± 9.419 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 3.036 ± 0.018 | 2.605 ± 0.021 | 1.113 ± 14.835 | 0.911 ± 0.066 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 285.049 ± 15.139 | 337.957 ± 5.932 | 2.976 ± 0.027 | 3.738 ± 0.069 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.535 ± 0.065 | 2.596 ± 0.019 | 1.235 ± 0.051 | 0.968 ± 0.090 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 128.159 ± 9.004 | 313.992 ± 10.797 | 4.753 ± 0.064 | 3.721 ± 0.050 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 36.026 ± 0.859 | 24.976 ± 0.234 | 16.676 ± 0.318 | 18.368 ± 0.808 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 237.270 ± 1.267 | 175.595 ± 0.987 | 93.540 ± 4.346 | 98.969 ± 13.975 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 20.288 ± 0.183 | 26.954 ± 0.376 | 14.761 ± 0.141 | 18.831 ± 1.092 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 124.529 ± 0.436 | 177.022 ± 1.795 | 83.162 ± 16.526 | 104.744 ± 12.575 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.267 ± 0.003 | 0.044 ± 0.000 | 0.198 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.283 ± 0.007 | 0.052 ± 0.001 | 0.202 ± 0.002 | 0.016 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.291 ± 0.001 | 0.059 ± 0.001 | 0.207 ± 0.004 | 0.018 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.111 ± 0.002 | 0.046 ± 0.000 | 0.195 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.123 ± 0.003 | 0.053 ± 0.001 | 0.200 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.125 ± 0.001 | 0.060 ± 0.001 | 0.205 ± 0.004 | 0.020 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.746 ± 0.013 | 0.045 ± 0.001 | 0.383 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.741 ± 0.009 | 0.046 ± 0.001 | 0.380 ± 0.012 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.749 ± 0.017 | 0.047 ± 0.000 | 0.381 ± 0.002 | 0.021 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.223 ± 0.001 | 0.042 ± 0.000 | 0.332 ± 0.006 | 0.014 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.228 ± 0.004 | 0.043 ± 0.000 | 0.336 ± 0.305 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.228 ± 0.002 | 0.044 ± 0.001 | 0.335 ± 0.009 | 0.019 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.996 ± 0.008 | 0.075 ± 0.001 | 0.286 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 1.010 ± 0.006 | 0.078 ± 0.001 | 0.285 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 1.021 ± 0.009 | 0.081 ± 0.000 | 0.289 ± 0.003 | 0.022 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.212 ± 0.003 | 0.077 ± 0.000 | 0.315 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.218 ± 0.002 | 0.080 ± 0.001 | 0.316 ± 0.005 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.221 ± 0.004 | 0.084 ± 0.001 | 0.319 ± 0.006 | 0.019 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.201 ± 0.003 | 0.034 ± 0.001 | 0.410 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.202 ± 0.005 | 0.034 ± 0.000 | 0.413 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.204 ± 0.001 | 0.036 ± 0.000 | 0.412 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.110 ± 0.004 | 0.033 ± 0.000 | 0.242 ± 0.008 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.108 ± 0.001 | 0.034 ± 0.000 | 0.246 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.109 ± 0.001 | 0.035 ± 0.000 | 0.248 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.236 ± 0.008 | 0.028 ± 0.000 | 0.267 ± 0.002 | 0.014 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.250 ± 0.001 | 0.037 ± 0.000 | 0.271 ± 0.001 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.275 ± 0.003 | 0.051 ± 0.000 | 0.285 ± 0.005 | 0.032 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.102 ± 0.018 | 0.026 ± 0.000 | 0.212 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.109 ± 0.001 | 0.035 ± 0.000 | 0.220 ± 0.004 | 0.020 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.123 ± 0.002 | 0.051 ± 0.001 | 0.228 ± 0.004 | 0.033 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

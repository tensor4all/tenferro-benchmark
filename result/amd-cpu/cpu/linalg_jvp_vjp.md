# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `74775e4a2ad3480fb9ee9f859e0037b8cac731f2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261125_175248/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261125_175248/run.yaml`
- Timestamp: `2026-11-25T17:52:48.228690+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261125_175248`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
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

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/amd-cpu/cpu/einsum/20261125_175248/cpu_ops_t1_20261125_175248.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 16.347 ± 0.090 | 12.097 ± 0.109 | 10.420 ± 0.181 | 8.889 ± 0.061 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 89.850 ± 0.693 | 66.374 ± 0.191 | 66.581 ± 0.504 | 58.320 ± 0.458 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 8.966 ± 0.060 | 12.005 ± 0.075 | 8.538 ± 0.108 | 8.993 ± 0.032 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 55.056 ± 0.819 | 65.933 ± 0.628 | 52.279 ± 0.440 | 58.115 ± 0.221 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 9.720 ± 0.086 | 4.784 ± 0.022 | 12.431 ± 0.315 | 4.345 ± 0.090 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 71.865 ± 1.130 | 34.295 ± 0.202 | 79.900 ± 0.646 | 31.439 ± 1.514 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 8.935 ± 0.192 | 5.056 ± 0.057 | 5.437 ± 0.099 | 4.402 ± 0.049 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 68.942 ± 0.390 | 39.059 ± 0.484 | 42.656 ± 0.701 | 32.865 ± 2.201 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 14.605 ± 0.660 | 8.227 ± 0.092 | 7.425 ± 0.071 | 7.172 ± 0.087 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 106.820 ± 1.519 | 60.035 ± 0.355 | 52.373 ± 0.800 | 45.843 ± 2.429 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 11.202 ± 0.112 | 8.235 ± 0.026 | 7.389 ± 0.207 | 7.185 ± 0.044 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 92.257 ± 2.636 | 59.761 ± 1.799 | 52.075 ± 0.523 | 50.766 ± 3.406 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.324 ± 0.020 | 1.096 ± 0.593 | 1.235 ± 0.020 | 0.677 ± 0.019 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 6.635 ± 0.107 | 6.411 ± 0.074 | 5.529 ± 0.093 | 4.165 ± 0.057 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.725 ± 0.012 | 1.121 ± 0.005 | 1.721 ± 0.031 | 0.658 ± 0.004 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 3.557 ± 0.692 | 6.378 ± 0.047 | 9.634 ± 0.697 | 4.050 ± 0.029 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 27.396 ± 0.237 | 20.393 ± 0.100 | 21.055 ± 0.269 | 15.639 ± 0.126 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 179.224 ± 0.459 | 132.675 ± 1.156 | 134.890 ± 0.163 | 110.251 ± 0.642 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 14.666 ± 0.318 | 20.287 ± 0.267 | 16.049 ± 0.098 | 15.603 ± 0.055 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 94.141 ± 0.397 | 130.002 ± 0.649 | 107.698 ± 0.623 | 116.704 ± 1.164 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.249 ± 0.003 | 0.033 ± 0.000 | 0.195 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.261 ± 0.003 | 0.039 ± 0.000 | 0.198 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.270 ± 0.002 | 0.048 ± 0.000 | 0.206 ± 0.008 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.095 ± 0.002 | 0.034 ± 0.000 | 0.192 ± 0.004 | 0.012 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.101 ± 0.002 | 0.040 ± 0.000 | 0.198 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.108 ± 0.002 | 0.049 ± 0.001 | 0.205 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.685 ± 0.008 | 0.033 ± 0.000 | 0.274 ± 0.006 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.696 ± 0.009 | 0.034 ± 0.000 | 0.271 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.692 ± 0.005 | 0.037 ± 0.001 | 0.276 ± 0.006 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.178 ± 0.006 | 0.031 ± 0.000 | 0.255 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.178 ± 0.004 | 0.032 ± 0.002 | 0.258 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.181 ± 0.004 | 0.035 ± 0.000 | 0.259 ± 0.006 | 0.015 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.797 ± 0.006 | 0.052 ± 0.001 | 0.216 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.815 ± 0.007 | 0.053 ± 0.001 | 0.216 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.818 ± 0.012 | 0.058 ± 0.011 | 0.221 ± 0.003 | 0.017 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.171 ± 0.002 | 0.056 ± 0.000 | 0.246 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.171 ± 0.003 | 0.057 ± 0.000 | 0.245 ± 0.007 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.177 ± 0.003 | 0.063 ± 0.001 | 0.247 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.187 ± 0.002 | 0.028 ± 0.000 | 0.408 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.190 ± 0.001 | 0.028 ± 0.000 | 0.414 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.193 ± 0.002 | 0.030 ± 0.000 | 0.410 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.100 ± 0.002 | 0.028 ± 0.000 | 0.244 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.101 ± 0.002 | 0.029 ± 0.000 | 0.244 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.105 ± 0.001 | 0.030 ± 0.000 | 0.245 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.222 ± 0.004 | 0.021 ± 0.000 | 0.261 ± 0.009 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.228 ± 0.002 | 0.025 ± 0.000 | 0.268 ± 0.001 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.244 ± 0.003 | 0.036 ± 0.000 | 0.282 ± 0.005 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.087 ± 0.001 | 0.020 ± 0.000 | 0.212 ± 0.009 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.091 ± 0.003 | 0.024 ± 0.001 | 0.218 ± 0.003 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.100 ± 0.002 | 0.036 ± 0.000 | 0.228 ± 0.005 | 0.019 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261125_183303/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261125_183303/run.yaml`
- Timestamp: `2026-11-25T18:33:03.871566+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261125_183303`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
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

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/amd-cpu/cpu/einsum/20261125_183303/cpu_ops_t4_20261125_183303.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 18.777 ± 0.170 | 14.716 ± 0.267 | 7.158 ± 5.914 | 10.984 ± 1.008 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 89.712 ± 0.481 | 64.909 ± 0.632 | 41.898 ± 1.089 | 49.952 ± 2.975 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 10.990 ± 0.107 | 14.799 ± 0.036 | 6.059 ± 0.280 | 10.994 ± 1.414 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 49.525 ± 0.968 | 64.863 ± 0.928 | 30.238 ± 1.429 | 57.418 ± 4.442 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 7.511 ± 0.063 | 3.921 ± 0.026 | 7.370 ± 0.191 | 5.721 ± 0.812 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 40.193 ± 0.133 | 21.322 ± 0.041 | 47.786 ± 0.621 | 31.654 ± 4.598 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 6.097 ± 0.069 | 3.972 ± 0.011 | 3.580 ± 0.039 | 6.187 ± 0.501 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 36.908 ± 0.066 | 23.524 ± 0.383 | 24.505 ± 0.238 | 36.024 ± 3.821 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 13.371 ± 0.069 | 9.341 ± 0.176 | 4.832 ± 0.255 | 8.988 ± 1.928 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 58.522 ± 1.138 | 33.434 ± 0.398 | 33.477 ± 0.430 | 47.860 ± 2.041 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 9.067 ± 0.060 | 9.317 ± 0.243 | 5.388 ± 1.865 | 8.967 ± 1.217 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 48.256 ± 1.060 | 33.787 ± 0.219 | 32.802 ± 0.322 | 55.978 ± 7.049 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.481 ± 0.009 | 1.251 ± 0.034 | 2.932 ± 3.892 | 0.902 ± 0.146 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 5.388 ± 0.099 | 5.003 ± 0.107 | 3.207 ± 0.570 | 3.793 ± 0.085 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.870 ± 0.024 | 1.256 ± 0.024 | 1.242 ± 0.023 | 0.987 ± 0.058 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 2.954 ± 0.093 | 4.980 ± 0.007 | 4.784 ± 0.060 | 3.732 ± 0.038 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 29.287 ± 0.216 | 22.302 ± 0.114 | 16.444 ± 0.085 | 17.360 ± 0.922 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 175.841 ± 0.811 | 126.839 ± 0.580 | 94.623 ± 1.097 | 107.797 ± 10.801 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 16.338 ± 0.096 | 22.198 ± 0.096 | 14.849 ± 0.300 | 17.647 ± 0.626 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 90.115 ± 0.977 | 126.274 ± 1.515 | 83.882 ± 0.483 | 105.779 ± 10.787 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.258 ± 0.004 | 0.043 ± 0.000 | 0.197 ± 0.024 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.271 ± 0.029 | 0.048 ± 0.000 | 0.196 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.284 ± 0.001 | 0.058 ± 0.001 | 0.204 ± 0.005 | 0.018 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.103 ± 0.001 | 0.043 ± 0.000 | 0.189 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.113 ± 0.002 | 0.048 ± 0.001 | 0.196 ± 0.033 | 0.016 ± 0.004 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.114 ± 0.001 | 0.058 ± 0.001 | 0.200 ± 0.003 | 0.019 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.716 ± 0.003 | 0.044 ± 0.001 | 0.371 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.746 ± 0.020 | 0.045 ± 0.001 | 0.371 ± 0.010 | 0.014 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.733 ± 0.011 | 0.047 ± 0.000 | 0.374 ± 0.004 | 0.020 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.203 ± 0.052 | 0.041 ± 0.000 | 0.328 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.202 ± 0.002 | 0.042 ± 0.000 | 0.327 ± 0.008 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.206 ± 0.001 | 0.045 ± 0.001 | 0.330 ± 0.010 | 0.021 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.838 ± 0.009 | 0.067 ± 0.001 | 0.281 ± 0.006 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.851 ± 0.006 | 0.068 ± 0.000 | 0.279 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.863 ± 0.005 | 0.074 ± 0.001 | 0.283 ± 0.004 | 0.021 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.196 ± 0.001 | 0.070 ± 0.000 | 0.310 ± 0.003 | 0.013 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.197 ± 0.002 | 0.071 ± 0.001 | 0.310 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.205 ± 0.003 | 0.076 ± 0.001 | 0.315 ± 0.010 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.201 ± 0.006 | 0.036 ± 0.000 | 0.398 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.198 ± 0.002 | 0.036 ± 0.000 | 0.403 ± 0.009 | 0.013 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.201 ± 0.001 | 0.039 ± 0.001 | 0.400 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.108 ± 0.003 | 0.036 ± 0.000 | 0.238 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.107 ± 0.000 | 0.037 ± 0.000 | 0.239 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.109 ± 0.000 | 0.039 ± 0.000 | 0.242 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.223 ± 0.001 | 0.027 ± 0.000 | 0.258 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.232 ± 0.004 | 0.032 ± 0.000 | 0.263 ± 0.006 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.249 ± 0.002 | 0.044 ± 0.006 | 0.277 ± 0.005 | 0.032 ± 0.007 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.092 ± 0.001 | 0.026 ± 0.000 | 0.208 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.097 ± 0.003 | 0.031 ± 0.000 | 0.215 ± 0.002 | 0.020 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.105 ± 0.001 | 0.042 ± 0.000 | 0.226 ± 0.006 | 0.032 ± 0.006 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

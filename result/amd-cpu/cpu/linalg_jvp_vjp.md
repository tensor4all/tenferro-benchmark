# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `4edf89377c13d5c8a287dd562b529de6bb85f4dc`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261112_180742/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261112_180742/run.yaml`
- Timestamp: `2026-11-12T18:07:42.528102+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261112_180742`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261112_180742/cpu_ops_t1_20261112_180742.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 17.264 ± 0.121 | 11.821 ± 0.103 | 10.517 ± 0.204 | 9.074 ± 0.109 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 107.927 ± 0.579 | 75.720 ± 0.313 | 66.070 ± 0.998 | 57.411 ± 0.350 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 9.658 ± 0.098 | 11.999 ± 0.020 | 8.411 ± 0.036 | 8.923 ± 0.105 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 61.910 ± 0.853 | 75.846 ± 0.369 | 51.991 ± 0.262 | 57.410 ± 0.521 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 11.216 ± 0.049 | 5.688 ± 0.036 | 12.214 ± 0.111 | 4.539 ± 0.097 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 76.224 ± 0.698 | 37.939 ± 0.349 | 79.313 ± 0.731 | 31.489 ± 0.747 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 10.259 ± 0.086 | 5.925 ± 0.076 | 5.454 ± 0.048 | 4.475 ± 0.079 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 71.115 ± 0.857 | 40.545 ± 1.050 | 41.979 ± 0.299 | 33.599 ± 0.374 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 16.951 ± 0.092 | 10.225 ± 0.121 | 7.281 ± 0.010 | 7.118 ± 0.059 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 118.205 ± 0.490 | 73.784 ± 0.379 | 51.427 ± 0.629 | 46.973 ± 4.012 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 12.945 ± 0.035 | 10.037 ± 0.043 | 7.394 ± 0.086 | 7.221 ± 0.097 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 92.403 ± 0.539 | 69.937 ± 0.609 | 51.433 ± 0.113 | 51.354 ± 0.824 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.943 ± 0.041 | 1.395 ± 0.005 | 1.244 ± 0.013 | 0.677 ± 0.009 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 8.687 ± 0.225 | 8.317 ± 0.161 | 5.591 ± 0.119 | 4.368 ± 0.074 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.946 ± 0.014 | 1.391 ± 0.010 | 1.711 ± 0.024 | 0.685 ± 0.022 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.555 ± 0.106 | 8.083 ± 0.430 | 9.389 ± 0.072 | 4.309 ± 0.135 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 29.330 ± 0.377 | 20.253 ± 0.258 | 18.689 ± 0.248 | 15.561 ± 0.171 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 196.569 ± 2.254 | 141.493 ± 0.991 | 134.565 ± 0.692 | 109.069 ± 0.693 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 15.124 ± 0.040 | 20.146 ± 0.068 | 16.006 ± 0.344 | 15.424 ± 0.098 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 104.005 ± 0.232 | 141.356 ± 0.917 | 107.162 ± 0.374 | 116.059 ± 0.816 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.254 ± 0.003 | 0.037 ± 0.001 | 0.195 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.267 ± 0.002 | 0.046 ± 0.001 | 0.200 ± 0.003 | 0.015 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.281 ± 0.008 | 0.052 ± 0.000 | 0.206 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.100 ± 0.002 | 0.039 ± 0.012 | 0.197 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.109 ± 0.001 | 0.048 ± 0.000 | 0.203 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.116 ± 0.001 | 0.053 ± 0.001 | 0.207 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.694 ± 0.007 | 0.036 ± 0.000 | 0.276 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.696 ± 0.013 | 0.037 ± 0.000 | 0.275 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.711 ± 0.007 | 0.038 ± 0.001 | 0.280 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.192 ± 0.003 | 0.034 ± 0.000 | 0.259 ± 0.005 | 0.013 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.194 ± 0.007 | 0.035 ± 0.000 | 0.263 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.199 ± 0.003 | 0.037 ± 0.001 | 0.263 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.963 ± 0.008 | 0.062 ± 0.002 | 0.218 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.970 ± 0.025 | 0.064 ± 0.001 | 0.219 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.982 ± 0.022 | 0.069 ± 0.001 | 0.222 ± 0.006 | 0.018 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.186 ± 0.002 | 0.066 ± 0.001 | 0.248 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.187 ± 0.004 | 0.068 ± 0.001 | 0.247 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.197 ± 0.021 | 0.073 ± 0.001 | 0.251 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.189 ± 0.002 | 0.027 ± 0.001 | 0.410 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.189 ± 0.002 | 0.026 ± 0.000 | 0.415 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.194 ± 0.001 | 0.028 ± 0.000 | 0.422 ± 0.009 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.097 ± 0.001 | 0.026 ± 0.001 | 0.246 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.098 ± 0.002 | 0.027 ± 0.000 | 0.249 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.099 ± 0.003 | 0.028 ± 0.001 | 0.248 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.225 ± 0.001 | 0.025 ± 0.000 | 0.264 ± 0.008 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.242 ± 0.001 | 0.034 ± 0.000 | 0.272 ± 0.003 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.274 ± 0.014 | 0.048 ± 0.000 | 0.285 ± 0.003 | 0.020 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.093 ± 0.000 | 0.024 ± 0.000 | 0.215 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.101 ± 0.004 | 0.033 ± 0.000 | 0.220 ± 0.002 | 0.018 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.116 ± 0.001 | 0.048 ± 0.001 | 0.234 ± 0.011 | 0.020 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261112_184910/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261112_184910/run.yaml`
- Timestamp: `2026-11-12T18:49:10.227921+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261112_184910`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261112_184910/cpu_ops_t4_20261112_184910.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 32.646 ± 0.267 | 27.017 ± 0.077 | 7.320 ± 0.195 | 9.710 ± 1.105 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 155.361 ± 1.112 | 119.680 ± 0.533 | 42.914 ± 1.248 | 57.580 ± 8.998 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 19.330 ± 0.112 | 26.931 ± 0.099 | 6.187 ± 0.273 | 11.550 ± 0.276 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 85.773 ± 0.476 | 119.826 ± 0.405 | 31.378 ± 6.548 | 57.212 ± 1.436 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 17.053 ± 0.165 | 9.557 ± 0.079 | 9.084 ± 0.351 | 6.099 ± 2.136 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 365.556 ± 12.996 | 303.509 ± 23.105 | 51.046 ± 2.310 | 31.601 ± 6.483 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 14.915 ± 0.205 | 10.082 ± 0.059 | 3.704 ± 0.090 | 5.250 ± 1.037 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 175.354 ± 23.991 | 313.509 ± 20.882 | 24.803 ± 0.429 | 29.596 ± 8.143 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 220.525 ± 44.640 | 15.897 ± 0.071 | 4.986 ± 0.070 | 10.242 ± 1.202 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 438.899 ± 14.927 | 396.137 ± 4.121 | 34.224 ± 0.377 | 45.298 ± 6.720 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 80.913 ± 8.868 | 30.339 ± 38.504 | 5.454 ± 0.075 | 10.734 ± 0.897 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 260.098 ± 7.717 | 383.429 ± 8.516 | 34.450 ± 1.242 | 55.114 ± 8.393 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 3.203 ± 0.072 | 2.650 ± 0.012 | 1.069 ± 0.146 | 0.891 ± 0.219 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 285.298 ± 3.033 | 296.967 ± 11.970 | 3.102 ± 0.055 | 3.872 ± 0.048 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.558 ± 0.021 | 2.631 ± 0.013 | 1.283 ± 0.895 | 0.960 ± 0.033 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 137.026 ± 32.458 | 287.992 ± 11.948 | 4.896 ± 0.251 | 3.739 ± 0.041 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 37.073 ± 0.807 | 25.192 ± 0.319 | 17.148 ± 0.806 | 18.640 ± 0.594 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 239.759 ± 1.066 | 179.747 ± 0.885 | 95.775 ± 1.240 | 104.805 ± 10.057 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 20.865 ± 0.088 | 26.674 ± 0.160 | 15.499 ± 0.273 | 18.659 ± 0.267 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 126.971 ± 0.862 | 181.496 ± 0.579 | 84.120 ± 6.385 | 106.752 ± 14.418 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.271 ± 0.002 | 0.046 ± 0.000 | 0.199 ± 0.060 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.283 ± 0.003 | 0.054 ± 0.007 | 0.199 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.296 ± 0.004 | 0.061 ± 0.000 | 0.209 ± 0.004 | 0.019 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.114 ± 0.002 | 0.048 ± 0.000 | 0.197 ± 0.003 | 0.012 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.123 ± 0.006 | 0.056 ± 0.002 | 0.201 ± 0.005 | 0.016 ± 0.004 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.129 ± 0.001 | 0.063 ± 0.001 | 0.209 ± 0.005 | 0.017 ± 0.003 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.743 ± 0.011 | 0.047 ± 0.000 | 1.264 ± 0.191 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.749 ± 0.002 | 0.047 ± 0.000 | 0.382 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.770 ± 0.008 | 0.049 ± 0.001 | 0.390 ± 0.004 | 0.020 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.227 ± 0.001 | 0.044 ± 0.000 | 1.217 ± 0.322 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.230 ± 0.003 | 0.045 ± 0.002 | 0.337 ± 0.008 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.233 ± 0.001 | 0.047 ± 0.000 | 0.342 ± 0.011 | 0.018 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 1.002 ± 0.014 | 0.078 ± 0.001 | 0.833 ± 0.718 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 1.021 ± 0.011 | 0.080 ± 0.001 | 0.284 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 1.024 ± 0.008 | 0.083 ± 0.000 | 0.290 ± 0.005 | 0.022 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.217 ± 0.003 | 0.080 ± 0.000 | 1.013 ± 2.023 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.219 ± 0.003 | 0.082 ± 0.001 | 0.317 ± 0.009 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.225 ± 0.005 | 0.085 ± 0.001 | 0.323 ± 0.008 | 0.019 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.208 ± 0.004 | 0.034 ± 0.000 | 0.411 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.204 ± 0.004 | 0.035 ± 0.001 | 0.417 ± 0.009 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.202 ± 0.002 | 0.036 ± 0.001 | 0.422 ± 0.031 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.121 ± 0.079 | 0.034 ± 0.000 | 0.245 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.110 ± 0.001 | 0.034 ± 0.000 | 0.248 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.112 ± 0.003 | 0.036 ± 0.000 | 0.248 ± 0.005 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.239 ± 0.002 | 0.031 ± 0.000 | 0.269 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.256 ± 0.004 | 0.039 ± 0.000 | 0.271 ± 0.004 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.282 ± 0.006 | 0.054 ± 0.001 | 0.288 ± 0.002 | 0.026 ± 0.006 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.104 ± 0.002 | 0.029 ± 0.000 | 0.224 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.112 ± 0.002 | 0.038 ± 0.001 | 0.221 ± 0.003 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.126 ± 0.005 | 0.052 ± 0.001 | 0.233 ± 0.008 | 0.026 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

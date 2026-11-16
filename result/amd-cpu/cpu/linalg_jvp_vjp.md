# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `5c1d5a382dd88d20901598318730d850d3568ac8`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261116_180206/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261116_180206/run.yaml`
- Timestamp: `2026-11-16T18:02:06.211309+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261116_180206`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Intel(R) Xeon(R) 6975P-C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1016-aws-x86_64-with-glibc2.39`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261116_180206/cpu_ops_t1_20261116_180206.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 8.415 ± 0.073 | 5.859 ± 0.048 | 4.941 ± 0.036 | 6.141 ± 0.069 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 48.488 ± 0.345 | 33.362 ± 0.580 | 29.130 ± 0.112 | 38.017 ± 0.216 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 4.607 ± 0.120 | 5.944 ± 0.086 | 4.146 ± 0.051 | 6.099 ± 0.043 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 27.248 ± 0.256 | 34.819 ± 0.087 | 22.843 ± 0.369 | 37.459 ± 0.334 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 5.257 ± 0.075 | 2.299 ± 0.010 | 4.204 ± 0.124 | 4.175 ± 0.057 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 34.042 ± 0.379 | 15.540 ± 0.118 | 36.366 ± 0.582 | 28.535 ± 0.237 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 4.712 ± 0.028 | 2.352 ± 0.030 | 2.533 ± 0.056 | 4.180 ± 0.047 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 31.973 ± 0.653 | 15.895 ± 0.051 | 19.386 ± 0.236 | 29.854 ± 0.532 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 8.048 ± 0.102 | 4.205 ± 0.054 | 3.575 ± 0.095 | 5.583 ± 0.074 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 50.190 ± 0.430 | 29.398 ± 0.186 | 24.916 ± 0.250 | 34.468 ± 0.848 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 6.198 ± 0.052 | 4.387 ± 0.051 | 3.502 ± 0.023 | 5.545 ± 0.115 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 40.961 ± 0.339 | 29.815 ± 0.115 | 23.448 ± 0.219 | 39.549 ± 0.705 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.100 ± 0.018 | 0.640 ± 0.006 | 0.750 ± 0.025 | 0.545 ± 0.021 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 4.192 ± 0.094 | 3.813 ± 0.070 | 2.987 ± 0.003 | 3.272 ± 0.089 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.493 ± 0.009 | 0.612 ± 0.008 | 0.918 ± 0.037 | 0.550 ± 0.011 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 2.356 ± 0.027 | 3.759 ± 0.009 | 4.708 ± 0.027 | 3.052 ± 0.122 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 15.007 ± 0.297 | 10.352 ± 0.089 | 9.228 ± 0.449 | 10.197 ± 0.153 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 97.115 ± 1.426 | 68.486 ± 0.495 | 61.576 ± 0.942 | 69.446 ± 0.377 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 7.434 ± 0.045 | 10.340 ± 0.089 | 7.576 ± 0.342 | 10.109 ± 0.157 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 50.999 ± 1.153 | 70.111 ± 1.007 | 43.585 ± 0.454 | 72.429 ± 1.599 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.142 ± 0.001 | 0.016 ± 0.000 | 0.084 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.152 ± 0.002 | 0.019 ± 0.000 | 0.089 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.161 ± 0.002 | 0.024 ± 0.000 | 0.094 ± 0.001 | 0.015 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.034 ± 0.001 | 0.015 ± 0.000 | 0.079 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.038 ± 0.001 | 0.018 ± 0.000 | 0.081 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.042 ± 0.001 | 0.023 ± 0.000 | 0.086 ± 0.000 | 0.017 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.547 ± 0.008 | 0.020 ± 0.000 | 0.129 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.569 ± 0.007 | 0.021 ± 0.000 | 0.131 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.584 ± 0.010 | 0.021 ± 0.000 | 0.133 ± 0.002 | 0.018 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.090 ± 0.001 | 0.018 ± 0.000 | 0.111 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.093 ± 0.005 | 0.019 ± 0.000 | 0.113 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.096 ± 0.002 | 0.019 ± 0.000 | 0.115 ± 0.001 | 0.018 ± 0.004 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.781 ± 0.006 | 0.027 ± 0.000 | 0.097 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.814 ± 0.004 | 0.029 ± 0.001 | 0.099 ± 0.006 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.822 ± 0.015 | 0.030 ± 0.000 | 0.101 ± 0.002 | 0.017 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.081 ± 0.001 | 0.027 ± 0.000 | 0.105 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.084 ± 0.000 | 0.028 ± 0.000 | 0.108 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.086 ± 0.001 | 0.030 ± 0.001 | 0.109 ± 0.001 | 0.016 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.093 ± 0.001 | 0.015 ± 0.000 | 0.206 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.095 ± 0.003 | 0.016 ± 0.000 | 0.211 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.098 ± 0.003 | 0.017 ± 0.000 | 0.213 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.037 ± 0.001 | 0.015 ± 0.000 | 0.099 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.037 ± 0.001 | 0.015 ± 0.001 | 0.102 ± 0.000 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.039 ± 0.002 | 0.016 ± 0.001 | 0.103 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.124 ± 0.001 | 0.013 ± 0.000 | 0.120 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.133 ± 0.002 | 0.017 ± 0.000 | 0.126 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.152 ± 0.008 | 0.027 ± 0.000 | 0.137 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.034 ± 0.002 | 0.011 ± 0.000 | 0.085 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.038 ± 0.001 | 0.015 ± 0.000 | 0.090 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.047 ± 0.001 | 0.026 ± 0.000 | 0.100 ± 0.012 | 0.015 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261116_183058/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261116_183058/run.yaml`
- Timestamp: `2026-11-16T18:30:58.322744+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261116_183058`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Intel(R) Xeon(R) 6975P-C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1016-aws-x86_64-with-glibc2.39`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261116_183058/cpu_ops_t4_20261116_183058.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 14.571 ± 0.115 | 10.704 ± 0.187 | 4.141 ± 0.083 | 6.683 ± 1.621 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 64.653 ± 0.526 | 48.554 ± 0.534 | 21.066 ± 0.335 | 33.811 ± 4.892 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 7.660 ± 0.145 | 10.871 ± 0.071 | 3.618 ± 0.059 | 7.973 ± 1.335 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 35.395 ± 0.270 | 48.625 ± 0.309 | 15.519 ± 0.534 | 38.614 ± 0.372 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 437.083 ± 60.434 | 401.935 ± 20.005 | 2.905 ± 0.043 | 4.975 ± 2.003 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 1084.687 ± 19.481 | 1054.399 ± 36.409 | 20.978 ± 0.649 | 27.637 ± 3.430 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 225.680 ± 27.063 | 431.148 ± 12.151 | 2.117 ± 0.039 | 4.991 ± 1.480 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 547.382 ± 29.971 | 1011.971 ± 56.951 | 11.275 ± 0.279 | 30.954 ± 3.676 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 10.194 ± 0.084 | 5.677 ± 0.042 | 2.532 ± 0.119 | 9.968 ± 2.004 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 942.750 ± 26.785 | 893.078 ± 105.707 | 13.721 ± 0.201 | 41.109 ± 11.940 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 7.355 ± 0.037 | 5.741 ± 0.038 | 2.529 ± 0.062 | 7.959 ± 3.692 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 511.777 ± 26.474 | 911.702 ± 55.166 | 13.028 ± 0.313 | 44.362 ± 0.267 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 439.345 ± 21.225 | 431.899 ± 17.883 | 1.680 ± 0.754 | 0.705 ± 0.006 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1015.794 ± 28.866 | 1017.884 ± 48.129 | 2.041 ± 0.086 | 3.038 ± 0.198 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 229.189 ± 6.128 | 417.012 ± 24.154 | 0.957 ± 0.052 | 0.714 ± 0.018 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 485.035 ± 39.221 | 1043.052 ± 21.047 | 2.929 ± 0.038 | 2.957 ± 0.102 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 17.767 ± 0.185 | 11.416 ± 0.038 | 8.211 ± 0.672 | 24.971 ± 4.844 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 114.241 ± 1.034 | 80.280 ± 0.397 | 47.185 ± 1.012 | 75.977 ± 1.997 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 9.042 ± 0.108 | 12.006 ± 0.136 | 7.199 ± 0.150 | 13.954 ± 3.673 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 59.295 ± 1.460 | 81.001 ± 0.407 | 34.960 ± 0.603 | 74.522 ± 7.938 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.147 ± 0.001 | 0.018 ± 0.000 | 0.085 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.154 ± 0.004 | 0.021 ± 0.000 | 0.086 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.161 ± 0.006 | 0.026 ± 0.000 | 0.092 ± 0.002 | 0.014 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.040 ± 0.002 | 0.018 ± 0.000 | 0.078 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.043 ± 0.001 | 0.020 ± 0.000 | 0.079 ± 0.008 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.047 ± 0.001 | 0.025 ± 0.000 | 0.081 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.565 ± 0.006 | 0.023 ± 0.000 | 35.978 ± 35.823 | 0.009 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.571 ± 0.003 | 0.024 ± 0.000 | 0.168 ± 0.004 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.600 ± 0.018 | 0.024 ± 0.000 | 0.170 ± 0.003 | 0.016 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.105 ± 0.002 | 0.021 ± 0.000 | 0.134 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.109 ± 0.004 | 0.021 ± 0.000 | 0.135 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.109 ± 0.001 | 0.022 ± 0.000 | 0.137 ± 0.001 | 0.015 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.797 ± 0.002 | 0.032 ± 0.000 | 0.122 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.833 ± 0.014 | 0.033 ± 0.000 | 0.120 ± 0.016 | 0.010 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.824 ± 0.005 | 0.035 ± 0.000 | 0.124 ± 0.002 | 0.017 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.096 ± 0.002 | 0.031 ± 0.001 | 0.124 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.097 ± 0.003 | 0.032 ± 0.000 | 0.123 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.099 ± 0.002 | 0.034 ± 0.000 | 0.128 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.099 ± 0.003 | 0.018 ± 0.000 | 0.207 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.099 ± 0.002 | 0.018 ± 0.000 | 0.207 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.099 ± 0.001 | 0.019 ± 0.000 | 0.208 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.042 ± 0.001 | 0.017 ± 0.000 | 0.099 ± 0.000 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.042 ± 0.001 | 0.017 ± 0.000 | 0.098 ± 0.000 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.042 ± 0.002 | 0.018 ± 0.000 | 0.098 ± 0.000 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.125 ± 0.001 | 0.014 ± 0.000 | 0.121 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.136 ± 0.002 | 0.018 ± 0.000 | 0.126 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.154 ± 0.002 | 0.027 ± 0.000 | 0.134 ± 0.002 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.037 ± 0.001 | 0.012 ± 0.000 | 0.085 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.041 ± 0.005 | 0.016 ± 0.000 | 0.087 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.049 ± 0.001 | 0.026 ± 0.000 | 0.094 ± 0.001 | 0.016 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

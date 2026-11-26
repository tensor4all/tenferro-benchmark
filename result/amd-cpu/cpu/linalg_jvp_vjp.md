# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `9b03dd718025b3350d93eaab13a8b93dcc66cab2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261126_175240/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261126_175240/run.yaml`
- Timestamp: `2026-11-26T17:52:40.572826+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261126_175240`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261126_175240/cpu_ops_t1_20261126_175240.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 16.280 ± 0.227 | 12.031 ± 0.128 | 10.255 ± 0.162 | 8.995 ± 0.038 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 88.872 ± 0.393 | 65.132 ± 0.406 | 65.601 ± 0.626 | 57.472 ± 0.302 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 9.056 ± 0.023 | 12.299 ± 0.034 | 8.446 ± 0.148 | 8.901 ± 0.071 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 54.203 ± 0.458 | 65.346 ± 0.362 | 51.701 ± 0.462 | 56.722 ± 0.237 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 9.618 ± 0.026 | 4.814 ± 0.018 | 12.255 ± 0.045 | 4.347 ± 0.076 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 70.834 ± 0.262 | 33.747 ± 0.074 | 79.702 ± 1.759 | 31.273 ± 0.738 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 8.812 ± 0.133 | 5.095 ± 0.014 | 5.384 ± 0.180 | 4.380 ± 0.038 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 69.137 ± 0.248 | 38.902 ± 0.643 | 42.388 ± 0.459 | 32.779 ± 0.853 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 14.080 ± 0.198 | 8.211 ± 0.045 | 7.376 ± 0.274 | 7.075 ± 0.068 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 109.374 ± 0.833 | 57.585 ± 0.286 | 51.516 ± 0.767 | 45.341 ± 2.433 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 11.225 ± 0.047 | 8.210 ± 0.063 | 7.263 ± 0.096 | 7.208 ± 0.069 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 91.691 ± 2.703 | 57.603 ± 0.280 | 50.472 ± 0.578 | 52.675 ± 3.036 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.310 ± 0.007 | 1.080 ± 0.008 | 1.237 ± 0.037 | 0.668 ± 0.002 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 6.660 ± 0.031 | 6.356 ± 0.159 | 5.441 ± 0.009 | 4.135 ± 0.060 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.712 ± 0.009 | 1.088 ± 0.003 | 1.730 ± 0.007 | 0.664 ± 0.006 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 3.510 ± 0.108 | 6.282 ± 0.040 | 9.426 ± 0.090 | 4.069 ± 0.077 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 27.086 ± 0.162 | 20.519 ± 0.282 | 20.814 ± 0.100 | 15.558 ± 0.109 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 177.903 ± 0.726 | 130.345 ± 0.488 | 134.894 ± 1.347 | 106.551 ± 0.542 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 14.267 ± 0.076 | 20.165 ± 0.103 | 15.849 ± 0.126 | 15.484 ± 0.075 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 94.215 ± 0.219 | 128.410 ± 0.805 | 107.340 ± 1.243 | 113.245 ± 0.927 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.256 ± 0.004 | 0.035 ± 0.000 | 0.194 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.265 ± 0.002 | 0.041 ± 0.001 | 0.199 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.277 ± 0.003 | 0.050 ± 0.001 | 0.204 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.098 ± 0.002 | 0.036 ± 0.000 | 0.194 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.101 ± 0.001 | 0.041 ± 0.000 | 0.199 ± 0.006 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.107 ± 0.001 | 0.051 ± 0.001 | 0.202 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.696 ± 0.005 | 0.034 ± 0.000 | 0.277 ± 0.006 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.694 ± 0.006 | 0.034 ± 0.000 | 0.275 ± 0.008 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.723 ± 0.003 | 0.037 ± 0.000 | 0.275 ± 0.001 | 0.015 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.179 ± 0.003 | 0.033 ± 0.002 | 0.259 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.178 ± 0.002 | 0.033 ± 0.001 | 0.255 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.181 ± 0.004 | 0.036 ± 0.000 | 0.259 ± 0.007 | 0.015 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.816 ± 0.007 | 0.054 ± 0.001 | 0.220 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.833 ± 0.004 | 0.056 ± 0.001 | 0.218 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.841 ± 0.008 | 0.060 ± 0.001 | 0.221 ± 0.004 | 0.018 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.177 ± 0.005 | 0.058 ± 0.000 | 0.245 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.174 ± 0.001 | 0.060 ± 0.000 | 0.245 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.179 ± 0.001 | 0.065 ± 0.007 | 0.249 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.191 ± 0.002 | 0.028 ± 0.000 | 0.409 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.192 ± 0.004 | 0.029 ± 0.000 | 0.410 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.196 ± 0.004 | 0.031 ± 0.000 | 0.410 ± 0.005 | 0.014 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.101 ± 0.002 | 0.029 ± 0.000 | 0.245 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.101 ± 0.002 | 0.029 ± 0.001 | 0.244 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.104 ± 0.001 | 0.031 ± 0.000 | 0.243 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.224 ± 0.002 | 0.022 ± 0.000 | 0.263 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.229 ± 0.002 | 0.027 ± 0.000 | 0.268 ± 0.039 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.248 ± 0.004 | 0.038 ± 0.000 | 0.285 ± 0.005 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.087 ± 0.001 | 0.022 ± 0.000 | 0.213 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.093 ± 0.002 | 0.027 ± 0.000 | 0.216 ± 0.003 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.099 ± 0.000 | 0.038 ± 0.000 | 0.229 ± 0.003 | 0.019 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261126_183225/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261126_183225/run.yaml`
- Timestamp: `2026-11-26T18:32:25.628209+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261126_183225`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261126_183225/cpu_ops_t4_20261126_183225.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 18.236 ± 0.088 | 14.268 ± 0.249 | 7.161 ± 0.104 | 10.333 ± 1.617 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 88.518 ± 0.395 | 64.315 ± 0.475 | 41.759 ± 2.749 | 55.920 ± 7.338 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 11.037 ± 0.096 | 14.251 ± 0.027 | 5.929 ± 0.019 | 10.957 ± 0.864 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 49.667 ± 1.043 | 64.426 ± 0.189 | 29.446 ± 1.808 | 56.880 ± 2.192 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 7.455 ± 0.049 | 3.869 ± 0.045 | 7.179 ± 0.152 | 4.445 ± 3.566 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 39.715 ± 0.176 | 21.020 ± 0.052 | 46.903 ± 0.359 | 32.890 ± 4.151 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 6.139 ± 0.072 | 3.922 ± 0.012 | 3.646 ± 0.075 | 4.006 ± 0.850 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 36.735 ± 1.082 | 23.591 ± 0.215 | 24.510 ± 0.258 | 31.925 ± 4.816 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 13.528 ± 0.293 | 9.111 ± 0.044 | 4.897 ± 0.244 | 8.930 ± 4.454 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 57.640 ± 3.613 | 33.652 ± 0.582 | 33.495 ± 0.836 | 54.959 ± 13.108 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 9.116 ± 0.075 | 9.181 ± 0.095 | 5.070 ± 0.105 | 9.671 ± 0.613 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 47.815 ± 0.758 | 33.759 ± 0.199 | 32.563 ± 0.796 | 52.962 ± 6.966 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.472 ± 0.008 | 1.232 ± 0.023 | 5.020 ± 5.572 | 0.905 ± 0.034 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 5.260 ± 0.082 | 4.988 ± 0.119 | 2.993 ± 0.106 | 3.772 ± 0.092 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.848 ± 0.003 | 1.236 ± 0.015 | 1.640 ± 3.980 | 0.955 ± 0.135 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 2.878 ± 0.104 | 5.019 ± 0.059 | 4.736 ± 0.091 | 10.053 ± 4.690 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 29.055 ± 0.059 | 22.062 ± 0.065 | 16.551 ± 0.368 | 17.988 ± 1.009 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 174.194 ± 1.323 | 124.191 ± 1.217 | 93.037 ± 3.291 | 102.269 ± 3.077 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 16.449 ± 0.136 | 21.993 ± 0.113 | 14.648 ± 0.389 | 17.988 ± 1.755 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 88.247 ± 0.918 | 123.468 ± 0.308 | 81.867 ± 2.821 | 106.948 ± 5.227 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.259 ± 0.015 | 0.043 ± 0.000 | 0.194 ± 0.007 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.269 ± 0.005 | 0.049 ± 0.001 | 0.201 ± 0.113 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.283 ± 0.002 | 0.059 ± 0.001 | 0.206 ± 0.002 | 0.019 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.105 ± 0.001 | 0.043 ± 0.000 | 0.193 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.112 ± 0.002 | 0.049 ± 0.001 | 0.199 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.117 ± 0.002 | 0.059 ± 0.001 | 0.203 ± 0.003 | 0.019 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.745 ± 0.028 | 0.044 ± 0.000 | 0.385 ± 0.008 | 0.013 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.755 ± 0.010 | 0.045 ± 0.000 | 1.325 ± 0.586 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.749 ± 0.013 | 0.048 ± 0.000 | 0.378 ± 0.003 | 0.020 ± 0.003 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.204 ± 0.002 | 0.041 ± 0.000 | 0.338 ± 0.010 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.204 ± 0.002 | 0.042 ± 0.011 | 0.337 ± 1.554 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.210 ± 0.003 | 0.045 ± 0.001 | 0.332 ± 0.009 | 0.020 ± 0.003 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.854 ± 0.011 | 0.067 ± 0.000 | 0.287 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.880 ± 0.010 | 0.069 ± 0.000 | 0.862 ± 0.169 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.896 ± 0.014 | 0.075 ± 0.001 | 0.292 ± 0.004 | 0.022 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.198 ± 0.002 | 0.070 ± 0.001 | 0.317 ± 0.009 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.199 ± 0.002 | 0.072 ± 0.001 | 1.266 ± 0.398 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.205 ± 0.001 | 0.077 ± 0.000 | 0.317 ± 0.003 | 0.019 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.200 ± 0.003 | 0.036 ± 0.000 | 0.407 ± 0.009 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.200 ± 0.003 | 0.036 ± 0.000 | 0.408 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.207 ± 0.002 | 0.037 ± 0.001 | 0.407 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.107 ± 0.002 | 0.036 ± 0.000 | 0.243 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.110 ± 0.001 | 0.036 ± 0.001 | 0.244 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.112 ± 0.001 | 0.039 ± 0.003 | 0.242 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.225 ± 0.003 | 0.027 ± 0.000 | 0.264 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.234 ± 0.002 | 0.031 ± 0.000 | 0.370 ± 0.008 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.251 ± 0.006 | 0.043 ± 0.001 | 0.286 ± 0.005 | 0.032 ± 0.007 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.094 ± 0.002 | 0.026 ± 0.001 | 0.215 ± 0.038 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.098 ± 0.002 | 0.030 ± 0.000 | 0.219 ± 0.012 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.108 ± 0.002 | 0.042 ± 0.000 | 0.231 ± 0.003 | 0.032 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

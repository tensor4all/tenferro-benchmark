# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `9e383937df4e365bd68bf5b16dbfcaaa269bcc82`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261115_180744/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261115_180744/run.yaml`
- Timestamp: `2026-11-15T18:07:44.075207+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261115_180744`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261115_180744/cpu_ops_t1_20261115_180744.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 17.079 ± 0.083 | 11.701 ± 0.071 | 10.363 ± 0.170 | 8.987 ± 0.152 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 107.736 ± 0.925 | 75.499 ± 0.528 | 66.714 ± 0.198 | 57.633 ± 1.333 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 9.551 ± 0.094 | 11.792 ± 0.197 | 8.498 ± 0.089 | 9.073 ± 0.024 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 60.881 ± 0.431 | 75.751 ± 0.129 | 52.871 ± 2.890 | 56.907 ± 0.733 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 11.006 ± 0.049 | 5.729 ± 0.095 | 12.668 ± 0.321 | 4.455 ± 0.089 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 74.355 ± 0.526 | 39.187 ± 0.337 | 81.228 ± 3.709 | 32.796 ± 1.484 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 10.175 ± 0.134 | 5.986 ± 0.083 | 5.517 ± 0.107 | 4.460 ± 0.033 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 71.228 ± 1.391 | 40.230 ± 1.081 | 42.739 ± 0.292 | 32.734 ± 0.838 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 16.678 ± 0.077 | 10.253 ± 0.315 | 7.317 ± 0.079 | 7.205 ± 0.051 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 119.255 ± 0.544 | 72.784 ± 0.096 | 53.224 ± 0.589 | 48.953 ± 0.463 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 12.700 ± 0.148 | 10.042 ± 0.176 | 7.310 ± 0.087 | 7.322 ± 0.114 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 93.841 ± 0.615 | 69.156 ± 0.817 | 52.166 ± 0.971 | 51.692 ± 2.203 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.860 ± 0.018 | 1.402 ± 0.008 | 1.241 ± 0.019 | 0.677 ± 0.017 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 8.836 ± 0.173 | 8.195 ± 0.093 | 5.604 ± 0.043 | 4.139 ± 0.082 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.883 ± 0.016 | 1.409 ± 0.003 | 1.734 ± 0.034 | 0.686 ± 0.016 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.805 ± 0.216 | 8.062 ± 0.103 | 9.522 ± 0.042 | 4.138 ± 0.059 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 29.205 ± 0.343 | 20.323 ± 0.318 | 18.979 ± 0.077 | 15.875 ± 0.282 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 191.679 ± 3.484 | 137.302 ± 0.553 | 136.092 ± 1.364 | 107.898 ± 1.395 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 15.130 ± 0.135 | 20.227 ± 0.184 | 15.936 ± 0.056 | 15.856 ± 0.140 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 102.092 ± 1.339 | 136.095 ± 1.266 | 108.490 ± 0.377 | 113.814 ± 1.172 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.253 ± 0.006 | 0.035 ± 0.001 | 0.196 ± 0.009 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.261 ± 0.010 | 0.041 ± 0.000 | 0.200 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.269 ± 0.003 | 0.047 ± 0.000 | 0.205 ± 0.005 | 0.014 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.096 ± 0.001 | 0.036 ± 0.000 | 0.196 ± 0.010 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.105 ± 0.001 | 0.043 ± 0.001 | 0.200 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.108 ± 0.002 | 0.049 ± 0.000 | 0.202 ± 0.004 | 0.015 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.703 ± 0.010 | 0.035 ± 0.000 | 0.278 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.699 ± 0.018 | 0.036 ± 0.000 | 0.279 ± 0.008 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.694 ± 0.020 | 0.037 ± 0.000 | 0.276 ± 0.005 | 0.016 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.186 ± 0.006 | 0.033 ± 0.000 | 0.257 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.188 ± 0.003 | 0.034 ± 0.000 | 0.257 ± 0.008 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.189 ± 0.001 | 0.035 ± 0.000 | 0.258 ± 0.005 | 0.017 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.939 ± 0.016 | 0.059 ± 0.001 | 0.218 ± 0.011 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.967 ± 0.016 | 0.061 ± 0.000 | 0.217 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.956 ± 0.007 | 0.064 ± 0.000 | 0.220 ± 0.005 | 0.018 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.177 ± 0.003 | 0.063 ± 0.001 | 0.248 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.181 ± 0.004 | 0.065 ± 0.001 | 0.249 ± 0.006 | 0.016 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.183 ± 0.004 | 0.068 ± 0.000 | 0.251 ± 0.007 | 0.015 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.185 ± 0.001 | 0.026 ± 0.000 | 0.413 ± 0.008 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.183 ± 0.009 | 0.026 ± 0.000 | 0.407 ± 0.008 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.185 ± 0.002 | 0.027 ± 0.000 | 0.408 ± 0.009 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.094 ± 0.002 | 0.026 ± 0.000 | 0.244 ± 0.004 | 0.016 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.096 ± 0.078 | 0.026 ± 0.000 | 0.244 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.094 ± 0.001 | 0.028 ± 0.000 | 0.243 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.222 ± 0.004 | 0.023 ± 0.000 | 0.263 ± 0.007 | 0.014 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.240 ± 0.006 | 0.031 ± 0.000 | 0.271 ± 0.003 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.264 ± 0.001 | 0.044 ± 0.000 | 0.283 ± 0.006 | 0.019 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.090 ± 0.002 | 0.023 ± 0.001 | 0.214 ± 0.015 | 0.015 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.099 ± 0.003 | 0.030 ± 0.000 | 0.217 ± 0.004 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.109 ± 0.002 | 0.045 ± 0.001 | 0.227 ± 0.005 | 0.020 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261115_184936/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261115_184936/run.yaml`
- Timestamp: `2026-11-15T18:49:36.321391+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261115_184936`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261115_184936/cpu_ops_t4_20261115_184936.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 32.623 ± 0.131 | 26.930 ± 0.094 | 7.165 ± 1.732 | 11.994 ± 1.971 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 155.695 ± 0.423 | 119.868 ± 0.769 | 43.299 ± 2.671 | 56.961 ± 1.285 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 18.774 ± 0.375 | 26.600 ± 0.144 | 5.977 ± 0.070 | 11.035 ± 1.463 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 86.186 ± 0.275 | 119.245 ± 0.219 | 30.709 ± 0.182 | 54.395 ± 9.059 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 16.770 ± 0.404 | 9.487 ± 0.104 | 8.790 ± 0.478 | 6.353 ± 0.716 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 361.531 ± 23.529 | 327.542 ± 9.755 | 50.575 ± 5.425 | 35.012 ± 9.836 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 14.482 ± 0.096 | 9.996 ± 0.038 | 3.593 ± 0.034 | 5.984 ± 1.002 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 214.753 ± 9.797 | 324.988 ± 44.644 | 24.331 ± 0.372 | 30.301 ± 6.882 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 268.663 ± 17.699 | 16.050 ± 59.148 | 4.836 ± 0.041 | 6.561 ± 2.645 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 322.530 ± 154.186 | 391.820 ± 15.321 | 33.579 ± 0.319 | 50.945 ± 4.075 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 66.104 ± 59.636 | 57.343 ± 35.456 | 5.365 ± 0.057 | 7.987 ± 2.990 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 257.304 ± 11.257 | 390.937 ± 2.909 | 33.512 ± 0.524 | 52.166 ± 8.537 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 3.103 ± 0.028 | 2.677 ± 0.034 | 1.047 ± 0.009 | 0.894 ± 0.098 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 285.251 ± 9.903 | 313.985 ± 23.988 | 2.933 ± 0.016 | 3.868 ± 0.289 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.537 ± 0.036 | 2.642 ± 0.174 | 1.241 ± 0.069 | 0.971 ± 0.060 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 136.895 ± 18.009 | 299.005 ± 12.114 | 4.733 ± 0.071 | 3.835 ± 0.084 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 37.190 ± 4.087 | 25.418 ± 0.210 | 16.434 ± 0.157 | 16.632 ± 1.302 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 241.224 ± 4.091 | 175.090 ± 0.285 | 93.537 ± 0.588 | 101.027 ± 6.267 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 20.776 ± 0.134 | 26.815 ± 0.349 | 14.735 ± 0.080 | 17.973 ± 0.753 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 127.239 ± 1.427 | 175.813 ± 1.320 | 82.953 ± 1.123 | 191.812 ± 38.114 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.263 ± 0.001 | 0.044 ± 0.001 | 0.195 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.277 ± 0.002 | 0.051 ± 0.000 | 0.199 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.287 ± 0.007 | 0.058 ± 0.000 | 0.209 ± 0.004 | 0.019 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.111 ± 0.002 | 0.046 ± 0.001 | 0.193 ± 0.004 | 0.013 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.119 ± 0.002 | 0.053 ± 0.001 | 0.201 ± 0.005 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.124 ± 0.000 | 0.060 ± 0.001 | 0.204 ± 0.003 | 0.018 ± 0.004 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.737 ± 0.010 | 0.045 ± 0.000 | 0.382 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.729 ± 0.004 | 0.046 ± 0.000 | 0.379 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.754 ± 0.001 | 0.047 ± 0.001 | 0.382 ± 0.006 | 0.021 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.220 ± 0.003 | 0.042 ± 0.001 | 0.330 ± 0.010 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.221 ± 0.002 | 0.043 ± 0.001 | 0.332 ± 0.009 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.228 ± 0.002 | 0.045 ± 0.000 | 0.334 ± 0.015 | 0.018 ± 0.003 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.984 ± 0.013 | 0.074 ± 0.000 | 0.287 ± 0.009 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.994 ± 0.010 | 0.076 ± 0.000 | 0.285 ± 0.006 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.996 ± 0.005 | 0.080 ± 0.001 | 0.286 ± 0.003 | 0.022 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.211 ± 0.003 | 0.076 ± 0.000 | 0.313 ± 0.013 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.213 ± 0.001 | 0.079 ± 0.002 | 0.313 ± 0.007 | 0.016 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.218 ± 0.002 | 0.083 ± 0.001 | 0.319 ± 0.007 | 0.018 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.197 ± 0.001 | 0.033 ± 0.001 | 0.408 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.198 ± 0.003 | 0.034 ± 0.000 | 0.408 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.200 ± 0.001 | 0.035 ± 0.000 | 0.416 ± 0.010 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.106 ± 0.001 | 0.033 ± 0.000 | 0.246 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.106 ± 0.001 | 0.034 ± 0.001 | 0.246 ± 0.034 | 0.015 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.108 ± 0.001 | 0.035 ± 0.000 | 0.243 ± 0.005 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.234 ± 0.005 | 0.029 ± 0.000 | 0.267 ± 0.145 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.250 ± 0.004 | 0.037 ± 0.000 | 0.269 ± 0.007 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.272 ± 0.002 | 0.051 ± 0.001 | 0.289 ± 0.008 | 0.026 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.099 ± 0.001 | 0.028 ± 0.000 | 0.212 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.108 ± 0.001 | 0.037 ± 0.000 | 0.218 ± 0.004 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.120 ± 0.000 | 0.052 ± 0.001 | 0.231 ± 0.006 | 0.027 ± 0.006 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

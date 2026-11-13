# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `f3a05ec9a2a7262482348fd5bddccbe90389866b`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261113_180725/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261113_180725/run.yaml`
- Timestamp: `2026-11-13T18:07:25.705019+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261113_180725`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261113_180725/cpu_ops_t1_20261113_180725.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 16.822 ± 0.116 | 11.497 ± 0.043 | 10.184 ± 0.113 | 8.915 ± 0.017 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 106.028 ± 0.942 | 75.075 ± 0.532 | 65.360 ± 0.290 | 57.118 ± 0.353 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 9.346 ± 0.046 | 11.554 ± 0.026 | 8.444 ± 0.036 | 8.899 ± 0.044 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 59.931 ± 0.807 | 75.796 ± 0.856 | 51.259 ± 0.556 | 56.758 ± 0.562 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 10.760 ± 0.124 | 5.622 ± 0.051 | 12.114 ± 0.054 | 4.274 ± 0.048 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 74.106 ± 1.428 | 38.334 ± 0.076 | 79.148 ± 0.771 | 30.755 ± 0.511 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 9.906 ± 0.066 | 5.871 ± 0.017 | 5.362 ± 0.090 | 4.278 ± 0.012 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 70.277 ± 0.324 | 39.186 ± 1.080 | 42.702 ± 0.256 | 32.080 ± 0.143 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 16.258 ± 0.050 | 10.009 ± 0.103 | 7.215 ± 0.018 | 7.104 ± 0.057 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 115.036 ± 0.685 | 72.578 ± 0.592 | 50.909 ± 0.223 | 45.358 ± 2.272 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 12.437 ± 0.043 | 9.912 ± 0.066 | 7.223 ± 0.057 | 7.118 ± 0.027 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 90.339 ± 0.484 | 69.210 ± 0.785 | 51.096 ± 0.557 | 53.361 ± 0.347 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.824 ± 0.039 | 1.407 ± 0.013 | 1.252 ± 0.017 | 0.658 ± 0.005 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 8.525 ± 0.133 | 7.914 ± 0.029 | 5.445 ± 0.107 | 4.086 ± 0.025 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.866 ± 0.009 | 1.403 ± 0.025 | 2.119 ± 0.015 | 0.665 ± 0.004 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.431 ± 0.083 | 7.889 ± 0.056 | 9.372 ± 0.098 | 4.003 ± 0.043 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 28.669 ± 0.339 | 19.723 ± 0.136 | 20.555 ± 0.274 | 15.373 ± 0.805 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 192.025 ± 1.349 | 141.119 ± 0.165 | 133.788 ± 0.693 | 108.117 ± 0.901 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 14.897 ± 0.065 | 19.682 ± 0.079 | 15.832 ± 0.164 | 15.377 ± 0.033 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 101.633 ± 0.401 | 140.025 ± 0.646 | 106.551 ± 0.595 | 114.858 ± 0.978 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.250 ± 0.002 | 0.036 ± 0.001 | 0.191 ± 0.004 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.265 ± 0.001 | 0.043 ± 0.001 | 0.196 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.278 ± 0.002 | 0.051 ± 0.000 | 0.203 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.099 ± 0.001 | 0.038 ± 0.001 | 0.192 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.108 ± 0.002 | 0.046 ± 0.001 | 0.196 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.114 ± 0.002 | 0.052 ± 0.001 | 0.202 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.688 ± 0.012 | 0.035 ± 0.000 | 0.269 ± 0.009 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.696 ± 0.008 | 0.036 ± 0.000 | 0.266 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.696 ± 0.005 | 0.037 ± 0.000 | 0.272 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.188 ± 0.001 | 0.034 ± 0.001 | 0.253 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.190 ± 0.002 | 0.034 ± 0.000 | 0.253 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.195 ± 0.003 | 0.036 ± 0.000 | 0.257 ± 0.005 | 0.016 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.946 ± 0.027 | 0.059 ± 0.000 | 0.213 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.948 ± 0.006 | 0.061 ± 0.001 | 0.212 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.948 ± 0.010 | 0.065 ± 0.000 | 0.216 ± 0.004 | 0.017 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.180 ± 0.001 | 0.062 ± 0.001 | 0.242 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.183 ± 0.001 | 0.065 ± 0.000 | 0.244 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.189 ± 0.000 | 0.069 ± 0.000 | 0.246 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.186 ± 0.001 | 0.026 ± 0.000 | 0.403 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.189 ± 0.002 | 0.027 ± 0.000 | 0.404 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.190 ± 0.002 | 0.028 ± 0.001 | 0.403 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.094 ± 0.001 | 0.026 ± 0.000 | 0.240 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.095 ± 0.002 | 0.027 ± 0.000 | 0.241 ± 0.006 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.098 ± 0.002 | 0.028 ± 0.000 | 0.241 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.223 ± 0.002 | 0.023 ± 0.000 | 0.256 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.239 ± 0.001 | 0.032 ± 0.000 | 0.263 ± 0.003 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.262 ± 0.009 | 0.046 ± 0.006 | 0.278 ± 0.004 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.092 ± 0.002 | 0.023 ± 0.000 | 0.208 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.100 ± 0.001 | 0.031 ± 0.000 | 0.215 ± 0.004 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.112 ± 0.001 | 0.045 ± 0.000 | 0.225 ± 0.004 | 0.019 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261113_184804/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261113_184804/run.yaml`
- Timestamp: `2026-11-13T18:48:04.785108+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261113_184804`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261113_184804/cpu_ops_t4_20261113_184804.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 31.288 ± 0.248 | 26.233 ± 0.054 | 7.136 ± 6.316 | 10.474 ± 0.989 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 152.667 ± 0.591 | 116.296 ± 0.407 | 42.654 ± 1.450 | 57.033 ± 5.921 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 18.559 ± 0.302 | 26.072 ± 0.150 | 5.913 ± 0.192 | 10.033 ± 0.857 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 83.913 ± 0.425 | 116.946 ± 0.512 | 30.933 ± 8.038 | 56.753 ± 3.381 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 16.308 ± 0.275 | 9.288 ± 0.018 | 8.886 ± 0.360 | 6.833 ± 1.812 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 332.526 ± 30.283 | 309.134 ± 35.759 | 46.945 ± 0.688 | 30.970 ± 5.157 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 14.337 ± 0.271 | 9.853 ± 0.046 | 3.588 ± 0.045 | 5.805 ± 0.713 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 204.774 ± 27.075 | 323.223 ± 32.445 | 24.040 ± 0.313 | 33.606 ± 8.323 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 169.854 ± 107.086 | 17.660 ± 35.381 | 22.049 ± 3.456 | 8.684 ± 1.591 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 433.802 ± 14.493 | 373.132 ± 163.877 | 33.498 ± 0.444 | 53.328 ± 9.748 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 44.041 ± 48.751 | 15.280 ± 11.350 | 5.190 ± 0.073 | 10.464 ± 2.196 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 264.197 ± 12.897 | 382.293 ± 14.663 | 32.454 ± 0.389 | 55.912 ± 11.714 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 3.065 ± 0.004 | 2.623 ± 0.011 | 3.018 ± 2.977 | 0.992 ± 0.003 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 279.107 ± 20.002 | 311.145 ± 24.081 | 2.934 ± 0.161 | 3.740 ± 0.163 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.495 ± 0.018 | 2.614 ± 0.008 | 1.444 ± 0.077 | 0.964 ± 0.099 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 124.857 ± 17.964 | 304.917 ± 20.887 | 4.783 ± 0.112 | 3.686 ± 0.017 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 35.600 ± 0.358 | 24.588 ± 0.145 | 66.351 ± 15.872 | 18.306 ± 0.689 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 237.262 ± 0.946 | 174.782 ± 1.639 | 92.135 ± 1.412 | 100.959 ± 3.761 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 20.405 ± 0.192 | 26.208 ± 0.097 | 56.047 ± 14.996 | 17.038 ± 1.250 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 125.724 ± 1.340 | 176.963 ± 1.671 | 81.293 ± 0.909 | 101.957 ± 5.852 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.262 ± 0.004 | 0.044 ± 0.000 | 0.192 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.274 ± 0.003 | 0.052 ± 0.000 | 0.196 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.286 ± 0.005 | 0.058 ± 0.001 | 0.202 ± 0.003 | 0.019 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.109 ± 0.002 | 0.046 ± 0.000 | 0.189 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.118 ± 0.002 | 0.052 ± 0.000 | 0.196 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.123 ± 0.002 | 0.059 ± 0.001 | 0.201 ± 0.004 | 0.019 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.729 ± 0.006 | 0.045 ± 0.000 | 0.372 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.723 ± 0.008 | 0.046 ± 0.001 | 0.369 ± 0.006 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.733 ± 0.006 | 0.047 ± 0.000 | 0.371 ± 0.004 | 0.020 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.217 ± 0.002 | 0.042 ± 0.000 | 0.317 ± 0.018 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.219 ± 0.002 | 0.043 ± 0.000 | 0.325 ± 0.008 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.224 ± 0.005 | 0.044 ± 0.000 | 0.327 ± 0.008 | 0.018 ± 0.003 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.969 ± 0.008 | 0.074 ± 0.000 | 0.281 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.977 ± 0.002 | 0.077 ± 0.000 | 0.279 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.977 ± 0.015 | 0.081 ± 0.001 | 0.281 ± 0.005 | 0.022 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.207 ± 0.001 | 0.077 ± 0.001 | 0.303 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.211 ± 0.003 | 0.079 ± 0.001 | 0.310 ± 0.005 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.214 ± 0.002 | 0.083 ± 0.002 | 0.312 ± 0.007 | 0.019 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.195 ± 0.004 | 0.033 ± 0.000 | 0.400 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.198 ± 0.004 | 0.034 ± 0.000 | 0.399 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.198 ± 0.003 | 0.035 ± 0.000 | 0.398 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.104 ± 0.002 | 0.033 ± 0.000 | 0.239 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.106 ± 0.001 | 0.034 ± 0.001 | 0.242 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.106 ± 0.001 | 0.035 ± 0.000 | 0.241 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.231 ± 0.002 | 0.029 ± 0.000 | 0.259 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.246 ± 0.002 | 0.037 ± 0.000 | 0.263 ± 0.006 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.271 ± 0.003 | 0.051 ± 0.001 | 0.278 ± 0.002 | 0.033 ± 0.006 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.099 ± 0.001 | 0.027 ± 0.000 | 0.208 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.107 ± 0.001 | 0.036 ± 0.000 | 0.214 ± 0.004 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.119 ± 0.001 | 0.050 ± 0.000 | 0.227 ± 0.009 | 0.033 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

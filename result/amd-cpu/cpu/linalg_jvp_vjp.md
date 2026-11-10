# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `15cf8f674203aeca079999fdb339cf4d35801b88`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261110_180539/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261110_180539/run.yaml`
- Timestamp: `2026-11-10T18:05:39.033513+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261110_180539`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261110_180539/cpu_ops_t1_20261110_180539.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 15.678 ± 0.037 | 9.838 ± 0.055 | 8.721 ± 0.058 | 7.066 ± 0.048 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 95.118 ± 0.308 | 63.776 ± 0.557 | 57.167 ± 0.254 | 44.731 ± 0.070 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 8.959 ± 0.057 | 9.848 ± 0.028 | 7.142 ± 0.021 | 7.027 ± 0.014 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 55.604 ± 0.424 | 63.930 ± 0.570 | 44.899 ± 0.368 | 44.178 ± 0.115 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 9.226 ± 0.087 | 4.693 ± 0.029 | 10.651 ± 0.128 | 4.077 ± 0.013 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 77.711 ± 25.997 | 32.624 ± 0.469 | 67.888 ± 0.568 | 27.558 ± 0.289 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 8.492 ± 0.096 | 4.912 ± 0.018 | 4.576 ± 0.023 | 3.955 ± 0.017 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 62.871 ± 0.551 | 33.236 ± 0.080 | 36.252 ± 0.319 | 27.062 ± 0.073 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 14.561 ± 0.051 | 8.921 ± 0.043 | 6.264 ± 0.021 | 5.596 ± 0.047 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 100.540 ± 0.150 | 63.722 ± 0.484 | 44.928 ± 0.153 | 37.069 ± 0.282 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 10.917 ± 0.092 | 8.606 ± 0.022 | 6.128 ± 0.039 | 5.520 ± 0.035 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 78.247 ± 0.454 | 60.480 ± 0.258 | 44.357 ± 0.195 | 41.380 ± 0.973 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.572 ± 0.032 | 1.095 ± 0.006 | 1.008 ± 0.021 | 0.566 ± 0.003 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 7.967 ± 0.068 | 6.696 ± 0.055 | 4.678 ± 0.061 | 3.499 ± 0.012 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.687 ± 0.001 | 1.099 ± 0.004 | 1.407 ± 0.022 | 0.575 ± 0.003 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.182 ± 0.093 | 6.596 ± 0.039 | 8.069 ± 0.041 | 3.591 ± 0.030 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 24.562 ± 0.171 | 17.098 ± 0.097 | 18.163 ± 0.286 | 12.279 ± 0.076 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 164.370 ± 1.160 | 116.991 ± 0.530 | 118.266 ± 0.665 | 87.351 ± 0.512 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 12.601 ± 0.152 | 17.054 ± 0.051 | 13.495 ± 0.126 | 12.212 ± 0.079 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 86.393 ± 0.487 | 117.111 ± 0.200 | 94.439 ± 0.460 | 91.948 ± 0.559 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.191 ± 0.001 | 0.022 ± 0.000 | 0.119 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.202 ± 0.003 | 0.025 ± 0.000 | 0.124 ± 0.002 | 0.009 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.208 ± 0.003 | 0.030 ± 0.000 | 0.131 ± 0.000 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.060 ± 0.000 | 0.022 ± 0.001 | 0.113 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.068 ± 0.001 | 0.025 ± 0.000 | 0.119 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.072 ± 0.000 | 0.030 ± 0.000 | 0.123 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.560 ± 0.014 | 0.027 ± 0.000 | 0.184 ± 0.006 | 0.009 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.565 ± 0.003 | 0.027 ± 0.000 | 0.182 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.561 ± 0.005 | 0.028 ± 0.000 | 0.184 ± 0.002 | 0.014 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.139 ± 0.001 | 0.028 ± 0.007 | 0.166 ± 0.005 | 0.009 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.140 ± 0.001 | 0.025 ± 0.000 | 0.168 ± 0.003 | 0.010 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.143 ± 0.001 | 0.026 ± 0.000 | 0.168 ± 0.004 | 0.014 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.749 ± 0.003 | 0.038 ± 0.000 | 0.139 ± 0.004 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.781 ± 0.008 | 0.039 ± 0.000 | 0.139 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.772 ± 0.020 | 0.041 ± 0.000 | 0.143 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.134 ± 0.000 | 0.038 ± 0.001 | 0.159 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.137 ± 0.003 | 0.040 ± 0.000 | 0.160 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.140 ± 0.001 | 0.042 ± 0.000 | 0.162 ± 0.002 | 0.012 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.138 ± 0.000 | 0.020 ± 0.000 | 0.272 ± 0.004 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.139 ± 0.002 | 0.020 ± 0.000 | 0.271 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.142 ± 0.005 | 0.021 ± 0.000 | 0.273 ± 0.009 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.058 ± 0.002 | 0.020 ± 0.000 | 0.151 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.058 ± 0.000 | 0.020 ± 0.000 | 0.153 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.059 ± 0.001 | 0.021 ± 0.000 | 0.154 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.165 ± 0.001 | 0.016 ± 0.000 | 0.181 ± 0.007 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.179 ± 0.005 | 0.020 ± 0.000 | 0.184 ± 0.004 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.197 ± 0.003 | 0.031 ± 0.000 | 0.198 ± 0.003 | 0.017 ± 0.009 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.054 ± 0.001 | 0.015 ± 0.000 | 0.138 ± 0.040 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.061 ± 0.000 | 0.019 ± 0.000 | 0.139 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.071 ± 0.000 | 0.030 ± 0.000 | 0.149 ± 0.001 | 0.015 ± 0.005 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261110_184045/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261110_184045/run.yaml`
- Timestamp: `2026-11-10T18:40:45.384615+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261110_184045`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261110_184045/cpu_ops_t4_20261110_184045.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 28.647 ± 0.206 | 22.014 ± 0.428 | 6.384 ± 3.218 | 8.569 ± 0.613 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 130.020 ± 0.802 | 98.924 ± 0.485 | 37.077 ± 4.289 | 42.076 ± 4.818 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 15.309 ± 0.087 | 22.245 ± 0.124 | 5.403 ± 0.125 | 9.089 ± 0.095 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 71.663 ± 0.780 | 99.694 ± 0.293 | 26.349 ± 0.654 | 45.485 ± 2.713 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 14.195 ± 0.420 | 8.064 ± 0.064 | 6.945 ± 0.254 | 4.987 ± 0.597 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 268.178 ± 20.144 | 190.309 ± 44.728 | 40.674 ± 0.374 | 30.178 ± 3.839 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 11.971 ± 0.039 | 8.393 ± 0.042 | 3.116 ± 0.027 | 4.038 ± 1.835 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 182.095 ± 9.797 | 198.739 ± 50.505 | 21.255 ± 0.373 | 27.908 ± 0.717 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 158.103 ± 20.572 | 13.374 ± 0.162 | 4.208 ± 0.172 | 7.133 ± 1.316 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 414.144 ± 24.871 | 374.480 ± 12.098 | 28.694 ± 0.861 | 44.656 ± 3.096 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 14.972 ± 0.721 | 13.308 ± 0.136 | 5.372 ± 0.102 | 7.323 ± 1.483 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 243.828 ± 13.961 | 362.015 ± 20.379 | 28.158 ± 0.248 | 39.979 ± 10.815 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 3.209 ± 0.008 | 2.740 ± 0.007 | 0.868 ± 0.192 | 0.738 ± 0.043 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 190.085 ± 35.449 | 176.151 ± 59.742 | 2.640 ± 0.674 | 2.992 ± 0.003 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.515 ± 0.005 | 2.718 ± 0.013 | 1.028 ± 0.020 | 0.814 ± 0.029 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 111.847 ± 8.896 | 258.014 ± 65.017 | 4.174 ± 0.062 | 3.184 ± 0.015 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 30.698 ± 0.099 | 21.538 ± 0.307 | 17.034 ± 0.148 | 14.981 ± 1.004 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 205.002 ± 4.898 | 153.182 ± 0.562 | 81.685 ± 0.579 | 90.247 ± 4.668 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 16.875 ± 0.050 | 22.477 ± 0.323 | 13.697 ± 1.789 | 14.958 ± 1.071 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 106.217 ± 0.192 | 154.291 ± 0.901 | 66.103 ± 0.582 | 84.218 ± 5.486 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.204 ± 0.001 | 0.029 ± 0.000 | 0.117 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.214 ± 0.003 | 0.032 ± 0.000 | 0.124 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.223 ± 0.002 | 0.037 ± 0.000 | 0.131 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.070 ± 0.000 | 0.029 ± 0.000 | 0.113 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.078 ± 0.001 | 0.032 ± 0.000 | 0.121 ± 0.002 | 0.011 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.084 ± 0.001 | 0.037 ± 0.000 | 0.122 ± 0.001 | 0.014 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.583 ± 0.005 | 0.036 ± 0.000 | 0.247 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.593 ± 0.004 | 0.037 ± 0.000 | 0.245 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.599 ± 0.010 | 0.038 ± 0.000 | 0.249 ± 0.005 | 0.016 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.170 ± 0.001 | 0.033 ± 0.000 | 0.211 ± 0.005 | 0.009 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.172 ± 0.000 | 0.034 ± 0.000 | 0.210 ± 0.004 | 0.010 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.175 ± 0.002 | 0.035 ± 0.000 | 0.212 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.771 ± 0.003 | 0.051 ± 0.000 | 0.182 ± 0.011 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.807 ± 0.005 | 0.053 ± 0.001 | 0.179 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.805 ± 0.009 | 0.054 ± 0.000 | 0.179 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.163 ± 0.001 | 0.051 ± 0.000 | 0.196 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.166 ± 0.000 | 0.052 ± 0.001 | 0.197 ± 0.004 | 0.010 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.170 ± 0.000 | 0.055 ± 0.000 | 0.197 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.148 ± 0.002 | 0.027 ± 0.000 | 0.273 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.150 ± 0.000 | 0.028 ± 0.000 | 0.275 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.150 ± 0.001 | 0.029 ± 0.000 | 0.277 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.068 ± 0.001 | 0.027 ± 0.000 | 0.153 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.069 ± 0.001 | 0.027 ± 0.000 | 0.156 ± 0.004 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.071 ± 0.001 | 0.028 ± 0.000 | 0.155 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.176 ± 0.006 | 0.020 ± 0.000 | 0.180 ± 0.007 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.188 ± 0.001 | 0.025 ± 0.000 | 0.185 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.206 ± 0.002 | 0.035 ± 0.000 | 0.198 ± 0.002 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.062 ± 0.001 | 0.019 ± 0.000 | 0.133 ± 0.006 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.069 ± 0.001 | 0.023 ± 0.000 | 0.139 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.080 ± 0.001 | 0.034 ± 0.000 | 0.149 ± 0.001 | 0.019 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

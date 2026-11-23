# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Benchmark commit: `d9f6406557a0d1066f37785ec85f9b9afdbb4781`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261123_175252/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261123_175252/run.yaml`
- Timestamp: `2026-11-23T17:52:52.587828+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261123_175252`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261123_175252/cpu_ops_t1_20261123_175252.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 25.408 ± 0.197 | 12.968 ± 0.061 | 11.082 ± 0.061 | 9.482 ± 0.025 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 132.043 ± 0.577 | 66.707 ± 0.076 | 71.739 ± 1.248 | 59.246 ± 0.161 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 17.467 ± 0.012 | 12.996 ± 0.020 | 9.069 ± 0.191 | 9.436 ± 0.014 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 94.202 ± 2.250 | 66.782 ± 0.581 | 55.873 ± 0.278 | 58.965 ± 0.283 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 11.226 ± 0.042 | 5.237 ± 0.019 | 13.256 ± 0.054 | 4.383 ± 0.026 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 80.719 ± 0.377 | 36.260 ± 0.409 | 87.316 ± 1.076 | 31.484 ± 0.284 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 10.271 ± 0.032 | 5.433 ± 0.029 | 5.845 ± 0.117 | 4.460 ± 0.021 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 78.036 ± 0.314 | 39.789 ± 2.324 | 46.338 ± 0.972 | 31.743 ± 0.679 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 17.787 ± 0.138 | 8.793 ± 0.021 | 8.061 ± 0.029 | 6.954 ± 0.050 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 128.193 ± 2.774 | 61.005 ± 0.359 | 57.500 ± 0.260 | 48.698 ± 2.681 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 14.384 ± 0.027 | 8.696 ± 0.043 | 7.942 ± 0.096 | 7.168 ± 0.066 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 109.087 ± 1.710 | 60.057 ± 0.834 | 57.340 ± 0.544 | 52.068 ± 0.108 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 2.224 ± 0.007 | 1.118 ± 0.011 | 1.296 ± 0.019 | 0.686 ± 0.005 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 11.659 ± 0.096 | 6.731 ± 0.068 | 5.962 ± 0.039 | 4.386 ± 0.022 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.281 ± 0.007 | 1.127 ± 0.001 | 1.807 ± 0.021 | 0.688 ± 0.003 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 7.074 ± 0.061 | 6.555 ± 0.037 | 10.189 ± 0.076 | 4.303 ± 0.041 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 45.611 ± 0.150 | 22.668 ± 0.143 | 22.479 ± 0.036 | 16.430 ± 0.203 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 278.138 ± 0.332 | 139.289 ± 0.474 | 149.708 ± 0.698 | 115.710 ± 0.352 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 30.764 ± 0.138 | 22.672 ± 0.093 | 17.048 ± 0.213 | 16.308 ± 0.103 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 186.972 ± 0.218 | 137.223 ± 0.667 | 119.788 ± 0.428 | 122.908 ± 0.794 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.260 ± 0.003 | 0.026 ± 0.000 | 0.156 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.271 ± 0.005 | 0.029 ± 0.000 | 0.164 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.292 ± 0.003 | 0.039 ± 0.001 | 0.170 ± 0.004 | 0.015 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.088 ± 0.001 | 0.025 ± 0.000 | 0.148 ± 0.005 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.096 ± 0.001 | 0.029 ± 0.001 | 0.157 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.108 ± 0.002 | 0.038 ± 0.001 | 0.161 ± 0.003 | 0.016 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.755 ± 0.004 | 0.033 ± 0.000 | 0.236 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.767 ± 0.007 | 0.033 ± 0.000 | 0.231 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.761 ± 0.004 | 0.035 ± 0.000 | 0.237 ± 0.004 | 0.019 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.187 ± 0.001 | 0.030 ± 0.000 | 0.215 ± 0.004 | 0.011 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.189 ± 0.001 | 0.030 ± 0.000 | 0.217 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.197 ± 0.003 | 0.032 ± 0.000 | 0.222 ± 0.003 | 0.018 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 1.006 ± 0.003 | 0.043 ± 0.000 | 0.179 ± 0.006 | 0.011 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 1.020 ± 0.005 | 0.043 ± 0.000 | 0.178 ± 0.006 | 0.012 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 1.025 ± 0.006 | 0.048 ± 0.001 | 0.183 ± 0.005 | 0.018 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.182 ± 0.003 | 0.043 ± 0.000 | 0.205 ± 0.005 | 0.011 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.182 ± 0.004 | 0.044 ± 0.000 | 0.205 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.190 ± 0.002 | 0.048 ± 0.001 | 0.211 ± 0.006 | 0.014 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.207 ± 0.001 | 0.026 ± 0.000 | 0.356 ± 0.010 | 0.011 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.209 ± 0.004 | 0.026 ± 0.000 | 0.360 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.212 ± 0.001 | 0.027 ± 0.000 | 0.357 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.100 ± 0.002 | 0.025 ± 0.000 | 0.198 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.102 ± 0.001 | 0.025 ± 0.000 | 0.199 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.105 ± 0.001 | 0.027 ± 0.001 | 0.199 ± 0.006 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.227 ± 0.001 | 0.019 ± 0.000 | 0.231 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.237 ± 0.001 | 0.023 ± 0.000 | 0.240 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.267 ± 0.003 | 0.035 ± 0.000 | 0.258 ± 0.008 | 0.020 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.084 ± 0.002 | 0.018 ± 0.000 | 0.174 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.089 ± 0.005 | 0.022 ± 0.000 | 0.181 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.107 ± 0.028 | 0.034 ± 0.000 | 0.194 ± 0.005 | 0.020 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261123_183450/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261123_183450/run.yaml`
- Timestamp: `2026-11-23T18:34:50.691833+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261123_183450`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261123_183450/cpu_ops_t4_20261123_183450.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 26.864 ± 0.115 | 15.030 ± 0.171 | 7.829 ± 0.236 | 11.348 ± 1.492 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 134.371 ± 0.947 | 67.603 ± 0.195 | 44.634 ± 1.271 | 59.131 ± 3.493 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 19.648 ± 0.785 | 15.041 ± 0.039 | 6.513 ± 0.208 | 9.029 ± 0.807 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 91.307 ± 1.456 | 67.447 ± 0.072 | 32.643 ± 11.369 | 59.922 ± 6.762 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 10.242 ± 0.152 | 4.175 ± 0.042 | 8.306 ± 0.320 | 5.245 ± 1.869 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 49.323 ± 0.112 | 23.011 ± 0.067 | 49.249 ± 0.353 | 31.529 ± 12.717 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 8.842 ± 0.420 | 4.231 ± 0.007 | 3.890 ± 0.019 | 4.280 ± 0.923 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 44.559 ± 0.095 | 24.705 ± 1.044 | 25.669 ± 0.421 | 32.405 ± 4.471 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 17.083 ± 0.107 | 9.093 ± 0.013 | 5.151 ± 0.080 | 9.122 ± 2.614 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 72.881 ± 5.956 | 34.564 ± 0.813 | 35.611 ± 0.333 | 60.773 ± 10.622 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 12.515 ± 0.094 | 9.305 ± 0.033 | 5.713 ± 0.199 | 9.504 ± 1.243 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 58.988 ± 0.841 | 35.639 ± 0.620 | 34.394 ± 0.711 | 54.961 ± 7.163 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 2.959 ± 0.517 | 1.185 ± 0.019 | 1.041 ± 0.036 | 0.927 ± 0.083 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 9.300 ± 0.226 | 5.070 ± 0.033 | 3.124 ± 0.038 | 4.052 ± 0.113 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.856 ± 0.140 | 1.229 ± 0.031 | 1.292 ± 0.012 | 0.953 ± 0.177 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 5.592 ± 0.250 | 5.210 ± 0.028 | 5.049 ± 0.078 | 3.877 ± 0.209 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 47.075 ± 0.055 | 24.295 ± 0.084 | 17.877 ± 0.108 | 17.269 ± 1.236 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 280.426 ± 0.864 | 135.353 ± 0.359 | 100.667 ± 15.142 | 107.955 ± 3.010 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 32.238 ± 0.520 | 24.297 ± 0.025 | 16.406 ± 0.463 | 17.134 ± 1.592 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 187.035 ± 0.366 | 134.984 ± 1.406 | 89.841 ± 1.556 | 121.022 ± 10.560 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.277 ± 0.002 | 0.037 ± 0.000 | 0.155 ± 0.006 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.288 ± 0.002 | 0.041 ± 0.000 | 0.161 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.305 ± 0.002 | 0.050 ± 0.000 | 0.171 ± 0.003 | 0.017 ± 0.007 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.106 ± 0.002 | 0.035 ± 0.000 | 0.146 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.116 ± 0.003 | 0.038 ± 0.000 | 0.154 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.127 ± 0.002 | 0.048 ± 0.001 | 0.162 ± 0.005 | 0.018 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.791 ± 0.022 | 0.047 ± 0.000 | 0.325 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.795 ± 0.009 | 0.048 ± 0.000 | 0.323 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.807 ± 0.002 | 0.050 ± 0.000 | 0.322 ± 0.005 | 0.021 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.235 ± 0.004 | 0.042 ± 0.000 | 0.284 ± 0.048 | 0.011 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.237 ± 0.005 | 0.042 ± 0.011 | 0.269 ± 0.006 | 0.013 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.243 ± 0.002 | 0.045 ± 0.000 | 0.271 ± 0.005 | 0.019 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 1.029 ± 0.004 | 0.061 ± 0.000 | 0.234 ± 0.006 | 0.011 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 1.060 ± 0.023 | 0.063 ± 0.001 | 0.230 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 1.062 ± 0.012 | 0.067 ± 0.001 | 0.237 ± 0.007 | 0.020 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.227 ± 0.001 | 0.060 ± 0.001 | 0.257 ± 0.003 | 0.011 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.228 ± 0.001 | 0.061 ± 0.001 | 0.253 ± 0.005 | 0.013 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.240 ± 0.006 | 0.066 ± 0.014 | 0.255 ± 0.006 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.219 ± 0.001 | 0.036 ± 0.000 | 0.359 ± 0.014 | 0.011 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.222 ± 0.001 | 0.036 ± 0.000 | 0.355 ± 0.011 | 0.011 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.228 ± 0.001 | 0.038 ± 0.000 | 0.356 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.117 ± 0.001 | 0.035 ± 0.000 | 0.194 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.116 ± 0.001 | 0.036 ± 0.000 | 0.197 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.120 ± 0.001 | 0.037 ± 0.000 | 0.197 ± 0.005 | 0.013 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.237 ± 0.001 | 0.025 ± 0.000 | 0.233 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.248 ± 0.002 | 0.029 ± 0.000 | 0.242 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.277 ± 0.002 | 0.042 ± 0.001 | 0.260 ± 0.010 | 0.024 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.096 ± 0.003 | 0.023 ± 0.000 | 0.170 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.102 ± 0.002 | 0.027 ± 0.000 | 0.179 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.120 ± 0.002 | 0.040 ± 0.000 | 0.192 ± 0.004 | 0.025 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Benchmark commit: `9185e8ca26c3fc600458361d0f477495a717daf3`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261124_175224/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261124_175224/run.yaml`
- Timestamp: `2026-11-24T17:52:24.640647+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261124_175224`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261124_175224/cpu_ops_t1_20261124_175224.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 15.406 ± 0.070 | 11.195 ± 0.238 | 9.907 ± 0.147 | 7.688 ± 0.070 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 80.615 ± 0.628 | 63.265 ± 0.709 | 65.704 ± 0.386 | 49.623 ± 0.670 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 8.292 ± 0.312 | 11.173 ± 0.169 | 8.321 ± 0.131 | 7.860 ± 0.148 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 47.212 ± 0.588 | 62.515 ± 0.697 | 51.857 ± 0.370 | 50.047 ± 1.235 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 9.307 ± 0.220 | 4.346 ± 0.086 | 12.378 ± 0.283 | 4.660 ± 0.122 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 60.730 ± 1.022 | 29.940 ± 0.263 | 78.713 ± 1.411 | 30.270 ± 0.291 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 8.264 ± 0.036 | 4.562 ± 0.102 | 5.302 ± 0.160 | 4.377 ± 0.032 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 57.499 ± 0.302 | 31.941 ± 0.345 | 42.740 ± 1.269 | 30.258 ± 0.913 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 12.921 ± 0.157 | 7.293 ± 0.103 | 7.431 ± 0.384 | 6.245 ± 0.081 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 85.779 ± 1.265 | 49.284 ± 0.138 | 51.906 ± 1.080 | 41.626 ± 0.224 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 9.971 ± 0.133 | 7.203 ± 0.044 | 7.081 ± 0.164 | 6.030 ± 0.057 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 71.252 ± 1.996 | 48.893 ± 0.386 | 51.349 ± 2.711 | 46.079 ± 0.944 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.179 ± 0.022 | 0.993 ± 0.012 | 1.110 ± 0.021 | 0.629 ± 0.006 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 5.939 ± 0.099 | 5.694 ± 0.078 | 5.552 ± 0.250 | 3.936 ± 0.102 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.651 ± 0.007 | 0.997 ± 0.007 | 1.583 ± 0.032 | 0.633 ± 0.010 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 3.020 ± 0.079 | 5.519 ± 0.063 | 9.241 ± 0.820 | 3.907 ± 0.062 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 27.167 ± 0.073 | 20.196 ± 0.358 | 20.050 ± 0.052 | 13.808 ± 0.115 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 153.664 ± 2.162 | 115.866 ± 1.776 | 135.070 ± 1.196 | 97.333 ± 1.702 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 14.203 ± 0.078 | 20.099 ± 0.221 | 15.558 ± 0.919 | 13.478 ± 0.791 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 82.414 ± 1.047 | 114.690 ± 1.860 | 107.584 ± 0.971 | 103.130 ± 0.683 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.213 ± 0.000 | 0.024 ± 0.000 | 0.138 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.219 ± 0.004 | 0.026 ± 0.000 | 0.140 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.233 ± 0.006 | 0.035 ± 0.000 | 0.143 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.065 ± 0.002 | 0.022 ± 0.001 | 0.128 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.070 ± 0.001 | 0.025 ± 0.000 | 0.134 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.075 ± 0.001 | 0.034 ± 0.001 | 0.135 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.628 ± 0.017 | 0.029 ± 0.000 | 0.203 ± 0.009 | 0.010 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.634 ± 0.010 | 0.029 ± 0.000 | 0.198 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.627 ± 0.006 | 0.032 ± 0.001 | 0.205 ± 0.007 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.140 ± 0.002 | 0.026 ± 0.000 | 0.183 ± 0.003 | 0.010 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.146 ± 0.001 | 0.026 ± 0.000 | 0.184 ± 0.007 | 0.011 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.147 ± 0.001 | 0.030 ± 0.000 | 0.189 ± 0.005 | 0.016 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.828 ± 0.007 | 0.037 ± 0.000 | 0.156 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.870 ± 0.016 | 0.039 ± 0.000 | 0.156 ± 0.010 | 0.011 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.855 ± 0.010 | 0.044 ± 0.001 | 0.154 ± 0.005 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.139 ± 0.004 | 0.037 ± 0.001 | 0.178 ± 0.005 | 0.010 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.142 ± 0.001 | 0.039 ± 0.000 | 0.178 ± 0.005 | 0.011 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.146 ± 0.004 | 0.044 ± 0.000 | 0.182 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.157 ± 0.002 | 0.023 ± 0.000 | 0.300 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.159 ± 0.003 | 0.023 ± 0.000 | 0.306 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.165 ± 0.002 | 0.025 ± 0.000 | 0.307 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.069 ± 0.000 | 0.023 ± 0.000 | 0.166 ± 0.004 | 0.011 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.071 ± 0.002 | 0.023 ± 0.000 | 0.168 ± 0.007 | 0.011 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.075 ± 0.001 | 0.024 ± 0.000 | 0.170 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.177 ± 0.006 | 0.016 ± 0.000 | 0.200 ± 0.007 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.187 ± 0.003 | 0.020 ± 0.000 | 0.206 ± 0.006 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.205 ± 0.005 | 0.031 ± 0.001 | 0.214 ± 0.003 | 0.017 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.058 ± 0.000 | 0.015 ± 0.000 | 0.150 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.062 ± 0.003 | 0.019 ± 0.000 | 0.163 ± 0.006 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.072 ± 0.002 | 0.031 ± 0.001 | 0.161 ± 0.006 | 0.017 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261124_182830/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261124_182830/run.yaml`
- Timestamp: `2026-11-24T18:28:30.406333+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261124_182830`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261124_182830/cpu_ops_t4_20261124_182830.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 17.139 ± 0.087 | 13.027 ± 0.186 | 6.902 ± 0.437 | 9.955 ± 1.173 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 53.424 ± 1.017 | 42.798 ± 0.150 | 40.973 ± 1.096 | 46.288 ± 1.800 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 8.954 ± 0.123 | 12.969 ± 0.272 | 5.971 ± 0.142 | 9.632 ± 1.616 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 30.274 ± 0.203 | 42.860 ± 0.761 | 29.389 ± 0.468 | 50.127 ± 4.730 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 6.682 ± 0.198 | 3.494 ± 0.042 | 7.832 ± 0.172 | 6.338 ± 1.265 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 35.804 ± 0.426 | 18.865 ± 0.363 | 45.012 ± 0.593 | 32.297 ± 2.863 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 5.277 ± 0.101 | 3.580 ± 0.022 | 3.517 ± 0.070 | 6.504 ± 0.063 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 32.350 ± 0.305 | 19.785 ± 0.146 | 23.236 ± 1.032 | 29.864 ± 6.303 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 8.816 ± 0.243 | 4.937 ± 0.078 | 4.591 ± 0.064 | 8.097 ± 1.895 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 48.733 ± 1.607 | 28.846 ± 0.731 | 32.126 ± 1.110 | 52.026 ± 13.698 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 6.131 ± 0.120 | 4.969 ± 0.016 | 5.176 ± 0.216 | 9.604 ± 2.306 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 38.684 ± 0.727 | 29.470 ± 0.383 | 31.023 ± 0.508 | 52.464 ± 21.578 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.235 ± 0.108 | 0.954 ± 0.012 | 1.727 ± 1.535 | 0.804 ± 0.042 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 4.697 ± 0.092 | 4.122 ± 0.089 | 2.808 ± 0.024 | 3.366 ± 0.054 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.658 ± 0.011 | 1.006 ± 0.047 | 1.432 ± 0.066 | 0.901 ± 0.146 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 2.362 ± 0.046 | 4.134 ± 0.142 | 4.507 ± 0.138 | 3.539 ± 0.019 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 30.490 ± 0.275 | 21.128 ± 0.147 | 15.994 ± 0.328 | 15.891 ± 1.315 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 100.848 ± 2.253 | 79.919 ± 1.969 | 90.797 ± 5.254 | 97.875 ± 4.602 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 15.354 ± 0.082 | 21.092 ± 0.230 | 14.433 ± 0.173 | 16.972 ± 2.952 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 52.753 ± 0.565 | 81.127 ± 2.602 | 80.205 ± 3.685 | 95.060 ± 6.728 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.225 ± 0.003 | 0.034 ± 0.001 | 0.134 ± 0.006 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.237 ± 0.011 | 0.038 ± 0.001 | 0.134 ± 0.004 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.253 ± 0.001 | 0.045 ± 0.001 | 0.146 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.078 ± 0.001 | 0.032 ± 0.000 | 0.124 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.084 ± 0.001 | 0.036 ± 0.007 | 0.131 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.091 ± 0.001 | 0.043 ± 0.000 | 0.138 ± 0.002 | 0.015 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.661 ± 0.007 | 0.042 ± 0.000 | 0.277 ± 0.005 | 0.010 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.678 ± 0.011 | 0.042 ± 0.000 | 0.363 ± 0.134 | 0.011 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.702 ± 0.004 | 0.045 ± 0.001 | 0.275 ± 0.008 | 0.019 ± 0.005 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.180 ± 0.001 | 0.037 ± 0.001 | 0.231 ± 0.004 | 0.009 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.184 ± 0.005 | 0.038 ± 0.000 | 0.233 ± 0.025 | 0.011 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.188 ± 0.001 | 0.040 ± 0.000 | 0.235 ± 0.004 | 0.017 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.887 ± 0.002 | 0.056 ± 0.002 | 0.202 ± 0.059 | 0.010 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.885 ± 0.011 | 0.059 ± 0.001 | 0.193 ± 0.007 | 0.011 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.898 ± 0.028 | 0.062 ± 0.001 | 0.202 ± 0.160 | 0.018 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.176 ± 0.025 | 0.055 ± 0.001 | 0.212 ± 0.005 | 0.009 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.172 ± 0.003 | 0.057 ± 0.001 | 0.215 ± 0.009 | 0.011 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.185 ± 0.005 | 0.060 ± 0.000 | 0.222 ± 0.008 | 0.015 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.169 ± 0.002 | 0.033 ± 0.000 | 0.298 ± 0.006 | 0.010 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.177 ± 0.005 | 0.033 ± 0.000 | 0.301 ± 0.007 | 0.010 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.183 ± 0.007 | 0.034 ± 0.000 | 0.298 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.081 ± 0.004 | 0.032 ± 0.001 | 0.165 ± 0.005 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.084 ± 0.002 | 0.032 ± 0.000 | 0.164 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.089 ± 0.001 | 0.033 ± 0.001 | 0.164 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.186 ± 0.002 | 0.023 ± 0.000 | 0.197 ± 0.005 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.198 ± 0.003 | 0.027 ± 0.001 | 0.202 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.214 ± 0.001 | 0.038 ± 0.000 | 0.213 ± 0.003 | 0.021 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.069 ± 0.002 | 0.021 ± 0.001 | 0.144 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.073 ± 0.002 | 0.025 ± 0.000 | 0.150 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.077 ± 0.001 | 0.036 ± 0.000 | 0.162 ± 0.004 | 0.021 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `89c207f1ca97cde3ae0f31586747a039ca4e5cc6`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261114_172310/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261114_172310/run.yaml`
- Timestamp: `2026-11-14T17:23:10.539309+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261114_172310`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

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

- tenferro-rs features: `blas-accelerate`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20261114_172310/cpu_ops_t1_20261114_172310.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 12.255 ± 0.498 | 9.389 ± 0.625 | 6.449 ± 1.293 | 7.259 ± 0.529 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 90.071 ± 18.023 | 54.490 ± 2.465 | 32.597 ± 0.772 | 58.667 ± 14.537 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 6.035 ± 0.263 | 9.138 ± 0.212 | 6.241 ± 0.919 | 7.106 ± 0.530 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 45.810 ± 9.098 | 53.964 ± 3.711 | 30.992 ± 0.658 | 43.401 ± 4.135 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 3.608 ± 0.454 | 1.416 ± 0.198 | 3.063 ± 0.271 | 4.084 ± 0.894 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 32.337 ± 7.579 | 10.192 ± 0.969 | 19.139 ± 1.295 | 20.580 ± 1.086 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 2.869 ± 0.207 | 1.648 ± 0.093 | 1.988 ± 0.833 | 2.755 ± 0.464 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 24.249 ± 11.935 | 9.946 ± 1.339 | 10.633 ± 0.850 | 21.258 ± 1.295 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 6.447 ± 0.784 | 4.793 ± 0.801 | 3.337 ± 0.498 | 4.954 ± 0.419 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 35.756 ± 1.531 | 25.999 ± 1.314 | 19.798 ± 1.755 | 35.518 ± 6.476 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 4.337 ± 0.343 | 4.303 ± 0.499 | 3.010 ± 0.310 | 5.215 ± 0.698 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 25.520 ± 1.860 | 26.584 ± 2.358 | 17.222 ± 0.876 | 41.845 ± 13.957 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.881 ± 0.134 | 0.609 ± 0.090 | 0.543 ± 0.136 | 0.403 ± 0.284 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 4.069 ± 0.619 | 3.103 ± 0.507 | 2.551 ± 0.616 | 2.147 ± 0.816 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.462 ± 0.027 | 0.583 ± 0.212 | 0.624 ± 0.061 | 0.382 ± 0.110 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 2.273 ± 0.369 | 2.894 ± 0.131 | 4.206 ± 1.541 | 1.638 ± 0.336 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 21.577 ± 0.428 | 16.691 ± 0.704 | 11.782 ± 1.997 | 11.528 ± 1.537 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 124.030 ± 2.993 | 101.907 ± 8.137 | 76.212 ± 5.742 | 79.624 ± 7.114 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 10.572 ± 0.466 | 17.381 ± 0.801 | 10.149 ± 1.144 | 11.054 ± 0.299 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 63.164 ± 1.205 | 107.224 ± 5.688 | 58.541 ± 6.449 | 80.074 ± 5.570 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.122 ± 0.008 | 0.020 ± 0.003 | 0.077 ± 0.015 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.186 ± 0.037 | 0.026 ± 0.033 | 0.092 ± 0.021 | 0.011 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.161 ± 0.080 | 0.034 ± 0.002 | 0.121 ± 0.064 | 0.010 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.042 ± 0.003 | 0.018 ± 0.006 | 0.069 ± 0.046 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.065 ± 0.014 | 0.029 ± 0.012 | 0.074 ± 0.012 | 0.012 ± 0.003 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.047 ± 0.024 | 0.028 ± 0.002 | 0.121 ± 0.089 | 0.010 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.577 ± 0.256 | 0.023 ± 0.003 | 0.127 ± 0.038 | 0.007 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.482 ± 0.063 | 0.023 ± 0.007 | 0.121 ± 0.022 | 0.009 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.604 ± 0.183 | 0.023 ± 0.003 | 0.283 ± 0.267 | 0.008 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.111 ± 0.009 | 0.022 ± 0.003 | 0.104 ± 0.015 | 0.009 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.129 ± 0.028 | 0.020 ± 0.007 | 0.085 ± 0.022 | 0.010 ± 0.004 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.147 ± 0.113 | 0.022 ± 0.001 | 0.230 ± 0.120 | 0.020 ± 0.016 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.749 ± 0.293 | 0.034 ± 0.003 | 0.091 ± 0.009 | 0.008 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 1.015 ± 0.389 | 0.036 ± 0.003 | 0.095 ± 0.022 | 0.009 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.909 ± 0.200 | 0.032 ± 0.002 | 0.147 ± 0.033 | 0.006 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.081 ± 0.014 | 0.034 ± 0.007 | 0.094 ± 0.019 | 0.009 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.123 ± 0.008 | 0.055 ± 0.007 | 0.120 ± 0.030 | 0.010 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.122 ± 0.007 | 0.042 ± 0.009 | 0.157 ± 0.039 | 0.008 ± 0.003 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.107 ± 0.013 | 0.017 ± 0.002 | 0.197 ± 0.024 | 0.009 ± 0.002 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.125 ± 0.007 | 0.019 ± 0.005 | 0.216 ± 0.030 | 0.009 ± 0.002 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.119 ± 0.030 | 0.017 ± 0.001 | 0.226 ± 0.250 | 0.010 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.045 ± 0.002 | 0.017 ± 0.002 | 0.089 ± 0.018 | 0.008 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.053 ± 0.017 | 0.016 ± 0.005 | 0.099 ± 0.008 | 0.010 ± 0.012 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.064 ± 0.013 | 0.017 ± 0.000 | 0.125 ± 0.040 | 0.010 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.110 ± 0.021 | 0.014 ± 0.001 | 0.135 ± 0.036 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.146 ± 0.164 | 0.025 ± 0.013 | 0.130 ± 0.030 | 0.012 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.189 ± 0.044 | 0.031 ± 0.009 | 0.181 ± 0.083 | 0.019 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.040 ± 0.003 | 0.013 ± 0.001 | 0.070 ± 0.011 | 0.011 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.057 ± 0.019 | 0.019 ± 0.003 | 0.090 ± 0.026 | 0.013 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.062 ± 0.018 | 0.035 ± 0.017 | 0.152 ± 0.076 | 0.019 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261114_175950/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261114_175950/run.yaml`
- Timestamp: `2026-11-14T17:59:50.064820+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261114_175950`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

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

- tenferro-rs features: `blas-accelerate`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20261114_175950/cpu_ops_t4_20261114_175950.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 11.767 ± 3.416 | 8.651 ± 0.297 | 5.903 ± 0.678 | 6.838 ± 0.201 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 59.290 ± 3.947 | 50.505 ± 0.605 | 44.096 ± 6.250 | 45.592 ± 5.041 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 6.353 ± 0.562 | 8.714 ± 0.421 | 5.586 ± 0.865 | 6.950 ± 0.637 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 32.500 ± 0.367 | 50.913 ± 0.661 | 31.901 ± 3.922 | 50.943 ± 18.774 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 3.340 ± 0.719 | 1.406 ± 0.278 | 2.788 ± 0.818 | 2.291 ± 0.221 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 19.335 ± 1.378 | 9.351 ± 0.804 | 17.771 ± 1.374 | 13.566 ± 1.611 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 2.732 ± 0.342 | 1.493 ± 0.026 | 2.191 ± 0.395 | 2.357 ± 0.434 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 18.097 ± 1.061 | 10.648 ± 1.103 | 10.569 ± 0.921 | 13.625 ± 0.692 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 7.098 ± 1.303 | 4.149 ± 0.516 | 3.742 ± 0.477 | 4.087 ± 0.851 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 38.332 ± 2.366 | 26.342 ± 1.138 | 23.745 ± 6.467 | 26.454 ± 4.993 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 5.150 ± 0.577 | 3.800 ± 0.498 | 3.907 ± 0.831 | 4.382 ± 0.716 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 27.281 ± 2.513 | 24.487 ± 5.301 | 22.584 ± 9.027 | 32.966 ± 5.751 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.869 ± 0.031 | 0.511 ± 0.159 | 0.600 ± 0.087 | 0.397 ± 0.110 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 4.154 ± 0.910 | 3.435 ± 0.254 | 2.113 ± 0.144 | 2.339 ± 0.599 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.761 ± 0.338 | 0.505 ± 0.114 | 0.690 ± 0.108 | 0.454 ± 0.209 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 2.211 ± 0.350 | 3.162 ± 0.201 | 3.746 ± 0.749 | 1.983 ± 0.655 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 19.336 ± 1.409 | 16.202 ± 1.204 | 13.536 ± 1.350 | 11.838 ± 0.774 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 129.219 ± 19.205 | 109.728 ± 7.096 | 61.964 ± 4.448 | 63.033 ± 2.446 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 10.575 ± 0.619 | 15.150 ± 1.615 | 11.499 ± 0.645 | 12.054 ± 1.839 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 68.931 ± 5.624 | 123.181 ± 18.203 | 70.455 ± 7.621 | 69.345 ± 4.762 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.127 ± 0.023 | 0.023 ± 0.003 | 0.076 ± 0.021 | 0.009 ± 0.003 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.135 ± 0.025 | 0.028 ± 0.004 | 0.072 ± 0.016 | 0.013 ± 0.003 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.145 ± 0.017 | 0.036 ± 0.001 | 0.081 ± 0.016 | 0.016 ± 0.004 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.046 ± 0.005 | 0.023 ± 0.005 | 0.069 ± 0.013 | 0.009 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.053 ± 0.006 | 0.028 ± 0.002 | 0.070 ± 0.018 | 0.011 ± 0.003 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.062 ± 0.004 | 0.033 ± 0.002 | 0.077 ± 0.008 | 0.022 ± 0.012 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.448 ± 0.051 | 0.031 ± 0.005 | 0.384 ± 0.104 | 0.009 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.476 ± 0.059 | 0.029 ± 0.004 | 0.319 ± 0.071 | 0.011 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.451 ± 0.046 | 0.030 ± 0.005 | 0.285 ± 0.080 | 0.017 ± 0.012 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.125 ± 0.006 | 0.026 ± 0.002 | 0.286 ± 0.050 | 0.008 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.126 ± 0.017 | 0.027 ± 0.003 | 0.231 ± 0.065 | 0.009 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.133 ± 0.010 | 0.031 ± 0.002 | 0.247 ± 0.026 | 0.014 ± 0.004 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.588 ± 0.046 | 0.046 ± 0.010 | 0.148 ± 0.009 | 0.009 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.585 ± 0.100 | 0.042 ± 0.007 | 0.167 ± 0.071 | 0.011 ± 0.003 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.662 ± 0.061 | 0.044 ± 0.007 | 0.170 ± 0.020 | 0.022 ± 0.005 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.108 ± 0.027 | 0.043 ± 0.004 | 0.164 ± 0.028 | 0.008 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.110 ± 0.017 | 0.049 ± 0.008 | 0.165 ± 0.006 | 0.011 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.128 ± 0.022 | 0.051 ± 0.010 | 0.173 ± 0.033 | 0.016 ± 0.018 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.092 ± 0.020 | 0.027 ± 0.005 | 0.201 ± 0.048 | 0.008 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.096 ± 0.018 | 0.025 ± 0.002 | 0.175 ± 0.024 | 0.010 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.091 ± 0.015 | 0.023 ± 0.004 | 0.180 ± 0.112 | 0.008 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.052 ± 0.006 | 0.023 ± 0.010 | 0.091 ± 0.027 | 0.010 ± 0.003 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.051 ± 0.011 | 0.024 ± 0.002 | 0.079 ± 0.008 | 0.010 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.049 ± 0.006 | 0.023 ± 0.002 | 0.078 ± 0.023 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.106 ± 0.049 | 0.019 ± 0.001 | 0.113 ± 0.020 | 0.011 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.114 ± 0.018 | 0.024 ± 0.001 | 0.106 ± 0.017 | 0.014 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.140 ± 0.011 | 0.036 ± 0.001 | 0.122 ± 0.013 | 0.032 ± 0.024 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.041 ± 0.009 | 0.017 ± 0.002 | 0.073 ± 0.018 | 0.011 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.062 ± 0.029 | 0.022 ± 0.001 | 0.074 ± 0.020 | 0.015 ± 0.003 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.056 ± 0.011 | 0.033 ± 0.003 | 0.076 ± 0.011 | 0.019 ± 0.011 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `4edf89377c13d5c8a287dd562b529de6bb85f4dc`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261112_172325/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261112_172325/run.yaml`
- Timestamp: `2026-11-12T17:23:25.841634+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261112_172325`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261112_172325/cpu_ops_t1_20261112_172325.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 11.230 ± 1.026 | 8.206 ± 0.287 | 5.907 ± 2.732 | 6.859 ± 0.076 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 55.807 ± 0.972 | 45.645 ± 1.465 | 31.191 ± 1.268 | 45.815 ± 12.957 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 5.877 ± 0.852 | 8.119 ± 0.434 | 5.705 ± 0.338 | 7.047 ± 3.590 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 29.589 ± 0.990 | 44.321 ± 0.531 | 29.741 ± 5.527 | 46.658 ± 11.449 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 2.984 ± 0.487 | 1.263 ± 0.210 | 2.328 ± 0.764 | 3.255 ± 1.594 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 16.492 ± 0.493 | 7.500 ± 0.309 | 18.449 ± 3.239 | 20.307 ± 2.299 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 2.620 ± 1.410 | 1.329 ± 0.074 | 1.886 ± 1.830 | 2.579 ± 4.864 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 14.974 ± 0.228 | 7.949 ± 0.191 | 8.076 ± 2.214 | 23.284 ± 25.756 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 5.666 ± 1.654 | 3.698 ± 0.041 | 3.362 ± 0.869 | 5.131 ± 0.505 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 29.483 ± 3.215 | 19.816 ± 0.776 | 17.616 ± 6.010 | 26.095 ± 2.536 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 3.586 ± 0.202 | 3.552 ± 0.064 | 3.514 ± 0.907 | 5.719 ± 2.358 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 21.859 ± 1.648 | 19.870 ± 0.619 | 16.347 ± 5.808 | 31.911 ± 3.592 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.695 ± 0.143 | 0.582 ± 0.187 | 0.530 ± 0.113 | 0.351 ± 0.296 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 3.153 ± 0.164 | 2.474 ± 0.029 | 2.149 ± 2.251 | 2.893 ± 3.352 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.371 ± 0.012 | 0.481 ± 0.024 | 0.810 ± 0.393 | 0.315 ± 0.036 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 1.657 ± 0.112 | 2.386 ± 0.057 | 3.045 ± 0.074 | 2.634 ± 1.653 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 19.888 ± 3.333 | 13.722 ± 0.739 | 11.896 ± 2.239 | 11.223 ± 0.786 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 121.425 ± 14.129 | 89.426 ± 3.868 | 62.705 ± 3.398 | 64.096 ± 9.845 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 9.556 ± 0.130 | 13.817 ± 0.220 | 10.800 ± 2.689 | 10.778 ± 0.674 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 58.620 ± 1.384 | 80.038 ± 2.170 | 58.338 ± 3.612 | 59.276 ± 5.632 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.148 ± 0.124 | 0.017 ± 0.001 | 0.088 ± 0.016 | 0.010 ± 0.004 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.140 ± 0.089 | 0.017 ± 0.000 | 0.070 ± 0.005 | 0.010 ± 0.006 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.122 ± 0.005 | 0.026 ± 0.002 | 0.079 ± 0.009 | 0.011 ± 0.005 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.034 ± 0.067 | 0.018 ± 0.001 | 0.079 ± 0.053 | 0.009 ± 0.005 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.035 ± 0.002 | 0.017 ± 0.000 | 0.065 ± 0.017 | 0.017 ± 0.009 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.044 ± 0.003 | 0.023 ± 0.001 | 0.120 ± 0.135 | 0.014 ± 0.003 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.744 ± 0.191 | 0.020 ± 0.001 | 0.122 ± 0.012 | 0.008 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.359 ± 0.017 | 0.018 ± 0.000 | 0.105 ± 0.027 | 0.010 ± 0.006 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.372 ± 0.050 | 0.018 ± 0.000 | 0.128 ± 0.103 | 0.007 ± 0.003 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.080 ± 0.009 | 0.020 ± 0.003 | 0.087 ± 0.039 | 0.011 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.111 ± 0.046 | 0.016 ± 0.000 | 0.081 ± 0.003 | 0.010 ± 0.003 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.087 ± 0.004 | 0.021 ± 0.000 | 0.101 ± 0.052 | 0.007 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.877 ± 0.497 | 0.031 ± 0.005 | 0.080 ± 0.010 | 0.010 ± 0.005 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.490 ± 0.054 | 0.027 ± 0.001 | 0.081 ± 0.004 | 0.021 ± 0.006 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.529 ± 0.060 | 0.029 ± 0.008 | 0.098 ± 0.109 | 0.017 ± 0.014 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.160 ± 0.026 | 0.038 ± 0.031 | 0.098 ± 0.052 | 0.009 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.077 ± 0.036 | 0.027 ± 0.003 | 0.084 ± 0.042 | 0.014 ± 0.008 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.080 ± 0.040 | 0.036 ± 0.005 | 0.108 ± 0.033 | 0.011 ± 0.006 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.084 ± 0.059 | 0.015 ± 0.001 | 0.193 ± 0.076 | 0.010 ± 0.007 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.078 ± 0.018 | 0.014 ± 0.001 | 0.164 ± 0.017 | 0.010 ± 0.003 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.073 ± 0.004 | 0.014 ± 0.000 | 0.235 ± 0.191 | 0.010 ± 0.007 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.041 ± 0.014 | 0.016 ± 0.001 | 0.154 ± 0.083 | 0.011 ± 0.003 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.040 ± 0.017 | 0.014 ± 0.002 | 0.078 ± 0.085 | 0.010 ± 0.003 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.036 ± 0.002 | 0.014 ± 0.000 | 0.096 ± 0.159 | 0.012 ± 0.007 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.106 ± 0.129 | 0.019 ± 0.005 | 0.092 ± 0.007 | 0.012 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.184 ± 0.167 | 0.014 ± 0.000 | 0.095 ± 0.002 | 0.028 ± 0.024 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.126 ± 0.017 | 0.029 ± 0.001 | 0.157 ± 0.058 | 0.019 ± 0.005 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.043 ± 0.027 | 0.012 ± 0.008 | 0.077 ± 0.042 | 0.012 ± 0.004 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.041 ± 0.028 | 0.013 ± 0.001 | 0.067 ± 0.008 | 0.023 ± 0.009 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.044 ± 0.002 | 0.025 ± 0.001 | 0.111 ± 0.112 | 0.018 ± 0.002 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261112_175944/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261112_175944/run.yaml`
- Timestamp: `2026-11-12T17:59:44.081935+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261112_175944`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261112_175944/cpu_ops_t4_20261112_175944.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 10.513 ± 0.566 | 10.205 ± 1.500 | 6.207 ± 0.353 | 6.342 ± 0.123 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 65.440 ± 5.947 | 56.045 ± 6.331 | 33.850 ± 3.121 | 35.953 ± 0.825 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 5.574 ± 0.192 | 10.492 ± 0.760 | 5.132 ± 0.089 | 7.223 ± 1.269 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 36.763 ± 3.035 | 60.552 ± 4.967 | 31.693 ± 1.919 | 36.267 ± 1.953 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 3.378 ± 0.190 | 1.373 ± 0.503 | 3.404 ± 0.361 | 2.535 ± 0.078 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 19.338 ± 0.765 | 9.310 ± 1.691 | 17.700 ± 4.374 | 14.204 ± 0.291 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 2.479 ± 0.094 | 1.307 ± 0.090 | 1.798 ± 0.550 | 2.396 ± 0.704 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 18.017 ± 3.101 | 10.187 ± 2.140 | 8.434 ± 1.116 | 14.698 ± 8.774 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 5.837 ± 0.408 | 3.982 ± 1.147 | 3.475 ± 1.179 | 6.077 ± 4.039 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 28.479 ± 4.686 | 18.373 ± 0.471 | 14.859 ± 3.057 | 22.994 ± 1.447 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 3.391 ± 0.251 | 3.670 ± 0.569 | 6.614 ± 2.884 | 4.065 ± 0.170 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 24.332 ± 0.688 | 19.333 ± 4.811 | 18.480 ± 1.279 | 23.025 ± 0.658 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.670 ± 0.102 | 0.527 ± 0.011 | 0.565 ± 0.119 | 0.347 ± 0.005 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 3.776 ± 0.352 | 3.312 ± 0.660 | 2.079 ± 0.697 | 3.640 ± 0.797 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.377 ± 0.019 | 0.553 ± 0.079 | 0.663 ± 0.088 | 0.391 ± 0.028 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 2.104 ± 0.127 | 3.653 ± 0.403 | 2.874 ± 0.530 | 1.701 ± 0.012 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 18.409 ± 1.342 | 15.046 ± 3.474 | 12.862 ± 4.638 | 13.110 ± 4.461 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 102.358 ± 2.581 | 98.124 ± 4.660 | 59.568 ± 18.021 | 62.838 ± 2.814 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 10.552 ± 2.555 | 13.269 ± 4.305 | 12.273 ± 3.700 | 14.328 ± 4.621 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 60.920 ± 13.917 | 110.261 ± 7.983 | 54.991 ± 2.645 | 63.133 ± 2.049 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.145 ± 0.076 | 0.039 ± 0.013 | 0.074 ± 0.007 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.129 ± 0.065 | 0.031 ± 0.006 | 0.064 ± 0.004 | 0.016 ± 0.007 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.139 ± 0.007 | 0.035 ± 0.000 | 0.092 ± 0.021 | 0.022 ± 0.008 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.042 ± 0.001 | 0.033 ± 0.011 | 0.058 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.047 ± 0.023 | 0.040 ± 0.005 | 0.057 ± 0.005 | 0.010 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.062 ± 0.022 | 0.030 ± 0.001 | 0.066 ± 0.002 | 0.026 ± 0.010 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.378 ± 0.006 | 0.031 ± 0.011 | 0.293 ± 0.018 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.515 ± 0.149 | 0.033 ± 0.009 | 0.271 ± 0.028 | 0.010 ± 0.004 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.387 ± 0.009 | 0.026 ± 0.002 | 0.264 ± 0.037 | 0.016 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.108 ± 0.024 | 0.037 ± 0.016 | 0.220 ± 0.011 | 0.007 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.109 ± 0.009 | 0.027 ± 0.012 | 0.226 ± 0.045 | 0.008 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.118 ± 0.002 | 0.027 ± 0.002 | 0.221 ± 0.025 | 0.013 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.553 ± 0.044 | 0.043 ± 0.004 | 0.158 ± 0.016 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.685 ± 0.311 | 0.052 ± 0.021 | 0.222 ± 0.068 | 0.016 ± 0.010 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.522 ± 0.047 | 0.043 ± 0.002 | 0.171 ± 0.022 | 0.014 ± 0.003 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.100 ± 0.113 | 0.046 ± 0.036 | 0.169 ± 0.020 | 0.007 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.106 ± 0.012 | 0.060 ± 0.010 | 0.203 ± 0.024 | 0.014 ± 0.023 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.116 ± 0.036 | 0.050 ± 0.001 | 0.202 ± 0.031 | 0.015 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.078 ± 0.003 | 0.029 ± 0.002 | 0.151 ± 0.006 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.082 ± 0.001 | 0.028 ± 0.011 | 0.144 ± 0.001 | 0.007 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.104 ± 0.083 | 0.023 ± 0.001 | 0.149 ± 0.006 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.044 ± 0.009 | 0.034 ± 0.007 | 0.076 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.046 ± 0.002 | 0.023 ± 0.000 | 0.081 ± 0.004 | 0.008 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.050 ± 0.002 | 0.025 ± 0.001 | 0.072 ± 0.002 | 0.008 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.091 ± 0.020 | 0.016 ± 0.000 | 0.114 ± 0.028 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.093 ± 0.003 | 0.022 ± 0.003 | 0.102 ± 0.016 | 0.020 ± 0.016 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.123 ± 0.003 | 0.037 ± 0.003 | 0.117 ± 0.008 | 0.045 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.038 ± 0.001 | 0.016 ± 0.001 | 0.067 ± 0.007 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.053 ± 0.009 | 0.022 ± 0.003 | 0.060 ± 0.001 | 0.014 ± 0.007 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.080 ± 0.037 | 0.032 ± 0.002 | 0.077 ± 0.002 | 0.046 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

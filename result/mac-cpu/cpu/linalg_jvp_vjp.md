# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `9e383937df4e365bd68bf5b16dbfcaaa269bcc82`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261115_172247/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261115_172247/run.yaml`
- Timestamp: `2026-11-15T17:22:47.463268+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261115_172247`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261115_172247/cpu_ops_t1_20261115_172247.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 10.124 ± 0.516 | 12.458 ± 1.576 | 5.582 ± 0.297 | 6.789 ± 0.097 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 78.209 ± 5.461 | 64.058 ± 7.768 | 46.227 ± 8.496 | 47.877 ± 0.945 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 5.155 ± 0.164 | 12.051 ± 3.535 | 6.289 ± 0.518 | 6.735 ± 0.156 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 39.174 ± 5.077 | 69.384 ± 21.636 | 38.816 ± 5.809 | 45.753 ± 3.697 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 3.503 ± 0.426 | 2.071 ± 0.228 | 3.131 ± 0.682 | 3.106 ± 0.833 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 29.891 ± 3.410 | 14.259 ± 7.593 | 20.183 ± 5.934 | 20.697 ± 0.986 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 2.984 ± 0.356 | 2.212 ± 0.736 | 2.024 ± 1.496 | 2.892 ± 0.210 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 22.881 ± 3.819 | 16.384 ± 6.610 | 10.511 ± 2.267 | 19.954 ± 0.760 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 6.325 ± 0.793 | 6.386 ± 0.941 | 2.934 ± 0.431 | 4.411 ± 0.157 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 32.019 ± 7.400 | 22.939 ± 1.314 | 27.099 ± 4.466 | 35.236 ± 1.651 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 3.378 ± 1.386 | 5.543 ± 1.118 | 3.452 ± 0.163 | 4.620 ± 1.311 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 40.596 ± 9.270 | 27.249 ± 8.068 | 22.620 ± 4.618 | 33.917 ± 4.371 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.664 ± 0.075 | 0.855 ± 0.262 | 0.594 ± 0.098 | 0.321 ± 0.066 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 4.172 ± 3.220 | 4.127 ± 4.815 | 3.842 ± 2.272 | 1.798 ± 0.102 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.359 ± 0.039 | 0.931 ± 0.353 | 0.661 ± 0.336 | 0.353 ± 0.071 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 2.744 ± 1.784 | 5.009 ± 1.409 | 3.231 ± 0.115 | 1.568 ± 0.069 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 21.541 ± 2.086 | 19.751 ± 4.153 | 13.759 ± 4.683 | 12.098 ± 0.558 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 163.039 ± 6.594 | 135.279 ± 6.563 | 94.224 ± 21.357 | 75.602 ± 5.973 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 10.232 ± 0.471 | 20.122 ± 2.902 | 13.471 ± 3.516 | 11.129 ± 0.196 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 72.936 ± 22.795 | 116.524 ± 22.008 | 76.338 ± 13.087 | 78.520 ± 5.762 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.113 ± 0.008 | 0.016 ± 0.017 | 0.126 ± 0.066 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.116 ± 0.013 | 0.019 ± 0.001 | 0.087 ± 0.005 | 0.014 ± 0.011 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.139 ± 0.019 | 0.028 ± 0.000 | 0.082 ± 0.006 | 0.010 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.037 ± 0.002 | 0.017 ± 0.001 | 0.080 ± 0.077 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.036 ± 0.001 | 0.020 ± 0.002 | 0.078 ± 0.031 | 0.013 ± 0.011 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.052 ± 0.002 | 0.028 ± 0.010 | 0.073 ± 0.002 | 0.011 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.393 ± 0.061 | 0.020 ± 0.008 | 0.111 ± 0.091 | 0.008 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.359 ± 0.010 | 0.021 ± 0.003 | 0.114 ± 0.029 | 0.014 ± 0.008 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.437 ± 0.053 | 0.025 ± 0.019 | 0.127 ± 0.022 | 0.009 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.097 ± 0.026 | 0.019 ± 0.020 | 0.091 ± 0.042 | 0.008 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.082 ± 0.004 | 0.018 ± 0.001 | 0.103 ± 0.021 | 0.012 ± 0.005 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.103 ± 0.008 | 0.028 ± 0.008 | 0.099 ± 0.004 | 0.008 ± 0.014 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.598 ± 0.516 | 0.037 ± 0.009 | 0.101 ± 0.025 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.481 ± 0.015 | 0.028 ± 0.018 | 0.085 ± 0.030 | 0.009 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.548 ± 0.030 | 0.039 ± 0.012 | 0.090 ± 0.038 | 0.008 ± 0.004 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.081 ± 0.025 | 0.034 ± 0.006 | 0.080 ± 0.004 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.071 ± 0.004 | 0.031 ± 0.002 | 0.101 ± 0.030 | 0.012 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.083 ± 0.006 | 0.051 ± 0.046 | 0.086 ± 0.001 | 0.008 ± 0.004 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.102 ± 0.078 | 0.016 ± 0.008 | 0.167 ± 0.011 | 0.007 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.088 ± 0.026 | 0.016 ± 0.005 | 0.205 ± 0.054 | 0.012 ± 0.005 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.092 ± 0.011 | 0.029 ± 0.016 | 0.185 ± 0.026 | 0.007 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.074 ± 0.011 | 0.015 ± 0.002 | 0.083 ± 0.042 | 0.009 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.035 ± 0.002 | 0.015 ± 0.001 | 0.097 ± 0.034 | 0.010 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.043 ± 0.010 | 0.024 ± 0.006 | 0.083 ± 0.016 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.106 ± 0.005 | 0.015 ± 0.005 | 0.107 ± 0.008 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.091 ± 0.010 | 0.021 ± 0.009 | 0.150 ± 0.042 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.116 ± 0.004 | 0.035 ± 0.005 | 0.117 ± 0.006 | 0.019 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.039 ± 0.010 | 0.016 ± 0.007 | 0.084 ± 0.054 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.034 ± 0.001 | 0.015 ± 0.004 | 0.081 ± 0.025 | 0.013 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.046 ± 0.009 | 0.030 ± 0.007 | 0.074 ± 0.018 | 0.020 ± 0.004 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261115_175655/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261115_175655/run.yaml`
- Timestamp: `2026-11-15T17:56:55.332197+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261115_175655`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261115_175655/cpu_ops_t4_20261115_175655.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 9.684 ± 0.710 | 8.675 ± 0.324 | 6.290 ± 0.333 | 7.034 ± 5.794 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 74.469 ± 6.712 | 58.884 ± 3.401 | 41.652 ± 8.425 | 41.021 ± 4.385 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 5.535 ± 0.295 | 8.684 ± 0.309 | 5.658 ± 0.404 | 6.634 ± 1.903 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 39.127 ± 7.108 | 65.623 ± 2.654 | 36.932 ± 4.754 | 59.147 ± 10.550 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 2.808 ± 0.115 | 1.289 ± 0.086 | 3.720 ± 3.720 | 4.004 ± 1.025 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 27.799 ± 8.368 | 13.567 ± 1.858 | 19.984 ± 6.542 | 24.414 ± 7.687 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 2.664 ± 0.060 | 1.440 ± 0.089 | 4.715 ± 0.899 | 2.517 ± 0.133 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 16.963 ± 1.511 | 13.229 ± 1.299 | 11.236 ± 0.946 | 24.330 ± 6.137 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 5.976 ± 0.228 | 4.057 ± 0.161 | 3.890 ± 0.298 | 6.077 ± 4.373 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 55.064 ± 5.517 | 27.283 ± 4.716 | 16.327 ± 0.508 | 29.585 ± 4.978 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 3.844 ± 0.235 | 3.999 ± 0.116 | 3.596 ± 0.261 | 4.259 ± 0.551 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 36.501 ± 3.824 | 26.671 ± 1.895 | 33.256 ± 9.861 | 29.758 ± 14.661 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.806 ± 0.088 | 0.566 ± 0.010 | 0.560 ± 0.022 | 0.387 ± 0.021 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 3.416 ± 0.537 | 3.995 ± 0.157 | 2.537 ± 0.622 | 2.213 ± 0.149 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.451 ± 0.043 | 0.543 ± 0.011 | 0.732 ± 0.023 | 0.429 ± 0.021 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.790 ± 0.124 | 3.809 ± 0.923 | 3.731 ± 0.438 | 2.030 ± 0.742 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 19.580 ± 0.423 | 15.215 ± 0.766 | 12.385 ± 2.438 | 11.268 ± 1.439 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 137.659 ± 10.773 | 87.715 ± 2.140 | 65.683 ± 12.400 | 70.142 ± 8.747 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 9.856 ± 0.107 | 15.486 ± 1.201 | 10.711 ± 1.192 | 10.507 ± 0.018 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 79.588 ± 13.876 | 90.287 ± 1.040 | 54.545 ± 3.669 | 67.438 ± 1.796 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.118 ± 0.016 | 0.023 ± 0.000 | 0.083 ± 0.007 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.136 ± 0.015 | 0.025 ± 0.002 | 0.070 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.150 ± 0.010 | 0.040 ± 0.002 | 0.114 ± 0.028 | 0.026 ± 0.005 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.042 ± 0.013 | 0.023 ± 0.003 | 0.073 ± 0.006 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.049 ± 0.001 | 0.025 ± 0.000 | 0.060 ± 0.001 | 0.010 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.059 ± 0.003 | 0.036 ± 0.003 | 0.101 ± 0.036 | 0.026 ± 0.004 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.367 ± 0.022 | 0.027 ± 0.002 | 0.425 ± 0.231 | 0.009 ± 0.003 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.441 ± 0.022 | 0.035 ± 0.018 | 0.399 ± 0.224 | 0.009 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.374 ± 0.024 | 0.035 ± 0.004 | 0.428 ± 0.069 | 0.014 ± 0.003 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.110 ± 0.079 | 0.026 ± 0.000 | 0.341 ± 0.037 | 0.008 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.147 ± 0.017 | 0.025 ± 0.002 | 0.179 ± 0.054 | 0.009 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.129 ± 0.007 | 0.051 ± 0.009 | 0.322 ± 0.061 | 0.015 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.587 ± 0.055 | 0.040 ± 0.001 | 0.201 ± 0.044 | 0.009 ± 0.006 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.582 ± 0.046 | 0.040 ± 0.002 | 0.285 ± 0.132 | 0.010 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.650 ± 0.038 | 0.106 ± 0.028 | 0.234 ± 0.081 | 0.015 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.127 ± 0.035 | 0.042 ± 0.003 | 0.172 ± 0.023 | 0.008 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.100 ± 0.018 | 0.039 ± 0.001 | 0.236 ± 0.065 | 0.009 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.134 ± 0.018 | 0.053 ± 0.013 | 0.188 ± 0.099 | 0.015 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.083 ± 0.023 | 0.021 ± 0.000 | 0.180 ± 0.018 | 0.009 ± 0.002 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.097 ± 0.011 | 0.021 ± 0.001 | 0.161 ± 0.011 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.084 ± 0.005 | 0.026 ± 0.018 | 0.226 ± 0.114 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.046 ± 0.002 | 0.022 ± 0.003 | 0.087 ± 0.003 | 0.010 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.052 ± 0.004 | 0.032 ± 0.004 | 0.079 ± 0.010 | 0.008 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.043 ± 0.001 | 0.026 ± 0.001 | 0.118 ± 0.032 | 0.009 ± 0.004 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.119 ± 0.042 | 0.017 ± 0.001 | 0.115 ± 0.027 | 0.014 ± 0.014 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.118 ± 0.003 | 0.020 ± 0.001 | 0.148 ± 0.051 | 0.013 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.162 ± 0.017 | 0.047 ± 0.005 | 0.142 ± 0.032 | 0.042 ± 0.003 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.052 ± 0.022 | 0.016 ± 0.001 | 0.085 ± 0.024 | 0.012 ± 0.004 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.070 ± 0.014 | 0.018 ± 0.000 | 0.097 ± 0.047 | 0.013 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.050 ± 0.006 | 0.039 ± 0.002 | 0.085 ± 0.010 | 0.043 ± 0.037 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

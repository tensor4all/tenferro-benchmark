# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `74775e4a2ad3480fb9ee9f859e0037b8cac731f2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261125_172313/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261125_172313/run.yaml`
- Timestamp: `2026-11-25T17:23:13.361566+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261125_172313`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20261125_172313/cpu_ops_t1_20261125_172313.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 18.056 ± 6.633 | 12.751 ± 0.402 | 5.724 ± 0.099 | 8.175 ± 1.276 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 89.593 ± 14.236 | 64.242 ± 5.327 | 36.277 ± 1.387 | 44.390 ± 2.072 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 10.937 ± 2.472 | 12.757 ± 1.919 | 5.274 ± 0.164 | 10.532 ± 5.492 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 61.972 ± 9.740 | 63.201 ± 1.174 | 35.166 ± 5.723 | 45.318 ± 9.481 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 11.906 ± 3.735 | 6.780 ± 0.323 | 2.228 ± 0.076 | 5.220 ± 2.240 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 82.919 ± 13.885 | 35.963 ± 1.065 | 27.903 ± 6.868 | 24.012 ± 0.825 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 10.247 ± 0.962 | 6.593 ± 0.285 | 1.312 ± 0.209 | 9.236 ± 2.452 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 85.995 ± 11.596 | 41.381 ± 2.421 | 21.506 ± 7.935 | 23.582 ± 2.844 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 14.982 ± 0.943 | 10.123 ± 0.427 | 3.320 ± 0.522 | 6.204 ± 2.497 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 115.625 ± 16.650 | 60.691 ± 6.124 | 18.368 ± 5.011 | 33.576 ± 1.839 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 11.893 ± 3.855 | 9.015 ± 0.450 | 3.075 ± 0.118 | 5.610 ± 3.880 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 87.730 ± 1.621 | 56.019 ± 0.639 | 19.113 ± 0.841 | 34.190 ± 1.153 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.536 ± 0.217 | 1.284 ± 0.086 | 0.910 ± 0.812 | 0.743 ± 0.033 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 8.490 ± 0.706 | 12.064 ± 5.679 | 4.605 ± 2.268 | 2.413 ± 0.244 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.309 ± 0.539 | 1.402 ± 0.170 | 0.646 ± 0.081 | 0.657 ± 0.518 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.358 ± 0.096 | 9.891 ± 6.722 | 6.896 ± 2.775 | 2.277 ± 0.184 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 26.851 ± 1.573 | 23.088 ± 0.769 | 12.052 ± 3.866 | 16.209 ± 2.592 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 251.873 ± 35.166 | 151.403 ± 9.623 | 60.929 ± 10.133 | 81.103 ± 5.019 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 14.237 ± 1.304 | 22.629 ± 0.437 | 10.765 ± 4.506 | 16.168 ± 3.099 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 114.685 ± 19.432 | 147.581 ± 7.865 | 55.434 ± 0.618 | 74.309 ± 3.365 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.133 ± 0.013 | 0.026 ± 0.014 | 0.103 ± 0.024 | 0.009 ± 0.011 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.121 ± 0.031 | 0.028 ± 0.025 | 0.070 ± 0.006 | 0.010 ± 0.002 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.116 ± 0.003 | 0.024 ± 0.000 | 0.103 ± 0.020 | 0.023 ± 0.006 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.046 ± 0.040 | 0.019 ± 0.017 | 0.085 ± 0.041 | 0.012 ± 0.009 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.037 ± 0.002 | 0.024 ± 0.009 | 0.061 ± 0.004 | 0.010 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.042 ± 0.005 | 0.024 ± 0.000 | 0.112 ± 0.015 | 0.014 ± 0.005 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.667 ± 0.146 | 0.028 ± 0.008 | 0.129 ± 0.015 | 0.014 ± 0.005 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.378 ± 0.025 | 0.024 ± 0.006 | 0.100 ± 0.005 | 0.008 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.401 ± 0.039 | 0.023 ± 0.012 | 0.152 ± 0.010 | 0.018 ± 0.015 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.123 ± 0.035 | 0.025 ± 0.006 | 0.108 ± 0.016 | 0.011 ± 0.003 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.074 ± 0.011 | 0.022 ± 0.013 | 0.085 ± 0.041 | 0.009 ± 0.003 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.079 ± 0.013 | 0.019 ± 0.000 | 0.096 ± 0.061 | 0.009 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.568 ± 0.052 | 0.030 ± 0.004 | 0.121 ± 0.078 | 0.012 ± 0.008 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.461 ± 0.146 | 0.038 ± 0.007 | 0.077 ± 0.002 | 0.010 ± 0.010 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.423 ± 0.033 | 0.029 ± 0.001 | 0.109 ± 0.014 | 0.010 ± 0.004 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.090 ± 0.008 | 0.029 ± 0.001 | 0.086 ± 0.012 | 0.009 ± 0.003 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.074 ± 0.004 | 0.036 ± 0.018 | 0.078 ± 0.021 | 0.010 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.072 ± 0.004 | 0.028 ± 0.001 | 0.159 ± 0.081 | 0.008 ± 0.007 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.097 ± 0.020 | 0.026 ± 0.014 | 0.227 ± 0.027 | 0.012 ± 0.009 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.072 ± 0.001 | 0.015 ± 0.000 | 0.166 ± 0.039 | 0.009 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.071 ± 0.008 | 0.015 ± 0.001 | 0.721 ± 0.880 | 0.007 ± 0.003 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.043 ± 0.003 | 0.030 ± 0.007 | 0.092 ± 0.013 | 0.015 ± 0.007 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.036 ± 0.001 | 0.015 ± 0.003 | 0.087 ± 0.008 | 0.009 ± 0.004 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.037 ± 0.000 | 0.015 ± 0.000 | 0.137 ± 0.099 | 0.009 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.122 ± 0.129 | 0.013 ± 0.000 | 0.142 ± 0.172 | 0.012 ± 0.009 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.102 ± 0.033 | 0.015 ± 0.004 | 0.097 ± 0.007 | 0.019 ± 0.009 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.110 ± 0.008 | 0.024 ± 0.013 | 0.141 ± 0.052 | 0.019 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.042 ± 0.001 | 0.012 ± 0.000 | 0.072 ± 0.003 | 0.013 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.050 ± 0.061 | 0.015 ± 0.004 | 0.069 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.044 ± 0.003 | 0.022 ± 0.000 | 0.092 ± 0.022 | 0.022 ± 0.006 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261125_180002/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261125_180002/run.yaml`
- Timestamp: `2026-11-25T18:00:02.301847+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261125_180002`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20261125_180002/cpu_ops_t4_20261125_180002.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 16.804 ± 1.643 | 12.245 ± 0.777 | 5.665 ± 0.142 | 6.620 ± 0.317 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 91.042 ± 3.049 | 71.507 ± 17.997 | 35.591 ± 1.669 | 37.689 ± 2.347 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 9.662 ± 1.047 | 12.215 ± 0.605 | 5.030 ± 0.191 | 7.245 ± 0.530 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 62.727 ± 9.025 | 68.165 ± 19.798 | 32.167 ± 2.234 | 35.575 ± 1.113 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 14.144 ± 7.947 | 5.970 ± 2.906 | 3.527 ± 0.905 | 2.784 ± 0.203 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 63.794 ± 8.114 | 32.298 ± 2.641 | 18.504 ± 0.671 | 16.816 ± 8.098 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 9.840 ± 4.356 | 6.188 ± 3.854 | 1.997 ± 0.140 | 2.576 ± 0.198 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 64.804 ± 13.086 | 30.845 ± 0.946 | 11.392 ± 1.687 | 17.876 ± 8.677 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 13.601 ± 1.662 | 9.449 ± 0.827 | 3.198 ± 0.111 | 5.274 ± 2.483 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 87.789 ± 11.859 | 67.362 ± 9.002 | 26.014 ± 16.528 | 23.012 ± 2.538 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 8.268 ± 1.205 | 9.140 ± 0.933 | 3.055 ± 0.283 | 5.430 ± 1.265 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 61.242 ± 4.026 | 60.598 ± 12.409 | 17.433 ± 1.045 | 25.401 ± 5.900 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 2.753 ± 0.301 | 1.921 ± 0.500 | 0.525 ± 0.347 | 0.389 ± 0.080 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 12.932 ± 5.541 | 8.806 ± 0.200 | 2.770 ± 0.532 | 3.343 ± 1.492 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.023 ± 0.053 | 1.964 ± 0.764 | 0.694 ± 0.019 | 0.902 ± 0.359 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 5.695 ± 3.252 | 8.925 ± 0.152 | 3.917 ± 0.259 | 1.862 ± 0.225 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 29.990 ± 6.095 | 21.992 ± 0.963 | 10.691 ± 0.050 | 11.393 ± 1.163 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 165.301 ± 6.530 | 128.953 ± 16.119 | 85.012 ± 11.511 | 83.365 ± 9.413 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 14.601 ± 3.233 | 23.276 ± 8.639 | 9.972 ± 3.744 | 11.253 ± 3.256 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 92.573 ± 2.157 | 134.196 ± 25.976 | 70.085 ± 7.879 | 67.067 ± 5.758 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.122 ± 0.022 | 0.022 ± 0.013 | 0.077 ± 0.029 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.183 ± 0.046 | 0.024 ± 0.002 | 0.077 ± 0.016 | 0.010 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.115 ± 0.008 | 0.030 ± 0.003 | 0.097 ± 0.009 | 0.021 ± 0.004 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.037 ± 0.002 | 0.022 ± 0.020 | 0.076 ± 0.147 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.049 ± 0.014 | 0.023 ± 0.006 | 0.074 ± 0.078 | 0.010 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.044 ± 0.002 | 0.028 ± 0.001 | 0.093 ± 0.014 | 0.024 ± 0.005 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.436 ± 0.035 | 0.022 ± 0.001 | 0.557 ± 0.397 | 0.008 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.407 ± 0.150 | 0.021 ± 0.005 | 0.713 ± 0.415 | 0.009 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.407 ± 0.040 | 0.024 ± 0.001 | 0.732 ± 0.069 | 0.013 ± 0.005 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.095 ± 0.016 | 0.020 ± 0.001 | 0.431 ± 0.177 | 0.008 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.146 ± 0.044 | 0.022 ± 0.001 | 0.604 ± 0.215 | 0.010 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.099 ± 0.017 | 0.023 ± 0.002 | 0.476 ± 0.190 | 0.020 ± 0.006 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.459 ± 0.028 | 0.041 ± 0.008 | 0.216 ± 0.096 | 0.010 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.652 ± 0.132 | 0.033 ± 0.001 | 0.157 ± 0.034 | 0.010 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.518 ± 0.121 | 0.031 ± 0.003 | 0.194 ± 0.025 | 0.016 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.080 ± 0.003 | 0.032 ± 0.008 | 0.203 ± 0.061 | 0.009 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.115 ± 0.059 | 0.032 ± 0.002 | 0.175 ± 0.027 | 0.010 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.099 ± 0.014 | 0.034 ± 0.003 | 0.179 ± 0.009 | 0.016 ± 0.002 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.097 ± 0.020 | 0.020 ± 0.003 | 0.311 ± 0.159 | 0.007 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.135 ± 0.051 | 0.018 ± 0.001 | 0.222 ± 0.155 | 0.008 ± 0.004 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.082 ± 0.003 | 0.018 ± 0.001 | 0.200 ± 0.014 | 0.026 ± 0.016 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.042 ± 0.002 | 0.017 ± 0.000 | 0.100 ± 0.046 | 0.009 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.065 ± 0.027 | 0.018 ± 0.001 | 0.096 ± 0.051 | 0.009 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.040 ± 0.001 | 0.018 ± 0.001 | 0.089 ± 0.010 | 0.009 ± 0.008 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.107 ± 0.010 | 0.013 ± 0.002 | 0.098 ± 0.055 | 0.012 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.149 ± 0.051 | 0.020 ± 0.002 | 0.111 ± 0.026 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.128 ± 0.028 | 0.027 ± 0.005 | 0.178 ± 0.116 | 0.040 ± 0.023 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.038 ± 0.003 | 0.013 ± 0.001 | 0.149 ± 0.064 | 0.012 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.067 ± 0.012 | 0.017 ± 0.002 | 0.066 ± 0.004 | 0.014 ± 0.003 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.062 ± 0.028 | 0.027 ± 0.009 | 0.090 ± 0.004 | 0.043 ± 0.003 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `7da876a214acb65eb9ea099adef67b41ef8efc98`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261127_172420/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261127_172420/run.yaml`
- Timestamp: `2026-11-27T17:24:20.827597+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261127_172420`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261127_172420/cpu_ops_t1_20261127_172420.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 17.521 ± 0.636 | 12.975 ± 1.910 | 6.051 ± 1.865 | 6.885 ± 0.269 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 83.653 ± 0.437 | 86.892 ± 6.576 | 26.823 ± 3.029 | 43.397 ± 1.072 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 8.801 ± 0.429 | 12.391 ± 4.770 | 6.599 ± 1.996 | 7.052 ± 0.179 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 47.669 ± 1.238 | 99.560 ± 27.620 | 26.650 ± 1.856 | 40.476 ± 3.874 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 10.920 ± 1.364 | 7.213 ± 2.147 | 2.459 ± 0.342 | 3.005 ± 0.181 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 73.664 ± 2.749 | 57.193 ± 9.592 | 15.371 ± 2.489 | 22.470 ± 2.241 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 9.762 ± 1.268 | 6.914 ± 1.960 | 1.191 ± 0.162 | 2.866 ± 0.225 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 67.272 ± 3.803 | 55.343 ± 3.812 | 9.104 ± 0.687 | 20.056 ± 1.548 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 17.429 ± 3.681 | 11.299 ± 4.178 | 2.890 ± 0.553 | 4.813 ± 0.276 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 115.987 ± 29.366 | 83.371 ± 8.270 | 14.558 ± 2.547 | 33.950 ± 3.652 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 14.141 ± 1.798 | 10.219 ± 3.054 | 2.154 ± 0.157 | 4.785 ± 0.072 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 80.514 ± 1.677 | 77.974 ± 12.019 | 12.994 ± 1.623 | 33.230 ± 2.806 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.274 ± 0.293 | 1.457 ± 0.915 | 0.465 ± 0.081 | 0.328 ± 0.020 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 6.875 ± 0.255 | 8.206 ± 3.975 | 1.903 ± 0.277 | 2.333 ± 0.238 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.611 ± 0.029 | 1.199 ± 0.335 | 0.513 ± 0.050 | 0.364 ± 0.027 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 3.652 ± 0.185 | 10.070 ± 3.268 | 2.716 ± 0.623 | 1.732 ± 0.130 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 27.603 ± 0.923 | 22.182 ± 0.715 | 10.822 ± 1.702 | 13.227 ± 3.046 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 173.900 ± 50.585 | 141.394 ± 11.704 | 56.611 ± 6.873 | 69.978 ± 7.356 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 16.087 ± 0.511 | 26.622 ± 3.193 | 9.141 ± 0.414 | 13.983 ± 2.197 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 95.458 ± 7.905 | 144.072 ± 9.832 | 54.434 ± 4.777 | 85.483 ± 16.884 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.116 ± 0.041 | 0.016 ± 0.000 | 0.074 ± 0.010 | 0.007 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.121 ± 0.029 | 0.019 ± 0.001 | 0.097 ± 0.090 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.244 ± 0.074 | 0.026 ± 0.001 | 0.084 ± 0.008 | 0.013 ± 0.004 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.051 ± 0.033 | 0.016 ± 0.000 | 0.070 ± 0.013 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.040 ± 0.015 | 0.020 ± 0.005 | 0.070 ± 0.068 | 0.009 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.081 ± 0.028 | 0.027 ± 0.002 | 0.077 ± 0.004 | 0.011 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.584 ± 0.118 | 0.018 ± 0.000 | 0.103 ± 0.017 | 0.008 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.435 ± 0.175 | 0.022 ± 0.003 | 0.112 ± 0.005 | 0.008 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.573 ± 0.180 | 0.022 ± 0.004 | 0.117 ± 0.007 | 0.010 ± 0.004 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.104 ± 0.036 | 0.016 ± 0.002 | 0.100 ± 0.017 | 0.009 ± 0.003 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.085 ± 0.007 | 0.018 ± 0.001 | 0.103 ± 0.024 | 0.009 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.127 ± 0.077 | 0.020 ± 0.003 | 0.091 ± 0.011 | 0.008 ± 0.008 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.494 ± 0.032 | 0.025 ± 0.000 | 0.084 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.431 ± 0.024 | 0.027 ± 0.001 | 0.088 ± 0.015 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.798 ± 0.264 | 0.032 ± 0.004 | 0.112 ± 0.040 | 0.010 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.076 ± 0.002 | 0.025 ± 0.001 | 0.087 ± 0.010 | 0.008 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.073 ± 0.002 | 0.029 ± 0.002 | 0.082 ± 0.024 | 0.011 ± 0.009 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.099 ± 0.119 | 0.032 ± 0.005 | 0.100 ± 0.012 | 0.010 ± 0.006 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.080 ± 0.023 | 0.015 ± 0.002 | 0.169 ± 0.008 | 0.008 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.080 ± 0.054 | 0.015 ± 0.001 | 0.167 ± 0.197 | 0.007 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.122 ± 0.070 | 0.020 ± 0.008 | 0.160 ± 0.013 | 0.008 ± 0.005 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.058 ± 0.045 | 0.014 ± 0.000 | 0.114 ± 0.153 | 0.008 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.058 ± 0.030 | 0.016 ± 0.000 | 0.087 ± 0.010 | 0.008 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.045 ± 0.026 | 0.026 ± 0.007 | 0.078 ± 0.017 | 0.009 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.106 ± 0.005 | 0.012 ± 0.000 | 0.124 ± 0.100 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.112 ± 0.043 | 0.016 ± 0.002 | 0.114 ± 0.019 | 0.011 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.317 ± 0.086 | 0.027 ± 0.006 | 0.166 ± 0.066 | 0.018 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.041 ± 0.007 | 0.010 ± 0.000 | 0.116 ± 0.365 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.047 ± 0.026 | 0.016 ± 0.001 | 0.072 ± 0.112 | 0.012 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.072 ± 0.051 | 0.025 ± 0.003 | 0.114 ± 0.098 | 0.017 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261127_180034/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261127_180034/run.yaml`
- Timestamp: `2026-11-27T18:00:34.399034+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261127_180034`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261127_180034/cpu_ops_t4_20261127_180034.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 15.049 ± 0.425 | 11.396 ± 0.235 | 5.893 ± 0.143 | 5.725 ± 0.091 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 86.868 ± 7.589 | 60.612 ± 1.384 | 29.631 ± 0.916 | 31.236 ± 2.281 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 8.306 ± 1.185 | 11.038 ± 0.199 | 5.221 ± 0.082 | 5.824 ± 0.082 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 43.357 ± 1.165 | 61.307 ± 0.619 | 26.724 ± 1.433 | 31.907 ± 4.881 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 9.046 ± 1.072 | 4.765 ± 0.106 | 3.273 ± 0.085 | 1.836 ± 0.029 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 53.986 ± 3.676 | 25.200 ± 2.646 | 17.729 ± 0.776 | 14.513 ± 2.204 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 8.822 ± 0.255 | 4.861 ± 0.333 | 2.293 ± 0.508 | 1.839 ± 0.034 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 49.953 ± 4.815 | 24.689 ± 1.506 | 10.353 ± 1.345 | 10.778 ± 0.270 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 13.532 ± 1.481 | 7.657 ± 0.266 | 3.596 ± 1.693 | 4.653 ± 0.323 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 63.414 ± 9.091 | 40.636 ± 3.578 | 16.011 ± 0.133 | 17.456 ± 0.088 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 10.007 ± 1.321 | 7.520 ± 0.284 | 5.433 ± 1.423 | 3.248 ± 0.022 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 58.267 ± 0.787 | 40.624 ± 5.152 | 15.763 ± 0.346 | 18.033 ± 0.512 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.990 ± 0.339 | 1.637 ± 0.143 | 0.532 ± 0.109 | 0.324 ± 0.003 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 7.338 ± 0.449 | 7.038 ± 0.330 | 1.718 ± 0.042 | 1.590 ± 0.032 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.134 ± 0.381 | 1.641 ± 0.065 | 0.829 ± 1.432 | 0.321 ± 0.017 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 4.635 ± 0.099 | 6.957 ± 0.158 | 2.977 ± 0.228 | 1.417 ± 0.013 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 27.379 ± 3.088 | 21.909 ± 2.242 | 16.394 ± 5.981 | 13.020 ± 2.807 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 180.443 ± 12.554 | 121.227 ± 3.295 | 57.476 ± 7.315 | 63.936 ± 1.163 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 13.787 ± 2.353 | 20.075 ± 2.596 | 10.539 ± 2.969 | 11.417 ± 0.640 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 83.418 ± 12.693 | 117.067 ± 3.329 | 52.689 ± 6.509 | 55.607 ± 2.814 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.124 ± 0.012 | 0.017 ± 0.001 | 0.120 ± 0.047 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.106 ± 0.002 | 0.020 ± 0.000 | 0.072 ± 0.003 | 0.010 ± 0.003 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.111 ± 0.004 | 0.025 ± 0.001 | 0.079 ± 0.004 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.035 ± 0.016 | 0.017 ± 0.000 | 0.057 ± 0.034 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.034 ± 0.001 | 0.020 ± 0.001 | 0.062 ± 0.002 | 0.012 ± 0.004 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.043 ± 0.004 | 0.026 ± 0.001 | 0.068 ± 0.007 | 0.011 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.515 ± 0.053 | 0.019 ± 0.000 | 0.392 ± 0.052 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.353 ± 0.007 | 0.021 ± 0.001 | 0.298 ± 0.082 | 0.010 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.373 ± 0.078 | 0.028 ± 0.007 | 0.356 ± 0.027 | 0.008 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.085 ± 0.004 | 0.018 ± 0.002 | 0.211 ± 0.013 | 0.007 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.081 ± 0.004 | 0.019 ± 0.000 | 0.215 ± 0.018 | 0.010 ± 0.003 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.084 ± 0.008 | 0.022 ± 0.008 | 0.284 ± 0.112 | 0.008 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.391 ± 0.038 | 0.026 ± 0.000 | 0.280 ± 0.058 | 0.008 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.415 ± 0.021 | 0.029 ± 0.001 | 0.155 ± 0.008 | 0.015 ± 0.006 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.422 ± 0.028 | 0.029 ± 0.000 | 0.148 ± 0.020 | 0.015 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.113 ± 0.069 | 0.028 ± 0.002 | 0.273 ± 0.095 | 0.008 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.068 ± 0.002 | 0.029 ± 0.001 | 0.145 ± 0.010 | 0.012 ± 0.006 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.076 ± 0.005 | 0.029 ± 0.000 | 0.144 ± 0.047 | 0.009 ± 0.003 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.078 ± 0.003 | 0.015 ± 0.000 | 0.158 ± 0.007 | 0.007 ± 0.002 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.079 ± 0.004 | 0.016 ± 0.000 | 0.154 ± 0.018 | 0.008 ± 0.002 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.076 ± 0.003 | 0.020 ± 0.003 | 0.165 ± 0.016 | 0.014 ± 0.010 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.047 ± 0.006 | 0.015 ± 0.000 | 0.078 ± 0.008 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.036 ± 0.003 | 0.016 ± 0.001 | 0.074 ± 0.003 | 0.009 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.038 ± 0.001 | 0.020 ± 0.004 | 0.077 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.091 ± 0.002 | 0.012 ± 0.000 | 0.096 ± 0.013 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.096 ± 0.035 | 0.015 ± 0.000 | 0.092 ± 0.001 | 0.013 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.103 ± 0.002 | 0.024 ± 0.000 | 0.113 ± 0.001 | 0.046 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.033 ± 0.003 | 0.011 ± 0.001 | 0.074 ± 0.024 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.033 ± 0.001 | 0.015 ± 0.000 | 0.063 ± 0.001 | 0.014 ± 0.003 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.041 ± 0.023 | 0.023 ± 0.002 | 0.075 ± 0.006 | 0.047 ± 0.002 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

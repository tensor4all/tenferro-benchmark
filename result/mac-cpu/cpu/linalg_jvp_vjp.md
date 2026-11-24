# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Benchmark commit: `9185e8ca26c3fc600458361d0f477495a717daf3`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261124_172410/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261124_172410/run.yaml`
- Timestamp: `2026-11-24T17:24:10.690570+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261124_172410`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261124_172410/cpu_ops_t1_20261124_172410.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 15.392 ± 0.401 | 11.764 ± 0.307 | 5.640 ± 0.427 | 6.737 ± 0.388 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 87.331 ± 4.772 | 74.686 ± 10.132 | 39.605 ± 4.585 | 42.608 ± 2.021 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 8.778 ± 0.257 | 13.187 ± 3.405 | 5.193 ± 0.462 | 7.066 ± 0.559 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 61.626 ± 16.545 | 63.795 ± 1.898 | 32.961 ± 3.555 | 40.389 ± 2.090 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 10.580 ± 0.742 | 5.409 ± 0.555 | 2.629 ± 0.913 | 2.837 ± 0.221 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 92.454 ± 18.485 | 36.482 ± 1.753 | 19.761 ± 5.723 | 20.739 ± 1.578 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 9.659 ± 0.511 | 5.550 ± 2.464 | 2.201 ± 0.965 | 2.815 ± 1.074 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 76.742 ± 19.231 | 36.626 ± 0.563 | 13.545 ± 5.166 | 20.108 ± 0.858 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 15.512 ± 2.046 | 8.322 ± 0.261 | 3.183 ± 1.457 | 4.721 ± 0.204 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 99.599 ± 1.673 | 66.065 ± 5.862 | 19.475 ± 7.620 | 31.960 ± 0.568 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 11.841 ± 5.430 | 8.173 ± 0.300 | 2.625 ± 0.285 | 4.661 ± 0.166 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 81.907 ± 1.528 | 62.676 ± 1.663 | 19.365 ± 2.674 | 31.259 ± 0.862 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.251 ± 0.116 | 1.117 ± 0.129 | 0.492 ± 0.039 | 0.300 ± 0.015 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 7.421 ± 0.463 | 6.892 ± 0.419 | 1.999 ± 0.273 | 1.802 ± 0.127 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.643 ± 0.012 | 1.022 ± 0.289 | 0.564 ± 0.065 | 0.357 ± 0.105 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.025 ± 1.729 | 6.558 ± 0.109 | 3.291 ± 0.394 | 1.562 ± 0.164 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 31.117 ± 8.124 | 21.940 ± 0.513 | 10.398 ± 0.495 | 11.135 ± 0.451 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 172.388 ± 3.481 | 145.910 ± 33.336 | 62.474 ± 1.411 | 68.352 ± 1.244 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 15.558 ± 0.532 | 22.200 ± 0.119 | 9.400 ± 0.509 | 11.188 ± 0.549 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 91.825 ± 0.698 | 147.098 ± 23.061 | 54.971 ± 2.893 | 67.593 ± 0.866 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.186 ± 0.155 | 0.017 ± 0.001 | 0.071 ± 0.018 | 0.007 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.136 ± 0.052 | 0.019 ± 0.001 | 0.074 ± 0.029 | 0.011 ± 0.006 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.136 ± 0.153 | 0.025 ± 0.001 | 0.085 ± 0.030 | 0.013 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.062 ± 0.019 | 0.016 ± 0.000 | 0.061 ± 0.007 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.066 ± 0.046 | 0.017 ± 0.002 | 0.060 ± 0.004 | 0.010 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.039 ± 0.004 | 0.025 ± 0.002 | 0.075 ± 0.005 | 0.013 ± 0.003 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.483 ± 0.572 | 0.019 ± 0.001 | 0.123 ± 0.013 | 0.008 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.398 ± 0.241 | 0.020 ± 0.008 | 0.104 ± 0.009 | 0.011 ± 0.004 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.355 ± 0.007 | 0.023 ± 0.001 | 0.119 ± 0.020 | 0.015 ± 0.012 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.088 ± 0.022 | 0.017 ± 0.000 | 0.080 ± 0.003 | 0.008 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.075 ± 0.002 | 0.018 ± 0.003 | 0.080 ± 0.004 | 0.009 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.078 ± 0.004 | 0.020 ± 0.000 | 0.117 ± 0.067 | 0.011 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.568 ± 0.365 | 0.026 ± 0.000 | 0.078 ± 0.004 | 0.008 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.904 ± 0.591 | 0.032 ± 0.004 | 0.090 ± 0.030 | 0.010 ± 0.007 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.496 ± 0.010 | 0.028 ± 0.001 | 0.087 ± 0.024 | 0.011 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.108 ± 0.018 | 0.026 ± 0.002 | 0.077 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.162 ± 0.061 | 0.027 ± 0.001 | 0.087 ± 0.067 | 0.011 ± 0.007 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.074 ± 0.029 | 0.028 ± 0.000 | 0.094 ± 0.032 | 0.010 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.102 ± 0.036 | 0.014 ± 0.000 | 0.163 ± 0.016 | 0.008 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.075 ± 0.002 | 0.015 ± 0.000 | 0.204 ± 0.144 | 0.012 ± 0.008 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.073 ± 0.008 | 0.016 ± 0.002 | 0.174 ± 0.022 | 0.008 ± 0.003 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.052 ± 0.038 | 0.014 ± 0.001 | 0.079 ± 0.003 | 0.008 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.035 ± 0.004 | 0.015 ± 0.000 | 0.079 ± 0.015 | 0.012 ± 0.008 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.036 ± 0.002 | 0.017 ± 0.001 | 0.078 ± 0.004 | 0.008 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.089 ± 0.002 | 0.011 ± 0.000 | 0.102 ± 0.018 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.101 ± 0.090 | 0.014 ± 0.000 | 0.183 ± 0.084 | 0.020 ± 0.006 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.103 ± 0.005 | 0.023 ± 0.001 | 0.113 ± 0.008 | 0.018 ± 0.004 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.032 ± 0.004 | 0.011 ± 0.000 | 0.063 ± 0.008 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.041 ± 0.016 | 0.015 ± 0.002 | 0.074 ± 0.068 | 0.016 ± 0.010 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.041 ± 0.001 | 0.022 ± 0.001 | 0.075 ± 0.006 | 0.018 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261124_175807/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261124_175807/run.yaml`
- Timestamp: `2026-11-24T17:58:07.132443+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261124_175807`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261124_175807/cpu_ops_t4_20261124_175807.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 17.746 ± 1.750 | 11.479 ± 0.215 | 5.800 ± 0.257 | 6.563 ± 0.391 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 83.438 ± 1.888 | 62.405 ± 1.056 | 33.783 ± 1.842 | 37.916 ± 2.696 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 9.138 ± 0.244 | 11.423 ± 0.165 | 5.413 ± 0.145 | 6.751 ± 3.226 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 46.830 ± 1.688 | 63.816 ± 2.099 | 30.266 ± 1.411 | 43.236 ± 11.168 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 10.304 ± 0.316 | 5.868 ± 0.225 | 3.481 ± 0.281 | 3.523 ± 1.396 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 69.559 ± 12.824 | 33.569 ± 4.396 | 18.489 ± 0.996 | 17.594 ± 5.130 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 9.405 ± 1.420 | 6.046 ± 0.428 | 2.143 ± 0.434 | 4.245 ± 2.460 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 60.698 ± 11.616 | 31.070 ± 1.451 | 10.759 ± 0.597 | 17.032 ± 3.917 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 13.335 ± 0.257 | 8.681 ± 3.064 | 5.000 ± 2.151 | 4.901 ± 1.951 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 101.932 ± 17.410 | 55.497 ± 21.920 | 16.935 ± 0.998 | 26.568 ± 4.195 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 10.372 ± 0.175 | 8.338 ± 0.313 | 7.511 ± 3.444 | 5.057 ± 1.148 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 64.881 ± 2.031 | 49.879 ± 1.354 | 16.299 ± 1.178 | 24.397 ± 4.869 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 2.122 ± 0.053 | 1.879 ± 0.123 | 0.607 ± 0.102 | 0.515 ± 0.284 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 9.586 ± 1.210 | 8.340 ± 1.817 | 2.328 ± 0.653 | 2.460 ± 0.712 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.492 ± 0.409 | 1.982 ± 0.140 | 0.758 ± 0.075 | 0.988 ± 0.259 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 4.849 ± 0.078 | 8.565 ± 0.507 | 3.927 ± 0.351 | 1.879 ± 0.198 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 29.460 ± 1.133 | 22.107 ± 1.097 | 11.314 ± 0.634 | 11.386 ± 1.653 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 180.388 ± 29.579 | 128.032 ± 11.317 | 62.603 ± 1.387 | 81.435 ± 9.488 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 15.375 ± 0.556 | 21.699 ± 0.546 | 10.051 ± 5.028 | 13.232 ± 1.677 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 92.907 ± 15.236 | 128.281 ± 3.072 | 57.318 ± 1.926 | 66.222 ± 5.725 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.114 ± 0.002 | 0.024 ± 0.001 | 0.072 ± 0.007 | 0.007 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.117 ± 0.007 | 0.040 ± 0.018 | 0.070 ± 0.002 | 0.009 ± 0.003 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.136 ± 0.029 | 0.032 ± 0.001 | 0.087 ± 0.040 | 0.026 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.043 ± 0.001 | 0.023 ± 0.001 | 0.065 ± 0.017 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.044 ± 0.000 | 0.047 ± 0.017 | 0.060 ± 0.004 | 0.011 ± 0.005 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.050 ± 0.012 | 0.032 ± 0.001 | 0.091 ± 0.027 | 0.025 ± 0.004 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.382 ± 0.019 | 0.028 ± 0.001 | 0.363 ± 0.067 | 0.008 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.408 ± 0.037 | 0.058 ± 0.017 | 0.352 ± 0.030 | 0.009 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.387 ± 0.034 | 0.030 ± 0.001 | 0.400 ± 0.291 | 0.013 ± 0.003 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.100 ± 0.004 | 0.025 ± 0.000 | 0.244 ± 0.064 | 0.008 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.103 ± 0.002 | 0.037 ± 0.010 | 0.261 ± 0.029 | 0.009 ± 0.004 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.104 ± 0.003 | 0.028 ± 0.002 | 0.291 ± 0.084 | 0.013 ± 0.004 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.497 ± 0.021 | 0.038 ± 0.001 | 0.228 ± 0.117 | 0.008 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.493 ± 0.045 | 0.045 ± 0.030 | 0.134 ± 0.041 | 0.009 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.518 ± 0.072 | 0.041 ± 0.001 | 0.316 ± 0.200 | 0.017 ± 0.003 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.094 ± 0.001 | 0.037 ± 0.001 | 0.180 ± 0.131 | 0.008 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.095 ± 0.001 | 0.067 ± 0.028 | 0.147 ± 0.040 | 0.012 ± 0.010 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.119 ± 0.052 | 0.041 ± 0.000 | 0.228 ± 0.067 | 0.015 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.080 ± 0.002 | 0.022 ± 0.000 | 0.164 ± 0.015 | 0.008 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.081 ± 0.005 | 0.023 ± 0.003 | 0.163 ± 0.011 | 0.013 ± 0.009 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.085 ± 0.001 | 0.023 ± 0.000 | 0.166 ± 0.023 | 0.008 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.047 ± 0.002 | 0.022 ± 0.001 | 0.083 ± 0.074 | 0.009 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.045 ± 0.002 | 0.023 ± 0.001 | 0.076 ± 0.008 | 0.010 ± 0.005 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.047 ± 0.001 | 0.023 ± 0.000 | 0.079 ± 0.009 | 0.009 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.095 ± 0.002 | 0.017 ± 0.001 | 0.127 ± 0.097 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.098 ± 0.001 | 0.035 ± 0.009 | 0.095 ± 0.006 | 0.013 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.113 ± 0.003 | 0.029 ± 0.003 | 0.189 ± 0.053 | 0.046 ± 0.003 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.039 ± 0.001 | 0.016 ± 0.000 | 0.066 ± 0.020 | 0.011 ± 0.005 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.042 ± 0.002 | 0.023 ± 0.009 | 0.064 ± 0.003 | 0.013 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.066 ± 0.070 | 0.028 ± 0.008 | 0.082 ± 0.054 | 0.023 ± 0.017 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

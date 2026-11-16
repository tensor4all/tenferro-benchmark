# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `5c1d5a382dd88d20901598318730d850d3568ac8`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261116_172433/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261116_172433/run.yaml`
- Timestamp: `2026-11-16T17:24:33.485160+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261116_172433`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Apple M2 Pro (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `5`
- Physical CPUs: `5`
- Sockets: `1`
- Cores per socket: `5`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 5 physical / 5 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 5 CPUs/L2)`
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


- CSV: `data/results/mac-cpu/cpu/einsum/20261116_172433/cpu_ops_t1_20261116_172433.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 10.198 ± 0.071 | 8.257 ± 0.079 | 5.054 ± 0.208 | 6.292 ± 0.232 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 50.832 ± 2.033 | 44.363 ± 1.123 | 26.093 ± 3.156 | 33.217 ± 1.113 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 5.175 ± 0.226 | 8.347 ± 0.146 | 4.614 ± 0.257 | 6.265 ± 0.227 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 26.392 ± 1.712 | 44.127 ± 0.333 | 19.700 ± 0.866 | 32.259 ± 4.806 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 2.450 ± 0.335 | 1.063 ± 0.091 | 1.713 ± 0.147 | 2.599 ± 0.207 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 14.414 ± 1.284 | 7.825 ± 0.715 | 11.301 ± 0.917 | 14.820 ± 0.799 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 2.057 ± 0.045 | 1.220 ± 0.073 | 1.047 ± 0.076 | 2.458 ± 0.165 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 12.903 ± 0.313 | 7.719 ± 0.463 | 6.854 ± 0.532 | 14.286 ± 0.196 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 4.659 ± 0.186 | 3.483 ± 0.220 | 2.558 ± 0.263 | 4.561 ± 0.409 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 28.115 ± 1.148 | 20.216 ± 0.670 | 14.872 ± 1.424 | 29.194 ± 4.892 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 3.026 ± 0.498 | 3.752 ± 0.778 | 2.160 ± 0.246 | 4.293 ± 0.406 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 20.284 ± 0.819 | 19.959 ± 1.255 | 12.992 ± 0.665 | 29.132 ± 1.539 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.599 ± 0.049 | 0.445 ± 0.022 | 0.431 ± 0.211 | 0.300 ± 0.027 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 2.946 ± 0.498 | 2.417 ± 0.185 | 1.479 ± 0.054 | 1.174 ± 0.022 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.412 ± 0.020 | 0.494 ± 0.028 | 0.551 ± 0.062 | 0.307 ± 0.052 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 1.692 ± 0.245 | 2.494 ± 0.175 | 2.360 ± 0.210 | 1.130 ± 0.076 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 16.610 ± 0.811 | 12.079 ± 0.282 | 9.138 ± 0.221 | 9.230 ± 0.651 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 89.683 ± 7.567 | 65.420 ± 3.459 | 44.431 ± 5.941 | 47.416 ± 1.060 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 8.040 ± 0.196 | 12.765 ± 0.569 | 8.312 ± 0.414 | 9.555 ± 0.871 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 39.999 ± 1.750 | 69.017 ± 2.372 | 41.556 ± 2.392 | 46.420 ± 1.532 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.097 ± 0.005 | 0.017 ± 0.003 | 0.065 ± 0.007 | 0.007 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.093 ± 0.007 | 0.017 ± 0.001 | 0.072 ± 0.019 | 0.007 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.122 ± 0.020 | 0.025 ± 0.000 | 0.072 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.028 ± 0.003 | 0.014 ± 0.003 | 0.061 ± 0.004 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.030 ± 0.003 | 0.017 ± 0.001 | 0.056 ± 0.009 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.040 ± 0.001 | 0.021 ± 0.001 | 0.067 ± 0.008 | 0.008 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.354 ± 0.084 | 0.018 ± 0.001 | 0.110 ± 0.013 | 0.006 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.339 ± 0.026 | 0.019 ± 0.001 | 0.098 ± 0.026 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.351 ± 0.036 | 0.019 ± 0.004 | 0.114 ± 0.009 | 0.005 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.077 ± 0.004 | 0.017 ± 0.001 | 0.082 ± 0.009 | 0.006 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.071 ± 0.004 | 0.017 ± 0.002 | 0.082 ± 0.008 | 0.006 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.089 ± 0.010 | 0.021 ± 0.000 | 0.083 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.485 ± 0.019 | 0.026 ± 0.001 | 0.079 ± 0.009 | 0.007 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.441 ± 0.130 | 0.026 ± 0.001 | 0.084 ± 0.010 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.495 ± 0.020 | 0.030 ± 0.004 | 0.077 ± 0.003 | 0.006 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.064 ± 0.003 | 0.026 ± 0.000 | 0.077 ± 0.017 | 0.007 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.068 ± 0.003 | 0.027 ± 0.001 | 0.074 ± 0.015 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.076 ± 0.003 | 0.032 ± 0.000 | 0.074 ± 0.007 | 0.005 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.071 ± 0.002 | 0.014 ± 0.001 | 0.165 ± 0.027 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.066 ± 0.005 | 0.014 ± 0.001 | 0.155 ± 0.009 | 0.005 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.072 ± 0.006 | 0.015 ± 0.001 | 0.144 ± 0.013 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.033 ± 0.005 | 0.013 ± 0.001 | 0.076 ± 0.007 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.031 ± 0.002 | 0.014 ± 0.002 | 0.072 ± 0.005 | 0.006 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.034 ± 0.001 | 0.015 ± 0.001 | 0.070 ± 0.005 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.104 ± 0.014 | 0.012 ± 0.001 | 0.099 ± 0.005 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.088 ± 0.004 | 0.015 ± 0.001 | 0.128 ± 0.098 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.114 ± 0.003 | 0.029 ± 0.003 | 0.108 ± 0.006 | 0.016 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.028 ± 0.001 | 0.011 ± 0.000 | 0.068 ± 0.008 | 0.008 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.029 ± 0.006 | 0.014 ± 0.001 | 0.060 ± 0.007 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.037 ± 0.005 | 0.024 ± 0.001 | 0.064 ± 0.001 | 0.016 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261116_174236/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261116_174236/run.yaml`
- Timestamp: `2026-11-16T17:42:36.879712+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261116_174236`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Apple M2 Pro (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `5`
- Physical CPUs: `5`
- Sockets: `1`
- Cores per socket: `5`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 5 physical / 5 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 5 CPUs/L2)`
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


- CSV: `data/results/mac-cpu/cpu/einsum/20261116_174236/cpu_ops_t4_20261116_174236.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 10.420 ± 0.132 | 7.866 ± 0.268 | 4.796 ± 0.122 | 5.563 ± 0.513 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 50.802 ± 4.602 | 43.825 ± 5.245 | 25.127 ± 1.450 | 27.906 ± 1.558 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 5.502 ± 0.303 | 7.819 ± 0.093 | 4.902 ± 0.554 | 5.560 ± 0.079 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 28.346 ± 1.869 | 44.228 ± 2.933 | 23.431 ± 0.824 | 27.984 ± 1.509 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 2.561 ± 0.288 | 1.054 ± 0.041 | 2.608 ± 0.216 | 1.992 ± 0.054 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 12.591 ± 0.596 | 6.970 ± 0.374 | 11.834 ± 1.055 | 8.848 ± 0.898 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 2.240 ± 0.049 | 1.201 ± 0.032 | 1.735 ± 0.133 | 1.888 ± 0.133 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 11.792 ± 4.077 | 7.100 ± 0.219 | 7.385 ± 0.313 | 7.980 ± 0.650 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 5.174 ± 0.356 | 3.351 ± 0.075 | 2.723 ± 0.360 | 2.876 ± 0.250 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 23.928 ± 4.595 | 17.653 ± 1.542 | 12.223 ± 1.483 | 15.867 ± 1.071 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 3.247 ± 0.125 | 3.265 ± 0.201 | 2.711 ± 0.043 | 2.897 ± 0.076 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 19.269 ± 4.546 | 18.012 ± 7.658 | 12.001 ± 0.599 | 17.032 ± 1.290 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.656 ± 0.015 | 0.466 ± 0.021 | 0.543 ± 0.049 | 0.343 ± 0.042 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 2.581 ± 0.133 | 2.208 ± 0.095 | 1.553 ± 0.186 | 1.707 ± 0.182 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.336 ± 0.024 | 0.470 ± 0.011 | 0.590 ± 0.129 | 0.383 ± 0.025 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.397 ± 0.042 | 2.234 ± 0.083 | 2.332 ± 0.398 | 1.583 ± 0.209 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 16.315 ± 0.234 | 12.069 ± 0.408 | 8.859 ± 1.143 | 8.629 ± 0.787 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 79.456 ± 2.827 | 63.509 ± 2.712 | 44.074 ± 1.691 | 39.774 ± 3.326 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 8.308 ± 0.260 | 11.934 ± 0.252 | 7.177 ± 0.234 | 8.467 ± 0.505 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 38.882 ± 1.343 | 64.610 ± 4.076 | 37.525 ± 4.010 | 43.605 ± 4.302 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.109 ± 0.017 | 0.019 ± 0.001 | 0.060 ± 0.006 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.097 ± 0.006 | 0.020 ± 0.001 | 0.068 ± 0.009 | 0.007 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.109 ± 0.002 | 0.031 ± 0.001 | 0.074 ± 0.004 | 0.008 ± 0.003 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.037 ± 0.007 | 0.019 ± 0.001 | 0.047 ± 0.006 | 0.005 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.037 ± 0.002 | 0.020 ± 0.000 | 0.052 ± 0.006 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.048 ± 0.003 | 0.024 ± 0.001 | 0.068 ± 0.005 | 0.009 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.350 ± 0.013 | 0.023 ± 0.001 | 0.310 ± 0.090 | 0.006 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.363 ± 0.021 | 0.023 ± 0.000 | 0.373 ± 0.025 | 0.006 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.344 ± 0.017 | 0.025 ± 0.001 | 0.369 ± 0.045 | 0.008 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.090 ± 0.002 | 0.021 ± 0.001 | 0.272 ± 0.009 | 0.006 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.092 ± 0.007 | 0.021 ± 0.001 | 0.273 ± 0.056 | 0.007 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.108 ± 0.010 | 0.027 ± 0.001 | 0.286 ± 0.017 | 0.008 ± 0.004 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.482 ± 0.078 | 0.034 ± 0.001 | 0.181 ± 0.012 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.436 ± 0.014 | 0.035 ± 0.007 | 0.179 ± 0.006 | 0.008 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.470 ± 0.015 | 0.036 ± 0.002 | 0.182 ± 0.011 | 0.009 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.093 ± 0.023 | 0.032 ± 0.001 | 0.196 ± 0.022 | 0.006 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.081 ± 0.005 | 0.033 ± 0.001 | 0.188 ± 0.023 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.089 ± 0.003 | 0.042 ± 0.002 | 0.199 ± 0.008 | 0.008 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.071 ± 0.010 | 0.018 ± 0.000 | 0.136 ± 0.011 | 0.005 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.066 ± 0.005 | 0.018 ± 0.000 | 0.143 ± 0.009 | 0.006 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.072 ± 0.004 | 0.019 ± 0.001 | 0.137 ± 0.013 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.035 ± 0.002 | 0.018 ± 0.003 | 0.064 ± 0.004 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.037 ± 0.004 | 0.017 ± 0.000 | 0.066 ± 0.005 | 0.006 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.039 ± 0.002 | 0.019 ± 0.000 | 0.068 ± 0.006 | 0.006 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.085 ± 0.003 | 0.014 ± 0.000 | 0.077 ± 0.005 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.093 ± 0.008 | 0.017 ± 0.002 | 0.084 ± 0.005 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.119 ± 0.004 | 0.031 ± 0.004 | 0.106 ± 0.010 | 0.016 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.036 ± 0.006 | 0.013 ± 0.000 | 0.056 ± 0.009 | 0.008 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.034 ± 0.003 | 0.017 ± 0.001 | 0.056 ± 0.005 | 0.011 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.044 ± 0.007 | 0.024 ± 0.000 | 0.061 ± 0.010 | 0.016 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

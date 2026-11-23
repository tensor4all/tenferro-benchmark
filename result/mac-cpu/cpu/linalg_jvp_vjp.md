# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Benchmark commit: `d9f6406557a0d1066f37785ec85f9b9afdbb4781`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261123_172252/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261123_172252/run.yaml`
- Timestamp: `2026-11-23T17:22:52.769897+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261123_172252`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261123_172252/cpu_ops_t1_20261123_172252.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 20.839 ± 0.265 | 10.627 ± 0.173 | 4.963 ± 0.075 | 6.256 ± 0.103 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 116.245 ± 2.519 | 61.065 ± 0.609 | 27.377 ± 0.100 | 37.995 ± 0.674 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 14.473 ± 0.007 | 10.445 ± 0.094 | 4.605 ± 0.132 | 6.257 ± 0.072 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 92.209 ± 2.618 | 61.028 ± 0.616 | 25.144 ± 1.105 | 37.067 ± 0.449 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 10.367 ± 0.031 | 4.717 ± 0.080 | 1.919 ± 0.181 | 2.634 ± 0.011 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 70.834 ± 0.795 | 34.628 ± 0.200 | 14.283 ± 0.175 | 17.080 ± 0.034 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 9.475 ± 0.028 | 4.721 ± 0.019 | 1.189 ± 0.082 | 2.551 ± 0.019 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 67.773 ± 0.409 | 34.176 ± 1.360 | 7.958 ± 0.222 | 16.797 ± 0.058 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 14.877 ± 0.039 | 7.753 ± 0.021 | 2.490 ± 0.093 | 4.365 ± 0.039 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 102.102 ± 0.833 | 50.800 ± 0.344 | 13.222 ± 0.136 | 27.898 ± 0.396 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 12.996 ± 0.215 | 7.789 ± 0.097 | 2.375 ± 0.064 | 4.287 ± 0.060 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 91.736 ± 1.007 | 50.280 ± 0.266 | 12.674 ± 0.331 | 28.415 ± 0.897 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.629 ± 0.031 | 0.937 ± 0.001 | 0.425 ± 0.007 | 0.288 ± 0.004 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 9.368 ± 0.237 | 5.860 ± 0.104 | 1.682 ± 0.048 | 1.452 ± 0.045 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.072 ± 0.010 | 0.939 ± 0.005 | 0.544 ± 0.024 | 0.281 ± 0.007 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 6.283 ± 0.094 | 5.767 ± 0.066 | 2.687 ± 0.070 | 1.426 ± 0.013 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 38.935 ± 0.134 | 21.032 ± 0.163 | 9.500 ± 0.072 | 10.234 ± 0.047 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 233.182 ± 9.516 | 123.334 ± 0.629 | 54.181 ± 0.778 | 62.560 ± 8.535 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 26.384 ± 0.058 | 21.138 ± 0.046 | 8.617 ± 0.077 | 10.370 ± 0.147 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 163.539 ± 10.863 | 123.453 ± 0.535 | 47.611 ± 2.319 | 60.589 ± 3.006 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.101 ± 0.001 | 0.016 ± 0.000 | 0.076 ± 0.003 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.109 ± 0.003 | 0.017 ± 0.000 | 0.069 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.118 ± 0.002 | 0.026 ± 0.001 | 0.077 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.037 ± 0.000 | 0.016 ± 0.000 | 0.067 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.041 ± 0.001 | 0.017 ± 0.002 | 0.059 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.048 ± 0.001 | 0.025 ± 0.001 | 0.067 ± 0.000 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.333 ± 0.003 | 0.018 ± 0.000 | 0.113 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.351 ± 0.002 | 0.020 ± 0.001 | 0.102 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.354 ± 0.012 | 0.021 ± 0.001 | 0.106 ± 0.000 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.082 ± 0.001 | 0.016 ± 0.000 | 0.091 ± 0.008 | 0.007 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.086 ± 0.001 | 0.018 ± 0.001 | 0.081 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.088 ± 0.002 | 0.019 ± 0.001 | 0.084 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.444 ± 0.003 | 0.024 ± 0.000 | 0.086 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.460 ± 0.002 | 0.024 ± 0.000 | 0.078 ± 0.003 | 0.007 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.462 ± 0.012 | 0.030 ± 0.000 | 0.081 ± 0.004 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.075 ± 0.001 | 0.025 ± 0.000 | 0.085 ± 0.005 | 0.008 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.077 ± 0.000 | 0.024 ± 0.000 | 0.076 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.081 ± 0.001 | 0.028 ± 0.001 | 0.080 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.076 ± 0.001 | 0.014 ± 0.000 | 0.166 ± 0.022 | 0.007 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.079 ± 0.002 | 0.015 ± 0.000 | 0.152 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.081 ± 0.007 | 0.016 ± 0.000 | 0.146 ± 0.000 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.042 ± 0.001 | 0.014 ± 0.000 | 0.075 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.044 ± 0.001 | 0.015 ± 0.000 | 0.076 ± 0.000 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.045 ± 0.000 | 0.016 ± 0.000 | 0.072 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.093 ± 0.003 | 0.011 ± 0.000 | 0.095 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.099 ± 0.005 | 0.014 ± 0.000 | 0.100 ± 0.004 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.115 ± 0.001 | 0.024 ± 0.001 | 0.106 ± 0.001 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.039 ± 0.003 | 0.011 ± 0.000 | 0.072 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.043 ± 0.001 | 0.013 ± 0.000 | 0.064 ± 0.000 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.054 ± 0.001 | 0.024 ± 0.001 | 0.069 ± 0.000 | 0.019 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261123_174752/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261123_174752/run.yaml`
- Timestamp: `2026-11-23T17:47:52.092097+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261123_174752`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261123_174752/cpu_ops_t4_20261123_174752.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 21.653 ± 1.542 | 10.311 ± 0.075 | 5.123 ± 0.038 | 5.971 ± 0.084 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 112.687 ± 4.279 | 58.949 ± 5.635 | 28.979 ± 0.225 | 29.619 ± 0.310 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 14.817 ± 0.116 | 10.095 ± 0.046 | 4.714 ± 0.042 | 5.974 ± 0.218 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 77.532 ± 0.459 | 57.911 ± 0.066 | 26.320 ± 0.233 | 29.857 ± 0.347 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 8.050 ± 0.097 | 3.495 ± 0.011 | 2.389 ± 0.072 | 1.539 ± 0.175 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 43.276 ± 0.344 | 21.060 ± 0.564 | 14.802 ± 0.076 | 8.966 ± 0.556 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 7.331 ± 0.237 | 3.527 ± 0.019 | 1.629 ± 0.026 | 1.384 ± 0.013 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 39.934 ± 0.237 | 21.954 ± 0.947 | 8.396 ± 0.697 | 8.550 ± 0.111 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 12.176 ± 0.083 | 6.637 ± 0.060 | 2.733 ± 0.055 | 3.460 ± 0.106 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 65.073 ± 0.886 | 32.872 ± 0.420 | 13.020 ± 0.121 | 15.173 ± 0.328 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 9.396 ± 0.071 | 6.493 ± 0.052 | 2.590 ± 0.047 | 3.603 ± 0.216 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 54.482 ± 0.264 | 36.044 ± 3.591 | 12.428 ± 0.028 | 15.664 ± 0.222 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.853 ± 0.086 | 0.983 ± 0.009 | 0.500 ± 0.009 | 0.303 ± 0.002 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 7.659 ± 0.152 | 5.363 ± 0.078 | 1.713 ± 0.396 | 1.436 ± 0.011 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.239 ± 0.111 | 0.981 ± 0.012 | 0.653 ± 0.010 | 0.294 ± 0.003 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 5.271 ± 0.140 | 5.304 ± 0.201 | 2.818 ± 0.158 | 1.320 ± 0.023 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 40.338 ± 0.228 | 20.526 ± 0.152 | 9.752 ± 0.076 | 9.521 ± 0.173 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 232.829 ± 0.863 | 117.547 ± 0.616 | 52.058 ± 2.035 | 49.334 ± 0.351 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 26.733 ± 0.249 | 20.444 ± 0.138 | 8.737 ± 0.136 | 10.798 ± 0.243 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 159.608 ± 5.473 | 117.652 ± 0.593 | 46.259 ± 0.901 | 52.476 ± 0.651 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.118 ± 0.001 | 0.023 ± 0.000 | 0.064 ± 0.002 | 0.007 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.115 ± 0.002 | 0.024 ± 0.000 | 0.066 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.130 ± 0.002 | 0.032 ± 0.000 | 0.090 ± 0.004 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.052 ± 0.003 | 0.023 ± 0.000 | 0.055 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.051 ± 0.000 | 0.023 ± 0.000 | 0.060 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.061 ± 0.001 | 0.030 ± 0.000 | 0.079 ± 0.004 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.386 ± 0.011 | 0.027 ± 0.000 | 0.235 ± 0.008 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.363 ± 0.003 | 0.026 ± 0.000 | 0.284 ± 0.014 | 0.008 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.382 ± 0.002 | 0.028 ± 0.000 | 0.261 ± 0.010 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.114 ± 0.001 | 0.024 ± 0.000 | 0.177 ± 0.005 | 0.007 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.111 ± 0.001 | 0.025 ± 0.003 | 0.212 ± 0.014 | 0.009 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.119 ± 0.005 | 0.026 ± 0.001 | 0.207 ± 0.013 | 0.005 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.495 ± 0.012 | 0.036 ± 0.000 | 0.119 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.472 ± 0.001 | 0.035 ± 0.003 | 0.120 ± 0.001 | 0.008 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.505 ± 0.012 | 0.039 ± 0.000 | 0.146 ± 0.003 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.106 ± 0.001 | 0.036 ± 0.001 | 0.120 ± 0.003 | 0.008 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.102 ± 0.001 | 0.035 ± 0.000 | 0.121 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.109 ± 0.001 | 0.039 ± 0.001 | 0.154 ± 0.012 | 0.005 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.093 ± 0.003 | 0.021 ± 0.002 | 0.147 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.089 ± 0.000 | 0.021 ± 0.000 | 0.158 ± 0.011 | 0.008 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.095 ± 0.002 | 0.022 ± 0.000 | 0.185 ± 0.014 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.057 ± 0.001 | 0.021 ± 0.000 | 0.073 ± 0.005 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.054 ± 0.000 | 0.021 ± 0.000 | 0.076 ± 0.002 | 0.007 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.057 ± 0.001 | 0.022 ± 0.000 | 0.090 ± 0.006 | 0.007 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.103 ± 0.001 | 0.016 ± 0.000 | 0.092 ± 0.003 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.102 ± 0.000 | 0.018 ± 0.000 | 0.093 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.125 ± 0.001 | 0.027 ± 0.000 | 0.113 ± 0.003 | 0.018 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.050 ± 0.001 | 0.015 ± 0.000 | 0.060 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.050 ± 0.001 | 0.017 ± 0.000 | 0.061 ± 0.001 | 0.012 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.063 ± 0.000 | 0.026 ± 0.000 | 0.080 ± 0.008 | 0.018 ± 0.003 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `35979dcd7110a29d0728b2e128c72914f4c3f67b`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261111_172341/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261111_172341/run.yaml`
- Timestamp: `2026-11-11T17:23:41.760374+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261111_172341`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261111_172341/cpu_ops_t1_20261111_172341.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 12.605 ± 1.991 | 7.298 ± 0.057 | 5.009 ± 0.391 | 6.885 ± 0.419 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 54.256 ± 1.426 | 48.794 ± 0.942 | 29.343 ± 0.284 | 37.186 ± 0.996 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 5.193 ± 0.781 | 7.238 ± 0.070 | 4.801 ± 0.256 | 6.575 ± 0.112 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 28.028 ± 0.284 | 48.591 ± 1.766 | 26.929 ± 0.494 | 36.268 ± 1.584 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 3.021 ± 0.456 | 1.054 ± 0.005 | 1.981 ± 0.084 | 2.826 ± 0.173 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 15.953 ± 0.594 | 8.120 ± 0.737 | 14.644 ± 0.100 | 19.342 ± 1.500 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 3.035 ± 0.442 | 1.347 ± 0.020 | 1.216 ± 0.033 | 3.019 ± 0.389 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 13.858 ± 1.084 | 8.437 ± 0.047 | 8.168 ± 0.073 | 19.829 ± 1.196 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 5.588 ± 0.268 | 3.207 ± 0.013 | 2.567 ± 0.208 | 4.696 ± 0.261 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 37.092 ± 4.434 | 20.804 ± 0.750 | 14.698 ± 0.340 | 28.258 ± 0.738 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 3.741 ± 0.215 | 3.226 ± 0.015 | 2.386 ± 0.037 | 4.344 ± 0.150 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 32.110 ± 4.344 | 20.243 ± 0.432 | 13.961 ± 0.150 | 28.742 ± 0.272 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.673 ± 0.089 | 0.523 ± 0.070 | 0.441 ± 0.037 | 0.430 ± 0.152 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 2.879 ± 0.141 | 2.721 ± 0.175 | 1.763 ± 0.146 | 1.892 ± 0.059 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.386 ± 0.084 | 0.539 ± 0.013 | 0.537 ± 0.011 | 0.425 ± 0.087 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 1.475 ± 0.046 | 2.601 ± 0.035 | 2.821 ± 0.098 | 1.802 ± 0.147 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 29.401 ± 1.474 | 13.031 ± 0.052 | 9.527 ± 0.132 | 10.872 ± 0.175 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 120.055 ± 6.667 | 88.228 ± 3.853 | 50.846 ± 1.213 | 64.457 ± 12.811 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 12.883 ± 1.562 | 13.071 ± 0.069 | 8.697 ± 0.215 | 11.029 ± 0.310 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 71.922 ± 12.036 | 80.685 ± 0.917 | 45.994 ± 0.196 | 61.595 ± 1.034 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.096 ± 0.001 | 0.015 ± 0.001 | 0.061 ± 0.002 | 0.007 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.095 ± 0.001 | 0.017 ± 0.002 | 0.075 ± 0.004 | 0.009 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.167 ± 0.024 | 0.025 ± 0.000 | 0.076 ± 0.004 | 0.010 ± 0.003 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.031 ± 0.001 | 0.016 ± 0.000 | 0.053 ± 0.001 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.033 ± 0.003 | 0.018 ± 0.001 | 0.066 ± 0.007 | 0.010 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.044 ± 0.017 | 0.022 ± 0.000 | 0.066 ± 0.000 | 0.010 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.335 ± 0.014 | 0.018 ± 0.001 | 0.095 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.339 ± 0.004 | 0.018 ± 0.001 | 0.112 ± 0.016 | 0.009 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.388 ± 0.017 | 0.018 ± 0.000 | 0.098 ± 0.003 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.075 ± 0.001 | 0.017 ± 0.001 | 0.077 ± 0.004 | 0.007 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.072 ± 0.004 | 0.017 ± 0.000 | 0.091 ± 0.031 | 0.010 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.091 ± 0.017 | 0.020 ± 0.001 | 0.081 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.466 ± 0.014 | 0.027 ± 0.000 | 0.071 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.435 ± 0.002 | 0.025 ± 0.000 | 0.084 ± 0.005 | 0.009 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.480 ± 0.038 | 0.029 ± 0.000 | 0.092 ± 0.029 | 0.012 ± 0.016 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.066 ± 0.001 | 0.027 ± 0.000 | 0.070 ± 0.004 | 0.008 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.063 ± 0.000 | 0.025 ± 0.000 | 0.080 ± 0.007 | 0.010 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.091 ± 0.018 | 0.032 ± 0.000 | 0.083 ± 0.012 | 0.018 ± 0.008 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.069 ± 0.001 | 0.014 ± 0.000 | 0.146 ± 0.027 | 0.007 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.065 ± 0.005 | 0.014 ± 0.001 | 0.169 ± 0.040 | 0.011 ± 0.004 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.076 ± 0.022 | 0.014 ± 0.000 | 0.144 ± 0.006 | 0.008 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.033 ± 0.001 | 0.014 ± 0.001 | 0.071 ± 0.002 | 0.008 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.031 ± 0.001 | 0.014 ± 0.000 | 0.085 ± 0.032 | 0.011 ± 0.012 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.062 ± 0.011 | 0.014 ± 0.000 | 0.129 ± 0.006 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.085 ± 0.007 | 0.011 ± 0.001 | 0.124 ± 0.025 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.084 ± 0.000 | 0.014 ± 0.000 | 0.104 ± 0.004 | 0.012 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.107 ± 0.002 | 0.027 ± 0.000 | 0.102 ± 0.019 | 0.020 ± 0.011 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.031 ± 0.001 | 0.011 ± 0.001 | 0.058 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.031 ± 0.001 | 0.013 ± 0.000 | 0.073 ± 0.006 | 0.012 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.040 ± 0.002 | 0.024 ± 0.000 | 0.080 ± 0.012 | 0.020 ± 0.003 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261111_175728/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261111_175728/run.yaml`
- Timestamp: `2026-11-11T17:57:28.429255+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261111_175728`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261111_175728/cpu_ops_t4_20261111_175728.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 10.217 ± 0.371 | 8.204 ± 2.493 | 5.917 ± 1.227 | 8.263 ± 0.898 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 69.528 ± 10.145 | 66.952 ± 12.116 | 37.774 ± 8.718 | 42.727 ± 8.784 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 5.241 ± 0.046 | 7.589 ± 0.495 | 5.446 ± 1.033 | 9.776 ± 4.289 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 29.725 ± 2.192 | 60.806 ± 6.247 | 33.270 ± 2.450 | 38.838 ± 2.494 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 2.871 ± 0.156 | 1.054 ± 0.003 | 2.878 ± 0.318 | 2.463 ± 1.128 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 20.343 ± 2.073 | 12.201 ± 3.666 | 17.283 ± 1.672 | 14.711 ± 3.995 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 2.415 ± 0.211 | 1.155 ± 0.069 | 1.774 ± 0.611 | 2.580 ± 0.960 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 16.107 ± 2.422 | 9.671 ± 0.230 | 9.851 ± 1.812 | 20.153 ± 4.618 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 5.933 ± 0.806 | 3.901 ± 0.636 | 3.483 ± 1.165 | 4.495 ± 1.468 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 56.756 ± 12.536 | 37.245 ± 13.258 | 14.488 ± 1.757 | 26.365 ± 14.969 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 3.515 ± 0.040 | 3.290 ± 0.064 | 3.049 ± 1.182 | 4.292 ± 1.619 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 27.411 ± 0.925 | 35.892 ± 2.218 | 17.554 ± 1.343 | 32.778 ± 6.916 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.198 ± 0.571 | 0.455 ± 0.009 | 0.637 ± 0.145 | 0.340 ± 0.016 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 4.129 ± 0.586 | 3.944 ± 0.531 | 2.256 ± 0.648 | 2.431 ± 0.783 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.420 ± 0.034 | 0.466 ± 0.015 | 0.825 ± 0.157 | 0.848 ± 0.315 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 2.978 ± 1.708 | 4.105 ± 0.958 | 3.355 ± 0.909 | 2.975 ± 3.034 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 18.681 ± 1.645 | 13.554 ± 1.221 | 11.135 ± 2.361 | 11.450 ± 0.610 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 117.618 ± 14.527 | 86.610 ± 7.660 | 67.920 ± 8.212 | 71.221 ± 6.581 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 9.493 ± 0.668 | 13.234 ± 0.062 | 9.794 ± 1.560 | 11.110 ± 0.413 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 61.883 ± 3.275 | 95.785 ± 29.428 | 55.392 ± 5.740 | 70.783 ± 8.852 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.126 ± 0.006 | 0.023 ± 0.003 | 0.071 ± 0.004 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.159 ± 0.020 | 0.025 ± 0.000 | 0.106 ± 0.029 | 0.010 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.143 ± 0.010 | 0.035 ± 0.001 | 0.086 ± 0.017 | 0.023 ± 0.007 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.047 ± 0.002 | 0.023 ± 0.003 | 0.062 ± 0.004 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.055 ± 0.008 | 0.027 ± 0.005 | 0.101 ± 0.038 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.061 ± 0.005 | 0.030 ± 0.001 | 0.083 ± 0.011 | 0.023 ± 0.004 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.450 ± 0.019 | 0.027 ± 0.002 | 0.328 ± 0.076 | 0.012 ± 0.003 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.472 ± 0.042 | 0.027 ± 0.000 | 0.200 ± 0.098 | 0.009 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.463 ± 0.056 | 0.027 ± 0.000 | 0.326 ± 0.038 | 0.009 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.110 ± 0.002 | 0.023 ± 0.000 | 0.233 ± 0.074 | 0.010 ± 0.004 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.125 ± 0.009 | 0.032 ± 0.002 | 0.253 ± 0.273 | 0.008 ± 0.003 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.125 ± 0.029 | 0.031 ± 0.001 | 0.293 ± 0.059 | 0.017 ± 0.007 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.564 ± 0.020 | 0.039 ± 0.001 | 0.167 ± 0.022 | 0.008 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.633 ± 0.021 | 0.041 ± 0.002 | 0.182 ± 0.160 | 0.012 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.615 ± 0.033 | 0.051 ± 0.011 | 0.211 ± 0.106 | 0.014 ± 0.004 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.102 ± 0.014 | 0.041 ± 0.004 | 0.168 ± 0.024 | 0.009 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.128 ± 0.005 | 0.040 ± 0.003 | 0.242 ± 0.070 | 0.011 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.119 ± 0.011 | 0.054 ± 0.004 | 0.180 ± 0.019 | 0.017 ± 0.015 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.084 ± 0.006 | 0.022 ± 0.003 | 0.174 ± 0.029 | 0.011 ± 0.020 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.095 ± 0.006 | 0.024 ± 0.004 | 0.253 ± 0.132 | 0.008 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.092 ± 0.006 | 0.022 ± 0.001 | 0.217 ± 0.058 | 0.008 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.061 ± 0.010 | 0.021 ± 0.001 | 0.080 ± 0.004 | 0.010 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.057 ± 0.015 | 0.021 ± 0.002 | 0.102 ± 0.066 | 0.007 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.044 ± 0.008 | 0.024 ± 0.003 | 0.084 ± 0.019 | 0.009 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.108 ± 0.008 | 0.017 ± 0.000 | 0.096 ± 0.007 | 0.011 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.116 ± 0.007 | 0.020 ± 0.003 | 0.215 ± 0.144 | 0.025 ± 0.016 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.135 ± 0.005 | 0.036 ± 0.008 | 0.129 ± 0.025 | 0.026 ± 0.019 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.044 ± 0.002 | 0.015 ± 0.000 | 0.069 ± 0.009 | 0.020 ± 0.012 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.051 ± 0.002 | 0.019 ± 0.001 | 0.092 ± 0.050 | 0.018 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.058 ± 0.003 | 0.030 ± 0.001 | 0.080 ± 0.006 | 0.025 ± 0.004 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

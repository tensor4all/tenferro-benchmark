# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `f3a05ec9a2a7262482348fd5bddccbe90389866b`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261113_172248/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261113_172248/run.yaml`
- Timestamp: `2026-11-13T17:22:48.940279+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261113_172248`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261113_172248/cpu_ops_t1_20261113_172248.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 11.869 ± 0.248 | 9.612 ± 0.471 | 7.023 ± 2.268 | 8.637 ± 2.028 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 68.239 ± 10.794 | 55.743 ± 3.309 | 45.577 ± 13.738 | 62.110 ± 5.214 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 6.140 ± 1.286 | 9.372 ± 0.753 | 5.700 ± 0.521 | 8.784 ± 4.492 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 36.506 ± 4.852 | 53.868 ± 2.455 | 33.949 ± 22.827 | 47.086 ± 10.271 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 3.313 ± 0.608 | 1.354 ± 0.035 | 2.558 ± 0.207 | 2.928 ± 0.265 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 19.388 ± 0.887 | 12.347 ± 2.302 | 19.719 ± 11.764 | 26.997 ± 5.904 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 2.625 ± 0.252 | 1.508 ± 0.104 | 1.594 ± 0.425 | 5.687 ± 2.054 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 19.898 ± 1.267 | 12.539 ± 4.042 | 11.115 ± 7.215 | 31.519 ± 6.650 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 6.557 ± 0.218 | 4.459 ± 0.293 | 3.553 ± 3.009 | 9.088 ± 5.942 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 37.242 ± 8.269 | 22.373 ± 0.970 | 24.335 ± 3.399 | 36.262 ± 3.421 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 4.170 ± 0.217 | 4.553 ± 0.153 | 3.221 ± 0.400 | 10.168 ± 2.704 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 33.225 ± 16.011 | 22.389 ± 3.767 | 23.683 ± 3.343 | 36.357 ± 7.240 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.839 ± 0.104 | 0.632 ± 0.030 | 0.550 ± 0.031 | 0.597 ± 0.273 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 4.200 ± 0.614 | 3.355 ± 0.801 | 2.713 ± 3.017 | 2.164 ± 0.271 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.513 ± 0.110 | 0.549 ± 0.025 | 0.713 ± 0.115 | 0.777 ± 0.223 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 2.138 ± 0.242 | 3.473 ± 0.740 | 4.841 ± 2.583 | 1.800 ± 0.187 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 21.404 ± 1.237 | 16.743 ± 1.241 | 16.188 ± 7.072 | 13.421 ± 1.348 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 127.501 ± 12.786 | 104.009 ± 4.205 | 79.142 ± 15.616 | 81.714 ± 17.910 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 10.720 ± 0.553 | 16.366 ± 1.742 | 11.167 ± 0.405 | 13.845 ± 4.125 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 60.245 ± 1.601 | 104.808 ± 7.185 | 67.967 ± 7.273 | 74.641 ± 24.574 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.119 ± 0.023 | 0.025 ± 0.008 | 0.103 ± 0.121 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.245 ± 0.101 | 0.026 ± 0.023 | 0.084 ± 0.003 | 0.016 ± 0.011 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.156 ± 0.006 | 0.027 ± 0.001 | 0.240 ± 0.195 | 0.022 ± 0.015 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.049 ± 0.059 | 0.023 ± 0.008 | 0.091 ± 0.022 | 0.010 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.122 ± 0.022 | 0.033 ± 0.008 | 0.071 ± 0.003 | 0.022 ± 0.004 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.050 ± 0.004 | 0.027 ± 0.001 | 0.125 ± 0.122 | 0.010 ± 0.016 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.623 ± 0.306 | 0.026 ± 0.008 | 0.155 ± 0.013 | 0.011 ± 0.005 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.649 ± 0.661 | 0.022 ± 0.007 | 0.125 ± 0.007 | 0.015 ± 0.004 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.477 ± 0.087 | 0.022 ± 0.002 | 0.212 ± 0.032 | 0.017 ± 0.009 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.095 ± 0.089 | 0.029 ± 0.016 | 0.119 ± 0.051 | 0.008 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.143 ± 0.090 | 0.022 ± 0.017 | 0.096 ± 0.002 | 0.016 ± 0.004 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.169 ± 0.098 | 0.024 ± 0.002 | 0.143 ± 0.085 | 0.018 ± 0.006 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.883 ± 0.349 | 0.054 ± 0.012 | 0.111 ± 0.013 | 0.010 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 1.402 ± 0.501 | 0.039 ± 0.009 | 0.104 ± 0.026 | 0.011 ± 0.005 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.604 ± 0.125 | 0.034 ± 0.001 | 0.100 ± 0.035 | 0.018 ± 0.009 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.172 ± 0.169 | 0.036 ± 0.018 | 0.118 ± 0.011 | 0.010 ± 0.004 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.209 ± 0.033 | 0.048 ± 0.061 | 0.091 ± 0.004 | 0.014 ± 0.006 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.087 ± 0.005 | 0.036 ± 0.001 | 0.257 ± 0.174 | 0.014 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.134 ± 0.021 | 0.017 ± 0.005 | 0.282 ± 0.159 | 0.011 ± 0.003 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.193 ± 0.057 | 0.029 ± 0.009 | 0.181 ± 0.010 | 0.017 ± 0.007 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.131 ± 0.072 | 0.020 ± 0.001 | 0.334 ± 0.264 | 0.018 ± 0.007 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.053 ± 0.027 | 0.016 ± 0.006 | 0.092 ± 0.023 | 0.007 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.132 ± 0.160 | 0.018 ± 0.004 | 0.092 ± 0.003 | 0.013 ± 0.011 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.058 ± 0.046 | 0.018 ± 0.001 | 0.282 ± 0.211 | 0.016 ± 0.008 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.266 ± 0.145 | 0.017 ± 0.020 | 0.144 ± 0.033 | 0.011 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.342 ± 0.117 | 0.043 ± 0.014 | 0.119 ± 0.011 | 0.012 ± 0.005 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.143 ± 0.008 | 0.034 ± 0.020 | 0.206 ± 0.097 | 0.020 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.069 ± 0.071 | 0.013 ± 0.002 | 0.088 ± 0.017 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.088 ± 0.033 | 0.034 ± 0.024 | 0.076 ± 0.002 | 0.014 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.058 ± 0.013 | 0.029 ± 0.000 | 0.125 ± 0.131 | 0.026 ± 0.009 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261113_180121/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261113_180121/run.yaml`
- Timestamp: `2026-11-13T18:01:21.706892+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261113_180121`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261113_180121/cpu_ops_t4_20261113_180121.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 11.315 ± 0.817 | 10.003 ± 4.599 | 6.473 ± 0.179 | 5.205 ± 0.092 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 58.316 ± 3.012 | 58.305 ± 1.377 | 33.313 ± 0.188 | 50.020 ± 10.105 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 5.411 ± 2.341 | 8.787 ± 4.692 | 6.184 ± 0.493 | 7.240 ± 2.205 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 40.545 ± 3.743 | 56.548 ± 2.097 | 31.074 ± 4.944 | 34.829 ± 4.918 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 3.591 ± 1.250 | 1.271 ± 0.386 | 3.584 ± 0.122 | 2.613 ± 0.115 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 24.255 ± 3.371 | 10.369 ± 0.521 | 27.721 ± 10.168 | 19.587 ± 11.729 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 2.794 ± 0.387 | 1.508 ± 1.198 | 2.609 ± 0.579 | 2.597 ± 0.478 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 22.547 ± 4.765 | 11.189 ± 0.845 | 16.518 ± 5.660 | 17.275 ± 4.456 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 5.992 ± 1.013 | 4.140 ± 1.006 | 3.960 ± 0.240 | 5.336 ± 1.054 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 38.110 ± 1.721 | 27.264 ± 1.613 | 29.513 ± 7.727 | 33.605 ± 4.394 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 3.839 ± 0.296 | 4.657 ± 1.646 | 3.681 ± 0.144 | 2.738 ± 0.140 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 26.522 ± 10.687 | 26.911 ± 12.989 | 27.706 ± 1.491 | 38.773 ± 5.289 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.899 ± 0.178 | 0.678 ± 0.439 | 0.710 ± 0.084 | 0.471 ± 0.179 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 5.531 ± 0.100 | 3.895 ± 0.346 | 2.533 ± 0.576 | 2.655 ± 3.040 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.473 ± 0.063 | 0.546 ± 0.022 | 0.904 ± 0.110 | 1.133 ± 0.242 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 3.145 ± 0.806 | 3.564 ± 0.190 | 5.283 ± 2.639 | 1.498 ± 0.098 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 23.579 ± 3.526 | 15.007 ± 0.160 | 11.421 ± 1.631 | 13.125 ± 1.291 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 116.721 ± 7.576 | 99.603 ± 6.758 | 72.832 ± 11.886 | 94.419 ± 24.457 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 10.070 ± 3.030 | 19.709 ± 3.009 | 10.892 ± 0.505 | 12.406 ± 1.071 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 65.656 ± 2.652 | 104.267 ± 3.370 | 65.266 ± 4.776 | 79.911 ± 19.847 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.170 ± 0.037 | 0.026 ± 0.004 | 0.170 ± 0.064 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.180 ± 0.016 | 0.046 ± 0.002 | 0.092 ± 0.006 | 0.010 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.150 ± 0.049 | 0.041 ± 0.021 | 0.095 ± 0.006 | 0.015 ± 0.007 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.059 ± 0.009 | 0.023 ± 0.006 | 0.182 ± 0.118 | 0.009 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.065 ± 0.041 | 0.042 ± 0.010 | 0.077 ± 0.003 | 0.010 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.067 ± 0.002 | 0.045 ± 0.044 | 0.081 ± 0.006 | 0.010 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.607 ± 0.364 | 0.046 ± 0.016 | 0.509 ± 0.441 | 0.008 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.429 ± 0.012 | 0.044 ± 0.013 | 0.477 ± 0.058 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.498 ± 0.011 | 0.051 ± 0.013 | 0.404 ± 0.262 | 0.018 ± 0.005 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.122 ± 0.005 | 0.029 ± 0.002 | 0.232 ± 0.070 | 0.008 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.164 ± 0.038 | 0.036 ± 0.031 | 0.392 ± 0.089 | 0.009 ± 0.004 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.166 ± 0.065 | 0.051 ± 0.021 | 0.845 ± 0.111 | 0.017 ± 0.010 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 1.120 ± 0.791 | 0.051 ± 0.003 | 0.219 ± 0.196 | 0.016 ± 0.006 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.797 ± 0.219 | 0.062 ± 0.006 | 0.190 ± 0.019 | 0.010 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.697 ± 0.124 | 0.077 ± 0.015 | 0.194 ± 0.027 | 0.014 ± 0.007 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.181 ± 0.064 | 0.047 ± 0.001 | 0.300 ± 0.041 | 0.009 ± 0.003 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.149 ± 0.074 | 0.075 ± 0.015 | 0.168 ± 0.050 | 0.010 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.126 ± 0.007 | 0.069 ± 0.015 | 0.194 ± 0.046 | 0.015 ± 0.005 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.120 ± 0.011 | 0.022 ± 0.001 | 0.184 ± 0.016 | 0.007 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.115 ± 0.008 | 0.035 ± 0.015 | 0.203 ± 0.017 | 0.008 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.144 ± 0.042 | 0.047 ± 0.007 | 0.985 ± 0.611 | 0.015 ± 0.006 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.078 ± 0.024 | 0.024 ± 0.001 | 0.087 ± 0.003 | 0.008 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.069 ± 0.051 | 0.032 ± 0.011 | 0.106 ± 0.016 | 0.025 ± 0.026 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.064 ± 0.069 | 0.034 ± 0.008 | 0.370 ± 0.163 | 0.019 ± 0.010 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.106 ± 0.023 | 0.019 ± 0.004 | 0.110 ± 0.038 | 0.012 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.131 ± 0.024 | 0.032 ± 0.027 | 0.128 ± 0.023 | 0.016 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.187 ± 0.054 | 0.051 ± 0.021 | 0.133 ± 0.021 | 0.035 ± 0.019 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.064 ± 0.036 | 0.018 ± 0.000 | 0.120 ± 0.190 | 0.024 ± 0.007 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.063 ± 0.022 | 0.021 ± 0.006 | 0.081 ± 0.025 | 0.015 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.054 ± 0.003 | 0.081 ± 0.014 | 0.091 ± 0.004 | 0.045 ± 0.016 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

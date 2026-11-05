# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`
- Benchmark commit: `ce2efd37d9e132703dfd2ebefbf4353249cf53ea`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261105_172310/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261105_172310/run.yaml`
- Timestamp: `2026-11-05T17:23:10.950764+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261105_172310`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261105_172310/cpu_ops_t1_20261105_172310.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 9.383 ± 0.070 | 4.683 ± 0.243 | 4.950 ± 0.240 | 6.271 ± 0.615 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 45.848 ± 1.594 | 23.226 ± 1.610 | 28.168 ± 1.958 | 35.280 ± 6.335 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 5.230 ± 0.160 | 4.473 ± 0.109 | 4.606 ± 0.310 | 5.925 ± 0.364 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 24.360 ± 1.230 | 21.326 ± 0.941 | 25.553 ± 0.797 | 34.273 ± 5.618 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 1.248 ± 0.169 | 0.801 ± 0.005 | 1.710 ± 0.168 | 2.473 ± 0.154 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 7.058 ± 0.384 | 5.096 ± 0.131 | 12.797 ± 1.405 | 18.662 ± 1.428 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 1.205 ± 0.041 | 0.812 ± 0.007 | 0.992 ± 0.063 | 2.404 ± 0.196 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 6.194 ± 0.864 | 5.234 ± 0.195 | 7.739 ± 0.327 | 19.773 ± 1.638 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 3.512 ± 0.045 | 2.026 ± 0.081 | 2.571 ± 0.234 | 4.032 ± 0.211 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 17.975 ± 0.227 | 11.102 ± 0.044 | 13.524 ± 1.014 | 26.813 ± 2.682 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 2.128 ± 0.084 | 1.965 ± 0.042 | 2.309 ± 0.120 | 3.816 ± 0.351 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 11.953 ± 0.235 | 11.213 ± 0.053 | 13.695 ± 0.584 | 27.308 ± 0.655 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.645 ± 0.050 | 0.233 ± 0.002 | 0.403 ± 0.020 | 0.271 ± 0.018 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 2.828 ± 0.401 | 1.380 ± 0.392 | 1.662 ± 0.475 | 1.340 ± 0.212 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.388 ± 0.023 | 0.250 ± 0.005 | 0.512 ± 0.083 | 0.291 ± 0.040 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 1.781 ± 0.060 | 1.042 ± 0.092 | 2.599 ± 0.469 | 1.467 ± 0.131 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 14.849 ± 0.357 | 7.683 ± 0.139 | 8.670 ± 0.183 | 9.507 ± 1.439 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 75.666 ± 2.735 | 40.412 ± 0.786 | 42.432 ± 3.511 | 50.588 ± 5.593 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 7.639 ± 0.144 | 7.891 ± 0.191 | 7.380 ± 0.613 | 8.938 ± 0.905 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 38.493 ± 0.400 | 39.435 ± 2.136 | 36.384 ± 1.680 | 47.488 ± 2.194 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.114 ± 0.003 | 0.016 ± 0.000 | 0.061 ± 0.002 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.126 ± 0.006 | 0.019 ± 0.002 | 0.060 ± 0.006 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.132 ± 0.001 | 0.022 ± 0.001 | 0.080 ± 0.006 | 0.009 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.062 ± 0.003 | 0.016 ± 0.000 | 0.046 ± 0.004 | 0.006 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.068 ± 0.004 | 0.018 ± 0.002 | 0.051 ± 0.004 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.083 ± 0.004 | 0.017 ± 0.002 | 0.068 ± 0.008 | 0.009 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.127 ± 0.008 | 0.021 ± 0.005 | 0.084 ± 0.010 | 0.006 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.137 ± 0.007 | 0.016 ± 0.001 | 0.095 ± 0.003 | 0.007 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.144 ± 0.008 | 0.016 ± 0.001 | 0.106 ± 0.008 | 0.006 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.097 ± 0.005 | 0.019 ± 0.006 | 0.072 ± 0.011 | 0.006 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.092 ± 0.011 | 0.014 ± 0.001 | 0.069 ± 0.005 | 0.007 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.109 ± 0.006 | 0.019 ± 0.000 | 0.083 ± 0.005 | 0.005 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.270 ± 0.007 | 0.027 ± 0.002 | 0.065 ± 0.002 | 0.007 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.289 ± 0.014 | 0.026 ± 0.005 | 0.076 ± 0.007 | 0.008 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.276 ± 0.014 | 0.025 ± 0.001 | 0.081 ± 0.006 | 0.006 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.159 ± 0.010 | 0.027 ± 0.001 | 0.065 ± 0.003 | 0.006 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.160 ± 0.010 | 0.027 ± 0.014 | 0.067 ± 0.005 | 0.009 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.161 ± 0.002 | 0.030 ± 0.001 | 0.081 ± 0.004 | 0.006 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.096 ± 0.007 | 0.014 ± 0.003 | 0.140 ± 0.013 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.094 ± 0.002 | 0.011 ± 0.002 | 0.140 ± 0.012 | 0.006 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.102 ± 0.034 | 0.011 ± 0.000 | 0.145 ± 0.030 | 0.006 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.063 ± 0.005 | 0.012 ± 0.009 | 0.070 ± 0.003 | 0.007 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.067 ± 0.002 | 0.010 ± 0.001 | 0.067 ± 0.003 | 0.007 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.066 ± 0.005 | 0.011 ± 0.000 | 0.068 ± 0.004 | 0.007 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.109 ± 0.005 | 0.010 ± 0.000 | 0.088 ± 0.016 | 0.009 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.111 ± 0.009 | 0.013 ± 0.001 | 0.093 ± 0.014 | 0.011 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.134 ± 0.000 | 0.022 ± 0.001 | 0.102 ± 0.007 | 0.019 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.060 ± 0.002 | 0.011 ± 0.000 | 0.053 ± 0.002 | 0.008 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.067 ± 0.003 | 0.012 ± 0.001 | 0.055 ± 0.003 | 0.012 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.071 ± 0.004 | 0.018 ± 0.001 | 0.061 ± 0.007 | 0.020 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261105_174004/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261105_174004/run.yaml`
- Timestamp: `2026-11-05T17:40:04.858371+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261105_174004`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261105_174004/cpu_ops_t4_20261105_174004.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 9.788 ± 0.384 | 5.305 ± 0.109 | 4.973 ± 0.554 | 5.353 ± 0.630 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 60.357 ± 3.091 | 23.371 ± 0.376 | 23.360 ± 2.605 | 28.068 ± 3.511 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 5.452 ± 0.295 | 5.081 ± 0.132 | 4.530 ± 0.376 | 5.147 ± 0.109 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 26.199 ± 2.548 | 23.241 ± 1.509 | 21.260 ± 0.474 | 30.505 ± 3.650 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 1.276 ± 0.040 | 0.918 ± 0.041 | 2.372 ± 0.085 | 1.499 ± 0.095 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 7.584 ± 0.450 | 5.715 ± 0.109 | 10.206 ± 0.481 | 8.866 ± 0.358 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 1.239 ± 0.030 | 1.024 ± 0.080 | 1.631 ± 0.059 | 1.422 ± 0.088 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 6.505 ± 0.052 | 6.369 ± 0.079 | 5.645 ± 0.048 | 7.484 ± 0.671 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 3.661 ± 0.063 | 2.063 ± 0.163 | 2.474 ± 0.118 | 3.063 ± 0.130 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 18.226 ± 4.946 | 10.716 ± 0.193 | 10.521 ± 0.156 | 11.905 ± 0.590 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 2.293 ± 0.013 | 2.188 ± 0.078 | 2.353 ± 0.177 | 2.617 ± 0.139 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 11.517 ± 6.371 | 10.681 ± 0.244 | 10.522 ± 0.191 | 14.558 ± 3.554 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.592 ± 0.041 | 0.284 ± 0.039 | 0.500 ± 0.032 | 0.343 ± 0.027 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 3.033 ± 0.177 | 1.299 ± 0.016 | 1.213 ± 0.013 | 1.562 ± 0.234 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.430 ± 0.065 | 0.272 ± 0.006 | 0.620 ± 0.039 | 0.312 ± 0.022 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.910 ± 0.058 | 1.212 ± 0.015 | 1.981 ± 0.015 | 1.226 ± 0.072 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 15.338 ± 0.445 | 8.213 ± 0.259 | 8.346 ± 0.242 | 7.902 ± 0.321 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 80.225 ± 3.525 | 37.109 ± 2.188 | 39.991 ± 2.994 | 39.308 ± 3.015 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 7.217 ± 0.791 | 8.101 ± 0.231 | 7.375 ± 0.026 | 7.877 ± 0.330 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 42.162 ± 3.090 | 39.676 ± 0.456 | 34.298 ± 1.513 | 36.256 ± 4.411 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.124 ± 0.004 | 0.017 ± 0.000 | 0.056 ± 0.003 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.142 ± 0.008 | 0.021 ± 0.002 | 0.057 ± 0.002 | 0.009 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.146 ± 0.003 | 0.028 ± 0.000 | 0.073 ± 0.002 | 0.009 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.068 ± 0.002 | 0.016 ± 0.000 | 0.048 ± 0.000 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.078 ± 0.005 | 0.018 ± 0.001 | 0.049 ± 0.001 | 0.009 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.088 ± 0.016 | 0.021 ± 0.000 | 0.059 ± 0.002 | 0.009 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.144 ± 0.002 | 0.020 ± 0.001 | 0.338 ± 0.030 | 0.006 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.145 ± 0.007 | 0.021 ± 0.002 | 0.326 ± 0.035 | 0.008 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.138 ± 0.005 | 0.022 ± 0.001 | 0.340 ± 0.027 | 0.006 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.106 ± 0.007 | 0.018 ± 0.000 | 0.241 ± 0.018 | 0.007 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.102 ± 0.004 | 0.019 ± 0.000 | 0.232 ± 0.022 | 0.008 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.115 ± 0.005 | 0.026 ± 0.000 | 0.236 ± 0.050 | 0.007 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.265 ± 0.022 | 0.029 ± 0.002 | 0.163 ± 0.022 | 0.007 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.300 ± 0.031 | 0.032 ± 0.001 | 0.164 ± 0.019 | 0.008 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.299 ± 0.002 | 0.032 ± 0.000 | 0.166 ± 0.014 | 0.007 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.151 ± 0.004 | 0.030 ± 0.001 | 0.153 ± 0.024 | 0.006 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.170 ± 0.005 | 0.031 ± 0.001 | 0.160 ± 0.027 | 0.009 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.175 ± 0.005 | 0.038 ± 0.001 | 0.175 ± 0.028 | 0.007 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.108 ± 0.004 | 0.013 ± 0.000 | 0.133 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.103 ± 0.007 | 0.014 ± 0.000 | 0.133 ± 0.003 | 0.007 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.101 ± 0.006 | 0.014 ± 0.000 | 0.189 ± 0.030 | 0.006 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.071 ± 0.000 | 0.013 ± 0.000 | 0.062 ± 0.002 | 0.007 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.071 ± 0.004 | 0.016 ± 0.000 | 0.062 ± 0.004 | 0.007 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.068 ± 0.006 | 0.014 ± 0.000 | 0.079 ± 0.011 | 0.006 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.103 ± 0.007 | 0.011 ± 0.000 | 0.078 ± 0.002 | 0.009 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.123 ± 0.010 | 0.013 ± 0.000 | 0.080 ± 0.001 | 0.014 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.136 ± 0.004 | 0.026 ± 0.003 | 0.100 ± 0.001 | 0.018 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.062 ± 0.006 | 0.011 ± 0.000 | 0.051 ± 0.000 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.070 ± 0.004 | 0.013 ± 0.000 | 0.053 ± 0.002 | 0.014 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.077 ± 0.004 | 0.019 ± 0.000 | 0.059 ± 0.002 | 0.019 ± 0.002 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

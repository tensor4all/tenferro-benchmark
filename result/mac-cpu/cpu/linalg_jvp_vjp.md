# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`
- Benchmark commit: `ef930a2e0f075f6b42bbbf9608afa9ea69ed412b`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261104_172845/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261104_172845/run.yaml`
- Timestamp: `2026-11-04T17:28:45.718036+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261104_172845`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261104_172845/cpu_ops_t1_20261104_172845.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 8.667 ± 0.939 | 4.263 ± 0.004 | 4.956 ± 0.681 | 5.722 ± 0.562 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 46.419 ± 5.971 | 26.933 ± 2.054 | 25.591 ± 0.196 | 34.388 ± 2.931 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 4.435 ± 0.111 | 4.447 ± 0.480 | 4.208 ± 0.059 | 5.711 ± 0.032 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 24.827 ± 1.963 | 23.591 ± 1.527 | 23.293 ± 1.399 | 38.395 ± 1.567 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 1.174 ± 0.008 | 0.789 ± 0.003 | 2.047 ± 0.404 | 2.367 ± 0.354 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 9.209 ± 1.293 | 5.574 ± 0.135 | 11.959 ± 0.614 | 18.538 ± 0.118 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 1.298 ± 0.228 | 0.858 ± 0.044 | 0.997 ± 0.023 | 2.364 ± 0.269 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 8.068 ± 0.665 | 5.976 ± 0.028 | 6.674 ± 0.286 | 15.915 ± 2.092 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 3.427 ± 0.132 | 1.832 ± 0.007 | 2.169 ± 0.048 | 3.933 ± 0.053 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 17.478 ± 1.912 | 12.173 ± 3.025 | 14.941 ± 1.981 | 30.534 ± 1.977 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 2.407 ± 0.418 | 2.109 ± 0.233 | 2.058 ± 0.298 | 3.957 ± 0.199 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 11.707 ± 2.254 | 12.203 ± 1.261 | 13.351 ± 1.432 | 30.498 ± 1.877 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.587 ± 0.055 | 0.226 ± 0.002 | 0.377 ± 0.006 | 0.252 ± 0.000 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 2.897 ± 0.987 | 1.153 ± 0.011 | 1.420 ± 0.030 | 1.282 ± 0.041 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.399 ± 0.032 | 0.228 ± 0.001 | 0.499 ± 0.078 | 0.244 ± 0.002 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 1.765 ± 0.282 | 1.139 ± 0.055 | 2.243 ± 0.017 | 1.547 ± 0.265 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 15.773 ± 0.039 | 7.779 ± 0.024 | 10.242 ± 1.735 | 9.520 ± 0.309 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 88.634 ± 2.147 | 44.512 ± 3.295 | 51.485 ± 3.744 | 53.888 ± 4.290 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 7.993 ± 0.350 | 7.880 ± 0.073 | 7.815 ± 0.032 | 9.357 ± 0.055 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 44.450 ± 2.302 | 54.649 ± 6.568 | 47.874 ± 6.316 | 58.819 ± 1.210 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.148 ± 0.019 | 0.015 ± 0.000 | 0.064 ± 0.009 | 0.007 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.120 ± 0.000 | 0.015 ± 0.000 | 0.063 ± 0.002 | 0.007 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.132 ± 0.001 | 0.021 ± 0.000 | 0.106 ± 0.023 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.079 ± 0.020 | 0.015 ± 0.000 | 0.055 ± 0.002 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.073 ± 0.007 | 0.015 ± 0.000 | 0.054 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.071 ± 0.003 | 0.018 ± 0.001 | 0.086 ± 0.022 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.161 ± 0.006 | 0.031 ± 0.007 | 0.093 ± 0.003 | 0.008 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.126 ± 0.002 | 0.015 ± 0.001 | 0.093 ± 0.005 | 0.006 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.127 ± 0.012 | 0.015 ± 0.000 | 0.125 ± 0.025 | 0.005 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.112 ± 0.026 | 0.016 ± 0.000 | 0.079 ± 0.008 | 0.008 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.092 ± 0.005 | 0.014 ± 0.001 | 0.073 ± 0.002 | 0.007 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.095 ± 0.002 | 0.018 ± 0.001 | 0.102 ± 0.011 | 0.005 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.385 ± 0.065 | 0.026 ± 0.001 | 0.071 ± 0.002 | 0.009 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.271 ± 0.001 | 0.024 ± 0.001 | 0.070 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.272 ± 0.002 | 0.024 ± 0.000 | 0.105 ± 0.013 | 0.005 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.188 ± 0.042 | 0.025 ± 0.001 | 0.069 ± 0.002 | 0.008 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.156 ± 0.016 | 0.025 ± 0.000 | 0.072 ± 0.006 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.154 ± 0.006 | 0.027 ± 0.000 | 0.094 ± 0.031 | 0.005 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.125 ± 0.012 | 0.011 ± 0.000 | 0.142 ± 0.006 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.132 ± 0.049 | 0.010 ± 0.000 | 0.150 ± 0.009 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.090 ± 0.001 | 0.010 ± 0.000 | 0.195 ± 0.030 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.085 ± 0.023 | 0.011 ± 0.000 | 0.072 ± 0.014 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.062 ± 0.009 | 0.010 ± 0.001 | 0.069 ± 0.000 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.059 ± 0.004 | 0.010 ± 0.000 | 0.098 ± 0.023 | 0.007 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.115 ± 0.013 | 0.009 ± 0.000 | 0.086 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.106 ± 0.000 | 0.011 ± 0.001 | 0.093 ± 0.007 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.125 ± 0.001 | 0.023 ± 0.002 | 0.100 ± 0.001 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.090 ± 0.010 | 0.009 ± 0.001 | 0.064 ± 0.006 | 0.011 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.064 ± 0.006 | 0.011 ± 0.000 | 0.058 ± 0.000 | 0.011 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.072 ± 0.010 | 0.018 ± 0.001 | 0.066 ± 0.001 | 0.019 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261104_175043/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261104_175043/run.yaml`
- Timestamp: `2026-11-04T17:50:43.176758+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261104_175043`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261104_175043/cpu_ops_t4_20261104_175043.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 8.491 ± 0.254 | 4.277 ± 0.013 | 5.310 ± 0.526 | 4.863 ± 0.039 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 46.589 ± 3.805 | 27.521 ± 1.942 | 26.189 ± 2.962 | 26.913 ± 0.986 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 4.509 ± 0.661 | 4.712 ± 0.235 | 4.277 ± 0.087 | 4.912 ± 0.081 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 28.838 ± 1.326 | 27.156 ± 3.541 | 23.382 ± 1.988 | 27.164 ± 1.918 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 1.200 ± 0.008 | 0.797 ± 0.003 | 2.450 ± 0.140 | 1.351 ± 0.069 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 9.318 ± 0.236 | 5.645 ± 0.910 | 14.342 ± 2.880 | 9.204 ± 1.743 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 1.286 ± 0.196 | 0.868 ± 0.006 | 1.326 ± 0.016 | 1.349 ± 0.045 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 7.094 ± 0.193 | 6.211 ± 1.339 | 6.770 ± 0.065 | 8.072 ± 0.216 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 3.396 ± 0.091 | 1.851 ± 0.005 | 2.736 ± 0.519 | 2.965 ± 0.077 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 17.246 ± 2.663 | 12.399 ± 0.531 | 11.678 ± 0.154 | 17.159 ± 1.089 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 2.114 ± 0.389 | 2.129 ± 0.115 | 2.253 ± 0.053 | 2.582 ± 0.118 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 11.386 ± 0.333 | 13.162 ± 1.600 | 11.123 ± 0.234 | 14.099 ± 0.278 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.572 ± 0.029 | 0.232 ± 0.002 | 0.433 ± 0.010 | 0.326 ± 0.043 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 2.973 ± 0.129 | 1.516 ± 0.052 | 1.361 ± 0.010 | 1.300 ± 0.030 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.375 ± 0.045 | 0.235 ± 0.002 | 0.634 ± 0.091 | 0.275 ± 0.004 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.782 ± 0.105 | 1.200 ± 0.031 | 2.303 ± 0.054 | 1.433 ± 0.048 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 15.832 ± 0.092 | 7.856 ± 0.047 | 9.814 ± 0.800 | 8.570 ± 0.063 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 89.301 ± 4.714 | 45.112 ± 1.933 | 49.438 ± 3.547 | 56.619 ± 4.413 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 8.866 ± 0.113 | 7.897 ± 0.646 | 7.889 ± 0.073 | 9.100 ± 0.948 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 45.474 ± 3.366 | 43.643 ± 2.019 | 43.350 ± 2.095 | 57.929 ± 6.603 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.138 ± 0.012 | 0.019 ± 0.000 | 0.062 ± 0.006 | 0.007 ± 0.002 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.130 ± 0.002 | 0.021 ± 0.000 | 0.065 ± 0.004 | 0.009 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.167 ± 0.006 | 0.027 ± 0.000 | 0.070 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.070 ± 0.000 | 0.020 ± 0.001 | 0.051 ± 0.000 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.079 ± 0.008 | 0.021 ± 0.001 | 0.053 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.090 ± 0.002 | 0.026 ± 0.000 | 0.063 ± 0.008 | 0.008 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.150 ± 0.010 | 0.022 ± 0.000 | 0.219 ± 0.006 | 0.007 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.137 ± 0.000 | 0.024 ± 0.002 | 0.269 ± 0.089 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.164 ± 0.006 | 0.023 ± 0.000 | 0.241 ± 0.031 | 0.005 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.100 ± 0.003 | 0.020 ± 0.000 | 0.183 ± 0.030 | 0.007 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.101 ± 0.014 | 0.021 ± 0.001 | 0.193 ± 0.008 | 0.007 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.129 ± 0.012 | 0.025 ± 0.000 | 0.171 ± 0.013 | 0.006 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.288 ± 0.003 | 0.033 ± 0.002 | 0.105 ± 0.003 | 0.006 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.287 ± 0.005 | 0.034 ± 0.002 | 0.106 ± 0.003 | 0.008 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.317 ± 0.006 | 0.037 ± 0.001 | 0.118 ± 0.012 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.163 ± 0.003 | 0.033 ± 0.000 | 0.110 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.164 ± 0.003 | 0.033 ± 0.000 | 0.113 ± 0.010 | 0.008 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.205 ± 0.014 | 0.040 ± 0.000 | 0.151 ± 0.021 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.097 ± 0.001 | 0.016 ± 0.002 | 0.139 ± 0.004 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.098 ± 0.000 | 0.015 ± 0.000 | 0.158 ± 0.022 | 0.008 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.135 ± 0.028 | 0.016 ± 0.001 | 0.150 ± 0.015 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.076 ± 0.009 | 0.015 ± 0.000 | 0.067 ± 0.002 | 0.006 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.067 ± 0.001 | 0.016 ± 0.000 | 0.080 ± 0.014 | 0.009 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.085 ± 0.014 | 0.015 ± 0.000 | 0.067 ± 0.001 | 0.007 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.138 ± 0.026 | 0.014 ± 0.002 | 0.092 ± 0.025 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.114 ± 0.001 | 0.015 ± 0.001 | 0.084 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.133 ± 0.001 | 0.025 ± 0.000 | 0.137 ± 0.027 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.073 ± 0.001 | 0.013 ± 0.000 | 0.059 ± 0.005 | 0.009 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.067 ± 0.010 | 0.014 ± 0.000 | 0.057 ± 0.006 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.084 ± 0.003 | 0.022 ± 0.000 | 0.065 ± 0.010 | 0.017 ± 0.002 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

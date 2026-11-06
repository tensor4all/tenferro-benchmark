# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`
- Benchmark commit: `8eb4f087ca7cfae8e8f724cdb0b344ad5123e372`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261106_172313/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261106_172313/run.yaml`
- Timestamp: `2026-11-06T17:23:13.669553+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261106_172313`.

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
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`
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


- CSV: `data/results/mac-cpu/cpu/einsum/20261106_172313/cpu_ops_t1_20261106_172313.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 15.424 ± 5.548 | 5.285 ± 2.436 | 5.446 ± 0.187 | 6.450 ± 0.211 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 83.208 ± 25.861 | 36.832 ± 7.217 | 31.352 ± 0.961 | 39.403 ± 1.068 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 6.912 ± 0.514 | 8.476 ± 2.641 | 5.063 ± 0.092 | 7.170 ± 0.285 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 44.661 ± 11.395 | 38.927 ± 6.349 | 29.130 ± 0.774 | 36.424 ± 0.614 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 1.778 ± 0.076 | 1.080 ± 0.153 | 2.505 ± 0.401 | 3.068 ± 0.078 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 10.215 ± 9.960 | 7.514 ± 3.077 | 18.161 ± 3.094 | 18.153 ± 0.568 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 2.628 ± 1.077 | 1.242 ± 0.214 | 1.280 ± 0.068 | 2.706 ± 0.126 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 10.346 ± 3.341 | 9.553 ± 3.137 | 9.464 ± 0.363 | 20.135 ± 2.097 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 6.894 ± 5.921 | 2.926 ± 1.661 | 2.772 ± 0.385 | 4.277 ± 0.263 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 28.935 ± 4.279 | 19.847 ± 14.490 | 17.147 ± 1.436 | 31.500 ± 4.534 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 2.976 ± 0.309 | 3.043 ± 1.613 | 2.575 ± 0.077 | 4.979 ± 0.418 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 19.727 ± 2.450 | 19.048 ± 10.362 | 15.770 ± 0.399 | 34.491 ± 6.592 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.126 ± 0.373 | 0.421 ± 0.338 | 0.476 ± 0.030 | 0.303 ± 0.013 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 3.635 ± 6.184 | 1.606 ± 2.128 | 2.029 ± 0.059 | 1.733 ± 0.293 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.540 ± 0.098 | 0.434 ± 0.222 | 0.583 ± 0.068 | 0.373 ± 0.018 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 2.114 ± 0.404 | 1.533 ± 0.155 | 3.088 ± 0.241 | 1.377 ± 0.067 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 30.039 ± 2.850 | 11.598 ± 0.640 | 10.710 ± 1.863 | 10.789 ± 0.763 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 139.624 ± 16.431 | 78.523 ± 15.808 | 63.652 ± 11.188 | 59.078 ± 6.351 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 10.309 ± 8.337 | 9.780 ± 4.740 | 9.182 ± 0.050 | 10.613 ± 0.757 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 74.118 ± 12.941 | 79.903 ± 12.008 | 54.992 ± 1.286 | 82.337 ± 13.433 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.156 ± 0.085 | 0.016 ± 0.002 | 0.102 ± 0.131 | 0.013 ± 0.011 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.168 ± 0.020 | 0.028 ± 0.008 | 0.107 ± 0.080 | 0.009 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.307 ± 0.122 | 0.027 ± 0.009 | 0.116 ± 0.151 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.087 ± 0.130 | 0.020 ± 0.007 | 0.114 ± 0.059 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.110 ± 0.029 | 0.027 ± 0.002 | 0.063 ± 0.044 | 0.009 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.192 ± 0.077 | 0.028 ± 0.011 | 0.178 ± 0.132 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.161 ± 0.066 | 0.027 ± 0.011 | 0.132 ± 0.063 | 0.007 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.168 ± 0.075 | 0.020 ± 0.015 | 0.104 ± 0.028 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.189 ± 0.043 | 0.025 ± 0.004 | 0.270 ± 0.207 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.129 ± 0.116 | 0.028 ± 0.027 | 0.123 ± 0.157 | 0.008 ± 0.004 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.112 ± 0.076 | 0.017 ± 0.004 | 0.087 ± 0.050 | 0.008 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.121 ± 0.005 | 0.029 ± 0.028 | 0.309 ± 0.353 | 0.006 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.587 ± 0.136 | 0.025 ± 0.001 | 0.184 ± 0.149 | 0.011 ± 0.003 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.314 ± 0.033 | 0.034 ± 0.063 | 0.114 ± 0.079 | 0.008 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.356 ± 0.123 | 0.034 ± 0.031 | 0.096 ± 0.020 | 0.006 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.373 ± 0.121 | 0.025 ± 0.000 | 0.095 ± 0.023 | 0.009 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.161 ± 0.037 | 0.053 ± 0.052 | 0.076 ± 0.004 | 0.009 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.534 ± 0.277 | 0.048 ± 0.016 | 0.394 ± 0.322 | 0.005 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.116 ± 0.086 | 0.014 ± 0.002 | 0.191 ± 0.186 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.168 ± 0.134 | 0.012 ± 0.003 | 0.225 ± 0.098 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.120 ± 0.171 | 0.015 ± 0.011 | 0.298 ± 0.328 | 0.008 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.075 ± 0.013 | 0.014 ± 0.007 | 0.137 ± 0.093 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.111 ± 0.060 | 0.017 ± 0.013 | 0.080 ± 0.046 | 0.007 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.091 ± 0.024 | 0.015 ± 0.012 | 0.203 ± 0.199 | 0.007 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.188 ± 0.058 | 0.010 ± 0.001 | 0.096 ± 0.008 | 0.013 ± 0.003 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.117 ± 0.010 | 0.015 ± 0.009 | 0.131 ± 0.035 | 0.013 ± 0.003 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.311 ± 0.345 | 0.040 ± 0.010 | 0.261 ± 0.188 | 0.020 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.232 ± 0.074 | 0.010 ± 0.001 | 0.149 ± 0.171 | 0.015 ± 0.005 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.067 ± 0.006 | 0.012 ± 0.001 | 0.118 ± 0.091 | 0.013 ± 0.003 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.087 ± 0.103 | 0.028 ± 0.007 | 0.122 ± 0.111 | 0.020 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261106_175840/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261106_175840/run.yaml`
- Timestamp: `2026-11-06T17:58:40.229051+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261106_175840`.

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
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`
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


- CSV: `data/results/mac-cpu/cpu/einsum/20261106_175840/cpu_ops_t4_20261106_175840.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 11.221 ± 1.706 | 5.280 ± 0.294 | 5.426 ± 0.275 | 5.767 ± 0.190 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 58.935 ± 1.434 | 35.760 ± 3.395 | 38.224 ± 7.201 | 27.412 ± 0.986 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 5.579 ± 0.179 | 5.317 ± 0.360 | 5.362 ± 0.230 | 6.197 ± 0.632 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 30.262 ± 4.902 | 32.706 ± 5.762 | 29.769 ± 4.726 | 27.169 ± 1.305 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 1.713 ± 0.089 | 0.951 ± 0.021 | 2.831 ± 0.500 | 1.720 ± 0.190 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 17.678 ± 4.803 | 9.893 ± 3.174 | 16.780 ± 3.013 | 9.763 ± 0.542 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 1.537 ± 0.118 | 0.974 ± 0.019 | 1.988 ± 0.391 | 1.742 ± 0.307 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 10.571 ± 3.003 | 9.462 ± 0.955 | 12.069 ± 2.010 | 9.394 ± 0.311 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 5.113 ± 0.711 | 2.368 ± 0.443 | 2.949 ± 0.493 | 2.703 ± 0.102 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 25.337 ± 2.049 | 15.203 ± 0.410 | 27.085 ± 11.722 | 13.762 ± 9.154 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 2.829 ± 0.157 | 2.475 ± 0.101 | 3.144 ± 0.612 | 2.755 ± 0.582 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 15.168 ± 0.966 | 18.562 ± 1.769 | 17.070 ± 0.754 | 14.748 ± 3.956 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.900 ± 0.180 | 0.257 ± 0.010 | 0.604 ± 0.046 | 0.316 ± 0.045 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 3.573 ± 0.489 | 3.429 ± 2.305 | 1.601 ± 0.299 | 1.364 ± 0.160 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.488 ± 0.116 | 0.278 ± 0.007 | 0.742 ± 0.055 | 0.324 ± 0.017 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 2.001 ± 0.450 | 1.980 ± 0.217 | 2.552 ± 0.185 | 1.224 ± 0.027 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 19.043 ± 5.879 | 12.905 ± 2.262 | 14.165 ± 5.314 | 8.634 ± 0.081 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 109.651 ± 23.616 | 57.662 ± 4.155 | 71.366 ± 26.376 | 46.899 ± 0.556 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 9.811 ± 1.663 | 9.623 ± 0.356 | 9.505 ± 0.654 | 8.579 ± 0.051 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 60.821 ± 12.578 | 56.358 ± 2.643 | 62.518 ± 5.913 | 47.872 ± 3.032 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.237 ± 0.074 | 0.021 ± 0.001 | 0.071 ± 0.006 | 0.007 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.155 ± 0.030 | 0.026 ± 0.006 | 0.066 ± 0.003 | 0.010 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.193 ± 0.044 | 0.030 ± 0.003 | 0.088 ± 0.021 | 0.009 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.155 ± 0.049 | 0.021 ± 0.001 | 0.065 ± 0.006 | 0.007 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.093 ± 0.002 | 0.022 ± 0.001 | 0.059 ± 0.011 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.116 ± 0.080 | 0.026 ± 0.000 | 0.075 ± 0.006 | 0.009 ± 0.003 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.213 ± 0.061 | 0.024 ± 0.002 | 0.434 ± 0.037 | 0.007 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.236 ± 0.058 | 0.022 ± 0.000 | 0.258 ± 0.010 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.178 ± 0.062 | 0.026 ± 0.002 | 0.332 ± 0.202 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.158 ± 0.063 | 0.022 ± 0.001 | 0.206 ± 0.084 | 0.008 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.139 ± 0.037 | 0.025 ± 0.002 | 0.210 ± 0.024 | 0.008 ± 0.003 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.165 ± 0.041 | 0.027 ± 0.000 | 0.229 ± 0.011 | 0.005 ± 0.005 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.508 ± 0.372 | 0.036 ± 0.001 | 0.136 ± 0.017 | 0.008 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.415 ± 0.339 | 0.038 ± 0.011 | 0.138 ± 0.047 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.480 ± 0.082 | 0.038 ± 0.003 | 0.151 ± 0.022 | 0.005 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.253 ± 0.073 | 0.035 ± 0.001 | 0.201 ± 0.048 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.246 ± 0.033 | 0.043 ± 0.004 | 0.145 ± 0.019 | 0.009 ± 0.005 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.286 ± 0.158 | 0.044 ± 0.001 | 0.171 ± 0.018 | 0.010 ± 0.005 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.153 ± 0.007 | 0.016 ± 0.001 | 0.169 ± 0.029 | 0.006 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.150 ± 0.018 | 0.017 ± 0.000 | 0.171 ± 0.018 | 0.008 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.129 ± 0.093 | 0.017 ± 0.001 | 0.171 ± 0.021 | 0.006 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.122 ± 0.043 | 0.016 ± 0.000 | 0.079 ± 0.008 | 0.007 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.112 ± 0.021 | 0.025 ± 0.011 | 0.076 ± 0.006 | 0.007 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.083 ± 0.006 | 0.017 ± 0.001 | 0.099 ± 0.067 | 0.007 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.158 ± 0.058 | 0.014 ± 0.000 | 0.103 ± 0.068 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.172 ± 0.068 | 0.017 ± 0.002 | 0.103 ± 0.017 | 0.012 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.173 ± 0.027 | 0.027 ± 0.003 | 0.115 ± 0.014 | 0.018 ± 0.004 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.130 ± 0.120 | 0.014 ± 0.014 | 0.064 ± 0.011 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.133 ± 0.023 | 0.015 ± 0.001 | 0.067 ± 0.009 | 0.013 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.098 ± 0.082 | 0.024 ± 0.000 | 0.075 ± 0.009 | 0.018 ± 0.005 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

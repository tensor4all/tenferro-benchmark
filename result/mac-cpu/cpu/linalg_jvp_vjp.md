# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`
- Benchmark commit: `12e0b50f983ca3253ff46841eb15912448ae3e20`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261022_172907/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20261022_172907`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261022_172907`.

- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261022_172907/cpu_ops_t1_20261022_172907.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20261022_172907/linalg_jvp_vjp_t1_20261022_172907.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 6.952 ± 1.671 | 5.125 ± 0.163 | 7.793 ± 0.839 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 35.534 ± 1.741 | 31.135 ± 0.564 | 47.590 ± 4.674 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 6.376 ± 2.432 | 4.646 ± 0.052 | 7.311 ± 0.780 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 33.753 ± 5.131 | 28.590 ± 1.189 | 47.115 ± 1.831 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 1.329 ± 0.823 | 1.967 ± 0.073 | 3.090 ± 1.115 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 8.995 ± 5.001 | 19.545 ± 6.710 | 25.033 ± 0.928 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 2.222 ± 1.279 | 1.187 ± 0.045 | 3.915 ± 1.104 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 9.105 ± 1.276 | 9.509 ± 0.343 | 24.691 ± 4.952 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 4.166 ± 3.103 | 2.533 ± 0.025 | 5.904 ± 0.330 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 22.111 ± 1.716 | 16.211 ± 0.202 | 47.472 ± 10.732 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 3.179 ± 0.459 | 2.362 ± 0.131 | 4.842 ± 0.245 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 20.213 ± 6.780 | 15.459 ± 0.309 | 38.095 ± 7.792 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.377 ± 0.329 | 0.429 ± 0.048 | 0.372 ± 0.063 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 1.665 ± 0.146 | 2.101 ± 0.320 | 2.585 ± 2.490 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.348 ± 0.125 | 0.885 ± 0.285 | 0.392 ± 0.288 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 1.627 ± 0.436 | 3.133 ± 0.185 | 3.360 ± 2.711 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 12.159 ± 2.005 | 9.511 ± 0.135 | 12.429 ± 0.966 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 73.868 ± 29.011 | 61.947 ± 4.444 | 95.544 ± 14.805 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 11.879 ± 2.180 | 8.773 ± 0.565 | 14.423 ± 2.880 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 78.424 ± 8.373 | 54.050 ± 2.694 | 86.763 ± 7.660 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.033 ± 0.014 | 0.076 ± 0.012 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.026 ± 0.018 | 0.136 ± 0.241 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.027 ± 0.014 | 0.101 ± 0.011 | 0.012 ± 0.004 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.033 ± 0.013 | 0.062 ± 0.013 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.028 ± 0.005 | 0.139 ± 0.087 | 0.010 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.035 ± 0.016 | 0.088 ± 0.016 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.033 ± 0.016 | 0.118 ± 0.059 | 0.008 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.026 ± 0.015 | 0.102 ± 0.016 | 0.007 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.028 ± 0.015 | 0.176 ± 0.040 | 0.007 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.023 ± 0.015 | 0.096 ± 0.033 | 0.008 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.021 ± 0.045 | 0.093 ± 0.015 | 0.010 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.022 ± 0.009 | 0.100 ± 0.094 | 0.005 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.031 ± 0.029 | 0.095 ± 0.022 | 0.008 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.078 ± 0.008 | 0.098 ± 0.031 | 0.010 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.042 ± 0.041 | 0.112 ± 0.034 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.045 ± 0.034 | 0.088 ± 0.005 | 0.008 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.043 ± 0.020 | 0.129 ± 0.214 | 0.011 ± 0.003 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.061 ± 0.025 | 0.112 ± 0.034 | 0.005 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.027 ± 0.015 | 0.172 ± 0.034 | 0.008 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.015 ± 0.008 | 0.607 ± 0.362 | 0.011 ± 0.004 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.019 ± 0.009 | 0.307 ± 0.082 | 0.008 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.025 ± 0.019 | 0.078 ± 0.016 | 0.011 ± 0.005 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.024 ± 0.014 | 0.158 ± 0.084 | 0.011 ± 0.003 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.031 ± 0.019 | 0.146 ± 0.065 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.022 ± 0.010 | 0.128 ± 0.027 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.034 ± 0.013 | 0.110 ± 0.016 | 0.012 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.039 ± 0.024 | 0.108 ± 0.003 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.021 ± 0.015 | 0.081 ± 0.022 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.032 ± 0.003 | 0.076 ± 0.013 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.038 ± 0.014 | 0.082 ± 0.070 | 0.020 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261022_180005/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20261022_180005`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261022_180005`.

- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261022_180005/cpu_ops_t4_20261022_180005.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20261022_180005/linalg_jvp_vjp_t4_20261022_180005.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 6.688 ± 0.737 | 5.228 ± 0.260 | 6.296 ± 0.407 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 48.278 ± 15.955 | 32.531 ± 2.187 | 43.258 ± 4.707 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 6.116 ± 0.715 | 5.058 ± 0.199 | 7.147 ± 1.307 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 39.861 ± 8.019 | 30.043 ± 3.329 | 40.748 ± 2.086 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 2.334 ± 0.408 | 2.639 ± 0.322 | 3.382 ± 1.431 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 9.299 ± 2.156 | 14.459 ± 0.581 | 18.185 ± 7.641 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 2.562 ± 0.243 | 1.621 ± 0.350 | 2.686 ± 0.107 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 9.464 ± 5.154 | 8.340 ± 0.066 | 15.593 ± 1.240 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 4.202 ± 0.848 | 2.772 ± 0.199 | 3.566 ± 0.353 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 20.838 ± 9.148 | 13.973 ± 0.349 | 22.497 ± 3.800 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 4.285 ± 1.426 | 2.877 ± 0.109 | 3.446 ± 1.094 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 22.780 ± 5.210 | 13.267 ± 0.204 | 23.351 ± 1.812 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.532 ± 0.086 | 0.484 ± 0.039 | 0.454 ± 0.094 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 2.909 ± 0.919 | 1.616 ± 0.318 | 2.267 ± 0.598 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.532 ± 0.028 | 0.704 ± 0.051 | 0.518 ± 0.044 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.920 ± 0.426 | 2.420 ± 0.148 | 1.824 ± 0.188 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 11.367 ± 1.074 | 9.919 ± 0.247 | 10.419 ± 1.017 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 78.919 ± 4.919 | 51.415 ± 0.802 | 74.219 ± 15.440 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 12.193 ± 1.402 | 8.749 ± 0.067 | 10.496 ± 0.879 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 73.897 ± 21.659 | 49.751 ± 0.276 | 72.146 ± 7.666 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.019 ± 0.005 | 0.070 ± 0.029 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.023 ± 0.004 | 0.076 ± 0.043 | 0.012 ± 0.003 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.038 ± 0.039 | 0.189 ± 0.076 | 0.020 ± 0.004 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.023 ± 0.002 | 0.070 ± 0.021 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.024 ± 0.020 | 0.067 ± 0.013 | 0.011 ± 0.004 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.030 ± 0.003 | 0.152 ± 0.073 | 0.013 ± 0.008 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.025 ± 0.003 | 0.375 ± 0.096 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.020 ± 0.002 | 0.303 ± 0.063 | 0.008 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.030 ± 0.004 | 0.268 ± 0.260 | 0.010 ± 0.006 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.026 ± 0.006 | 0.318 ± 0.098 | 0.007 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.036 ± 0.018 | 0.243 ± 0.091 | 0.009 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.028 ± 0.017 | 0.208 ± 0.052 | 0.009 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.047 ± 0.019 | 0.161 ± 0.011 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.044 ± 0.012 | 0.182 ± 0.092 | 0.010 ± 0.003 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.060 ± 0.016 | 0.437 ± 0.207 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.042 ± 0.004 | 0.169 ± 0.027 | 0.008 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.034 ± 0.030 | 0.176 ± 0.027 | 0.010 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.045 ± 0.019 | 0.183 ± 0.032 | 0.012 ± 0.007 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.020 ± 0.006 | 0.257 ± 0.064 | 0.007 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.016 ± 0.010 | 0.169 ± 0.014 | 0.009 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.016 ± 0.006 | 0.333 ± 0.136 | 0.008 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.018 ± 0.004 | 0.097 ± 0.022 | 0.008 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.016 ± 0.012 | 0.084 ± 0.010 | 0.010 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.030 ± 0.014 | 0.079 ± 0.005 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.021 ± 0.015 | 0.099 ± 0.012 | 0.013 ± 0.003 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.018 ± 0.006 | 0.111 ± 0.049 | 0.015 ± 0.005 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.041 ± 0.009 | 0.222 ± 0.110 | 0.018 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.023 ± 0.003 | 0.067 ± 0.004 | 0.011 ± 0.009 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.044 ± 0.036 | 0.079 ± 0.020 | 0.017 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.029 ± 0.012 | 0.123 ± 0.024 | 0.018 ± 0.002 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

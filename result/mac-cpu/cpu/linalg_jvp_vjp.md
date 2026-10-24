# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`
- Benchmark commit: `50a7dc9b3ea08f5dba14db07df1e09314cac56f0`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261024_172655/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20261024_172655`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261024_172655`.

- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261024_172655/cpu_ops_t1_20261024_172655.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20261024_172655/linalg_jvp_vjp_t1_20261024_172655.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 4.807 ± 0.314 | 5.343 ± 0.725 | 7.517 ± 2.509 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 33.565 ± 12.031 | 36.606 ± 11.451 | 51.909 ± 18.163 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 4.927 ± 0.268 | 4.946 ± 0.117 | 7.005 ± 0.255 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 33.334 ± 5.175 | 35.708 ± 4.609 | 60.939 ± 18.449 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 0.995 ± 0.033 | 1.914 ± 0.038 | 3.802 ± 0.922 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 8.200 ± 1.946 | 24.376 ± 20.491 | 35.051 ± 10.814 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 1.075 ± 0.044 | 1.289 ± 0.300 | 3.796 ± 1.515 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 8.175 ± 1.337 | 10.589 ± 2.059 | 31.258 ± 15.452 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 2.429 ± 0.320 | 2.926 ± 0.663 | 5.327 ± 0.727 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 13.877 ± 0.770 | 17.945 ± 1.000 | 40.722 ± 12.038 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 2.212 ± 0.047 | 2.766 ± 0.528 | 6.897 ± 2.459 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 20.412 ± 2.561 | 13.315 ± 0.260 | 42.093 ± 17.108 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.266 ± 0.012 | 0.416 ± 0.055 | 0.396 ± 0.396 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 1.655 ± 0.189 | 3.936 ± 1.767 | 1.952 ± 1.375 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.310 ± 0.032 | 0.526 ± 0.097 | 0.318 ± 0.026 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 2.022 ± 0.283 | 3.960 ± 1.148 | 2.117 ± 1.724 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 8.946 ± 0.492 | 10.149 ± 1.658 | 19.100 ± 9.923 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 72.511 ± 9.293 | 68.084 ± 14.200 | 79.313 ± 5.835 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 8.669 ± 0.098 | 9.303 ± 0.279 | 14.267 ± 3.689 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 65.672 ± 4.578 | 58.899 ± 1.562 | 78.765 ± 24.212 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.016 ± 0.001 | 0.069 ± 0.003 | 0.013 ± 0.008 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.022 ± 0.011 | 0.066 ± 0.004 | 0.025 ± 0.020 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.023 ± 0.001 | 0.093 ± 0.035 | 0.018 ± 0.010 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.017 ± 0.006 | 0.060 ± 0.006 | 0.016 ± 0.005 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.018 ± 0.006 | 0.062 ± 0.018 | 0.015 ± 0.013 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.025 ± 0.002 | 0.176 ± 0.108 | 0.022 ± 0.005 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.020 ± 0.010 | 0.107 ± 0.014 | 0.011 ± 0.013 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.018 ± 0.001 | 0.096 ± 0.003 | 0.016 ± 0.018 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.018 ± 0.004 | 0.113 ± 0.006 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.016 ± 0.003 | 0.090 ± 0.014 | 0.024 ± 0.019 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.018 ± 0.002 | 0.078 ± 0.004 | 0.025 ± 0.016 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.019 ± 0.001 | 0.096 ± 0.017 | 0.018 ± 0.003 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.025 ± 0.001 | 0.077 ± 0.009 | 0.012 ± 0.014 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.027 ± 0.004 | 0.079 ± 0.007 | 0.031 ± 0.004 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.028 ± 0.001 | 0.095 ± 0.024 | 0.012 ± 0.009 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.028 ± 0.012 | 0.080 ± 0.015 | 0.015 ± 0.009 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.027 ± 0.001 | 0.074 ± 0.003 | 0.013 ± 0.018 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.031 ± 0.003 | 0.102 ± 0.016 | 0.017 ± 0.010 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.014 ± 0.006 | 0.178 ± 0.067 | 0.012 ± 0.007 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.012 ± 0.000 | 0.153 ± 0.014 | 0.015 ± 0.013 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.012 ± 0.002 | 0.176 ± 0.022 | 0.009 ± 0.009 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.013 ± 0.005 | 0.083 ± 0.036 | 0.018 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.015 ± 0.005 | 0.084 ± 0.007 | 0.017 ± 0.012 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.014 ± 0.003 | 0.089 ± 0.018 | 0.012 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.011 ± 0.007 | 0.099 ± 0.014 | 0.013 ± 0.011 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.013 ± 0.001 | 0.094 ± 0.011 | 0.038 ± 0.027 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.026 ± 0.001 | 0.140 ± 0.023 | 0.034 ± 0.018 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.011 ± 0.001 | 0.065 ± 0.007 | 0.018 ± 0.006 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.014 ± 0.001 | 0.064 ± 0.008 | 0.019 ± 0.019 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.021 ± 0.001 | 0.082 ± 0.018 | 0.036 ± 0.018 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261024_175546/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20261024_175546`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261024_175546`.

- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261024_175546/cpu_ops_t4_20261024_175546.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20261024_175546/linalg_jvp_vjp_t4_20261024_175546.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 4.732 ± 0.240 | 4.633 ± 0.161 | 4.860 ± 0.054 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 23.539 ± 0.155 | 25.609 ± 0.328 | 26.903 ± 0.153 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 4.707 ± 0.306 | 4.269 ± 0.059 | 5.664 ± 0.778 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 25.019 ± 0.069 | 23.269 ± 0.237 | 27.006 ± 0.185 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 1.046 ± 0.110 | 2.022 ± 0.066 | 1.727 ± 0.064 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 6.258 ± 0.115 | 12.306 ± 1.450 | 8.589 ± 0.533 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 1.097 ± 0.092 | 1.498 ± 0.134 | 1.702 ± 0.101 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 6.764 ± 0.211 | 7.604 ± 0.625 | 13.488 ± 3.408 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 2.184 ± 0.189 | 2.756 ± 0.042 | 2.647 ± 0.055 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 11.290 ± 0.028 | 11.736 ± 0.293 | 13.707 ± 0.131 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 2.309 ± 0.164 | 2.357 ± 0.350 | 2.575 ± 0.038 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 10.559 ± 0.206 | 11.123 ± 0.105 | 14.098 ± 0.436 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.298 ± 0.007 | 0.434 ± 0.033 | 0.309 ± 0.015 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.340 ± 0.033 | 1.362 ± 0.045 | 1.315 ± 0.031 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.296 ± 0.008 | 0.628 ± 0.057 | 0.332 ± 0.032 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.320 ± 0.006 | 2.262 ± 0.012 | 1.219 ± 0.019 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 8.447 ± 0.176 | 10.154 ± 0.327 | 8.574 ± 0.058 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 46.901 ± 11.610 | 47.561 ± 0.306 | 64.596 ± 7.163 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 8.804 ± 0.101 | 8.971 ± 0.068 | 8.538 ± 0.059 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 46.731 ± 0.477 | 43.461 ± 0.251 | 46.734 ± 0.230 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.015 ± 0.001 | 0.066 ± 0.005 | 0.006 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.017 ± 0.000 | 0.067 ± 0.007 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.030 ± 0.005 | 0.076 ± 0.005 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.015 ± 0.000 | 0.078 ± 0.008 | 0.007 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.016 ± 0.000 | 0.058 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.022 ± 0.006 | 0.068 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.016 ± 0.003 | 0.433 ± 0.081 | 0.007 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.016 ± 0.001 | 0.277 ± 0.026 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.022 ± 0.003 | 0.252 ± 0.019 | 0.005 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.014 ± 0.000 | 0.439 ± 0.157 | 0.007 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.015 ± 0.000 | 0.257 ± 0.046 | 0.008 ± 0.004 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.023 ± 0.004 | 0.189 ± 0.011 | 0.005 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.028 ± 0.001 | 0.131 ± 0.012 | 0.007 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.025 ± 0.000 | 0.180 ± 0.034 | 0.009 ± 0.005 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.033 ± 0.001 | 0.131 ± 0.048 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.024 ± 0.000 | 0.155 ± 0.171 | 0.006 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.025 ± 0.001 | 0.162 ± 0.072 | 0.009 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.036 ± 0.009 | 0.162 ± 0.033 | 0.005 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.010 ± 0.000 | 0.205 ± 0.030 | 0.008 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.011 ± 0.000 | 0.176 ± 0.033 | 0.006 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.017 ± 0.005 | 0.158 ± 0.014 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.011 ± 0.001 | 0.080 ± 0.014 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.011 ± 0.000 | 0.082 ± 0.008 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.015 ± 0.002 | 0.074 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.011 ± 0.000 | 0.091 ± 0.013 | 0.010 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.012 ± 0.000 | 0.092 ± 0.012 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.026 ± 0.004 | 0.103 ± 0.002 | 0.017 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.011 ± 0.001 | 0.060 ± 0.004 | 0.012 ± 0.005 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.012 ± 0.001 | 0.062 ± 0.003 | 0.013 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.023 ± 0.003 | 0.069 ± 0.001 | 0.017 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A
